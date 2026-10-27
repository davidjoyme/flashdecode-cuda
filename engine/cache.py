"""
KV cache.

Phase 1 uses a dead-simple contiguous cache: one [B, max_len, n_kv, d] tensor
per layer for K and for V. It wastes memory proportional to max_len and
cannot handle ragged batches efficiently. That is fine -- its job is to be
obviously correct so the Phase 3 paged cache has something to diff against.

The interface is deliberately one a PAGED cache can also satisfy, so
model.py does not change in Phase 3.
"""

from __future__ import annotations
import torch


class ContiguousKVCache:
    def __init__(self, n_layers: int, batch_size: int, max_len: int,
                 n_kv_heads: int, head_dim: int, dtype, device):
        self.n_layers = n_layers
        self.max_len = max_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        shape = (batch_size, max_len, n_kv_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        # Per-sequence token count. Ragged from the start, even though the
        # Phase 1 kernel handles raggedness with a mask rather than efficiently.
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self) -> None:
        self.seq_lens.zero_()

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [B, n_kv_heads, head_dim] -- one token per sequence.

        Writes at each sequence's CURRENT length and does not advance. Call
        advance() once per decode step after every layer has written --
        advancing per-layer would make layer 1 write to a different slot
        than layer 0, which is a nasty, quiet bug.
        """
        idx = self.seq_lens.long()
        b = torch.arange(k.shape[0], device=k.device)
        self.k[layer][b, idx] = k
        self.v[layer][b, idx] = v

    def append_prefill(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [B, S, n_kv_heads, head_dim] -- whole prompt at once.

        Phase 1 assumes uniform prompt length across the batch. Ragged
        prefill is a Phase 3 problem; assert rather than silently corrupt.
        """
        assert torch.all(self.seq_lens == 0), "prefill into a non-empty cache"
        S = k.shape[1]
        self.k[layer][:, :S] = k
        self.v[layer][:, :S] = v

    def advance(self, n: int = 1) -> None:
        self.seq_lens += n
        assert int(self.seq_lens.max()) <= self.max_len, "KV cache overflow"

    def layer(self, i: int):
        return self.k[i], self.v[i]

    @property
    def positions(self) -> torch.Tensor:
        """RoPE position of the token about to be written. [B]"""
        return self.seq_lens.long()

    def kv_bytes_per_step(self) -> int:
        """KV bytes a perfectly efficient decode step must read -- the
        numerator of the Phase 4 speed-of-light calculation.

        Counts only VALID entries, because that is what an ideal kernel
        touches. The contiguous implementation actually reads max_len and is
        therefore worse than SOL by construction.
        """
        tokens = int(self.seq_lens.sum())
        per_tok = self.n_kv_heads * self.head_dim * self.k[0].element_size()
        return 2 * tokens * per_tok * self.n_layers    # 2 = K and V