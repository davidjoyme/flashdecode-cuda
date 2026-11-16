from __future__ import annotations
from typing import List, Optional
import torch


class BlockAllocator:
    """O(1) LIFO allocator over a fixed pool of block ids."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # Reversed so the first allocations hand out 0, 1, 2, ... which makes
        # test failures far easier to read than an arbitrary order.
        self._free: List[int] = list(range(num_blocks - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def allocate(self, n: int = 1) -> List[int]:
        if n > len(self._free):
            raise MemoryError(
                f"paged cache exhausted: asked for {n} blocks, {len(self._free)} "
                f"free of {self.num_blocks}. Raise num_blocks, lower the batch "
                f"size, or shorten the context.")
        out = [self._free.pop() for _ in range(n)]
        return out

    def free(self, blocks: List[int]) -> None:
        # Catch double frees here rather than letting two sequences silently
        # share a block and corrupt each other's history.
        seen = set(self._free)
        for b in blocks:
            if b in seen:
                raise ValueError(f"double free of block {b}")
            seen.add(b)
        self._free.extend(blocks)


class PagedKVCache:
    """Paged KV cache with a per-sequence block table.

    Shapes:
      k[layer], v[layer] : [num_blocks, n_kv_heads, block_size, head_dim]
      block_table        : [max_seqs, max_blocks_per_seq]  int32
      seq_lens           : [max_seqs]                      int32

    The interface deliberately mirrors ContiguousKVCache (append, advance,
    seq_lens, positions) so engine/model.py needs no restructuring. The one
    addition is `block_table`, which the paged attention op requires.
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 num_blocks: int, max_seqs: int, max_blocks_per_seq: int,
                 block_size: int = 16, dtype=torch.bfloat16, device="cuda"):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_seqs = max_seqs
        self.max_blocks_per_seq = max_blocks_per_seq
        self.device = device
        self.dtype = dtype

        shape = (num_blocks, n_kv_heads, block_size, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]

        self.block_table = torch.full((max_seqs, max_blocks_per_seq), -1,
                                      dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros(max_seqs, dtype=torch.int32, device=device)
        self.alloc = BlockAllocator(num_blocks)
        # Python-side mirror of the table. The GPU tensor is what the kernel
        # reads; this is what the allocator reasons about, so freeing a
        # sequence does not need a device-to-host copy.
        self._blocks: List[List[int]] = [[] for _ in range(max_seqs)]
        self._active: List[int] = []

    # ------------------------------------------------------------------
    @property
    def bytes_per_block(self) -> int:
        return (self.n_kv_heads * self.block_size * self.head_dim
                * self.k[0].element_size())

    def reset(self) -> None:
        """Release all blocks and zero all lengths, but keep the sequences
        REGISTERED with 0 tokens.

        This must mirror ContiguousKVCache.reset(), which zeroes seq_lens and
        leaves the batch intact -- LlamaEngine.generate() calls reset() and
        then immediately prefills, so deregistering here left `_active` empty
        and prefill asserted with "3 rows of k but 0 sequences". reset means
        "same batch, empty histories"; free_sequence() is how a sequence
        actually leaves.
        """
        for s in list(self._active):
            if self._blocks[s]:
                self.alloc.free(self._blocks[s])
            self._blocks[s] = []
            self.block_table[s].fill_(-1)
        self.seq_lens.zero_()

    # ------------------------------------------------------------------
    def blocks_needed(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def add_sequence(self, seq_id: int, n_tokens: int = 0) -> None:
        """Register a sequence and reserve blocks for n_tokens."""
        if seq_id >= self.max_seqs:
            raise IndexError(f"seq_id {seq_id} >= max_seqs {self.max_seqs}")
        if self._blocks[seq_id]:
            raise ValueError(f"seq {seq_id} already active; free it first")
        nb = self.blocks_needed(n_tokens)
        if nb > self.max_blocks_per_seq:
            raise ValueError(
                f"{n_tokens} tokens needs {nb} blocks but max_blocks_per_seq "
                f"is {self.max_blocks_per_seq}")
        blocks = self.alloc.allocate(nb) if nb else []
        self._blocks[seq_id] = blocks
        self._active.append(seq_id)
        if blocks:
            self.block_table[seq_id, :nb] = torch.tensor(
                blocks, dtype=torch.int32, device=self.device)
        self.seq_lens[seq_id] = n_tokens

    def free_sequence(self, seq_id: int) -> None:
        blocks = self._blocks[seq_id]
        if blocks:
            self.alloc.free(blocks)
        self._blocks[seq_id] = []
        self.block_table[seq_id].fill_(-1)
        self.seq_lens[seq_id] = 0
        if seq_id in self._active:
            self._active.remove(seq_id)

    # ------------------------------------------------------------------
    def ensure_capacity(self, seq_ids: Optional[List[int]] = None, extra: int = 1) -> None:
        """Make room for `extra` more tokens on each listed sequence.

        MUST be called before append() on any step where a sequence might
        cross a block boundary. Allocation is separated from the write on
        purpose: append() is the hot path and runs once per layer, while a
        new block is needed at most once per step, so folding allocation into
        append would do 28x the work and would also mean the Python-side
        allocator ran inside the per-layer loop.
        """
        for s in (seq_ids if seq_ids is not None else self._active):
            need = self.blocks_needed(int(self.seq_lens[s]) + extra)
            have = len(self._blocks[s])
            if need > have:
                if need > self.max_blocks_per_seq:
                    raise ValueError(
                        f"seq {s} would need {need} blocks, max_blocks_per_seq "
                        f"is {self.max_blocks_per_seq}")
                new = self.alloc.allocate(need - have)
                self._blocks[s].extend(new)
                self.block_table[s, have:need] = torch.tensor(
                    new, dtype=torch.int32, device=self.device)

    # ------------------------------------------------------------------
    @property
    def positions(self) -> torch.Tensor:
        """RoPE position of the token about to be written, for the ACTIVE
        sequences in order. [B]

        Same name and meaning as ContiguousKVCache.positions, which is what
        lets engine/model.py read it without caring which cache it has. Note
        it is indexed by active sequence, not by seq_id, so it stays [B] even
        when max_seqs is larger.
        """
        rows = torch.as_tensor(self._active, device=self.device, dtype=torch.long)
        return self.seq_lens[rows].long()

    def slot_of(self, seq_id: int, token_idx: int) -> tuple:
        """(block_id, offset) for a logical token position. The inverse of
        this mapping is what the kernel computes, so the tests assert it
        directly rather than trusting it."""
        blk = token_idx // self.block_size
        return self._blocks[seq_id][blk], token_idx % self.block_size

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor,
               seq_ids: Optional[List[int]] = None) -> None:
        """k, v: [B, n_kv_heads, head_dim] -- one token per sequence.

        Writes at each sequence's current length and does NOT advance, for
        the same reason as the contiguous cache: advancing per-layer would
        make layer 1 write to a different slot than layer 0.
        """
        ids = seq_ids if seq_ids is not None else self._active
        pos = self.seq_lens[ids].long()                       # [B]
        blk_idx = pos // self.block_size
        off = pos % self.block_size
        rows = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        blocks = self.block_table[rows, blk_idx].long()       # [B]
        if bool((blocks < 0).any()):
            raise RuntimeError(
                "append into an unallocated block -- call ensure_capacity() "
                "before the first append of the step")
        # k_cache is [num_blocks, n_kv, block_size, head_dim]; index
        # (block, :, offset, :) for each sequence.
        self.k[layer][blocks, :, off, :] = k
        self.v[layer][blocks, :, off, :] = v

    def append_prefill(self, layer: int, k: torch.Tensor, v: torch.Tensor,
                       seq_ids: Optional[List[int]] = None) -> None:
        """k, v: [B, S, n_kv_heads, head_dim] -- a whole prompt at once.

        Vectorised scatter, no Python loop over tokens. The indexing trick:
        with `blk` and `off` both shaped [B, S], the expression

            k_cache[blk, :, off, :] = k

        uses advanced indexing on dims 0 and 2 (which broadcast together to
        [B, S]) and plain slices on dims 1 and 3, so the assigned shape is
        [B, S, n_kv_heads, head_dim] -- exactly k. One kernel, no gather.

        Caller must have reserved enough blocks (add_sequence(n_tokens) or
        ensure_capacity(extra=S)); the -1 check below catches it if not.
        """
        ids = seq_ids if seq_ids is not None else self._active
        B, S = k.shape[0], k.shape[1]
        assert len(ids) == B, f"{B} rows of k but {len(ids)} sequences"
        rows = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        t = torch.arange(S, device=self.device)
        blk = self.block_table[rows][:, t // self.block_size].long()   # [B, S]
        if bool((blk < 0).any()):
            raise RuntimeError(
                f"prefill of {S} tokens needs {self.blocks_needed(S)} blocks; "
                "reserve them with add_sequence(n) or ensure_capacity(extra=S)")
        off = (t % self.block_size).unsqueeze(0).expand(B, S)           # [B, S]
        self.k[layer][blk, :, off, :] = k
        self.v[layer][blk, :, off, :] = v

    def advance(self, n: int = 1, seq_ids: Optional[List[int]] = None) -> None:
        ids = seq_ids if seq_ids is not None else self._active
        rows = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self.seq_lens[rows] += n

    # ------------------------------------------------------------------
    def active_view(self, seq_ids: Optional[List[int]] = None):
        """(block_table_rows, seq_lens) for the active sequences, in order --
        what the attention op needs."""
        ids = seq_ids if seq_ids is not None else self._active
        rows = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        return self.block_table[rows], self.seq_lens[rows]

    def gather_sequence(self, layer: int, seq_id: int) -> tuple:
        """Materialise one sequence's KV history as [n, n_kv_heads, head_dim].

        Reference/debug only. This is what the paged kernel avoids doing: it
        copies, and it is the straightforward way to prove the block mapping
        is right by comparing against a contiguous cache.
        """
        n = int(self.seq_lens[seq_id])
        nb = self.blocks_needed(n)
        if nb == 0:
            return (torch.empty(0, self.n_kv_heads, self.head_dim,
                                dtype=self.dtype, device=self.device),) * 2
        blocks = torch.as_tensor(self._blocks[seq_id][:nb],
                                 device=self.device, dtype=torch.long)
        # [nb, n_kv, block_size, d] -> [nb*block_size, n_kv, d] -> trim
        def pull(cache):
            sl = cache[blocks]                                  # gather blocks
            return (sl.permute(0, 2, 1, 3)
                      .reshape(nb * self.block_size, self.n_kv_heads, self.head_dim)[:n])
        return pull(self.k[layer]), pull(self.v[layer])

    # ------------------------------------------------------------------
    def kv_bytes_per_step(self, seq_ids: Optional[List[int]] = None) -> int:
        """KV bytes an ideal kernel must read for one decode step.

        Counts VALID tokens only, which is the honest SOL numerator. Note the
        paged kernel will actually touch whole blocks, so its floor is
        slightly higher: blocks_needed(n) * block_size tokens rather than n.
        Both numbers are reported by bench/sol so the gap from internal
        fragmentation is visible rather than hidden.
        """
        ids = seq_ids if seq_ids is not None else self._active
        tokens = sum(int(self.seq_lens[s]) for s in ids)
        per_tok = self.n_kv_heads * self.head_dim * self.k[0].element_size()
        return 2 * tokens * per_tok * self.n_layers

    def fragmentation(self, seq_ids: Optional[List[int]] = None) -> float:
        """Fraction of allocated slots that hold no valid token."""
        ids = seq_ids if seq_ids is not None else self._active
        valid = sum(int(self.seq_lens[s]) for s in ids)
        slots = sum(len(self._blocks[s]) * self.block_size for s in ids)
        return 0.0 if slots == 0 else 1.0 - valid / slots

    def __repr__(self) -> str:
        return (f"PagedKVCache(layers={self.n_layers}, block_size={self.block_size}, "
                f"blocks={self.alloc.num_used}/{self.num_blocks} used, "
                f"active={len(self._active)}, "
                f"frag={self.fragmentation()*100:.1f}%)")