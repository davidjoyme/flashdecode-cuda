# ==========================================================================
# FILE LOCATION:  flashdecode/engine/model.py
# IMPORTED AS:    engine.model
# ==========================================================================
"""
Naive decode engine for Llama and Qwen2.

RESIDUAL THREADING (changed 2026-10-27 to use the Phase 2 fused kernel).

The textbook way to write a transformer block is

    residual = h
    hn = rmsnorm(h)
    h  = residual + attn(hn)

which puts the add at the END of one block and the norm at the START of the
next -- opposite sides of the loop boundary, so they can never be fused.

This engine instead carries two values across the loop, the way vLLM and
TensorRT-LLM do:

    hidden    the DELTA produced by the last sub-block (o_proj or down_proj)
    residual  the running accumulated hidden state

and each norm site becomes one call:

    hn, residual = rmsnorm_residual(hidden, residual, w, eps)
                   # residual <- residual + hidden   (the deferred add)
                   # hn       <- rmsnorm(new residual)

Now the add and the norm are adjacent inside a single op, which is exactly
what kernels/rmsnorm.cu fuses: it writes both outputs in one pass over the
data and turns 2 kernel launches into 1, 56 times per decode step.

Equivalence is exact, not approximate. The old form computed
`residual + linear(...)`; the new one computes `x + residual` inside the op.
IEEE addition is commutative, so the bits are identical -- which means the
restructure can be validated with ops.use("rmsnorm_residual", "ref") BEFORE
the CUDA kernel is switched on, separating "did I restructure correctly" from
"is my kernel correct".

One wrinkle: the diff harness traces each layer's accumulated output, which
in delta form is `residual + hidden` and is not materialised anywhere. It is
computed only when tracing is on, so the hot path pays nothing.

---

This is OUR forward pass, not HuggingFace's. That is the point: every op that
later becomes a CUDA kernel goes through ops.call(), so swapping in a kernel
is a registry change and the parity harness immediately reports if it broke.

Phase 1 goal is correctness, not speed. Prefill uses SDPA and is not an
optimisation target -- decode is.

Qwen2 note: q_proj/k_proj/v_proj carry a bias. It is fetched once in __init__
and passed to F.linear, which accepts None for the Llama case, so there is no
per-layer branch in the hot path.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from config import LlamaConfig
from engine import ops
from engine.cache import ContiguousKVCache


class LlamaEngine:
    def __init__(self, cfg: LlamaConfig, weights: Dict[str, torch.Tensor],
                 device="cuda", dtype=torch.bfloat16):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.w = weights
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.cos, self.sin = ops.build_rope_cache(
            cfg.head_dim, cfg.max_position_embeddings,
            cfg.rope_theta, cfg.rope_scaling, device)
        # Resolve biases once. None for Llama, tensors for Qwen2.
        self.qb = [self._bias(i, "q_proj") for i in range(cfg.num_hidden_layers)]
        self.kb = [self._bias(i, "k_proj") for i in range(cfg.num_hidden_layers)]
        self.vb = [self._bias(i, "v_proj") for i in range(cfg.num_hidden_layers)]
        # Activation trace for the diff harness. None = recording off.
        self.trace: Optional[Dict[str, torch.Tensor]] = None

    def _bias(self, layer: int, proj: str) -> Optional[torch.Tensor]:
        return self.w.get(f"model.layers.{layer}.self_attn.{proj}.bias")

    # ------------------------------------------------------------------
    def _rec(self, name: str, t: torch.Tensor) -> None:
        if self.trace is not None:
            self.trace[name] = t.detach().float().cpu().clone()

    def _w(self, key: str) -> torch.Tensor:
        return self.w[key]

    def new_cache(self, batch_size: int, max_len: int) -> ContiguousKVCache:
        return ContiguousKVCache(self.cfg.num_hidden_layers, batch_size, max_len,
                                 self.cfg.num_key_value_heads, self.cfg.head_dim,
                                 self.dtype, self.device)

    # ------------------------------------------------------------------
    def prefill(self, input_ids: torch.Tensor, cache: ContiguousKVCache,
                logits_all: bool = False) -> torch.Tensor:
        """input_ids: [B, S]. Returns logits for the LAST position: [B, vocab].
        Uniform prompt length across the batch (Phase 1 restriction).

        logits_all: compute the head over ALL S positions and then slice the
        last, instead of projecting only the last hidden state. The returned
        value is the same either way, but the arithmetic is not: HF runs
        [B, S, hidden] @ [hidden, vocab] as a GEMM and slices, while the
        default path here runs a single-row GEMV. cuBLAS selects different
        kernels and split-k for those two shapes, so the bf16 accumulation
        rounds differently and the logits differ by about one ULP -- enough to
        trip the diff harness even when every layer is bit-exact.

        The diff harness passes True so the comparison is apples-to-apples.
        Inference leaves it False, because the full projection costs
        B * S * vocab and is pure waste when only the last row is used
        (at ctx=4096 that is ~1.2 GB for a 152k vocab)."""
        cfg = self.cfg
        B, S = input_ids.shape
        h = F.embedding(input_ids, self._w("model.embed_tokens.weight"))
        self._rec("embed", h)

        pos = torch.arange(S, device=self.device)
        c = self.cos[pos].view(1, S, 1, cfg.head_dim)
        s = self.sin[pos].view(1, S, 1, cfg.head_dim)

        residual = None
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}"
            if residual is None:
                # First layer: nothing to add yet, so a plain norm.
                residual = h
                hn = ops.call("rmsnorm", h,
                              self._w(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)
            else:
                hn, residual = ops.call("rmsnorm_residual", h, residual,
                                        self._w(f"{p}.input_layernorm.weight"),
                                        cfg.rms_norm_eps)
            if i == 0:
                self._rec("L0.input_layernorm", hn)

            q = F.linear(hn, self._w(f"{p}.self_attn.q_proj.weight"), self.qb[i])
            k = F.linear(hn, self._w(f"{p}.self_attn.k_proj.weight"), self.kb[i])
            v = F.linear(hn, self._w(f"{p}.self_attn.v_proj.weight"), self.vb[i])
            q = q.view(B, S, cfg.num_attention_heads, cfg.head_dim)
            k = k.view(B, S, cfg.num_key_value_heads, cfg.head_dim)
            v = v.view(B, S, cfg.num_key_value_heads, cfg.head_dim)

            cq, sq = c.to(q.dtype), s.to(q.dtype)
            q = q * cq + ops.rotate_half(q) * sq
            k = k * cq + ops.rotate_half(k) * sq

            cache.append_prefill(i, k, v)

            kx = k.repeat_interleave(cfg.num_kv_groups, dim=2)
            vx = v.repeat_interleave(cfg.num_kv_groups, dim=2)
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), kx.transpose(1, 2), vx.transpose(1, 2),
                is_causal=True).transpose(1, 2).reshape(B, S, -1)

            # DELTA, not accumulated -- the add is deferred into the next norm.
            h = F.linear(attn, self._w(f"{p}.self_attn.o_proj.weight"))

            hn, residual = ops.call("rmsnorm_residual", h, residual,
                                    self._w(f"{p}.post_attention_layernorm.weight"),
                                    cfg.rms_norm_eps)
            gate = F.linear(hn, self._w(f"{p}.mlp.gate_proj.weight"))
            up = F.linear(hn, self._w(f"{p}.mlp.up_proj.weight"))
            h = F.linear(ops.call("swiglu", gate, up),
                         self._w(f"{p}.mlp.down_proj.weight"))

            # The layer's accumulated output is residual + h, which the delta
            # form never materialises. Only build it when tracing.
            if self.trace is not None:
                self._rec(f"layer.{i}", residual + h)

        cache.advance(S)
        # Final norm also settles the last deferred add: residual becomes the
        # true final hidden state, hn its normalised form.
        hn, residual = ops.call("rmsnorm_residual", h, residual,
                                self._w("model.norm.weight"), cfg.rms_norm_eps)
        self._rec("final_norm", hn)
        if logits_all:
            logits = F.linear(hn, self._w("lm_head.weight"))[:, -1]
        else:
            logits = F.linear(hn[:, -1], self._w("lm_head.weight"))
        self._rec("logits", logits)
        return logits.float()

    # ------------------------------------------------------------------
    def decode_step(self, token: torch.Tensor, cache: ContiguousKVCache) -> torch.Tensor:
        """token: [B] -- one new token per sequence. Returns logits [B, vocab].

        The hot path. Every ops.call() here is a future kernel.
        """
        cfg = self.cfg
        B = token.shape[0]
        h = F.embedding(token, self._w("model.embed_tokens.weight"))

        positions = cache.positions                     # [B]
        cos = self.cos[positions]                       # [B, head_dim]
        sin = self.sin[positions]

        residual = None
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}"
            if residual is None:
                residual = h
                hn = ops.call("rmsnorm", h,
                              self._w(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)
            else:
                hn, residual = ops.call("rmsnorm_residual", h, residual,
                                        self._w(f"{p}.input_layernorm.weight"),
                                        cfg.rms_norm_eps)

            q = F.linear(hn, self._w(f"{p}.self_attn.q_proj.weight"), self.qb[i])
            k = F.linear(hn, self._w(f"{p}.self_attn.k_proj.weight"), self.kb[i])
            v = F.linear(hn, self._w(f"{p}.self_attn.v_proj.weight"), self.vb[i])
            q = q.view(B, cfg.num_attention_heads, cfg.head_dim)
            k = k.view(B, cfg.num_key_value_heads, cfg.head_dim)
            v = v.view(B, cfg.num_key_value_heads, cfg.head_dim)

            q, k = ops.call("rope", q, k, cos, sin)
            cache.append(i, k, v)

            k_cache, v_cache = cache.layer(i)
            # seq_lens + 1: the token just written is attendable this step.
            attn = ops.call("decode_attention", q, k_cache, v_cache,
                            cache.seq_lens + 1, self.scale)      # [B, n_q, d]

            h = F.linear(attn.reshape(B, -1),
                         self._w(f"{p}.self_attn.o_proj.weight"))

            hn, residual = ops.call("rmsnorm_residual", h, residual,
                                    self._w(f"{p}.post_attention_layernorm.weight"),
                                    cfg.rms_norm_eps)
            gate = F.linear(hn, self._w(f"{p}.mlp.gate_proj.weight"))
            up = F.linear(hn, self._w(f"{p}.mlp.up_proj.weight"))
            h = F.linear(ops.call("swiglu", gate, up),
                         self._w(f"{p}.mlp.down_proj.weight"))

            if self.trace is not None:
                self._rec(f"decode.layer.{i}", residual + h)

        cache.advance(1)
        hn, residual = ops.call("rmsnorm_residual", h, residual,
                                self._w("model.norm.weight"), cfg.rms_norm_eps)
        self._rec("decode.final_norm", hn)
        logits = F.linear(hn, self._w("lm_head.weight"))
        self._rec("decode.logits", logits)
        return logits.float()

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 cache: Optional[ContiguousKVCache] = None) -> torch.Tensor:
        """Greedy decode. Greedy, not sampled -- determinism is the entire
        point of the parity test. Returns [B, max_new_tokens]."""
        if cache is None:
            cache = self.new_cache(input_ids.shape[0], input_ids.shape[1] + max_new_tokens + 8)
        cache.reset()
        logits = self.prefill(input_ids, cache)
        tok = logits.argmax(-1)
        out = [tok]
        for _ in range(max_new_tokens - 1):
            tok = self.decode_step(tok, cache).argmax(-1)
            out.append(tok)
        return torch.stack(out, dim=1)