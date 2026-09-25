"""
Naive Llama decode engine.

This is OUR forward pass, not HuggingFace's. That is the point: every op that
later becomes a CUDA kernel goes through ops.call(), so swapping in a kernel
is a registry change and the parity harness immediately reports if it broke.

Phase 1 goal is correctness, not speed. Prefill uses SDPA and is not an
optimisation target -- decode is.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

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
        # Activation trace for the diff harness. None = recording off.
        self.trace: Optional[Dict[str, torch.Tensor]] = None

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
    def prefill(self, input_ids: torch.Tensor, cache: ContiguousKVCache) -> torch.Tensor:
        """input_ids: [B, S]. Returns logits for the LAST position: [B, vocab].
        Uniform prompt length across the batch (Phase 1 restriction)."""
        cfg = self.cfg
        B, S = input_ids.shape
        h = torch.nn.functional.embedding(input_ids, self._w("model.embed_tokens.weight"))
        self._rec("embed", h)

        pos = torch.arange(S, device=self.device)
        c = self.cos[pos].view(1, S, 1, cfg.head_dim)
        s = self.sin[pos].view(1, S, 1, cfg.head_dim)

        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}"
            residual = h
            hn = ops.call("rmsnorm", h, self._w(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)
            if i == 0:
                self._rec("L0.input_layernorm", hn)

            q = torch.nn.functional.linear(hn, self._w(f"{p}.self_attn.q_proj.weight"))
            k = torch.nn.functional.linear(hn, self._w(f"{p}.self_attn.k_proj.weight"))
            v = torch.nn.functional.linear(hn, self._w(f"{p}.self_attn.v_proj.weight"))
            q = q.view(B, S, cfg.num_attention_heads, cfg.head_dim)
            k = k.view(B, S, cfg.num_key_value_heads, cfg.head_dim)
            v = v.view(B, S, cfg.num_key_value_heads, cfg.head_dim)

            cq, sq = c.to(q.dtype), s.to(q.dtype)
            q = q * cq + ops.rotate_half(q) * sq
            k = k * cq + ops.rotate_half(k) * sq

            cache.append_prefill(i, k, v)

            kx = k.repeat_interleave(cfg.num_kv_groups, dim=2)
            vx = v.repeat_interleave(cfg.num_kv_groups, dim=2)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), kx.transpose(1, 2), vx.transpose(1, 2),
                is_causal=True).transpose(1, 2).reshape(B, S, -1)

            h = residual + torch.nn.functional.linear(
                attn, self._w(f"{p}.self_attn.o_proj.weight"))

            residual = h
            hn = ops.call("rmsnorm", h, self._w(f"{p}.post_attention_layernorm.weight"),
                          cfg.rms_norm_eps)
            gate = torch.nn.functional.linear(hn, self._w(f"{p}.mlp.gate_proj.weight"))
            up = torch.nn.functional.linear(hn, self._w(f"{p}.mlp.up_proj.weight"))
            h = residual + torch.nn.functional.linear(
                ops.call("swiglu", gate, up), self._w(f"{p}.mlp.down_proj.weight"))

            self._rec(f"layer.{i}", h)

        cache.advance(S)
        h = ops.call("rmsnorm", h, self._w("model.norm.weight"), cfg.rms_norm_eps)
        self._rec("final_norm", h)
        logits = torch.nn.functional.linear(h[:, -1], self._w("lm_head.weight"))
        self._rec("logits", logits)
        return logits.float()

    # ------------------------------------------------------------------
    def decode_step(self, token: torch.Tensor, cache: ContiguousKVCache) -> torch.Tensor:
        """token: [B] -- one new token per sequence. Returns logits [B, vocab].

        The hot path. Every ops.call() here is a future kernel.
        """
        cfg = self.cfg
        B = token.shape[0]
        h = torch.nn.functional.embedding(token, self._w("model.embed_tokens.weight"))

        positions = cache.positions                     # [B]
        cos = self.cos[positions]                       # [B, head_dim]
        sin = self.sin[positions]

        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}"
            residual = h
            hn = ops.call("rmsnorm", h, self._w(f"{p}.input_layernorm.weight"), cfg.rms_norm_eps)

            q = torch.nn.functional.linear(hn, self._w(f"{p}.self_attn.q_proj.weight"))
            k = torch.nn.functional.linear(hn, self._w(f"{p}.self_attn.k_proj.weight"))
            v = torch.nn.functional.linear(hn, self._w(f"{p}.self_attn.v_proj.weight"))
            q = q.view(B, cfg.num_attention_heads, cfg.head_dim)
            k = k.view(B, cfg.num_key_value_heads, cfg.head_dim)
            v = v.view(B, cfg.num_key_value_heads, cfg.head_dim)

            q, k = ops.call("rope", q, k, cos, sin)
            cache.append(i, k, v)

            k_cache, v_cache = cache.layer(i)
            # seq_lens + 1: the token just written is attendable this step.
            attn = ops.call("decode_attention", q, k_cache, v_cache,
                            cache.seq_lens + 1, self.scale)      # [B, n_q, d]

            h = residual + torch.nn.functional.linear(
                attn.reshape(B, -1), self._w(f"{p}.self_attn.o_proj.weight"))

            residual = h
            hn = ops.call("rmsnorm", h, self._w(f"{p}.post_attention_layernorm.weight"),
                          cfg.rms_norm_eps)
            gate = torch.nn.functional.linear(hn, self._w(f"{p}.mlp.gate_proj.weight"))
            up = torch.nn.functional.linear(hn, self._w(f"{p}.mlp.up_proj.weight"))
            h = residual + torch.nn.functional.linear(
                ops.call("swiglu", gate, up), self._w(f"{p}.mlp.down_proj.weight"))

            self._rec(f"decode.layer.{i}", h)

        cache.advance(1)
        h = ops.call("rmsnorm", h, self._w("model.norm.weight"), cfg.rms_norm_eps)
        self._rec("decode.final_norm", h)
        logits = torch.nn.functional.linear(h, self._w("lm_head.weight"))
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