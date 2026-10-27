"""Weight loading. Handles Llama (no biases) and Qwen2 (QKV biases)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from config import LlamaConfig


def load(model_id: str, device="cuda", dtype=torch.bfloat16
         ) -> Tuple[LlamaConfig, Dict[str, torch.Tensor]]:
    """Returns (cfg, weights) with weights on `device` in `dtype`.

    Goes through transformers rather than reading safetensors directly so HF
    handles shard indexing and checkpoint-format quirks. Wasteful (builds the
    nn.Module first) but runs once.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = LlamaConfig.from_hf(AutoConfig.from_pretrained(model_id))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
    weights = {k: v.to(device=device, dtype=dtype).contiguous()
               for k, v in model.state_dict().items()}
    del model

    # Detect QKV bias from the CHECKPOINT, not from config.json. HF hardcodes
    # bias=True inside the Qwen2 modeling code rather than exposing it as a
    # config field, so config.json cannot be trusted to say. The presence of
    # the tensor is ground truth and cannot be wrong.
    cfg.qkv_bias = "model.layers.0.self_attn.q_proj.bias" in weights

    # Tied embeddings: Qwen2.5-0.5B/1.5B/3B and Llama-3.2-1B/3B tie lm_head to
    # embed_tokens and the checkpoint simply omits lm_head.weight. Our forward
    # always looks up 'lm_head.weight', so alias it here rather than branching
    # in the hot path.
    if "lm_head.weight" not in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

    validate(cfg, weights)
    return cfg, weights


def validate(cfg: LlamaConfig, w: Dict[str, torch.Tensor]) -> None:
    """Fail loudly on shape surprises now rather than as a silent broadcast
    twenty layers later. Catches head_dim != hidden/n_heads models, MLA-style
    variants, and a bias set that does not match cfg.qkv_bias."""
    hid, d = cfg.hidden_size, cfg.head_dim
    checks = [
        ("model.embed_tokens.weight", (cfg.vocab_size, hid)),
        ("model.layers.0.self_attn.q_proj.weight", (cfg.num_attention_heads * d, hid)),
        ("model.layers.0.self_attn.k_proj.weight", (cfg.num_key_value_heads * d, hid)),
        ("model.layers.0.self_attn.v_proj.weight", (cfg.num_key_value_heads * d, hid)),
        ("model.layers.0.self_attn.o_proj.weight", (hid, cfg.num_attention_heads * d)),
        ("model.layers.0.mlp.gate_proj.weight", (cfg.intermediate_size, hid)),
        ("model.layers.0.mlp.up_proj.weight", (cfg.intermediate_size, hid)),
        ("model.layers.0.mlp.down_proj.weight", (hid, cfg.intermediate_size)),
        ("model.norm.weight", (hid,)),
    ]
    if cfg.qkv_bias:
        checks += [
            ("model.layers.0.self_attn.q_proj.bias", (cfg.num_attention_heads * d,)),
            ("model.layers.0.self_attn.k_proj.bias", (cfg.num_key_value_heads * d,)),
            ("model.layers.0.self_attn.v_proj.bias", (cfg.num_key_value_heads * d,)),
        ]

    for name, want in checks:
        if name not in w:
            raise KeyError(f"missing weight {name!r}")
        got = tuple(w[name].shape)
        if got != want:
            raise ValueError(f"{name}: expected {want}, got {got}. This engine "
                             f"assumes standard Llama/Qwen2-style projections.")

    # Biases we do NOT support. o_proj or MLP biases would be silently dropped
    # by the forward pass, producing a small consistent error that looks like
    # a numerics problem rather than a missing term.
    for bad in ("model.layers.0.self_attn.o_proj.bias",
                "model.layers.0.mlp.gate_proj.bias",
                "model.layers.0.mlp.up_proj.bias",
                "model.layers.0.mlp.down_proj.bias"):
        if bad in w:
            raise ValueError(
                f"{bad} present but this engine ignores it. Add it to model.py "
                f"before proceeding -- a dropped bias term is a silent "
                f"correctness bug, not a rounding difference.")

    # Per-head QK-norm (Qwen3). Refuse rather than silently skip it.
    if "model.layers.0.self_attn.q_norm.weight" in w:
        raise ValueError(
            "q_norm/k_norm found -- this looks like Qwen3, which applies "
            "per-head RMSNorm to Q and K before RoPE. This engine does not "
            "implement that. Use a Qwen2.5 or Llama checkpoint.")