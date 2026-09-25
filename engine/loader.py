"""Weight loading."""

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

    # Tied embeddings: Llama-3.2-1B/3B tie lm_head to embed_tokens and the
    # checkpoint simply omits lm_head.weight. Our forward always looks up
    # 'lm_head.weight', so alias it here rather than branching in the hot path.
    if "lm_head.weight" not in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

    validate(cfg, weights)
    return cfg, weights


def validate(cfg: LlamaConfig, w: Dict[str, torch.Tensor]) -> None:
    """Fail loudly on shape surprises now rather than as a silent broadcast
    twenty layers later. Catches head_dim != hidden/n_heads models and
    MLA-style variants this engine does not support."""
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
    for name, want in checks:
        if name not in w:
            raise KeyError(f"missing weight {name!r}")
        got = tuple(w[name].shape)
        if got != want:
            raise ValueError(f"{name}: expected {want}, got {got}. This engine "
                             f"assumes standard Llama-style projections.")