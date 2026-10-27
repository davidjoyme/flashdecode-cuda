"""
Frozen target configuration.

TARGET CHANGED (2026-10-27): Llama-3.2-1B -> Qwen2.5-1.5B.
Llama is gated on HuggingFace and needs an accepted license; Qwen2.5 is not.

Why 1.5B specifically, on a 32 GB card:

  model          GQA  weights  KV budget  max B@4k   KV% of traffic
  Qwen2.5-0.5B     7    0.99      30.1       598        96.8%
  Qwen2.5-1.5B     6    3.09      28.0       238        90.1%
  Qwen2.5-3B       8    6.17      24.9       165        80.1%
  Qwen2.5-7B       7   15.23      15.9        67        50.8%

KV% is what decides whether the attention kernel shows up end to end. 7B
is capped near 50% by its own weights, so the kernel could never be worth
more than ~2x there. 1.5B reaches 90% and its GQA group of 6 is LARGER than
Llama-3.2-1B's 4, which makes the Phase 4 v3 GQA-reuse optimisation worth
more, not less. 0.5B scores better still but H=896 does not tile cleanly
(896/8 = 112 threads = 3.5 warps), so the kernels would need a tail path.

ARCHITECTURE DELTA vs Llama -- the one thing that actually changes code:
Qwen2 uses bias=True on q_proj, k_proj and v_proj. o_proj and all three MLP
projections stay bias-free, as in Llama. Nothing else differs: same RMSNorm,
same rotate_half RoPE, same SwiGLU, same HF module names.

Qwen3 was considered and rejected for now: it adds per-head QK-norm applied
BEFORE RoPE, which would complicate the Phase 5 fused RoPE+KV-write kernel.
One architectural delta is enough.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

# --------------------------------------------------------------------------
# The frozen target
# --------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-1.5B"
DTYPE = torch.bfloat16
DEVICE = "cuda"

PARITY_PROMPT = "The capital of France is"
PARITY_MAX_NEW_TOKENS = 32


@dataclass
class LlamaConfig:
    """Covers Llama and Qwen2. Name kept for continuity.

    Populated from HF config.json rather than hardcoded, so the same code
    runs 0.5B through 7B unchanged.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool = False
    rope_scaling: Optional[dict] = field(default=None)
    # Set by loader.load() from the actual checkpoint contents, NOT from
    # config.json. HF hardcodes bias=True in the Qwen2 modeling code rather
    # than exposing it as a config field, so the checkpoint is the only
    # reliable source. See loader.load().
    qkv_bias: bool = False

    @property
    def num_kv_groups(self) -> int:
        """Query heads per KV head -- the GQA sharing factor, and what the
        Phase 4 v3 kernel exploits to load each KV head once."""
        return self.num_attention_heads // self.num_key_value_heads

    @staticmethod
    def rope_from_hf(hf) -> tuple:
        """Returns (theta, scaling_or_None) across transformers versions.

        transformers 4.x: flat `hf.rope_theta`, plus `hf.rope_scaling` which is
        None unless the model actually scales.

        transformers 5.x: both consolidated into
        `hf.rope_parameters = {"rope_theta": ..., "rope_type": "default"|...}`,
        with `hf.rope_scaling` kept as an ALIAS to the same dict. `rope_theta`
        no longer exists as an attribute -- on Qwen2Config or on Llama's.

        Two traps in the 5.x shape:
          - hf.rope_theta raises AttributeError (this is what broke stage 3).
          - hf.rope_scaling is now always truthy, so passing it straight
            through would make build_rope_cache believe every model scales.
            rope_type == "default" means NO scaling, so it maps to None.
        """
        params = getattr(hf, "rope_parameters", None)
        if not isinstance(params, dict):
            params = getattr(hf, "rope_scaling", None)
        if not isinstance(params, dict):
            params = {}

        theta = getattr(hf, "rope_theta", None)
        if theta is None:
            theta = params.get("rope_theta")
        if theta is None:
            raise ValueError(
                "could not find rope_theta on the HF config (checked "
                "hf.rope_theta, hf.rope_parameters, hf.rope_scaling). "
                f"Available rope keys: "
                f"{[k for k in hf.to_dict() if 'rope' in k.lower()]}")

        rope_type = params.get("rope_type") or params.get("type")
        # "default" (5.x) and absent (4.x) both mean plain RoPE.
        scaling = None if rope_type in (None, "default") else dict(params)
        return float(theta), scaling

    @classmethod
    def from_hf(cls, hf) -> "LlamaConfig":
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        theta, scaling = cls.rope_from_hf(hf)
        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=hf.vocab_size,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=theta,
            max_position_embeddings=hf.max_position_embeddings,
            tie_word_embeddings=bool(getattr(hf, "tie_word_embeddings", False)),
            rope_scaling=scaling,
        )

    def summary(self) -> str:
        return (f"layers={self.num_hidden_layers} hidden={self.hidden_size} "
                f"q_heads={self.num_attention_heads} kv_heads={self.num_key_value_heads} "
                f"(GQA group={self.num_kv_groups}) head_dim={self.head_dim} "
                f"rope_theta={self.rope_theta} "
                f"scaling={'yes' if self.rope_scaling else 'no'} "
                f"qkv_bias={'yes' if self.qkv_bias else 'no'}")

    def kernel_threads(self) -> int:
        """Threads per block for the H-wide kernels, at 8 bf16 per thread
        (one 128-bit load). Must be a multiple of 32 -- see kernels/rmsnorm.cu.

        Qwen2.5-1.5B: 1536/8 = 192 threads = 6 warps.
        """
        return self.hidden_size // 8


# --------------------------------------------------------------------------
# Diff tolerances.
#
# bf16 carries ~8 mantissa bits, so ~1e-2 relative error from reduction-order
# differences alone is expected. 1e-1 is a real bug, 1e-3 is noise.
# --------------------------------------------------------------------------

DIFF_ATOL = 2e-2
DIFF_RTOL = 2e-2