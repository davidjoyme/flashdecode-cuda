"""
Frozen target configuration.

Phase 0 rule: pick ONE model, ONE dtype, ONE architecture and freeze it.
Every benchmark number in the writeup is against this config. Changing
anything here invalidates every previously recorded measurement -- note the
change in logs/EXPERIMENTS.md and re-run.

Import style note: every script is run as `python -m pkg.module` from the
project root, so the root is on sys.path and `import config` works from any
subpackage. No sys.path manipulation anywhere.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

# --------------------------------------------------------------------------
# The frozen target
# --------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B"   # move to Llama-3-8B once 1B is green
DTYPE = torch.bfloat16
DEVICE = "cuda"

# Deterministic prompt for the oracle and every parity test. Keep fixed.
PARITY_PROMPT = "The capital of France is"
PARITY_MAX_NEW_TOKENS = 32


@dataclass
class LlamaConfig:
    """Populated from HF config.json rather than hardcoded, so the same code
    runs 1B and 8B unchanged."""

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

    @property
    def num_kv_groups(self) -> int:
        """Query heads per KV head -- the GQA sharing factor, and the thing
        the Phase 4 v3 kernel exploits to load each KV head once."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf(cls, hf) -> "LlamaConfig":
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=hf.vocab_size,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=hf.rope_theta,
            max_position_embeddings=hf.max_position_embeddings,
            tie_word_embeddings=bool(getattr(hf, "tie_word_embeddings", False)),
            rope_scaling=getattr(hf, "rope_scaling", None),
        )

    def summary(self) -> str:
        return (f"layers={self.num_hidden_layers} hidden={self.hidden_size} "
                f"q_heads={self.num_attention_heads} kv_heads={self.num_key_value_heads} "
                f"(GQA group={self.num_kv_groups}) head_dim={self.head_dim} "
                f"rope_theta={self.rope_theta} "
                f"scaling={'yes' if self.rope_scaling else 'no'}")


# --------------------------------------------------------------------------
# Diff tolerances.
#
# Not arbitrary: bf16 carries ~8 mantissa bits, so ~1e-2 relative error from
# reduction-order differences alone is expected. 1e-1 is a real bug, 1e-3 is
# noise. Tighten once the implementation is known good so regressions bite.
# --------------------------------------------------------------------------

DIFF_ATOL = 2e-2
DIFF_RTOL = 2e-2