"""
The op boundary.

The load-bearing architectural decision. Every operation destined to become a
CUDA kernel lives here as a pure-PyTorch reference with a FROZEN signature.
Model code calls ops.call(name, ...) and never an implementation directly.

Consequences:
  - Landing a kernel is ops.use("rmsnorm", "cuda") -- model code untouched.
  - You can run any MIX of reference and custom kernels, so a correctness bug
    bisects to one kernel in minutes instead of days.
  - Signatures are shaped for what a kernel wants (flat, contiguous, explicit
    dims), not for what reads nicely in PyTorch. Decode ops take
    [B, heads, head_dim] with no sequence axis, because that is all a decode
    kernel ever sees.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_IMPL: Dict[str, str] = {}
_REGISTRY: Dict[str, Dict[str, Callable]] = {}


def register(op: str, impl: str):
    def deco(fn):
        _REGISTRY.setdefault(op, {})[impl] = fn
        _IMPL.setdefault(op, "ref")
        return fn
    return deco


def use(op: str, impl: str) -> None:
    if impl not in _REGISTRY.get(op, {}):
        raise KeyError(f"no impl {impl!r} for op {op!r}; have {list(_REGISTRY.get(op, {}))}")
    _IMPL[op] = impl


def use_all(impl: str) -> None:
    """Switch every op that has this impl. Ops without it stay on ref."""
    for op, impls in _REGISTRY.items():
        if impl in impls:
            _IMPL[op] = impl


def reset() -> None:
    for op in _IMPL:
        _IMPL[op] = "ref"


def active() -> Dict[str, str]:
    return dict(_IMPL)


def call(op: str, *args, **kwargs):
    return _REGISTRY[op][_IMPL[op]](*args, **kwargs)


# --------------------------------------------------------------------------
# rmsnorm -- Phase 2 target
# --------------------------------------------------------------------------

@register("rmsnorm", "ref")
def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [..., hidden]   weight: [hidden]

    Order of operations matters and must match HF exactly: variance in fp32,
    scale in fp32, cast BACK to input dtype, and only then multiply by the
    weight. Doing the weight multiply in fp32 gives a slightly different
    answer that shows up as a small persistent diff in the parity harness.
    """
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return weight * x32.to(in_dtype)


@register("rmsnorm_residual", "ref")
def rmsnorm_residual_ref(x, residual, weight, eps):
    """Fused pattern: h = x + residual; returns (rmsnorm(h), h).

    Returns BOTH because the fused kernel writes both in one pass. Keeping
    this signature from day one means the kernel drops in without touching
    model.py.
    """
    h = x + residual
    return rmsnorm_ref(h, weight, eps), h


# --------------------------------------------------------------------------
# rope -- Phase 5 target (fused rope + kv write)
# --------------------------------------------------------------------------

def build_rope_cache(head_dim: int, max_pos: int, theta: float,
                     scaling: Optional[dict], device, dtype=torch.float32):
    """Precompute cos/sin tables: [max_pos, head_dim].

    cat((freqs, freqs)) is the HF 'rotate_half' convention -- pairs element i
    with i + head_dim/2. This is NOT the interleaved convention (i with i+1)
    used in the original Llama repo and in many CUDA implementations. Getting
    it wrong produces output that is grammatical but subtly wrong, which is
    the worst failure mode. If layer 0 diffs small and layer 31 is garbage,
    suspect this first.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    if scaling is not None and scaling.get("rope_type", scaling.get("type")) == "llama3":
        inv_freq = _llama3_scaling(inv_freq, scaling)

    t = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # [max_pos, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)          # [max_pos, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _llama3_scaling(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    """Llama 3.1/3.2 NTK-style frequency scaling.

    Llama-3.2-1B ships with this; Llama-3-8B does not. Skipping it on a model
    that needs it leaves short-prompt output fine and long-context output
    degraded -- a silent failure the parity test catches only if you test long
    prompts. Test long prompts.
    """
    factor = scaling["factor"]
    low_f = scaling["low_freq_factor"]
    high_f = scaling["high_freq_factor"]
    old_ctx = scaling["original_max_position_embeddings"]

    low_wavelen = old_ctx / low_f
    high_wavelen = old_ctx / high_f
    wavelen = 2 * math.pi / inv_freq

    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low_f) / (high_f - low_f)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_mid = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(is_mid, smoothed, scaled)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@register("rope", "ref")
def rope_ref(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Decode-shaped RoPE.

    q:   [B, n_q_heads, head_dim]
    k:   [B, n_kv_heads, head_dim]
    cos/sin: [B, head_dim] -- already gathered at each sequence's position

    No sequence axis, deliberately: one token per sequence is all the decode
    kernel ever sees.
    """
    c = cos.unsqueeze(1).to(q.dtype)
    s = sin.unsqueeze(1).to(q.dtype)
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


# --------------------------------------------------------------------------
# decode attention -- Phase 4 target (THE kernel)
# --------------------------------------------------------------------------

@register("decode_attention", "ref")
def decode_attention_ref(q: torch.Tensor,
                         k_cache: torch.Tensor,
                         v_cache: torch.Tensor,
                         seq_lens: torch.Tensor,
                         scale: float) -> torch.Tensor:
    """Contiguous-cache decode attention. Phase 1 only.

    q:        [B, n_q_heads, head_dim]
    k_cache:  [B, max_len, n_kv_heads, head_dim]   (v_cache same)
    seq_lens: [B]  number of VALID cache entries per sequence
    returns:  [B, n_q_heads, head_dim]

    Phase 3 replaces this with a paged variant taking a block table. That
    signature change is why paged gets its own op name rather than being
    another impl of this one.
    """
    B, n_q, d = q.shape
    _, max_len, n_kv, _ = k_cache.shape
    groups = n_q // n_kv

    # repeat_interleave materialises a full copy of the cache. That is
    # exactly the bandwidth waste the Phase 4 v3 kernel eliminates by loading
    # each KV head once and reusing it across its query group.
    k = k_cache.repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)   # [B,n_q,max_len,d]
    v = v_cache.repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)

    scores = torch.einsum("bhd,bhld->bhl", q.float(), k.float()) * scale

    pos = torch.arange(max_len, device=q.device).view(1, 1, max_len)
    scores = scores.masked_fill(pos >= seq_lens.view(B, 1, 1), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhl,bhld->bhd", probs, v.float()).to(q.dtype)


# --------------------------------------------------------------------------
# swiglu -- Phase 5 target
# --------------------------------------------------------------------------

@register("swiglu", "ref")
def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """gate, up: [..., intermediate]. Elementwise and memory-bound -- a good
    second kernel after rmsnorm."""
    return torch.nn.functional.silu(gate) * up