"""
The op boundary. UNCHANGED by the Qwen switch.

Qwen2 uses the same RMSNorm, the same rotate_half RoPE convention, the same
SwiGLU and the same attention maths as Llama. The only delta is the QKV bias,
which lives in the projections (model.py), not in any of these ops.

Every operation destined to become a CUDA kernel lives here as a pure-PyTorch
reference with a FROZEN signature. Model code calls ops.call(name, ...) and
never an implementation directly.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch

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

    Order of operations must match HF exactly: variance in fp32, scale in
    fp32, cast BACK to input dtype, and only then multiply by the weight.
    """
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return weight * x32.to(in_dtype)


@register("rmsnorm_residual", "ref")
def rmsnorm_residual_ref(x, residual, weight, eps):
    """Fused pattern: h = x + residual; returns (rmsnorm(h), h)."""
    h = x + residual
    return rmsnorm_ref(h, weight, eps), h


# --------------------------------------------------------------------------
# rope -- Phase 5 target
# --------------------------------------------------------------------------

def build_rope_cache(head_dim: int, max_pos: int, theta: float,
                     scaling: Optional[dict], device, dtype=torch.float32):
    """Precompute cos/sin tables: [max_pos, head_dim].

    cat((freqs, freqs)) is the HF 'rotate_half' convention -- pairs element i
    with i + head_dim/2, NOT the interleaved convention (i with i+1).

    Qwen2.5-1.5B: rope_theta=1000000.0, no rope_scaling. The large theta is
    the only numeric difference from Llama here and needs no code change,
    since theta comes from config.json.

    max_pos note: Qwen2.5 advertises max_position_embeddings=131072, so this
    table would be 131072 x 128 fp32 x 2 = 134 MB. The caller should pass a
    smaller max_pos for benchmarking; see LlamaEngine.__init__ if that becomes
    a problem.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    if scaling is not None and scaling.get("rope_type", scaling.get("type")) == "llama3":
        inv_freq = _llama3_scaling(inv_freq, scaling)

    t = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _llama3_scaling(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    """Llama 3.1/3.2 NTK-style frequency scaling. Not used by Qwen2.5, kept
    so a Llama checkpoint still works if you switch back."""
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

    q: [B, n_q_heads, head_dim]   k: [B, n_kv_heads, head_dim]
    cos/sin: [B, head_dim] -- already gathered at each sequence's position
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
    """
    B, n_q, d = q.shape
    _, max_len, n_kv, _ = k_cache.shape
    G = n_q // n_kv

    # GROUP THE QUERIES instead of expanding the KV cache.
    #
    # The obvious implementation is k_cache.repeat_interleave(G, dim=2)
    # followed by .float(). That materialises a transient of
    # B * n_q * max_len * d * 4 bytes -- G times too big from the expansion
    # and another 2x from the fp32 upcast, so 12x the real KV footprint at
    # Qwen2.5-1.5B's GQA group of 6. At B=128 ctx=4096 that is a single
    # 3.25 GB allocation and it OOMs a 32 GB card, which is exactly what
    # capped the Phase 1 baseline at B=64.
    #
    # Viewing q as [B, n_kv, G, d] gives the same head mapping for free:
    # query head k*G+g lands at [k, g], and k*G+g // G == k, which is the
    # repeat_interleave mapping. Verified bitwise identical to the expanding
    # version, and it is also what the Phase 4 v3 kernel does -- load each KV
    # head once, reuse it across its query group.
    qg = q.view(B, n_kv, G, d).float()

    scores = torch.einsum("bkgd,blkd->bkgl", qg, k_cache.float()) * scale

    pos = torch.arange(max_len, device=q.device).view(1, 1, 1, max_len)
    scores = scores.masked_fill(pos >= seq_lens.view(B, 1, 1, 1), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bkgl,blkd->bkgd", probs, v_cache.float())
    return out.reshape(B, n_q, d).to(q.dtype)


# --------------------------------------------------------------------------
# swiglu -- Phase 5 target
# --------------------------------------------------------------------------

@register("swiglu", "ref")
def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """gate, up: [..., intermediate]. Elementwise and memory-bound."""
    return torch.nn.functional.silu(gate) * up