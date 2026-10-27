"""
Layer-by-layer diff harness -- the Phase 1 gate.

When output is garbage this tells you WHICH layer first went wrong, turning a
twelve-component mystery into a one-component bug.

Layers are compared in EXECUTION order, and only the first divergence means
anything: once layer 7 is wrong, layers 8..31 are wrong too and their numbers
are noise. Read the first flagged line, ignore everything below it.

TOLERANCE (revised 2026-10-27). A flat absolute tolerance is wrong here,
because the tensors in this trace do not share a scale. Hidden states run
~1, logits run ~10-30. 2e-2 is a sensible atol for a hidden state and far
too tight for a logit: one bf16 ULP at magnitude 8-16 is already 6.25e-02,
so a single-ULP rounding difference in the final matmul trips an atol that
is perfectly calibrated for every other row in the table.

That is not hypothetical -- it is what the first green Qwen run did. Every
one of the 28 layers and final_norm came back bit-exact, greedy parity was
32/32 exact, and 'logits' alone reported 6.250e-02 and got flagged.

So the gate compares against the TENSOR'S OWN SCALE, in units of bf16 ULPs:

    ulps = max|a - b| / (max|b| * 2^-8)

2^-8 is the relative spacing of bf16 (8 bits of significand). A result
within a couple of ULPs of the tensor scale is at the floor of what bf16
can represent and cannot be improved by fixing code -- only by matching the
reference's exact reduction order. Both measures are printed; the verdict
uses whichever is satisfied.

    python -m testing.diff
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

import config

# Relative spacing of bf16: 1 sign + 8 exponent + 7 explicit mantissa bits,
# so 8 bits of significand precision.
BF16_REL = 2.0 ** -8

# How many ULPs of tensor scale to accept. 2 covers a different matmul
# reduction order; anything much larger is a real bug, not rounding.
ULP_BUDGET = 2.0


def execution_order(names) -> List[str]:
    def key(n: str):
        if n == "embed":
            return (0, 0)
        if n.startswith("L0."):
            return (1, 0)
        m = re.match(r"(?:decode\.)?layer\.(\d+)$", n)
        if m:
            return (2, int(m.group(1)))
        if n.endswith("final_norm"):
            return (3, 0)
        if n.endswith("logits"):
            return (4, 0)
        return (5, 0)
    return sorted(names, key=key)


def compare(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float, float]:
    """Returns (max_abs_err, max_rel_err, cosine_sim, ulps_of_scale).

    cosine is the useful discriminator and worth about a week of your life:
    high absolute error with cosine near 1.0 means the tensor points the right
    way and you have a SCALING bug (wrong eps, wrong attention scale, missing
    residual). Cosine well below 1.0 means the computation is STRUCTURALLY
    wrong -- indexing, RoPE convention, or head layout.

    ulps_of_scale is max_abs expressed in bf16 ULPs at the tensor's own
    magnitude. It is the right measure for a matmul output, where the rounding
    error tracks the size of the PARTIAL SUMS rather than the size of the
    individual result -- which is why a logit of 1.56 can legitimately carry
    an absolute error of 0.0625.
    """
    a, b = a.float().flatten(), b.float().flatten()
    if a.shape != b.shape:
        return float("inf"), float("inf"), 0.0, float("inf")
    diff = (a - b).abs()
    max_abs = diff.max().item()
    scale = b.abs().max().item()
    ulps = max_abs / (scale * BF16_REL) if scale > 0 else 0.0
    return (max_abs,
            (diff / b.abs().clamp_min(1e-6)).max().item(),
            torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
            ulps)


def report(mine: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor],
           atol: float = config.DIFF_ATOL) -> bool:
    shared = set(mine) & set(ref)
    missing = set(ref) - set(mine)
    if missing:
        print(f"[warn] not traced by engine: {sorted(missing)}\n")

    print(f"{'tensor':<24} {'max_abs':>10} {'max_rel':>10} {'cos':>9} "
          f"{'ULPs':>7}   status")
    print("-" * 76)
    first_bad = None
    for name in execution_order(shared):
        ma, mr, cos, ulps = compare(mine[name], ref[name])
        # Pass on EITHER measure: tight absolute error, or within a couple of
        # bf16 ULPs of the tensor's own scale.
        ok = (ma <= atol) or (ulps <= ULP_BUDGET)
        if not ok and first_bad is None:
            first_bad = name
        if ma == 0.0:
            flag = "exact"
        elif ok:
            flag = f"ok ({ulps:.1f} ULP)"
        else:
            flag = "<<< FIRST DIVERGENCE" if first_bad == name else "bad"
        print(f"{name:<24} {ma:>10.3e} {mr:>10.3e} {cos:>9.5f} {ulps:>7.2f}   {flag}")

    print()
    if first_bad is None:
        print(f"PASS -- every tensor within atol={atol} or {ULP_BUDGET} bf16 ULP of scale")
        return True
    print(f"FAIL -- first divergence at {first_bad!r}")
    _hint(first_bad, mine, ref)
    return False


def _hint(name: str, mine, ref) -> None:
    ma, _, cos, ulps = compare(mine[name], ref[name])
    print()
    if name == "embed":
        print("  -> embedding lookup. Wrong tokenizer, wrong vocab slice, or the")
        print("     oracle's input_ids were not used verbatim.")
    elif name.startswith("L0."):
        print("  -> RMSNorm. Check eps, and check the weight multiply happens AFTER")
        print("     the cast back to bf16 (see ops.rmsnorm_ref). The CPU smoke test")
        print("     provably cannot catch this one -- it runs in fp32.")
    elif cos > 0.99:
        print(f"  -> right direction, wrong magnitude ({ulps:.1f} ULP of scale).")
        print("     Suspect a scale factor: attention scale 1/sqrt(head_dim),")
        print("     rms eps, or a missing residual. If this is 'logits' and every")
        print("     layer above it is exact, see the note at the top of this file --")
        print("     a few ULP there is matmul reduction order, not a bug.")
    else:
        print("  -> structurally wrong. In order of likelihood:")
        print("     1. RoPE convention: rotate_half vs interleaved. Most common.")
        print("     2. rope_scaling missing or applied when it should not be. In")
        print("        transformers 5.x rope_type=='default' means NO scaling.")
        print("     3. GQA mapping: query head hq must read KV head hq // G.")
        print("     4. q/k/v reshape: [B,S,heads,dim] vs [B,heads,S,dim].")
        print("     5. KV cache written at the wrong position index.")
    print()
    print("  If testing/smoke.py also fails, fix that first -- it isolates the")
    print("  same bugs without a model download in the loop.")


def main() -> int:
    from engine.cache import ContiguousKVCache
    from engine.loader import load as load_weights
    from engine.model import LlamaEngine
    from reference.oracle import load as load_oracle

    oracle = load_oracle()
    cfg, weights = load_weights(oracle["model_id"], config.DEVICE, config.DTYPE)
    print(cfg.summary(), "\n")

    engine = LlamaEngine(cfg, weights, config.DEVICE, config.DTYPE)
    engine.trace = {}
    ids = oracle["input_ids"].to(config.DEVICE)

    max_len = ids.shape[1] + config.PARITY_MAX_NEW_TOKENS + 8
    cache = ContiguousKVCache(cfg.num_hidden_layers, ids.shape[0], max_len,
                              cfg.num_key_value_heads, cfg.head_dim,
                              config.DTYPE, config.DEVICE)
    with torch.inference_mode():
        # logits_all=True matches HF's GEMM shape so the logits row is an
        # apples-to-apples comparison. See LlamaEngine.prefill.
        engine.prefill(ids, cache, logits_all=True)

    ok = report(engine.trace, oracle["activations"])

    print("\n" + "=" * 72)
    engine.trace = None
    toks = engine.generate(ids, config.PARITY_MAX_NEW_TOKENS, cache)[0].tolist()
    want = oracle["greedy_tokens"]
    n = min(len(toks), len(want))
    match = sum(1 for i in range(n) if toks[i] == want[i])
    first_div = next((i for i in range(n) if toks[i] != want[i]), None)
    print(f"greedy parity: {match}/{n} tokens match")
    if first_div is None:
        print("  exact match")
    else:
        print(f"  first mismatch at position {first_div}")
        print(f"  ours: {toks[:n]}")
        print(f"  ref : {want[:n]}")
        print("\n  Divergence past ~20 tokens with every activation passing is")
        print("  usually benign: greedy argmax amplifies bf16 noise once two")
        print("  logits are nearly tied. Divergence in the first few tokens is a")
        print("  real bug.")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())