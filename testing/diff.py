"""
Layer-by-layer diff harness -- the Phase 1 gate.

When output is garbage this tells you WHICH layer first went wrong, turning a
twelve-component mystery into a one-component bug.

Layers are compared in EXECUTION order, and only the first divergence means
anything: once layer 7 is wrong, layers 8..31 are wrong too and their numbers
are noise. Read the first flagged line, ignore everything below it.

    python -m testing.diff
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

import config


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


def compare(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    """Returns (max_abs_err, max_rel_err, cosine_sim).

    Cosine is the useful third number and worth about a week of your life:
    high absolute error with cosine near 1.0 means the tensor points the right
    way and you have a SCALING bug (wrong eps, wrong attention scale, missing
    residual). Cosine well below 1.0 means the computation is STRUCTURALLY
    wrong -- indexing, RoPE convention, or head layout.
    """
    a, b = a.float().flatten(), b.float().flatten()
    if a.shape != b.shape:
        return float("inf"), float("inf"), 0.0
    diff = (a - b).abs()
    return (diff.max().item(),
            (diff / b.abs().clamp_min(1e-6)).max().item(),
            torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def report(mine: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor],
           atol: float = config.DIFF_ATOL) -> bool:
    shared = set(mine) & set(ref)
    missing = set(ref) - set(mine)
    if missing:
        print(f"[warn] not traced by engine: {sorted(missing)}\n")

    print(f"{'tensor':<28} {'max_abs':>10} {'max_rel':>10} {'cos':>9}   status")
    print("-" * 72)
    first_bad = None
    for name in execution_order(shared):
        ma, mr, cos = compare(mine[name], ref[name])
        if ma > atol and first_bad is None:
            first_bad = name
        flag = "ok" if ma <= atol else ("<<< FIRST DIVERGENCE" if first_bad == name else "bad")
        print(f"{name:<28} {ma:>10.3e} {mr:>10.3e} {cos:>9.5f}   {flag}")

    print()
    if first_bad is None:
        print(f"PASS -- all tensors within atol={atol}")
        return True
    print(f"FAIL -- first divergence at {first_bad!r}")
    _hint(first_bad, mine, ref)
    return False


def _hint(name: str, mine, ref) -> None:
    _, _, cos = compare(mine[name], ref[name])
    print()
    if name == "embed":
        print("  -> embedding lookup. Wrong tokenizer, wrong vocab slice, or the")
        print("     oracle's input_ids were not used verbatim.")
    elif name.startswith("L0."):
        print("  -> RMSNorm. Check eps, and check the weight multiply happens AFTER")
        print("     the cast back to bf16 (see ops.rmsnorm_ref). The CPU smoke test")
        print("     provably cannot catch this one -- it runs in fp32.")
    elif cos > 0.99:
        print("  -> right direction, wrong magnitude. Suspect a scale factor:")
        print("     attention scale 1/sqrt(head_dim), rms eps, or a missing residual.")
    else:
        print("  -> structurally wrong. In order of likelihood:")
        print("     1. RoPE convention: rotate_half vs interleaved. Most common.")
        print("     2. rope_scaling missing (Llama 3.1/3.2) or applied when it")
        print("        should not be (Llama 3.0).")
        print("     3. GQA expansion: repeat_interleave vs repeat. Both are")
        print("        shape-valid and only one is right (repeat_interleave).")
        print("     4. q/k/v reshape: [B,S,heads,dim] vs [B,heads,S,dim].")
        print("     5. KV cache written at the wrong position index.")
    print()
    print("  If testing/smoke.py also fails, fix that first -- it isolates the")
    print("  same bugs without a 16 GB download in the loop.")


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
        engine.prefill(ids, cache)

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