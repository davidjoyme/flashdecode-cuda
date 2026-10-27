"""
Speed-of-light model for Llama decode. No GPU needed (downloads config.json
only, not weights).

Phase 2 rule: compute SOL BEFORE writing the kernel. Measure first and you
anchor on whatever you happened to get, then stop optimising when it stops
being easy. Compute first and you know exactly how much is left.

The point worth leading the writeup with: a decode step reads the ENTIRE
weight matrix to produce one token per sequence. Arithmetic intensity is
~batch_size FLOPs per byte. At batch 1, on a card whose compute:bandwidth
ratio is in the hundreds, you are orders of magnitude from compute-bound.
Decode is a memory problem wearing a matmul costume.

    python -m bench.sol --batch 32 --ctx 4096 --bw 1500
"""

from __future__ import annotations
import argparse
from config import LlamaConfig


def weight_bytes(cfg: LlamaConfig, elem: int = 2) -> int:
    """Weight bytes that must be read once per decode step regardless of
    batch size. The fixed cost that dominates at small batch."""
    hid, d = cfg.hidden_size, cfg.head_dim
    per_layer = (
        cfg.num_attention_heads * d * hid          # q_proj
        + 2 * cfg.num_key_value_heads * d * hid    # k_proj, v_proj
        + cfg.num_attention_heads * d * hid        # o_proj
        + 3 * cfg.intermediate_size * hid          # gate, up, down
        + 2 * hid                                  # two rmsnorm weights
    )
    return elem * (per_layer * cfg.num_hidden_layers + cfg.vocab_size * hid)


def kv_bytes(cfg: LlamaConfig, batch: int, ctx: int, elem: int = 2) -> int:
    """KV bytes read per decode step. Scales with batch AND context -- the
    term that eventually overtakes weights, and the reason Phase 4 matters."""
    return (2 * batch * ctx * cfg.num_key_value_heads * cfg.head_dim
            * elem * cfg.num_hidden_layers)


def analyse(cfg: LlamaConfig, batch: int, ctx: int, bw_gbs: float, elem: int = 2) -> dict:
    w, kv = weight_bytes(cfg, elem), kv_bytes(cfg, batch, ctx, elem)
    total = w + kv
    t = total / (bw_gbs * 1e9)
    return {"weight_GB": w / 1e9, "kv_GB": kv / 1e9, "total_GB": total / 1e9,
            "kv_fraction": kv / total, "sol_ms": t * 1e3, "sol_tok_s": batch / t}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--bw", type=float, required=True,
                    help="measured GB/s from bench.bandwidth (no default on "
                         "purpose -- never use a spec-sheet figure)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    import config as C
    from transformers import AutoConfig
    cfg = LlamaConfig.from_hf(AutoConfig.from_pretrained(args.model or C.MODEL_ID))
    print(cfg.summary(), "\n")

    r = analyse(cfg, args.batch, args.ctx, args.bw)
    print(f"batch={args.batch} ctx={args.ctx} bw={args.bw:.0f} GB/s\n")
    print(f"  weights read/step : {r['weight_GB']:.3f} GB")
    print(f"  KV read/step      : {r['kv_GB']:.3f} GB  ({r['kv_fraction']*100:.1f}% of traffic)")
    print(f"  total             : {r['total_GB']:.3f} GB\n")
    print(f"  SOL latency       : {r['sol_ms']:.3f} ms/step")
    print(f"  SOL throughput    : {r['sol_tok_s']:.0f} tok/s\n")
    if r["kv_fraction"] < 0.25:
        print("  NOTE: KV is a minority of traffic here, so attention-kernel wins")
        print("  will be nearly invisible end-to-end at this batch/context. Not a")
        print("  problem with the kernel -- it is the regime. Benchmark the kernel")
        print("  in isolation and pick a larger batch/ctx for the e2e story.")
    else:
        print("  KV dominates -- attention improvements should show up end-to-end.")


if __name__ == "__main__":
    main()