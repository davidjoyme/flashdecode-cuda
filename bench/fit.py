# ==========================================================================
# FILE LOCATION:  flashdecode/bench/fit.py
# RUN AS:         python -m bench.fit                  (uses live free VRAM)
#                 python -m bench.fit --budget 20      (assume 20 GB free)
# ==========================================================================
"""
Which (batch, context) shapes actually fit right now?

On a shared GPU the answer changes minute to minute, and guessing wastes a
run: an OOM mid-benchmark leaves fragmented memory that spoils the shapes
after it, and the pre-flight skip in testing/phase1.py only tells you a
shape did not fit AFTER you asked for it.

This prints the feasible envelope first, so you pick shapes you can keep.

Memory model for one decode configuration:

    weights    fixed, ~3.09 GB for Qwen2.5-1.5B in bf16
    rope       2 tables of max_pos x head_dim fp32  (cap this, see below)
    KV cache   2 (K,V) x B x ctx x n_kv_heads x head_dim x 2B x n_layers
    transient  the reference attention upcasts one layer's KV to fp32:
               B x ctx x n_kv_heads x head_dim x 4B, times 2 for K and V

That transient term is why the measured ceiling is lower than a naive
weights+cache estimate. It is also the thing the Phase 4 kernel deletes
entirely -- a real paged kernel reads the bf16 cache in place and never
materialises an fp32 copy, so the same card will hold a noticeably larger
batch once Phase 4 lands. Worth stating in the writeup: paging plus a
fused kernel buys batch size, and batch size is throughput.
"""

from __future__ import annotations

import argparse

from config import LlamaConfig


def weight_bytes(cfg: LlamaConfig, elem: int = 2) -> int:
    hid, d = cfg.hidden_size, cfg.head_dim
    per_layer = (cfg.num_attention_heads * d * hid
                 + 2 * cfg.num_key_value_heads * d * hid
                 + cfg.num_attention_heads * d * hid
                 + 3 * cfg.intermediate_size * hid + 2 * hid)
    emb = cfg.vocab_size * hid if cfg.tie_word_embeddings else 2 * cfg.vocab_size * hid
    return elem * (per_layer * cfg.num_hidden_layers + emb)


def kv_bytes_per_token(cfg: LlamaConfig, elem: int = 2) -> int:
    """All layers, K and V, one token of one sequence."""
    return 2 * cfg.num_key_value_heads * cfg.head_dim * elem * cfg.num_hidden_layers


def plan(cfg: LlamaConfig, budget_bytes: int, rope_max_pos: int = 8192,
         paged: bool = False, block_size: int = 16) -> None:
    w = weight_bytes(cfg)
    rope = 2 * rope_max_pos * cfg.head_dim * 4
    per_tok = kv_bytes_per_token(cfg)
    avail = budget_bytes - w - rope

    print(f"  weights          {w/1e9:>7.2f} GB")
    print(f"  rope tables      {rope/1e9:>7.2f} GB  (max_pos={rope_max_pos})")
    print(f"  left for KV      {avail/1e9:>7.2f} GB")
    print(f"  KV per token     {per_tok/1024:>7.1f} KB  (all {cfg.num_hidden_layers} layers, K+V)\n")

    if avail <= 0:
        print("  NOTHING FITS -- the weights alone exceed the budget.")
        return

    print(f"{'ctx':>6} {'max B':>7} {'KV GB':>7} {'+transient':>11} {'safe B':>7}")
    for ctx in (512, 1024, 2048, 4096, 8192):
        slots = ((ctx + block_size - 1) // block_size) * block_size if paged else ctx
        # fp32 upcast of ONE layer's K and V, which the reference attention does
        trans_per_seq = 2 * slots * cfg.num_key_value_heads * cfg.head_dim * 4
        per_seq = per_tok * slots + trans_per_seq
        max_b = int(avail / (per_tok * slots))
        safe_b = int(avail * 0.90 / per_seq)
        if max_b < 1:
            print(f"{ctx:>6} {'<1':>7}")
            continue
        print(f"{ctx:>6} {max_b:>7} {per_tok*slots*max_b/1e9:>7.2f} "
              f"{trans_per_seq*safe_b/1e9:>10.2f}G {safe_b:>7}")

    print("\n  'max B' ignores transients; 'safe B' includes them and keeps 10%")
    print("  headroom. Use safe B. On a shared card, re-check before each run.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None,
                    help="GB to assume free; default reads the live figure")
    ap.add_argument("--rope-max-pos", type=int, default=8192)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    import config as C
    from transformers import AutoConfig
    cfg = LlamaConfig.from_hf(AutoConfig.from_pretrained(args.model or C.MODEL_ID))
    print(cfg.summary(), "\n")

    if args.budget is not None:
        budget = int(args.budget * 1e9)
        print(f"  assuming {args.budget:.1f} GB free\n")
    else:
        import torch
        if not torch.cuda.is_available():
            print("  no CUDA device; pass --budget to plan offline")
            return
        free, total = torch.cuda.mem_get_info()
        used_by_others = total - free
        print(f"  live: {free/1e9:.1f} GB free of {total/1e9:.1f} GB "
              f"({used_by_others/1e9:.1f} GB held by others)\n")
        budget = free

    print("--- contiguous cache ---")
    plan(cfg, budget, args.rope_max_pos, paged=False)
    print("\n--- paged cache ---")
    plan(cfg, budget, args.rope_max_pos, paged=True, block_size=args.block_size)
    print("\nIf the numbers are too small to be interesting, the levers in")
    print("order of effect are: shorter ctx, then smaller batch, then")
    print("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to cut")
    print("fragmentation, then capping the rope table.")


if __name__ == "__main__":
    main()