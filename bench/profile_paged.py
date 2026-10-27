# ==========================================================================
# FILE LOCATION:  flashdecode/bench/profile_paged.py
# RUN AS:         python -m bench.profile_paged --batch 128 --ctx 2048
# REQUIRES:       ncu on PATH (/usr/local/cuda-13.0/bin/ncu)
# ==========================================================================
"""
Hardware-counter profile of the Phase 4 paged attention kernel.

The one metric worth collecting even on a contended GPU is
`dram__bytes.sum`. It is a hardware counter, so it is unaffected by what
else is running -- unlike duration and throughput, which contention
distorts. That makes it the right thing to measure first here.

It also tests a specific claim. The kernel is supposed to read each KV head
ONCE and reuse it across the GROUP=6 query heads sharing it, where the
PyTorch reference expands with repeat_interleave and then upcasts to fp32.
So the prediction is:

    dram__bytes  ~=  2 (K,V) x B x ctx x n_kv_heads x head_dim x 2 bytes

with NO factor of GROUP and NO factor of 2 for fp32. At B=128, ctx=2048 that
is 268 MB against the reference's 3221 MB. If the counter comes back near
268 MB, the 12x traffic reduction is measured rather than asserted. If it
comes back near 1600 MB, the GQA reuse is not happening and the kernel is
re-reading per query head.

Reusing _parse from bench/profile_rmsnorm so the two profilers cannot drift
in how they read ncu's two different CSV layouts.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from bench.profile_rmsnorm import _parse

METRICS = [
    "dram__bytes.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__waves_per_multiprocessor",
    "launch__occupancy_limit_registers",
    "gpu__time_duration.sum",
]

DRIVER = r'''
import sys, torch
sys.path.insert(0, %(root)r)
from engine import ops, ops_cuda
assert ops_cuda.AVAILABLE, ops_cuda.status()
from engine.paged_cache import PagedKVCache

B, ctx, bs = %(B)d, %(ctx)d, %(bs)d
n_kv, G, d = 2, 6, 128
nblk = ((ctx + bs - 1) // bs) * B + 8
pc = PagedKVCache(1, n_kv, d, nblk, B, (ctx + bs - 1)//bs + 2, bs,
                  dtype=torch.bfloat16, device="cuda")
filler = []
for b in range(B):
    filler.append(pc.alloc.allocate(1))      # scramble block ownership
    pc.add_sequence(b, ctx)
for f in filler:
    pc.alloc.free(f)
pc.k[0].normal_(0, 0.5); pc.v[0].normal_(0, 0.5)
q = torch.randn(B, n_kv*G, d, device="cuda", dtype=torch.bfloat16) * 0.5
table, slens = pc.active_view(list(range(B)))
fn = ops._REGISTRY["paged_decode_attention"]["cuda"]
for _ in range(%(skip)d + %(count)d):
    fn(q, pc.k[0], pc.v[0], table, slens, d ** -0.5)
torch.cuda.synchronize()
print("driver ok")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--ncu", default=None)
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()

    ncu = args.ncu or shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
    if not os.path.exists(ncu) and not shutil.which(ncu):
        print(f"ncu not found at {ncu!r}; pass --ncu <path>")
        return 1

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    driver = DRIVER % {"root": root, "B": args.batch, "ctx": args.ctx,
                       "bs": args.block_size, "skip": args.skip,
                       "count": args.count}

    n_kv, d = 2, 128
    predicted = 2 * args.batch * args.ctx * n_kv * d * 2
    ref_traffic = predicted * 6 * 2        # GROUP expansion x fp32 upcast

    print(f"B={args.batch} ctx={args.ctx} block_size={args.block_size}")
    print(f"  predicted dram__bytes (GQA-aware) : {predicted/1e6:>8.1f} MB")
    print(f"  what the PyTorch reference reads   : {ref_traffic/1e6:>8.1f} MB"
          f"  ({ref_traffic/predicted:.0f}x)\n")

    cmd = [ncu, "--metrics", ",".join(METRICS),
           "--kernel-name", "regex:paged_decode_attention_kernel",
           "--launch-skip", str(args.skip), "--launch-count", str(args.count),
           "--csv", "--page", "raw", sys.executable, "-c", driver]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if args.raw or proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        if proc.returncode != 0:
            print("\nncu failed. Usual cause is profiling permissions:")
            print("  sudo, or set NVreg_RestrictProfilingToAdminUsers=0")
            return proc.returncode

    vals = _parse(proc.stdout)
    if not vals:
        print("could not parse metrics; first 25 lines of ncu output:")
        for line in proc.stdout.splitlines()[:25]:
            print("  " + line)
        return 1

    def avg(k):
        v = vals.get(k) or next((x for n, x in vals.items() if n.startswith(k)), None)
        return sum(v) / len(v) if v else None

    dram = avg("dram__bytes.sum")
    dur = avg("gpu__time_duration.sum")

    print("=" * 70)
    if dram is not None:
        print(f"dram__bytes.sum         {dram/1e6:>9.1f} MB  "
              f"({dram/predicted*100:.0f}% of predicted)")
        if dram < predicted * 0.6:
            print("  -> BELOW the GQA-aware prediction: L2 is serving part of it.")
            print("     Expected at small B*ctx, where the whole cache fits in")
            print("     ~101 MB of L2. Raise B or ctx to measure true DRAM.")
        elif dram > predicted * 2.5:
            print("  -> WELL ABOVE prediction: the GQA reuse is NOT happening.")
            print("     The kernel is re-reading KV per query head. Check that")
            print("     one block really serves all GROUP heads.")
        else:
            print("  -> matches the GQA-aware prediction: each KV head is read")
            print(f"     once, not {6}x. The traffic reduction is measured.")
    for label, key in [
        ("DRAM throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("SM throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("achieved occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"),
        ("waves per SM", "launch__waves_per_multiprocessor"),
        ("occupancy limit (regs)", "launch__occupancy_limit_registers"),
    ]:
        v = avg(key)
        if v is not None:
            unit = "" if ("waves" in key or "limit" in key) else "%"
            print(f"{label:<24}{v:>9.2f}{unit}")
    if dur:
        print(f"{'duration':<24}{dur/1e3:>9.2f} us")
        if dram:
            print(f"{'achieved DRAM BW':<24}{dram/(dur*1e-9)/1e9:>9.0f} GB/s")
    print("=" * 70)
    print("\nOn a contended GPU, dram__bytes is still valid (hardware counter);")
    print("duration, throughput and occupancy are not. Re-run the latter three")
    print("on an idle card before quoting them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())