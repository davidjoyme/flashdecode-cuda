from __future__ import annotations
import argparse
import csv as _csv
import io
import os
import shutil
import subprocess
import sys
from bench.profile_rmsnorm import _parse  # noqa: F401  (kept for compatibility)

N_KV, GROUP, HEAD_DIM = 2, 6, 128
L2_BYTES = 96 * 1024 * 1024          # cudaDeviceProp.l2CacheSize on a 5090

# Matches v0, v1, and both halves of v2.
KERNEL_RE = "regex:paged_(decode_attention|attn_v2)"

METRICS = [
    "dram__bytes.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__waves_per_multiprocessor",
    "gpu__time_duration.sum",
]

# One sequence at ctx=2048 is 2.10 MB of KV, so L2 is exactly 48 sequences.
# These cross it deliberately: 0.67x, 1.33x, 2.67x, 5.33x, 8.0x.
SWEEP_BATCHES = [32, 64, 128, 256, 384]

# For --sweep-splits. fd_choose_splits targets 850 resident blocks, so at
# B=384 it picks ~1 -- which degrades v2 back to v1 behaviour (one block per
# (kv_head, seq) streaming the whole context). That is a candidate explanation
# for the odd large-B rows, and pinning the count is how to test it.
SPLIT_VALUES = [1, 2, 4, 8, 16, 32]

DRIVER = r'''
import os, sys
os.environ["FD_ATTN_VERSION"] = "%(version)d"
%(splits_env)s
import torch
sys.path.insert(0, %(root)r)
from engine import ops, ops_cuda
assert ops_cuda.AVAILABLE, ops_cuda.status()
from engine.paged_cache import PagedKVCache

B, ctx, bs = %(B)d, %(ctx)d, %(bs)d
n_kv, G, d = 2, 6, 128
bpr = (ctx + bs - 1) // bs
nblk = bpr * B + B + 8
need = nblk * n_kv * bs * d * 2 * 2        # k and v
free, total = torch.cuda.mem_get_info()
if need > free * 0.8:
    print("SKIP: needs %%.1f GB of KV pool, only %%.1f GB free" %% (need/1e9, free/1e9))
    raise SystemExit(0)

pc = PagedKVCache(1, n_kv, d, nblk, B, bpr + 2, bs,
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


def logical_bytes(B: int, ctx: int) -> dict:
    """Compulsory DRAM traffic for one paged decode-attention call.

    KV dominates by two orders of magnitude, but q and out are counted so the
    '% of logical' ratio is honest rather than flattering. v2's (m, l, acc)
    scratch is excluded: it is a few percent and depends on the split count,
    which the kernel chooses internally.
    """
    kv = 2 * B * ctx * N_KV * HEAD_DIM * 2          # K and V, bf16, read once
    q = B * N_KV * GROUP * HEAD_DIM * 2             # query, one token each
    out = B * N_KV * GROUP * HEAD_DIM * 2           # output
    return {"kv": kv, "q": q, "out": out, "total": kv + q + out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--version", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--splits", type=int, default=0,
                    help="pin v2's split count (0 = let the heuristic choose)")
    ap.add_argument("--sweep", action="store_true",
                    help=f"profile B in {SWEEP_BATCHES} to find where traffic "
                         "converges on the logical footprint")
    ap.add_argument("--sweep-splits", action="store_true",
                    help=f"at one --batch, profile splits in {SPLIT_VALUES} "
                         "plus the heuristic's own choice. All measurements "
                         "come from one session back-to-back, so contention "
                         "is as close to constant as it gets.")
    # NOTE: argparse runs `help % params` on every help string, and Python
    # 3.14 validates them eagerly in add_argument rather than only on --help.
    # So a bare "%" is a crash at import time: "%-of" parses as a format spec
    # (left-justified octal) and raises "badly formed help string". Write
    # "percent" in prose, or escape as "%%".
    ap.add_argument("--bw", type=float, default=None,
                    help="measured HBM bandwidth in GB/s (bench.bandwidth). "
                         "No default: without it the percent-of-peak lines "
                         "are skipped rather than computed against a guess.")
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--ncu", default=None)
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()

    ncu = args.ncu or shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
    if not os.path.exists(ncu) and not shutil.which(ncu):
        print(f"ncu not found at {ncu!r}; pass --ncu <path>")
        return 1

    if args.bw is None:
        print("[note] no --bw given; %-of-peak lines are skipped. Run")
        print("       python -m bench.bandwidth   and pass the read figure.\n")

    per_seq = 2 * args.ctx * N_KV * HEAD_DIM * 2
    print(f"one sequence @ctx{args.ctx} = {per_seq/1e6:.2f} MB of KV, "
          f"so L2 ({L2_BYTES/1e6:.0f} MB) is exactly "
          f"{L2_BYTES/per_seq:.0f} sequences\n")

    # --- split sweep: one batch size, several pinned split counts ---------
    if args.sweep_splits:
        plan = [0] + SPLIT_VALUES          # 0 = whatever the heuristic picks
        rows = []
        for i, sp in enumerate(plan):
            label = "auto" if sp == 0 else str(sp)
            print(f"[{i+1}/{len(plan)}] B={args.batch} splits={label}")
            got = _profile_one(ncu, args.batch, args, splits=sp)
            if got:
                got["splits"] = label
                rows.append(got)
        if rows:
            _splits_table(rows, args)
        return 0 if rows else 1

    batches = SWEEP_BATCHES if args.sweep else [args.batch]
    rows = []
    for i, B in enumerate(batches):
        if len(batches) > 1:
            print(f"[{i+1}/{len(batches)}] B={B}")
        got = _profile_one(ncu, B, args)
        if got:
            rows.append(got)

    if len(rows) > 1:
        _sweep_table(rows, args)
    return 0 if rows else 1


def _profile_one(ncu: str, B: int, args, splits: int | None = None) -> dict | None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pin = args.splits if splits is None else splits
    splits_env = (f'os.environ["FD_ATTN_SPLITS"] = "{pin}"' if pin else "")
    driver = DRIVER % {"root": root, "B": B, "ctx": args.ctx,
                       "bs": args.block_size, "skip": args.skip,
                       "count": args.count, "version": args.version,
                       "splits_env": splits_env}

    lb = logical_bytes(B, args.ctx)
    ref = lb["kv"] * GROUP * 2        # reference: GROUP expansion x fp32 upcast
    print(f"  B={B} ctx={args.ctx} v{args.version}  "
          f"logical {lb['total']/1e6:.1f} MB "
          f"({lb['total']/L2_BYTES:.2f}x L2)  "
          f"reference would read {ref/1e6:.0f} MB ({ref/lb['total']:.0f}x)")

    cmd = [ncu, "--metrics", ",".join(METRICS),
           "--kernel-name", KERNEL_RE,
           # v2 launches 2 kernels per call, so take every launch and group
           # them below rather than assuming one kernel per invocation.
           "--launch-skip", str(args.skip * (2 if args.version == 2 else 1)),
           "--launch-count", str(args.count * (2 if args.version == 2 else 1)),
           "--cache-control", "all",
           "--csv", "--page", "raw", sys.executable, "-c", driver]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if "SKIP:" in proc.stdout:
        line = next(l for l in proc.stdout.splitlines() if "SKIP:" in l)
        print(f"  {line.strip()}\n")
        return None

    if args.raw or proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        if proc.returncode != 0:
            print("\nncu failed. Usual cause is profiling permissions:")
            print("  sudo, or set NVreg_RestrictProfilingToAdminUsers=0")
            return None

    per_kernel = _parse_by_kernel(proc.stdout)
    if not per_kernel:
        print("  could not parse metrics; first 25 lines of ncu output:")
        for line in proc.stdout.splitlines()[:25]:
            print("    " + line)
        print("  (if this says 'No kernels were profiled', the kernel-name")
        print("   regex missed. v0/v1/v2 have different names -- see the")
        print("   module docstring.)\n")
        return None

    # Sum across the kernels of ONE invocation. v2 = partial + combine, so
    # dividing total traffic by the number of ncu launches would halve it.
    n_launch = sum(len(v["dram__bytes.sum"]) for v in per_kernel.values()
                   if "dram__bytes.sum" in v)
    n_inv = max(1, n_launch // (2 if args.version == 2 else 1))
    dram = sum(sum(v.get("dram__bytes.sum", [])) for v in per_kernel.values()) / n_inv
    dur = sum(sum(v.get("gpu__time_duration.sum", [])) for v in per_kernel.values()) / n_inv

    print(f"  kernels profiled: {', '.join(sorted(per_kernel))}")
    out = {"B": B, "dram": dram, "dur": dur, "logical": lb["total"],
           "kv": lb["kv"], "ref": ref}
    _report_one(out, per_kernel, args)
    return out


def _report_one(r: dict, per_kernel: dict, args) -> None:
    dram, dur, logical = r["dram"], r["dur"], r["logical"]
    print("  " + "-" * 66)
    print(f"  {'logical (compulsory)':<26}{logical/1e6:>9.1f} MB"
          f"  ({logical/L2_BYTES:.2f}x L2)")
    print(f"  {'dram__bytes.sum':<26}{dram/1e6:>9.1f} MB"
          f"  ({dram/logical*100:.0f}% of logical)")

    # --- question 1: is GQA reuse real? -----------------------------------
    if dram > logical * 2.5:
        print("   -> GQA REUSE IS NOT HAPPENING. Traffic is near the")
        print(f"      per-query-head figure ({r['ref']/1e6:.0f} MB). Check that one")
        print("      block really serves all GROUP heads.")
    elif dram > logical * 0.9:
        print(f"   -> matches the GQA-aware prediction: each KV head read once,")
        print(f"      not {GROUP}x. The {r['ref']/logical:.0f}x reduction is measured, not asserted.")
    else:
        print("   -> below the prediction, so L2 served part of it. The GQA")
        print("      claim is untestable at this shape (too little traffic to")
        print("      tell 'read once' from 'read once and cached'). Raise B.")

    # --- question 2: fraction of peak -------------------------------------
    for label, key in [
        ("DRAM throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("SM throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("achieved occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"),
        ("waves per SM", "launch__waves_per_multiprocessor"),
    ]:
        vals = [x for v in per_kernel.values() for x in v.get(key, [])]
        if vals:
            unit = "" if "waves" in key else "%"
            print(f"  {label:<26}{sum(vals)/len(vals):>9.2f}{unit}")
    if dur:
        print(f"  {'duration (per call)':<26}{dur/1e3:>9.2f} us")
        if args.bw:
            # TASK rate: the fixed work this kernel must do, over the time it
            # took. This is the efficiency figure, because the compulsory byte
            # count is a property of the problem and does not move when the
            # kernel changes.
            #
            # MOVED rate: whatever crossed the bus, over the same time. This
            # is memory-system utilisation, NOT efficiency -- a kernel that
            # generates overhead traffic scores HIGHER on it while doing no
            # more useful work. The split sweep demonstrated exactly that:
            # splits 1 -> 32 at B=384 raised moved-rate 789 -> 974 GB/s while
            # the duration stayed flat, because v2's (m,l,acc) scratch grew
            # 140 MB. Faster-looking, identical work.
            task = logical / (dur * 1e-9) / 1e9
            moved = dram / (dur * 1e-9) / 1e9
            print(f"  {'task rate (compulsory)':<26}{task:>9.0f} GB/s"
                  f"  = {task/args.bw*100:.0f}% of peak   <-- the efficiency figure")
            print(f"  {'moved rate (all traffic)':<26}{moved:>9.0f} GB/s"
                  f"  = {moved/args.bw*100:.0f}% of peak   (utilisation, not efficiency)")
            if dram > logical * 1.08:
                print(f"   -> {(dram-logical)/1e6:.0f} MB ({dram/logical*100-100:.0f}%) of")
                print("      traffic is OVERHEAD above the compulsory footprint, so")
                print("      the moved rate flatters this shape. Use the task rate.")
            elif dram < logical * 0.9:
                print("   -> traffic is BELOW the footprint, so some of it was")
                print("      absorbed by cache and this duration is short by that")
                print("      much. Neither rate is trustworthy here; raise B.")
    print()


def _parse_by_kernel(csv_text: str) -> dict:
    """Like _parse, but keyed by kernel name.

    Needed because v2 is two kernels per call and they must be summed per
    invocation, not averaged together. _parse() deliberately drops the kernel
    name, and its signature is shared with profile_rmsnorm, so this is a
    separate reader rather than a change there.
    """
    lines = csv_text.splitlines()
    start = next((i for i, l in enumerate(lines) if "Kernel Name" in l), None)
    if start is None:
        return {}
    rows = list(_csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    if not rows:
        return {}

    skip = {"ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
            "Kernel Time", "Context", "Stream", "Block Size", "Grid Size",
            "Device", "CC", "Section Name", "Metric Name", "Metric Value",
            "Metric Unit"}
    scale = {"Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
             "usecond": 1e3, "msecond": 1e6, "nsecond": 1.0}
    out: dict = {}
    for r in rows:
        name = (r.get("Kernel Name") or "").strip().strip('"')
        # Trim template arguments so v2's partial/combine stay distinguishable
        # but every instantiation of one kernel groups together.
        name = name.split("<")[0].replace("void ", "").strip()
        # ncu emits blank/summary rows among the per-launch ones. They were
        # landing in a "?" bucket and being summed into the traffic total.
        # Harmless when empty, wrong if not -- so drop them explicitly.
        if not name:
            continue
        bucket = out.setdefault(name, {})
        if "Metric Name" in r:                                  # LONG
            k = (r.get("Metric Name") or "").strip()
            raw = (r.get("Metric Value") or "").replace(",", "").strip()
            unit = (r.get("Metric Unit") or "").strip()
            try:
                bucket.setdefault(k, []).append(float(raw) * scale.get(unit, 1.0))
            except ValueError:
                pass
        else:                                                   # WIDE
            for col, raw in r.items():
                if not col or col in skip:
                    continue
                raw = (raw or "").replace(",", "").strip()
                try:
                    bucket.setdefault(col.strip(), []).append(float(raw))
                except ValueError:
                    pass
    return out


def _splits_table(rows: list[dict], args) -> None:
    """Split count vs achieved bandwidth, all from one session.

    The question this answers: is the odd large-B bandwidth a SPLIT artifact
    or CONTENTION? Comparing against a run from an hour ago cannot tell,
    because contention drifts between sessions. Back-to-back it can.
    """
    print("=" * 74)
    print(f"SPLIT SWEEP -- B={args.batch} ctx={args.ctx}, one session")
    print("=" * 74)
    hdr = f"{'splits':>8}{'dram MB':>10}{'%log':>7}{'overhead':>10}{'us':>10}"
    if args.bw:
        hdr += f"{'task':>7}{'moved':>7}"
    print(hdr)
    base = min(r["dram"] for r in rows)
    for r in rows:
        line = (f"{r['splits']:>8}{r['dram']/1e6:>10.1f}"
                f"{r['dram']/r['logical']*100:>7.0f}"
                f"{(r['dram']-base)/1e6:>9.1f}M{r['dur']/1e3:>10.2f}")
        if args.bw:
            task = r["logical"]/(r["dur"]*1e-9)/1e9
            moved = r["dram"]/(r["dur"]*1e-9)/1e9
            line += f"{task/args.bw*100:>6.0f}%{moved/args.bw*100:>6.0f}%"
        print(line)
    print("-" * 74)
    auto = next((r for r in rows if r["splits"] == "auto"), None)
    pinned = [r for r in rows if r["splits"] != "auto"]
    if auto and pinned:
        best = min(pinned, key=lambda r: r["dur"])
        gain = auto["dur"] / best["dur"]
        extra = (best["dram"] - auto["dram"]) / auto["dram"] * 100
        print(f"heuristic ({auto['dur']/1e3:.1f} us, {auto['dram']/1e6:.0f} MB) vs "
              f"best-latency splits={best['splits']} "
              f"({best['dur']/1e3:.1f} us, {best['dram']/1e6:.0f} MB)")
        print(f"  latency {gain:.2f}x better for {extra:+.0f}% traffic")
        if gain > 1.15:
            print("  -> the heuristic leaves real latency on the table here.")
            print("     Calibrate fd_choose_splits from this data.")
        else:
            print("  -> within noise on latency, so at this shape the heuristic")
            print("     is trading a few percent of latency for less traffic --")
            print("     a defensible call, not a miscalibration.")
    print("\nWATCH THE OVERHEAD COLUMN. Splitting changes the access pattern, not")
    print("the compulsory bytes, so extra traffic is v2's (m,l,acc) scratch:")
    print("  acc[B, n_q, splits, head_dim] fp32, written by partial, read by")
    print("  combine -- so it grows LINEARLY in the split count.")
    print("That is why 'moved' can climb while the duration stays flat: the")
    print("kernel moves more bytes for identical work. 'task' is the figure")
    print("that does not reward that, which is why it is the one to quote.")
    print("\nThis sweep runs in ONE session, so cross-session contention is")
    print("excluded by construction. To test whether a shape is contention-")
    print("sensitive, re-run it and compare -- B=384 reproduced to 0.2%.")
    print("=" * 74)


def _sweep_table(rows: list[dict], args) -> None:
    print("=" * 78)
    print("SWEEP -- where does attention traffic converge on its footprint?")
    print("=" * 78)
    hdr = (f"{'B':>5}{'logical':>10}{'xL2':>7}{'dram MB':>10}{'%log':>7}"
           f"{'us':>9}{'us/MB':>8}")
    if args.bw:
        hdr += f"{'task':>7}{'moved':>7}"
    print(hdr)
    for r in rows:
        line = (f"{r['B']:>5}{r['logical']/1e6:>10.1f}"
                f"{r['logical']/L2_BYTES:>7.2f}{r['dram']/1e6:>10.1f}"
                f"{r['dram']/r['logical']*100:>7.0f}{r['dur']/1e3:>9.2f}"
                f"{r['dur']/1e3/(r['logical']/1e6):>8.3f}")
        if args.bw:
            task = r["logical"]/(r["dur"]*1e-9)/1e9
            moved = r["dram"]/(r["dur"]*1e-9)/1e9
            line += f"{task/args.bw*100:>6.0f}%{moved/args.bw*100:>6.0f}%"
        print(line)
    print("-" * 78)
    print("'task' prices the duration against the COMPULSORY bytes -- the work")
    print("that must happen -- so it is the efficiency figure. 'moved' uses all")
    print("traffic, so overhead bytes inflate it; it is utilisation, not")
    print("efficiency. %log is the model check: under 100 means cache absorbed")
    print("some, over 100 means the kernel moved more than the task requires.")
    print("us/MB is the cleanest cross-shape comparison, since it needs no peak.")
    good = [r for r in rows if 0.9 <= r["dram"]/r["logical"] <= 1.1]
    if good:
        b = min(good, key=lambda r: r["dur"]/r["logical"])
        print(f"\nBest efficiency: B={b['B']} at "
              f"{b['dur']/1e3/(b['logical']/1e6):.3f} us/MB", end="")
        if args.bw:
            task = b["logical"]/(b["dur"]*1e-9)/1e9
            print(f" = {task/args.bw*100:.0f}% of peak.")
            print("If that is well under 100%, the kernel is latency-bound and")
            print("the next step is `ncu --set full` stall reasons, not more bytes.")
        else:
            print(".")
        spread = (max(r["dur"]/r["logical"] for r in good)
                  / min(r["dur"]/r["logical"] for r in good))
        if spread > 1.2:
            print(f"\nNote: us/MB varies {spread:.2f}x across valid rows. A")
            print("bandwidth-bound kernel would be near-flat, so this spread is")
            print("itself a finding -- compare stall reasons at the extremes.")
    else:
        print("\nNo row landed within 10% of its footprint; see %log above.")
    print("=" * 78)
    print("\nRecord in logs/EXPERIMENTS.md as the corrected Phase 4 bandwidth row.")


if __name__ == "__main__":
    raise SystemExit(main())