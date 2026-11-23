from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys

# One metric per question. Keeping the list short keeps ncu's replay count
# low, which matters because ncu replays each kernel once per metric pass.
METRICS = [
    "dram__bytes.sum",                                      # ACTUAL DRAM traffic
    # Direction split. dram__bytes_{read,write}.sum are NOT available on
    # sm_120 (both came back n/a on a 5090); dram__sectors_{read,write}.sum
    # are the fallback, 32 B per sector. Run --query to see what this card
    # actually exposes. When none resolve, the split is DERIVED instead --
    # see _rw_bytes().
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",   # % of peak DRAM
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",     # % of peak SM
    "sm__warps_active.avg.pct_of_peak_sustained_active",    # achieved occupancy
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "launch__waves_per_multiprocessor",
    "gpu__time_duration.sum",
]

# Batch sizes for --sweep. The point of the large end is to get the working
# set clear of L2: at H=1536 the footprint is 4*B*H*2 bytes, so
#   B= 2048 ->  25 MB  (0.25x L2)  entirely cache-resident, writes absorbed
#   B= 8192 -> 101 MB  (1.0x  L2)  exactly L2-sized, still not safe
#   B=32768 -> 403 MB  (4.0x  L2)  writes must drain during the kernel
#   B=65536 -> 805 MB  (8.0x  L2)  the row a bandwidth claim can rest on
# Only rows at >=2x L2 support a statement about DRAM efficiency.
SWEEP_BATCHES = [2048, 8192, 32768, 65536]

DRIVER = r'''
import sys, torch
sys.path.insert(0, %(root)r)
from engine import ops_cuda
assert ops_cuda.AVAILABLE, ops_cuda.status()
from engine.ops import _REGISTRY
k = _REGISTRY["rmsnorm_residual"]["cuda"]
B, H = %(B)d, %(H)d
x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
r = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
w = torch.randn(H, device="cuda", dtype=torch.bfloat16)
# A few warmup launches, then the ones ncu profiles. ncu is told to skip the
# warmups with --launch-skip so it never profiles a cold cache.
for _ in range(%(skip)d + %(count)d):
    k(x, r, w, 1e-6)
torch.cuda.synchronize()
print("driver ok")
'''


def logical_split(B: int, H: int) -> tuple[int, int]:
    """(read bytes, write bytes) a perfect kernel would move.

    Reads  : x [B,H] + residual [B,H] + weight [H], the weight ONCE -- a
             byte floor charges each distinct byte a single time, however
             many blocks happen to request it.
    Writes : out [B,H] + new_residual [B,H].
    """
    read = 2 * B * H * 2 + H * 2
    write = 2 * B * H * 2
    return read, write


def _rw_bytes(avg, r_log: int, dram: float | None):
    """(read bytes, write bytes, how) -- measured if the card exposes a
    direction split, otherwise DERIVED from the aggregate.

    The derivation rests on one assumption, which --cache-control all makes
    true: L1 and L2 are flushed before the profiled launch, so every logical
    input byte must come from DRAM and the read half is pinned at r_log.
    Whatever dram__bytes.sum reports beyond that is write-back that drained
    inside the kernel window.

    The assumption is checkable, and it checked out: at B=2048 the aggregate
    came in at 12.59 MB against a 12.586 MB read footprint -- 100.03%, with
    writes contributing nothing at all. A derived split that lands on the
    read footprint to three decimal places is not a guess.

    Returns bytes, plus "measured" or "derived" so nothing downstream can
    quietly present one as the other.
    """
    for rk, wk, scale in (("dram__bytes_read.sum", "dram__bytes_write.sum", 1.0),
                          ("dram__sectors_read.sum", "dram__sectors_write.sum", 32.0)):
        rd, wr = avg(rk), avg(wk)
        if rd is not None and wr is not None:
            return rd * scale, wr * scale, "measured"
    if dram is None:
        return None, None, "unavailable"
    # Derived. Clamp at zero: a tiny negative means the reads themselves were
    # partly cached, which the caller reports separately.
    return min(dram, float(r_log)), max(0.0, dram - r_log), "derived"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2048,
                    help="rows. Use a saturated size (>= SMs*8) to judge the "
                         "body; use 128 to see what a decode step really does.")
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--sweep", action="store_true",
                    help=f"profile {SWEEP_BATCHES} instead of one shape, to "
                         "find where the working set clears L2")
    ap.add_argument("--bw", type=float, default=None,
                    help="measured HBM bandwidth in GB/s (bench.bandwidth). "
                         "No default on purpose: without it the SOL verdict "
                         "is skipped rather than computed against a guess.")
    ap.add_argument("--l2-mb", type=float, default=101.0,
                    help="L2 size in MB, for the working-set ratio (5090: 101)")
    ap.add_argument("--count", type=int, default=3, help="launches to profile")
    ap.add_argument("--skip", type=int, default=5, help="warmup launches to skip")
    ap.add_argument("--ncu", default=None, help="path to ncu")
    ap.add_argument("--raw", action="store_true", help="print ncu output verbatim")
    ap.add_argument("--query", action="store_true",
                    help="list the DRAM/L2 metrics this card exposes and exit "
                         "(cheap: no kernel is profiled)")
    args = ap.parse_args()

    ncu = args.ncu or shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
    if not os.path.exists(ncu) and not shutil.which(ncu):
        print(f"ncu not found at {ncu!r}.")
        print("It ships with the CUDA toolkit; try /usr/local/cuda-13.0/bin/ncu")
        print("or pass --ncu <path>.")
        return 1

    if args.query:
        return _query_metrics(ncu)

    if args.bw is None:
        print("[note] no --bw given, so the SOL verdict is skipped. Run")
        print("       python -m bench.bandwidth   and pass the read figure.\n")

    batches = SWEEP_BATCHES if args.sweep else [args.batch]
    rows: list[dict] = []
    failures = 0

    for i, B in enumerate(batches):
        if len(batches) > 1:
            print(f"\n[{i+1}/{len(batches)}] B={B}")
        got = _profile_one(ncu, B, args)
        if got is None:
            failures += 1
        else:
            rows.append(got)

    if len(rows) > 1:
        _sweep_table(rows, args)

    if not rows:
        return 1
    return 1 if failures else 0


def _query_metrics(ncu: str) -> int:
    """Ask ncu which DRAM/L2 counters exist here, instead of guessing.

    dram__bytes_read.sum and dram__bytes_write.sum are documented but absent
    on sm_120, which cost one full sweep to discover. This is the cheap way
    to find the right names on a new card.
    """
    proc = subprocess.run([ncu, "--query-metrics"], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    want = ("dram__bytes", "dram__sectors", "lts__t_sectors_aperture_device")
    hits = [l for l in proc.stdout.splitlines()
            if any(w in l for w in want)]
    print(f"{len(hits)} DRAM/L2 metric(s) exposed by this device:\n")
    for line in hits:
        print("  " + line.strip())
    print("\nLooking for a read/write pair. Put whichever pair exists at the")
    print("front of the name list in _rw_bytes(); 'sectors' variants are")
    print("32 bytes each. If no pair exists, the derived split is the answer.")
    return 0


def _profile_one(ncu: str, B: int, args) -> dict | None:
    """Profile one batch size. Returns parsed metrics, or None on failure."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    driver = DRIVER % {"root": root, "B": B, "H": args.hidden,
                       "skip": args.skip, "count": args.count}

    cmd = [
        ncu,
        "--metrics", ",".join(METRICS),
        "--kernel-name", "regex:rmsnorm_residual_kernel",
        "--launch-skip", str(args.skip),
        "--launch-count", str(args.count),
        # Be explicit rather than relying on the default: flush L1 and L2
        # before each profiled launch, so the READ half is genuinely cold and
        # dram__bytes_read is comparable to the logical read footprint. With
        # caches left warm, back-to-back launches on the same buffers serve
        # reads out of L2 and the read row collapses for a second, unrelated
        # reason -- making the write-absorption question unanswerable.
        "--cache-control", "all",
        "--csv",
        "--page", "raw",
        sys.executable, "-c", driver,
    ]

    r_log, w_log = logical_split(B, args.hidden)
    print(f"profiling B={B} H={args.hidden}  "
          f"(logical {(r_log+w_log)/1e6:.1f} MB = {r_log/1e6:.1f} R "
          f"+ {w_log/1e6:.1f} W)\n")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if args.raw or proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        if proc.returncode != 0:
            print("\nncu failed. The usual cause is profiling permissions:")
            print("  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\")
            print("    | sudo tee /etc/modprobe.d/nvidia-profiler.conf")
            print("  sudo update-initramfs -u && sudo reboot")
            print("Or run under sudo. See NVIDIA ERR_NVGPUCTRPERM.")
            return None

    vals = _summarize(proc.stdout, B, args)
    if vals is None:
        print("\n--- ncu stdout (first 25 lines) ---")
        for line in proc.stdout.splitlines()[:25]:
            print("  " + line)
        if proc.stderr.strip():
            print("--- ncu stderr (first 10 lines) ---")
            for line in proc.stderr.splitlines()[:10]:
                print("  " + line)
        print("\nIf those lines show metric data, the parser needs adjusting.")
        print("If they show a permissions error, see ERR_NVGPUCTRPERM above.")
        return None
    vals["B"] = B
    return vals


def _parse(csv_text: str) -> dict[str, list[float]]:
    """Parse ncu --csv output into {metric_name: [values]}.

    ncu emits TWO different CSV shapes and which one you get depends on
    --page, which is exactly the trap the first version of this fell into:

      --page raw      WIDE  -- one row per kernel launch, one COLUMN per
                               metric, values as plain unscaled numbers
      --page details  LONG  -- one row per (launch, metric), with
                               "Metric Name" / "Metric Value" / "Metric Unit"
                               columns and values scaled by the unit prefix

    This handles both. It also skips ncu's preamble lines, which are not CSV
    at all but do contain commas, so a naive "line contains a comma" filter
    feeds them to the reader and produces garbage rows.

    NOTE: bench/profile_paged.py imports this function. Keep the signature.
    """
    import csv as _csv
    import io

    lines = csv_text.splitlines()
    # The real CSV starts at the header row, which always names the kernel.
    start = next((i for i, l in enumerate(lines)
                  if "Kernel Name" in l or "ID" == l.split(",")[0].strip('"')), None)
    if start is None:
        return {}

    rows = list(_csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    if not rows:
        return {}

    vals: dict[str, list[float]] = {}

    def add(name: str, raw: str, unit: str = "") -> None:
        raw = (raw or "").replace(",", "").strip()
        if not raw or raw in ("n/a", "N/A", "-"):
            return
        try:
            v = float(raw)
        except ValueError:
            return
        # Long format scales by unit prefix; raw format does not.
        scale = {"Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
                 "Kbyte/s": 1e3, "Mbyte/s": 1e6, "Gbyte/s": 1e9,
                 "usecond": 1e3, "msecond": 1e6, "nsecond": 1.0}.get(unit.strip(), 1.0)
        vals.setdefault(name.strip(), []).append(v * scale)

    if "Metric Name" in rows[0]:                      # LONG
        for r in rows:
            add(r.get("Metric Name", ""), r.get("Metric Value", ""),
                r.get("Metric Unit", ""))
    else:                                             # WIDE
        skip = {"ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
                "Kernel Time", "Context", "Stream", "Block Size", "Grid Size",
                "Device", "CC", "Section Name"}
        for r in rows:
            for col, raw in r.items():
                if col and col not in skip:
                    add(col, raw)
    return vals


def _summarize(csv_text: str, B: int, args) -> dict | None:
    vals = _parse(csv_text)
    if not vals:
        print("could not parse any metrics out of the ncu output.")
        return None

    def avg(k: str):
        # Tolerate ncu decorating names, e.g. a trailing unit in the header.
        v = vals.get(k)
        if v is None:
            for name, got in vals.items():
                if name.startswith(k):
                    v = got
                    break
        return sum(v) / len(v) if v else None

    H = args.hidden
    r_log, w_log = logical_split(B, H)
    logical = r_log + w_log
    dram = avg("dram__bytes.sum")
    rd, wr, how = _rw_bytes(avg, r_log, dram)
    dur = avg("gpu__time_duration.sum")           # nanoseconds
    l2 = args.l2_mb * 1e6

    out = {"dram": dram, "read": rd, "write": wr, "how": how, "dur": dur,
           "logical": logical, "r_log": r_log, "w_log": w_log}

    print("=" * 72)
    print(f"working set            {logical/1e6:>10.2f} MB  "
          f"({logical/l2:.2f}x the {args.l2_mb:.0f} MB L2)")
    if logical < 2 * l2:
        print("  -> FITS IN L2. Writes need never reach DRAM while the kernel")
        print("     runs, and dram__bytes only counts sectors inside the")
        print("     kernel window, so the write row below will undercount and")
        print("     any efficiency figure from this shape is not about DRAM.")
        print("     Use --sweep, or B >= %d, for a defensible row."
              % (int(2 * l2 / (4 * H * 2)) + 1))

    # ---- the read/write split: the measurement that settles it -------------
    print("-" * 72)
    tag = {"measured": "dram measured", "derived": "dram (derived)",
           "unavailable": "dram"}[how]
    print(f"{'':<10}{'logical':>14}{tag:>16}{'ratio':>10}")
    for label, lg, got in (("reads", r_log, rd), ("writes", w_log, wr),
                           ("total", logical, dram)):
        if got is None:
            print(f"{label:<10}{lg/1e6:>12.2f} MB{'n/a':>16}")
            continue
        print(f"{label:<10}{lg/1e6:>12.2f} MB{got/1e6:>13.2f} MB"
              f"{got/lg*100:>9.0f}%")
    if how == "derived":
        print("  [the direction counters are absent on this card, so the split")
        print("   is derived: reads are pinned at the logical footprint because")
        print("   --cache-control all flushed the caches, and the remainder of")
        print("   dram__bytes.sum is write-back that drained in-window]")

    if rd is not None and wr is not None:
        if wr < 0.9 * w_log:
            print(f"\n  -> WRITE-BACK ABSORBED: only {wr/w_log*100:.0f}% of the "
                  f"stores reached DRAM inside the")
            print("     kernel window. The kernel retired before its own writes")
            print("     drained, so `duration` excludes work the byte floor")
            print("     charges in full. This is what produces a spurious")
            print("     >100%-of-SOL reading. Not a fast kernel; a short window.")
        if rd < 0.9 * r_log:
            print(f"\n  -> READS SERVED BY CACHE: {rd/r_log*100:.0f}% of the "
                  f"logical read footprint hit DRAM.")
            print("     With --cache-control all this should be ~100%; if it is")
            print("     not, the inputs are smaller than they look or the launch")
            print("     being profiled is not the one assumed.")
        if rd > 1.15 * r_log:
            print("\n  -> EXCESS READ TRAFFIC: uncoalesced access, or")
            print("     write-allocate pulling lines in before overwriting them.")

    # ---- throughput and occupancy -----------------------------------------
    print("-" * 72)
    for label, key in [
        ("DRAM throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("SM throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("achieved occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"),
        ("waves per SM", "launch__waves_per_multiprocessor"),
    ]:
        v = avg(key)
        if v is not None:
            unit = "" if "waves" in key else "%"
            print(f"{label:<22} {v:>10.2f}{unit}")
            if "waves" in key and v < 1.0:
                print(f"  -> under one wave: {(1-v)*100:.0f}% of the card idle. "
                      "Launch-bound, not bandwidth-bound.")

    # ---- the SOL verdict ---------------------------------------------------
    if dur is not None:
        dur_us = dur / 1e3
        print("-" * 72)
        print(f"kernel duration        {dur_us:>10.2f} us")
        if dram is not None:
            print(f"achieved DRAM BW       {dram/(dur*1e-9)/1e9:>10.0f} GB/s"
                  "   (measured bytes / duration)")
        if args.bw:
            sol_us = logical / (args.bw * 1e9) * 1e6
            pct = sol_us / dur_us * 100
            # Floor for the bytes that ACTUALLY moved. Needed by the
            # decomposition below, so compute it before anything prints.
            sol_meas = dram / (args.bw * 1e9) * 1e6 if dram else None
            print(f"SOL @ {args.bw:.0f} GB/s      {sol_us:>10.2f} us  "
                  f"-> {pct:.0f}% of SOL")
            # %SOL factors exactly into two terms, and printing them is worth
            # more than printing %SOL itself:
            #
            #   %SOL = (logical / measured) x (efficiency on measured traffic)
            #        =  absorption factor   x  how good the kernel really is
            #
            # The first term is pure measurement error -- bytes the floor was
            # charged for that never crossed the bus. The second is the real
            # result. They move in opposite directions as B grows (absorption
            # falls, efficiency rises), which is why %SOL is not monotonic and
            # why the worst reading is NOT at the smallest shape.
            if sol_meas is not None and dram:
                absorb = logical / dram
                print(f"\n  %SOL decomposes: {absorb:.2f}x absorption"
                      f" x {sol_meas/dur_us:.2f} efficiency = {absorb*sol_meas/dur_us:.2f}")
            if pct > 100.0:
                print("\n  *** >100% OF SOL IS NOT A RESULT. ***")
                print("  A kernel cannot move its bytes faster than the memory")
                print("  system allows. Something in the comparison is unequal.")
                if dram is not None:
                    print(f"  Here: SOL is priced on {logical/1e6:.1f} MB but only"
                          f" {dram/1e6:.1f} MB crossed")
                    print("  the bus in that duration. The gap is cache, not speed.")
                # The sting in the tail: a kernel that is BAD hides this,
                # because its own inefficiency cancels part of the inflation.
                # So a plausible-looking %SOL says nothing about whether the
                # measurement is sound. Improving the kernel makes the lie
                # louder, not quieter.
                if sol_meas is not None and sol_meas / dur_us > 0.7:
                    print("  Note: this shape reads worse than a less efficient")
                    print("  one would, because there is less inefficiency left")
                    print("  to cancel the inflation. Plausibility is not a check.")
            # The honest per-shape efficiency: how close the kernel came to
            # moving the bytes it ACTUALLY moved at full speed. Immune to the
            # absorption artifact, because both sides use measured traffic.
            if dram is not None:
                sol_meas = dram / (args.bw * 1e9) * 1e6
                print(f"\nefficiency on measured traffic: "
                      f"{sol_meas/dur_us*100:.0f}% "
                      f"({sol_meas:.2f} us floor for {dram/1e6:.1f} MB)")
                out["eff"] = sol_meas / dur_us * 100
    print("=" * 72)

    missing = [m for m in METRICS if avg(m) is None]
    if missing:
        print(f"\n[warn] {len(missing)} metric(s) not reported: {missing}")
        print("       Some counters are unavailable on consumer parts; the")
        print("       dram__bytes rows and duration are the ones that matter.")
    return out


def _qualifies(r: dict, l2: float) -> bool:
    """Is this row's duration a fair thing to price against a byte floor?

    Two conditions, and the second has a fallback. The working set must clear
    L2 by 2x, and the stores must have actually drained in-window. When the
    card exposes no direction counters the aggregate stands in: if total DRAM
    traffic is within 10% of the logical footprint then little was absorbed,
    whichever direction it went.

    Without the fallback a missing counter silently disqualifies every row --
    which is what the first sweep did, rejecting a perfectly good B=65536 at
    94% of logical traffic because W% read as 0 rather than as unknown.
    """
    if r["logical"] < 2 * l2:
        return False
    if r.get("how") == "measured" and r["write"] is not None:
        return r["write"] >= 0.9 * r["w_log"]
    return (r["dram"] or 0) >= 0.9 * r["logical"]


def _sweep_table(rows: list[dict], args) -> None:
    """One table across batch sizes, so the L2 crossover is visible."""
    l2 = args.l2_mb * 1e6
    derived = any(r.get("how") == "derived" for r in rows)
    print("\n" + "=" * 78)
    print("SWEEP -- find the first row whose working set clears L2")
    print("=" * 78)
    hdr = (f"{'B':>7}{'set MB':>9}{'xL2':>7}{'dram MB':>10}{'%log':>7}"
           f"{'W drained':>11}{'us':>9}{'GB/s':>8}")
    if args.bw:
        hdr += f"{'%SOL':>7}{'eff':>6}"
    print(hdr)
    for r in rows:
        dur_us = r["dur"] / 1e3 if r["dur"] else float("nan")
        dram = r["dram"] or 0.0
        bw = dram / (r["dur"] * 1e-9) / 1e9 if r["dur"] else float("nan")
        wpct = (f"{r['write']/r['w_log']*100:.0f}%"
                if r["write"] is not None else "n/a")
        line = (f"{r['B']:>7}{r['logical']/1e6:>9.1f}{r['logical']/l2:>7.2f}"
                f"{dram/1e6:>10.1f}{dram/r['logical']*100:>7.0f}"
                f"{wpct:>11}{dur_us:>9.2f}{bw:>8.0f}")
        if args.bw:
            sol_us = r["logical"] / (args.bw * 1e9) * 1e6
            line += f"{sol_us/dur_us*100:>7.0f}{r.get('eff', float('nan')):>6.0f}"
        print(line)
    print("-" * 78)
    print("%log is measured DRAM traffic over the logical footprint. Well under")
    print("100 means traffic was absorbed by L2, so that row's duration -- and")
    print("any %SOL built on it -- understates the real cost. 'eff' prices the")
    print("duration against the bytes that DID move, so it is immune to that.")
    if derived:
        print("W drained is derived, not measured: this card exposes no")
        print("dram__bytes_{read,write} counters. See --query and _rw_bytes().")
    usable = [r for r in rows if _qualifies(r, l2)]
    if usable:
        print(f"\nDefensible rows: B = {', '.join(str(r['B']) for r in usable)}")
        if args.bw:
            best = max(usable, key=lambda r: r["logical"])
            dur_us = best["dur"] / 1e3
            sol_us = best["logical"] / (args.bw * 1e9) * 1e6
            logical_rate = best["logical"] / (dur_us * 1e-6) / 1e9
            print(f"\nPhase 2 headline, B={best['B']} "
                  f"({best['logical']/1e6:.0f} MB, {best['logical']/l2:.1f}x L2, "
                  f"{best['dram']/best['logical']*100:.0f}% of traffic reaching DRAM):")
            print(f"  {sol_us/dur_us*100:.0f}% of the logical byte floor")
            print(f"  {logical_rate:.0f} GB/s logical-byte rate "
                  f"= {logical_rate/args.bw*100:.0f}% of measured peak")
            print(f"  {best['dram']/(best['dur']*1e-9)/1e9:.0f} GB/s on counted "
                  f"sectors = {best.get('eff', float('nan')):.0f}% of peak")
            print("Quote the range. The spread between the last two lines is")
            print("exactly the traffic still being absorbed, so both are true")
            print("and neither alone is the whole picture.")
    else:
        print("\nNo row qualified. Raise B until the working set is >=2x L2 and")
        print("%log reaches ~100.")
    print("=" * 78)
    print("\nRecord these in logs/EXPERIMENTS.md as the corrected Phase 2 row.")


if __name__ == "__main__":
    raise SystemExit(main())