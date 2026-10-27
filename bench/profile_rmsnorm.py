# ==========================================================================
# FILE LOCATION:  flashdecode/bench/profile_rmsnorm.py
# RUN AS:         python -m bench.profile_rmsnorm            (from project root)
#                 python -m bench.profile_rmsnorm --batch 2048
# REQUIRES:       ncu on PATH (ships with the CUDA toolkit:
#                 /usr/local/cuda-13.0/bin/ncu)
# ==========================================================================
"""
Profile the rmsnorm kernel with hardware counters instead of wall-clock.

WHY THIS EXISTS. bench/kernel_rmsnorm.py tried to measure achieved bandwidth
by timing a loop and dividing bytes by seconds. Three successive attempts all
failed, in different ways:

  attempt 1  reused one buffer      -> 4829 GB/s, 292% of peak (pure L2)
  attempt 2  rotated 3x L2          -> 2374 GB/s, 143% of peak (still L2)
  attempt 3  rotated 6x L2 + graph  -> kernel 57/78/64% across three shapes
                                       that should be identical, and a plain
                                       torch copy_ control came back at 38%
                                       of peak, which is nonsense

The lesson is not "try harder at the arithmetic". Wall-clock bandwidth is the
wrong instrument: bytes/second computed from a timing loop cannot distinguish
traffic that came from DRAM from traffic that came from L2, and that
distinction IS the measurement. ncu reads the hardware counters and reports
DRAM sectors directly, so L2 hits simply do not appear as DRAM traffic.

The decisive metric is dram__bytes.sum. Compare it to the logical footprint
(4 * B * H * 2 bytes for this kernel):

  dram__bytes ~= logical   -> the kernel really is streaming from memory, and
                              dram__throughput.pct_of_peak is a trustworthy
                              efficiency number
  dram__bytes << logical   -> L2 served most of it; any wall-clock "bandwidth"
                              figure from that configuration was fiction

Also collected, because they answer the questions Phase 2 actually asked:
  achieved occupancy   -- ptxas said 8 blocks/SM fit; what happened in reality
  stall reasons        -- what the warps were waiting on
  SM vs DRAM throughput-- which unit is the limiter
"""

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
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",   # % of peak DRAM
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",     # % of peak SM
    "sm__warps_active.avg.pct_of_peak_sustained_active",    # achieved occupancy
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "launch__waves_per_multiprocessor",
    "gpu__time_duration.sum",
]

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2048,
                    help="rows. Use a saturated size (>= SMs*8) to judge the "
                         "body; use 128 to see what a decode step really does.")
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--count", type=int, default=3, help="launches to profile")
    ap.add_argument("--skip", type=int, default=5, help="warmup launches to skip")
    ap.add_argument("--ncu", default=None, help="path to ncu")
    ap.add_argument("--raw", action="store_true", help="print ncu output verbatim")
    args = ap.parse_args()

    ncu = args.ncu or shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
    if not os.path.exists(ncu) and not shutil.which(ncu):
        print(f"ncu not found at {ncu!r}.")
        print("It ships with the CUDA toolkit; try /usr/local/cuda-13.0/bin/ncu")
        print("or pass --ncu <path>.")
        return 1

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    driver = DRIVER % {"root": root, "B": args.batch, "H": args.hidden,
                       "skip": args.skip, "count": args.count}

    cmd = [
        ncu,
        "--metrics", ",".join(METRICS),
        "--kernel-name", "regex:rmsnorm_residual_kernel",
        "--launch-skip", str(args.skip),
        "--launch-count", str(args.count),
        "--csv",
        "--page", "raw",
        sys.executable, "-c", driver,
    ]

    print(f"profiling B={args.batch} H={args.hidden} "
          f"(logical footprint {4*args.batch*args.hidden*2/1e6:.1f} MB)\n")
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
            return proc.returncode

    ok = _summarize(proc.stdout, args.batch, args.hidden)
    if not ok:
        print("\n--- ncu stdout (first 25 lines) ---")
        for line in proc.stdout.splitlines()[:25]:
            print("  " + line)
        if proc.stderr.strip():
            print("--- ncu stderr (first 10 lines) ---")
            for line in proc.stderr.splitlines()[:10]:
                print("  " + line)
        print("\nIf those lines show metric data, the parser needs adjusting.")
        print("If they show a permissions error, see ERR_NVGPUCTRPERM above.")
        return 1
    return 0


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


def _summarize(csv_text: str, B: int, H: int) -> bool:
    vals = _parse(csv_text)
    if not vals:
        print("could not parse any metrics out of the ncu output.")
        return False

    def avg(k: str):
        # Tolerate ncu decorating names, e.g. a trailing unit in the header.
        v = vals.get(k)
        if v is None:
            for name, got in vals.items():
                if name.startswith(k):
                    v = got
                    break
        return sum(v) / len(v) if v else None

    logical = 4 * B * H * 2
    dram = avg("dram__bytes.sum")

    print("=" * 68)
    print(f"logical footprint      {logical/1e6:>10.2f} MB  (4 tensors x B x H x 2B)")
    if dram is not None:
        print(f"dram__bytes.sum        {dram/1e6:>10.2f} MB  "
              f"({dram/logical*100:.0f}% of logical)")
        if dram < logical * 0.6:
            print("  -> L2 served most of this. Any wall-clock bandwidth number")
            print("     measured at this shape is meaningless. Increase B or")
            print("     rotate buffers until dram__bytes approaches logical.")
        elif dram > logical * 1.3:
            print("  -> more DRAM traffic than the data requires: uncoalesced")
            print("     access or write-allocate reading lines before overwriting.")
        else:
            print("  -> genuinely streaming from DRAM; the throughput figures")
            print("     below are trustworthy.")
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
    dur = avg("gpu__time_duration.sum")
    if dur and dram:
        print(f"kernel duration        {dur/1e3:>10.2f} us")
        print(f"achieved DRAM BW       {dram/(dur*1e-9)/1e9:>10.0f} GB/s"
              "   <-- the honest number")
    print("=" * 68)
    missing = [m for m in METRICS if avg(m) is None]
    if missing:
        print(f"\n[warn] {len(missing)} metric(s) not reported: {missing}")
        print("       Some counters are unavailable on consumer parts; the")
        print("       dram__bytes and duration rows are the ones that matter.")
    print("\nRecord these in logs/EXPERIMENTS.md as the Phase 2 ncu row.")
    return True


if __name__ == "__main__":
    raise SystemExit(main())