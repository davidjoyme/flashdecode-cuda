"""
Achievable HBM bandwidth. GPU required.

Every speed-of-light claim divides by a bandwidth number. Use the MEASURED
one, never the spec sheet. Real kernels top out around 80-93% of spec
depending on architecture, so quoting spec makes your own kernels look worse
than they are AND makes you look like you did not check.

    python -m bench.bandwidth

Record the result in logs/EXPERIMENTS.md with the device name. Re-run if you
change GPUs; a thermally throttled card reads ~10% low.
"""

from __future__ import annotations

import torch


def time_ms(fn, warmup=20, iters=100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def measure(n_bytes: int = 1 << 30, dtype=torch.bfloat16) -> dict:
    """Three access patterns, because the honest denominator depends on which
    one your kernel resembles."""
    n = n_bytes // torch.tensor([], dtype=dtype).element_size()
    a = torch.randn(n, dtype=dtype, device="cuda")
    b = torch.empty_like(a)
    c = torch.empty_like(a)
    out = {}
    out["copy"] = 2 * n_bytes / (time_ms(lambda: b.copy_(a)) * 1e-3) / 1e9
    # read-only reduction: closest to what a decode-attention kernel does
    out["read"] = n_bytes / (time_ms(lambda: torch.sum(a)) * 1e-3) / 1e9
    out["triad"] = 3 * n_bytes / (time_ms(lambda: torch.add(a, b, out=c)) * 1e-3) / 1e9
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device -- this script requires a GPU")
    p = torch.cuda.get_device_properties(0)
    print(f"device : {p.name}")
    print(f"SMs    : {p.multi_processor_count}")
    print(f"VRAM   : {p.total_memory / 1e9:.1f} GB")
    print(f"cc     : {p.major}.{p.minor}\n")
    r = measure()
    for k, v in r.items():
        print(f"{k:<8} {v:8.1f} GB/s")
    print(f"\nUSE THIS as the SOL denominator: {max(r.values()):.1f} GB/s")
    print("Record it in logs/EXPERIMENTS.md with the device name.")