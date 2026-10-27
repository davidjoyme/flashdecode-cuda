# ==========================================================================
# FILE LOCATION:  flashdecode/bench/kernel_rmsnorm.py
# RUN AS:         python -m bench.kernel_rmsnorm --bw 1655
#                 (from the project root; needs bench/__init__.py to exist,
#                  and kernels/fd_kernels*.so to have been built)
# ==========================================================================
"""
Phase 2 kernel harness: correctness, SOL, and the launch-overhead story.

    python -m bench.kernel_rmsnorm --bw 1655

Three timings per shape, and the gap between them is the finding:

  amortized   many launches back to back, queue stays full. This is the
              number that reflects the KERNEL's quality.
  per-launch  one launch, synchronized. Includes dispatch overhead. This is
              what the kernel actually costs in an eager decode loop.
  graphed     replayed inside a CUDA graph. Shows how much of the per-launch
              cost was dispatch rather than work.

At decode batch sizes this kernel moves ~1 MB, which is well under a
microsecond of work against several microseconds of launch overhead. Expect
amortized to look great and per-launch to look terrible. That is not a
broken benchmark -- it is the reason fusion and CUDA graphs matter more than
shaving the last 5% of bandwidth here.
"""

from __future__ import annotations

import argparse

import torch

from engine import ops

try:
    from engine import ops_cuda  # noqa: F401  (registers the kernel)

    HAVE_CUDA_IMPL = ops_cuda.AVAILABLE
    IMPL_STATUS = ops_cuda.status()
except Exception as _exc:
    HAVE_CUDA_IMPL = False
    IMPL_STATUS = f"engine.ops_cuda failed to import: {_exc}"


def ref(x, residual, weight, eps):
    return ops.rmsnorm_residual_ref(x, residual, weight, eps)


def cuda(x, residual, weight, eps):
    return ops._REGISTRY["rmsnorm_residual"]["cuda"](x, residual, weight, eps)


def time_amortized(fn, iters=200, warmup=50) -> float:
    """Back-to-back launches. Measures the kernel, not the dispatch."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def time_per_launch(fn, iters=200, warmup=50) -> float:
    """Sync after every call -- a CPU-side wall-clock measurement.

    MEDIAN, not mean. This path times python call + pybind + two
    torch::empty_like allocations + launch + full device sync, so a single
    scheduler hiccup or an allocator miss that needs a real cudaMalloc adds
    hundreds of microseconds. The first version of this used the mean and
    reported 35 to 532 us across batch sizes with no monotonic pattern --
    pure noise. The median is stable and is the number an eager decode loop
    actually experiences.
    """
    import statistics
    import time

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def time_graphed(fn, iters=200, warmup=50):
    """Capture into a CUDA graph and replay. Dispatch cost mostly vanishes."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1e3
    except Exception as exc:
        print(f"    [graph capture failed: {type(exc).__name__}: {exc}]")
        return float("nan")


def check(B, H, device="cuda", dtype=torch.bfloat16) -> dict:
    torch.manual_seed(0)
    x = torch.randn(B, H, device=device, dtype=dtype)
    res = torch.randn(B, H, device=device, dtype=dtype)
    w = torch.randn(H, device=device, dtype=dtype)
    eps = 1e-5

    o_ref, r_ref = ref(x, res, w, eps)
    o_cu, r_cu = cuda(x, res, w, eps)

    # new_residual is a plain bf16 add in both paths -- it should match
    # EXACTLY. Any difference here means the kernel's add or its store is
    # wrong, which is a much more basic bug than a reduction mismatch.
    res_exact = torch.equal(r_ref, r_cu)
    o_err = (o_ref.float() - o_cu.float()).abs().max().item()
    o_cos = torch.nn.functional.cosine_similarity(
        o_ref.float().flatten(), o_cu.float().flatten(), dim=0
    ).item()
    return {"residual_exact": res_exact, "out_max_abs": o_err, "out_cos": o_cos}


def sweep(H: int, bw: float, sms: int, threads: int) -> None:
    """Asymptotic bandwidth: how good is the KERNEL BODY?

    MEASUREMENT CORRECTED 2026-10-27. The first version of this reported
    4829 GB/s against a 1655 GB/s HBM peak -- 292%, i.e. physically
    impossible. Two separate artifacts, both classics:

    1. L2 RESIDENCY. The timing loop re-read the SAME two buffers 100 times.
       This card has ~96 MB of L2, so any working set below that was served
       entirely from cache after the first iteration and the loop measured L2
       bandwidth (several TB/s), not HBM. Fixed by rotating over enough
       distinct buffer sets that the total exceeds L2 by ~3x, so nothing is
       still cached by the time it comes round again.

    2. ALLOCATOR CHURN. At B=16384 the working set finally exceeded L2, but
       every call also allocated two 50 MB output tensors; 100 iterations of
       that gave 332 us against a 122 us floor. That was the caching
       allocator, not bandwidth. Fixed by capturing the whole rotation into
       one CUDA graph, which uses a private memory pool and takes both the
       allocator and the dispatch out of the timed region.

    Worth keeping in the writeup: an above-peak bandwidth number is never a
    fast kernel, it is always a broken benchmark, and the usual cause is a
    hot cache.

    Why this is a separate measurement. The kernel runs one block per row, so
    the grid is exactly B blocks. ptxas reports 36 registers and 24 B of smem
    at THREADS=192, which allows 8 blocks per SM, so the machine does not fill
    until B >= sms * 8 (1360 on a 170-SM card). Every decode batch size is far
    below that: at B=128, 42 of 170 SMs get no work at all, and the whole
    kernel moves 1.5 MB -- less than 1 us of traffic. Under those conditions
    the measurement is dominated by launch and wave latency, and %SOL says
    nothing about the quality of the loads.

    Sweeping into the saturated regime separates the two questions:
      - is the kernel body efficient?        -> this sweep
      - is it worth optimising for decode?   -> the table above (it is not;
        there the win is launch count, which fusion already delivered)

    A good memory-bound kernel reaches 80-90% of measured peak here. If this
    plateaus much lower, the body has a real problem -- look at whether 8
    bf16 per thread is enough work to keep enough loads in flight, and try
    2 or 4 vectors per thread.
    """
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", 0) or 96 * 1024 * 1024
    # 6x L2, not 3x. At 3x, B=1360 still reported 143% of peak -- a rotation
    # wide enough on paper can still leave a useful fraction resident once
    # write-back traffic for the output buffers is counted, and any number
    # above peak means the cache is still helping.
    target = 6 * l2

    # ---- CALIBRATE THE HARNESS FIRST ----
    # Without this, a kernel reading 65% of peak is ambiguous: is the kernel
    # mediocre, or is this measurement method incapable of showing more?
    # torch's copy_ is a pure, well-optimised stream. Running it through the
    # SAME rotation and the SAME graph timing gives the harness's own ceiling,
    # and every kernel number below is quoted relative to it.
    cal_gbs = 0.0
    try:
        # n buffer PAIRS; each pair moves 2*sz*2 bytes (read src + write dst),
        # so the rotation footprint is n * 2 * sz * 2. Size sz so that total
        # reaches `target`, matching the kernel sweep's L2 margin.
        n = 8
        sz = int(target / (2 * 2 * n))        # elements per buffer (bf16)
        src = [torch.randn(sz, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
        dst = [torch.empty_like(src[0]) for _ in range(n)]

        def cal_round():
            for i in range(n):
                dst[i].copy_(src[i])

        for _ in range(3):
            cal_round()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cal_round()
        torch.cuda.synchronize()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for _ in range(20):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 20
        moved = 2 * 2 * sz * n                # read + write, bf16
        cal_gbs = moved / (ms * 1e-3) / 1e9
        print(f"\n  harness calibration (torch copy_, same rotation + graph):")
        print(f"    {cal_gbs:.0f} GB/s = {cal_gbs/bw*100:.0f}% of the {bw:.0f} GB/s "
              f"figure from bench.bandwidth")
        if cal_gbs > bw * 1.05:
            print(f"    WARNING: control is above peak too -- the rotation is still")
            print(f"    not defeating L2, so treat every number below as suspect.")
        del src, dst
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"\n  harness calibration failed: {exc}")

    print(f"\n--- asymptotic bandwidth (L2-defeating, graph-timed) ---")
    print(f"  L2 = {l2/1e6:.0f} MB; each shape rotates over enough distinct")
    print(f"  buffers to touch >= {target/1e6:.0f} MB, so nothing is still cached")
    print(f"  when it is read again. Grid == B blocks; {sms} SMs x 8 blocks/SM")
    print(f"  means saturation needs B >= {sms*8}.\n")
    print(f"{'B':>7} {'MB/call':>8} {'bufs':>5} {'total MB':>9} {'blk/SM':>7} "
          f"{'SOL us':>8} {'meas us':>8} {'GB/s':>7} {'%peak':>6} {'%ctrl':>6}")

    best = 0.0
    for B in (2048, 4096, 8192):
        per_call = 4 * B * H * 2                      # 2 reads + 2 writes
        # Inputs are 2 of the 4 tensors; graph-pool outputs add the other 2.
        nbuf = max(2, min(24, int(target / per_call) + 1))
        total = per_call * nbuf
        if total > 0.45 * props.total_memory:
            nbuf = max(2, int(0.45 * props.total_memory / per_call))
            total = per_call * nbuf
        sol = per_call / bw / 1e9 * 1e6
        try:
            xs = [torch.randn(B, H, device="cuda", dtype=torch.bfloat16) for _ in range(nbuf)]
            rs = [torch.randn(B, H, device="cuda", dtype=torch.bfloat16) for _ in range(nbuf)]
            w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

            def one_round():
                for i in range(nbuf):
                    cuda(xs[i], rs[i], w, 1e-6)

            # Capture the whole rotation in one graph: no allocator, no
            # dispatch, and nbuf distinct buffers so L2 cannot serve them.
            for _ in range(3):
                one_round()
            torch.cuda.synchronize()
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    one_round()
                torch.cuda.synchronize()
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                reps = 20
                s.record()
                for _ in range(reps):
                    g.replay()
                e.record()
                torch.cuda.synchronize()
                t = s.elapsed_time(e) / reps / nbuf * 1e3        # us per call
                how = ""
            except Exception:
                t = time_amortized(one_round, iters=20, warmup=5) / nbuf
                how = " (no graph)"

            gbs = per_call / (t * 1e-6) / 1e9
            best = max(best, gbs)
            pct_ctrl = f"{gbs/cal_gbs*100:>5.0f}%" if cal_gbs else "    - "
            print(f"{B:>7} {per_call/1e6:>8.1f} {nbuf:>5} {total/1e6:>9.0f} "
                  f"{B/sms:>7.1f} {sol:>8.2f} {t:>8.2f} {gbs:>7.0f} "
                  f"{gbs/bw*100:>5.0f}% {pct_ctrl}{how}")
            del xs, rs, w
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>7} {'OOM':>8}")
            torch.cuda.empty_cache()

    print(f"\n  best achieved: {best:.0f} GB/s = {best/bw*100:.0f}% of peak", end="")
    if cal_gbs:
        print(f", {best/cal_gbs*100:.0f}% of the control")
    else:
        print()

    if best > bw * 1.05:
        print("  -> STILL above peak: the measurement is wrong, not the kernel.")
        print("     Raise `target` further or check the reported L2 size.")
    elif cal_gbs and best / cal_gbs > 0.85:
        print("  -> within 15% of a pure torch copy through the identical harness,")
        print("     so the body is about as good as a simple stream gets here.")
        print("     Its poor %SOL at decode shapes is occupancy and latency, not")
        print("     the loads. The Phase 2 win is launch count and graph capture.")
    elif cal_gbs:
        print(f"  -> the control reached {cal_gbs:.0f} GB/s through the same harness,")
        print(f"     so roughly {(1-best/cal_gbs)*100:.0f}% is being left on the table by the")
        print("     kernel itself, not by the measurement.")
        print("\n     Most likely cause: too little memory-level parallelism per")
        print("     thread. Each thread issues one uint4 from x, one from residual,")
        print("     then must WAIT for both before it can square anything -- only")
        print("     two loads in flight per thread, which cannot cover HBM latency.")
        print("     Next experiment (v1): give each block several rows, so each")
        print("     thread has several independent loads outstanding. Grid becomes")
        print("     B/rows, which is fine at these sizes and irrelevant at decode")
        print("     shapes where the kernel is launch-bound anyway.")
        print("     Log the result either way -- a v1 that does NOT help is still")
        print("     a writeup entry, and this one may not, since prefill is the")
        print("     only regime where it would show.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, required=True, help="measured GB/s")
    ap.add_argument("--sweep", action="store_true",
                    help="also measure asymptotic bandwidth at large B")
    ap.add_argument("--hidden", type=int, default=1536)  # Qwen2.5-1.5B
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 8, 32, 64, 128, 213, 256])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    if not HAVE_CUDA_IMPL:
        print(IMPL_STATUS)
        print("\nBuild it with:  cd kernels && python setup.py build_ext --inplace")
        print("The .so lands in kernels/; engine/ops_cuda.py adds that directory")
        print("to sys.path, so no PYTHONPATH is needed.")
        return 1

    H = args.hidden
    print(IMPL_STATUS)
    print(f"{torch.cuda.get_device_properties(0).name}  H={H}  bw={args.bw:.0f} GB/s")
    print(f"SMs={torch.cuda.get_device_properties(0).multi_processor_count}\n")

    print("--- correctness ---")
    ok = True
    for B in [1, 7, 64, 256]:
        r = check(B, H)
        good = r["residual_exact"] and r["out_max_abs"] < 2e-2
        ok &= good
        print(f"  B={B:<4} residual_exact={str(r['residual_exact']):<5} "
              f"out_max_abs={r['out_max_abs']:.3e} cos={r['out_cos']:.6f} "
              f"{'ok' if good else 'FAIL'}")
    if not ok:
        print("\ncorrectness failed -- do not benchmark a wrong kernel")
        return 1

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    threads = H // 8

    print("\n--- decode shapes (us) ---")
    print("  blk/SM is the real story: grid == B blocks, so anything under 1.0")
    print("  leaves SMs idle and %SOL measures launch latency, not bandwidth.\n")
    print(f"{'B':>5} {'bytes':>9} {'blk/SM':>7} {'SOL':>7} {'ref':>8} {'amort':>8} "
          f"{'%SOL':>6} {'launch':>8} {'graph':>8} {'vs ref':>7}")
    for B in args.batches:
        x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
        res = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        byts = 4 * B * H * 2          # read x, read res; write out, write res
        sol = byts / args.bw / 1e9 * 1e6

        t_ref = time_amortized(lambda: ref(x, res, w, 1e-6))
        t_am = time_amortized(lambda: cuda(x, res, w, 1e-6))
        t_la = time_per_launch(lambda: cuda(x, res, w, 1e-6))
        t_gr = time_graphed(lambda: cuda(x, res, w, 1e-6))

        print(f"{B:>5} {byts:>9} {B/sms:>7.2f} {sol:>7.2f} {t_ref:>8.2f} {t_am:>8.2f} "
              f"{sol/t_am*100:>5.0f}% {t_la:>8.2f} {t_gr:>8.2f} {t_ref/t_am:>6.1f}x")
        del x, res, w
        torch.cuda.empty_cache()

    print("\nHow to read this:")
    print("  'amort' flat across small B  -> you are measuring a launch floor,")
    print("     not the kernel. Ignore %SOL in that range.")
    print("  'graph' == 'amort'           -> dispatch is fully hidden by graph")
    print("     capture; the remaining cost is the kernel plus wave latency.")
    print("  'vs ref'                     -> the real Phase 2 win. PyTorch runs")
    print("     the add and the norm as ~6 separate kernels; this is 1. That is")
    print("     a launch-count result, not a bandwidth result -- say so in the")
    print("     writeup rather than claiming a bandwidth speedup.")
    print("  Run with --sweep to find out whether the BODY is any good.")

    if args.sweep:
        sweep(H, args.bw, sms, threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())