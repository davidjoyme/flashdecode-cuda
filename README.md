<!-- ======================================================================
     FILE LOCATION: flashdecode/README.md
     ====================================================================== -->

# Paged GQA decode attention for consumer Blackwell (sm_120)

Hand-written CUDA kernels for LLM decode on an architecture no existing
attention library supports, with a minimal Qwen2.5 inference engine to run and
measure them end to end.

**The paged decode-attention kernel runs 6.2–6.7× faster than torch SDPA at
working sets beyond L2, and is bit-exact against HuggingFace inside the model.**

RTX 5090 (sm_120, 170 SMs, 101 MB L2, **1655 GB/s measured** read bandwidth,
92% of spec) · Qwen2.5-1.5B bf16 · 28 layers, hidden 1536, 12 query / 2 KV
heads (GQA group 6), head_dim 128.

> Timings came from a GPU shared with unrelated processes at 90–95%
> utilisation. Speedup *ratios* are apples-to-apples — both kernels were
> contended identically on the same data — but absolute microseconds are
> provisional until a clean-card re-run. Correctness is unaffected.

---

## Why this architecture

The RTX 5090 is *consumer* Blackwell, a different chip family from datacenter
Blackwell (sm_100). It has **neither** Hopper's `wgmma` **nor** datacenter
Blackwell's `tcgen05`/TMEM — so FlashAttention refuses to run, FlashAttention 4
is SM100-only, FlashInfer and FlashMLA have no SM12x backend, Triton compiles
sm_12x as sm_80, and vLLM needs `--enforce-eager`. There is no tuned paged
decode-attention kernel for a card a very large number of people own.

**That gap is favourable rather than limiting.** Decode attention is
bandwidth-bound — essentially GEMV plus softmax, arithmetic intensity
≈ batch size. The instructions sm_120 lacks are matmul-throughput features
that matter for prefill and GEMM, not for this kernel. The one thing this
architecture cannot do is the one thing the target kernel does not need.

---

## Correctness

Validated **bit-exact** against HuggingFace, not within a tolerance — 32/32
activation tensors at `0.000e+00` and 32/32 greedy tokens exact, in all four
configurations (contiguous/paged cache × reference/CUDA ops).

Plus 18 CPU structural tests and 21 paged-cache tests, both **mutation-tested**
— each classic bug injected deliberately to confirm the suite catches it:
`seq_lens` off-by-one (2 tests), interleaved RoPE (1), `repeat` vs
`repeat_interleave` (3), per-layer `advance()` (4), QKV bias dropped from
decode (2), block/offset swapped (5), block table ignored (2), whole-block
masking (2).

One mutation is caught by **zero** tests — rmsnorm's weight multiply before the
dtype cast. That is a documented blind spot, not an oversight: the CPU suite
runs in fp32 where the cast is a no-op, so dtype-ordering bugs are invisible to
it by construction and surface only in the bf16 parity run. Structure and
numerics are tested by different harnesses on purpose.

---

## Results

### Paged GQA decode attention (Phase 4)

Three versions kept compiled and runtime-selectable, so before/after comes from
one binary on one dataset. v0: one warp per query head, scalar loads. v1:
thread groups of 16, one 16-byte load per thread, 2 tokens/warp. v2: v1 +
split-KV with a combine pass (Flash-Decoding). All 44–48 registers, 0 spills.

One call, one layer, ctx=2048, block_size=16:

| B | KV MB | MB/L2 | v0 µs | v1 µs | v2 µs | SDPA µs | **v2 vs SDPA** | GB/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.1 | 0.02 | 3265 | 1278 | 23.0 | 403 | 17.5× | 91 |
| 8 | 16.8 | 0.17 | 3673 | 1474 | 25.1 | 345 | 13.8× | 669 |
| 32 | 67.1 | 0.66 | 3689 | 1469 | 52.9 | 1487 | 28.1× | 1268 |
| 64 | 134.2 | **1.33** | 4485 | 1719 | 406.9 | 2510 | **6.17×** | 330 |
| 128 | 268.4 | **2.66** | 4451 | 2524 | 792.4 | 5343 | **6.74×** | 339 |

**Read the MB/L2 column before the GB/s column.** Below B=64 the working set is
L2-resident, so the apparent 1268 GB/s at B=32 is L2 bandwidth, not HBM. Past
L2 it settles at ~330 GB/s — **20% of measured peak, so the kernel is still
latency-bound, not bandwidth-bound.** The defensible rows are B=64 and B=128.
Speedups hold at every batch size, since both kernels read the same data
through the same cache; only absolute bandwidth is inflated.

Baseline is `F.scaled_dot_product_attention` with `enable_gqa=True`, so it pays
no `repeat_interleave` expansion and reads exactly the same byte count — a pure
efficiency comparison at equal traffic. The Python reference is 50–4000× slower
and is never quoted as a speedup.

### Fused RMSNorm + residual (Phase 2)

36 registers, 0 spills, 24 B smem → 8 blocks/SM, 100% theoretical occupancy, so
register pressure was eliminated as a suspect before profiling.

Saturated (B=2048), by `ncu`: 14.58 µs against a 15.21 µs logical-byte floor —
**104% of SOL**, faster than a naive byte count allows, because the 25% gap
between logical and `dram__bytes.sum` traffic is the weight broadcast (2048
blocks × 1536 × 2 B = 6.29 MB of load instructions vs a 6.24 MB measured gap)
and L2 serves all but the first.

At decode shapes it is the opposite problem: **0.09 waves/SM** at B=128, so 42
of 170 SMs get no work, and 2.75 of 3.70 µs is pure launch latency. Not fixable
in the kernel — 128 × 1536 elements is 1.6 MB, and a decode step cannot fill
this card at any geometry. End to end the fused op saves ~5.2 ms/step at *both*
B=1 and B=64; constant in batch, which is the signature of dispatch cost
(56 calls × ~93 µs eager overhead), not bandwidth. The levers are launch
*count* (fusion) and launch *cost* (graph capture).

### Paged KV cache (Phase 3)

Block table, O(1) LIFO free-list allocator, ragged batches. Bit-exact against
the contiguous engine at block_size 2/4/8/16 across repeated boundary
crossings, with no block leak on cache reuse.

---

## Run it

```bash
pip install -r requirements.txt
cd kernels && python setup.py build_ext --inplace && cd ..

python -m testing.smoke          # 18 structural tests, CPU, ~2 s
python -m testing.test_paged     # 21 paged-cache tests, CPU
python -m bench.bandwidth        # measured HBM bandwidth
python -m reference.oracle       # build the HF ground truth, once

python -m testing.phase1                  # full gate, contiguous cache
FD_CACHE=paged python -m testing.phase1   # full gate, paged cache

python -m bench.kernel_paged --bw 1655 --sweep-splits
python -m bench.profile_paged --batch 128 --ctx 2048   # ncu counters
```

`FD_IMPL=ref|cuda`, `FD_CACHE=contiguous|paged`, `FD_BLOCK_SIZE=N`,
`FD_ATTN_VERSION=0|1|2`, `FD_ATTN_SPLITS=N`, `FD_SHAPES="64x2048,128x2048"`.

---

## Measurement discipline

Three successive attempts to measure achieved bandwidth by timing a loop gave
**292%**, then **143%**, then **38%** of a 1655 GB/s peak. The first two were L2
residency; the third was the shared GPU, which also made a plain `torch.copy_`
control read 38% of peak. The resolution was not better arithmetic but a
different instrument — wall-clock bandwidth cannot separate DRAM traffic from
L2 traffic, and that distinction *is* the measurement. **An above-peak
bandwidth number is never a fast kernel; it is always a broken benchmark.**

Timings are best-of-5 minima (contention is strictly additive, so the minimum
is the least-polluted sample) with per-row trial spread reported; rows above 15%
are marked unusable. Across two independent runs the flagged rows disagreed by
9–27% while unflagged rows matched to three significant figures.

---

## Open work

- **~20% of measured HBM peak** — latency-bound, with real headroom. `ncu` stall
  analysis on the attention kernel is the next step; the 20% figure is
  wall-clock arithmetic, not a hardware counter.
- All timings from a contended card; ratios are sound, absolutes provisional.
- v2's split count comes from a heuristic targeting 850 resident blocks.
  `--sweep-splits` measures the real optimum at 16–32, so it is not calibrated.
- Phase 5 (fused RoPE + KV-write, fused SwiGLU) not started.
- Uniform prompt length per batch; ragged *prefill* asserts rather than
  silently corrupting. Prefill uses SDPA and is not an optimisation target.
- Shape limits raise rather than silently falling back: rmsnorm needs
  `hidden_size/8` to be a warp multiple in [64, 1024] (H=1536 and 2048 work,
  Qwen2.5-0.5B's 896 would need a tail path); attention is built for head_dim
  128 and GQA groups {1,2,3,4,6,8}.

---

