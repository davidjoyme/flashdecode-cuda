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

Measured with `ncu` across four working-set sizes, because a single shape
cannot tell a bandwidth result from a cache artifact:

| B | set MB | ×L2 | dram MB | % of logical | µs | GB/s | % SOL |
|---|---|---|---|---|---|---|---|
| 2048 | 25.2 | 0.25 | 12.6 | 50% | 13.85 | 909 | 110%  |
| 8192 | 100.7 | 1.00 | 55.0 | 55% | 42.25 | 1302 | 144%  |
| 32768 | 402.7 | 3.99 | 350.0 | 87% | 241.46 | 1450 | 101%  |
| 65536 | 805.3 | 7.97 | 758.9 | **94%** | 519.13 | 1462 | **94%**  |

**Only the last row is a bandwidth measurement.** Below ~4× L2 the cache
absorbs write-back that never reaches DRAM inside the kernel window, so the
duration excludes stores the byte floor charges in full — which is what
produces the impossible 110% and 144% readings. As the working set grows, the
measured fraction of the logical footprint climbs 50% → 55% → 87% → 94% and
the artifact disappears on its own.

At B=2048 the aggregate came in at 12.59 MB against a 12.586 MB logical *read*
footprint — 100.03%, with the stores contributing **nothing at all**. That
exact coincidence is what identifies the mechanism.

**Phase 2 result, B=65536 (805 MB, 8× L2, 94% of traffic reaching DRAM):
94% of the logical byte floor — 1551 GB/s logical-byte rate, 1462 GB/s on
counted sectors, so 88–94% of measured peak.** The spread between those two
figures is exactly the 6% still being absorbed, so both are true and the range
is the honest claim.


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
python -m bench.profile_rmsnorm --sweep --bw 1655      # ncu, past L2
python -m bench.profile_paged --batch 128 --ctx 2048   # ncu counters
```

`FD_IMPL=ref|cuda`, `FD_CACHE=contiguous|paged`, `FD_BLOCK_SIZE=N`,
`FD_ATTN_VERSION=0|1|2`, `FD_ATTN_SPLITS=N`, `FD_SHAPES="64x2048,128x2048"`.

---

## Measurement discipline

Three attempts to measure achieved bandwidth by timing a loop gave **292%**,
then **143%**, then **38%** of a 1655 GB/s peak. Those are not one error
shrinking — they are **two independent confounds with opposite signs**, which is
why no amount of fixing the harness converged.

| | setup | reading | cause |
|---|---|---|---|
| 1 | one buffer, reused | 4829 GB/s = **292%** | fully L2-resident |
| 2 | rotate over 3× L2 | 2374 GB/s = **143%** | ~46% still L2-resident |
| 3 | rotate 6× L2 + graph | `copy_` control **38%** | GPU shared with other users |

Attempt 1 measured L2 faithfully and divided by an HBM denominator — L2 read
bandwidth here is ~4800 GB/s, so 292% is arithmetically exactly right and
answering the wrong question. Attempt 2's rotation genuinely reduced residency
but could not eliminate it, and partial residency is indistinguishable from a
working harness: since time is additive, mixed residency combines harmonically
as `1/B_eff = f/B_L2 + (1−f)/B_HBM`, and f = 0.46 reproduces 143% exactly.

Attempt 3 is the one that mattered, because **caches can only inflate.** A
plain `torch.copy_` cannot read at 38% of its own peak through any cache
effect, so the cause had to be external — other processes taking bandwidth.

That combination is unfixable by arithmetic: cache inflates, contention
deflates, and wall-clock time is a single observable. 143% is equally
consistent with f = 0.46 on a quiet card or f = 0.7 with contention clawing it
back. One equation, two unknowns — and attempt 3's spread of 57/78/64% across
three shapes that should have been identical is what that looks like in
practice.

So the resolution was a different instrument, not better arithmetic.
`dram__bytes.sum` **counts** the DRAM sectors a kernel moved rather than
inferring them from elapsed time, so cache hits never enter the number at all.
**An above-peak bandwidth figure is never a fast kernel; it is always a broken
benchmark** — and a below-peak one on a pure stream is never a slow kernel.

Timings are best-of-5 minima (contention is strictly additive, so the minimum
is the least-polluted sample) with per-row trial spread reported; rows above 15%
are marked unusable. Across two independent runs the flagged rows disagreed by
9–27% while unflagged rows matched to three significant figures.

---

## Open work

- **~20% of measured HBM peak** — latency-bound, with real headroom. `ncu` stall
  analysis on the attention kernel is the next step; the 20% figure is
  wall-clock arithmetic, not a hardware counter.
- Attention timings are from a contended card; ratios are sound, absolutes
  provisional. The Phase 2 ncu sweep is only *partly* insulated: the byte
  counts are contention-immune, since another process cannot change how many
  sectors this kernel moves, but `ncu` still reports a duration and contention
  inflates it. So the `dram MB` and `% logical` columns are solid while the
  µs / GB/s / %SOL columns are not — they understate, making **94% of SOL a
  floor rather than a ceiling**.
- This card exposes no `dram__bytes_{read,write}` counters, so the Phase 2
  read/write split is derived from the aggregate rather than measured. The
  derivation is pinned by `--cache-control all` and corroborated by the
  100.03% read-footprint match at B=2048, but it is an inference.
- The paged attention kernel has the same latent absorption question and has
  not been swept this way yet.
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