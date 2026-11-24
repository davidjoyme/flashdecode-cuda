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

| B | KV MB | MB/L2 | v0 µs | v1 µs | v2 µs | SDPA µs | **v2 vs SDPA** |
|---|---|---|---|---|---|---|---|
| 1 | 2.1 | 0.02 | 3265 | 1278 | 23.0 | 403 | 17.5× |
| 8 | 16.8 | 0.17 | 3673 | 1474 | 25.1 | 345 | 13.8× |
| 32 | 67.1 | 0.66 | 3689 | 1469 | 52.9 | 1487 | 28.1× |
| 64 | 134.2 | **1.33** | 4485 | 1719 | 406.9 | 2510 | **6.17×** |
| 128 | 268.4 | **2.66** | 4451 | 2524 | 792.4 | 5343 | **6.74×** |

Each ratio is a fair comparison *at its own shape* — both kernels saw identical
data under identical cache conditions. They are **not comparable across the L2
boundary**, though. One sequence at ctx 2048 holds 2.10 MB of KV, so L2 (96 MiB)
is **exactly 48 sequences**, and crossing it (B=32 → 64, data 2×) costs v2 7.7×
while costing SDPA only 1.7×. v2 gains far more from cache residency, so the
sub-L2 ratios of 13–28× do not transfer. **6.2–6.7× is the figure to quote**,
because that is the DRAM-resident regime deployment actually runs in: in the
engine ~3 GB of weights stream past per decode step and flush L2 roughly 31
times over, so KV always comes from DRAM.

#### Hardware counters (`profile_paged.py --sweep`)

Wall-clock bandwidth is not quoted here at all; these are `dram__bytes.sum`
figures with caches flushed before each launch:

| B | logical MB | ×L2 | dram MB | % logical | µs | µs/MB | % peak |
|---|---|---|---|---|---|---|---|
| 32 | 67.3 | 0.67 | 69.9 | 104% | 91.8 | 1.364 | 44% |
| 64 | 134.6 | 1.34 | 141.7 | 105% | 229.7 | 1.706 | 35% |
| 128 | 269.2 | 2.67 | 276.9 | 103% | 498.6 | 1.852 | 33% |
| 256 | 538.4 | 5.35 | 546.9 | 102% | 986.9 | 1.833 | 33% |
| 384 | 807.7 | 8.02 | 819.9 | 102% | 1014.3 | **1.256** | **48%** |

`% peak` prices the duration against the **compulsory** bytes, not against
measured traffic. That distinction turned out to matter — see the split sweep
below.

**The GQA reuse is now measured, not asserted.** Traffic sits at 102–105% of
the GQA-aware prediction at every shape. Had the kernel been re-reading KV per
query head it would read ~1200% — so each KV head is provably read once and
reused across its group of 6 in registers. The 2–5% overage is v2's `(m, l,
acc)` split scratch, which the logical count deliberately excludes.

**No cache absorption here, and for a structural reason.** Measured traffic
*exceeds* the footprint at every shape, including 0.67× L2, so nothing is
hiding in cache — unlike Phase 2's rmsnorm, where half the traffic is stores
and up to 50% went unaccounted. Attention output is 0.39 MB against 268 MB of
KV reads, **0.15% of traffic**. Write-back absorption needs writes to absorb,
and there essentially are none. An earlier draft of this section predicted the
Phase 2 problem would recur here; it does not, and the write fraction is why.

**Bandwidth: 33–48% of measured peak, latency-bound.** Every utilisation metric
is low at once — DRAM throughput 30–41%, SM throughput 18–30%, achieved
occupancy 49–69%, **waves/SM 0.71–2.03** — so nothing is saturated anywhere and
there is real headroom. This replaces an earlier claim of ~330 GB/s / 20% of
peak, which came from contended wall-clock timings; `ncu` durations are 1.6–1.8×
shorter at B=64 and B=128, so contention had been *understating* throughput.

**The open finding: µs/MB varies 1.47× across shapes** — 1.852 at B=128, 1.833
at B=256, then **1.256 at B=384**. A bandwidth-bound kernel would be near-flat,
so that spread is itself the result. B=384 reproduced to **0.2%** across two
sessions (1014.26 → 1016.04 µs), so it is not contention, and the split sweep
below rules out the split count. Cause unknown; `ncu --set full` stall reasons
at B=256 vs B=384 is the next instrument.

#### Split sweep — why "GB/s" needs the right denominator

Seven runs at B=384 in one session, so cross-session contention is excluded by
construction:

| splits | dram MB | % logical | overhead | µs | task | moved |
|---|---|---|---|---|---|---|
| auto | 820.8 | 102% | +7 MB | 1016.0 | 48% | 49% |
| 1 | 813.8 | 101% | — | 1032.0 | 47% | 48% |
| 4 | 832.8 | 103% | +19 MB | 1016.6 | 48% | 50% |
| 16 | 882.1 | 109% | +68 MB | 959.6 | 51% | 56% |
| 32 | 953.7 | 118% | **+140 MB** | 978.7 | 50% | **59%** |

**Duration is flat (1032 → 960 µs, 8%) while traffic grows 18%.** The extra
bytes are v2's `(m, l, acc)` scratch, which scales linearly in the split count:
predicted deltas of 4.8 / 14.4 / 33.5 / 71.9 / 148.6 MB against measured
6.0 / 19.0 / 39.4 / 68.3 / 139.9 MB.

So a rate computed against *measured* traffic climbs 789 → 974 GB/s (48% → 59%)
purely because the kernel moves more bytes for identical work. **Efficiency must
be priced against the compulsory bytes.** Measured traffic is a model check —
below 100% means cache absorbed some, above means overhead — never the
efficiency denominator. Phase 2 needed the opposite correction (bytes that
should have moved and didn't), which is why both tables now report the task
rate and keep measured traffic as a diagnostic.

Secondary result: splits=16 is 1.06× faster for +7% traffic, so the heuristic
picking ~1–2 here is trading a little latency for less traffic — defensible,
not miscalibrated. `kernel_paged --sweep-splits` had called it miscalibrated on
wall-clock alone, blind to the scratch cost.

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

- **33–48% of measured HBM peak** — latency-bound (waves/SM 0.71–2.03, so the
  card is barely filled once at most shapes), with real headroom. `ncu --set
  full` stall reasons are the next instrument.
- **µs/MB varies 1.47× across shapes and nothing explains it yet.** Not
  contention (B=384 reproduced to 0.2%), not the split count (duration flat
  across seven pinned values). This is the most interesting open question in
  the project.
- v2's split count heuristic targets 850 resident blocks. At B=384 that picks
  ~1–2 splits, which costs 6% latency but saves 7% traffic — a reasonable
  trade, so the earlier "miscalibrated" verdict is withdrawn for this shape.
  Whether it holds at small B is untested.
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
