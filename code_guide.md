<!-- ======================================================================
     FILE LOCATION: flashdecode/CODE_GUIDE.md
     A walkthrough of every file: what it does, why it is built that way,
     and the interview question it answers.
     ====================================================================== -->

# Code guide

`README.md` says what the project achieved. This says how it is put together
and *why each decision was made*, which is what an interviewer actually
probes. Every section ends with the question that file is the answer to.

---

## The 60-second version

A decode step for one token flows like this:

```
token id
  → embedding
  → for each of 28 layers:
       rmsnorm_residual   ← kernels/rmsnorm.cu          (Phase 2)
       q/k/v projections   (torch matmuls)
       RoPE
       write K,V into the paged cache
       paged decode attention ← kernels/paged_attention.cu  (Phase 4)
       o_proj
       rmsnorm_residual   ← kernels/rmsnorm.cu
       SwiGLU MLP
  → final rmsnorm
  → lm_head → logits → argmax
```

Three ideas hold it together:

1. **An op registry** (`engine/ops.py`). Every operation that might become a
   CUDA kernel is called through a lookup table, never directly. Swapping a
   kernel in is a one-line change, and a correctness bug bisects to a single
   kernel in minutes.
2. **A ground-truth oracle** (`reference/oracle.py`). HuggingFace's own
   activations are dumped once and every change is diffed against them,
   layer by layer, for bit-exactness rather than "close enough".
3. **Speed-of-light first** (`bench/sol.py`). The byte-traffic floor is
   computed *before* a kernel is written, so "fast" means a fraction of a
   known limit rather than a feeling.

---

## Project root

### `config.py`
The frozen target: model ID, dtype, device, the parity prompt, and the diff
tolerances. Holds `LlamaConfig`, a dataclass populated from HuggingFace's
`config.json` rather than hardcoded, so the same code runs 0.5B through 8B.

Two things worth knowing:

- **`rope_from_hf()`** handles both transformers layouts. transformers 5.x
  removed the flat `rope_theta` attribute and moved it into
  `rope_parameters`, keeping `rope_scaling` as an *alias to the same dict*.
  So a naive read makes every model look frequency-scaled —
  `rope_type == "default"` means no scaling, and missing that would silently
  corrupt long-context output.
- **The tolerances are reasoned, not guessed.** bf16 carries ~8 mantissa
  bits, so ~1e-2 relative error from reduction-order differences alone is
  expected. 1e-1 is a bug; 1e-3 is noise.

> *"Why freeze the model and dtype up front?"* Every benchmark number is
> relative to one configuration. Change the config and every recorded
> measurement becomes meaningless, so it is pinned in one place and changes
> are logged.

### `requirements.txt`, `.gitignore`
`.gitignore` matters more than it looks: it blocks `*.ncu-rep` (a single
Nsight Compute report can exceed GitHub's 100 MB limit), `kernels/build/`,
`*.so`, and the generated `oracle.pt`. Once a large blob is in git history it
stays there without a rewrite.

---

## `engine/` — the forward pass

### `engine/ops.py` — **the load-bearing file**
A registry mapping `(op_name, impl_name) → function`. Model code calls
`ops.call("rmsnorm_residual", ...)`, never an implementation directly.

```python
ops.use("rmsnorm_residual", "cuda")   # swap in a kernel, no model changes
ops.use_all("cuda")                   # everything that has a CUDA impl
ops.reset()                           # back to pure PyTorch
```

Also holds the pure-PyTorch reference implementation of every op that becomes
a kernel: `rmsnorm`, `rmsnorm_residual`, `rope`, `decode_attention`,
`paged_decode_attention`, `swiglu`.

Two deliberate choices:

- **Signatures are shaped for what a kernel wants**, not what reads nicely in
  PyTorch. Decode ops take `[B, heads, head_dim]` with no sequence axis,
  because one token per sequence is all a decode kernel ever sees. Getting
  that right up front means no reshaping at the kernel boundary later.
- **`decode_attention_ref` groups queries instead of expanding KV.** The
  obvious implementation is `k_cache.repeat_interleave(G, dim=2)` then
  `.float()`, which materialises a transient `G × 2` times the real KV
  footprint — 3.25 GB at B=128, and it OOM'd the first benchmark run.
  Viewing `q` as `[B, n_kv, G, d]` gives the identical head mapping for free
  and is bitwise identical.

> *"How do you keep a kernel and its reference from drifting apart?"* They
> share a frozen signature and a registry entry, so both are callable on the
> same inputs and the test suite runs the comparison every time.

### `engine/ops_cuda.py`
Registers the CUDA implementations against those same op names. Importing it
registers; it does *not* activate — activation is explicit.

Also adds `kernels/` to `sys.path`, because `setup.py build_ext --inplace`
puts the `.so` there while everything runs as `python -m` from the project
root. Without that you get "not built, run setup.py" seconds after a
successful build. It distinguishes *never built* from *built but won't load*
(an ABI mismatch needing a rebuild, not a first build).

`FD_ATTN_VERSION=0|1|2` and `FD_ATTN_SPLITS=N` select the attention kernel at
runtime.

### `engine/model.py` — the forward pass
`LlamaEngine` with `prefill`, `decode_step`, and `generate`. Handles Llama and
Qwen2; the only architectural difference is that Qwen2 carries a bias on
`q_proj`/`k_proj`/`v_proj`, resolved once in `__init__` and passed to
`F.linear`, which accepts `None` for Llama — no per-layer branch in the hot
path.

**The deferred-residual restructure is the thing to be able to explain.** The
textbook block is:

```python
residual = h
hn = rmsnorm(h)
h  = residual + attn(hn)       # add at the END of one block,
                               # norm at the START of the next
```

Those sit on opposite sides of the loop boundary, so they can never fuse.
This engine carries two values across the loop instead, the way vLLM and
TensorRT-LLM do:

```python
hn, residual = rmsnorm_residual(hidden, residual, w, eps)
#   residual <- residual + hidden    (the deferred add)
#   hn       <- rmsnorm(new residual)
```

`hidden` is now the *delta* from the last sub-block and `residual`
accumulates. The add and the norm are adjacent inside one op — exactly what
`rmsnorm.cu` fuses, 56 times per decode step.

Equivalence is **exact, not approximate**: the old form computed
`residual + linear(...)`, the new one computes `x + residual` inside the op,
and IEEE addition is commutative. That matters practically — it means the
restructure can be validated with the *reference* implementation before the
kernel is switched on, separating "did I restructure correctly" from "is my
kernel correct".

Also here: `logits_all` (compute the LM head over all positions and slice,
matching HF's GEMM shape so the logits row is a like-for-like comparison —
see `testing/diff.py`), and a capped RoPE table (Qwen2.5 advertises
`max_position_embeddings=131072`, which would build two 67 MB fp32 tables for
positions no benchmark reaches).

> *"Why not just fuse the add into the attention output?"* Because the norm
> that consumes it belongs to the *next* sub-block. Deferring the add is what
> brings the two adjacent.

### `engine/cache.py` — contiguous KV cache
One `[B, max_len, n_kv_heads, head_dim]` tensor per layer for K and for V.
Deliberately simple: its job is to be *obviously correct* so the paged cache
has something to diff against.

The one subtlety: `append()` writes at each sequence's current length and does
**not** advance. `advance()` is called once per decode step after every layer
has written — advancing per-layer would make layer 1 write to a different slot
than layer 0, which is a quiet and very confusing bug.

### `engine/paged_cache.py` — paged KV cache
Block table, O(1) LIFO free-list allocator, ragged batches. Three decisions
worth defending:

**Layout `[num_blocks, n_kv_heads, block_size, head_dim]`.** A decode kernel
fixes a head and walks tokens, so consecutive tokens *of the same head* must
be contiguous. This gives a 4 KB fully-coalesced slab per (block, head) —
exactly 32 cache lines at head_dim 128 in bf16. The token-major alternative,
`[num_blocks, block_size, n_kv_heads, head_dim]`, makes one token's heads
contiguous instead, so reading one head across tokens strides by
`n_kv_heads × head_dim` and wastes half of every fetch at 2 KV heads. vLLM
splits K further to suit its particular vector loads; that was deliberately
*not* copied, because committing to it before writing the kernel would be
cargo-culting.

**`block_size = 16`**, giving that clean 4 KB slab and matching vLLM's default
so the comparison is like-for-like. Swept via `FD_BLOCK_SIZE`.

**Unused slots are not zeroed.** Zero-filling costs a write per allocation
*and* hides mask off-by-ones behind harmless-looking zeros. A test poisons the
tail of the last block with 1e4 and asserts the result is bit-identical.

Also: `ensure_capacity()` runs once per *step*, before the layer loop, not
inside `append()`. A sequence needs a new block at most once per step while
`append` runs 28 times, so folding them would do 28× the work and run the
Python allocator in the hot loop.

> *"What does paging actually buy you?"* Memory, on **ragged** workloads. With
> every sequence at full length there is nothing to reclaim, so a uniform
> full-length batch sees no benefit. Claiming otherwise would be wrong, and
> the code says so in a comment.

### `engine/loader.py`
HuggingFace checkpoint → GPU weight dict, plus `validate()`, which checks
every projection shape and fails loudly on surprises.

Two details: QKV bias is detected from the **checkpoint**, not `config.json`,
because transformers hardcodes `bias=True` in the Qwen2 modeling code rather
than exposing it as a config field — the tensor's presence is ground truth.
And tied embeddings are aliased (`Qwen2.5-1.5B` omits `lm_head.weight`
entirely) so the forward pass never branches.

`validate()` also *refuses* things it cannot do: an `o_proj` bias, MLP biases,
or Qwen3's per-head QK-norm. A dropped bias term is a silent correctness bug,
not a rounding difference.

---

## `kernels/` — the CUDA

### `kernels/rmsnorm.cu` — fused RMSNorm + residual (Phase 2)
Computes `h = x + residual` and `out = weight * rmsnorm(h)` in **one pass**,
writing both outputs. One block per row, `THREADS = H/8` so each thread does
exactly one 128-bit load per tensor.

**The detail that made it bit-exact:** round `x + residual` to bf16 *before*
squaring it for the variance. The reference adds two bf16 tensors (producing a
bf16 result) and computes the variance from the rounded value. Accumulating
the unrounded fp32 sum instead changes 1309 of 131072 outputs by up to
6.25e-02 — three times the diff tolerance — and would present as a mysterious
persistent error with cosine ≈ 1.0.

Likewise the weight multiply happens *after* the cast back to bf16, because HF
does `weight * x32.to(dtype)`, not `(weight * x32).to(dtype)`.

**ptxas:** 36 registers, 0 spills, 24 B shared memory → 8 blocks/SM, 100%
theoretical occupancy, with the hardware thread limit binding rather than this
kernel. So register pressure was eliminated as a suspect *before* profiling.

**What the profiler said.** At B=2048 the kernel runs in 14.58 µs against a
15.21 µs logical-byte floor — 104% of SOL, *faster* than a naive byte count
allows. The 25% gap between logical and measured DRAM traffic is the **weight
broadcast**: all 2048 blocks read the whole 3 KB weight vector, which is
2048 × 1536 × 2 = 6.29 MB of load instructions against a measured gap of
6.24 MB. L2 serves all but the first.

At decode shapes it is a different story: **0.09 waves/SM** at B=128, so 42 of
170 SMs get no work at all, 12% occupancy, and 2.75 of 3.70 µs is pure launch
latency. Not fixable in the kernel — 128 rows × 1536 elements is 1.6 MB, and
there is not enough data in a decode step to fill the card at any geometry.
The levers are launch *count* (fusion) and launch *cost* (graph capture).

> *"Your kernel beat the speed-of-light number. How?"* It did not; the SOL was
> computed on logical bytes and L2 absorbs the weight re-reads. That is why
> `dram__bytes.sum` is the metric to trust, not bytes ÷ seconds.

### `kernels/paged_attention.cu` — paged GQA decode attention (Phase 4)
Three versions, **all kept compiled** and selectable at runtime, so before and
after come from one binary on one dataset rather than remembered numbers.

Common to all: grid is `(n_kv_heads, batch)`, and each block serves all
`GROUP` query heads that share its KV head — so KV is read **once** and reused
across the group, where the reference's `repeat_interleave` reads it 6 times.
Accumulation is fp32; the cache is read as bf16 in place and never upcast in
memory.

**Online softmax, single pass.** A running max and running sum, rescaling the
accumulator when the max moves:

```
m_new = max(m, s)
corr  = exp(m - m_new)
p     = exp(s - m_new)
acc   = acc*corr + p*V
l     = l*corr + p
```

The alternative — scores to scratch, then max, exp, sum, weighted sum — reads
KV twice and needs O(seq_len) storage. This is FlashAttention's recurrence
applied to a single query row. Verified against standard max-then-exp to fp32
epsilon, including a score of 115,200 where naive `exp` would overflow.

**v0** — one warp per query head, scalar loads, whole context per block.
44 registers, 0 spills, **0 barriers** (no `__syncthreads()` at all, since
each warp owns one query head independently). Correct, and **8× slower than
torch SDPA**. Per token: 4 scalar bf16 loads of K, 5 `__shfl_xor` rounds to
reduce 32 lanes, 4 scalar loads of V — one token at a time, zero token-level
parallelism in a warp. Over ctx=2048 that is 10,240 *dependent* shuffle rounds
per head, ~107 µs of pure reduction latency before any memory. **v0 was
latency-bound on its own reduction.**

**v1** — thread groups and 16-byte loads. `VEC=8` bf16 per thread (one
`uint4`, the widest load the pipe accepts) so `TG=16` threads cooperate on one
token and a warp covers 2 tokens concurrently:

| | v0 | v1 |
|---|---|---|
| loads per token per thread | 4 × 2 B | 1 × 16 B |
| reduction rounds | 5 (32 lanes) | 4 (16 lanes) |
| tokens in flight per warp | 1 | 2 |
| dependent shuffle rounds | 10,240 | 4,096 |

2.6–2.9× over v0. The cost is that each warp now holds **two independent
softmax partials** that must be merged — the same merge Flash-Decoding uses
across blocks, applied across thread groups.

But v1 measured **flat at ~1600–1800 µs from B=1 to B=32** while the work grew
32×. Each block streams the whole context, so per-block serial time is set by
*context length* and extra blocks run in parallel for free. Batch size is
irrelevant to the critical path.

**v2** — split-KV (Flash-Decoding) plus a combine pass. Grid becomes
`(n_kv_heads, batch, n_splits)`; each block handles a range of table blocks and
writes `(m, l, acc)` to scratch; a second kernel merges across splits. At B=1
that is 32 splits × 64 tokens instead of one block × 2048 tokens — a 32×
shorter dependency chain. **6.2–6.7× over torch SDPA** at working sets beyond
L2.

Prior art is cited, not reinvented: the algorithm is Flash-Decoding. What is
new is that no implementation of it exists for sm_120.

> *"Why not use wgmma or tcgen05?"* Neither exists on sm_120 — consumer
> Blackwell is a different chip family from datacenter Blackwell and has no
> TMEM. But decode attention is bandwidth-bound, essentially GEMV plus
> softmax, so the missing instructions are matmul-throughput features this
> kernel does not need. The one thing the architecture cannot do is the one
> thing the kernel does not require.

### `kernels/bindings.cpp`
pybind11 bindings. Validates dtype, contiguity, shapes and GQA divisibility,
allocates the v2 scratch, picks the split count, and `TORCH_CHECK`s that a
kernel was actually launched — a silent no-op would leave the output
uninitialised and look like a numerics bug rather than a configuration one.

Two bugs lived here and both are worth knowing:

- The project macro was originally named `CHECK`, which **shadows torch's own
  one-argument `CHECK`**. Renamed `FD_CHECK`.
- The block count was computed with `seq_lens.max().item<int>()`. `.item()` is
  a **device-to-host sync on every call**, breaking pipelining so each launch
  paid full latency — worth up to **23.8×**. `block_table.size(1)` is already
  a host-side int that bounds the block count, so the sync bought nothing.

> *"How do you profile a kernel's host wrapper?"* You have to look. A single
> `.item()` cost more than the kernel it wrapped.

### `kernels/setup.py`
The torch extension build. Two non-obvious settings:

- **`-std=c++20`.** torch ≥ 2.4 headers are C++20 and say so with an
  `#error`. With `c++17` you get ~100 cascading errors about
  `std::strong_ordering`, `requires` and `string_view::starts_with`, all from
  that one line. `c++17` is the common template default because torch ≤ 2.3
  *was* C++17.
- **`TORCH_CUDA_ARCH_LIST="12.0"`** plus an explicit
  `-gencode arch=compute_120,code=sm_120`. Without it, older build defaults
  stop at sm_90 and the extension compiles cleanly but fails at *runtime* with
  "no kernel image is available" — nothing in the build output hints at it.

`-Xptxas=-v` prints registers and spills; `-lineinfo` lets `ncu` attribute
stalls to source lines. `--use_fast_math` is deliberately **off**: it changes
`rsqrtf` and would introduce a systematic difference from the reference, which
is exactly what the diff harness is sensitive to.

---

## `reference/` — ground truth

### `reference/oracle.py`
Runs HuggingFace's own Qwen2 once and dumps per-layer hidden states (via
forward hooks) plus the greedy token sequence to `oracle.pt`. Everything the
engine produces is diffed against this.

Unchanged by the Llama → Qwen switch, because the HF module names
(`model.embed_tokens`, `model.layers`, `model.norm`) are identical.

> *"How do you know your engine is right?"* It reproduces HuggingFace's
> activations bit-exactly at every layer, and its greedy output token-for-token
> — not within a tolerance.

---

## `testing/` — anything that asserts

### `testing/smoke.py` — 18 structural tests, CPU, ~2 seconds
Runs on a tiny random-weight model with deliberately awkward dimensions
(GQA group 3, `head_dim ≠ hidden/n_heads`) so sloppy reshapes fail.

**The load-bearing test is `decode_equals_prefill`:** prefill over S tokens
must produce the same logits as prefill over S−1 plus one `decode_step`. That
single invariant catches, with no GPU and no real weights — RoPE applied at
the wrong position, KV written to or read from the wrong slot, the
`seq_lens` vs `seq_lens + 1` off-by-one, GQA expansion mismatched between
prefill and decode, and a QKV bias dropped from one path.

**Mutation-tested.** Each classic bug was injected deliberately to confirm the
suite catches it:

| injected bug | tests that caught it |
|---|---|
| `seq_lens` instead of `seq_lens + 1` | 2 |
| interleaved RoPE instead of `rotate_half` | 1 |
| `repeat` instead of `repeat_interleave` | 3 |
| `advance()` per-layer instead of per-step | 4 |
| QKV bias dropped from decode only | 2 |
| rmsnorm weight multiply before the cast | **0** |

That last row is a **documented blind spot, not an oversight**: the suite runs
in fp32, where the cast is a no-op, so dtype-ordering bugs are invisible to it
by construction. They surface only in the bf16 parity run. Structure and
numerics are tested by different harnesses on purpose.

> *"How do you know your tests are any good?"* Mutation testing. A test suite
> that has never failed has never been shown to work.

### `testing/test_paged.py` — 21 paged-cache tests, CPU
Allocator round-trips, exhaustion, double-free, disjointness; block-table
mapping; boundary crossing; fragmentation; and the engine-level checks that
the paged engine emits **identical tokens** to the contiguous one at block
sizes 2/4/8/16.

The key one builds the same logical KV twice — contiguous and paged — and
asserts the two attention paths agree, with the block table **deliberately
scrambled** (interleaved, non-monotonic allocations). A table where sequence
*b* owns blocks `b*nb …` passes even if the code ignores the table entirely
and computes positions arithmetically, which is the easiest way to write a
paged cache that is not actually paged. The `ignore_table` mutation confirms
the scrambling is what catches it.

Also mutation-tested: swapped block/offset (5 tests), ignored table (2),
whole-block masking (2), wrong layout permute (2), skipped allocation (2),
off-by-one `blocks_needed` (9).

### `testing/diff.py` — the layer-by-layer harness
Compares the engine's traced activations against the oracle **in execution
order** and stops mattering after the first divergence — once layer 7 is
wrong, layers 8–27 are wrong too and their numbers are noise.

Reports four things per tensor, and the last two are the useful ones:

- **cosine similarity** separates *scaling* bugs (cosine ≈ 1.0, wrong
  magnitude — wrong eps, wrong attention scale, missing residual) from
  *structural* bugs (cosine well below 1.0 — indexing, RoPE convention, head
  layout). Worth about a week of debugging time.
- **ULPs of tensor scale**, `max_abs / (max|ref| × 2⁻⁸)`. A flat absolute
  tolerance is wrong here because the tensors do not share a scale: hidden
  states run ~1, logits ~20. One bf16 ULP at magnitude 8–16 is already
  6.25e-02, so a single-ULP rounding difference in the final matmul trips a
  tolerance perfectly calibrated for every other row.

That is not hypothetical — it is what the first green Qwen run did. Every
layer bit-exact, greedy parity 32/32, and `logits` alone flagged. The cause
was HF computing the LM head as a GEMM over all positions and slicing, where
the engine ran a single-row GEMV; cuBLAS picks different kernels, so the bf16
accumulation rounds differently. Hence `logits_all` in `model.py`.

### `testing/phase1.py` — the full gate, one command
Five stages: smoke suite → oracle freshness → activation diff → greedy parity
→ decode baseline. Exits non-zero on failure, so "is the gate green" is not a
judgement call.

Things it does that are easy to skip:

- **Oracle freshness.** Compares the `model_id`, prompt and dtype stored in
  `oracle.pt` against the current config. Change `MODEL_ID` and forget to
  regenerate, and the diff compares the engine against a *different model* —
  errors that look exactly like real bugs.
- **Runs every implementation.** The diff executes for `ref` *and* `cuda`. An
  earlier version selected the impl only in stage 5, so the kernel ran inside
  the model for the first time during a *timing* loop with nothing checking
  its output. A kernel that is fast and wrong is worse than no kernel.
- **GPU contention check.** Reads `nvidia-smi` and refuses to present timings
  from a busy card. This exists because of a real incident: the GPU was at 60%
  utilisation from two unrelated processes, and it silently corrupted
  everything — the same shape measured 191 ms then 306 ms, and a plain
  `torch.copy_` control read 38% of peak bandwidth. Hours went into harness
  fixes for what was contention.
- **Best-of-5 with jitter reporting.** Contention is strictly additive —
  another process can only make our kernels wait, never finish sooner — so the
  minimum is the least-polluted sample. Each row reports trial-to-trial
  spread, and rows above 15% are marked unusable. Across two independent runs
  the flagged rows disagreed by 9–27% while the unflagged ones matched to
  three significant figures.

> *"Under what conditions were these measured?"* The harness answers this
> itself rather than leaving it to memory.

---

## `bench/` — anything that times

### `bench/bandwidth.py`
Measures achievable HBM bandwidth with copy, read and triad patterns.
**1655 GB/s** on this card, 92% of the spec figure. Every SOL claim divides by
this, never by the spec number — real kernels top out at 80–93% of spec, so
quoting spec makes your own kernels look worse *and* signals you did not
check.

### `bench/sol.py`
The speed-of-light model: weight bytes and KV bytes per decode step, and the
resulting latency and throughput floor. Requires `--bw` with no default, on
purpose, so you cannot accidentally produce a number against an unmeasured
figure.

The point worth leading a writeup with: **a decode step reads the entire
weight matrix to produce one token per sequence.** Arithmetic intensity is
~batch_size FLOPs per byte, so at batch 1 on a card whose compute:bandwidth
ratio is in the hundreds you are orders of magnitude from compute-bound.
Decode is a memory problem wearing a matmul costume.

Computing this *first* matters: measure first and you anchor on whatever
number you happened to get, then stop optimising when it stops being easy.

### `bench/fit.py`
Given live free VRAM, prints which `(batch, context)` shapes actually fit.
Needed because the GPU is shared and the answer changes minute to minute; an
OOM mid-benchmark leaves fragmentation that spoils every later shape.

Surfaced a useful insight: KV% of traffic is set by `B × ctx`, so ctx=2048 at
B=240 costs the same memory as ctx=4096 at B=120 but gives **twice as many
sequences** — and the decode kernel's grid scales with sequence count. Shorter
context at larger batch is strictly the better benchmark.

### `bench/kernel_rmsnorm.py`
Correctness plus three timings per shape: *amortized* (back-to-back, judges
the kernel), *per-launch* (median, what an eager loop actually pays), and
*graph-replayed* (shows how much was dispatch).

The finding: `amort` is pinned near 2.06 µs while the work varies 128×, which
is the signature of a launch floor rather than a bandwidth measurement, and
`graph == amort` everywhere. So the Phase 2 result is **11–32× over unfused
PyTorch, and it is a launch-count win, not a bandwidth win** — PyTorch runs
the add and the norm as ~6 kernels, this is 1. Calling it a bandwidth speedup
would be wrong and an interviewer would take it apart in one question.

### `bench/kernel_paged.py`
The headline benchmark: v0 vs v1 vs v2 vs **torch SDPA with
`enable_gqa=True`**, so SDPA pays no `repeat_interleave` expansion and reads
exactly the same byte count — a pure efficiency comparison at equal traffic.
It also probes for vLLM's paged kernels and reports the exact error if they
do not load.

**The baseline choice matters more than the number.** Beating the Python
reference by 50× proves nothing; it was written to be obviously correct. The
output says so explicitly and caps the reference at B≤8.

`--sweep-splits` **measures** the best v2 split count per shape rather than
trusting the built-in heuristic, which targets 850 resident blocks from
occupancy arithmetic and turns out to be miscalibrated.

The split sweep is also what caught a real bug: v2 was exact at splits
1/2/4/8 and `rel 1.00e+00` at 16, because ctx=128 at block_size 16 is 8 table
blocks, so splits 8–15 got no work, and `(-inf) − (-inf)` is NaN rather than a
harmless zero. The combine could not recover, because **`NaN × 0` is still
NaN**. Both merges now test `isfinite` explicitly.

### `bench/profile_rmsnorm.py`, `bench/profile_paged.py`
Nsight Compute wrappers. These exist because **three successive attempts to
measure bandwidth with a timing loop produced 292%, then 143%, then 38% of a
1655 GB/s peak.** The first two were L2 residency — the loop re-read buffers
small enough to stay cached, so it measured L2 at several TB/s. The third was
the shared GPU.

The resolution was not better arithmetic but a different instrument:
wall-clock bandwidth cannot distinguish DRAM traffic from L2 traffic, and that
distinction *is* the measurement. `dram__bytes.sum` reports actual DRAM
sectors, so cache hits simply do not appear — and being a hardware counter it
is unaffected by contention, unlike duration and throughput.

`profile_paged.py` uses it to test a specific claim: the kernel should read
each KV head once, so `dram__bytes ≈ 2 × B × ctx × n_kv_heads × head_dim × 2`
with **no** factor of GROUP and no factor of 2 for fp32 — 268 MB at B=128
against the reference's 3221 MB. Near 268 and the 12× reduction is measured;
near 1600 and the GQA reuse is not happening.

> *"An above-peak bandwidth number is never a fast kernel. It is always a
> broken benchmark, and the usual cause is a hot cache."*

---

## `logs/EXPERIMENTS.md`
The append-only experiment log, and the raw material the writeup is assembled
from. Every kernel version records hypothesis → change → result → **what
actually happened**, including the versions that got slower and the
measurements that turned out to be wrong.

It is the most valuable file in the repo for an interview, because it contains
the things that cannot be reconstructed from the code: v0's entry says the
hypothesis was right about the grid and wrong about what mattered; the failed
bandwidth measurements are recorded with their diagnoses; and both Phase 4
bugs are written up with before/after tables.

---

## Three things to be ready to say

**On the architecture.** sm_120 has neither Hopper's `wgmma` nor datacenter
Blackwell's `tcgen05`/TMEM, so FlashAttention, FlashInfer and FlashMLA have no
backend and Triton compiles it as sm_80. That gap is favourable rather than
limiting: decode attention is bandwidth-bound, so the missing instructions are
matmul-throughput features the kernel does not need.

**On what is still open.** The kernel sits at ~20% of measured HBM peak, so it
is latency-bound, not bandwidth-bound — there is real headroom and `ncu` stall
reasons are the next step. Timings came from a contended GPU, so the ratios
are sound but the absolute microseconds are provisional. The v2 split
heuristic is measurably miscalibrated. Saying this before being asked is
worth more than any number in the table.

**On the two bugs.** Both were caught by test *design*, not luck. The NaN
needed a split-count sweep, because splits > table blocks is the only way to
produce an empty partial and the auto heuristic never picks it. The device
sync needed looking at the host wrapper rather than only the kernel. Those are
the stories to tell.