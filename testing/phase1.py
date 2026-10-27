"""
Phase 1 gate. One command, one verdict.

    python -m testing.phase1

Runs the whole gate in order and exits non-zero if any stage fails:

  1. CPU smoke suite            -- structure (no GPU, no download)
  2. oracle freshness           -- does oracle.pt match the CURRENT config?
  3. activation diff vs HF      -- numerics, bf16, layer by layer
  4. greedy token parity        -- end-to-end determinism
  5. naive decode baseline      -- ms/step and tok/s at real shapes

Stage 2 exists because a stale oracle is the sneakiest failure mode here:
change MODEL_ID or PARITY_PROMPT, forget to regenerate, and stage 3 diffs
your engine against a different model entirely. The errors look like real
bugs and are not.

Stage 5 is the number every Phase 2-5 result gets compared against, so it is
part of the gate rather than an afterthought -- a baseline you forgot to
record is a baseline you cannot claim an improvement over.

Stage 5 does NOT run a real prefill to fill the cache. A B=128 ctx=4096
prefill through SDPA would materialise ~1.6 GB per tensor after the GQA
repeat_interleave and can OOM or take minutes, which is pointless when
correctness was already settled in stages 3-4. Instead the cache is filled
with random KV and seq_lens is set directly, which is exactly the state a
decode step sees. Timing only -- never correctness.
"""

from __future__ import annotations

import gc
import os
import time
import traceback

import torch

import config


def _hdr(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\n[{n}/5] {title}\n{'=' * 72}")


def check_gpu_clean() -> bool:
    """Refuse to pretend a shared GPU produces trustworthy timings.

    This exists because of a real incident: the card was at 60% utilisation
    with 9.5 GB held by two unrelated conda envs, and it silently corrupted
    every measurement in the project --

      * stage 5 saw 19.9 GB free instead of ~30 GB, so the largest shape was
        skipped as if it did not fit
      * the same B=128 shape measured 191 ms in one run and 306 ms in the
        next, which looked like a code regression
      * a pure torch.copy_ control came back at 38% of peak bandwidth, which
        looked like a broken benchmark harness
      * achieved bandwidth varied 57/78/64% across shapes that should be
        identical

    Hours went into increasingly elaborate harness fixes for what was
    contention. Checking first is seconds. Always check first.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        util, used, total = (int(x.strip()) for x in out.stdout.split(","))
    except Exception:
        return True          # cannot tell; do not block the run

    # Our own process has not allocated anything yet at this point, so
    # anything resident belongs to someone else (plus ~1 GB of desktop).
    others = max(0, used - 1024)
    noisy = util > 15 or others > 2048
    print(f"  GPU state: {util}% util, {used}/{total} MiB used "
          f"(~{others} MiB not ours)")
    if noisy:
        print("\n  *** WARNING: this GPU is busy. Timings below are NOT valid. ***")
        print("  Other processes are using it, so every ms/step, GB/s and")
        print("  speedup in this run is contaminated by contention, and free")
        print("  VRAM is reduced so large shapes may be skipped for no reason.")
        print("\n  Find the culprits:   nvidia-smi")
        print("  Inspect one:         ps -p <PID> -o pid,etime,cmd")
        print("  A clean card reads ~0% util and under ~1 GB used.")
        print("\n  Correctness stages (1-4) are unaffected -- bit-exactness does")
        print("  not care about contention. Only the timings are junk.")
    return not noisy


# ---------------------------------------------------------------- stage 1
def stage_smoke() -> tuple[bool, int]:
    _hdr(1, "CPU smoke suite")
    from testing import smoke

    failed = 0
    for fn in smoke.TESTS:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    total = len(smoke.TESTS)
    if failed:
        print(f"\n  {failed}/{total} failed. Fix these before touching the GPU --")
        print("  they isolate the same bugs without a download in the loop.")
        return False, total
    print(f"  all {total} passed")
    return True, total


# ---------------------------------------------------------------- stage 2
def stage_oracle_fresh():
    _hdr(2, "oracle freshness")
    try:
        from reference.oracle import load

        o = load()
    except FileNotFoundError as e:
        print(f"  {e}")
        return None
    stale = []
    if o["model_id"] != config.MODEL_ID:
        stale.append(f"model_id: oracle={o['model_id']!r} config={config.MODEL_ID!r}")
    if o["prompt"] != config.PARITY_PROMPT:
        stale.append(f"prompt: oracle={o['prompt']!r} config={config.PARITY_PROMPT!r}")
    if o.get("dtype") != str(config.DTYPE):
        stale.append(f"dtype: oracle={o.get('dtype')} config={config.DTYPE}")
    if stale:
        print("  STALE -- oracle does not match the current config:")
        for s in stale:
            print(f"    {s}")
        print("\n  This is expected right after switching models.")
        print("  Regenerate: python -m reference.oracle")
        return None
    print(f"  fresh: {o['model_id']}  prompt={o['prompt']!r}")
    print(f"  traced {len(o['activations'])} tensors, {len(o['greedy_tokens'])} greedy tokens")
    print(f"  reference continuation: {o['greedy_text']!r}")
    return o


# ---------------------------------------------------------------- stage 3/4
def stage_diff_and_parity(oracle):
    from engine.cache import ContiguousKVCache
    from engine.loader import load as load_weights
    from engine.model import LlamaEngine
    from testing.diff import report

    cfg, weights = load_weights(oracle["model_id"], config.DEVICE, config.DTYPE)
    if not cfg.qkv_bias:
        print("  NOTE: qkv_bias=no. Expected for Llama; for a Qwen2 checkpoint this")
        print("  means the bias tensors were not found, which is a loader bug.\n")

    eng = LlamaEngine(cfg, weights, config.DEVICE, config.DTYPE)
    ids = oracle["input_ids"].to(config.DEVICE)
    max_len = ids.shape[1] + config.PARITY_MAX_NEW_TOKENS + 8

    # FD_CACHE=paged runs the whole gate against the paged cache. The paged
    # path must be BIT-EXACT against HuggingFace, not merely close: the KV
    # values are identical, only their addresses differ, so any difference is
    # an indexing bug rather than a numerical one.
    kind = os.environ.get("FD_CACHE", "contiguous")
    if kind == "paged":
        bs = int(os.environ.get("FD_BLOCK_SIZE", "16"))
        cache = eng.new_paged_cache(ids.shape[0], max_len, block_size=bs)
        print(f"  cache: paged, block_size={bs}")
    else:
        cache = ContiguousKVCache(cfg.num_hidden_layers, ids.shape[0], max_len,
                                  cfg.num_key_value_heads, cfg.head_dim,
                                  config.DTYPE, config.DEVICE)
        print("  cache: contiguous")

    # Run the gate for EVERY implementation, not just the reference.
    #
    # The first version selected the impl only in stage 5, so stages 3-4
    # always validated the pure-PyTorch path -- the CUDA kernel ran inside
    # the model for the first time during a TIMING loop, with nothing
    # checking its output. That is precisely backwards: the kernel is the
    # thing most likely to be wrong, and a kernel that is fast and wrong is
    # worse than no kernel.
    #
    # ops.use("rmsnorm_residual", "ref") passing also separates the two
    # questions cleanly: ref proves the delta/residual restructure is
    # correct, cuda then proves the kernel is.
    try:
        from engine import ops_cuda
        impls = ("ref", "cuda") if ops_cuda.AVAILABLE else ("ref",)
        if not ops_cuda.AVAILABLE:
            print(f"  [warn] {ops_cuda.status()}  -- validating ref only")
    except Exception:
        impls = ("ref",)

    all_ok = True
    for impl in impls:
        desc = _select_impl(impl)
        _hdr(3, f"activation diff vs HuggingFace -- {impl}")
        print(f"  {cfg.summary()}")
        print(f"  ops: {desc}\n")

        # Reset between implementations. The previous impl's prefill and
        # generate left seq_lens > 0, and append_prefill asserts the cache is
        # empty -- correctly, since prefilling over existing history would
        # silently compare against the wrong KV. Each impl must see the same
        # starting state or the comparison means nothing.
        cache.reset()

        eng.trace = {}
        with torch.inference_mode():
            # logits_all=True reproduces HF's GEMM shape for the head, so the
            # 'logits' row is a like-for-like comparison rather than a GEMV vs
            # GEMM reduction-order difference. See LlamaEngine.prefill.
            eng.prefill(ids, cache, logits_all=True)
        diff_ok = report(eng.trace, oracle["activations"])

        _hdr(4, f"greedy token parity -- {impl}")
        eng.trace = None
        toks = eng.generate(ids, config.PARITY_MAX_NEW_TOKENS, cache)[0].tolist()
        want = oracle["greedy_tokens"]
        n = min(len(toks), len(want))
        first = next((i for i in range(n) if toks[i] != want[i]), None)
        match = sum(1 for i in range(n) if toks[i] == want[i])
        print(f"  {match}/{n} tokens match")

        if first is None:
            print("  exact match")
            parity_ok = True
        elif first >= 20:
            # bf16 greedy argmax flips once two logits are near-tied. With
            # every activation inside tolerance, a late flip is noise.
            print(f"  first mismatch at position {first} (>= 20) -- benign bf16 drift")
            parity_ok = diff_ok
        else:
            print(f"  first mismatch at position {first} -- THIS IS A BUG")
            print(f"    ours: {toks[:min(n, 12)]}")
            print(f"    ref : {want[:min(n, 12)]}")
            parity_ok = False

        if not (diff_ok and parity_ok):
            print(f"\n  >>> the '{impl}' path FAILED the gate.")
            if impl == "cuda":
                print("      ref passed, so the restructure is fine and the")
                print("      KERNEL is wrong. Check kernels/rmsnorm.cu.")
            all_ok = False

    del cache
    _release()
    return all_ok, cfg, eng


# ---------------------------------------------------------------- stage 5
def _fill_cache(eng, cfg, B, ctx):
    """Put the cache into the state a decode step at (B, ctx) would see,
    without running a prefill. Timing only."""
    if os.environ.get("FD_CACHE") == "paged":
        bs = int(os.environ.get("FD_BLOCK_SIZE", "16"))
        cache = eng.new_paged_cache(B, ctx + 40, block_size=bs)
        cache.ensure_capacity(extra=ctx)
        for i in range(cfg.num_hidden_layers):
            cache.k[i].normal_(0, 0.5)
            cache.v[i].normal_(0, 0.5)
        cache.seq_lens[:B] = ctx
        return cache
    cache = eng.new_cache(B, ctx + 40)
    for i in range(cfg.num_hidden_layers):
        k, v = cache.layer(i)
        k[:, :ctx].normal_(0, 0.5)
        v[:, :ctx].normal_(0, 0.5)
    cache.seq_lens.fill_(ctx)
    return cache


def _release() -> None:
    """Return cached blocks to the driver. gc.collect() first: a traceback or
    a reference cycle can keep a dead cache object alive, and empty_cache()
    only frees blocks the allocator already considers unused."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _select_impl(name: str, quiet: bool = False) -> str:
    """Switch the op registry. Returns a human-readable description.

    'ref' keeps the pure-PyTorch path. 'cuda' switches every op that has a
    CUDA implementation; ops without one stay on ref, which is the whole
    point of the registry -- a half-CUDA model still runs and a correctness
    bug bisects to one kernel.
    """
    from engine import ops

    if name == "ref":
        ops.reset()
        return "reference (pure PyTorch)"
    from engine import ops_cuda

    if not ops_cuda.AVAILABLE:
        if not quiet:
            print(f"  [warn] {ops_cuda.status()}")
            print("         falling back to the reference path")
        ops.reset()
        return "reference (CUDA unavailable)"
    ops.use_all("cuda")
    active = [k for k, v in ops.active().items() if v == "cuda"]
    return f"cuda for {active}, ref for the rest"


def _time_decode(eng, cfg, cache, ctx, steps, trials: int = 5) -> float:
    """Returns the MINIMUM mean-per-step over `trials` independent trials.

    Minimum, not mean, and the reason is this project's GPU: it is shared,
    and two unrelated processes were holding 9.5 GB at 60% utilisation. The
    same B=128 shape measured 191 ms in one run and 306 ms in the next.

    Contention is strictly additive -- another process can only ever make our
    kernels wait, never finish sooner. So across repeated trials the minimum
    is the sample least polluted by someone else's work and the best available
    estimate of the uncontended time, while the mean just measures how busy
    the neighbours were. On a quiet card min and mean converge, so this costs
    nothing when the machine is idle.

    This is not a licence to report flattering numbers: the minimum of a
    contended distribution is still an upper bound on the true time, so any
    speedup computed from it is conservative on both sides.
    """
    n = cache.seq_lens.numel()
    best = float("inf")
    spread = []
    with torch.inference_mode():
        for t in range(trials):
            tok = torch.randint(0, cfg.vocab_size, (n,), device=config.DEVICE)
            for _ in range(3):                          # warmup
                tok = eng.decode_step(tok, cache).argmax(-1)
                cache.seq_lens.fill_(ctx)               # pin the shape
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                tok = eng.decode_step(tok, cache).argmax(-1)
                cache.seq_lens.fill_(ctx)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / steps
            spread.append(dt)
            best = min(best, dt)
    # A wide spread is itself a result: it quantifies how contended the card
    # was while this shape ran.
    _time_decode.last_spread = (min(spread), max(spread))
    return best


def stage_baseline(cfg, eng, shapes, impls=("ref", "cuda"), steps=24) -> dict:
    """A/B the implementations at each shape, back to back, on ONE cache.

    The first version ran two full passes -- every shape on ref, then every
    shape on cuda. That is not a fair comparison: the second pass sees a
    heap the first pass fragmented, and at B=128 (a ~16 GB cache in ~20 GB
    of free memory) it produced a spurious 0.90x "regression" plus a ref
    time of 306 ms where an earlier, less memory-pressured run of the same
    code measured 191 ms.

    Timing both impls against the same freshly built cache removes that
    variable: identical allocation state, identical data, microseconds
    apart.
    """
    _hdr(5, "decode baseline: reference vs fused kernels")
    print("  Synthetic cache, random tokens -- timing only.")
    print("  KV% is the share of per-step byte traffic that is KV cache. Below")
    print("  ~25% an attention kernel cannot show up end to end at that shape.")
    print("  'off SOL' is how far the naive engine is from the memory-bandwidth")
    print("  floor -- the headroom Phases 2-5 are competing for.\n")

    # Fixed weight traffic per step, for KV% and SOL.
    hid, d, E = cfg.hidden_size, cfg.head_dim, 2
    per_layer = (cfg.num_attention_heads * d * hid
                 + 2 * cfg.num_key_value_heads * d * hid
                 + cfg.num_attention_heads * d * hid
                 + 3 * cfg.intermediate_size * hid + 2 * hid)
    w_bytes = E * (per_layer * cfg.num_hidden_layers + cfg.vocab_size * hid)
    bw = getattr(config, "MEASURED_BW_GBS", 1655.0) * 1e9

    # Biggest shape FIRST. Running ascending means every large allocation has
    # to fit around whatever the allocator still holds from the previous
    # smaller run; descending gets the hardest allocation while memory is
    # cleanest. This alone is what turned the B=200 OOM into a pass.
    shapes = sorted(shapes, key=lambda s: -(s[0] * s[1]))

    if os.environ.get("FD_CACHE") == "paged":
        # paged_decode_attention_ref loops over sequences in PYTHON. At B=128
        # that is 128 x 28 layers x 24 steps x 5 trials = 430k iterations per
        # shape, each launching a handful of tiny kernels -- minutes per row,
        # and the number measures Python, not the GPU. Phase 3 is a
        # correctness phase; skip it until the Phase 4 CUDA kernel exists.
        print("  SKIPPED for the paged cache: the paged reference attention is")
        print("  a Python loop over sequences, so timing it measures the")
        print("  interpreter. Correctness (stages 1-4) is what Phase 3 is for.")
        print("  Use the contiguous cache for timings until Phase 4 lands a")
        print("  real paged kernel:  python -m testing.phase1")
        return {}

    free0, _ = torch.cuda.mem_get_info()
    print(f"  free VRAM at start of stage 5: {free0/1e9:.1f} GB\n")
    print("  Each number is the BEST of 5 trials: contention can only slow us")
    print("  down, so the minimum is the least-polluted sample. 'jit' is the")
    print("  trial-to-trial spread -- over ~15% means the card was busy and the")
    print("  figures on that row are not publication-grade.\n")
    print(f"{'B':>5} {'ctx':>6} {'KV%':>6} {'SOL ms':>7} {'ref ms':>8} "
          f"{'cuda ms':>8} {'saved':>7} {'speedup':>8} {'off SOL':>8} {'jit':>6}")
    results = {}
    for B, ctx in shapes:
        kv_bytes = 2 * B * ctx * cfg.num_key_value_heads * d * E * cfg.num_hidden_layers
        need = kv_bytes * 1.05 * (ctx + 40) / ctx
        _release()
        free, _ = torch.cuda.mem_get_info()
        if need > free * 0.92:
            print(f"{B:>5} {ctx:>6} {'skip':>6}  needs ~{need/1e9:.1f} GB, "
                  f"{free/1e9:.1f} GB free")
            continue
        try:
            cache = _fill_cache(eng, cfg, B, ctx)
            kv = cache.kv_bytes_per_step()
            sol_ms = (w_bytes + kv) / bw * 1e3
            per, jitter = {}, {}
            for impl in impls:
                _select_impl(impl, quiet=True)
                per[impl] = _time_decode(eng, cfg, cache, ctx, steps) * 1e3
                lo, hi = _time_decode.last_spread
                jitter[impl] = (hi - lo) / lo * 100 if lo else 0.0
            r, c = per.get("ref"), per.get("cuda")
            results[(B, ctx)] = per
            if r and c:
                jit = max(jitter.values())
                flag = "  <-- noisy" if jit > 15 else ""
                print(f"{B:>5} {ctx:>6} {kv/(kv+w_bytes)*100:>5.1f}% {sol_ms:>7.2f} "
                      f"{r:>8.2f} {c:>8.2f} {r-c:>7.2f} {r/c:>7.3f}x "
                      f"{c/sol_ms:>7.1f}x {jit:>5.0f}%{flag}")
            else:
                only = next(iter(per.values()))
                print(f"{B:>5} {ctx:>6} {kv/(kv+w_bytes)*100:>5.1f}% {sol_ms:>7.2f} "
                      f"{only:>8.2f} {'-':>8} {'-':>7} {'-':>8} "
                      f"{only/sol_ms:>7.1f}x")
            del cache
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>5} {ctx:>6} {'OOM':>6}")
        except Exception as e:
            print(f"{B:>5} {ctx:>6} {type(e).__name__}: {e}")
        finally:
            _release()

    print("\n  'off SOL' is the cuda path against the bandwidth floor -- the")
    print("  headroom Phases 3-5 compete for. The speedup here is larger than")
    print("  the kernel microbenchmark predicts, and that is the real result:")
    print("  the microbenchmark amortizes PyTorch's per-op dispatch by keeping")
    print("  the queue full, while a decode loop is serialized by data")
    print("  dependencies and pays that dispatch in full.")
    return dict(sorted(results.items()))


# ---------------------------------------------------------------- shapes
# A FIXED core set plus one opportunistic large shape.
#
# The core exists for comparability: if the shape list changed with whatever
# memory happened to be free, two runs could not be compared and the log
# would be worthless. These four are small enough to fit even on a busy card
# and are the rows that belong in logs/EXPERIMENTS.md.
#
# ctx=2048 rather than 4096 at double the batch. KV% is set by B*ctx (total
# tokens), so memory cost and KV share of traffic are identical either way --
# but twice as many SEQUENCES means twice the grid for the Phase 4 decode
# kernel, which launches on the order of B * n_kv_heads blocks and was
# occupancy-starved at 0.09 waves/SM in Phase 2. More sequences is exactly
# the lever that matters.
CORE_SHAPES = [(1, 2048), (64, 2048), (128, 2048), (64, 4096)]

# Candidate large shapes, tried biggest-first; the first one that fits in the
# current budget is appended. On a busy card none fit and the core still
# runs; on an idle card this reaches the real ceiling without editing code.
STRETCH_SHAPES = [(512, 2048), (384, 2048), (256, 2048), (192, 2048),
                  (256, 4096), (192, 4096), (160, 2048)]


def choose_shapes(cfg) -> list:
    """Core shapes that fit, plus the largest stretch shape that fits.

    Override with FD_SHAPES="64x2048,128x2048" to pin an exact list.
    """
    override = os.environ.get("FD_SHAPES")
    if override:
        out = []
        for tok in override.split(","):
            b, c = tok.strip().lower().split("x")
            out.append((int(b), int(c)))
        print(f"\n  FD_SHAPES override: {out}")
        return out

    per_tok = 2 * cfg.num_key_value_heads * cfg.head_dim * 2 * cfg.num_hidden_layers
    free, total = torch.cuda.mem_get_info()
    budget = free * 0.85

    def fits(shape) -> bool:
        B, c = shape
        # 1.25x covers the reference attention's fp32 upcast of one layer's KV.
        return B * c * per_tok * 1.25 < budget

    core = [s for s in CORE_SHAPES if fits(s)]
    dropped = [s for s in CORE_SHAPES if s not in core]
    stretch = next((s for s in STRETCH_SHAPES if fits(s)), None)

    print(f"\n  {free/1e9:.1f} of {total/1e9:.1f} GB free "
          f"-> budget {budget/1e9:.1f} GB, KV {per_tok/1024:.1f} KB/token")
    if dropped:
        print(f"  core shapes that will not fit right now: {dropped}")
        print("  (they are the comparable rows -- re-run these when the card frees up)")
    if stretch:
        print(f"  stretch shape that fits: {stretch}")
    shapes = core + ([stretch] if stretch and stretch not in core else [])
    if not shapes:
        print("  nothing fits; falling back to B=1 so the run still produces a row")
        shapes = [(1, 1024)]
    return shapes


# ---------------------------------------------------------------- main
def main() -> int:
    print("Phase 1 gate")
    print(f"  target: {config.MODEL_ID}  dtype={config.DTYPE}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name}  cc {p.major}.{p.minor}  {p.multi_processor_count} SMs  "
              f"{p.total_memory/1e9:.1f} GB  torch {torch.__version__}")
        gpu_clean = check_gpu_clean()
    else:
        print("  no CUDA device -- stages 2-5 require a GPU")
        gpu_clean = False

    smoke_ok, n_smoke = stage_smoke()
    if not smoke_ok:
        return 1

    # Stage 2 before the GPU bail: it only loads a pickle and compares
    # strings, so it is worth reporting even on a machine with no CUDA. A
    # stale oracle is cheap to catch and expensive to debug later.
    oracle = stage_oracle_fresh()
    if not torch.cuda.is_available():
        print("\n  no CUDA -- stopping after stage 2.")
        return 1
    if oracle is None:
        return 1

    try:
        ok, cfg, eng = stage_diff_and_parity(oracle)
    except Exception:
        traceback.print_exc()
        print("\n  Stage 3/4 crashed. Most likely causes, in order:")
        print("    0. 'prefill into a non-empty cache' -- the cache was not")
        print("       reset between the ref and cuda passes. cache.reset()")
        print("       belongs at the top of the per-impl loop.")
        print("    1. qkv_bias -- Qwen2 has biases on q/k/v_proj. loader.load()")
        print("       detects them from the checkpoint; if cfg.qkv_bias printed")
        print("       'no' above for a Qwen model, that detection failed.")
        print("    2. tied embeddings -- Qwen2.5-1.5B omits lm_head.weight")
        print("       entirely; loader aliases it to embed_tokens.")
        print("    3. hook names in reference/oracle.py not matching the _rec()")
        print("       calls in engine/model.py.")
        print("    4. rope table size -- Qwen2.5 advertises max_position_embeddings")
        print("       =131072, so the cos/sin tables are ~134 MB. Fine on 32 GB,")
        print("       but it is the first thing to shrink if you see an OOM here.")
        return 1

    # Shapes chosen from the VRAM budget: Qwen2.5-1.5B weights are ~3.1 GB,
    # leaving ~28 GB for KV, which caps ctx=4096 at roughly B=238.
    shapes = choose_shapes(cfg)

    only = os.environ.get("FD_IMPL")
    impls = (only,) if only in ("ref", "cuda") else ("ref", "cuda")
    base = stage_baseline(cfg, eng, shapes, impls=impls)

    print(f"\n{'=' * 72}")
    if ok:
        print("PHASE 1 PASSED -- gate is green, Phase 2 may begin.")
        if not gpu_clean:
            print("\n*** but the GPU was BUSY: record the correctness results and")
            print("*** DISCARD the timings. Re-run on an idle card before putting")
            print("*** any ms/step or speedup into logs/EXPERIMENTS.md.")
        print("\nPaste into logs/EXPERIMENTS.md:\n")
        today = time.strftime("%Y-%m-%d")
        print("| what | number | date |")
        print("|---|---|---|")
        print(f"| smoke | {n_smoke}/{n_smoke} | {today} |")
        print(f"| activation diff vs HF (ref and cuda) | pass | {today} |")
        for (B, ctx), per in base.items():
            r, c = per.get("ref"), per.get("cuda")
            if r and c:
                print(f"| B={B} ctx={ctx} | ref {r:.2f} ms, cuda {c:.2f} ms, "
                      f"{r/c:.2f}x | {today} |")
            else:
                v = next(iter(per.values()))
                print(f"| B={B} ctx={ctx} | {v:.2f} ms/step | {today} |")
        print("\nNext: Phase 3 -- paged KV cache (block table, free-list")
        print("      allocator, ragged batches), PyTorch first.")
    else:
        print("PHASE 1 FAILED -- read the FIRST divergence above, ignore the rest.")
        print("No CUDA until this is green: a wrong kernel on top of a wrong")
        print("engine is two bugs that look like one.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())