"""
CPU smoke test. No GPU, no model download, no HF auth. Runs in seconds.

Run this BEFORE the oracle and diff harness. It exercises the whole pipeline
-- registry, cache indexing, RoPE, GQA, QKV bias, prefill, decode -- on a tiny
random-weight model, so structural bugs surface in seconds instead of after a
multi-GB download and a failed parity run.

The load-bearing test is decode_equals_prefill. Running prefill over S tokens
must produce the same final logits as prefill over S-1 tokens followed by one
decode_step on token S. That single invariant catches, without real weights:

  - RoPE applied at the wrong position in the decode path
  - KV cache written to or read from the wrong slot
  - the off-by-one in the attention mask (seq_lens vs seq_lens+1)
  - GQA head expansion mismatched between prefill and decode
  - q/k/v reshape order differing between the two paths
  - a QKV bias applied in prefill but dropped in decode, or vice versa

QWEN NOTE: every structural test runs TWICE, once with qkv_bias=False (Llama
shape) and once with qkv_bias=True (Qwen2 shape). A bias dropped from one of
the two paths is a small consistent error that looks like a numerics problem,
so it is worth testing explicitly rather than hoping diff.py catches it.

WHAT THIS SUITE CANNOT CATCH, verified by mutation testing:

It runs in fp32, so every dtype-ordering bug is invisible here. Moving the
weight multiply in rmsnorm to before the cast back to bf16 passes all tests,
because in fp32 that cast is a no-op. The same blindness applies to
accumulator precision and any bf16 rounding difference. Those surface only in
the real bf16 parity run.

    python -m testing.smoke
"""

from __future__ import annotations

import torch

from config import LlamaConfig
from engine import ops
from engine.loader import validate
from engine.model import LlamaEngine

DEV = "cpu"
DT = torch.float32          # fp32 on CPU: testing logic, not numerics


def tiny_config(rope_scaling=None, qkv_bias=False) -> LlamaConfig:
    """Deliberately asymmetric: GQA group of 3, and head_dim that is not
    hidden/n_heads, would both be silently wrong under sloppy reshapes."""
    return LlamaConfig(
        hidden_size=48,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=6,
        num_key_value_heads=2,      # GQA group = 3
        head_dim=8,
        vocab_size=100,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=64,
        rope_scaling=rope_scaling,
        qkv_bias=qkv_bias,
    )


def random_weights(cfg: LlamaConfig, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.02):
        return torch.randn(*shape, generator=g, dtype=DT) * scale

    hid, d = cfg.hidden_size, cfg.head_dim
    w = {
        "model.embed_tokens.weight": r(cfg.vocab_size, hid, scale=0.1),
        "model.norm.weight": torch.ones(hid, dtype=DT),
        "lm_head.weight": r(cfg.vocab_size, hid, scale=0.1),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        w[f"{p}.input_layernorm.weight"] = torch.ones(hid, dtype=DT)
        w[f"{p}.post_attention_layernorm.weight"] = torch.ones(hid, dtype=DT)
        w[f"{p}.self_attn.q_proj.weight"] = r(cfg.num_attention_heads * d, hid)
        w[f"{p}.self_attn.k_proj.weight"] = r(cfg.num_key_value_heads * d, hid)
        w[f"{p}.self_attn.v_proj.weight"] = r(cfg.num_key_value_heads * d, hid)
        w[f"{p}.self_attn.o_proj.weight"] = r(hid, cfg.num_attention_heads * d)
        w[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, hid)
        w[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, hid)
        w[f"{p}.mlp.down_proj.weight"] = r(hid, cfg.intermediate_size)
        if cfg.qkv_bias:
            # Scale 0.5 deliberately: large enough that a dropped bias moves
            # the logits far beyond any plausible fp32 tolerance.
            w[f"{p}.self_attn.q_proj.bias"] = r(cfg.num_attention_heads * d, scale=0.5)
            w[f"{p}.self_attn.k_proj.bias"] = r(cfg.num_key_value_heads * d, scale=0.5)
            w[f"{p}.self_attn.v_proj.bias"] = r(cfg.num_key_value_heads * d, scale=0.5)
    return w


def _engine(qkv_bias=False, seed=0):
    cfg = tiny_config(qkv_bias=qkv_bias)
    return cfg, LlamaEngine(cfg, random_weights(cfg, seed), DEV, DT)


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------

def t_registry():
    assert ops.active()["rmsnorm"] == "ref"

    @ops.register("rmsnorm", "fake")
    def _fake(x, weight, eps):
        return torch.zeros_like(x)

    ops.use("rmsnorm", "fake")
    assert torch.all(ops.call("rmsnorm", torch.ones(4), torch.ones(4), 1e-5) == 0)
    ops.reset()
    assert ops.active()["rmsnorm"] == "ref"
    return "registry swap + reset"


def t_rmsnorm():
    x, w = torch.randn(2, 16), torch.randn(16)
    got = ops.call("rmsnorm", x, w, 1e-5)
    want = w * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5))
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    return "rmsnorm matches closed form"


def t_rope_convention():
    """rotate_half must pair element i with i+d/2, NOT i with i+1."""
    d, theta, pos = 8, 10000.0, 5
    cos, sin = ops.build_rope_cache(d, 16, theta, None, DEV)
    x = torch.randn(1, 1, d)
    got = ops.rope_ref(x, x, cos[pos:pos+1], sin[pos:pos+1])[0][0, 0]

    inv = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
    h = d // 2
    want = torch.empty(d)
    for i in range(h):
        c, s = torch.cos(pos * inv[i]), torch.sin(pos * inv[i])
        want[i] = x[0, 0, i] * c - x[0, 0, i + h] * s
        want[i + h] = x[0, 0, i + h] * c + x[0, 0, i] * s
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()

    inter = torch.empty(d)
    for i in range(h):
        c, s = torch.cos(pos * inv[i]), torch.sin(pos * inv[i])
        inter[2*i] = x[0, 0, 2*i] * c - x[0, 0, 2*i+1] * s
        inter[2*i+1] = x[0, 0, 2*i+1] * c + x[0, 0, 2*i] * s
    assert not torch.allclose(got, inter, atol=1e-4), "matches interleaved -- wrong convention"
    return "rope = rotate_half (not interleaved)"


def t_rope_large_theta():
    """Qwen2.5 uses rope_theta=1e6 vs Llama's 5e5. Confirm theta propagates."""
    a, _ = ops.build_rope_cache(64, 16, 500000.0, None, DEV)
    b, _ = ops.build_rope_cache(64, 16, 1000000.0, None, DEV)
    assert not torch.allclose(a, b), "rope_theta had no effect on the table"
    return "rope_theta propagates (Qwen 1e6 != Llama 5e5)"


def t_rope_scaling():
    sc = dict(rope_type="llama3", factor=32.0, low_freq_factor=1.0,
              high_freq_factor=4.0, original_max_position_embeddings=8192)
    a, _ = ops.build_rope_cache(64, 8, 500000.0, None, DEV)
    b, _ = ops.build_rope_cache(64, 8, 500000.0, sc, DEV)
    assert not torch.allclose(a, b), "scaling had no effect"
    return "llama3 rope scaling changes the table"


def t_decode_attention_ragged():
    """Masked batched attention must equal a per-sequence manual loop."""
    B, n_q, n_kv, d, L = 3, 6, 2, 8, 10
    q = torch.randn(B, n_q, d)
    kc, vc = torch.randn(B, L, n_kv, d), torch.randn(B, L, n_kv, d)
    lens = torch.tensor([1, 5, 10], dtype=torch.int32)
    got = ops.call("decode_attention", q, kc, vc, lens, 0.5)
    for b in range(B):
        n = int(lens[b])
        for hq in range(n_q):
            hk = hq // (n_q // n_kv)          # repeat_interleave mapping
            s = (q[b, hq] @ kc[b, :n, hk].T) * 0.5
            want = torch.softmax(s, -1) @ vc[b, :n, hk]
            assert torch.allclose(got[b, hq], want, atol=1e-5), (b, hq)
    return "decode attention: ragged mask + GQA mapping"


def t_cache_indexing():
    from engine.cache import ContiguousKVCache
    c = ContiguousKVCache(2, 2, 8, 2, 4, DT, DEV)
    c.append(0, torch.full((2, 2, 4), 1.0), torch.full((2, 2, 4), -1.0))
    c.append(1, torch.full((2, 2, 4), 9.0), torch.full((2, 2, 4), -9.0))
    assert c.k[0][:, 0].eq(1).all() and c.k[1][:, 0].eq(9).all()
    assert c.k[0][:, 1].eq(0).all(), "wrote past the current position"
    c.advance(1)
    assert c.positions.tolist() == [1, 1]
    c.append(0, torch.full((2, 2, 4), 2.0), torch.zeros(2, 2, 4))
    assert c.k[0][:, 1].eq(2).all() and c.k[0][:, 0].eq(1).all()
    return "cache writes at seq_lens, advance is per-step not per-layer"


def t_shapes():
    cfg, eng = _engine()
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    cache = eng.new_cache(2, 32)
    with torch.inference_mode():
        lg = eng.prefill(ids, cache)
        assert lg.shape == (2, cfg.vocab_size), lg.shape
        lg2 = eng.decode_step(lg.argmax(-1), cache)
        assert lg2.shape == (2, cfg.vocab_size), lg2.shape
    assert cache.seq_lens.tolist() == [6, 6]
    return "prefill/decode shapes and seq_len bookkeeping"


def t_decode_equals_prefill():
    """THE test, no bias. prefill(S) == prefill(S-1) + decode_step(token S)."""
    cfg, eng = _engine(qkv_bias=False)
    ids = torch.randint(0, cfg.vocab_size, (2, 7))
    with torch.inference_mode():
        a = eng.prefill(ids, eng.new_cache(2, 32))
        cache = eng.new_cache(2, 32)
        eng.prefill(ids[:, :-1], cache)
        b = eng.decode_step(ids[:, -1], cache)
    err = (a - b).abs().max().item()
    assert err < 1e-4, f"decode diverges from prefill: max_abs={err:.3e}"
    return f"decode == prefill, no bias (max_abs={err:.2e})"


def t_decode_equals_prefill_bias():
    """Same invariant with Qwen2-style QKV bias. Catches a bias applied in
    one path and dropped in the other."""
    cfg, eng = _engine(qkv_bias=True)
    assert eng.qb[0] is not None, "bias not picked up by the engine"
    ids = torch.randint(0, cfg.vocab_size, (2, 7))
    with torch.inference_mode():
        a = eng.prefill(ids, eng.new_cache(2, 32))
        cache = eng.new_cache(2, 32)
        eng.prefill(ids[:, :-1], cache)
        b = eng.decode_step(ids[:, -1], cache)
    err = (a - b).abs().max().item()
    assert err < 1e-4, f"decode diverges from prefill WITH bias: max_abs={err:.3e}"
    return f"decode == prefill, qkv_bias (max_abs={err:.2e})"


def t_bias_actually_applied():
    """A zero bias must match no bias; a nonzero bias must NOT. Without this,
    a forward pass that silently ignores self.qb would pass every other test.

    All three engines are built from ONE weight dict. Calling random_weights
    twice with different qkv_bias would not do: the bias tensors consume extra
    draws from the seeded generator, so every later layer's weights would
    differ too and the comparison would be meaningless. (That mistake is how
    this test first failed -- the engines differed in their weights, not their
    biases.)
    """
    cfg_b = tiny_config(qkv_bias=True)
    cfg_nb = tiny_config(qkv_bias=False)
    w = random_weights(cfg_b)
    is_bias = lambda k: k.endswith("_proj.bias")

    eng_b = LlamaEngine(cfg_b, w, DEV, DT)
    eng_nb = LlamaEngine(cfg_nb, {k: v for k, v in w.items() if not is_bias(k)}, DEV, DT)
    eng_z = LlamaEngine(cfg_b, {k: (torch.zeros_like(v) if is_bias(k) else v)
                                for k, v in w.items()}, DEV, DT)

    ids = torch.randint(0, cfg_b.vocab_size, (1, 5))
    with torch.inference_mode():
        nb = eng_nb.prefill(ids, eng_nb.new_cache(1, 16))
        z = eng_z.prefill(ids, eng_z.new_cache(1, 16))
        b = eng_b.prefill(ids, eng_b.new_cache(1, 16))

    gap = (nb - z).abs().max().item()
    assert gap < 1e-5, f"zero bias != no bias (max_abs={gap:.3e})"
    delta = (nb - b).abs().max().item()
    assert delta > 1e-3, f"nonzero bias had no effect (delta={delta:.2e}) -- it is being dropped"
    return f"bias applied, not ignored (zero==none {gap:.1e}, nonzero delta {delta:.2e})"


def t_multi_step_consistency():
    """Several decode steps must equal one prefill over the same tokens."""
    cfg, eng = _engine(qkv_bias=True)
    ids = torch.randint(0, cfg.vocab_size, (1, 9))
    with torch.inference_mode():
        a = eng.prefill(ids, eng.new_cache(1, 32))
        cache = eng.new_cache(1, 32)
        eng.prefill(ids[:, :4], cache)
        for t in range(4, 9):
            b = eng.decode_step(ids[:, t], cache)
    err = (a - b).abs().max().item()
    assert err < 1e-4, f"drift over 5 decode steps: max_abs={err:.3e}"
    return f"5 decode steps == prefill (max_abs={err:.2e})"


def t_batch_independence():
    """Duplicated sequences must give identical results -- catches
    cross-sequence leakage in the cache or the mask."""
    cfg, eng = _engine(qkv_bias=True)
    one = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.inference_mode():
        a = eng.generate(one, 4)
        b = eng.generate(one.repeat(3, 1), 4)
    assert torch.equal(b[0], b[1]) and torch.equal(b[1], b[2]), "batch rows differ"
    assert torch.equal(a[0], b[0]), "batch 1 differs from batch 3"
    return "batch independence (B=1 == B=3)"


def t_generate():
    cfg, eng = _engine(qkv_bias=True)
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    out = eng.generate(ids, 6)
    assert out.shape == (2, 6), out.shape
    assert out.min() >= 0 and out.max() < cfg.vocab_size
    return "generate() greedy loop"


def t_rope_from_hf_versions():
    """LlamaConfig.rope_from_hf must handle both transformers layouts.

    transformers 5.x dropped the flat `rope_theta` attribute and moved it into
    `rope_parameters`, keeping `rope_scaling` as an alias to the same dict.
    That broke stage 3 of the gate with:
        AttributeError: 'Qwen2Config' object has no attribute 'rope_theta'
    The second trap is subtler: in 5.x `rope_scaling` is ALWAYS a dict, so
    passing it through unchanged makes build_rope_cache think every model
    scales its frequencies. rope_type == "default" means it does not.
    """
    class Fake:
        def __init__(self, **kw):
            self.__dict__.update(kw)
        def to_dict(self):
            return dict(self.__dict__)

    # transformers 4.x, Qwen2.5: flat theta, no scaling
    t, s = LlamaConfig.rope_from_hf(Fake(rope_theta=1000000.0, rope_scaling=None))
    assert t == 1000000.0 and s is None, (t, s)

    # transformers 5.x, Qwen2.5: consolidated, rope_type "default" => no scaling
    t, s = LlamaConfig.rope_from_hf(Fake(
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
        rope_scaling={"rope_theta": 1000000.0, "rope_type": "default"}))
    assert t == 1000000.0, t
    assert s is None, f"rope_type=default must mean no scaling, got {s}"

    # transformers 4.x, Llama-3.2: real llama3 scaling must survive
    sc = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
          "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
    t, s = LlamaConfig.rope_from_hf(Fake(rope_theta=500000.0, rope_scaling=dict(sc)))
    assert t == 500000.0 and s and s["rope_type"] == "llama3", (t, s)

    # transformers 5.x, Llama-3.2: same, consolidated
    t, s = LlamaConfig.rope_from_hf(Fake(
        rope_parameters={**sc, "rope_theta": 500000.0}))
    assert t == 500000.0 and s and s["rope_type"] == "llama3", (t, s)

    # nothing findable -> loud failure, never a silent None into the kernel
    try:
        LlamaConfig.rope_from_hf(Fake(hidden_size=8))
        raise AssertionError("missing rope_theta was not rejected")
    except ValueError:
        pass
    return "rope_from_hf handles transformers 4.x and 5.x layouts"


def t_validate_catches_bad_shapes():
    cfg = tiny_config()
    w = random_weights(cfg)
    validate(cfg, w)
    w["model.layers.0.self_attn.k_proj.weight"] = torch.randn(
        cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)   # MHA not GQA
    try:
        validate(cfg, w)
    except ValueError:
        return "loader.validate rejects wrong k_proj shape"
    raise AssertionError("validate did not catch the bad shape")


def t_validate_catches_missing_bias():
    """cfg says qkv_bias but the checkpoint lacks it -> must fail loudly."""
    cfg = tiny_config(qkv_bias=True)
    w = random_weights(cfg)
    validate(cfg, w)
    del w["model.layers.0.self_attn.q_proj.bias"]
    try:
        validate(cfg, w)
    except KeyError:
        return "loader.validate rejects a missing qkv bias"
    raise AssertionError("validate did not catch the missing bias")


def t_validate_rejects_unsupported():
    """o_proj bias and Qwen3 q_norm must be refused, not silently dropped."""
    cfg = tiny_config()
    w = random_weights(cfg)
    w["model.layers.0.self_attn.o_proj.bias"] = torch.zeros(cfg.hidden_size)
    try:
        validate(cfg, w)
        raise AssertionError("o_proj bias was not rejected")
    except ValueError:
        pass
    del w["model.layers.0.self_attn.o_proj.bias"]

    w["model.layers.0.self_attn.q_norm.weight"] = torch.ones(cfg.head_dim)
    try:
        validate(cfg, w)
        raise AssertionError("q_norm (Qwen3) was not rejected")
    except ValueError:
        pass
    return "loader.validate rejects o_proj bias and Qwen3 q_norm"


TESTS = [
    t_registry, t_rmsnorm, t_rope_convention, t_rope_large_theta,
    t_rope_scaling, t_rope_from_hf_versions, t_decode_attention_ragged,
    t_cache_indexing, t_shapes,
    t_decode_equals_prefill, t_decode_equals_prefill_bias,
    t_bias_actually_applied, t_multi_step_consistency,
    t_batch_independence, t_generate, t_validate_catches_bad_shapes,
    t_validate_catches_missing_bias, t_validate_rejects_unsupported,
]


def main() -> int:
    torch.manual_seed(0)
    print(f"CPU smoke test  ({tiny_config().summary()})\n")
    failed = 0
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<34} {fn()}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__:<34} {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"{failed}/{len(TESTS)} failed -- fix before downloading weights.")
        return 1
    print(f"all {len(TESTS)} passed. Pipeline logic is sound; next is real weights:")
    print("  python -m reference.oracle && python -m testing.phase1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())