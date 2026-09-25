"""
CPU smoke test. No GPU, no model download, no HF auth. Runs in seconds.

Run this BEFORE the oracle and diff harness. It exercises the whole pipeline
-- registry, cache indexing, RoPE, GQA, prefill, decode -- on a tiny
random-weight model, so structural bugs surface in seconds instead of after
a 16 GB download and a failed parity run.

The load-bearing test is decode_equals_prefill. Running prefill over S tokens
must produce the same final logits as prefill over S-1 tokens followed by one
decode_step on token S. That single invariant catches, without needing real
weights:

  - RoPE applied at the wrong position in the decode path
  - KV cache written to or read from the wrong slot
  - the off-by-one in the attention mask (seq_lens vs seq_lens+1)
  - GQA head expansion mismatched between prefill and decode
  - q/k/v reshape order differing between the two paths

WHAT THIS SUITE CANNOT CATCH, verified by mutation testing:

It runs in fp32, so every dtype-ordering bug is invisible here. Concretely,
moving the weight multiply in rmsnorm to before the cast back to bf16 passes
all 12 tests, because in fp32 that cast is a no-op. The same blindness
applies to accumulator precision choices and any bf16 rounding difference.

Those only surface in the real bf16 parity run. So: smoke test first to
catch structure, then testing/diff.py against the oracle to catch numerics.
Neither replaces the other.

    python -m testing.smoke
"""

from __future__ import annotations

import torch

from config import LlamaConfig
from engine import ops
from engine.loader import validate
from engine.model import LlamaEngine

DEV = "cpu"
DT = torch.float32          # fp32 on CPU: we are testing logic, not numerics


def tiny_config(rope_scaling=None) -> LlamaConfig:
    """Deliberately asymmetric: GQA group of 3 and head_dim that is not
    hidden/n_heads would both be silently wrong under sloppy reshapes."""
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
    )


def random_weights(cfg: LlamaConfig, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g, dtype=DT) * scale)

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
    return w


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------

def t_registry():
    assert ops.active()["rmsnorm"] == "ref"

    @ops.register("rmsnorm", "fake")
    def _fake(x, weight, eps):
        return torch.zeros_like(x)

    ops.use("rmsnorm", "fake")
    assert ops.active()["rmsnorm"] == "fake"
    assert torch.all(ops.call("rmsnorm", torch.ones(4), torch.ones(4), 1e-5) == 0)
    ops.reset()
    assert ops.active()["rmsnorm"] == "ref"
    return "registry swap + reset"


def t_rmsnorm():
    x = torch.randn(2, 16)
    w = torch.randn(16)
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
    # Both layers wrote slot 0 because advance() had not been called.
    assert c.k[0][:, 0].eq(1).all() and c.k[1][:, 0].eq(9).all()
    assert c.k[0][:, 1].eq(0).all(), "wrote past the current position"
    c.advance(1)
    assert c.positions.tolist() == [1, 1]
    c.append(0, torch.full((2, 2, 4), 2.0), torch.zeros(2, 2, 4))
    assert c.k[0][:, 1].eq(2).all() and c.k[0][:, 0].eq(1).all()
    return "cache writes at seq_lens, advance is per-step not per-layer"


def t_shapes():
    cfg = tiny_config()
    eng = LlamaEngine(cfg, random_weights(cfg), DEV, DT)
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
    """THE test. prefill(S) must equal prefill(S-1) + decode_step(token S)."""
    cfg = tiny_config()
    eng = LlamaEngine(cfg, random_weights(cfg), DEV, DT)
    ids = torch.randint(0, cfg.vocab_size, (2, 7))

    with torch.inference_mode():
        a = eng.prefill(ids, eng.new_cache(2, 32))
        cache = eng.new_cache(2, 32)
        eng.prefill(ids[:, :-1], cache)
        b = eng.decode_step(ids[:, -1], cache)

    err = (a - b).abs().max().item()
    assert err < 1e-4, f"decode diverges from prefill: max_abs={err:.3e}"
    return f"decode == prefill (max_abs={err:.2e})"


def t_multi_step_consistency():
    """Several decode steps must equal one prefill over the same tokens."""
    cfg = tiny_config()
    eng = LlamaEngine(cfg, random_weights(cfg), DEV, DT)
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
    """Duplicated sequences in a batch must give identical results -- catches
    cross-sequence leakage in the cache or the mask."""
    cfg = tiny_config()
    eng = LlamaEngine(cfg, random_weights(cfg), DEV, DT)
    one = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.inference_mode():
        a = eng.generate(one, 4)
        b = eng.generate(one.repeat(3, 1), 4)
    assert torch.equal(b[0], b[1]) and torch.equal(b[1], b[2]), "batch rows differ"
    assert torch.equal(a[0], b[0]), "batch 1 differs from batch 3"
    return "batch independence (B=1 == B=3)"


def t_generate():
    cfg = tiny_config()
    eng = LlamaEngine(cfg, random_weights(cfg), DEV, DT)
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    out = eng.generate(ids, 6)
    assert out.shape == (2, 6), out.shape
    assert out.min() >= 0 and out.max() < cfg.vocab_size
    return "generate() greedy loop"


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


TESTS = [
    t_registry, t_rmsnorm, t_rope_convention, t_rope_scaling,
    t_decode_attention_ragged, t_cache_indexing, t_shapes,
    t_decode_equals_prefill, t_multi_step_consistency,
    t_batch_independence, t_generate, t_validate_catches_bad_shapes,
]


def main() -> int:
    torch.manual_seed(0)
    print(f"CPU smoke test  ({tiny_config().summary()})\n")
    failed = 0
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<32} {fn()}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__:<32} {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"{failed}/{len(TESTS)} failed -- fix before downloading weights.")
        return 1
    print(f"all {len(TESTS)} passed. Pipeline logic is sound; next is real weights:")
    print("  python -m reference.oracle && python -m testing.diff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())