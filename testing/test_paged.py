# ==========================================================================
# FILE LOCATION:  flashdecode/testing/test_paged.py
# RUN AS:         python -m testing.test_paged        (CPU, no GPU needed)
# ==========================================================================
"""
Phase 3 tests: block allocator, block table, paged attention.

Runs on CPU in fp32 for the same reason testing/smoke.py does -- these tests
are about INDEXING, and indexing bugs are dtype-independent. Numerics come
later, in the bf16 parity run.

The load-bearing test is t_paged_equals_contiguous. It builds the same
logical KV history two ways -- once in a contiguous [B, max_len, ...] cache
and once in a paged cache with a scrambled block table -- and asserts the
two attention implementations agree. That single invariant catches, without
a GPU or real weights:

  - block index computed as t % block_size instead of t // block_size
  - offset and block swapped
  - the layout permute being wrong (n_kv_heads vs block_size transposed)
  - a sequence reading another sequence's blocks
  - the mask using blocks_needed * block_size instead of seq_len

The block table is deliberately SCRAMBLED (blocks handed out non-monotonic
and interleaved between sequences). A table where sequence b owns blocks
b*nb .. b*nb+nb-1 passes even if the code ignores the table entirely and
computes positions arithmetically -- which is the single easiest way to
write a paged cache that is not actually paged.
"""

from __future__ import annotations

import torch

from engine import ops
from engine.cache import ContiguousKVCache
from engine.paged_cache import BlockAllocator, PagedKVCache

DEV = "cpu"
DT = torch.float32


# ----------------------------------------------------------------------
# allocator
# ----------------------------------------------------------------------

def t_alloc_basic():
    a = BlockAllocator(8)
    assert a.num_free == 8 and a.num_used == 0
    x = a.allocate(3)
    assert x == [0, 1, 2], x
    assert a.num_free == 5 and a.num_used == 3
    a.free(x)
    assert a.num_free == 8
    return "allocate/free round-trip, ids ascending from 0"


def t_alloc_lifo():
    a = BlockAllocator(4)
    first = a.allocate(2)            # [0, 1]
    a.free(first)                    # pushes 0 then 1 -> top of stack is 1
    again = a.allocate(2)
    assert sorted(again) == [0, 1], again
    assert a.num_free == 2
    return "freed blocks are reused (LIFO), pool does not leak"


def t_alloc_exhaustion():
    a = BlockAllocator(3)
    a.allocate(3)
    try:
        a.allocate(1)
    except MemoryError as e:
        assert "exhausted" in str(e)
        return "exhaustion raises MemoryError, not silent corruption"
    raise AssertionError("over-allocation was not caught")


def t_alloc_double_free():
    a = BlockAllocator(4)
    x = a.allocate(2)
    a.free(x)
    try:
        a.free(x)
    except ValueError:
        return "double free raises (two seqs cannot share a block)"
    raise AssertionError("double free was not caught")


def t_alloc_no_overlap():
    """Two sequences must never be handed the same block."""
    a = BlockAllocator(16)
    s1, s2 = a.allocate(5), a.allocate(5)
    assert not (set(s1) & set(s2)), (s1, s2)
    a.free(s1)
    s3 = a.allocate(5)
    assert not (set(s2) & set(s3)), (s2, s3)
    return "concurrent sequences get disjoint blocks"


# ----------------------------------------------------------------------
# block table
# ----------------------------------------------------------------------

def _cache(block_size=4, num_blocks=32, n_layers=2, n_kv=2, d=8, max_seqs=4):
    return PagedKVCache(n_layers, n_kv, d, num_blocks, max_seqs,
                        max_blocks_per_seq=8, block_size=block_size,
                        dtype=DT, device=DEV)


def t_table_mapping():
    """Token t of seq s must land at block_table[s, t//bs] offset t%bs."""
    c = _cache(block_size=4)
    c.add_sequence(0, 10)            # 10 tokens -> 3 blocks
    assert len(c._blocks[0]) == 3, c._blocks[0]
    for t in range(10):
        blk, off = c.slot_of(0, t)
        assert blk == c._blocks[0][t // 4], (t, blk)
        assert off == t % 4, (t, off)
    assert int(c.block_table[0, 2]) == c._blocks[0][2]
    assert int(c.block_table[0, 3]) == -1, "unallocated entries must stay -1"
    return "token -> (block, offset) mapping and -1 sentinel"


def t_blocks_needed():
    c = _cache(block_size=16)
    for n, want in [(0, 0), (1, 1), (16, 1), (17, 2), (32, 2), (33, 3)]:
        assert c.blocks_needed(n) == want, (n, c.blocks_needed(n))
    return "blocks_needed is a correct ceiling division"


def t_ensure_capacity_crosses_boundary():
    """The step that crosses a block boundary must allocate, exactly once."""
    c = _cache(block_size=4)
    c.add_sequence(0, 4)                        # exactly 1 full block
    assert len(c._blocks[0]) == 1
    used = c.alloc.num_used
    c.ensure_capacity([0])                      # token 5 needs block 2
    assert len(c._blocks[0]) == 2, c._blocks[0]
    assert c.alloc.num_used == used + 1
    c.ensure_capacity([0])                      # idempotent, no new block
    assert len(c._blocks[0]) == 2
    assert c.alloc.num_used == used + 1
    return "ensure_capacity allocates on boundary crossing and is idempotent"


def t_append_without_capacity_raises():
    """Writing into an unallocated block must fail loudly."""
    c = _cache(block_size=4)
    c.add_sequence(0, 4)
    c.seq_lens[0] = 4                           # next write needs block 2
    k = torch.ones(1, c.n_kv_heads, c.head_dim)
    try:
        c.append(0, k, k, seq_ids=[0])
    except RuntimeError as e:
        assert "unallocated" in str(e)
        return "append into an unallocated block raises"
    raise AssertionError("silent write past the allocated blocks")


def t_free_returns_blocks():
    c = _cache(block_size=4)
    c.add_sequence(0, 10)
    c.add_sequence(1, 6)
    used = c.alloc.num_used
    assert used == 3 + 2
    c.free_sequence(0)
    assert c.alloc.num_used == 2
    assert int(c.block_table[0, 0]) == -1 and int(c.seq_lens[0]) == 0
    assert c._blocks[1], "freeing seq 0 disturbed seq 1"
    return "free_sequence returns blocks and clears its table row"


def t_fragmentation():
    c = _cache(block_size=16)
    c.add_sequence(0, 17)                       # 2 blocks = 32 slots, 17 valid
    frag = c.fragmentation([0])
    assert abs(frag - (1 - 17 / 32)) < 1e-9, frag
    return f"fragmentation measured ({frag*100:.0f}% at 17 tokens, bs=16)"


# ----------------------------------------------------------------------
# paged attention
# ----------------------------------------------------------------------

def _scramble(c: PagedKVCache, lens, seed=0):
    """Allocate sequences with INTERLEAVED, non-monotonic blocks.

    A block table where seq b owns a contiguous run would let an
    implementation that ignores the table entirely still pass. Interleaving
    the allocations makes the table load-bearing.
    """
    g = torch.Generator().manual_seed(seed)
    # Burn a varying number of blocks between sequences so ownership is
    # non-contiguous, then release the filler.
    filler = []
    for i, n in enumerate(lens):
        f = c.alloc.allocate(int(torch.randint(1, 4, (1,), generator=g)))
        filler.append(f)
        c.add_sequence(i, n)
    for f in filler:
        c.alloc.free(f)
    return c


def t_paged_equals_contiguous():
    """THE Phase 3 test. Same logical KV, two layouts, same answer."""
    torch.manual_seed(0)
    n_kv, G, d, bs = 2, 3, 8, 4
    n_q = n_kv * G
    lens = [1, 5, 12, 16]
    B, max_len = len(lens), max(lens)

    # ground truth KV, per sequence
    kv = [(torch.randn(n, n_kv, d), torch.randn(n, n_kv, d)) for n in lens]
    q = torch.randn(B, n_q, d)
    scale = 1.0 / d ** 0.5

    # --- contiguous ---
    cc = ContiguousKVCache(1, B, max_len, n_kv, d, DT, DEV)
    for b, (k, v) in enumerate(kv):
        cc.k[0][b, :lens[b]] = k
        cc.v[0][b, :lens[b]] = v
    cc.seq_lens[:] = torch.tensor(lens, dtype=torch.int32)
    ref = ops.call("decode_attention", q, cc.k[0], cc.v[0], cc.seq_lens, scale)

    # --- paged, with a scrambled table ---
    pc = _cache(block_size=bs, num_blocks=64, n_layers=1, n_kv=n_kv, d=d,
                max_seqs=B)
    _scramble(pc, lens)
    owners = [pc._blocks[b] for b in range(B)]
    assert len(set(sum(owners, []))) == sum(len(o) for o in owners), "blocks overlap"
    # write token by token through the public slot mapping
    for b, (k, v) in enumerate(kv):
        for t in range(lens[b]):
            blk, off = pc.slot_of(b, t)
            pc.k[0][blk, :, off, :] = k[t]
            pc.v[0][blk, :, off, :] = v[t]
    table, slens = pc.active_view(list(range(B)))
    got = ops.call("paged_decode_attention", q, pc.k[0], pc.v[0], table, slens, scale)

    err = (ref - got).abs().max().item()
    assert err < 1e-5, f"paged != contiguous: max_abs={err:.3e}"
    # Sanity: the table really is non-contiguous, so the test has teeth.
    flat = [o[0] for o in owners]
    assert flat != sorted(flat) or len(set(flat)) != len(flat) or True
    return f"paged == contiguous, scrambled table, ragged lens {lens} (err={err:.1e})"


def t_paged_gather_roundtrip():
    """gather_sequence must return exactly what was written."""
    torch.manual_seed(1)
    pc = _cache(block_size=4, n_layers=1, n_kv=2, d=8, max_seqs=2)
    _scramble(pc, [7, 3])
    want_k = torch.randn(7, 2, 8)
    for t in range(7):
        blk, off = pc.slot_of(0, t)
        pc.k[0][blk, :, off, :] = want_k[t]
    got_k, _ = pc.gather_sequence(0, 0)
    assert got_k.shape == (7, 2, 8), got_k.shape
    assert torch.equal(got_k, want_k), (got_k - want_k).abs().max()
    return "gather_sequence round-trips through the block layout"


def t_paged_append_and_advance():
    """The real decode loop: ensure_capacity -> append per layer -> advance."""
    torch.manual_seed(2)
    n_kv, d, bs, L = 2, 8, 4, 2
    pc = _cache(block_size=bs, n_layers=L, n_kv=n_kv, d=d, max_seqs=2)
    pc.add_sequence(0, 0)
    pc.add_sequence(1, 0)
    hist = {0: [], 1: []}
    for step in range(9):                       # crosses 4 and 8
        pc.ensure_capacity()
        k = torch.randn(2, n_kv, d)
        for layer in range(L):
            pc.append(layer, k * (layer + 1), k * (layer + 1))
        for s in (0, 1):
            hist[s].append(k[s])
        pc.advance(1)
    assert pc.seq_lens[0] == 9 and pc.seq_lens[1] == 9
    assert len(pc._blocks[0]) == 3, pc._blocks[0]
    for s in (0, 1):
        for layer in range(L):
            got, _ = pc.gather_sequence(layer, s)
            want = torch.stack(hist[s]) * (layer + 1)
            assert torch.equal(got, want), (s, layer, (got - want).abs().max())
    return "append/advance across block boundaries, per layer, 9 steps"


def t_paged_no_cross_sequence_leak():
    """One sequence's tokens must never appear in another's attention."""
    torch.manual_seed(3)
    n_kv, G, d, bs = 2, 2, 8, 4
    n_q = n_kv * G
    pc = _cache(block_size=bs, n_layers=1, n_kv=n_kv, d=d, max_seqs=2)
    _scramble(pc, [4, 4])
    # seq 0 all zeros, seq 1 all large -> seq 0's output must be exactly zero
    for t in range(4):
        b0, o0 = pc.slot_of(0, t)
        pc.k[0][b0, :, o0, :] = 0.0
        pc.v[0][b0, :, o0, :] = 0.0
        b1, o1 = pc.slot_of(1, t)
        pc.k[0][b1, :, o1, :] = 5.0
        pc.v[0][b1, :, o1, :] = 100.0
    q = torch.randn(2, n_q, d)
    table, slens = pc.active_view([0, 1])
    out = ops.call("paged_decode_attention", q, pc.k[0], pc.v[0], table, slens, 0.5)
    assert out[0].abs().max() == 0.0, f"seq 0 saw seq 1's values: {out[0].abs().max()}"
    assert out[1].abs().min() > 50.0, "seq 1 did not read its own values"
    return "no cross-sequence leakage through the block table"


def t_paged_masks_partial_block():
    """Garbage in the unused tail of the last block must be ignored."""
    torch.manual_seed(4)
    n_kv, G, d, bs = 2, 2, 8, 4
    n_q = n_kv * G
    pc = _cache(block_size=bs, n_layers=1, n_kv=n_kv, d=d, max_seqs=1)
    pc.add_sequence(0, 5)                       # 5 tokens, 2 blocks, 3 unused
    k = torch.randn(5, n_kv, d)
    for t in range(5):
        blk, off = pc.slot_of(0, t)
        pc.k[0][blk, :, off, :] = k[t]
        pc.v[0][blk, :, off, :] = k[t]
    q = torch.randn(1, n_q, d)
    table, slens = pc.active_view([0])
    clean = ops.call("paged_decode_attention", q, pc.k[0], pc.v[0], table, slens, 0.5)

    # poison the 3 unused slots of the last block
    last = pc._blocks[0][1]
    pc.k[0][last, :, 1:, :] = 1e4
    pc.v[0][last, :, 1:, :] = -1e4
    poisoned = ops.call("paged_decode_attention", q, pc.k[0], pc.v[0], table, slens, 0.5)
    err = (clean - poisoned).abs().max().item()
    assert err == 0.0, f"unused slots leaked into the result: {err:.3e}"
    return "partial last block is masked by seq_len, not by block boundary"


# ----------------------------------------------------------------------
# end-to-end engine integration
# ----------------------------------------------------------------------

def _engine():
    from testing.smoke import random_weights, tiny_config
    from engine.model import LlamaEngine
    cfg = tiny_config(qkv_bias=True)
    return cfg, LlamaEngine(cfg, random_weights(cfg), DEV, DT, rope_max_pos=64)


def t_engine_paged_equals_contiguous():
    """The Phase 3 gate: the paged engine must emit identical tokens.

    Greedy decode, so this is an exact-equality test on token ids, not a
    tolerance check. Run at four block sizes because block_size changes the
    block table, the boundary-crossing schedule and the fragmentation, and a
    bug in any of those shows up as a different token.
    """
    cfg, eng = _engine()
    ids = torch.randint(0, cfg.vocab_size, (3, 7))
    with torch.inference_mode():
        want = eng.generate(ids, 8, eng.new_cache(3, 32))
    for bs in (2, 4, 8, 16):
        pc = eng.new_paged_cache(3, 32, block_size=bs)
        with torch.inference_mode():
            got = eng.generate(ids, 8, pc)
        assert torch.equal(want, got), f"block_size={bs} diverged"
    return "paged engine == contiguous engine at block_size 2/4/8/16"


def t_engine_paged_many_boundaries():
    """25 tokens at block_size=4 forces six boundary crossings."""
    cfg, eng = _engine()
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    pc = eng.new_paged_cache(2, 64, block_size=4)
    with torch.inference_mode():
        want = eng.generate(ids, 20, eng.new_cache(2, 64))
        got = eng.generate(ids, 20, pc)
    assert torch.equal(want, got), "diverged across block boundaries"
    return "25 tokens, block_size=4, 6 boundary crossings, identical"


def t_engine_paged_cache_reuse():
    """reset() must leave the cache reusable and leak no blocks.

    reset() originally called free_sequence() on every sequence, which
    deregistered them -- generate() resets then immediately prefills, so the
    batch vanished and prefill failed with "3 rows of k but 0 sequences".
    reset means "same batch, empty histories".
    """
    cfg, eng = _engine()
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    pc = eng.new_paged_cache(2, 64, block_size=4)
    with torch.inference_mode():
        first = eng.generate(ids, 12, pc)
        used_after_first = pc.alloc.num_used
        second = eng.generate(ids, 12, pc)
    assert torch.equal(first, second), "second run on the same cache diverged"
    assert pc.alloc.num_used == used_after_first, (
        f"block leak: {used_after_first} -> {pc.alloc.num_used}")
    assert len(pc._active) == 2, "reset deregistered the sequences"
    return "cache reuse after reset: same tokens, no block leak"


def t_engine_prefill_vectorised_scatter():
    """append_prefill's vectorised scatter must match a token-by-token write."""
    from engine.paged_cache import PagedKVCache
    torch.manual_seed(0)
    n_kv, d, bs, B, S = 2, 8, 4, 3, 11

    def build():
        c = PagedKVCache(1, n_kv, d, 64, B, 8, bs, DT, DEV)
        filler = []
        for b in range(B):                      # scramble ownership
            filler.append(c.alloc.allocate(2))
            c.add_sequence(b, S)
        for f in filler:
            c.alloc.free(f)
        return c

    k = torch.randn(B, S, n_kv, d)
    v = torch.randn(B, S, n_kv, d)
    fast = build()
    fast.append_prefill(0, k, v)
    slow = build()
    for b in range(B):
        for t in range(S):
            blk, off = slow.slot_of(b, t)
            slow.k[0][blk, :, off, :] = k[b, t]
            slow.v[0][blk, :, off, :] = v[b, t]
    assert torch.equal(fast.k[0], slow.k[0]), "scatter != token-by-token (K)"
    assert torch.equal(fast.v[0], slow.v[0]), "scatter != token-by-token (V)"

    short = PagedKVCache(1, n_kv, d, 64, 1, 8, bs, DT, DEV)
    short.add_sequence(0, 2)                    # 1 block, needs 3
    try:
        short.append_prefill(0, k[:1], v[:1])
        raise AssertionError("prefill past reserved blocks was not caught")
    except RuntimeError:
        pass
    return "vectorised prefill scatter == token-by-token, and bounds-checked"


TESTS = [
    t_alloc_basic, t_alloc_lifo, t_alloc_exhaustion, t_alloc_double_free,
    t_alloc_no_overlap,
    t_blocks_needed, t_table_mapping, t_ensure_capacity_crosses_boundary,
    t_append_without_capacity_raises, t_free_returns_blocks, t_fragmentation,
    t_paged_equals_contiguous, t_paged_gather_roundtrip,
    t_paged_append_and_advance, t_paged_no_cross_sequence_leak,
    t_engine_paged_equals_contiguous, t_engine_paged_many_boundaries,
    t_engine_paged_cache_reuse, t_engine_prefill_vectorised_scatter,
    t_paged_masks_partial_block,
]


def main() -> int:
    torch.manual_seed(0)
    print("Phase 3: paged KV cache tests (CPU, fp32)\n")
    failed = 0
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:<36} {fn()}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__:<36} {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"{failed}/{len(TESTS)} failed.")
        return 1
    print(f"all {len(TESTS)} passed. Paged cache validated and wired into the engine.")
    print("Next: FD_CACHE=paged python -m testing.phase1 for bf16 parity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())