#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

using bf16 = __nv_bfloat16;

static constexpr int WARP = 32;

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// --------------------------------------------------------------------------
// GROUP   = query heads per KV head (6 for Qwen2.5-1.5B)
// HEAD_DIM= 128
// THREADS = 256
//
// Thread mapping: one WARP per query head in the group, so GROUP warps are
// busy and the rest idle. Crude -- 6 of 8 warps used at GROUP=6 -- and
// deliberately so: it keeps v0 obviously correct, and the waste is a measured
// baseline for v1 rather than a guess.
// --------------------------------------------------------------------------
template <int GROUP, int HEAD_DIM, int BLOCK_SIZE, int THREADS>
__global__ void paged_decode_attention_kernel(
        const bf16 *__restrict__ q,            // [B, n_q_heads, HEAD_DIM]
        const bf16 *__restrict__ k_cache,      // [num_blocks, n_kv, BLOCK_SIZE, HEAD_DIM]
        const bf16 *__restrict__ v_cache,
        const int *__restrict__ block_table,   // [B, max_blocks]
        const int *__restrict__ seq_lens,      // [B]
        bf16 *__restrict__ out,                // [B, n_q_heads, HEAD_DIM]
        const int n_kv_heads,
        const int max_blocks,
        const float scale) {

    const int kv_head = blockIdx.x;
    const int seq = blockIdx.y;
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;

    const int seq_len = seq_lens[seq];
    const int n_q_heads = n_kv_heads * GROUP;

    // Warps beyond the group have no query head to own. They must still reach
    // every __syncthreads() below, so they fall through the loop doing no
    // work rather than returning early -- an early return here would hang the
    // block on the first barrier.
    const bool active = (warp < GROUP);
    const int q_head = kv_head * GROUP + warp;

    // q for this warp's head, in registers. HEAD_DIM/WARP = 4 floats/lane.
    constexpr int DPL = HEAD_DIM / WARP;          // dims per lane
    float qreg[DPL];
    if (active) {
        const bf16 *qp = q + ((size_t)seq * n_q_heads + q_head) * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < DPL; ++i)
            qreg[i] = __bfloat162float(qp[lane * DPL + i]) * scale;
    }

    // Online-softmax state, per warp (i.e. per query head).
    float m_i = -CUDART_INF_F;                    // running max
    float l_i = 0.0f;                             // running sum of exp
    float acc[DPL];                               // running weighted V
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.0f;

    // The table row is read once per block, not once per token.
    const int *table = block_table + (size_t)seq * max_blocks;
    const int n_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int blk = 0; blk < n_blocks; ++blk) {
        const int phys = table[blk];
        // A negative entry means the host under-reserved. Bail rather than
        // read arbitrary memory; the Python side raises on the same condition.
        if (phys < 0) break;

        const bf16 *kb = k_cache
            + (((size_t)phys * n_kv_heads + kv_head) * BLOCK_SIZE) * HEAD_DIM;
        const bf16 *vb = v_cache
            + (((size_t)phys * n_kv_heads + kv_head) * BLOCK_SIZE) * HEAD_DIM;

        const int tok_base = blk * BLOCK_SIZE;
        const int n_tok = min(BLOCK_SIZE, seq_len - tok_base);

        for (int t = 0; t < n_tok; ++t) {
            if (!active) continue;

            // score = q . k  -- each lane holds DPL dims, warp reduces.
            const bf16 *kt = kb + (size_t)t * HEAD_DIM;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < DPL; ++i)
                dot += qreg[i] * __bfloat162float(kt[lane * DPL + i]);
            const float s = warp_reduce_sum(dot);   // every lane gets it

            // --- online softmax update ---
            // Standard FlashAttention recurrence. Rescaling the accumulator
            // when the max moves is what makes a single pass numerically
            // equivalent to max-then-exp, and it is why no scratch buffer of
            // scores is needed.
            const float m_new = fmaxf(m_i, s);
            const float corr = __expf(m_i - m_new);  // 0 on the first token
            const float p = __expf(s - m_new);

            const bf16 *vt = vb + (size_t)t * HEAD_DIM;
#pragma unroll
            for (int i = 0; i < DPL; ++i)
                acc[i] = acc[i] * corr + p * __bfloat162float(vt[lane * DPL + i]);

            l_i = l_i * corr + p;
            m_i = m_new;
        }
    }

    if (active && seq_len > 0) {
        const float inv = 1.0f / l_i;
        bf16 *op = out + ((size_t)seq * n_q_heads + q_head) * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < DPL; ++i)
            op[lane * DPL + i] = __float2bfloat16(acc[i] * inv);
    } else if (active) {
        // seq_len == 0: define the output as zero rather than NaN from 1/0.
        bf16 *op = out + ((size_t)seq * n_q_heads + q_head) * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < DPL; ++i)
            op[lane * DPL + i] = __float2bfloat16(0.0f);
    }
}


template <int GROUP, int HEAD_DIM, int BLOCK_SIZE, int THREADS>
__global__ void paged_decode_attention_v1_kernel(
        const bf16 *__restrict__ q,
        const bf16 *__restrict__ k_cache,
        const bf16 *__restrict__ v_cache,
        const int *__restrict__ block_table,
        const int *__restrict__ seq_lens,
        bf16 *__restrict__ out,
        const int n_kv_heads,
        const int max_blocks,
        const float scale) {

    constexpr int VEC = 8;                    // bf16 per thread = 16 B
    constexpr int TG = HEAD_DIM / VEC;        // threads per token
    constexpr int TOKS = WARP / TG;           // tokens per warp, concurrently
    static_assert(TG > 0 && TG <= WARP, "bad thread-group size");
    static_assert(WARP % TG == 0, "TG must divide the warp");

    const int kv_head = blockIdx.x;
    const int seq = blockIdx.y;
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;
    const int tg = lane / TG;                 // which token slot: 0..TOKS-1
    const int tgl = lane % TG;                // which VEC dims: 0..TG-1

    const int seq_len = seq_lens[seq];
    const int n_q_heads = n_kv_heads * GROUP;
    const bool active = (warp < GROUP);
    const int q_head = kv_head * GROUP + warp;

    // q fragment: this thread's VEC dims, pre-scaled.
    float qf[VEC];
    if (active) {
        const bf16 *qp = q + ((size_t)seq * n_q_heads + q_head) * HEAD_DIM
                         + tgl * VEC;
        const uint4 raw = *reinterpret_cast<const uint4 *>(qp);
        const bf16 *qb = reinterpret_cast<const bf16 *>(&raw);
#pragma unroll
        for (int i = 0; i < VEC; ++i) qf[i] = __bfloat162float(qb[i]) * scale;
    }

    float m_i = -CUDART_INF_F, l_i = 0.0f;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

    const int *table = block_table + (size_t)seq * max_blocks;
    const int n_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int blk = 0; blk < n_blocks; ++blk) {
        const int phys = table[blk];
        if (phys < 0) break;
        const size_t base =
            (((size_t)phys * n_kv_heads + kv_head) * BLOCK_SIZE) * HEAD_DIM;
        const bf16 *kb = k_cache + base;
        const bf16 *vb = v_cache + base;
        const int n_tok = min(BLOCK_SIZE, seq_len - blk * BLOCK_SIZE);

        // Each warp advances TOKS tokens at a time; thread group `tg` owns
        // token t0 + tg.
        for (int t0 = 0; t0 < n_tok; t0 += TOKS) {
            const int tok = t0 + tg;
            if (!active || tok >= n_tok) continue;

            const uint4 kraw = *reinterpret_cast<const uint4 *>(
                kb + (size_t)tok * HEAD_DIM + tgl * VEC);
            const bf16 *kf = reinterpret_cast<const bf16 *>(&kraw);

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < VEC; ++i)
                dot += qf[i] * __bfloat162float(kf[i]);

            // Reduce across the TG lanes of THIS group only. xor offsets
            // below TG never cross a group boundary, so no mask surgery is
            // needed and the two groups reduce independently.
#pragma unroll
            for (int off = TG / 2; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, off);
            const float s = dot;

            const float m_new = fmaxf(m_i, s);
            const float corr = __expf(m_i - m_new);
            const float p = __expf(s - m_new);

            const uint4 vraw = *reinterpret_cast<const uint4 *>(
                vb + (size_t)tok * HEAD_DIM + tgl * VEC);
            const bf16 *vf = reinterpret_cast<const bf16 *>(&vraw);
#pragma unroll
            for (int i = 0; i < VEC; ++i)
                acc[i] = acc[i] * corr + p * __bfloat162float(vf[i]);

            l_i = l_i * corr + p;
            m_i = m_new;
        }
    }

    // ---- merge the TOKS partials held by this warp ----
    // Group g lives at lanes [g*TG, (g+1)*TG). Lane L and lane L^(TG*step)
    // own the SAME dims for DIFFERENT tokens, so an xor shuffle by TG pairs
    // them up. After log2(TOKS) rounds every lane holds the full result.
#pragma unroll
    for (int step = TG; step < WARP; step <<= 1) {
        const float m_o = __shfl_xor_sync(0xffffffffu, m_i, step);
        const float l_o = __shfl_xor_sync(0xffffffffu, l_i, step);
        const float m_new = fmaxf(m_i, m_o);
        // BOTH partials can be empty -- in v2 a split beyond the last table
        // block gets no work at all. Then m_i == m_o == -inf, m_new == -inf,
        // and m_i - m_new is (-inf) - (-inf) == NaN, so __expf returns NaN
        // and that NaN propagates into l and acc. The combine pass cannot
        // recover either: its isfinite(m) guard gives c = 0, but NaN * 0 is
        // still NaN. So test isfinite(m_new) explicitly rather than trusting
        // the arithmetic to degrade gracefully.
        //
        // Found by sweeping the split count in the correctness check: v2 was
        // exact at splits 1/2/4/8 and rel 1.0 at 16, because ctx=128 with
        // block_size=16 gives 8 table blocks, so splits 8..15 get no work.
        const bool any_work = isfinite(m_new);
        const float c_s = any_work ? __expf(m_i - m_new) : 0.0f;
        const float c_o = any_work ? __expf(m_o - m_new) : 0.0f;
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const float a_o = __shfl_xor_sync(0xffffffffu, acc[i], step);
            acc[i] = acc[i] * c_s + a_o * c_o;
        }
        l_i = l_i * c_s + l_o * c_o;
        m_i = m_new;
    }

    // Only the first thread group writes; all groups now hold the same value.
    if (active && tg == 0) {
        bf16 *op = out + ((size_t)seq * n_q_heads + q_head) * HEAD_DIM
                   + tgl * VEC;
        bf16 packed[VEC];
        const float inv = (seq_len > 0 && l_i > 0.0f) ? 1.0f / l_i : 0.0f;
#pragma unroll
        for (int i = 0; i < VEC; ++i)
            packed[i] = __float2bfloat16(acc[i] * inv);
        *reinterpret_cast<uint4 *>(op) =
            *reinterpret_cast<const uint4 *>(packed);
    }
}

template <int GROUP, int HEAD_DIM, int BLOCK_SIZE, int THREADS>
__global__ void paged_attn_v2_partial(
        const bf16 *__restrict__ q,
        const bf16 *__restrict__ k_cache,
        const bf16 *__restrict__ v_cache,
        const int *__restrict__ block_table,
        const int *__restrict__ seq_lens,
        float *__restrict__ m_part,      // [B, n_q_heads, n_splits]
        float *__restrict__ l_part,      // [B, n_q_heads, n_splits]
        float *__restrict__ acc_part,    // [B, n_q_heads, n_splits, HEAD_DIM]
        const int n_kv_heads,
        const int max_blocks,
        const int n_splits,
        const float scale) {

    constexpr int VEC = 8;
    constexpr int TG = HEAD_DIM / VEC;
    constexpr int TOKS = WARP / TG;

    const int kv_head = blockIdx.x;
    const int seq = blockIdx.y;
    const int split = blockIdx.z;
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;
    const int tg = lane / TG;
    const int tgl = lane % TG;

    const int seq_len = seq_lens[seq];
    const int n_q_heads = n_kv_heads * GROUP;
    const bool active = (warp < GROUP);
    const int q_head = kv_head * GROUP + warp;

    const int n_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Split in units of BLOCKS, not tokens, so block-table indexing stays
    // integral and no split straddles a table entry.
    const int bps = (n_blocks + n_splits - 1) / n_splits;
    const int blk_begin = split * bps;
    const int blk_end = min(n_blocks, blk_begin + bps);

    float qf[VEC];
    if (active) {
        const bf16 *qp = q + ((size_t)seq * n_q_heads + q_head) * HEAD_DIM
                         + tgl * VEC;
        const uint4 raw = *reinterpret_cast<const uint4 *>(qp);
        const bf16 *qb = reinterpret_cast<const bf16 *>(&raw);
#pragma unroll
        for (int i = 0; i < VEC; ++i) qf[i] = __bfloat162float(qb[i]) * scale;
    }

    float m_i = -CUDART_INF_F, l_i = 0.0f;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

    const int *table = block_table + (size_t)seq * max_blocks;

    for (int blk = blk_begin; blk < blk_end; ++blk) {
        const int phys = table[blk];
        if (phys < 0) break;
        const size_t base =
            (((size_t)phys * n_kv_heads + kv_head) * BLOCK_SIZE) * HEAD_DIM;
        const bf16 *kb = k_cache + base;
        const bf16 *vb = v_cache + base;
        const int n_tok = min(BLOCK_SIZE, seq_len - blk * BLOCK_SIZE);

        for (int t0 = 0; t0 < n_tok; t0 += TOKS) {
            const int tok = t0 + tg;
            if (!active || tok >= n_tok) continue;

            const uint4 kraw = *reinterpret_cast<const uint4 *>(
                kb + (size_t)tok * HEAD_DIM + tgl * VEC);
            const bf16 *kf = reinterpret_cast<const bf16 *>(&kraw);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < VEC; ++i)
                dot += qf[i] * __bfloat162float(kf[i]);
#pragma unroll
            for (int off = TG / 2; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, off);

            const float m_new = fmaxf(m_i, dot);
            const float corr = __expf(m_i - m_new);
            const float p = __expf(dot - m_new);

            const uint4 vraw = *reinterpret_cast<const uint4 *>(
                vb + (size_t)tok * HEAD_DIM + tgl * VEC);
            const bf16 *vf = reinterpret_cast<const bf16 *>(&vraw);
#pragma unroll
            for (int i = 0; i < VEC; ++i)
                acc[i] = acc[i] * corr + p * __bfloat162float(vf[i]);
            l_i = l_i * corr + p;
            m_i = m_new;
        }
    }

    // merge the TOKS thread-group partials, exactly as v1 does
#pragma unroll
    for (int step = TG; step < WARP; step <<= 1) {
        const float m_o = __shfl_xor_sync(0xffffffffu, m_i, step);
        const float l_o = __shfl_xor_sync(0xffffffffu, l_i, step);
        const float m_new = fmaxf(m_i, m_o);
        // BOTH partials can be empty -- in v2 a split beyond the last table
        // block gets no work at all. Then m_i == m_o == -inf, m_new == -inf,
        // and m_i - m_new is (-inf) - (-inf) == NaN, so __expf returns NaN
        // and that NaN propagates into l and acc. The combine pass cannot
        // recover either: its isfinite(m) guard gives c = 0, but NaN * 0 is
        // still NaN. So test isfinite(m_new) explicitly rather than trusting
        // the arithmetic to degrade gracefully.
        //
        // Found by sweeping the split count in the correctness check: v2 was
        // exact at splits 1/2/4/8 and rel 1.0 at 16, because ctx=128 with
        // block_size=16 gives 8 table blocks, so splits 8..15 get no work.
        const bool any_work = isfinite(m_new);
        const float c_s = any_work ? __expf(m_i - m_new) : 0.0f;
        const float c_o = any_work ? __expf(m_o - m_new) : 0.0f;
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const float a_o = __shfl_xor_sync(0xffffffffu, acc[i], step);
            acc[i] = acc[i] * c_s + a_o * c_o;
        }
        l_i = l_i * c_s + l_o * c_o;
        m_i = m_new;
    }

    if (active && tg == 0) {
        const size_t pidx = ((size_t)seq * n_q_heads + q_head) * n_splits + split;
        float *ap = acc_part + pidx * HEAD_DIM + tgl * VEC;
#pragma unroll
        for (int i = 0; i < VEC; ++i) ap[i] = acc[i];
        if (tgl == 0) {
            m_part[pidx] = m_i;
            l_part[pidx] = l_i;
        }
    }
}

// One block per (sequence, query head); one thread per output dim.
// n_splits <= 32, so re-reading the m/l arrays per thread costs nothing --
// they are in L1 after the first access and the loop is trivially short.
template <int HEAD_DIM>
__global__ void paged_attn_v2_combine(
        const float *__restrict__ m_part,
        const float *__restrict__ l_part,
        const float *__restrict__ acc_part,
        bf16 *__restrict__ out,
        const int n_q_heads,
        const int n_splits) {

    const int q_head = blockIdx.x;
    const int seq = blockIdx.y;
    const int dim = threadIdx.x;
    if (dim >= HEAD_DIM) return;

    const size_t base = ((size_t)seq * n_q_heads + q_head) * n_splits;

    float m = -CUDART_INF_F;
    for (int s = 0; s < n_splits; ++s) m = fmaxf(m, m_part[base + s]);

    float l = 0.0f, a = 0.0f;
    if (isfinite(m)) {
        for (int s = 0; s < n_splits; ++s) {
            const float ms = m_part[base + s];
            // isfinite guard, not arithmetic: exp(-inf - -inf) is NaN, and a
            // split that received no blocks legitimately has m = -inf.
            if (!isfinite(ms)) continue;        // split that got no work
            const float c = __expf(ms - m);
            const float ls = l_part[base + s];
            const float as = acc_part[(base + s) * HEAD_DIM + dim];
            // Skip rather than weight by zero: NaN * 0 == NaN, so a poisoned
            // partial would survive a plain weighted sum.
            if (isfinite(ls) && isfinite(as)) {
                l += ls * c;
                a += as * c;
            }
        }
    }
    out[((size_t)seq * n_q_heads + q_head) * HEAD_DIM + dim] =
        __float2bfloat16((l > 0.0f) ? a / l : 0.0f);
}

#define FD_DISPATCH(KERN, G, D, BS)                                           \
    KERN<G, D, BS, 256><<<grid, 256, 0, stream>>>(                             \
        q, k_cache, v_cache, block_table, seq_lens, out, n_kv_heads,           \
        max_blocks, scale);                                                    \
    return true;

#define FD_DISPATCH_V2(G, D, BS)                                              \
    paged_attn_v2_partial<G, D, BS, 256><<<grid2, 256, 0, stream>>>(           \
        q, k_cache, v_cache, block_table, seq_lens, m_part, l_part, acc_part,  \
        n_kv_heads, max_blocks, n_splits, scale);                              \
    paged_attn_v2_combine<D><<<dim3(n_kv_heads *(G), batch), (D), 0, stream>>>(\
        m_part, l_part, acc_part, out, n_kv_heads *(G), n_splits);             \
    return true;

#define FD_BY_VERSION(G, D, BS)                                               \
    if (version == 0) { FD_DISPATCH(paged_decode_attention_kernel, G, D, BS) } \
    if (version == 1) { FD_DISPATCH(paged_decode_attention_v1_kernel, G, D, BS) } \
    if (version == 2) { FD_DISPATCH_V2(G, D, BS) }

// How many context splits to use for v2.
//
// Target enough blocks to fill the machine once: 170 SMs x 5 resident blocks
// (ptxas reports 48 registers at 256 threads) = 850. Capped at 32 so the
// scratch stays small and the combine loop stays short, and never more
// splits than there are table blocks, since an empty split is pure overhead.
int fd_choose_splits(int batch, int n_kv_heads, int n_blocks, int cap) {
    const int target = 850;
    int s = (target + n_kv_heads * batch - 1) / (n_kv_heads * batch);
    if (s < 1) s = 1;
    if (s > cap) s = cap;
    if (s > n_blocks) s = n_blocks > 0 ? n_blocks : 1;
    return s;
}

// version selects the kernel so all three stay built and benchmarkable side
// by side. Keeping v0 and v1 alive is the point: the writeup needs
// before/after from the same binary on the same data, not remembered numbers.
bool paged_decode_attention_launch(
        const bf16 *q, const bf16 *k_cache, const bf16 *v_cache,
        const int *block_table, const int *seq_lens, bf16 *out,
        int batch, int n_kv_heads, int group, int head_dim,
        int block_size, int max_blocks, float scale, int version,
        float *m_part, float *l_part, float *acc_part, int n_splits,
        cudaStream_t stream) {

    dim3 grid(n_kv_heads, batch);
    dim3 grid2(n_kv_heads, batch, n_splits);

    if (head_dim != 128) return false;
    if (version < 0 || version > 2) return false;
    if (version == 2 && (m_part == nullptr || n_splits < 1)) return false;

    if (group == 6 && block_size == 16) { FD_BY_VERSION(6, 128, 16) }
    if (group == 6 && block_size == 32) { FD_BY_VERSION(6, 128, 32) }
    if (group == 4 && block_size == 16) { FD_BY_VERSION(4, 128, 16) }
    if (group == 8 && block_size == 16) { FD_BY_VERSION(8, 128, 16) }
    if (group == 1 && block_size == 16) { FD_BY_VERSION(1, 128, 16) }
    if (group == 2 && block_size == 16) { FD_BY_VERSION(2, 128, 16) }
    if (group == 3 && block_size == 16) { FD_BY_VERSION(3, 128, 16) }
    return false;
}
#undef FD_BY_VERSION
#undef FD_DISPATCH_V2
#undef FD_DISPATCH