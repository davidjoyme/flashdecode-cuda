// ==========================================================================
// FILE LOCATION:  flashdecode/kernels/paged_attention.cu
// BUILD:          cd kernels && python setup.py build_ext --inplace
// ==========================================================================
//
// Paged GQA decode attention. Phase 4 v0 -- CORRECT, not yet optimised.
//
// out[b,h,:] = sum_t softmax_t( q[b,h,:] . K[b,t,:] * scale ) * V[b,t,:]
//
// where K/V for sequence b token t live at
//   block = block_table[b, t / BLOCK_SIZE]
//   slot  = t % BLOCK_SIZE
//   k_cache[block, kv_head, slot, :]
//
// -------------------------------------------------------------------------
// v0 DESIGN, and what is deliberately left for later
// -------------------------------------------------------------------------
// One block per (sequence, KV head). Each block handles all GROUP query heads
// that share that KV head, which is the point of GQA: the KV data is read
// ONCE from memory and reused across the group, instead of the reference
// implementation's repeat_interleave that materialises GROUP copies.
//
// Grid  = (n_kv_heads, batch)        -> 2 x B blocks for this model
// Block = 256 threads
//
// At B=128 that is 256 blocks on 170 SMs. Phase 2 measured 0.09 waves/SM for
// a B-sized grid and showed that is the whole problem at decode shapes;
// multiplying the grid by n_kv_heads is the first structural improvement, and
// v2 (split-KV) will multiply it again.
//
// Online softmax, one pass. The whole KV history is streamed once, keeping a
// running max and running sum per query head and rescaling the accumulator
// when the max moves. The alternative -- scores to scratch, max, exp, sum,
// weighted sum -- would read KV twice and need O(seq_len) storage. This is
// the same recurrence FlashAttention uses, applied to a single query row.
//
// NOT in v0, by design. Each gets its own measurement so the writeup can
// attribute the gains:
//   v1  vectorised 128-bit loads, shared-memory staging for q
//   v2  split the KV range across blocks + a combine pass (Flash-Decoding),
//       which is what rescues batch-1 and short-context cases
//   v3  more aggressive register blocking across the query group
//   v4  sm_120 specifics -- cp.async pipelining; NOT wgmma/tcgen05, which do
//       not exist on this architecture
//
// Accumulation is fp32 throughout. The cache is bf16 and is read as bf16;
// nothing is ever upcast in memory, which is the single biggest difference
// from the PyTorch reference -- that upcast cost 3.25 GB of transient at
// B=128 and was what OOM'd the Phase 1 baseline.

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
    constexpr int NWARPS = THREADS / WARP;

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

// --------------------------------------------------------------------------
// Launcher. Returns false if the shape is not covered, so the binding can
// raise instead of leaving `out` uninitialised -- a silent no-op would look
// like a numerics bug rather than a configuration one.
// --------------------------------------------------------------------------
#define FD_LAUNCH(G, D, BS)                                                   \
    paged_decode_attention_kernel<G, D, BS, 256>                              \
        <<<grid, 256, 0, stream>>>(q, k_cache, v_cache, block_table,          \
                                   seq_lens, out, n_kv_heads, max_blocks,     \
                                   scale);                                    \
    return true;

bool paged_decode_attention_launch(
        const bf16 *q, const bf16 *k_cache, const bf16 *v_cache,
        const int *block_table, const int *seq_lens, bf16 *out,
        int batch, int n_kv_heads, int group, int head_dim,
        int block_size, int max_blocks, float scale, cudaStream_t stream) {

    dim3 grid(n_kv_heads, batch);

    if (head_dim != 128) return false;            // only shape built so far
    if (group == 6  && block_size == 16) { FD_LAUNCH(6, 128, 16) }
    if (group == 6  && block_size == 32) { FD_LAUNCH(6, 128, 32) }
    if (group == 4  && block_size == 16) { FD_LAUNCH(4, 128, 16) }
    if (group == 8  && block_size == 16) { FD_LAUNCH(8, 128, 16) }
    if (group == 1  && block_size == 16) { FD_LAUNCH(1, 128, 16) }
    if (group == 2  && block_size == 16) { FD_LAUNCH(2, 128, 16) }
    if (group == 3  && block_size == 16) { FD_LAUNCH(3, 128, 16) }
    return false;
}
#undef FD_LAUNCH