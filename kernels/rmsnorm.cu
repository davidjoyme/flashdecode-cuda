#include <cuda_bf16.h>
#include <cuda_runtime.h>

using bf16 = __nv_bfloat16;

static constexpr int WARP = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Two-level reduction: shuffle within each warp, then warp 0 reduces the
// per-warp partials. Shared memory holds only NWARPS floats (24 bytes at
// THREADS=192), so it never limits occupancy.
template <int THREADS>
__device__ __forceinline__ float block_reduce_sum(float v, float *smem) {
    constexpr int NWARPS = THREADS / WARP;
    static_assert(THREADS % WARP == 0, "THREADS must be a whole number of warps");
    static_assert(NWARPS <= WARP, "cross-warp reduction needs NWARPS <= 32");

    const int lane = threadIdx.x % WARP;
    const int wid = threadIdx.x / WARP;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    v = (threadIdx.x < NWARPS) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// THREADS * 8 == H. ITEMS=8 bf16 = 16 bytes = one 128-bit load, the widest
// the memory pipe accepts.
template <int THREADS>
__global__ void rmsnorm_residual_kernel(const bf16 *__restrict__ x,
                                        const bf16 *__restrict__ residual,
                                        const bf16 *__restrict__ weight,
                                        bf16 *__restrict__ out,
                                        bf16 *__restrict__ new_residual,
                                        const int H, const float eps) {
    constexpr int ITEMS = 8;

    const size_t base = (size_t)blockIdx.x * H;
    const int idx = threadIdx.x * ITEMS;
    __shared__ float smem[THREADS / WARP];

    // h kept in registers across both passes so the normalize pass does not
    // re-read it from memory. 8 floats/thread -- cheap, no spills expected.
    float h[ITEMS];

    // ---- pass 1: load, add, square-accumulate, store new residual ----
    const uint4 xv = *reinterpret_cast<const uint4 *>(x + base + idx);
    const uint4 rv = *reinterpret_cast<const uint4 *>(residual + base + idx);
    const bf16 *xb = reinterpret_cast<const bf16 *>(&xv);
    const bf16 *rb = reinterpret_cast<const bf16 *>(&rv);

    float ss = 0.0f;
    bf16 hb[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const float t = __bfloat162float(xb[i]) + __bfloat162float(rb[i]);
        // CRITICAL: round to bf16 BEFORE squaring. The reference adds two
        // bf16 tensors (producing a bf16 result) and only then computes the
        // variance. Accumulating the unrounded fp32 sum instead changes ~1%
        // of output elements by up to 6.25e-02 -- three times DIFF_ATOL, and
        // exactly the kind of drift that shows up as a persistent nonzero
        // diff with cosine ~1.0.
        const bf16 hbf = __float2bfloat16(t);
        const float tr = __bfloat162float(hbf);
        h[i] = tr;
        ss += tr * tr;
        hb[i] = hbf;
    }
    *reinterpret_cast<uint4 *>(new_residual + base + idx) =
        *reinterpret_cast<const uint4 *>(hb);

    ss = block_reduce_sum<THREADS>(ss, smem);
    const float rstd = rsqrtf(ss / (float)H + eps);

    // ---- pass 2: normalize, cast, then weight ----
    const uint4 wv = *reinterpret_cast<const uint4 *>(weight + idx);
    const bf16 *wb = reinterpret_cast<const bf16 *>(&wv);

    bf16 ob[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        // Cast to bf16 FIRST, then multiply by the weight. HF does
        // `weight * x32.to(dtype)`, not `(weight * x32).to(dtype)`. The two
        // differ in the last mantissa bit, and the fp32 CPU smoke test
        // provably cannot catch the difference.
        const bf16 n = __float2bfloat16(h[i] * rstd);
        ob[i] = __hmul(wb[i], n);
    }
    *reinterpret_cast<uint4 *>(out + base + idx) =
        *reinterpret_cast<const uint4 *>(ob);
}

// ---------------------------------------------------------------------
// Launcher. Returns true if a kernel was launched, false if H is unsupported.
// The caller MUST check: a silent no-op would leave `out` uninitialised and
// present as a numerics bug rather than a configuration one.
// ---------------------------------------------------------------------
#define FD_CASE(T)                                                            \
    case T:                                                                    \
        rmsnorm_residual_kernel<T><<<B, T, 0, stream>>>(                        \
            x, residual, weight, out, new_residual, H, eps);                   \
        return true;

bool rmsnorm_residual_launch(const bf16 *x, const bf16 *residual,
                             const bf16 *weight, bf16 *out, bf16 *new_residual,
                             int B, int H, float eps, cudaStream_t stream) {
    const int threads = H / 8;
    if (H % 8 != 0 || threads % WARP != 0 || threads < 64 || threads > 1024)
        return false;

    switch (threads) {
        FD_CASE(64)    // H=512
        FD_CASE(128)   // H=1024
        FD_CASE(192)   // H=1536  Qwen2.5-1.5B
        FD_CASE(256)   // H=2048  Qwen2.5-3B, Llama-3.2-1B
        FD_CASE(320)   // H=2560
        FD_CASE(384)   // H=3072
        FD_CASE(448)   // H=3584  Qwen2.5-7B
        FD_CASE(512)   // H=4096  Llama-3-8B
        FD_CASE(640)   // H=5120
        FD_CASE(768)   // H=6144
        FD_CASE(896)   // H=7168
        FD_CASE(1024)  // H=8192
    default:
        return false;
    }
}
#undef FD_CASE