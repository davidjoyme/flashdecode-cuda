// torch bindings for the Phase 2 kernels.
//
// Requires -std=c++20: torch >= 2.4 headers are C++20. See setup.py.

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <vector>

using bf16 = __nv_bfloat16;

bool rmsnorm_residual_launch(const bf16 *x, const bf16 *residual,
                             const bf16 *weight, bf16 *out, bf16 *new_residual,
                             int B, int H, float eps, cudaStream_t stream);

// NOT named CHECK. torch's c10/util/logging_is_not_google_glog.h already
// defines a one-argument CHECK(condition) macro, so defining a two-argument
// CHECK here shadowed it and produced a "CHECK redefined" warning -- and
// would silently break any torch header that used CHECK after this point.
// Project macros in a header-heavy build get a prefix.
#define FD_CHECK(cond, msg) TORCH_CHECK((cond), msg)

std::vector<torch::Tensor> rmsnorm_residual(torch::Tensor x,
                                            torch::Tensor residual,
                                            torch::Tensor weight, double eps) {
    FD_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda(),
             "all tensors must be on CUDA");
    FD_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    FD_CHECK(residual.scalar_type() == torch::kBFloat16, "residual must be bfloat16");
    FD_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");
    FD_CHECK(x.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
             "all tensors must be contiguous");
    FD_CHECK(x.sizes() == residual.sizes(), "x and residual must have the same shape");

    // Accept [B, H] (decode) or [B, S, H] (prefill); flatten leading dims.
    const int H = static_cast<int>(x.size(-1));
    const int B = static_cast<int>(x.numel() / H);
    FD_CHECK(weight.numel() == H, "weight length must equal the last dim of x");

    auto out = torch::empty_like(x);
    auto new_residual = torch::empty_like(x);

    // The launcher reports whether H tiles. Never let an unsupported H fall
    // through silently: `out` would stay uninitialised and the failure would
    // look like a numerics bug instead of a configuration one.
    const bool launched = rmsnorm_residual_launch(
        reinterpret_cast<const bf16 *>(x.data_ptr()),
        reinterpret_cast<const bf16 *>(residual.data_ptr()),
        reinterpret_cast<const bf16 *>(weight.data_ptr()),
        reinterpret_cast<bf16 *>(out.data_ptr()),
        reinterpret_cast<bf16 *>(new_residual.data_ptr()), B, H,
        static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(launched,
                "hidden_size ", H, " does not tile: needs H/8 to be a multiple "
                "of 32 in [64, 1024]. H=1536 (Qwen2.5-1.5B) and H=2048 work; "
                "H=896 (Qwen2.5-0.5B) gives 112 threads = 3.5 warps and would "
                "need a tail path.");

    // Surface launch errors here rather than at some unrelated later sync.
    C10_CUDA_CHECK(cudaGetLastError());
    return {out, new_residual};
}

// --------------------------------------------------------------------------
// paged GQA decode attention -- Phase 4
// --------------------------------------------------------------------------

bool paged_decode_attention_launch(
        const bf16 *q, const bf16 *k_cache, const bf16 *v_cache,
        const int *block_table, const int *seq_lens, bf16 *out,
        int batch, int n_kv_heads, int group, int head_dim,
        int block_size, int max_blocks, float scale, int version,
        float *m_part, float *l_part, float *acc_part, int n_splits,
        cudaStream_t stream);

int fd_choose_splits(int batch, int n_kv_heads, int n_blocks, int cap);

torch::Tensor paged_decode_attention(torch::Tensor q,
                                     torch::Tensor k_cache,
                                     torch::Tensor v_cache,
                                     torch::Tensor block_table,
                                     torch::Tensor seq_lens,
                                     double scale,
                                     int64_t version,
                                     int64_t splits) {
    FD_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
             "all tensors must be on CUDA");
    FD_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
    FD_CHECK(k_cache.scalar_type() == torch::kBFloat16, "k_cache must be bfloat16");
    FD_CHECK(v_cache.scalar_type() == torch::kBFloat16, "v_cache must be bfloat16");
    FD_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
    FD_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
    FD_CHECK(q.dim() == 3, "q must be [B, n_q_heads, head_dim]");
    FD_CHECK(k_cache.dim() == 4,
             "k_cache must be [num_blocks, n_kv_heads, block_size, head_dim]");
    FD_CHECK(q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous()
             && block_table.is_contiguous() && seq_lens.is_contiguous(),
             "all tensors must be contiguous");
    FD_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache shapes differ");

    const int batch = static_cast<int>(q.size(0));
    const int n_q_heads = static_cast<int>(q.size(1));
    const int head_dim = static_cast<int>(q.size(2));
    const int n_kv_heads = static_cast<int>(k_cache.size(1));
    const int block_size = static_cast<int>(k_cache.size(2));
    const int max_blocks = static_cast<int>(block_table.size(1));

    FD_CHECK(head_dim == k_cache.size(3), "q and k_cache head_dim differ");
    FD_CHECK(n_q_heads % n_kv_heads == 0,
             "n_q_heads must be a multiple of n_kv_heads for GQA");
    FD_CHECK(block_table.size(0) == batch, "block_table rows != batch");
    FD_CHECK(seq_lens.numel() == batch, "seq_lens length != batch");

    const int group = n_q_heads / n_kv_heads;
    auto out = torch::empty_like(q);

    // v2 scratch for the per-split partial softmaxes. Allocated here rather
    // than inside the kernel launch so its size is visible and so the
    // caching allocator reuses it across calls instead of cudaMalloc-ing.
    // At B=128 with 4 splits this is ~3 MB; at B=8 with 32 splits ~1.6 MB.
    torch::Tensor m_part, l_part, acc_part;
    int n_splits = 0;
    float *mp = nullptr, *lp = nullptr, *ap = nullptr;
    if (version == 2) {
        // block_table.size(1) is already a host-side int and bounds the
        // block count. The obvious seq_lens.max().item<int>() instead forces
        // a DEVICE-TO-HOST SYNC on every call, which breaks pipelining and
        // makes each launch pay full latency -- it was in the v2 path only
        // and is part of why v2 looked slower than v1 at shapes where
        // splitting should have been free.
        const int n_blocks = max_blocks;
        n_splits = splits > 0
                 ? static_cast<int>(splits)
                 : fd_choose_splits(batch, n_kv_heads, n_blocks, 32);
        auto f32 = q.options().dtype(torch::kFloat32);
        m_part = torch::empty({batch, n_q_heads, n_splits}, f32);
        l_part = torch::empty({batch, n_q_heads, n_splits}, f32);
        acc_part = torch::empty({batch, n_q_heads, n_splits, head_dim}, f32);
        mp = m_part.data_ptr<float>();
        lp = l_part.data_ptr<float>();
        ap = acc_part.data_ptr<float>();
    }

    const bool launched = paged_decode_attention_launch(
        reinterpret_cast<const bf16 *>(q.data_ptr()),
        reinterpret_cast<const bf16 *>(k_cache.data_ptr()),
        reinterpret_cast<const bf16 *>(v_cache.data_ptr()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        reinterpret_cast<bf16 *>(out.data_ptr()),
        batch, n_kv_heads, group, head_dim, block_size, max_blocks,
        static_cast<float>(scale), static_cast<int>(version),
        mp, lp, ap, n_splits, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(launched,
                "no paged attention kernel built for version=", version,
                " group=", group,
                " head_dim=", head_dim, " block_size=", block_size,
                ". v0 covers head_dim=128 with group in {1,2,3,4,6,8} and "
                "block_size in {16,32}; add a FD_LAUNCH line in "
                "kernels/paged_attention.cu.");

    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_residual", &rmsnorm_residual,
          "fused RMSNorm + residual (bf16)");
    m.def("paged_decode_attention", &paged_decode_attention,
          "paged GQA decode attention (bf16); version 0, 1 or 2",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("scale"),
          py::arg("version") = 2, py::arg("splits") = 0);
}