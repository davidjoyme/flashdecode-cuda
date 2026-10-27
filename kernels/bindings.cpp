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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_residual", &rmsnorm_residual,
          "fused RMSNorm + residual (bf16)");
}