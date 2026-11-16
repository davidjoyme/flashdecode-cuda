import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
from setuptools import setup                                   # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

CXX_STD = "c++20"   # required by torch >= 2.4 headers; see module docstring

setup(
    name="fd_kernels",
    ext_modules=[
        CUDAExtension(
            name="fd_kernels",
            sources=["bindings.cpp", "rmsnorm.cu", "paged_attention.cu"],
            extra_compile_args={
                "cxx": ["-O3", f"-std={CXX_STD}"],
                "nvcc": [
                    "-O3",
                    f"-std={CXX_STD}",
                    # -lineinfo lets ncu attribute stalls to source lines.
                    # Costs nothing at runtime, essential for profiling.
                    "-lineinfo",
                    # Register and spill counts at build time. Read these:
                    # spills are the usual cause of a kernel that profiles
                    # as occupancy-limited.
                    "-Xptxas=-v",
                    "-gencode=arch=compute_120,code=sm_120",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)