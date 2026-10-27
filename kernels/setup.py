"""
Build the CUDA extension.

    cd kernels && python setup.py build_ext --inplace

C++ STANDARD (fixed 2026-10-27): must be c++20, not c++17.

PyTorch 2.14's headers are C++20 and say so explicitly:

    torch/all.h:5: #error C++20 or later compatible compiler is required
                   to use PyTorch.

With -std=c++17 the build produces ~100 lines of cascading errors --
std::strong_ordering missing, `requires` not naming a type,
std::string_view::starts_with absent, operator<=> unavailable -- which all
trace back to that one #error. Those are C++20 features in torch's own
headers, not problems in this project's code. Set the standard and they all
disappear at once.

Older torch (<= 2.3) was C++17, which is why c++17 is such a common default
in CUDA-extension templates. It is wrong for this toolchain.

TORCH_CUDA_ARCH_LIST is set explicitly to 12.0. Without it, older PyTorch
build defaults stop at sm_90 and the extension compiles cleanly but fails at
RUNTIME with "no kernel image is available for execution on the device" --
a confusing way to lose an afternoon, because nothing in the build output
hints at the problem.

--use_fast_math is deliberately NOT enabled. It changes rsqrtf and would
introduce a small systematic difference from the PyTorch reference, which is
exactly what testing/diff.py is meant to be sensitive to. Turn it on later as
a measured experiment, not as a default.
"""

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
            sources=["bindings.cpp", "rmsnorm.cu"],
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