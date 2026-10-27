# ==========================================================================
# FILE LOCATION:  flashdecode/engine/ops_cuda.py
# IMPORTED AS:    engine.ops_cuda      (run everything from the project root)
# ==========================================================================
"""
CUDA implementations, registered against the existing op names.

Importing this module registers the kernels; it does not activate them.
Activate explicitly:

    from engine import ops, ops_cuda          # noqa: F401
    ops.use("rmsnorm_residual", "cuda")

or ops.use_all("cuda") to switch every op that has a CUDA impl. Ops without
one stay on the reference path, which is the whole point of the registry:
you can run a half-CUDA model and bisect a correctness bug to one kernel.

IMPORT PATH (fixed 2026-10-27): `setup.py build_ext --inplace` writes
fd_kernels*.so into kernels/, but every script here runs as
`python -m pkg.module` from the PROJECT ROOT, so kernels/ is not on
sys.path and a bare `import fd_kernels` fails. The result was the worst kind
of error message -- "not built, run setup.py" immediately after a successful
build. So add kernels/ to sys.path here rather than asking anyone to set
PYTHONPATH.
"""

from __future__ import annotations

import os
import torch
import sys
import warnings

from engine.ops import register

_KERNELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels")
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

try:
    import fd_kernels

    AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    fd_kernels = None
    AVAILABLE = False
    IMPORT_ERROR = exc
    # Distinguish "never built" from "built but failed to load" -- the second
    # is usually an ABI mismatch (extension built against a different torch
    # or python than the one importing it), and the fix is a rebuild, not a
    # first build.
    import glob

    built = glob.glob(os.path.join(_KERNELS_DIR, "fd_kernels*.so"))
    if built:
        warnings.warn(
            f"found {os.path.basename(built[0])} but importing it failed: {exc}. "
            f"Usually an ABI mismatch -- rebuild against the current torch:\n"
            f"  cd kernels && rm -rf build *.so && python setup.py build_ext --inplace",
            stacklevel=2,
        )
    else:
        warnings.warn(
            "fd_kernels not built -- CUDA impls unavailable, reference path only.\n"
            "  cd kernels && python setup.py build_ext --inplace",
            stacklevel=2,
        )


if AVAILABLE:

    @register("rmsnorm_residual", "cuda")
    def rmsnorm_residual_cuda(x, residual, weight, eps):
        """Returns (normed, new_residual) -- same contract as the reference.

        Signature is unchanged from ops.rmsnorm_residual_ref, which is why
        model.py needs no edit to adopt this beyond switching the impl.
        """
        out, new_res = fd_kernels.rmsnorm_residual(
            x.contiguous(), residual.contiguous(), weight.contiguous(), float(eps)
        )
        return out, new_res


def status() -> str:
    if AVAILABLE:
        return f"fd_kernels loaded from {fd_kernels.__file__}"
    return f"fd_kernels unavailable: {IMPORT_ERROR}"


if AVAILABLE:

    @register("paged_decode_attention", "cuda")
    def paged_decode_attention_cuda(q, k_cache, v_cache, block_table,
                                    seq_lens, scale):
        """Same signature as ops.paged_decode_attention_ref.

        The reference loops over sequences in Python and gathers each one's
        blocks; this reads the cache in place. Note seq_lens must be int32 --
        the engine passes `slens + 1`, which keeps the dtype, but a caller
        building it by hand can easily produce int64 and the binding will
        reject it rather than reinterpret the bytes.
        """
        return fd_kernels.paged_decode_attention(
            q.contiguous(), k_cache.contiguous(), v_cache.contiguous(),
            block_table.contiguous().to(torch.int32),
            seq_lens.contiguous().to(torch.int32), float(scale))