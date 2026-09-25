"""Environment fixes needed before importing vLLM on this machine.

Import this module before ``vllm``::

    import vllm_env  # noqa: F401

1. FlashInfer's CUDA-arch detection fails on the RTX 5090 (sm_120) with this
   CUDA toolkit ("FlashInfer requires GPUs with sm75 or higher" /
   "SM 12.x requires CUDA >= 12.9"). We pin the target arch to sm_120 and skip
   the check. vLLM uses FlashInfer for sampling even though attention runs on
   FLASH_ATTN.
2. Run the vLLM engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0). The engine
   otherwise starts in a spawned subprocess that does not inherit the patch
   above or our custom model registration, and scripts without an
   ``if __name__ == "__main__"`` guard crash on start-up.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import flashinfer.compilation_context as _fcc  # noqa: E402
import flashinfer.jit.core as _fcore  # noqa: E402

_fcore.current_compilation_context.TARGET_CUDA_ARCHS = {(12, "0")}
_fcore.check_cuda_arch = lambda: None

_orig_get_nvcc = _fcc.CompilationContext.get_nvcc_flags_list


def _patched_get_nvcc(self, *args, **kwargs):
    try:
        return _orig_get_nvcc(self, *args, **kwargs)
    except RuntimeError:
        return ["-gencode", "arch=compute_120,code=sm_120"]


_fcc.CompilationContext.get_nvcc_flags_list = _patched_get_nvcc
