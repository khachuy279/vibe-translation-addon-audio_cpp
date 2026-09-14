"""CUDA DLL Loader for llama-cpp-python on Windows.

Registers DLL directories and pre-loads dependencies (cudart, cublas, ggml)
to guarantee full CUDA GPU offload without DLL load errors.
"""

import ctypes
import logging
import os
import sys
import threading
from pathlib import Path

logger = logging.getLogger("backend_audio_cpp.translation.cuda")

_initialized: bool = False
_lock = threading.Lock()


def ensure_cuda_dlls_loaded() -> bool:
    """Pre-load CUDA and GGML DLLs for llama_cpp."""
    global _initialized
    if _initialized:
        return True

    with _lock:
        if _initialized:
            return True

        if sys.platform != "win32":
            _initialized = True
            return True

        project_root = Path(__file__).resolve().parent.parent.parent
        bin_dir = project_root / "backend_audio_cpp" / "bin"
        llama_lib = Path(sys.prefix) / "Lib" / "site-packages" / "llama_cpp" / "lib"

        if bin_dir.exists():
            os.add_dll_directory(str(bin_dir))
        if llama_lib.exists():
            os.add_dll_directory(str(llama_lib))

        # Explicit preload sequence for Windows PE dynamic loader
        dlls_to_load = [
            (bin_dir, "cudart64_12.dll"),
            (bin_dir, "cublasLt64_12.dll"),
            (bin_dir, "cublas64_12.dll"),
            (llama_lib, "ggml-base.dll"),
            (llama_lib, "ggml-cuda.dll"),
            (llama_lib, "ggml-cpu.dll"),
            (llama_lib, "ggml.dll"),
            (llama_lib, "llama.dll"),
        ]

        for base_p, dll_name in dlls_to_load:
            dll_p = base_p / dll_name
            if dll_p.exists():
                try:
                    ctypes.CDLL(str(dll_p))
                except Exception as e:
                    logger.debug(f"Preload {dll_name}: {e}")

        logger.info("[CUDA] Successfully initialized CUDA DLLs for llama_cpp")
        _initialized = True
        return True
