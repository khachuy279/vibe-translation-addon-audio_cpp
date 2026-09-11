"""CUDA / Vulkan DLL path setup for Windows."""

import os
import sys
import logging
import threading

logger = logging.getLogger(__name__)

_cuda_paths_initialized: bool = False
_init_lock = threading.Lock()


def discover_cuda_dll_dirs() -> set:
    """Directories whose DLLs must be findable for CUDA / cuBLAS / llama.cpp loading.

    Pure and side-effect free, so it can be unit tested -- see
    ``test_cuda_dll_discovery_actually_scans_site_packages``.

    Background: this logic used to be inlined in ``setup_cuda_dll_paths()`` with the
    per-site-package block OUTSIDE its ``for sp in site_packages_dirs`` loop. The loop therefore
    only performed the existence check, and every path was derived from the LAST entry --
    ``site.USER_SITE``, which does not exist on a normal install. The set came back EMPTY:

        Configured CUDA DLL directories: []

    so no directory was ever registered, defeating the function's stated purpose. That is why
    ``import llama_cpp`` fails in a fresh process: ``llama_cpp/lib/ggml.dll`` statically imports
    ``ggml-cuda.dll``, which needs cudart64_12.dll / cublas64_12.dll from
    ``nvidia/cuda_runtime/bin`` and ``nvidia/cublas/bin``. The app only manages to load the
    translation model because ASR imports ``transcribe_cpp`` first, which loads a ``ggml.dll``
    into the process that llama.dll then binds to by name (and that one uses Vulkan, not CUDA --
    measured at 65-67 tok/s vs 68.2 tok/s for the CUDA path, so the accident costs nothing but
    the coupling is worth knowing about).
    """
    dll_dirs = set()

    # 1. site-packages: torch/lib (CUDA runtime / cuBLAS), llama_cpp/lib (llama.cpp's own
    #    ggml + ggml-cuda), and the nvidia-*-cu12 runtime wheels.
    try:
        import site

        site_packages_dirs = [p for p in (site.getsitepackages() + [site.USER_SITE]) if p]
        for sp in site_packages_dirs:
            if not os.path.exists(sp):
                continue

            torch_lib = os.path.join(sp, "torch", "lib")
            if os.path.isdir(torch_lib):
                dll_dirs.add(os.path.abspath(torch_lib))

            llama_lib = os.path.join(sp, "llama_cpp", "lib")
            if os.path.isdir(llama_lib):
                dll_dirs.add(os.path.abspath(llama_lib))

            nvidia_dir = os.path.join(sp, "nvidia")
            if os.path.isdir(nvidia_dir):
                for item in os.listdir(nvidia_dir):
                    if item.lower() == "cudnn":
                        # Skip nvidia/cudnn/bin to prevent CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH with PyTorch
                        continue
                    sub_bin = os.path.join(nvidia_dir, item, "bin")
                    if os.path.isdir(sub_bin):
                        dll_dirs.add(os.path.abspath(sub_bin))
                    sub_lib = os.path.join(nvidia_dir, item, "lib")
                    if os.path.isdir(sub_lib):
                        dll_dirs.add(os.path.abspath(sub_lib))
    except Exception as e:
        logger.debug(f"Error scanning site-packages for CUDA DLLs: {e}")

    # 2. CUDA environment variables (CUDA_PATH, CUDA_PATH_V12_*, etc.)
    for env_key, env_val in os.environ.items():
        if env_key.startswith("CUDA_PATH") and env_val and os.path.exists(env_val):
            bin_dir = os.path.join(env_val, "bin")
            if os.path.isdir(bin_dir):
                dll_dirs.add(os.path.abspath(bin_dir))

    # 3. Common CUDA toolkit default installation paths on Windows
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_toolkit_base = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
    if os.path.isdir(cuda_toolkit_base):
        try:
            for ver_dir in os.listdir(cuda_toolkit_base):
                bin_dir = os.path.join(cuda_toolkit_base, ver_dir, "bin")
                if os.path.isdir(bin_dir):
                    dll_dirs.add(os.path.abspath(bin_dir))
        except Exception:
            pass

    return dll_dirs


def setup_cuda_dll_paths() -> None:
    """Ensure Windows dynamic loader (LoadLibrary) can find CUDA/cuBLAS/cuDNN and llama.dll dependencies."""
    global _cuda_paths_initialized
    if _cuda_paths_initialized:
        return

    with _init_lock:
        if _cuda_paths_initialized:
            return

        if sys.platform != "win32":
            _cuda_paths_initialized = True
            return

        dll_dirs = discover_cuda_dll_dirs()

        # Register directories with os.add_dll_directory and PATH
        path_env = os.environ.get("PATH", "")
        path_list = path_env.split(os.path.pathsep)

        for d in dll_dirs:
            try:
                os.add_dll_directory(d)
            except Exception as e:
                logger.debug(f"Could not add DLL directory {d}: {e}")

            if d not in path_list:
                path_list.insert(0, d)

        os.environ["PATH"] = os.path.pathsep.join(path_list)
        _cuda_paths_initialized = True
        logger.debug(f"Configured CUDA DLL directories: {list(dll_dirs)}")

