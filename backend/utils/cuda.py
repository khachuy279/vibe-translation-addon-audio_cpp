"""Thiết lập đường dẫn nạp DLL cho CUDA/cuBLAS/cuDNN và Vulkan trên môi trường Windows."""

import os
import sys
import threading
from typing import Set

from backend.utils.logger import logger

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_cuda_paths_initialized: bool = False
_init_lock = threading.RLock()


def setup_cuda_dll_paths() -> None:
    """Đăng ký đường dẫn chứa CUDA/Vulkan DLLs cho Windows dynamic loader."""
    global _cuda_paths_initialized
    if _cuda_paths_initialized:
        return

    with _init_lock:
        if _cuda_paths_initialized:
            return

        if sys.platform != "win32":
            _cuda_paths_initialized = True
            return

        dll_dirs: Set[str] = set()

        # 0. Thêm backend/bin của dự án
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bin_dir = os.path.join(here, "bin")
        if os.path.isdir(bin_dir):
            dll_dirs.add(os.path.abspath(bin_dir))

        # 1. Tìm trong site-packages (torch, llama_cpp, nvidia)
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
                            continue
                        sub_bin = os.path.join(nvidia_dir, item, "bin")
                        if os.path.isdir(sub_bin):
                            dll_dirs.add(os.path.abspath(sub_bin))
                        sub_lib = os.path.join(nvidia_dir, item, "lib")
                        if os.path.isdir(sub_lib):
                            dll_dirs.add(os.path.abspath(sub_lib))
        except Exception as e:
            logger.debug(f"Lỗi khi quét site-packages cho CUDA DLLs: {e}")

        # 2. Tìm trong biến môi trường CUDA_PATH
        for env_key, env_val in os.environ.items():
            if env_key.startswith("CUDA_PATH") and env_val and os.path.exists(env_val):
                bin_dir = os.path.join(env_val, "bin")
                if os.path.isdir(bin_dir):
                    dll_dirs.add(os.path.abspath(bin_dir))

        # 3. Tìm trong Program Files NVIDIA GPU Computing Toolkit
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

        # Đăng ký với Windows loader
        path_env = os.environ.get("PATH", "")
        path_list = path_env.split(os.path.pathsep)

        for d in dll_dirs:
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

            if d not in path_list:
                path_list.insert(0, d)

        os.environ["PATH"] = os.path.pathsep.join(path_list)
        _cuda_paths_initialized = True
        logger.debug(f"Đã đăng ký các thư mục CUDA DLL: {list(dll_dirs)}")
