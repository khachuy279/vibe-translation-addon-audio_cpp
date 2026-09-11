"""Why can a fresh process not import llama_cpp, while the running app can?

`llama_cpp/lib/ggml-cuda.dll` (795MB) exists, so the wheel IS a CUDA build. Importing it fails
with "Could not find module ... llama.dll (or one of its dependencies)", i.e. a DLL dependency
of ggml-cuda.dll is missing at load time.

This matters because if the dependency is only found by accident (import order), then a future
refactor that changes import order could silently drop translation to CPU while the code still
"works".

Usage:  python scratch/diag_llama_dll.py
"""

import ctypes
import os
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PKG = Path("C:/Users/khach/AppData/Local/Programs/Python/Python313/Lib/site-packages")


def cuda_dlls_in(directory: Path):
    if not directory.is_dir():
        return []
    return sorted(
        p.name for p in directory.iterdir()
        if p.suffix.lower() == ".dll" and "cuda" in p.name.lower()
        or p.suffix.lower() == ".dll" and any(
            k in p.name.lower() for k in ("cublas", "cudnn", "cudart", "nvrtc")
        )
    )


def main():
    print("site.getsitepackages():")
    for sp in site.getsitepackages():
        print("   ", sp)

    print("\ntorch/lib CUDA-ish DLLs:")
    torch_lib = PKG / "torch" / "lib"
    for name in cuda_dlls_in(torch_lib):
        print("   ", name)

    print("\nnvidia/*/bin|lib CUDA-ish DLLs:")
    nvidia = PKG / "nvidia"
    if nvidia.is_dir():
        for item in sorted(nvidia.iterdir()):
            for sub in ("bin", "lib"):
                d = item / sub
                found = cuda_dlls_in(d)
                if found:
                    print(f"    {item.name}/{sub}: {found}")
    else:
        print("    <no nvidia dir>")

    print("\nDirect load attempts (this is the whole question):")
    for cand in [
        torch_lib / "cublas64_13.dll",
        torch_lib / "cublas64_12.dll",
        torch_lib / "cudart64_13.dll",
        torch_lib / "cudart64_12.dll",
    ]:
        if not cand.exists():
            print(f"    MISSING  {cand.name}")
            continue
        try:
            ctypes.CDLL(str(cand))
            print(f"    OK       {cand.name}")
        except OSError as exc:
            print(f"    FAILED   {cand.name}: {exc}")

    print("\nllama_cpp import:")
    from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths

    setup_cuda_dll_paths()
    try:
        import llama_cpp

        print("    OK   gpu_offload =", llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {str(exc)[:160]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
