"""List the DLLs that llama_cpp's ggml-cuda.dll imports, without extra dependencies.

`import llama_cpp` fails in a fresh process; `llama_cpp/lib/ggml-base.dll` and `ggml-cpu.dll`
load fine but `ggml.dll`, `ggml-cuda.dll` and `llama.dll` do not. Windows reports only
"Could not find module" without naming the missing dependency, so read the PE import table.

Usage:  python scratch/pe_imports.py [dll ...]
"""

import struct
import sys
from pathlib import Path

DEFAULT_DIR = Path(
    "C:/Users/khach/AppData/Local/Programs/Python/Python313/Lib/site-packages/llama_cpp/lib"
)


def read_imports(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError("not a PE file")

    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_off:pe_off + 4] != b"PE\0\0":
        raise ValueError("bad PE signature")

    machine, n_sections = struct.unpack_from("<HH", data, pe_off + 4)
    opt_size = struct.unpack_from("<H", data, pe_off + 20)[0]
    magic = struct.unpack_from("<H", data, pe_off + 24)[0]
    pe32_plus = magic == 0x20B

    # Data directory 1 = import table; its offset differs between PE32 and PE32+.
    dd_off = pe_off + 24 + (112 if pe32_plus else 96)
    import_rva, import_size = struct.unpack_from("<II", data, dd_off + 8)
    if import_rva == 0:
        return []

    sections = []
    sec_off = pe_off + 24 + opt_size
    for i in range(n_sections):
        base = sec_off + i * 40
        name = data[base:base + 8].rstrip(b"\0").decode("latin-1")
        v_size, v_addr, raw_size, raw_ptr = struct.unpack_from("<IIII", data, base + 8)
        sections.append((v_addr, v_size, raw_ptr, raw_size, name))

    def rva_to_off(rva: int) -> int:
        for v_addr, v_size, raw_ptr, raw_size, _name in sections:
            if v_addr <= rva < v_addr + max(v_size, raw_size):
                return raw_ptr + (rva - v_addr)
        raise ValueError(f"RVA 0x{rva:x} not mapped")

    names = []
    off = rva_to_off(import_rva)
    while True:
        entry = data[off:off + 20]
        if len(entry) < 20 or entry == b"\0" * 20:
            break
        name_rva = struct.unpack_from("<I", entry, 12)[0]
        if name_rva == 0:
            break
        n_off = rva_to_off(name_rva)
        end = data.index(b"\0", n_off)
        names.append(data[n_off:end].decode("latin-1"))
        off += 20
    return names


def main() -> int:
    targets = sys.argv[1:] or [
        "ggml-base.dll",
        "ggml-cpu.dll",
        "ggml.dll",
        "ggml-cuda.dll",
        "llama.dll",
    ]
    for target in targets:
        path = Path(target)
        if not path.is_absolute():
            path = DEFAULT_DIR / target
        print(f"\n=== {path.name} ({path.stat().st_size / 1e6:.1f} MB) ===")
        try:
            imports = read_imports(path)
        except Exception as exc:
            print(f"   parse failed: {exc}")
            continue
        for name in imports:
            marker = ""
            if name.lower().startswith(("cublas", "cudart", "nvrtc", "cudnn", "nvcuda")):
                marker = "   <-- CUDA runtime (NOT in SysWOW64/System32)"
            print(f"   {name}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
