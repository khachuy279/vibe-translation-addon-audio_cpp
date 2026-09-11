"""A/B the translation backend: accidental Vulkan vs llama_cpp's own CUDA build.

Current behaviour (verified): `setup_cuda_dll_paths()` registers ZERO directories, so
`llama_cpp/lib/ggml.dll` cannot resolve `ggml-cuda.dll` (which imports cudart64_12.dll and
cublas64_12.dll). `import llama_cpp` therefore fails -- UNLESS `transcribe_cpp` was imported
first, which loads its own ggml.dll into the process and satisfies llama.dll's `ggml.dll`
dependency by name. The result is that llama.cpp silently runs on **transcribe.cpp's Vulkan
ggml**, and llama_cpp's own 795MB ggml-cuda.dll is never used.

This script runs llama_cpp against ITS OWN build by registering the directories that
`setup_cuda_dll_paths()` is supposed to register, and reports the resulting tok/s.

Usage:  python scratch/bench_translation_backend.py [--vulkan-fallback]
"""

import os
import site
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It is easy to tell the depth of a well.",
    "These days, a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the park truck.",
]


def register_dll_dirs() -> list:
    """Register the directories that setup_cuda_dll_paths() intends to register."""
    sp = site.getsitepackages()[-1]
    added = []
    candidates = [
        os.path.join(sp, "nvidia", "cublas", "bin"),
        os.path.join(sp, "nvidia", "cuda_runtime", "bin"),
        os.path.join(sp, "nvidia", "cuda_nvrtc", "bin"),
        os.path.join(sp, "llama_cpp", "lib"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            os.add_dll_directory(path)
            added.append(path)
    return added


def preload_llama_cpp_native() -> list:
    """Load llama_cpp's own native libraries in dependency order, by absolute path.

    `ggml.dll` statically imports `ggml-cuda.dll`, and `ggml-cuda.dll` imports
    `cudart64_12.dll` / `cublas64_12.dll`. The Windows loader will NOT name the missing
    dependency, and a plain `import llama_cpp` fails unless ggml-cuda.dll is already resident
    -- which is exactly what happens today by accident when `transcribe_cpp` (whose directory
    also contains a `ggml.dll`) was imported first.

    Preloading explicitly is what makes llama_cpp bind to ITS OWN CUDA build instead of
    transcribe.cpp's Vulkan one.
    """
    import ctypes

    sp = site.getsitepackages()[-1]
    lib = Path(sp) / "llama_cpp" / "lib"
    order = ["ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll"]
    loaded = []
    for name in order:
        path = lib / name
        if not path.exists():
            continue
        ctypes.WinDLL(str(path))
        loaded.append(name)
    return loaded


def main() -> int:
    use_vulkan_fallback = "--vulkan-fallback" in sys.argv

    if use_vulkan_fallback:
        import transcribe_cpp  # noqa: F401  -> binds llama.dll to transcribe's ggml (Vulkan)

        label = "VULKAN (accidental binding via transcribe_cpp)"
    else:
        added = register_dll_dirs()
        print("registered DLL dirs:")
        for path in added:
            print("   ", path)
        preloaded = preload_llama_cpp_native()
        print("preloaded llama_cpp native libs:", ", ".join(preloaded))
        label = "CUDA (llama_cpp's own ggml-cuda)"

    import llama_cpp
    from llama_cpp import Llama

    from backend_cpp.config import config

    cfg = config.translation
    model_path = Path("backend_cpp/models") / cfg.gguf_file
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        print(f"model missing: {model_path}")
        return 1

    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=cfg.n_gpu_layers,
        n_ctx=2048,
        n_batch=512,
        verbose=False,
    )
    load_ms = (time.perf_counter() - t0) * 1000.0

    from backend_cpp.translation.prompt_strategies import get_prompt_strategy

    strategy = get_prompt_strategy(
        model_name=cfg.gguf_file, repo_name=cfg.model, prompt_style=cfg.prompt_style
    )
    stop = strategy.get_stop_tokens()

    def run(text: str) -> dict:
        prompt = strategy.build_prompt(
            text=text, source_lang="auto", target_lang="vi",
            context="", use_context=cfg.use_context,
        )
        t = time.perf_counter()
        out = llm(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repetition_penalty,
            stop=stop,
        )
        ms = (time.perf_counter() - t) * 1000.0
        usage = out.get("usage", {})
        return {
            "ms": ms,
            "out_tok": usage.get("completion_tokens", 0),
            "text": out["choices"][0]["text"],
        }

    run(SENTENCES[0])  # warm-up

    print(f"\n=== {label} ===")
    print(f"llama_cpp {getattr(llama_cpp, '__version__', '?')} | load {load_ms:.0f} ms")
    print(f"{'#':>2} {'out_tok':>8} {'ms':>7} {'tok/s':>7}")
    tot_ms = 0.0
    tot_tok = 0
    first_text = ""
    for idx, sentence in enumerate(SENTENCES, 1):
        r = run(sentence)
        tot_ms += r["ms"]
        tot_tok += r["out_tok"]
        if idx == 1:
            first_text = r["text"]
        print(f"{idx:>2} {r['out_tok']:>8} {r['ms']:>7.0f} "
              f"{r['out_tok'] / max(0.001, r['ms'] / 1000.0):>7.1f}")
    print(
        f"   avg out_tok={tot_tok / len(SENTENCES):.1f}  avg ms={tot_ms / len(SENTENCES):.0f}  "
        f"overall tok/s={tot_tok / max(0.001, tot_ms / 1000.0):.1f}"
    )
    print(f"   sample: {first_text.strip()!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
