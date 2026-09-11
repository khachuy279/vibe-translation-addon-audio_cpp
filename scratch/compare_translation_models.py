"""Compare the available translation models: latency vs output, on the same 7 sentences.

Driven by the user's question: is the default Hy-MT2-7B (Q4_K_XL) slow because it is large,
or because it is loaded/run incorrectly?

Already answered for "loaded incorrectly":
  * GPU is used: 33/33 layers offloaded, ~67 tok/s (a CPU-only 7B Q4 tops out near 13 tok/s).
  * Prefill is only ~41 ms of the ~464 ms (the prompt is 52 tokens).
  * Decode is ~423 ms for ~30 output tokens; that is the whole latency.
  * max_tokens=128 never binds (observed max 43), so the cap costs nothing.
  * temperature 0 vs 0.7 gives the same length, so sampling is not the cause.
  * Backend A/B: llama_cpp's own CUDA = 68.2 tok/s vs the accidentally-bound Vulkan = 65-67
    tok/s, i.e. no practical difference.

So the remaining question is purely "which model", which is what this script measures.

Smaller models are in translation_models.yaml already. For each model we report latency AND
objective sanity flags (truncation at the token cap), plus the raw text so quality can be
judged by reading it -- there is no reference translation to score against.

Usage:  python scratch/compare_translation_models.py
"""

import gc
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

# Ordered cheapest -> most expensive so the tradeoff reads naturally.
MODELS = ["tencent-1.8b", "xiaomi", "tencent", "gemmax"]


def setup_llama_cpp():
    """Load llama_cpp against its own CUDA build (see bench_translation_backend.py)."""
    import ctypes

    sp = site.getsitepackages()[-1]
    for rel in ("nvidia/cublas/bin", "nvidia/cuda_runtime/bin", "nvidia/cuda_nvrtc/bin",
                "llama_cpp/lib"):
        d = os.path.join(sp, *rel.split("/"))
        if os.path.isdir(d):
            os.add_dll_directory(d)
    lib = Path(sp) / "llama_cpp" / "lib"
    for name in ("ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll"):
        if (lib / name).exists():
            ctypes.WinDLL(str(lib / name))


def main() -> int:
    setup_llama_cpp()

    from llama_cpp import Llama

    from backend_cpp.config import MODELS_DIR, TranslationConfig
    from backend_cpp.translation.prompt_strategies import get_prompt_strategy
    from backend_cpp.translation.model_registry import TranslationModelRegistry

    registry = TranslationModelRegistry.get_instance()
    available = set(registry.models.keys())
    print(f"registry models: {sorted(available)}\n")

    results = {}
    for base in MODELS:
        if base not in available:
            print(f"skip {base}: not in registry")
            continue

        cfg = TranslationConfig(base=base)
        path = Path(MODELS_DIR) / (cfg.gguf_file or "")
        if not path.exists():
            print(f"skip {base}: model file missing ({path.name})")
            continue

        print(f"{'=' * 78}\n{base}  ({path.name}, {path.stat().st_size / 1e9:.2f} GB)\n{'=' * 78}")

        t0 = time.perf_counter()
        llm = Llama(
            model_path=str(path),
            n_gpu_layers=cfg.n_gpu_layers,
            n_ctx=2048,
            n_batch=512,
            verbose=False,
        )
        load_ms = (time.perf_counter() - t0) * 1000.0

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
            n_tok = usage.get("completion_tokens", 0)
            return {
                "ms": ms,
                "out_tok": n_tok,
                "hit_cap": n_tok >= cfg.max_tokens,
                "text": out["choices"][0]["text"].strip(),
            }

        run(SENTENCES[0])  # warm-up

        rows = [run(s) for s in SENTENCES]
        tot_ms = sum(r["ms"] for r in rows)
        tot_tok = sum(r["out_tok"] for r in rows)
        truncated = sum(1 for r in rows if r["hit_cap"])

        print(f"load {load_ms:.0f} ms | avg {tot_ms / len(rows):.0f} ms | "
              f"avg {tot_tok / len(rows):.1f} out_tok | {tot_tok / max(0.001, tot_ms / 1000):.1f} tok/s"
              f" | truncated {truncated}/{len(rows)}")
        for idx, (sentence, row) in enumerate(zip(SENTENCES, rows), 1):
            flag = "  <-- HIT CAP" if row["hit_cap"] else ""
            print(f"  {idx}. {row['ms']:>5.0f}ms {row['out_tok']:>3}tok  {row['text']!r}{flag}")

        results[base] = {
            "avg_ms": tot_ms / len(rows),
            "out_tok": tot_tok / len(rows),
            "tok_s": tot_tok / max(0.001, tot_ms / 1000),
            "size_gb": path.stat().st_size / 1e9,
            "truncated": truncated,
        }

        del llm
        gc.collect()
        print()

    print("=" * 78)
    print(f"{'model':14s} {'GB':>5s} {'avg_ms':>8s} {'out_tok':>8s} {'tok/s':>7s} {'trunc':>6s} {'speedup':>8s}")
    print("-" * 78)
    baseline = results.get("tencent", {}).get("avg_ms") or 1.0
    for base, row in results.items():
        print(
            f"{base:14s} {row['size_gb']:>5.2f} {row['avg_ms']:>8.0f} {row['out_tok']:>8.1f} "
            f"{row['tok_s']:>7.1f} {row['truncated']:>6d} {baseline / row['avg_ms']:>7.2f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
