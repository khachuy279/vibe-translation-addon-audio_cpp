"""Diagnose where translation latency actually goes for the default model.

Question being answered: is the default Hy-MT2-7B (Q4_K_XL) simply "a big model", or is it
being loaded/run in a way that costs more than necessary?

Baseline evidence already recorded (report/baseline_perf.json):
    translation.infer_ms        p50 439ms, avg 497ms, p90 659ms
    translation.tokens_per_sec  avg 62.5 tok/s      <- GPU-class for a 7B Q4
    => ~31 completion tokens are generated per sentence.

62 tok/s rules out a CPU-only run (a 7B Q4 on 12 CPU cores tops out near 10-13 tok/s), so the
GPU offload is working. That redirects the investigation to:
    (a) prefill cost (prompt length),
    (b) how many tokens the model emits before it stops.

Usage (project root):  python scratch/diag_translation_latency.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# IMPORT ORDER IS LOAD-BEARING -- do not "clean this up".
#
# `setup_cuda_dll_paths()` currently registers ZERO directories (indentation bug: the
# torch/lib, llama_cpp/lib and nvidia/* derivation sits outside its `for sp in
# site_packages_dirs` loop and uses the non-existent site.USER_SITE). Verified:
#     Configured CUDA DLL directories: []
#
# Consequently `import llama_cpp` FAILS in a fresh process:
#     llama_cpp/lib/ggml-base.dll -> loads
#     llama_cpp/lib/ggml-cpu.dll  -> loads
#     llama_cpp/lib/ggml.dll      -> FAILS (missing dependency)
#     llama_cpp/lib/llama.dll     -> FAILS
#
# It only succeeds once `transcribe_cpp` has LOADED its own ggml.dll into the process, which
# then satisfies llama.dll's `ggml.dll` dependency. The app gets this for free because ASR is
# imported before translation starts, so translation works BY ACCIDENT of import order.
#
# This script reproduces the app's order deliberately.
# ---------------------------------------------------------------------------
from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths  # noqa: E402

setup_cuda_dll_paths()

import transcribe_cpp  # noqa: E402,F401  (loads the ggml.dll that llama.dll binds to)

import llama_cpp  # noqa: E402
from llama_cpp import Llama  # noqa: E402

from backend_cpp.config import config  # noqa: E402
from backend_cpp.translation.local_translator import LocalGGUFTranslator  # noqa: E402
from backend_cpp.translation.prompt_strategies import get_prompt_strategy  # noqa: E402

SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It is easy to tell the depth of a well.",
    "These days, a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the park truck.",
]


def show_config():
    cfg = config.translation
    print("=== effective TranslationConfig ===")
    for field in cfg.model_fields:
        print(f"  {field:20s} = {getattr(cfg, field)!r}")
    print(f"  {'n_ctx (fallback)':20s} = {getattr(cfg, 'n_ctx', 2048)!r}")
    print(f"  {'n_batch (fallback)':20s} = {getattr(cfg, 'n_batch', 512)!r}")
    print(f"  {'n_threads':20s} = <not set -> llama.cpp default = cpu_count()>")


def main():
    print(f"llama_cpp version : {getattr(llama_cpp, '__version__', '?')}")
    print(f"gpu_offload ok    : {llama_cpp.llama_supports_gpu_offload()}")
    print()

    show_config()
    print()

    translator = LocalGGUFTranslator()
    model_path = translator._resolve_gguf_path()
    print(f"model file        : {model_path}")
    if not model_path:
        print("!! no model resolved")
        return 1

    cfg = config.translation
    strategy = get_prompt_strategy(
        model_name=cfg.gguf_file, repo_name=cfg.model, prompt_style=cfg.prompt_style
    )
    prompt = strategy.build_prompt(
        text=SENTENCES[0], source_lang="auto", target_lang="vi",
        context="", use_context=cfg.use_context,
    )
    print("\n=== prompt for sentence 1 ===")
    print(repr(prompt))
    print("stop tokens:", strategy.get_stop_tokens())
    print()

    print("=== loading (verbose) ===")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=cfg.n_gpu_layers,
        n_ctx=getattr(cfg, "n_ctx", 2048),
        n_batch=getattr(cfg, "n_batch", 512),
        verbose=True,
    )
    print(f"load time: {(time.perf_counter() - t0) * 1000:.0f} ms\n")

    def one(text: str, **over):
        p = strategy.build_prompt(
            text=text, source_lang="auto", target_lang="vi", context="",
            use_context=cfg.use_context,
        )
        params = dict(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repetition_penalty,
            stop=strategy.get_stop_tokens(),
        )
        params.update(over)
        t = time.perf_counter()
        out = llm(p, **params)
        ms = (time.perf_counter() - t) * 1000.0
        usage = out.get("usage", {}) if isinstance(out, dict) else {}
        return {
            "ms": ms,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "text": out["choices"][0]["text"] if isinstance(out, dict) else str(out),
        }

    # Warm-up: the first call also pays for graph/KV allocation, which is not per-sentence cost.
    one(SENTENCES[0])
    llm.reset() if hasattr(llm, "reset") else None

    print("=== A. current config, per sentence ===")
    print(f"{'#':>2} {'prefill_tok':>11} {'out_tok':>8} {'ms':>8} {'tok/s':>7}  output")
    totals = {"out": 0, "ms": 0.0}
    for idx, sentence in enumerate(SENTENCES, 1):
        r = one(sentence)
        tps = r["completion_tokens"] / max(0.001, r["ms"] / 1000.0)
        totals["out"] += r["completion_tokens"]
        totals["ms"] += r["ms"]
        print(
            f"{idx:>2} {r['prompt_tokens']:>11} {r['completion_tokens']:>8} "
            f"{r['ms']:>8.0f} {tps:>7.1f}  {r['text']!r}"
        )
    print(
        f"   avg out_tok={totals['out'] / len(SENTENCES):.1f}  "
        f"avg ms={totals['ms'] / len(SENTENCES):.0f}"
    )

    print("\n=== B. prefill-only cost (max_tokens=1) ===")
    for sentence in SENTENCES[:3]:
        r = one(sentence, max_tokens=1)
        print(
            f"  prompt_tok={r['prompt_tokens']:>4}  "
            f"prefill+1tok={r['ms']:>7.0f} ms   {sentence[:38]!r}"
        )

    print("\n=== C. greedy decoding (temperature=0, top_k=1) vs current (0.7/20) ===")
    for sentence in SENTENCES:
        greedy = one(sentence, temperature=0.0, top_p=1.0, top_k=1)
        print(
            f"  out_tok={greedy['completion_tokens']:>3} ms={greedy['ms']:>6.0f}  "
            f"{greedy['text']!r}"
        )

    print("\n=== D. max_tokens sensitivity (does it ever run to the cap?) ===")
    for cap in (16, 32, 48, 64, 128):
        r = one(SENTENCES[0], max_tokens=cap)
        print(
            f"  max_tokens={cap:>4}  out_tok={r['completion_tokens']:>3}  "
            f"ms={r['ms']:>7.0f}  {r['text']!r}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
