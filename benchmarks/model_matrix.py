"""Compare ASR models on the repo's own test material (all languages).

Answers two questions at once:

1. Does switching the Japanese default to `cohere-transcribe` regress English /
   Chinese / Russian / multilingual?
2. Does the qwen3-asr-1.7b "continuation" failure also show up on the user's own
   captured Japanese drama session?

    python -m benchmarks.model_matrix --set wav_test
    python -m benchmarks.model_matrix --captured
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

import transcribe_cpp
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.dataset import discover_dataset
from benchmarks.ja_text import score_ja

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "backend_cpp" / "models"

CANDIDATES: Dict[str, Path] = {
    "qwen3-asr-1.7b": MODELS / "Qwen3-ASR-1.7B-Q8_0.gguf",
    "cohere-transcribe": MODELS / "cohere-transcribe-03-2026-Q8_0.gguf",
    "kotoba-whisper-v2.2": MODELS / "kotoba-whisper-v2.2-BF16.gguf",
}
LANG_ARG = {
    "qwen3-asr-1.7b": lambda lang: None if lang in ("multi", "UNKNOWN") else lang,
    "cohere-transcribe": lambda lang: None if lang in ("multi", "UNKNOWN") else lang,
    "kotoba-whisper-v2.2": lambda lang: "ja" if lang == "ja" else (None if lang in ("multi", "UNKNOWN") else lang),
}


def load_wav(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import soxr

        data = soxr.resample(data, in_rate=sr, out_rate=16000, quality="HQ")
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def run_matrix(items: List[dict]) -> None:
    print(f"{len(items)} items, {len(CANDIDATES)} models\n")
    table: Dict[str, Dict[str, float]] = {}

    for key, path in CANDIDATES.items():
        if not path.exists():
            print(f"SKIP {key}")
            continue
        model = transcribe_cpp.Model(str(path))
        session = model.session()
        # warm-up (first Vulkan run pays pipeline creation)
        session.run(load_wav(Path(items[0]["wav"])), language=None)

        row: Dict[str, float] = {}
        print(f"=== {key} ===")
        for it in items:
            pcm = load_wav(Path(it["wav"]))
            lang = it["lang"]
            try:
                res = session.run(pcm, language=LANG_ARG[key](lang))
                hyp = (getattr(res, "text", "") or "").strip()
            except Exception as e:
                partial = getattr(e, "partial_result", None)
                hyp = (getattr(partial, "text", "") or "").strip() if partial is not None else ""
            if lang == "ja":
                sc = score_ja(it["ref"], hyp)
                metric = sc.cer if sc.cer is not None else 0.0
            else:
                acc = evaluate_accuracy(raw_reference=it["ref"], raw_hypothesis=hyp, language=lang)
                metric = acc.cer if acc.cer is not None else 0.0
            row[it["name"]] = metric
            print(f"  {it['name']:<44} CER={metric*100:6.2f}%  {hyp[:60]}")
        table[key] = row
        session.close()
        model.close()
        print()

    names = [it["name"] for it in items]
    print("=" * 110)
    print("SUMMARY CER %")
    print("=" * 110)
    print(f"{'item':<44}" + "".join(f"{k[:20]:>22}" for k in table))
    for n in names:
        print(f"{n:<44}" + "".join(f"{table[k].get(n, float('nan'))*100:>21.2f}%" for k in table))


def wav_test_items() -> List[dict]:
    out = []
    for pair in discover_dataset("wav_test"):
        out.append({"name": pair.pair_id, "wav": str(pair.wav_path), "ref": pair.raw_reference, "lang": pair.inferred_language})
    return out


def captured_items() -> List[dict]:
    """The user's own captured Japanese drama session -- no ground truth, but the
    whole-stream decode is a strong reference for the per-segment decode."""
    session_dir = sorted((REPO / "debug_audio").glob("*"))[-1]
    items = [{"name": "captured:ingress_29s", "wav": str(session_dir / "00_ingress_stream.wav"), "ref": "", "lang": "ja"}]
    for f in sorted(session_dir.glob("*_vad_*.wav")):
        if f.stat().st_size > 3200:
            items.append({"name": f"captured:{f.name[:2]}", "wav": str(f), "ref": "", "lang": "ja"})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captured", action="store_true", help="use the captured drama session instead of wav_test")
    ap.add_argument("--out", default="report/model_matrix.json")
    args = ap.parse_args()

    items = captured_items() if args.captured else wav_test_items()
    if args.captured:
        # No ground truth: dump raw hypotheses side by side instead of scoring.
        rows = {}
        for key, path in CANDIDATES.items():
            model = transcribe_cpp.Model(str(path))
            session = model.session()
            session.run(load_wav(Path(items[0]["wav"])), language="ja")
            rows[key] = {}
            for it in items:
                res = session.run(load_wav(Path(it["wav"])), language="ja")
                rows[key][it["name"]] = (getattr(res, "text", "") or "").strip()
            session.close()
            model.close()
        print("=" * 100)
        for it in items:
            print(f"\n### {it['name']}")
            for key in CANDIDATES:
                print(f"  {key:<22}: {rows[key][it['name']]}")
        with open(REPO / args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return

    run_matrix(items)


if __name__ == "__main__":
    main()
