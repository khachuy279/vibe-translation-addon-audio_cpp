"""Japanese ASR model bake-off on every locally available model.

Scores each candidate model against the Japanese evaluation set and prints
CER/WER plus the raw hypotheses so errors can be read directly.

    python scratch/ja_bakeoff.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transcribe_cpp  # noqa: E402
from benchmarks.accuracy import evaluate_accuracy  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "backend_cpp" / "models"

# ---------------------------------------------------------------- eval set
EVAL = [
    {
        "id": "Japanese_5s",
        "wav": REPO / "wav_test" / "Japanese_5s.wav",
        "ref": "抜群の運動神経を持ち合わせ、どんな要求にも応えてきた。",
        "domain": "narration",
    },
    {
        "id": "ja_sample",
        "wav": REPO / "transcribe.cpp" / "samples" / "ja.wav",
        # Reference quoted by transcribe.cpp's own porting docs (funasr_nano.md
        # acceptance run). Punctuation/ITN differs between models -> CER only.
        "ref": "うちの中学は弁当制で、持っていけない場合は、五十円の学校販売のパンを買う",
        "domain": "conversational",
    },
]

# ---------------------------------------------------------------- candidates
CANDIDATES = [
    ("qwen3-asr-1.7b", MODELS / "Qwen3-ASR-1.7B-Q8_0.gguf", "ja"),
    ("qwen3-asr-1.7b-ja-ft", REPO / "backend_cpp" / "convert" / "temp" / "Qwen3-ASR-1.7B-JA" / "Qwen3-ASR-1.7B-JA-BF16.gguf", "ja"),
    ("qwen3-asr-0.6b", MODELS / "Qwen3-ASR-0.6B-Q8_0.gguf", "ja"),
    ("kotoba-whisper-v2.2", MODELS / "kotoba-whisper-v2.2-BF16.gguf", "ja"),
    ("sensevoice-small", MODELS / "SenseVoiceSmall-F32.gguf", "ja"),
    ("cohere-transcribe", MODELS / "cohere-transcribe-03-2026-Q8_0.gguf", "ja"),
    ("nemotron-3.5-streaming", MODELS / "nemotron-3.5-asr-streaming-0.6b-F16.gguf", "ja-JP"),
    ("voxtral-mini-4b", MODELS / "Voxtral-Mini-4B-Realtime-2602-Q5_K_M.gguf", "ja"),
]


def load_wav(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import soxr

        data = soxr.resample(data, in_rate=sr, out_rate=16000, quality="HQ")
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []

    for key, path, lang in CANDIDATES:
        if only and only not in key:
            continue
        if not path.exists():
            print(f"SKIP {key}: missing {path}")
            continue

        print(f"\n{'='*78}\nMODEL {key}  ({path.name})\n{'='*78}", flush=True)
        t_load = time.perf_counter()
        try:
            model = transcribe_cpp.Model(str(path))
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            rows.append({"model": key, "error": f"load: {e}"})
            continue
        caps = model.capabilities
        load_ms = (time.perf_counter() - t_load) * 1000.0
        print(f"  arch={model.arch} backend={model.backend} load={load_ms:.0f}ms")
        print(f"  caps: streaming={caps.supports_streaming} langs={len(caps.languages)} max_audio_ms={caps.max_audio_ms}")

        try:
            session = model.session()
        except Exception as e:
            print(f"  SESSION FAILED: {e}")
            model.close()
            continue

        for item in EVAL:
            pcm = load_wav(item["wav"])
            t0 = time.perf_counter()
            try:
                res = session.run(pcm, language=lang)
                hyp = (getattr(res, "text", "") or "").strip()
                err = None
            except Exception as e:
                partial = getattr(e, "partial_result", None)
                hyp = (getattr(partial, "text", "") or "").strip() if partial is not None else ""
                err = f"{type(e).__name__}: {e}"
            ms = (time.perf_counter() - t0) * 1000.0

            acc = evaluate_accuracy(raw_reference=item["ref"], raw_hypothesis=hyp, language="ja")
            print(f"\n  [{item['id']}] {ms:.0f}ms  CER={acc.cer*100:.2f}%  WER={acc.wer*100:.2f}%")
            print(f"    REF: {item['ref']}")
            print(f"    HYP: {hyp}")
            if err:
                print(f"    ERR: {err}")
            rows.append(
                {
                    "model": key,
                    "item": item["id"],
                    "inference_ms": round(ms, 1),
                    "cer": acc.cer,
                    "wer": acc.wer,
                    "ref": item["ref"],
                    "hyp": hyp,
                    "error": err,
                }
            )

        try:
            session.close()
        except Exception:
            pass
        model.close()

    out = REPO / "report" / "ja_bakeoff.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*78}\nSUMMARY (CER%)\n{'='*78}")
    print(f"{'model':<26} " + " ".join(f"{i['id']:>14}" for i in EVAL) + "    mean")
    for key, _p, _l in CANDIDATES:
        sub = [r for r in rows if r["model"] == key and "error" not in r or (r.get("model") == key and r.get("cer") is not None)]
        sub = {r["item"]: r for r in rows if r["model"] == key and r.get("cer") is not None}
        if not sub:
            continue
        cells = []
        vals = []
        for i in EVAL:
            r = sub.get(i["id"])
            if r is None:
                cells.append(f"{'-':>14}")
            else:
                cells.append(f"{r['cer']*100:>13.2f}%")
                vals.append(r["cer"])
        mean = 100.0 * sum(vals) / len(vals) if vals else float("nan")
        print(f"{key:<26} " + " ".join(cells) + f"  {mean:6.2f}%")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
