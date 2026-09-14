"""Japanese offline ASR bench: score candidate models on the ReazonSpeech eval set.

This is the *discriminating* measurement. The original Japanese material in the repo
(2 clean clips, ~10 s) scored ~0% for every competent model and could not separate
them; this set is 1000 natural TV-speech utterances (~88 min), which can.

    python -m benchmarks.ja_offline_bench --limit 200
    python -m benchmarks.ja_offline_bench --models qwen3-asr-1.7b,cohere-transcribe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

import transcribe_cpp
from benchmarks.ja_text import score_ja

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "backend_cpp" / "models"
DATA = REPO / "data" / "ja_eval"

CANDIDATES: Dict[str, Dict[str, Any]] = {
    "qwen3-asr-1.7b": {"file": MODELS_DIR / "Qwen3-ASR-1.7B-Q8_0.gguf", "lang": "ja"},
    "qwen3-asr-1.7b-ja-ft": {
        "file": REPO / "backend_cpp" / "convert" / "temp" / "Qwen3-ASR-1.7B-JA" / "Qwen3-ASR-1.7B-JA-BF16.gguf",
        "lang": "ja",
    },
    "qwen3-asr-0.6b": {"file": MODELS_DIR / "Qwen3-ASR-0.6B-Q8_0.gguf", "lang": "ja"},
    "cohere-transcribe": {"file": MODELS_DIR / "cohere-transcribe-03-2026-Q8_0.gguf", "lang": "ja"},
    "kotoba-whisper-v2.2": {"file": MODELS_DIR / "kotoba-whisper-v2.2-BF16.gguf", "lang": "ja"},
    "sensevoice-small": {"file": MODELS_DIR / "SenseVoiceSmall-F32.gguf", "lang": "ja"},
    "nemotron-3.5-streaming": {"file": MODELS_DIR / "nemotron-3.5-asr-streaming-0.6b-F16.gguf", "lang": "ja-JP"},
    "voxtral-mini-4b": {"file": MODELS_DIR / "Voxtral-Mini-4B-Realtime-2602-Q5_K_M.gguf", "lang": "ja"},
}


def load_manifest(limit: int, manifest_path: Path | None = None) -> List[dict]:
    rows = []
    with open(manifest_path or (DATA / "utts.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit > 0 else rows


_ACTIVE_DATA = DATA


def load_wav(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert sr == 16000, f"unexpected sample rate {sr} for {path}"
    return data.astype(np.float32)


def bench_model(key: str, spec: Dict[str, Any], manifest: List[dict], warmup: int = 3) -> Dict[str, Any]:
    path: Path = spec["file"]
    lang: str = spec["lang"]
    if not path.exists():
        return {"model": key, "error": f"missing model file {path}"}

    t0 = time.perf_counter()
    model = transcribe_cpp.Model(str(path))
    load_ms = (time.perf_counter() - t0) * 1000.0
    arch = model.arch
    session = model.session()

    for r in manifest[:warmup]:
        session.run(load_wav(_ACTIVE_DATA / r["wav"]), language=lang)

    per_utt: List[dict] = []
    total_audio = 0.0
    total_infer = 0.0
    for r in manifest:
        pcm = load_wav(_ACTIVE_DATA / r["wav"])
        t1 = time.perf_counter()
        try:
            res = session.run(pcm, language=lang)
            hyp = (getattr(res, "text", "") or "").strip()
            err = None
        except Exception as e:
            partial = getattr(e, "partial_result", None)
            hyp = (getattr(partial, "text", "") or "").strip() if partial is not None else ""
            err = type(e).__name__
        infer = time.perf_counter() - t1
        total_audio += len(pcm) / 16000.0
        total_infer += infer

        sc = score_ja(r["text"], hyp)
        per_utt.append(
            {
                "id": r["id"],
                "ref": r["text"],
                "hyp": hyp,
                "cer": sc.cer,
                "cer_itn": sc.cer_itn,
                "ref_len": sc.ref_len,
                "substitutions": sc.substitutions,
                "deletions": sc.deletions,
                "insertions": sc.insertions,
                "infer_ms": round(infer * 1000, 1),
                "error": err,
            }
        )

    session.close()
    model.close()

    # Corpus CER is character-weighted (total edits / total reference characters),
    # which is the standard definition and prevents a 1-character utterance from
    # carrying the same weight as a 30-character one.
    def corpus_from(field: str) -> float:
        num = sum((u[field] or 0.0) * u["ref_len"] for u in per_utt)
        den = sum(u["ref_len"] for u in per_utt)
        return 100.0 * num / den if den else 0.0

    return {
        "model": key,
        "arch": arch,
        "load_ms": round(load_ms, 1),
        "n_utts": len(per_utt),
        "audio_min": round(total_audio / 60.0, 2),
        "rtf": round(total_infer / max(1e-6, total_audio), 4),
        "corpus_cer_pct": round(corpus_from("cer"), 3),
        "corpus_cer_itn_pct": round(corpus_from("cer_itn"), 3),
        "mean_utt_cer_pct": round(100.0 * sum((u["cer"] or 0.0) for u in per_utt) / len(per_utt), 3),
        "total_substitutions": sum(u["substitutions"] for u in per_utt),
        "total_deletions": sum(u["deletions"] for u in per_utt),
        "total_insertions": sum(u["insertions"] for u in per_utt),
        "errors": sum(1 for u in per_utt if u["error"]),
        "worst": sorted(per_utt, key=lambda u: -(u["cer"] or 0.0))[:15],
        "per_utt": per_utt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="0 = all utterances")
    ap.add_argument("--models", default="", help="comma-separated model keys")
    ap.add_argument("--manifest", default="", help="alternative manifest path (e.g. utts_verbatim.jsonl)")
    ap.add_argument("--data-dir", default="", help="directory the manifest's wav paths are relative to")
    ap.add_argument("--out", default="report/ja_offline_bench.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA
    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.data_dir:
        manifest_path = data_dir / "utts.jsonl"
    else:
        manifest_path = None

    manifest = load_manifest(args.limit, manifest_path)
    global _ACTIVE_DATA
    _ACTIVE_DATA = data_dir
    keys = [k.strip() for k in args.models.split(",") if k.strip()] or list(CANDIDATES)
    print(f"Evaluating {len(manifest)} utterances ({sum(r['duration_sec'] for r in manifest)/60:.1f} min)")

    results = []
    for key in keys:
        spec = CANDIDATES.get(key)
        if spec is None:
            print(f"unknown model {key}")
            continue
        print(f"\n=== {key} ===")
        r = bench_model(key, spec, manifest)
        results.append(r)
        if "error" in r:
            print("  ", r["error"])
            continue
        print(
            f"  CER={r['corpus_cer_pct']:.2f}%  CER(ITN)={r['corpus_cer_itn_pct']:.2f}%  "
            f"RTF={r['rtf']:.3f}  load={r['load_ms']:.0f}ms  "
            f"S/D/I={r['total_substitutions']}/{r['total_deletions']}/{r['total_insertions']}"
        )

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_utts": len(manifest), "results": results}, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*76}\nSUMMARY  (corpus CER %, character-weighted)\n{'='*76}")
    print(f"{'model':<24}{'CER':>9}{'CER(ITN)':>11}{'RTF':>8}{'S':>7}{'D':>6}{'I':>6}")
    for r in sorted([x for x in results if "error" not in x], key=lambda x: x["corpus_cer_pct"]):
        print(
            f"{r['model']:<24}{r['corpus_cer_pct']:>8.2f}%{r['corpus_cer_itn_pct']:>10.2f}%"
            f"{r['rtf']:>8.3f}{r['total_substitutions']:>7}{r['total_deletions']:>6}{r['total_insertions']:>6}"
        )
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
