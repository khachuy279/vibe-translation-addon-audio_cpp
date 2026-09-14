"""Attribute the Japanese streaming accuracy loss between VAD and the commit controller.

`streaming_wer_bench` shows a large gap between offline accuracy and streaming accuracy
(e.g. cohere: 2.88% offline on isolated utterances vs 12.11% through the pipeline). This
splits that gap into its two structural causes, without real-time pacing so it runs fast:

  A. WHOLE-FILE decode      -- upper bound; the model sees all audio with full context
  B. VAD-SEGMENTED decode   -- same model, but on the exact chunks VAD hands to the engine
                               (includes production pre-roll/hangover)
  C. STREAMING commit text  -- taken from a previous streaming run's report, if given

A -> B isolates VAD segmentation (audio cut, context split, fragments).
B -> C isolates the commit controller (stability splits, dedupe, drop filters).

    python -m benchmarks.ja_stream_attribution --dataset data/ja_cv/streams --model cohere-transcribe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

import transcribe_cpp
from backend_cpp.config import config
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.dataset import DatasetPair, discover_dataset
from benchmarks.ja_text import score_ja

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "backend_cpp" / "models"
FILES = {
    "qwen3-asr-1.7b": MODELS / "Qwen3-ASR-1.7B-Q8_0.gguf",
    "cohere-transcribe": MODELS / "cohere-transcribe-03-2026-Q8_0.gguf",
    "kotoba-whisper-v2.2": MODELS / "kotoba-whisper-v2.2-BF16.gguf",
}


def load_wav(path: Path) -> np.ndarray:
    d, sr = sf.read(str(path), dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != 16000:
        import soxr

        d = soxr.resample(d, in_rate=sr, out_rate=16000, quality="HQ")
    return np.clip(d, -1.0, 1.0).astype(np.float32)


def vad_segments(pcm: np.ndarray) -> List[np.ndarray]:
    """Run the production VAD over the audio and return the chunks it would feed the engine."""
    int16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    segs: List[np.ndarray] = []
    cur: List[bytes] = []

    def on_chunk(b: bytes, *a) -> None:
        cur.append(b)

    def on_end() -> None:
        if cur:
            data = b"".join(cur)
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(arr) >= 1600:
                segs.append(arr)
            cur.clear()

    vad = VADProcessor(
        sample_rate=16000,
        vad_engine=config.vad.vad_engine,
        threshold=config.vad.threshold,
        silence_duration_ms=config.vad.silence_duration_ms,
        hangover_ms=config.vad.hangover_ms,
        pre_speech_buffer_ms=config.vad.pre_speech_buffer_ms,
        enabled=True,
        on_speech_chunk=on_chunk,
        on_speech_start=lambda: None,
        on_speech_end=on_end,
    )
    step = 1024 * 2
    for i in range(0, len(int16), step):
        vad.feed_chunk(int16[i : i + step], capture_timestamp=i / 32000.0)
    vad.force_end()
    on_end()
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/ja_cv/streams")
    ap.add_argument("--model", default="cohere-transcribe")
    ap.add_argument("--streaming-report", default="", help="streaming_wer_bench JSON for stage C")
    ap.add_argument("--out", default="report/ja_stream_attribution.json")
    args = ap.parse_args()

    model = transcribe_cpp.Model(str(FILES[args.model]))
    session = model.session()

    dataset: List[DatasetPair] = discover_dataset(args.dataset)
    stream_report = {}
    if args.streaming_report and Path(args.streaming_report).exists():
        sd = json.loads(Path(args.streaming_report).read_text(encoding="utf-8"))
        stream_report = {r["pair_id"]: r for r in sd["results"]}

    totals = {"full_edits": 0, "vad_edits": 0, "stream_edits": 0, "ref": 0}
    rows = []
    for pair in dataset:
        pcm = load_wav(pair.wav_path)
        dur = len(pcm) / 16000.0

        # A. whole-file decode, single call (both candidate models accept >60 s), so the
        #    model has full bidirectional context and no artificial cut.
        try:
            hyp_full = (session.run(pcm, language="ja").text or "").strip()
        except Exception as e:
            partial = getattr(e, "partial_result", None)
            hyp_full = (getattr(partial, "text", "") or "").strip() if partial is not None else ""

        # B. VAD-segmented decode on exactly the chunks the engine receives
        segs = vad_segments(pcm)
        seg_texts = []
        for seg in segs:
            try:
                seg_texts.append((session.run(seg, language="ja").text or "").strip())
            except Exception as e:
                partial = getattr(e, "partial_result", None)
                if partial is not None and getattr(partial, "text", ""):
                    seg_texts.append(partial.text.strip())
        hyp_vad = "".join(seg_texts)

        speech_sec = sum(len(s) for s in segs) / 16000.0

        def edits_of(hyp: str) -> int:
            sc = score_ja(pair.raw_reference, hyp)
            return sc.substitutions + sc.deletions + sc.insertions

        scorer = score_ja(pair.raw_reference, hyp_full)
        ref_chars = scorer.ref_len
        e_full = edits_of(hyp_full)
        e_vad = edits_of(hyp_vad)
        e_stream = None
        if pair.pair_id in stream_report:
            e_stream = edits_of(stream_report[pair.pair_id]["raw_hypothesis"])

        totals["full_edits"] += e_full
        totals["vad_edits"] += e_vad
        totals["ref"] += ref_chars
        if e_stream is not None:
            totals["stream_edits"] += e_stream

        rows.append(
            {
                "pair_id": pair.pair_id,
                "audio_sec": round(dur, 2),
                "vad_speech_sec": round(speech_sec, 2),
                "speech_coverage_pct": round(100.0 * speech_sec / max(1e-6, dur), 1),
                "vad_segment_count": len(segs),
                "cer_full_pct": round(100.0 * e_full / max(1, ref_chars), 2),
                "cer_vad_pct": round(100.0 * e_vad / max(1, ref_chars), 2),
                "cer_stream_pct": (round(100.0 * e_stream / max(1, ref_chars), 2) if e_stream is not None else None),
                "ref": pair.raw_reference[:200],
                "hyp_full": hyp_full[:200],
                "hyp_vad": hyp_vad[:200],
            }
        )
        print(
            f"{pair.pair_id}: full={rows[-1]['cer_full_pct']:6.2f}%  vad={rows[-1]['cer_vad_pct']:6.2f}%  "
            f"stream={rows[-1]['cer_stream_pct']}  coverage={rows[-1]['speech_coverage_pct']}%  segs={len(segs)}"
        )

    ref = max(1, totals["ref"])
    print(f"\n=== {args.model} ===")
    print(f"A. whole-file decode    : {100.0*totals['full_edits']/ref:6.2f}%")
    print(f"B. VAD-segmented decode : {100.0*totals['vad_edits']/ref:6.2f}%   (VAD costs {100.0*(totals['vad_edits']-totals['full_edits'])/ref:+.2f} pp)")
    if stream_report:
        print(f"C. streaming pipeline   : {100.0*totals['stream_edits']/ref:6.2f}%   (controller costs {100.0*(totals['stream_edits']-totals['vad_edits'])/ref:+.2f} pp)")

    out = REPO / args.out
    out.write_text(
        json.dumps({"model": args.model, "totals": totals, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"-> {out}")

    session.close()
    model.close()


if __name__ == "__main__":
    main()
