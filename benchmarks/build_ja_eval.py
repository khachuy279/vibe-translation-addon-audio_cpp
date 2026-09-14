"""Build the Japanese ASR evaluation set from the downloaded ReazonSpeech test parquet.

Outputs (all under `data/ja_eval/`, gitignored):

  utts.jsonl        one JSON object per utterance: id, wav, text, duration_sec
  utts/<id>.wav     16 kHz mono PCM16
  streams.jsonl     continuous ~60 s concatenations for streaming-path scoring
  streams/<id>.wav
  streams/<id>.txt  reference = concatenated utterance transcriptions

Streams exist because the production pipeline is a *streaming* system: VAD decides
where utterances start and end, and the commit path slices audio. Scoring isolated
utterances would never exercise that.

    python -m benchmarks.build_ja_eval --limit 1000 --streams 8
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ja_eval"
RAW = DATA / "raw"
TARGET_SR = 16000


def _decode(blob: bytes) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        import soxr

        data = soxr.resample(data, in_rate=sr, out_rate=TARGET_SR, quality="HQ")
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def load_all() -> list[dict]:
    rows: list[dict] = []
    for pq_path in sorted(RAW.glob("data/*.parquet")):
        pf = pq.ParquetFile(pq_path)
        for batch in pf.iter_batches(batch_size=256, columns=["audio", "transcription"]):
            for rec in batch.to_pylist():
                rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000, help="number of utterances to extract")
    ap.add_argument("--streams", type=int, default=8, help="number of continuous streams to build")
    ap.add_argument("--stream-sec", type=float, default=60.0, help="target seconds per stream")
    ap.add_argument("--gap-sec", type=float, default=0.35, help="silence inserted between utterances")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    utts_dir = DATA / "utts"
    streams_dir = DATA / "streams"
    utts_dir.mkdir(parents=True, exist_ok=True)
    streams_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all()
    print(f"loaded {len(rows)} raw rows")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    kept: list[dict] = []
    for idx, rec in enumerate(rows):
        if len(kept) >= args.limit:
            break
        blob = (rec.get("audio") or {}).get("bytes")
        text = (rec.get("transcription") or "").strip()
        if not blob or not text:
            continue
        try:
            pcm = _decode(blob)
        except Exception:
            continue
        dur = len(pcm) / TARGET_SR
        if dur < 0.6 or dur > 14.0:
            continue
        uid = f"rz{idx:05d}"
        sf.write(str(utts_dir / f"{uid}.wav"), pcm, TARGET_SR, subtype="PCM_16")
        kept.append({"id": uid, "wav": f"utts/{uid}.wav", "text": text, "duration_sec": round(dur, 3)})

    with open(DATA / "utts.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_audio = sum(r["duration_sec"] for r in kept)
    print(f"kept {len(kept)} utterances, {total_audio/60:.1f} min audio, {total_audio/len(kept):.2f}s avg")

    # ---- continuous streams: consecutive utterances joined by a short silence ----
    gap = np.zeros(int(TARGET_SR * args.gap_sec), dtype=np.float32)
    stream_rows = []
    si = 0
    cur: list[np.ndarray] = []
    cur_text: list[str] = []
    cur_sec = 0.0
    for r in kept:
        pcm, _sr = sf.read(str(DATA / r["wav"]), dtype="float32")
        if cur_sec > 0:
            cur.append(gap)
            cur_sec += args.gap_sec
        cur.append(pcm)
        cur_text.append(r["text"])
        cur_sec += r["duration_sec"]
        if cur_sec >= args.stream_sec:
            sid = f"stream{si:02d}"
            sf.write(str(streams_dir / f"{sid}.wav"), np.concatenate(cur), TARGET_SR, subtype="PCM_16")
            ref = "".join(cur_text)
            (streams_dir / f"{sid}.txt").write_text(ref, encoding="utf-8")
            stream_rows.append({"id": sid, "wav": f"streams/{sid}.wav", "duration_sec": round(cur_sec, 2), "n_utts": len(cur_text)})
            print(f"  {sid}: {cur_sec:.1f}s, {len(cur_text)} utterances")
            si += 1
            cur, cur_text, cur_sec = [], [], 0.0
            if si >= args.streams:
                break

    with open(DATA / "streams.jsonl", "w", encoding="utf-8") as f:
        for r in stream_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nBuilt {len(kept)} utterances and {len(stream_rows)} streams under {DATA}")


if __name__ == "__main__":
    main()
