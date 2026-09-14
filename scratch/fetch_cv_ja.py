"""Download the Common Voice Japanese (8.0) test split.

Common Voice is *verbatim by construction*: contributors read a prompt sentence, so
the transcript covers the whole clip. That makes it a metric-valid complement to the
ReazonSpeech set (which is caption-derived and therefore under-labelled).

151 MB, single parquet shard.

    python scratch/fetch_cv_ja.py
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ja_cv"
RAW = DATA / "raw"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ID = "japanese-asr/ja_asr.common_voice_8_0"
FILE = "data/test-00000-of-00001.parquet"
TARGET_SR = 16000


def main() -> None:
    from huggingface_hub import hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(repo_id=REPO_ID, filename=FILE, repo_type="dataset", local_dir=str(RAW))
    print(f"downloaded -> {p} ({Path(p).stat().st_size/1e6:.1f} MB)")

    pf = pq.ParquetFile(p)
    print("rows:", pf.metadata.num_rows)
    print(pf.schema_arrow)

    utts = DATA / "utts"
    utts.mkdir(parents=True, exist_ok=True)

    n = 0
    rows = []
    for batch in pf.iter_batches(batch_size=256):
        for rec in batch.to_pylist():
            audio = rec.get("audio") or rec.get("sentence_audio") or {}
            text = (rec.get("transcription") or rec.get("sentence") or "").strip()
            blob = audio.get("bytes") if isinstance(audio, dict) else None
            if not blob or not text:
                continue
            data, sr = sf.read(io.BytesIO(blob), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != TARGET_SR:
                import soxr

                data = soxr.resample(data, in_rate=sr, out_rate=TARGET_SR, quality="HQ")
            data = np.clip(data, -1.0, 1.0).astype(np.float32)
            dur = len(data) / TARGET_SR
            if dur < 0.6 or dur > 20.0:
                continue
            uid = f"cv{n:05d}"
            sf.write(str(utts / f"{uid}.wav"), data, TARGET_SR, subtype="PCM_16")
            rows.append({"id": uid, "wav": f"utts/{uid}.wav", "text": text, "duration_sec": round(dur, 3)})
            n += 1

    with open(DATA / "utts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {n} utterances ({sum(r['duration_sec'] for r in rows)/60:.1f} min) -> {DATA/'utts.jsonl'}")


if __name__ == "__main__":
    main()
