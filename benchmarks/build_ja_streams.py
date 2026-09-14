"""Build continuous streaming test material from any Japanese utterance manifest.

Why: `streaming_wer_bench` scores the concatenated committed text against the
concatenated references. ReazonSpeech references are TV captions, so they omit real
speech and make "emit more real speech" look like an error — a bad benchmark for
deciding whether to *stop dropping* short utterances.

Common Voice is verbatim, so streams built from it give an absolute, trustworthy
streaming CER. The audio is read speech rather than drama, which is the trade-off;
the user's own captured session remains the domain check.

    python -m benchmarks.build_ja_streams --src data/ja_cv/utts.jsonl --out data/ja_cv --streams 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
TARGET_SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="manifest with id/wav/text/duration_sec")
    ap.add_argument("--out", required=True, help="directory to write streams/ into")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--stream-sec", type=float, default=60.0)
    ap.add_argument("--gap-sec", type=float, default=0.35)
    ap.add_argument("--start", type=int, default=0, help="skip the first N utterances (e.g. ones used by another set)")
    args = ap.parse_args()

    src_dir = Path(args.src).parent
    out_dir = Path(args.out)
    streams_dir = out_dir / "streams"
    streams_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.src, encoding="utf-8") if l.strip()][args.start:]
    gap = np.zeros(int(TARGET_SR * args.gap_sec), dtype=np.float32)

    made = []
    cur: list[np.ndarray] = []
    cur_text: list[str] = []
    cur_sec = 0.0
    si = 0
    # Ground-truth boundary bookkeeping: the gaps we insert ARE the true sentence
    # boundaries, so they give an objective target to score any segmenter against.
    gap_spans: list[tuple[float, float]] = []
    sent_spans: list[tuple[float, float]] = []
    pos = 0.0
    for r in rows:
        if si >= args.streams:
            break
        pcm, sr = sf.read(str(src_dir / r["wav"]), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        if sr != TARGET_SR:
            import soxr

            pcm = soxr.resample(pcm, in_rate=sr, out_rate=TARGET_SR, quality="HQ")
        pcm = np.clip(pcm, -1.0, 1.0).astype(np.float32)
        dur = len(pcm) / TARGET_SR

        if cur_sec > 0:
            cur.append(gap)
            gap_spans.append((pos, pos + args.gap_sec))
            pos += args.gap_sec
            cur_sec += args.gap_sec
        cur.append(pcm)
        sent_spans.append((pos, pos + dur))
        pos += dur
        cur_text.append(r["text"])
        cur_sec += dur

        if cur_sec >= args.stream_sec:
            sid = f"stream{si:02d}"
            sf.write(str(streams_dir / f"{sid}.wav"), np.concatenate(cur), TARGET_SR, subtype="PCM_16")
            (streams_dir / f"{sid}.txt").write_text("".join(cur_text), encoding="utf-8")
            made.append(
                {
                    "id": sid,
                    "wav": f"streams/{sid}.wav",
                    "duration_sec": round(cur_sec, 2),
                    "n_utts": len(cur_text),
                    "true_gaps_sec": [[round(a, 3), round(b, 3)] for a, b in gap_spans],
                    "true_sentences_sec": [[round(a, 3), round(b, 3)] for a, b in sent_spans],
                    "reference_texts": cur_text,
                }
            )
            print(f"  {sid}: {cur_sec:.1f}s, {len(cur_text)} utterances, {len(gap_spans)} true boundaries")
            si += 1
            cur, cur_text, cur_sec = [], [], 0.0
            gap_spans, sent_spans, pos = [], [], 0.0

    with open(out_dir / "streams.jsonl", "w", encoding="utf-8") as f:
        for r in made:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nbuilt {len(made)} streams + ground-truth boundaries -> {streams_dir}")


if __name__ == "__main__":
    main()
