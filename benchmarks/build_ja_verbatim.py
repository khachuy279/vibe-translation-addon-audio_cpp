"""Build a *verbatim-verified* Japanese subset from the ReazonSpeech eval set.

Why this exists
---------------
`japanese-asr/ja_asr.reazonspeech_test` is derived from Japanese TV broadcasts, and
its `transcription` field is the broadcast **caption**, not a verbatim transcript.
Captions routinely drop back-channel, crosstalk, fillers and sometimes paraphrase.

Measured consequence on the 200-utterance screen: 45% of clips contain more speech
energy than the reference can account for, and 10% contain more than twice as much.
Several unrelated ASR architectures (qwen3, sensevoice, nemotron, voxtral, kotoba)
all emit the *same* unlabelled text on those clips, which means the audio really
contains it -- so a model that transcribes faithfully is scored as if it hallucinated.

Scoring models on the raw set therefore measures "how caption-like is the output",
not accuracy. This script keeps only clips where the reference plausibly accounts
for the speech in the file, producing a subset where CER means what it says.

    python -m benchmarks.build_ja_verbatim --min-ratio 0.80 --max-ratio 1.15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ja_eval"

# Japanese broadcast speech runs roughly 6-9 morae/s. 0.20 s per reference
# character is a slow (conservative) reading rate, so a clip whose speech energy
# still exceeds that much time is definitely under-labelled.
SEC_PER_CHAR = 0.20


def speech_seconds(pcm: np.ndarray, sr: int = 16000, win_ms: int = 25) -> float:
    w = int(sr * win_ms / 1000)
    n = len(pcm) // w
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(pcm[: n * w].reshape(n, w) ** 2, axis=1) + 1e-12)
    thr = 0.10 * np.percentile(rms, 90)
    return float(np.sum(rms > thr)) * (win_ms / 1000.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ratio", type=float, default=0.80)
    ap.add_argument("--max-ratio", type=float, default=1.15)
    ap.add_argument("--out", default="utts_verbatim.jsonl")
    args = ap.parse_args()

    src = DATA / "utts.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    kept = []
    for r in rows:
        pcm, sr = sf.read(str(DATA / r["wav"]), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        sp = speech_seconds(pcm.astype(np.float32), sr)
        implied = len(r["text"]) * SEC_PER_CHAR
        ratio = sp / max(1e-6, implied)
        if args.min_ratio <= ratio <= args.max_ratio:
            kept.append({**r, "speech_ratio": round(ratio, 3)})

    out = DATA / args.out
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"kept {len(kept)} / {len(rows)} utterances ({100.0*len(kept)/len(rows):.1f}%)")
    print(f"  band: {args.min_ratio} <= speech/implied-ref <= {args.max_ratio}")
    print(f"  audio: {sum(r['duration_sec'] for r in kept)/60:.1f} min")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
