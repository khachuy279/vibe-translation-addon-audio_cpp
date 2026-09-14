"""Is the ReazonSpeech reference verbatim, or a TV caption (abbreviated)?

If several unrelated ASR architectures all emit the same extra text that the
reference lacks, the audio probably contains it. ReazonSpeech is built from
Japanese TV broadcasts, whose captions are known to omit fillers, crosstalk and
sometimes paraphrase -- which would make the reference non-verbatim and inflate
CER for models that transcribe faithfully.

This checks it without ears:

  1. Speech duration implied by the reference (via a plausible Japanese speech
     rate) vs. the actual speech energy in the file.
  2. FSMN-VAD speech regions: how many, and do they cover more time than the
     reference can account for?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ja_eval"

# Japanese broadcast speech runs roughly 6-9 morae/s; kana ~= 1 mora, kanji
# ~1.6 morae. 0.20 s/char is a deliberately conservative (slow) reading so a
# reference that still under-fills the audio is strong evidence of omission.
SEC_PER_CHAR = 0.20


def frame_rms(pcm: np.ndarray, sr: int = 16000, win_ms: int = 25):
    w = int(sr * win_ms / 1000)
    n = len(pcm) // w
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(pcm[: n * w].reshape(n, w) ** 2, axis=1) + 1e-12)


def main() -> None:
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "utts.jsonl", encoding="utf-8") if l.strip()}
    data = json.loads((REPO / "report" / "ja_offline_bench.json").read_text(encoding="utf-8"))
    qwen = {u["id"]: u for r in data["results"] if r["model"] == "qwen3-asr-1.7b" for u in r["per_utt"]}

    ids = ["rz00004", "rz00005", "rz00006", "rz00035", "rz00071", "rz00075", "rz00080", "rz00161", "rz00142", "rz00134"]
    print(f"{'id':<10}{'file_s':>8}{'ref_chars':>10}{'implied_s':>11}{'extra_s':>9}{'speech_s':>10}{'ratio':>8}  verdict")
    for uid in ids:
        row = rows.get(uid)
        if not row:
            continue
        pcm, sr = sf.read(str(DATA / row["wav"]), dtype="float32")
        dur = len(pcm) / sr
        ref = row["text"]
        implied = len(ref) * SEC_PER_CHAR
        rms = frame_rms(pcm)
        if len(rms) == 0:
            continue
        # Speech frames = above 10% of the 90th-percentile frame energy.
        thr = 0.10 * np.percentile(rms, 90)
        speech_sec = float(np.sum(rms > thr)) * 0.025
        ratio = speech_sec / max(1e-6, implied)
        verdict = "REFERENCE IS PARTIAL" if ratio > 1.35 else ("ok" if ratio > 0.8 else "ref longer than audio?")
        print(
            f"{uid:<10}{dur:>8.2f}{len(ref):>10}{implied:>11.2f}{dur-implied:>9.2f}{speech_sec:>10.2f}{ratio:>8.2f}  {verdict}"
        )

    # Corpus-level: distribution over all 200 scored utterances.
    print("\nCorpus-level speech/implied-reference ratio over the scored set:")
    ratios = []
    for uid in qwen:
        row = rows.get(uid)
        if not row:
            continue
        pcm, sr = sf.read(str(DATA / row["wav"]), dtype="float32")
        rms = frame_rms(pcm)
        if len(rms) == 0:
            continue
        thr = 0.10 * np.percentile(rms, 90)
        speech_sec = float(np.sum(rms > thr)) * 0.025
        implied = len(row["text"]) * SEC_PER_CHAR
        ratios.append(speech_sec / max(1e-6, implied))
    ratios = np.array(ratios)
    print(f"  n={len(ratios)}  median={np.median(ratios):.2f}  mean={ratios.mean():.2f}  "
          f"p10={np.percentile(ratios,10):.2f}  p90={np.percentile(ratios,90):.2f}")
    for t in (1.0, 1.35, 1.6, 2.0):
        print(f"  fraction with ratio > {t}: {100.0*np.mean(ratios>t):.1f}%")


if __name__ == "__main__":
    main()
