"""Sanity-check the ground-truth gap offsets against the actual waveform and the VAD onsets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
DATA = REPO / "data" / "ja_cv"

from benchmarks.vad_boundary_audit import load_wav, speech_onsets  # noqa: E402

st = json.loads(open(DATA / "streams.jsonl", encoding="utf-8").readline())
pcm = load_wav(DATA / st["wav"])
dur = len(pcm) / 16000.0
print(f"{st['id']}: {dur:.2f}s   declared duration {st['duration_sec']}s")
print(f"true gaps  : {st['true_gaps_sec']}")
print(f"true sents : {[ (round(a,2), round(b,2)) for a,b in st['true_sentences_sec'] ]}")

onsets, ends = speech_onsets(
    pcm, engine="fsmn-vad", threshold=0.2, silence_ms=150, hangover_ms=250, pre_speech_ms=800
)
print(f"\nVAD onsets : {[round(o,2) for o in onsets]}")
print(f"VAD ends   : {[round(e,2) for e in ends]}")


def rms_db(a: float, b: float) -> float:
    s = pcm[int(max(0, a) * 16000): int(min(dur, b) * 16000)]
    if len(s) == 0:
        return float("nan")
    r = float(np.sqrt(np.mean(s ** 2) + 1e-12))
    return 20 * np.log10(max(r, 1e-9))


print("\nEnergy at declared TRUE GAPS (should be low):")
for a, b in st["true_gaps_sec"][:6]:
    print(f"   [{a:.2f},{b:.2f}]  {rms_db(a, b):6.1f} dBFS")

print("\nEnergy at declared SENTENCE CENTRES (should be high):")
for a, b in st["true_sentence_sec"][:6]:
    mid = (a + b) / 2
    print(f"   [{mid-0.15:.2f},{mid+0.15:.2f}]  {rms_db(mid - 0.15, mid + 0.15):6.1f} dBFS")

print("\nEnergy in the 0.4s window starting at each VAD onset (should be speech):")
for o in onsets[:6]:
    print(f"   onset {o:.2f}: {rms_db(o, o + 0.4):6.1f} dBFS")
