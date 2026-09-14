"""Exactly which samples does the VAD hand to the model, and is it reproducible?

Two questions raised by `pipeline_transparency_test`:

  1. The VAD path delivered 40.20 s in one run and 34.68 s in another although nothing that
     affects audio length was changed between them. Is the segmenter deterministic?
  2. Bypassing VAD made CER identical to the direct model (17.53% vs 17.53%), while the VAD
     path cost +2.06 pp. Where does VAD remove or shift audio?

Runs the VAD alone (CPU only, no model inference) and reports the fed spans against the file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend_cpp.config import config  # noqa: E402
from backend_cpp.vad.vad_processor import VADProcessor  # noqa: E402

CV = REPO / "data" / "ja_cv"
N = 10


def fed_spans(pcm: np.ndarray, *, silence_ms: int, pre_speech_ms: int, hangover_ms: int,
              threshold: float) -> Tuple[List[Tuple[float, float, int]], float]:
    """(span_start_sec, span_end_sec, vad_state) for every chunk handed to the engine."""
    int16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    spans: List[Tuple[float, float, int]] = []

    def on_chunk(b: bytes, ts: float = 0.0, state: int = 1, *a) -> None:
        spans.append((ts, ts + len(b) / 32000.0, state))

    vad = VADProcessor(
        sample_rate=16000,
        vad_engine=config.vad.vad_engine,
        threshold=threshold,
        silence_duration_ms=silence_ms,
        hangover_ms=hangover_ms,
        pre_speech_buffer_ms=pre_speech_ms,
        enabled=True,
        on_speech_chunk=on_chunk,
        on_speech_start=lambda: None,
        on_speech_end=lambda: None,
    )
    step = 1024 * 2
    for i in range(0, len(int16), step):
        vad.feed_chunk(int16[i : i + step], capture_timestamp=i / 32000.0)
    vad.force_end()
    total = sum(e - s for s, e, _st in spans)
    return spans, total


def main() -> None:
    rows = [json.loads(l) for l in open(CV / "utts.jsonl", encoding="utf-8") if l.strip()]
    items = [r for r in rows if 1.2 <= r["duration_sec"] <= 8.0][:N]

    print(f"VAD: engine={config.vad.vad_engine} threshold={config.vad.threshold} "
          f"silence={config.vad.silence_duration_ms} pre_speech={config.vad.pre_speech_buffer_ms}\n")

    print("--- neutered VAD (silence=60000ms), two passes to test determinism ---")
    print(f"{'id':<9}{'file_s':>8}{'first_fed':>11}{'lead_trim':>11}{'fed_s':>9}   pass1==pass2")
    for r in items:
        pcm, sr = sf.read(str(CV / r["wav"]), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        pcm = pcm.astype(np.float32)

        a, ta = fed_spans(pcm, silence_ms=60000, pre_speech_ms=config.vad.pre_speech_buffer_ms,
                          hangover_ms=config.vad.hangover_ms, threshold=config.vad.threshold)
        b, tb = fed_spans(pcm, silence_ms=60000, pre_speech_ms=config.vad.pre_speech_buffer_ms,
                          hangover_ms=config.vad.hangover_ms, threshold=config.vad.threshold)
        first = min((s for s, _e, _st in a), default=float("nan"))
        dur = len(pcm) / 16000.0
        print(f"{r['id']:<9}{dur:>8.2f}{first:>11.3f}{first:>11.3f}{ta:>9.2f}   {a == b}")
        _ = tb

    print("\n--- pre-speech buffer sweep (lead audio recovered), silence=60000ms ---")
    print(f"{'id':<9}{'file_s':>8}" + "".join(f"{f'pre={p}ms':>12}" for p in (0, 200, 800, 1500, 3000)))
    for r in items:
        pcm, _sr = sf.read(str(CV / r["wav"]), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        pcm = pcm.astype(np.float32)
        dur = len(pcm) / 16000.0
        cells = []
        for p in (0, 200, 800, 1500, 3000):
            spans, _t = fed_spans(pcm, silence_ms=60000, pre_speech_ms=p,
                                  hangover_ms=config.vad.hangover_ms, threshold=config.vad.threshold)
            first = min((s for s, _e, _st in spans), default=float("nan"))
            cells.append(f"{first:>11.3f}s")
        print(f"{r['id']:<9}{dur:>8.2f}" + "".join(cells))

    print("\n--- NON-SPEECH frames inside an utterance (silence=60000 means no cut can occur) ---")
    for r in items:
        pcm, _sr = sf.read(str(CV / r["wav"]), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        spans, _t = fed_spans(pcm.astype(np.float32), silence_ms=60000,
                              pre_speech_ms=config.vad.pre_speech_buffer_ms,
                              hangover_ms=config.vad.hangover_ms, threshold=config.vad.threshold)
        states = [st for _s, _e, st in spans]
        speech = sum(1 for st in states if st == 1)
        nonspeech = len(states) - speech
        print(f"  {r['id']}: chunks={len(spans)} speech={speech} non_speech={nonspeech}")


if __name__ == "__main__":
    main()
