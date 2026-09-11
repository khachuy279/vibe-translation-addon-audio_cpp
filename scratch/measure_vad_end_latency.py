"""How much trailing-silence latency does the VAD actually add, and is it tunable?

Motivation: measured post-speech subtitle latency is ~610 ms, of which ~447 ms is translation
and only ~165 ms is everything between `[VAD END]` and the delivered subtitle. The unmeasured
term is *before* `[VAD END]`: the VAD has to decide the speaker stopped.

`VADProcessor.feed_chunk()` has two end paths (backend_cpp/vad/vad_processor.py):

  1. the ENGINE emits an END event  -> `on_speech_end` fires immediately, no added latency;
  2. the SAFETY NET (engine emits nothing) -> after `silence_elapsed_ms >= silence_duration_ms`
     -> `on_speech_end` fires, i.e. up to `silence_duration_ms` (450 ms by default) of added
     latency.

`hangover_ms` does NOT add latency: it is only a grace period
(`min(hangover_ms, silence_duration_ms * 0.5)`) during which silent frames are still attached to
the committed audio. So the tunable latency knob is `silence_duration_ms` alone -- and ONLY on
path 2.

This script replays a real WAV through VADProcessor at several settings and reports, per
utterance:

  * which end path fired,
  * how long after the true speech end the end event fired (the latency cost),
  * the committed audio duration vs the true speech duration (a truncation check -- cutting the
    end of a word is the risk of lowering the threshold).

Usage:  python scratch/measure_vad_end_latency.py [wav]
"""

import sys
import wave
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend_cpp.vad.vad_processor import VADProcessor  # noqa: E402

DEFAULT_WAV = "wav_test/OSR_us_000_0010_16k.wav"
CHUNK_MS = 30

# (silence_duration_ms, hangover_ms) -- current config first.
SETTINGS = [
    (450, 400),  # config.vad defaults
    (300, 400),
    (250, 250),
    (150, 150),
    (100, 100),
]


def load_wav_pcm16(path: Path) -> tuple:
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(
                f"{path.name} must be 16 kHz mono 16-bit PCM "
                f"(got {wf.getframerate()} Hz, {wf.getnchannels()}ch, {wf.getsampwidth() * 8}bit)"
            )
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16), wf.getframerate()


def true_speech_segments(samples: np.ndarray, sr: int, frame_ms: int = 10,
                         margin_db: float = 10.0, min_speech_ms: int = 120,
                         min_gap_ms: int = 150) -> list:
    """Independent energy-based speech segmentation, used only as a positional reference."""
    hop = sr * frame_ms // 1000
    n = len(samples) // hop
    frames = samples[:n * hop].reshape(n, hop).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms / 32768.0 + 1e-12)

    noise_floor = float(np.percentile(db, 10))
    is_speech = db > (noise_floor + margin_db)

    # Bridge short gaps, then drop short bursts.
    min_gap_frames = max(1, min_gap_ms // frame_ms)
    min_speech_frames = max(1, min_speech_ms // frame_ms)

    segments = []
    start = None
    gap = 0
    for idx, flag in enumerate(is_speech):
        if flag:
            if start is None:
                start = idx
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap_frames:
                if idx - gap - start >= min_speech_frames:
                    segments.append((start * frame_ms, (idx - gap) * frame_ms))
                start = None
    if start is not None and n - start >= min_speech_frames:
        segments.append((start * frame_ms, n * frame_ms))
    return segments


def replay(samples: np.ndarray, sr: int, engine: str, silence_ms: int, hangover_ms: int) -> dict:
    """Feed the whole file and capture VAD events with the audio position they fired at."""
    events = []
    committed_ms = {"value": 0.0}
    pos = {"ms": 0.0}

    def on_start():
        events.append(("start", pos["ms"]))

    def on_end():
        events.append(("end", pos["ms"]))

    def on_chunk(_payload, *_rest):
        # Approximate the committed audio length; only used for the truncation comparison.
        committed_ms["value"] += 0.0

    vad = VADProcessor(
        sample_rate=sr,
        vad_engine=engine,
        threshold=None,
        silence_duration_ms=silence_ms,
        hangover_ms=hangover_ms,
        pre_speech_buffer_ms=550,
        enabled=True,
        on_speech_chunk=on_chunk,
        on_speech_start=on_start,
        on_speech_end=on_end,
    )

    chunk = sr * CHUNK_MS // 1000
    pcm_bytes = samples.tobytes()
    bytes_per_chunk = chunk * 2
    speech_chunks_ms = 0.0

    for offset in range(0, len(pcm_bytes), bytes_per_chunk):
        piece = pcm_bytes[offset:offset + bytes_per_chunk]
        pos["ms"] = (offset + len(piece)) / 2.0 / sr * 1000.0
        before = len(events)
        vad.feed_chunk(piece, capture_timestamp=pos["ms"] / 1000.0)
        # Attribute newly emitted events to this chunk's end position.
        for idx in range(before, len(events)):
            events[idx] = (events[idx][0], pos["ms"])
        if vad.get_stats().get("is_speech"):
            speech_chunks_ms += CHUNK_MS

    return {"events": events, "speech_ms_in_utterances": speech_chunks_ms}


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or [ROOT / DEFAULT_WAV]

    for raw in paths:
        path = raw if raw.is_absolute() else ROOT / raw
        samples, sr = load_wav_pcm16(path)
        duration_ms = len(samples) / sr * 1000.0
        segments = true_speech_segments(samples, sr)

        print(f"\n{'=' * 78}")
        print(f"{path.name}  ({duration_ms / 1000:.1f}s)  energy-segments={len(segments)}")
        print(f"{'=' * 78}")
        header = f"{'engine':11s} {'silence':>7} {'hangover':>8} {'ends':>5} {'median_trail':>13}"
        print(header)
        print("-" * len(header))

        for silence_ms, hangover_ms in SETTINGS:
            for engine in ("fsmn-vad", "silero-vad"):
                result = replay(samples, sr, engine, silence_ms, hangover_ms)
                ends = [t for kind, t in result["events"] if kind == "end"]
                costs = []
                for t_end in ends:
                    candidates = [e for _s, e in segments if e <= t_end]
                    if candidates:
                        costs.append(t_end - candidates[-1])
                label = f"{engine:11s} {silence_ms:>7} {hangover_ms:>8} {len(ends):>5}"
                if costs:
                    print(f"{label} {median(costs):>10.0f} ms")
                else:
                    print(f"{label} {'n/a':>13}")
        print("\n  `ends` = number of utterances the VAD would commit. If a lower silence setting "
              "raises it, shorter natural pauses start splitting sentences (subtitle fragmentation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
