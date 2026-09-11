"""Inspect candidate reference audio files (format, duration) for the W2.1 baseline."""

import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    "wav_test/OSR_us_000_0010_16k.wav",
    "wav_test/OSR_us_000_0010_8k.wav",
    "wav_test/test01_20s.wav",
    "wav_test/test_e2e_audio.wav",
    "wav_test/two_speakers_sample.wav",
    "transcribe.cpp/samples/ja.wav",
    "transcribe.cpp/samples/jfk.wav",
    "transcribe.cpp/samples/ko.wav",
    "transcribe.cpp/samples/german.wav",
]

print(f"{'file':<44} {'ch':>3} {'rate':>7} {'bits':>5} {'sec':>8}")
print("-" * 72)
for rel in CANDIDATES:
    path = PROJECT_ROOT / rel
    if not path.exists():
        print(f"{rel:<44} (missing)")
        continue
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            print(
                f"{rel:<44} {wf.getnchannels():>3} {rate:>7} "
                f"{wf.getsampwidth() * 8:>5} {frames / rate:>8.2f}"
            )
    except Exception as exc:
        print(f"{rel:<44} ERROR {type(exc).__name__}: {exc}")
