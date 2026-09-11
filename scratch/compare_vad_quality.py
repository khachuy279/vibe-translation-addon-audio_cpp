"""Compare VAD engines on REAL speech: cost vs segmentation vs transcripts.

Context (plan section 0.3): VAD is the largest CPU stage in the pipeline, and the CURRENT
default (`firered-vad`) is the MOST expensive engine:

    silero-vad   13.5 ms/audio-sec   (x1.00)
    fsmn-vad     55.0 ms/audio-sec   (x4.07)
    firered-vad 108.5 ms/audio-sec   (x8.04)   <- current default

Cost alone is not a reason to switch: a VAD that cuts speech in different places can change
the transcripts. This script runs the real-audio end-to-end scenario once per engine and
compares the FINAL transcripts, the delivery counts and the segmentation.

The reference audio is the Harvard sentence corpus, so the expected text is known and a
word-level diff is meaningful.

Usage (from the project root):
    python scratch/compare_vad_quality.py                 # firered-vad, fsmn-vad, silero-vad
    python scratch/compare_vad_quality.py fsmn-vad firered-vad
"""

import difflib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINES = ["firered-vad", "fsmn-vad", "silero-vad"]


def run_engine(engine: str) -> dict:
    """Run the real-audio scenario with one VAD engine and return its JSON section."""
    # The harness MERGES into an existing file, so the path must not pre-exist as a
    # zero-byte file (that makes it try to parse "" as JSON and silently write nothing).
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = Path(tmp.name)
    out.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "backend_cpp/run_perf_test.py",
        "--mode", "real-audio",
        "--vad-engine", engine,
        "--baseline-out", str(out),
    ]
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        print(f"  !! run failed for {engine}: {proc.stderr[-400:]}")

    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)
    return data["end_to_end_scenarios"]["real_audio"]


def word_diff(a: str, b: str) -> str:
    a_words, b_words = a.split(), b.split()
    if a_words == b_words:
        return "identical"
    sm = difflib.SequenceMatcher(None, a_words, b_words)
    ops = [
        f"{tag}:{'+' if tag != 'delete' else ''}{len(b_words[j1:j2])}{'-' if tag != 'insert' else ''}{len(a_words[i1:i2])}"
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal"
    ]
    return f"differs ({', '.join(ops)})" if ops else "differs"


def main() -> int:
    engines = sys.argv[1:] or DEFAULT_ENGINES

    # VAD cost comes from the micro-benchmark already stored in the baseline.
    baseline = json.loads((ROOT / "report" / "baseline_perf.json").read_text(encoding="utf-8"))
    cost = {
        row["engine"]: row
        for row in baseline["micro_benchmarks"]["vad_engine_tradeoff"]["rows"]
    }

    print(f"Running the real-audio scenario for: {', '.join(engines)}\n")
    results = {}
    for engine in engines:
        print(f"  ... {engine}")
        results[engine] = run_engine(engine)
    print()

    header = f"{'engine':14s} {'ms/audio-s':>11s} {'segs':>5s} {'speech%':>8s} "
    header += f"{'utt':>4s} {'prev':>5s} {'transl':>7s} {'infer_ms':>9s}"
    print(header)
    print("-" * len(header))
    for engine in engines:
        data = results[engine]
        row = cost.get(engine, {})
        counters = data["counters"]
        infer = data["asr"]["infer_ms"]
        print(
            f"{engine:14s} {row.get('ms_per_audio_sec', float('nan')):>11.1f} "
            f"{row.get('segments', 0):>5d} {row.get('speech_ratio', 0) * 100:>7.1f}% "
            f"{data['delivery']['final_utterances']:>4d} "
            f"{data['delivery']['preview_updates']:>5d} "
            f"{data['delivery']['translations']:>7d} "
            f"{counters.get('asr.total_inferences', 0) * infer['avg']:>9.0f}"
        )

    reference = engines[0]
    ref_texts = [t["text"] for t in results[reference]["transcripts"]]

    print(f"\nReference transcripts (from {reference}):")
    for idx, text in enumerate(ref_texts, 1):
        print(f"  {idx}. {text}")

    print("\nTranscript comparison against", reference)
    print("-" * 70)
    verdicts = {}
    for engine in engines[1:]:
        texts = [t["text"] for t in results[engine]["transcripts"]]
        if texts == ref_texts:
            print(f"  {engine:14s} IDENTICAL ({len(texts)} utterances)")
            verdicts[engine] = True
            continue

        print(f"  {engine:14s} DIFFERENT")
        verdicts[engine] = False
        for idx in range(max(len(ref_texts), len(texts))):
            a = ref_texts[idx] if idx < len(ref_texts) else "<missing>"
            b = texts[idx] if idx < len(texts) else "<missing>"
            if a != b:
                print(f"    [{idx + 1}] {reference}: {a!r}")
                print(f"    [{idx + 1}] {engine:14s}: {b!r}  -> {word_diff(a, b)}")

    print()
    if all(verdicts.values()):
        print("All engines produced identical final transcripts on this audio.")
    else:
        print("At least one engine changed the final transcripts -- inspect before switching.")
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
