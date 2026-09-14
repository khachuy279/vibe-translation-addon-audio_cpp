"""Score the segmenter against known ground-truth sentence boundaries.

The Common Voice streams are built by concatenating whole sentences with a 350 ms gap, so
the exact offsets of every true sentence boundary are known
(`streams.jsonl: true_gaps_sec`). That turns "does the segmenter cut in the right place?"
into an objective measurement instead of an inference from downstream CER.

What is measured
----------------
Every utterance the segmenter emits (after the first) implies a cut immediately before it.
The marker for "where the segmenter resumed speech" is the timestamp of that utterance's
first real SPEECH chunk -- pre-roll chunks carry an earlier timestamp and are ignored, and
`hangover_ms` only extends utterance *ends*, so neither can move the marker.

  * hit      -- the cut lands inside a true 350 ms gap (the segmenter broke where a sentence ended)
  * spurious -- the cut lands inside a sentence (this is what shreds words and produces the
                `インコイ`-style garbage / `はい。` hallucinations from the CER study)
  * missed   -- a true gap with no cut near it (two sentences merged into one decode)

    python -m benchmarks.vad_boundary_audit
    python -m benchmarks.vad_boundary_audit --sweep-silence 150,250,350,500,700,900
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

from backend_cpp.config import config
from backend_cpp.vad.vad_processor import VADProcessor

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ja_cv"
SPEECH_STATE = 1  # backend_cpp.asr.audio_buffer.VAD_STATE_SPEECH
TOL_SEC = 0.05


def load_streams() -> List[dict]:
    return [json.loads(l) for l in open(DATA / "streams.jsonl", encoding="utf-8") if l.strip()]


def load_wav(path: Path) -> np.ndarray:
    d, sr = sf.read(str(path), dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != 16000:
        import soxr

        d = soxr.resample(d, in_rate=sr, out_rate=16000, quality="HQ")
    return np.clip(d, -1.0, 1.0).astype(np.float32)


def speech_onsets(pcm: np.ndarray, *, engine: str, threshold: float, silence_ms: int,
                  hangover_ms: int, pre_speech_ms: int) -> Tuple[List[float], List[float]]:
    """Return (onset_seconds_per_utterance, utterance_end_seconds)."""
    int16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    onsets: List[float] = []
    ends: List[float] = []
    cur_first_speech: List[float] = []
    cur_last_speech_end: List[float] = []

    def on_chunk(b: bytes, ts: float = 0.0, state: int = SPEECH_STATE, *a) -> None:
        if state != SPEECH_STATE:
            return  # ignore pre-roll: it carries an earlier timestamp
        if not cur_first_speech:
            cur_first_speech.append(ts)
        cur_last_speech_end.append(ts + len(b) / 32000.0)

    def on_end() -> None:
        if cur_first_speech:
            onsets.append(cur_first_speech[0])
            ends.append(cur_last_speech_end[-1])
        cur_first_speech.clear()
        cur_last_speech_end.clear()

    vad = VADProcessor(
        sample_rate=16000,
        vad_engine=engine,
        threshold=threshold,
        silence_duration_ms=silence_ms,
        hangover_ms=hangover_ms,
        pre_speech_buffer_ms=pre_speech_ms,
        enabled=True,
        on_speech_chunk=on_chunk,
        on_speech_start=lambda: None,
        on_speech_end=on_end,
    )
    step = 1024 * 2
    for i in range(0, len(int16), step):
        vad.feed_chunk(int16[i : i + step], capture_timestamp=i / 32000.0)
    vad.force_end()
    on_end()
    return onsets, ends


def evaluate(streams: List[dict], **vad_cfg) -> Dict[str, float]:
    """Span-overlap segmentation quality.

    Onset *timestamps* are not used directly: the VAD reports an onset only once it has
    confirmed speech, which measured ~200 ms late, so timestamp matching would need an
    arbitrary tolerance. Span overlap avoids that entirely and answers the two questions
    that matter:

      * split    -- a true sentence covered by >= 2 utterances (over-segmentation: this is
                    what shreds words and produces fragment hallucination)
      * merged   -- an utterance whose span crosses a true gap (under-segmentation: two
                    turns decoded together)
    """
    sentences = utterances = 0
    split_sentences = extra_splits = merges = 0

    for st in streams:
        pcm = load_wav(DATA / st["wav"])
        _onsets, ends = speech_onsets(pcm, **vad_cfg)
        starts = _onsets
        spans = list(zip(starts, ends))
        sents = st["true_sentences_sec"]
        gaps = st["true_gaps_sec"]

        sentences += len(sents)
        utterances += len(spans)

        # Over-segmentation: how many utterances overlap each true sentence?
        for a, b in sents:
            covering = sum(1 for s, e in spans if e > a + 0.10 and s < b - 0.10)
            if covering >= 2:
                split_sentences += 1
                extra_splits += covering - 1

        # Under-segmentation: an utterance whose speech crosses a true gap.
        for s, e in spans:
            if any(s < a - 0.05 and e > b + 0.05 for a, b in gaps):
                merges += 1

    return {
        "true_sentences": sentences,
        "vad_utterances": utterances,
        "utterances_per_sentence": round(utterances / max(1, sentences), 2),
        "split_sentences": split_sentences,
        "split_sentence_pct": round(100 * split_sentences / max(1, sentences), 1),
        "extra_splits": extra_splits,
        "merged_utterances": merges,
        "merged_pct": round(100 * merges / max(1, utterances), 1),
    }


def silence_profile(streams: List[dict], **vad_cfg) -> Dict[str, float]:
    """How long was the quiet run that preceded each emitted utterance boundary?

    Measured as ``next_onset - previous_utterance_end``: the interval between the last
    speech frame of one utterance and the confirmed onset of the next. That is exactly the
    pause the segmenter chose to break on, and it needs no energy threshold guesswork.
    """
    true_runs: List[float] = []
    spurious_runs: List[float] = []

    for st in streams:
        pcm = load_wav(DATA / st["wav"])
        onsets, ends = speech_onsets(pcm, **vad_cfg)
        if len(onsets) < 2:
            continue
        for i in range(1, min(len(onsets), len(ends))):
            pause = onsets[i] - ends[i - 1]
            if any(g[0] - 0.25 <= onsets[i] <= g[1] + 0.45 for g in st["true_gaps_sec"]):
                true_runs.append(pause)
            else:
                spurious_runs.append(pause)

    def q(v, p):
        return round(float(np.percentile(v, p)), 3) if v else None

    return {
        "true_boundary_pause_median_sec": q(true_runs, 50),
        "true_boundary_pause_p10_sec": q(true_runs, 10),
        "spurious_cut_pause_median_sec": q(spurious_runs, 50),
        "spurious_cut_pause_p90_sec": q(spurious_runs, 90),
        "n_true": len(true_runs),
        "n_spurious": len(spurious_runs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=config.vad.vad_engine)
    ap.add_argument("--threshold", type=float, default=config.vad.threshold)
    ap.add_argument("--silence", type=int, default=config.vad.silence_duration_ms)
    ap.add_argument("--hangover", type=int, default=config.vad.hangover_ms)
    ap.add_argument("--pre-speech", type=int, default=config.vad.pre_speech_buffer_ms)
    ap.add_argument("--sweep-silence", default="")
    ap.add_argument("--sweep-hangover", default="")
    ap.add_argument("--out", default="report/vad_boundary_audit.json")
    args = ap.parse_args()

    streams = load_streams()
    print(f"{len(streams)} streams, {sum(len(s['true_gaps_sec']) for s in streams)} true sentence boundaries\n")

    results = []

    def show(label: str, **kw) -> None:
        r = evaluate(streams, **kw)
        r["silence_profile"] = silence_profile(streams, **kw)
        r["label"] = label
        r["config"] = kw
        results.append(r)
        sp = r["silence_profile"]
        print(
            f"{label:<24} utt/sent={r['utterances_per_sentence']:>5}  splits={r['split_sentences']:>3}/{r['true_sentences']:<3}"
            f" ({r['split_sentence_pct']:>5}%)  extra={r['extra_splits']:>3}  merged={r['merged_utterances']:>3}"
            f"  | pause true={sp['true_boundary_pause_median_sec']}s (n={sp['n_true']})"
            f" spurious={sp['spurious_cut_pause_median_sec']}s (n={sp['n_spurious']})"
        )

    base = dict(engine=args.engine, threshold=args.threshold, silence_ms=args.silence,
                hangover_ms=args.hangover, pre_speech_ms=args.pre_speech)
    show("production default", **base)

    if args.sweep_silence:
        print()
        for ms in [int(x) for x in args.sweep_silence.split(",")]:
            show(f"silence={ms}ms", **{**base, "silence_ms": ms})

    if args.sweep_hangover:
        print()
        for ms in [int(x) for x in args.sweep_hangover.split(",")]:
            show(f"hangover={ms}ms", **{**base, "hangover_ms": ms})

    out = REPO / args.out
    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
