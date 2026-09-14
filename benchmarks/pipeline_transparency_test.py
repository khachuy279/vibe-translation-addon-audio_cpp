"""Is the pipeline transparent? Direct model vs full pipeline with VAD neutered.

The controlled experiment
-------------------------
Take N short utterances with verbatim references and one model (qwen3-asr-1.7b).

  Run 1  audio -> model                       -> CER(1)   direct, no pipeline
  Run 2  audio -> real pipeline -> model      -> CER(2)   VAD silenced (60 s) so it
                                                          cannot cut, only the explicit
                                                          end-of-stream flush closes it

If CER(1) == CER(2) the pipeline is transparent and nothing is wrong with it.
If CER(1) << CER(2) then something in the pipeline is corrupting the audio or the text.

Run 2 walks the production objects exactly as a session does:
    VADProcessor -> TranscribeEngine.feed_audio -> AudioBufferManager
        -> SpeechNormalizer -> transcribe_cpp -> commit

To make a gap attributable rather than mysterious, Run 2 is repeated with one component
neutralised at a time:

    pipeline                 everything on (the claim under test)
    no_normalizer            asr.normalize_speech = False
    no_drop_filter           min_words_to_commit / min_words_to_emit_final = 1
    no_vad                   feed the engine directly, bypassing VADProcessor entirely

The script also verifies the audio actually delivered to the model, in seconds, against the
file duration -- a mismatch there is an audio-transport bug, not a model problem.

    python -m benchmarks.pipeline_transparency_test --n 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

import transcribe_cpp
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import config
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.ja_text import score_ja
from benchmarks.simulator import StreamingAudioSimulator

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "backend_cpp" / "models" / "Qwen3-ASR-1.7B-Q8_0.gguf"
CV = REPO / "data" / "ja_cv"

# VAD can never fire with this, so the only thing that can end an utterance is the flush.
VAD_NEUTERED_MS = 60_000
MAX_SEGMENT_SEC = 60.0


class TracedEngine(TranscribeEngine):
    """Records committed text and the exact audio handed to the model."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.commits: List[Dict[str, Any]] = []
        self.commit_audio_sec = 0.0

    def _emit_final(self, text, utt_id, reason, epoch=0, media_start_time=0.0, media_end_time=0.0):
        self.commits.append({"reason": reason, "text": (text or "").strip()})
        return super()._emit_final(text, utt_id, reason, epoch=epoch,
                                   media_start_time=media_start_time, media_end_time=media_end_time)

    def _run_inference(self, pcm_float32, frame_state=None, is_commit=False, utt_id=None):
        if is_commit:
            self.commit_audio_sec += len(pcm_float32) / 16000.0
        return super()._run_inference(pcm_float32, frame_state=frame_state,
                                      is_commit=is_commit, utt_id=utt_id)


def pick_utterances(n: int, min_sec: float = 1.2, max_sec: float = 8.0) -> List[dict]:
    rows = [json.loads(l) for l in open(CV / "utts.jsonl", encoding="utf-8") if l.strip()]
    out = [r for r in rows if min_sec <= r["duration_sec"] <= max_sec]
    return out[:n]


def corpus_cer(pairs: List[tuple]) -> float:
    """Character-weighted corpus CER over (reference, hypothesis) pairs."""
    edits = ref_len = 0
    for ref, hyp in pairs:
        s = score_ja(ref, hyp)
        edits += s.substitutions + s.deletions + s.insertions
        ref_len += s.ref_len
    return 100.0 * edits / max(1, ref_len)


def run_direct(model_path: Path, items: List[dict]) -> List[str]:
    model = transcribe_cpp.Model(str(model_path))
    session = model.session()
    session.run(np.zeros(16000, dtype=np.float32), language="ja")  # warm-up
    hyps = []
    for r in items:
        pcm, sr = sf.read(str(CV / r["wav"]), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        hyps.append((session.run(pcm.astype(np.float32), language="ja").text or "").strip())
    session.close()
    model.close()
    return hyps


async def run_pipeline(
    items: List[dict],
    *,
    use_vad: bool = True,
    normalize: bool = True,
    drop_filter: bool = True,
    speed: float = 1.0,
) -> Dict[str, Any]:
    """Drive one utterance at a time through the production objects.

    Returns per-utterance committed text plus the seconds of audio the model actually saw.
    """
    results = []
    was_norm = config.asr.normalize_speech
    config.asr.normalize_speech = normalize
    try:
        for r in items:
            engine = TracedEngine(session_id=f"transp_{r['id']}")
            engine.sentence_config.max_duration_sec = MAX_SEGMENT_SEC
            engine._segmenter.max_duration_sec = MAX_SEGMENT_SEC
            if not drop_filter:
                engine.sentence_config.min_words_to_commit = 1
                engine.sentence_config.min_words_to_emit_final = 1
                engine._segmenter.min_words_to_commit = 1
                engine._segmenter.min_words_to_emit_final = 1

            async def consume() -> None:
                try:
                    async for _ in engine.stream_tokens():
                        pass
                except asyncio.CancelledError:
                    pass

            consumer = asyncio.create_task(consume())

            sim = StreamingAudioSimulator(
                audio_source=CV / r["wav"], chunk_ms=64, speed=speed, trailing_silence_sec=0.5
            )
            vad: Optional[VADProcessor] = None
            if use_vad:
                vad = VADProcessor(
                    sample_rate=16000,
                    vad_engine=config.vad.vad_engine,
                    threshold=config.vad.threshold,
                    silence_duration_ms=VAD_NEUTERED_MS,
                    hangover_ms=config.vad.hangover_ms,
                    pre_speech_buffer_ms=config.vad.pre_speech_buffer_ms,
                    enabled=True,
                    on_speech_chunk=engine.feed_audio,
                    on_speech_start=engine.on_speech_start,
                    on_speech_end=engine.on_speech_end,
                )

            for chunk in sim.iter_chunks():
                if vad is not None:
                    vad.feed_chunk(chunk.pcm_bytes)
                else:
                    # Bypass VAD entirely: mark everything speech and feed the engine directly.
                    engine.feed_audio(chunk.pcm_bytes, timestamp=chunk.capture_timestamp, vad_state=1)
                if speed > 0:
                    await asyncio.sleep((chunk.duration_ms / 1000.0) / speed)

            # End-of-stream flush (what SessionState.cleanup -> vad.force_end does).
            if vad is not None:
                vad.force_end()
            else:
                engine.on_speech_end(reason="STREAM_END")

            deadline = time.perf_counter() + 15.0
            while not engine.commits and time.perf_counter() < deadline:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.2)

            await engine.cleanup()
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass

            results.append(
                {
                    "id": r["id"],
                    "file_sec": r["duration_sec"],
                    "commit_audio_sec": round(engine.commit_audio_sec, 3),
                    "text": " ".join(c["text"] for c in engine.commits if c["text"]).strip(),
                    "commit_count": len(engine.commits),
                }
            )
    finally:
        config.asr.normalize_speech = was_norm
    return {"per_utt": results}


async def main_async(args: argparse.Namespace) -> None:
    items = pick_utterances(args.n, args.min_sec, args.max_sec)
    refs = [r["text"] for r in items]
    print(f"{len(items)} utterances, {sum(r['duration_sec'] for r in items):.1f}s total, verbatim references\n")
    for r in items:
        print(f"  {r['id']}  {r['duration_sec']:.2f}s  {r['text']}")
    print()

    ASRModelManager().ensure_model(config.asr.active_model)

    print("Run 1: direct model ...")
    direct = run_direct(MODEL, items)
    cer1 = corpus_cer(list(zip(refs, direct)))

    variants = [
        ("pipeline (VAD neutered)", dict(use_vad=True, normalize=True, drop_filter=True)),
        ("  - no normalizer", dict(use_vad=True, normalize=False, drop_filter=True)),
        ("  - no drop filter", dict(use_vad=True, normalize=True, drop_filter=False)),
        ("  - VAD bypassed", dict(use_vad=False, normalize=True, drop_filter=False)),
    ]

    summary = {"direct_cer_pct": round(cer1, 3), "runs": []}
    for label, kw in variants:
        print(f"Run 2 [{label.strip()}] ...")
        res = await run_pipeline(items, speed=args.speed, **kw)
        hyps = [p["text"] for p in res["per_utt"]]
        cer2 = corpus_cer(list(zip(refs, hyps)))
        delivered = sum(p["commit_audio_sec"] for p in res["per_utt"])
        expected = sum(p["file_sec"] for p in res["per_utt"])
        empty = sum(1 for h in hyps if not h)
        identical = sum(1 for a, b in zip(direct, hyps) if a == b)
        summary["runs"].append(
            {
                "label": label.strip(),
                "cer_pct": round(cer2, 3),
                "delta_pp": round(cer2 - cer1, 3),
                "utterances_with_no_commit": empty,
                "text_identical_to_direct": identical,
                "commit_audio_sec": round(delivered, 2),
                "file_sec": round(expected, 2),
                "audio_ratio": round(delivered / max(1e-6, expected), 3),
            }
        )
        print(f"    CER={cer2:6.2f}%  (delta {cer2-cer1:+.2f} pp)  audio delivered {delivered:.2f}s / {expected:.2f}s"
              f"  no-commit={empty}  identical-to-direct={identical}/{len(items)}")

    # Per-utterance comparison for the headline variant.
    res = await run_pipeline(items, speed=args.speed, use_vad=True, normalize=True, drop_filter=True)
    summary["per_utterance"] = [
        {"id": p["id"], "ref": ref, "direct": d, "pipeline": p["text"], "commit_count": p["commit_count"]}
        for p, ref, d in zip(res["per_utt"], refs, direct)
    ]

    print(f"\n{'='*100}\nPER-UTTERANCE (direct vs pipeline)\n{'='*100}")
    for row in summary["per_utterance"]:
        mark = "==" if row["direct"] == row["pipeline"] else "!!"
        print(f"\n[{mark}] {row['id']}")
        print(f"   REF      : {row['ref']}")
        print(f"   DIRECT   : {row['direct']}")
        print(f"   PIPELINE : {row['pipeline']}")

    out = REPO / args.out
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*100}")
    print(f"Run 1 direct CER = {cer1:.2f}%")
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min-sec", type=float, default=1.2)
    ap.add_argument("--max-sec", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = realtime")
    ap.add_argument("--out", default="report/pipeline_transparency.json")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
