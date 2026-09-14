"""Streaming ASR accuracy bench (CER/WER) on the real production pipeline.

Purpose
-------
Iterate on ASR accuracy without booting the full server (translation/TTS/WebSocket).
It drives the *real* production components in-process:

    StreamingAudioSimulator -> VADProcessor -> feed_audio -> AudioBufferManager
        -> SpeechNormalizer -> transcribe_cpp.Session
        -> on_speech_end / STABLE_PREFIX -> committed finals

and scores the concatenated committed text against `wav_test/*.txt`.

Usage
-----
    python -m benchmarks.streaming_wer_bench                      # production defaults
    python -m benchmarks.streaming_wer_bench --mode silence       # disable stability split
    python -m benchmarks.streaming_wer_bench --speed 4            # faster (poller under-samples)
    python -m benchmarks.streaming_wer_bench --lang per-file      # force ground-truth language
    python -m benchmarks.streaming_wer_bench --set asr.normalize_speech=False
    python -m benchmarks.streaming_wer_bench --out report/my_run.json

`--set` accepts dotted paths into `backend_cpp.config.config`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import config
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.dataset import DatasetPair, discover_dataset
from benchmarks.ja_text import score_ja
from benchmarks.simulator import StreamingAudioSimulator

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("streaming_wer_bench")


# ---------------------------------------------------------------------------
# Instrumented engine
# ---------------------------------------------------------------------------

class TracedTranscribeEngine(TranscribeEngine):
    """TranscribeEngine that records every commit and inference for attribution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commits: List[Dict[str, Any]] = []
        self.inferences: List[Dict[str, Any]] = []

    def _emit_final(self, text, utt_id, reason, epoch=0, media_start_time=0.0, media_end_time=0.0):
        self.commits.append(
            {
                "utterance_id": utt_id,
                "reason": reason,
                "text": (text or "").strip(),
                "t_wall": time.time(),
                "buffer_duration_sec": round(self._audio_buffer_mgr.duration_sec, 3),
            }
        )
        return super()._emit_final(
            text,
            utt_id,
            reason,
            epoch=epoch,
            media_start_time=media_start_time,
            media_end_time=media_end_time,
        )

    def _run_inference(self, pcm_float32, frame_state=None, is_commit=False, utt_id=None):
        t0 = time.perf_counter()
        out = super()._run_inference(pcm_float32, frame_state=frame_state, is_commit=is_commit, utt_id=utt_id)
        self.inferences.append(
            {
                "is_commit": bool(is_commit),
                "audio_sec": round(len(pcm_float32) / 16000.0, 3),
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "text": (out or "").strip(),
            }
        )
        return out


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _apply_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Apply `section.key=value` overrides onto the global config; return the applied map."""
    applied: Dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects section.key=value, got: {item}")
        path, raw = item.split("=", 1)
        parts = path.split(".")
        obj: Any = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        leaf = parts[-1]
        old = getattr(obj, leaf)
        if isinstance(old, bool):
            val: Any = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(old, int):
            val = int(raw)
        elif isinstance(old, float):
            val = float(raw)
        else:
            val = raw
        setattr(obj, leaf, val)
        applied[path] = val
    return applied


def _snapshot_config() -> Dict[str, Any]:
    return {
        "vad_engine": config.vad.vad_engine,
        "vad_threshold": config.vad.threshold,
        "silence_duration_ms": config.vad.silence_duration_ms,
        "hangover_ms": config.vad.hangover_ms,
        "pre_speech_buffer_ms": config.vad.pre_speech_buffer_ms,
        "asr_language": config.asr.language,
        "normalize_speech": config.asr.normalize_speech,
        "normalize_target_rms": config.asr.normalize_target_rms,
        "preview_min_growth_ratio": config.asr.preview_min_growth_ratio,
        "poll_interval_ms": config.asr.poll_interval_ms,
        "min_transcribe_sec": config.asr.min_transcribe_sec,
        "split_on_stability": config.sentence.split_on_stability,
        "min_words_to_commit": config.sentence.min_words_to_commit,
        "stability_duration_sec": config.sentence.stability_duration_sec,
        "max_duration_sec": config.sentence.max_duration_sec,
        "max_duration_require_silence": config.sentence.max_duration_require_silence,
        "boundary_candidate_silence_ms": config.sentence.boundary_candidate_silence_ms,
    }


# ---------------------------------------------------------------------------
# Single-file run
# ---------------------------------------------------------------------------

async def run_file(
    pair: DatasetPair,
    mode: str,
    speed: float,
    lang_mode: str,
    chunk_ms: int = 64,
) -> Dict[str, Any]:
    session_id = f"werbench_{pair.pair_id}_{uuid.uuid4().hex[:6]}"
    engine = TracedTranscribeEngine(session_id=session_id)

    if mode == "silence":
        engine.sentence_config.split_on_stability = False
    if lang_mode in ("per-file", "ja"):
        target = "ja" if lang_mode == "ja" else pair.inferred_language
        if target not in ("UNKNOWN", "multi"):
            engine.set_language(target)

    vad = VADProcessor(
        sample_rate=16000,
        vad_engine=config.vad.vad_engine,
        threshold=config.vad.threshold,
        silence_duration_ms=config.vad.silence_duration_ms,
        hangover_ms=config.vad.hangover_ms,
        pre_speech_buffer_ms=config.vad.pre_speech_buffer_ms,
        enabled=True,
        on_speech_chunk=engine.feed_audio,
        on_speech_start=engine.on_speech_start,
        on_speech_end=engine.on_speech_end,
    )

    sim = StreamingAudioSimulator(
        audio_source=pair.wav_path, chunk_ms=chunk_ms, speed=speed, trailing_silence_sec=1.5
    )

    async def _consume():
        try:
            async for _ in engine.stream_tokens():
                pass
        except asyncio.CancelledError:
            pass

    t0 = time.perf_counter()
    consumer = asyncio.create_task(_consume())
    try:
        for chunk in sim.iter_chunks():
            vad.feed_chunk(chunk.pcm_bytes)
            if speed > 0:
                await asyncio.sleep((chunk.duration_ms / 1000.0) / speed)
    finally:
        await engine.cleanup()
        # Drain the commit thread pool / pending coroutines before scoring.
        await asyncio.sleep(0.3)
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
    wall = time.perf_counter() - t0

    # Hypothesis = committed finals in arrival order (this is what the user reads).
    raw_hyp = " ".join(c["text"] for c in engine.commits if c["text"]).strip()

    scoring_lang = "ja" if lang_mode == "ja" else pair.inferred_language
    if scoring_lang == "ja":
        # Japanese has no word boundaries; score at character level and report the
        # ITN-folded variant separately (五十円 vs 50円 must not read as an error).
        js = score_ja(pair.raw_reference, raw_hyp)
        wer_value = None
        cer_value = js.cer
        cer_itn = js.cer_itn
        subs, dels, ins = js.substitutions, js.deletions, js.insertions
        ref_len = js.ref_len
        acc = None
    else:
        acc = evaluate_accuracy(
            raw_reference=pair.raw_reference,
            raw_hypothesis=raw_hyp,
            language=pair.inferred_language,
        )
        wer_value = acc.wer
        cer_value = acc.cer
        cer_itn = None
        subs, dels, ins = acc.substitutions, acc.deletions, acc.insertions
        ref_len = acc.ref_length

    commit_audio = sum(i["audio_sec"] for i in engine.inferences if i["is_commit"])
    preview_audio = sum(i["audio_sec"] for i in engine.inferences if not i["is_commit"])

    return {
        "pair_id": pair.pair_id,
        "language": pair.inferred_language,
        "condition": pair.inferred_condition,
        "audio_duration_sec": round(pair.actual_duration_sec, 2),
        "wall_sec": round(wall, 2),
        "wer": wer_value,
        "cer": cer_value,
        "cer_itn": cer_itn,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "ref_length": ref_len,
        "hyp_length": len(raw_hyp),
        "raw_reference": pair.raw_reference,
        "raw_hypothesis": raw_hyp,
        "normalized_reference": (acc.normalized_reference if acc is not None else js.norm_ref),
        "normalized_hypothesis": (acc.normalized_hypothesis if acc is not None else js.norm_hyp),
        "commit_count": len([c for c in engine.commits if c["text"]]),
        "commit_reasons": [c["reason"] for c in engine.commits if c["text"]],
        "commit_audio_sec": round(commit_audio, 2),
        "preview_audio_sec": round(preview_audio, 2),
        "inference_count": len(engine.inferences),
        "audio_exposure_ratio": round((commit_audio + preview_audio) / max(0.001, pair.actual_duration_sec), 2),
        "commits": engine.commits,
        "inferences": engine.inferences,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> Dict[str, Any]:
    applied = _apply_overrides(args.set)
    dataset = discover_dataset(Path(args.dataset))
    if args.files:
        wanted = [f.lower() for f in args.files.split(",")]
        dataset = [p for p in dataset if any(w in p.pair_id.lower() for w in wanted)]

    if not dataset:
        raise SystemExit("No benchmark files matched.")

    logger.warning("Pre-warming model...")
    ASRModelManager().ensure_model(config.asr.active_model)

    cfg_snapshot = _snapshot_config()
    cfg_snapshot.update({f"override.{k}": v for k, v in applied.items()})
    logger.warning(f"Config: {json.dumps(cfg_snapshot, ensure_ascii=False)}")

    results: List[Dict[str, Any]] = []
    for pair in dataset:
        logger.warning(f"▶ {pair.pair_id} ({pair.actual_duration_sec:.1f}s)...")
        rec = await run_file(pair, mode=args.mode, speed=args.speed, lang_mode=args.lang, chunk_ms=args.chunk_ms)
        results.append(rec)
        wer_txt = f"{rec['wer']*100:.2f}%" if rec["wer"] is not None else "n/a"
        itn_txt = f" CER_ITN={rec['cer_itn']*100:.2f}%" if rec.get("cer_itn") is not None else ""
        logger.warning(
            f"   CER={rec['cer']*100:.2f}%{itn_txt} WER={wer_txt} "
            f"commits={rec['commit_count']} {rec['commit_reasons']} exposure={rec['audio_exposure_ratio']}x"
        )

    # Character-weighted corpus CER (total edits / total reference characters) is the
    # standard definition; a plain mean over files would over-weight short files.
    total_edits = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in results)
    total_ref = sum(r["ref_length"] for r in results)
    corpus_cer = 100.0 * total_edits / total_ref if total_ref else 0.0

    wer_vals = [r["wer"] for r in results if r["wer"] is not None]
    corpus_wer = 100.0 * sum(wer_vals) / len(wer_vals) if wer_vals else None

    itn_edits = sum(
        (r["substitutions"] + r["deletions"] + r["insertions"]) for r in results if r.get("cer_itn") is not None
    )
    itn_ref = sum(r["ref_length"] for r in results if r.get("cer_itn") is not None)
    corpus_cer_itn = 100.0 * itn_edits / itn_ref if itn_ref else None

    summary = {
        "mode": args.mode,
        "speed": args.speed,
        "lang_mode": args.lang,
        "config": cfg_snapshot,
        "corpus_cer_pct": round(corpus_cer, 3),
        "corpus_cer_itn_pct": (round(corpus_cer_itn, 3) if corpus_cer_itn is not None else None),
        "corpus_wer_pct": (round(corpus_wer, 3) if corpus_wer is not None else None),
        "total_ref_chars": total_ref,
        "total_edits": total_edits,
        "per_file_cer_pct": {r["pair_id"]: (round(r["cer"] * 100, 2) if r["cer"] is not None else None) for r in results},
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.warning("=" * 70)
    itn_msg = f"  CER(ITN)={corpus_cer_itn:.2f}%" if corpus_cer_itn is not None else ""
    logger.warning(f"CORPUS CER={corpus_cer:.2f}%{itn_msg}  ({total_edits} edits / {total_ref} ref chars) -> {out}")
    for pid, cer in summary["per_file_cer_pct"].items():
        logger.warning(f"   {pid:<48} CER={cer}%")
    logger.warning("=" * 70)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Streaming ASR WER/CER bench")
    ap.add_argument("--dataset", default="wav_test")
    ap.add_argument("--files", default="", help="comma-separated substrings to filter pair ids")
    ap.add_argument("--mode", default="stream", choices=["stream", "silence"])
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = realtime (production-faithful)")
    ap.add_argument("--lang", default="auto", choices=["auto", "per-file", "ja"])
    ap.add_argument("--chunk-ms", type=int, default=64)
    ap.add_argument("--set", action="append", default=[], help="section.key=value override (repeatable)")
    ap.add_argument("--out", default="report/streaming_wer_bench.json")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
