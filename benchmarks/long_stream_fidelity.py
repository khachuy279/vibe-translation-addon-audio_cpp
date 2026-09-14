"""Long-stream fidelity: can the real VAD segmentation approach a direct decode?

Test material: a ~3 minute stream built by concatenating whole Common Voice sentences with a
500 ms gap, so every true sentence boundary is known.

Three numbers, all on the same audio:

  (1)  ceiling       decode each ground-truth sentence span on its own and concatenate.
                     Ground-truth spans are single sentences (all well under 8 s), so this is
                     the best achievable output that also respects the subtitle length limit.
  (3)  pipeline      the real production path: VAD segmentation ON, normalizer, commit.
                     Subtitle constraint: every committed segment must be < --max-seg-sec.

Goal: (3) as close to (1) as possible.

(1) depends only on the stream and the model, not on any VAD setting, so it is cached in
`report/refs_<stream>.json` and reused. Use --recompute-refs to force a refresh.

    python -m benchmarks.long_stream_fidelity                            # uses cached (1)
    python -m benchmarks.long_stream_fidelity --set vad.silence_duration_ms=400
    python -m benchmarks.long_stream_fidelity --recompute-refs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

import transcribe_cpp
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import config
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.ja_text import normalize_ja, score_ja
from benchmarks.simulator import StreamingAudioSimulator

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "backend_cpp" / "models" / "Qwen3-ASR-1.7B-Q8_0.gguf"
REF_CACHE = REPO / "report" / "refs"


# --------------------------------------------------------------------------- scoring
def corpus_cer(pairs: List[tuple]) -> float:
    edits = ref_len = 0
    for ref, hyp in pairs:
        s = score_ja(ref, hyp)
        edits += s.substitutions + s.deletions + s.insertions
        ref_len += s.ref_len
    return 100.0 * edits / max(1, ref_len)


def edit_counts(ref: str, hyp: str):
    s = score_ja(ref, hyp)
    return s.substitutions + s.deletions + s.insertions, s.ref_len


def load_wav(path: Path) -> np.ndarray:
    d, sr = sf.read(str(path), dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    return d.astype(np.float32)


# --------------------------------------------------------------------------- pipeline
class TracedEngine(TranscribeEngine):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.commits: List[Dict[str, Any]] = []
        self.commit_audio_sec = 0.0

    def _emit_final(self, text, utt_id, reason, epoch=0, media_start_time=0.0, media_end_time=0.0):
        self.commits.append({"reason": reason, "text": (text or "").strip(),
                             "audio_sec": round(self.commit_audio_sec, 3)})
        self.commit_audio_sec = 0.0
        return super()._emit_final(text, utt_id, reason, epoch=epoch,
                                   media_start_time=media_start_time, media_end_time=media_end_time)

    def _run_inference(self, pcm_float32, frame_state=None, is_commit=False, utt_id=None):
        if is_commit:
            self.commit_audio_sec += len(pcm_float32) / 16000.0
        return super()._run_inference(pcm_float32, frame_state=frame_state,
                                      is_commit=is_commit, utt_id=utt_id)


async def run_pipeline(wav: Path, speed: float) -> Dict[str, Any]:
    engine = TracedEngine(session_id="long_stream")

    async def consume() -> None:
        try:
            async for _ in engine.stream_tokens():
                pass
        except asyncio.CancelledError:
            pass

    consumer = asyncio.create_task(consume())
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

    sim = StreamingAudioSimulator(audio_source=wav, chunk_ms=64, speed=speed, trailing_silence_sec=1.5)
    t0 = time.perf_counter()
    for chunk in sim.iter_chunks():
        vad.feed_chunk(chunk.pcm_bytes)
        if speed > 0:
            await asyncio.sleep((chunk.duration_ms / 1000.0) / speed)
    wall = time.perf_counter() - t0

    vad.force_end()
    deadline = time.perf_counter() + 30.0
    while time.perf_counter() < deadline:
        await asyncio.sleep(0.1)
        if engine.commits:
            break
    await asyncio.sleep(1.0)
    await engine.cleanup()
    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass

    return {
        "wall_sec": round(wall, 1),
        "commits": engine.commits,
        "text": " ".join(c["text"] for c in engine.commits if c["text"]).strip(),
    }


# --------------------------------------------------------------------------- reference cache
def compute_refs(stream_dir: Path, meta: dict, ref: str) -> Dict[str, Any]:
    pcm = load_wav(stream_dir / meta["wav"])
    model = transcribe_cpp.Model(str(MODEL))
    session = model.session()
    session.run(np.zeros(16000, dtype=np.float32), language="ja")

    parts = []
    for a, b in meta["true_sentences_sec"]:
        seg = pcm[int(a * 16000): int(b * 16000)]
        if len(seg) >= 1600:
            parts.append((session.run(seg, language="ja").text or "").strip())
    ceiling_text = " ".join(parts)

    try:
        single = (session.run(pcm, language="ja").text or "").strip()
        single_err = None
    except Exception as e:
        partial = getattr(e, "partial_result", None)
        single = (getattr(partial, "text", "") or "").strip() if partial is not None else ""
        single_err = type(e).__name__

    win = 30 * 16000
    wparts = []
    for s in range(0, len(pcm), win):
        seg = pcm[s: s + win]
        if len(seg) >= 1600:
            wparts.append((session.run(seg, language="ja").text or "").strip())

    session.close()
    model.close()
    return {
        "stream": meta["id"],
        "ceiling_cer_pct": round(corpus_cer([(ref, ceiling_text)]), 3),
        "ceiling_text": ceiling_text,
        "single_shot_cer_pct": round(corpus_cer([(ref, single)]), 3),
        "single_shot_error": single_err,
        "windows30_cer_pct": round(corpus_cer([(ref, " ".join(wparts))]), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams-dir", default="data/ja_long3")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--recompute-refs", action="store_true")
    ap.add_argument("--max-seg-sec", type=float, default=8.0,
                    help="subtitle constraint: no committed segment may exceed this")
    ap.add_argument("--set", action="append", default=[], help="section.key=value override")
    ap.add_argument("--out", default="report/long_stream_sweep_last.json")
    args = ap.parse_args()

    for item in args.set:
        path, raw = item.split("=", 1)
        parts = path.split(".")
        obj: Any = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        old = getattr(obj, parts[-1])
        val = (raw.strip().lower() in ("1", "true", "yes", "on")) if isinstance(old, bool) else (
            int(raw) if isinstance(old, int) else (float(raw) if isinstance(old, float) else raw)
        )
        setattr(obj, parts[-1], val)

    # The subtitle length constraint is part of the configuration under test.
    config.sentence.max_duration_sec = args.max_seg_sec

    stream_dir = REPO / args.streams_dir
    meta = json.loads((stream_dir / "streams.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    ref = (stream_dir / "streams" / f"{meta['id']}.txt").read_text(encoding="utf-8")
    wav = stream_dir / meta["wav"]

    print(f"{meta['id']}: {meta['duration_sec']:.0f}s, {len(meta['true_sentences_sec'])} sentences, "
          f"max sentence {max(b - a for a, b in meta['true_sentences_sec']):.1f}s")
    print(f"cfg: engine={config.vad.vad_engine} thr={config.vad.threshold} "
          f"silence={config.vad.silence_duration_ms}ms hangover={config.vad.hangover_ms}ms "
          f"pre_speech={config.vad.pre_speech_buffer_ms}ms max_seg={config.sentence.max_duration_sec}s "
          f"fresh_commit={not config.asr.reuse_stable_preview}")

    REF_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = REF_CACHE / f"{meta['id']}.json"
    if cache_file.exists() and not args.recompute_refs:
        refs = json.loads(cache_file.read_text(encoding="utf-8"))
        print(f"(1)  ceiling  CER={refs['ceiling_cer_pct']:6.2f}%   [cached]")
    else:
        ASRModelManager().ensure_model(config.asr.active_model)
        t0 = time.perf_counter()
        refs = compute_refs(stream_dir, meta, ref)
        cache_file.write_text(json.dumps(refs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"(1)  ceiling  CER={refs['ceiling_cer_pct']:6.2f}%   ({time.perf_counter()-t0:.0f}s, cached)")

    cer1 = refs["ceiling_cer_pct"]
    summary: Dict[str, Any] = {
        "stream": meta["id"], "ceiling_cer_pct": cer1,
        "config": {
            "vad_engine": config.vad.vad_engine, "threshold": config.vad.threshold,
            "silence_duration_ms": config.vad.silence_duration_ms,
            "hangover_ms": config.vad.hangover_ms,
            "pre_speech_buffer_ms": config.vad.pre_speech_buffer_ms,
            "max_duration_sec": config.sentence.max_duration_sec,
            "reuse_stable_preview": config.asr.reuse_stable_preview,
        },
    }

    if not args.skip_pipeline:
        ASRModelManager().ensure_model(config.asr.active_model)
        res = asyncio.run(run_pipeline(wav, args.speed))
        edits, ref_len = edit_counts(ref, res["text"])
        cer3 = 100.0 * edits / max(1, ref_len)
        durs = [c["audio_sec"] for c in res["commits"] if c["text"]]
        over = [d for d in durs if d > args.max_seg_sec]
        summary.update({
            "pipeline_cer_pct": round(cer3, 3),
            "gap_pp": round(cer3 - cer1, 3),
            "commits": len(durs),
            "segment_max_sec": round(max(durs), 2) if durs else 0,
            "segment_p95_sec": round(float(np.percentile(durs, 95)), 2) if durs else 0,
            "segments_over_limit": len(over),
            "wall_sec": res["wall_sec"],
            "pipeline_text": res["text"],
            "commit_detail": res["commits"],
        })
        print(f"(3)  pipeline CER={cer3:6.2f}%   gap={cer3 - cer1:+.2f}pp   "
              f"segments={len(durs)} max={summary['segment_max_sec']}s p95={summary['segment_p95_sec']}s "
              f"over{args.max_seg_sec}s={len(over)}   wall={res['wall_sec']:.0f}s")

    out = REPO / args.out
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
