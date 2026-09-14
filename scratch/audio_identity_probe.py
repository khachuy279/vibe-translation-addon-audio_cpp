"""Is the audio the pipeline feeds the model the same audio as the file?

`pipeline_transparency_test` showed the VAD path changes the transcript on 3/10 utterances
(+2.06 pp CER) while bypassing VAD reproduces the direct result exactly. This decides
whether that is an *audio* difference or an *engine-call* difference:

  A. capture the exact float32 array the engine hands to transcribe_cpp on commit
  B. compare it to the file (length, sample-by-sample)
  C. decode the captured array with a fresh session and compare that transcript to both

If (C) matches the pipeline text, the cause is the audio. If it matches the direct text, the
cause is how the engine calls the model.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import transcribe_cpp  # noqa: E402
from backend_cpp.asr.model_manager import ASRModelManager  # noqa: E402
from backend_cpp.asr.transcribe_engine import TranscribeEngine  # noqa: E402
from backend_cpp.config import config  # noqa: E402
from backend_cpp.vad.vad_processor import VADProcessor  # noqa: E402
from benchmarks.ja_text import score_ja  # noqa: E402
from benchmarks.simulator import StreamingAudioSimulator  # noqa: E402

CV = REPO / "data" / "ja_cv"
MODEL = REPO / "backend_cpp" / "models" / "Qwen3-ASR-1.7B-Q8_0.gguf"
TARGETS = ["cv00000", "cv00001", "cv00002", "cv00006", "cv00008"]


class CapturingEngine(TranscribeEngine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captured = None
        self.commits = []

    def _emit_final(self, text, utt_id, reason, epoch=0, media_start_time=0.0, media_end_time=0.0):
        self.commits.append((text or "").strip())
        return super()._emit_final(text, utt_id, reason, epoch=epoch,
                                   media_start_time=media_start_time, media_end_time=media_end_time)

    def _run_inference(self, pcm_float32, frame_state=None, is_commit=False, utt_id=None):
        if is_commit and self.captured is None:
            self.captured = np.array(pcm_float32, dtype=np.float32, copy=True)
        return super()._run_inference(pcm_float32, frame_state=frame_state,
                                      is_commit=is_commit, utt_id=utt_id)


async def capture(item: dict) -> np.ndarray:
    engine = CapturingEngine(session_id=f"cap_{item['id']}")
    engine.sentence_config.max_duration_sec = 60.0
    engine._segmenter.max_duration_sec = 60.0

    async def consume():
        try:
            async for _ in engine.stream_tokens():
                pass
        except asyncio.CancelledError:
            pass

    consumer = asyncio.create_task(consume())
    sim = StreamingAudioSimulator(audio_source=CV / item["wav"], chunk_ms=64, speed=0,
                                  trailing_silence_sec=0.5)
    vad = VADProcessor(
        sample_rate=16000, vad_engine=config.vad.vad_engine, threshold=config.vad.threshold,
        silence_duration_ms=60000, hangover_ms=config.vad.hangover_ms,
        pre_speech_buffer_ms=config.vad.pre_speech_buffer_ms, enabled=True,
        on_speech_chunk=engine.feed_audio, on_speech_start=engine.on_speech_start,
        on_speech_end=engine.on_speech_end,
    )
    for chunk in sim.iter_chunks():
        vad.feed_chunk(chunk.pcm_bytes)
    vad.force_end()

    import time
    deadline = time.perf_counter() + 20
    while engine.captured is None and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)
    await engine.cleanup()
    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass
    return engine.captured


async def main() -> None:
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(CV / "utts.jsonl", encoding="utf-8") if l.strip()}
    ASRModelManager().ensure_model(config.asr.active_model)

    ref_model = transcribe_cpp.Model(str(MODEL))
    ref_session = ref_model.session()
    ref_session.run(np.zeros(16000, dtype=np.float32), language="ja")

    for tid in TARGETS:
        item = rows[tid]
        file_pcm, _sr = sf.read(str(CV / item["wav"]), dtype="float32")
        if file_pcm.ndim > 1:
            file_pcm = file_pcm.mean(axis=1)
        file_pcm = file_pcm.astype(np.float32)

        direct = (ref_session.run(file_pcm, language="ja").text or "").strip()
        captured = await capture(item)

        print(f"\n{'='*100}\n{item['id']}  file={len(file_pcm)/16000:.2f}s")
        print(f"  REF      : {item['text']}")
        print(f"  DIRECT   : {direct}")
        if captured is None:
            print("  CAPTURE FAILED")
            continue

        n = min(len(file_pcm), len(captured))
        diff = np.abs(file_pcm[:n] - captured[:n])
        # Best alignment offset, searched over a small lag window.
        best = (0, float("inf"))
        for lag in range(-2400, 2401, 80):
            a = file_pcm[max(0, lag): max(0, lag) + n]
            b = captured[: len(a)]
            if len(a) < 1600:
                continue
            d = float(np.mean(np.abs(a - b)))
            if d < best[1]:
                best = (lag, d)

        print(f"  captured : {len(captured)/16000:.3f}s  ({len(captured) - len(file_pcm):+d} samples vs file)")
        print(f"  max|diff| (same index) = {float(diff.max()):.6f}   mean|diff| = {float(diff.mean()):.6f}")
        print(f"  best lag = {best[0]} samples ({best[0]/16:.1f} ms)  mean|diff| = {best[1]:.6f}")

        replay = (ref_session.run(captured, language="ja").text or "").strip()
        print(f"  REPLAY   : {replay}")
        print(f"  replay==pipeline_capture_source? direct={replay == direct} ")

    ref_session.close()
    ref_model.close()


if __name__ == "__main__":
    asyncio.run(main())
