"""Real-speech end-to-end scenario for the W2.1 baseline.

Why this exists
---------------
The synthetic test signal used by the other scenarios (`generate_pcm_audio`) is a harmonic
tone mix. Measured behaviour (see ``micro_benchmarks.vad_cost`` and the ``validity`` block
of ``preview_cost_e2e``):

* it is classified as **non-speech** by the FireRed VAD engine (0 frames) and as speech by
  FSMN, so scenarios must request ``fsmn-vad`` explicitly;
* even when utterances commit, the ASR returns an **empty transcript**, so
  ``TranscribeEngine._emit_final()`` drops it and the client receives nothing.

That makes the synthetic signal fine for measuring *cost* (``infer_ms``, RTF, lock waits,
RSS) but useless for measuring *delivery* (subtitles, translations, TTS). Real speech fixes
this, and it is also the reference audio the W2.2 transcript-equivalence gate needs.

The streaming loop lives here rather than in ``perf_benchmark.BenchmarkClient`` so the
existing benchmark module stays untouched.
"""

from __future__ import annotations

import asyncio
import logging
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE
from backend_cpp.tests.perf_benchmark import BenchmarkClient, pack_audio_frame
from backend_cpp.tests.perf_baseline import (
    BASELINE_VAD_ENGINE,
    _baseline_config,
    _rss_mb,
    _stats,
    _validity,
)
from backend_cpp.utils.perf_profiler import perf

logger = logging.getLogger("perf_baseline")

from backend_cpp.config import PROJECT_ROOT

# 16 kHz mono 16-bit speech, used when no --audio-file is given.
def _resolve_default_reference_audio() -> str:
    for cand in [
        "wav_test/OSR_us_000_0010_16k.wav",
        "wav_test/English_multiple_kinds_of_noise_88s.wav",
        "wav_test/Chinese_noise_28s.wav",
    ]:
        if (PROJECT_ROOT / cand).exists():
            return cand
    return "wav_test/English_multiple_kinds_of_noise_88s.wav"

DEFAULT_REFERENCE_AUDIO = _resolve_default_reference_audio()


def load_wav_pcm16(path: Path, max_sec: Optional[float] = None) -> bytes:
    """Load a WAV file as 16 kHz mono 16-bit PCM bytes.

    Only the exact format the pipeline consumes is accepted. Rather than silently
    resampling or downmixing (which would change what is being measured), anything else
    raises with a message that says what to do.
    """
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        if channels != 1 or rate != DEFAULT_SAMPLE_RATE or width != 2:
            raise ValueError(
                f"{path} is {channels}ch/{rate}Hz/{width * 8}-bit but the pipeline needs "
                f"1ch/{DEFAULT_SAMPLE_RATE}Hz/16-bit. Convert it first, e.g.:\n"
                f"  ffmpeg -i \"{path}\" -ac 1 -ar {DEFAULT_SAMPLE_RATE} -sample_fmt s16 out.wav"
            )
        frames = wf.getnframes()
        if max_sec is not None:
            frames = min(frames, int(max_sec * rate))
        return wf.readframes(frames)


def count_delivery(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Count delivered messages, collapsing the BY-DESIGN duplicate final utterances.

    Each committed sentence produces two ``is_final`` utterances:

    1. the ASR final from ``_stream_asr_tokens()``, and
    2. a repeated ``utterance_update`` emitted by ``_process_translation_item()``
       "for complete backward compatibility", which carries the translation.

    Naively counting messages therefore roughly doubles the sentence count and looks like a
    duplicate-transcript defect. ``final_utterances`` counts unique ``utterance_id`` values;
    ``final_messages`` keeps the raw count visible so the duplication is not hidden.
    """
    final_messages = [
        m for m in messages
        if m.get("type") == "utterance_update" and m.get("is_final")
    ]

    finals_by_id: Dict[str, Dict[str, Any]] = {}
    for message in final_messages:
        finals_by_id.setdefault(message.get("utterance_id"), message)

    return {
        "final_utterances": len(finals_by_id),
        "final_messages": len(final_messages),
        "unique_finals": list(finals_by_id.values()),
        "preview_updates": sum(
            1 for m in messages
            if m.get("type") == "utterance_update" and not m.get("is_final")
        ),
        "translations": sum(1 for m in messages if m.get("type") == "translation"),
        "tts_audio_messages": sum(1 for m in messages if m.get("type") == "tts_audio"),
    }


async def _stream_pcm(
    client: BenchmarkClient,
    pcm_bytes: bytes,
    chunk_ms: int = 100,
    realtime_factor: float = 1.0,
    start_chunk_idx: int = 1,
) -> int:
    """Stream raw PCM through the client at the requested rate, like the extension does."""
    chunk_bytes = int(DEFAULT_SAMPLE_RATE * chunk_ms / 1000.0) * 2
    total_chunks = len(pcm_bytes) // chunk_bytes
    chunk_idx = start_chunk_idx
    start_ts = time.time()

    for i in range(total_chunks):
        offset = i * chunk_bytes
        frame = pack_audio_frame(
            pcm_bytes[offset: offset + chunk_bytes],
            chunk_idx,
            start_ts + (i * chunk_ms / 1000.0),
            format_type="A",
        )
        await client.ws.send(frame)
        chunk_idx += 1
        if realtime_factor > 0:
            await asyncio.sleep((chunk_ms / 1000.0) / realtime_factor)

    return chunk_idx


async def run_scenario_real_audio(
    uri: str,
    ssl_context: Any = None,
    audio_path: Optional[Path] = None,
    max_sec: Optional[float] = 24.0,
    speed: float = 1.0,
    with_tts: bool = False,
    silence_tail_sec: float = 2.0,
    vad_engine: str = BASELINE_VAD_ENGINE,
    vad_silence_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Stream real speech and measure delivery as well as cost.

    This is the scenario that can answer "did subtitles and translations actually arrive",
    which the synthetic-signal scenarios cannot. It is also the harness the W2.2
    transcript-equivalence gate will build on, so ``transcripts`` is reported explicitly.

    ``vad_engine`` is selectable because VAD choice changes both cost (firered is ~8x
    silero) and WHERE the audio is cut, and only real speech can show whether that changes
    the transcripts.
    """
    path = Path(audio_path) if audio_path else Path(DEFAULT_REFERENCE_AUDIO)
    if not path.is_absolute():
        path = Path.cwd() / path

    pcm = load_wav_pcm16(path, max_sec=max_sec)
    audio_sec = len(pcm) / (DEFAULT_SAMPLE_RATE * 2)
    message = (
        "▶️ [BASELINE R] Real audio %s (%.1fs, speed=%.1fx, tts=%s, vad=%s, silence=%s)..."
    )
    logger.info(
        message,
        path.name,
        audio_sec,
        speed,
        with_tts,
        vad_engine,
        f"{vad_silence_ms}ms" if vad_silence_ms else "default",
    )

    perf.reset()
    rss_before = _rss_mb()

    client = BenchmarkClient(uri, ssl_context)
    await client.connect()
    session_config = _baseline_config(
        ttsEnabled=bool(with_tts), splitOnStability=True, vadEngine=vad_engine
    )
    if vad_silence_ms:
        # The VAD safety net fires after exactly this much trailing silence (measured: 450ms
        # config -> 480ms of silence, in 60ms frame steps). It is the single largest stage of
        # perceived subtitle latency and costs nothing in compute, so it is measured here.
        session_config["silenceDurationMs"] = int(vad_silence_ms)
    await client.send_config(session_config)
    await asyncio.sleep(0.5)

    started = time.perf_counter()
    chunk_idx = await _stream_pcm(client, pcm, chunk_ms=100, realtime_factor=speed, start_chunk_idx=1)

    # Trailing silence: must exceed VAD silence_duration_ms + hangover_ms or the last
    # utterance never commits.
    silence = b"\x00\x00" * int(DEFAULT_SAMPLE_RATE * silence_tail_sec)
    await _stream_pcm(
        client, silence, chunk_ms=100, realtime_factor=speed, start_chunk_idx=chunk_idx
    )

    # Drain: allow pending translation/TTS to finish and be delivered.
    await asyncio.sleep(max(4.0, 2.0 + silence_tail_sec))
    wall_sec = time.perf_counter() - started
    await client.disconnect()

    delivery = count_delivery(client.received_messages)
    finals = delivery["unique_finals"]

    counters = perf.generate_report()["counters"]
    total_preview_audio_ms = counters.get("asr.preview_audio_ms", 0)
    total_commit_audio_ms = counters.get("asr.commit_audio_ms", 0)
    rss_after = _rss_mb()

    logger.info(
        "✅ [BASELINE R] %.1fs wall | %d utterances (%d final messages), %d previews, "
        "%d translations, %d tts | RSS %.1f -> %.1f MB",
        wall_sec,
        delivery["final_utterances"],
        delivery["final_messages"],
        delivery["preview_updates"],
        delivery["translations"],
        delivery["tts_audio_messages"],
        rss_before,
        rss_after,
    )

    return {
        "benchmark": "real_audio_e2e",
        "audio_file": str(path),
        "audio_sec": round(audio_sec, 2),
        "speed": speed,
        "tts_enabled": with_tts,
        "wall_sec": round(wall_sec, 2),
        "validity": _validity(
            counters.get("asr.total_inferences", 0),
            delivered=len(finals),
        ),
        "vad_engine": vad_engine,
        "vad_silence_ms": vad_silence_ms,
        "delivery": {
            "final_utterances": delivery["final_utterances"],
            "final_messages": delivery["final_messages"],
            "preview_updates": delivery["preview_updates"],
            "translations": delivery["translations"],
            "tts_audio_messages": delivery["tts_audio_messages"],
            "note": (
                "final_messages is about 2x final_utterances BY DESIGN: "
                "_process_translation_item() re-sends a final utterance_update for backward "
                "compatibility. Do not read that as duplicate transcripts."
            ),
        },
        # Reference transcripts: the W2.2 equivalence gate compares against these.
        "transcripts": [
            {
                "utterance_id": m.get("utterance_id"),
                "text": m.get("text", ""),
                "language": m.get("language"),
            }
            for m in finals
        ],
        # Preview amplification.
        #
        # The active model (qwen3-asr-1.7b) advertises supports_streaming=False, so every
        # preview poll re-runs `session.run()` over the WHOLE utterance so far. The audio
        # processed by previews therefore exceeds the audio actually spoken, and this factor
        # is the real ASR waste -- not the copy chain (~0.7 ms/audio-sec).
        "preview_amplification": {
            "speech_ms": int(audio_sec * 1000.0),
            "preview_audio_ms": total_preview_audio_ms,
            "commit_audio_ms": total_commit_audio_ms,
            "factor_vs_speech": round(total_preview_audio_ms / max(1.0, audio_sec * 1000.0), 2),
        },
        "asr": {
            "infer_ms": _stats(perf.get_samples("asr", "infer_ms")),
            "rtf": _stats(perf.get_samples("asr", "rtf")),
            "lock_wait_ms": _stats(perf.get_samples("asr", "lock_wait_ms")),
        },
        "translation": {
            "queue_wait_ms": _stats(perf.get_samples("translation", "queue_wait_ms")),
            "infer_ms": _stats(perf.get_samples("translation", "infer_ms")),
            "tokens_per_sec": _stats(perf.get_samples("translation", "tokens_per_sec")),
        },
        # TTS is a real product feature (the extension has a toggle, voice and ducking) and was
        # never measured before: every earlier run used ttsEnabled=False. It sits AFTER
        # translation on the dubbing path, so its cost adds to the subtitle latency above.
        "tts": {
            "queue_wait_ms": _stats(perf.get_samples("tts", "queue_wait_ms")),
            "synthesis_ms": _stats(perf.get_samples("tts", "synthesis_ms")),
            "rtf": _stats(perf.get_samples("tts", "rtf")),
        },
        # End-to-end pipeline latencies recorded by ws_handler (measured, not derived).
        "pipeline_e2e": {
            "asr_to_sub_ms": _stats(perf.get_samples("pipeline", "e2e_asr_to_sub_ms")),
            "sub_to_tts_ms": _stats(perf.get_samples("pipeline", "e2e_sub_to_tts_ms")),
        },
        "counters": {
            key: counters.get(key, 0)
            for key in (
                "asr.total_inferences",
                "asr.preview_infers",
                "asr.commit_inferences",
                "asr.cached_preview_reuse",
                "asr.preview_audio_ms",
                "asr.commit_audio_ms",
                "asr.preview_skipped_growth",
                "translation.queue_full_dropped",
                "translation.short_words_filtered",
                "tts.queue_full_dropped",
                "tts.synthesized_utterances",
                "tts.dedup_skipped",
                "asr.preview_dropped",
                "asr.preview_evicted",
            )
        },
        "resources": {
            "ram_rss_mb_before": rss_before,
            "ram_rss_mb_after": rss_after,
            "ram_rss_delta_mb": round(rss_after - rss_before, 1),
        },
    }
