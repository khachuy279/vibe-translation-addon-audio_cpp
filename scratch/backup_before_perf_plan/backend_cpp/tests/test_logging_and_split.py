"""Test verifying VAD start/end logs and ASR COMMIT format [model] [reason]."""

import asyncio
import logging
import io
import pytest
import numpy as np

from backend_cpp.vad.vad_processor import VADProcessor
from backend_cpp.asr.transcribe_engine import TranscribeEngine

# Set up logging capture
log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s [%(name)s]: %(message)s")
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)


@pytest.mark.asyncio
async def test_vad_start_end_logging():
    log_stream.seek(0)
    log_stream.truncate(0)

    vad = VADProcessor(
        vad_engine="fsmn-vad",
        silence_duration_ms=100,
        hangover_ms=50,
    )

    # 1. Feed speech (synthetic 440Hz sine wave to trigger speech detection)
    t = np.linspace(0, 0.5, int(16000 * 0.5), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes()

    chunk_size = 1024  # 512 samples
    for i in range(0, len(sine), chunk_size):
        vad.feed_chunk(sine[i:i + chunk_size])

    # 2. Feed silence to trigger speech end
    silence = b"\x00" * int(16000 * 2 * 0.5)
    for i in range(0, len(silence), chunk_size):
        vad.feed_chunk(silence[i:i + chunk_size])
        await asyncio.sleep(0.01)

    vad.force_end()

    logs = log_stream.getvalue()
    print("CAPTURED LOGS:\n", logs)

    # Verify VAD logs exist
    assert "[VAD START]" in logs or "[VAD END]" in logs


@pytest.mark.asyncio
async def test_asr_commit_log_format():
    log_stream.seek(0)
    log_stream.truncate(0)

    asr = TranscribeEngine("qwen3-asr-1.7b")

    # Test _emit_final with VAD_SILENCE
    asr._emit_final("A cover for me.", "utt-2", reason="VAD_SILENCE")
    # Test _emit_final with STABLE_PREFIX
    asr._emit_final("Hello world", "utt-3", reason="STABLE_PREFIX")
    # Test _emit_final with MAX_DURATION
    asr._emit_final("Long sentence exceeding buffer limit", "utt-4", reason="MAX_DURATION")

    logs = log_stream.getvalue()
    print("ASR COMMIT LOGS:\n", logs)

    assert "[ASR COMMIT] [utt=utt-2] [qwen3-asr-1.7b] [VAD_SILENCE] (auto): 'A cover for me.'" in logs
    assert "[ASR COMMIT] [utt=utt-3] [qwen3-asr-1.7b] [STABLE_PREFIX] (auto): 'Hello world'" in logs
    assert "[ASR COMMIT] [utt=utt-4] [qwen3-asr-1.7b] [MAX_DURATION] (auto): 'Long sentence exceeding buffer limit'" in logs

