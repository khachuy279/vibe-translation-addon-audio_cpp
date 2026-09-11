"""Test suite for Audio Pipeline Audit.

Verifies:
1. Zero-loss resampling and carryover buffer preservation (simulation of 48kHz/44.1kHz -> 16kHz).
2. ASR tail audio preservation on commit (ensuring no skipping of 600ms tail endings).
3. ASR normalize_speech configuration bypass and safe gain ceiling.
4. Audio dumper functionality for end-to-end observability.
5. CJK content token counting and preservation of min_words_to_commit filtering.
"""

import math
from pathlib import Path
import tempfile
import wave
import numpy as np
import pytest
from unittest.mock import MagicMock

from backend_cpp.config import config, load_config
from backend_cpp.asr.sentence_segmenter import count_content_tokens
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.utils.audio_dumper import (
    dump_ingress_chunk,
    dump_vad_utterance,
    dump_asr_input,
    close_session_dumper,
    flush_dumper,
)


def test_extension_resampling_zero_loss_simulation():
    """Simulate AudioCapture carryover buffer algorithm on 48kHz and 44.1kHz input streams.

    Verifies that total output samples at 16kHz matches expected duration with 0% sample loss.
    """
    chunk_size = 1024
    buffer_size = 4096

    for input_sr in (48000, 44100):
        # 3 seconds of synthetic audio at input_sr
        total_input_sec = 3.0
        total_input_samples = int(total_input_sec * input_sr)
        input_signal = np.sin(2 * np.pi * 440 * np.arange(total_input_samples) / input_sr).astype(np.float32)

        # Simulation state
        residual_samples = np.zeros(0, dtype=np.float32)
        resample_phase = 0.0
        emitted_samples = 0
        ratio = input_sr / 16000.0

        for b_start in range(0, total_input_samples, buffer_size):
            input_buffer = input_signal[b_start : min(total_input_samples, b_start + buffer_size)]
            if len(input_buffer) == 0:
                break

            # Resample step
            src_idx = resample_phase
            out_list = []
            while src_idx < len(input_buffer):
                i = int(math.floor(src_idx))
                frac = src_idx - i
                s0 = input_buffer[i]
                s1 = input_buffer[i + 1] if (i + 1 < len(input_buffer)) else s0
                out_list.append(s0 + frac * (s1 - s0))
                src_idx += ratio

            resample_phase = src_idx - len(input_buffer)
            float16k = np.array(out_list, dtype=np.float32)

            # Combine with residual
            combined = np.concatenate([residual_samples, float16k])

            # Chunking
            offset = 0
            while len(combined) - offset >= chunk_size:
                emitted_samples += chunk_size
                offset += chunk_size

            residual_samples = combined[offset:]

        # Add remaining residual at stream end
        total_emitted_plus_residual = emitted_samples + len(residual_samples)
        expected_16k_samples = int(total_input_sec * 16000)

        # Tolerance within +/- 2 samples across 3 full seconds (fractional phase boundary)
        assert abs(total_emitted_plus_residual - expected_16k_samples) <= 2, (
            f"At {input_sr}Hz, expected ~{expected_16k_samples} samples, got {total_emitted_plus_residual}"
        )


def test_normalize_speech_config_controls():
    """Verify normalize_speech respects config bypass, selective boost, and safe max_gain limit."""
    # 1. Low energy speech (< 0.05 RMS): should be boosted up to max_gain = 3.0
    low_speech = np.array([0.02, -0.02] * 1000, dtype=np.float32)  # rms = 0.02
    config.asr.normalize_speech = True
    config.asr.normalize_min_rms_to_boost = 0.05
    config.asr.normalize_target_rms = 0.12
    config.asr.normalize_max_gain = 3.0
    config.debug.bypass_speech_normalization = False

    norm_low = TranscribeEngine.normalize_speech(low_speech, is_commit=True)
    orig_peak = np.max(np.abs(low_speech))
    norm_peak = np.max(np.abs(norm_low))
    gain_ratio = norm_peak / orig_peak
    assert 2.9 <= gain_ratio <= 3.001, f"Gain ratio {gain_ratio} should be ~3.0 for low speech"

    # 2. Adequately loud speech (>= 0.05 RMS): should maintain unity gain (1.0) without boosting noise
    loud_speech = np.array([0.08, -0.08] * 1000, dtype=np.float32)  # rms = 0.08 >= 0.05
    norm_loud = TranscribeEngine.normalize_speech(loud_speech, is_commit=True)
    np.testing.assert_allclose(norm_loud, loud_speech, atol=1e-5)

    # 3. Peak Limiter: transient spikes exceeding 0.95 are compressed
    spike_speech = np.array([0.02, -0.02] * 1000, dtype=np.float32)
    spike_speech[100] = 0.8  # When boosted x3, would exceed 2.4 without limiter
    norm_spike = TranscribeEngine.normalize_speech(spike_speech, is_commit=True)
    assert np.max(np.abs(norm_spike)) <= 0.9501, "Peak Limiter must keep max amplitude <= target_peak"

    # 4. Bypass via config.debug.bypass_speech_normalization
    config.debug.bypass_speech_normalization = True
    norm_bypassed = TranscribeEngine.normalize_speech(low_speech)
    np.testing.assert_array_equal(norm_bypassed, low_speech)

    # Reset
    config.debug.bypass_speech_normalization = False
    config.asr.normalize_speech = True


def test_asr_tail_audio_preservation():
    """Verify that on_speech_end executes full inference on pcm_combined if diff_samples > 0."""
    from unittest.mock import MagicMock

    engine = TranscribeEngine(session_id="test_tail")
    mock_run = MagicMock(return_value="full recognized sentence")
    engine._commit_sync = MagicMock()

    # Pre-populate state simulating preview has run on 16000 samples (1.0s)
    # But speech ended with 22400 samples (1.4s, e.g. 400ms trailing speech)
    with engine._state_lock:
        engine._is_speech_active = True
        engine._last_partial_text = "preview sentence"
        engine._last_partial_samples = 16000
        engine._current_utterance_id = "utt-12345"

    engine._audio_buffer_mgr.feed_bytes(b"\x00" * (22400 * 2))
    engine._loop = None

    # Call on_speech_end
    engine.on_speech_end(reason="VAD_SILENCE")

    # Wait briefly for background thread pool executor
    import time
    for _ in range(200):
        if engine._commit_sync.called:
            break
        time.sleep(0.01)

    # Verify that final_cached was None, forcing full commit inference
    assert engine._commit_sync.called
    args, kwargs = engine._commit_sync.call_args
    # args: (pcm_combined, utt_id, reason, final_cached)
    final_cached = args[3]
    assert final_cached is None, "Tail audio (> 0 diff samples) must NOT reuse preview text"


def test_audio_dumper_wav_generation():
    """Verify audio dumper correctly creates valid 16kHz WAV files with sequential utterance naming."""
    session_id = "test_dump_session"
    with tempfile.TemporaryDirectory() as tmpdir:
        config.debug.dump_audio = True
        config.debug.dump_dir = tmpdir

        # 1. Ingress dump
        pcm_bytes = b"\x00\x01" * 1600  # 100ms
        dump_ingress_chunk(session_id, 0, pcm_bytes)
        dump_ingress_chunk(session_id, 1, pcm_bytes)

        pcm_f32 = np.zeros(1600, dtype=np.float32)

        # 2. Utterance 1: Preview -> VAD -> Commit
        dump_asr_input(session_id, "utt-abc12345", pcm_f32, is_commit=False)
        dump_vad_utterance(session_id, "utt-abc12345", pcm_bytes * 5, reason="VAD_SILENCE")
        dump_asr_input(session_id, "utt-abc12345", pcm_f32, is_commit=True)

        # 3. Utterance 2: VAD -> Commit
        dump_vad_utterance(session_id, "utt-xyz98765", pcm_bytes * 5, reason="MAX_DURATION")
        dump_asr_input(session_id, "utt-xyz98765", pcm_f32, is_commit=True)

        close_session_dumper(session_id)
        flush_dumper(5.0)

        sess_dir = Path(tmpdir) / session_id
        wav_files = sorted([f.name for f in sess_dir.glob("*.wav")])

        # Verify exact sequential naming pattern
        assert "00_ingress_stream.wav" in wav_files
        assert "01_utt-abc1_asr_preview_01.wav" in wav_files
        assert "01_utt-abc1_vad_VAD_SILENCE.wav" in wav_files
        assert "01_utt-abc1_asr_commit.wav" in wav_files
        assert "02_utt-xyz9_vad_MAX_DURATION.wav" in wav_files
        assert "02_utt-xyz9_asr_commit.wav" in wav_files

        for wf_name in wav_files:
            wf_path = sess_dir / wf_name
            with wave.open(str(wf_path), "rb") as wf:
                assert wf.getframerate() == 16000
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getnframes() > 0

        config.debug.dump_audio = False


def test_min_words_to_commit_preservation():
    """Verify that min_words_to_commit filtering logic is preserved as designed."""
    # Latin words
    assert count_content_tokens("hello world") == 2
    assert count_content_tokens("hi") == 1
    assert count_content_tokens("   ") == 0

    # CJK characters (each counted as a content token)
    assert count_content_tokens("はい") == 2
    assert count_content_tokens("うん") == 2
    assert count_content_tokens("おはよう") == 4
    assert count_content_tokens("何？") == 1  # 1 CJK char + 1 punct
    assert count_content_tokens("こんにちは世界") == 7


def test_utterance_level_and_snr_logging(caplog):
    """Verify that on_speech_end logs [ASR UTT LEVEL] with duration, RMS, peak, and estimated SNR."""
    import logging
    caplog.set_level(logging.INFO)

    engine = TranscribeEngine("qwen3-asr-1.7b")
    engine._commit_sync = MagicMock()

    with engine._state_lock:
        engine._is_speech_active = True
        engine._current_utterance_id = "utt-test-snr"

    # Construct synthetic audio: 100ms silence head (1600 samples) + 300ms 0.1 amplitude tone + 100ms silence tail (1600 samples)
    # Total: 500ms = 8000 samples = 16000 bytes
    head = np.zeros(1600, dtype=np.float32)
    mid = np.ones(4800, dtype=np.float32) * 0.1
    tail = np.zeros(1600, dtype=np.float32)
    pcm = np.concatenate([head, mid, tail])
    pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()

    engine._audio_buffer_mgr.feed_bytes(pcm_bytes)
    engine.on_speech_end(reason="VAD_SILENCE")

    # Check for [ASR UTT LEVEL] in logs (short utterance -> (dur<2s))
    utt_logs = [rec.message for rec in caplog.records if "[ASR UTT LEVEL]" in rec.message]
    assert len(utt_logs) >= 1, f"Expected [ASR UTT LEVEL] log, found: {caplog.text}"
    log_msg = utt_logs[0]
    assert "utt=utt-test" in log_msg
    assert "dur=0.50s" in log_msg
    assert "snr~=" in log_msg
    assert "(dur<2s)" in log_msg
    assert "reason=VAD_SILENCE" in log_msg

    # Test long utterance (2.5s with clean speech -> (reliable))
    long_head = np.zeros(3200, dtype=np.float32)  # 200ms
    long_mid = np.ones(33600, dtype=np.float32) * 0.1  # 2100ms
    long_tail = np.zeros(3200, dtype=np.float32)  # 200ms
    long_pcm = np.concatenate([long_head, long_mid, long_tail])
    long_bytes = (long_pcm * 32767.0).astype(np.int16).tobytes()

    engine._audio_buffer_mgr.feed_bytes(long_bytes)
    engine.on_speech_end(reason="VAD_SILENCE")

    long_utt_logs = [rec.message for rec in caplog.records if "[ASR UTT LEVEL]" in rec.message]
    assert len(long_utt_logs) >= 2
    long_log_msg = long_utt_logs[1]
    assert "dur=2.50s" in long_log_msg
    assert "(reliable)" in long_log_msg


def test_vad_utterance_dump_skips_conversion_when_disabled(monkeypatch):
    """N-01: no float32 -> int16 conversion may happen when dump_audio is disabled.

    The audit found ``on_speech_end()`` always building the int16 dump buffer even
    though ``dump_vad_utterance()`` immediately discarded it when dumping was off.
    That cost 4 full-array passes + a ~256 KB allocation on every committed sentence.
    """
    import backend_cpp.utils.audio_dumper as dumper

    conversions = []
    monkeypatch.setattr(
        dumper,
        "float32_to_pcm16_bytes",
        lambda pcm: conversions.append(len(pcm)) or b"",
    )
    monkeypatch.setattr(config.debug, "dump_audio", False)

    dumper.dump_vad_utterance_f32("sess", "utt-1", np.zeros(16000, dtype=np.float32))
    dumper.flush_dumper(2.0)

    assert conversions == [], "conversion ran even though dump_audio was disabled"


def test_vad_utterance_dump_converts_when_enabled(monkeypatch):
    """The deferred conversion must still run (on the worker thread) when dumping is on."""
    import backend_cpp.utils.audio_dumper as dumper

    conversions = []
    monkeypatch.setattr(
        dumper,
        "float32_to_pcm16_bytes",
        lambda pcm: conversions.append(len(pcm)) or b"",
    )
    monkeypatch.setattr(config.debug, "dump_audio", True)

    dumper.dump_vad_utterance_f32("sess", "utt-1", np.zeros(16000, dtype=np.float32))
    dumper.flush_dumper(2.0)

    assert conversions == [16000]


def test_on_speech_end_defers_dump_conversion(monkeypatch):
    """on_speech_end() must hand float32 to the dumper, not pre-converted int16 bytes."""
    import backend_cpp.asr.transcribe_engine as transcribe_engine
    import backend_cpp.utils.audio_dumper as dumper

    conversions = []
    monkeypatch.setattr(
        dumper,
        "float32_to_pcm16_bytes",
        lambda pcm: conversions.append(len(pcm)) or b"",
    )

    received = []
    monkeypatch.setattr(
        transcribe_engine,
        "dump_vad_utterance_f32",
        lambda session_id, utt_id, pcm, reason="VAD": received.append(
            (utt_id, pcm.dtype, len(pcm))
        ),
    )
    monkeypatch.setattr(config.debug, "dump_audio", False)

    # Neutralize the commit path: this test is about the dump hook, and letting the
    # commit run would load a real native ASR model.
    monkeypatch.setattr(
        transcribe_engine.TranscribeEngine, "_commit_sync", lambda *a, **k: None
    )
    monkeypatch.setattr(
        transcribe_engine.TranscribeEngine, "_commit_async", lambda *a, **k: None
    )

    engine = TranscribeEngine()
    engine.on_speech_start()
    engine.feed_audio((np.ones(8000, dtype=np.int16) * 1000).tobytes(), vad_state=1)
    engine.on_speech_end(reason="VAD_SILENCE")

    assert received, "on_speech_end did not call the dumper hook"
    _, dtype, sample_count = received[0]
    assert dtype == np.float32
    assert sample_count == 8000
    assert conversions == [], "int16 conversion ran on the commit hot path"
