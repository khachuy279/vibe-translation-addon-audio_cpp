"""Test VAD multi-session concurrency, state isolation, and sample clock."""

import os
from pathlib import Path
import wave
import pytest

from backend_cpp.vad.vad_processor import VADProcessor

TEST_WAV = Path(__file__).resolve().parent.parent.parent / "wav_test" / "OSR_us_000_0010_16k.wav"


def _load_test_speech(start_sec: float = 1.0, duration_sec: float = 1.0) -> bytes:
    """Extract sample speech from the test WAV file."""
    assert TEST_WAV.exists(), f"Test wav not found at: {TEST_WAV}"
    with wave.open(str(TEST_WAV), "rb") as wf:
        wf.setpos(int(16000 * start_sec))
        return wf.readframes(int(16000 * duration_sec))


def _generate_silence(duration_sec: float, sample_rate: int = 16000) -> bytes:
    """Generate 16-bit mono PCM silence."""
    return b"\x00" * int(sample_rate * 2 * duration_sec)


@pytest.mark.parametrize("engine_name", ["fsmn-vad", "silero-vad", "firered-vad"])
def test_vad_session_isolation(engine_name: str):
    """Verify that multiple concurrent VADProcessor instances maintain independent state."""
    session1_started = []
    session1_ended = []
    session1_chunks = []

    session2_started = []
    session2_ended = []
    session2_chunks = []

    vad1 = VADProcessor(
        vad_engine=engine_name,
        silence_duration_ms=250,
        hangover_ms=100,
        on_speech_start=lambda: session1_started.append(True),
        on_speech_end=lambda: session1_ended.append(True),
        on_speech_chunk=lambda chunk, ts: session1_chunks.append(chunk),
    )

    vad2 = VADProcessor(
        vad_engine=engine_name,
        silence_duration_ms=250,
        hangover_ms=100,
        on_speech_start=lambda: session2_started.append(True),
        on_speech_end=lambda: session2_ended.append(True),
        on_speech_chunk=lambda chunk, ts: session2_chunks.append(chunk),
    )

    # 1. Feed speech to session 1
    speech_data = _load_test_speech(start_sec=1.0, duration_sec=0.8)
    chunk_size = 1024  # 32ms
    for i in range(0, len(speech_data), chunk_size):
        vad1.feed_chunk(speech_data[i : i + chunk_size])

    assert len(session1_started) >= 1, f"Session 1 should have detected speech onset on {engine_name}"
    assert len(session2_started) == 0, f"Session 2 should be completely untouched on {engine_name}"

    # 2. Concurrently feed silence to session 2 and call force_end() on session 2
    silence_data = _generate_silence(0.5)
    for i in range(0, len(silence_data), chunk_size):
        vad2.feed_chunk(silence_data[i : i + chunk_size])

    vad2.force_end()

    # Verify session 1 is still in speech state and was NOT wiped out by session 2 force_end
    assert vad1._state.is_speech is True, f"Session 1 state must remain speech after Session 2 force_end on {engine_name}"

    # 3. Session 1 continues receiving speech
    more_speech = _load_test_speech(start_sec=1.8, duration_sec=0.4)
    prev_chunk_count = len(session1_chunks)
    for i in range(0, len(more_speech), chunk_size):
        vad1.feed_chunk(more_speech[i : i + chunk_size])

    assert len(session1_chunks) > prev_chunk_count, "Session 1 should continue receiving speech chunks"

    # 4. Feed silence to session 1 to trigger speech end
    end_silence = _generate_silence(0.8)
    for i in range(0, len(end_silence), chunk_size):
        vad1.feed_chunk(end_silence[i : i + chunk_size])

    assert len(session1_ended) >= 1, f"Session 1 should trigger speech end on {engine_name}"


def test_vad_audio_sample_clock_burst():
    """Verify that a large burst of audio triggers speech start and end deterministically."""
    speech_ended = []

    vad = VADProcessor(
        vad_engine="silero-vad",
        silence_duration_ms=250,
        hangover_ms=100,
        on_speech_end=lambda: speech_ended.append(True),
    )

    # Combine 1s speech followed by 800ms silence into ONE single burst chunk (1.8s total)
    speech = _load_test_speech(start_sec=1.0, duration_sec=1.0)
    silence = _generate_silence(0.8)
    burst = speech + silence

    # Feed entire 1.8s burst in a single feed_chunk call
    vad.feed_chunk(burst)

    # With audio sample clock, the silence threshold (250ms) is reached inside the chunk loop
    assert len(speech_ended) == 1, "VAD should trigger speech end inside the burst processing via Sample Clock"


def test_vad_force_end_outside_lock():
    """Verify force_end calls on_speech_end outside self._lock (reentrant-safe)."""
    callback_executed = []

    def on_end():
        # Calling vad.reset() requires self._lock; if called while holding lock, this would deadlock
        vad.reset()
        callback_executed.append(True)

    vad = VADProcessor(
        vad_engine="silero-vad",
        on_speech_end=on_end,
    )

    # Set speech state
    with vad._lock:
        vad._state.is_speech = True

    vad.force_end()
    assert len(callback_executed) == 1
    assert vad._state.is_speech is False


def test_vad_engine_factory_official_engines():
    """Verify VADEngineFactory loads official engines from backend_cpp/models."""
    from backend_cpp.vad.engines import (
        VADEngineFactory,
        FireRedOfficialVADEngine,
        SileroOfficialVADEngine,
        FsmnOfficialVADEngine,
    )

    VADEngineFactory.reset_pool()

    engine_firered = VADEngineFactory.get_engine("firered-vad")
    assert isinstance(engine_firered, FireRedOfficialVADEngine)
    assert engine_firered.native_frame_samples == 400

    engine_silero = VADEngineFactory.get_engine("silero-vad")
    assert isinstance(engine_silero, SileroOfficialVADEngine)
    assert engine_silero.native_frame_samples == 512

    engine_fsmn = VADEngineFactory.get_engine("fsmn-vad")
    assert isinstance(engine_fsmn, FsmnOfficialVADEngine)
    assert engine_fsmn.native_frame_samples == 960

    # Singleton check
    assert VADEngineFactory.get_engine("firered-vad") is engine_firered
    assert VADEngineFactory.get_engine("silero-vad") is engine_silero
    assert VADEngineFactory.get_engine("fsmn-vad") is engine_fsmn

    VADEngineFactory.reset_pool()


def test_unified_vad_threshold_and_live_update():
    """Verify shared threshold in VADConfig and live dynamic threshold updates during streaming."""
    from backend_cpp.config import VADConfig

    # 1. Verify VADConfig threshold synchronization across all sub-configs
    cfg = VADConfig(threshold=0.55)
    assert cfg.firered.threshold == 0.55
    assert cfg.silero.threshold == 0.55
    assert cfg.fsmn.speech_noise_thres == 0.55
    assert cfg.fsmn.threshold == 0.55

    # Test dynamic mutation
    cfg.threshold = 0.72
    assert cfg.firered.threshold == 0.72
    assert cfg.silero.threshold == 0.72
    assert cfg.fsmn.speech_noise_thres == 0.72
    assert cfg.fsmn.threshold == 0.72

    # 2. Verify live threshold update on active VADProcessor for all 3 engines
    for engine_name in ["firered-vad", "silero-vad", "fsmn-vad"]:
        vad = VADProcessor(vad_engine=engine_name, threshold=0.40)
        # Feed 1 silent frame to initialize state
        silence_chunk = b"\x00" * (vad._frame_samples * 2)
        vad.feed_chunk(silence_chunk)

        assert vad.threshold == 0.40

        # Simulate live threshold slider update from popup
        vad.update_config(threshold=0.75)
        assert vad.threshold == 0.75

        # Feed next chunk and verify the engine's internal detector updated in real time
        vad.feed_chunk(silence_chunk)

        if engine_name == "firered-vad":
            assert vad._state.firered_postprocessor.speech_threshold == 0.75
        elif engine_name == "silero-vad":
            assert vad._state.silero_iterator.threshold == 0.75
        elif engine_name == "fsmn-vad":
            assert vad._state.fsmn_cache["stats"].speech_noise_thres == 0.75


def test_vad_bounded_buffer_overflow_protection():
    """Verify raw_buffer does not exceed MAX_BUFFER_BYTES (96000 bytes = 3s)."""
    vad = VADProcessor(vad_engine="silero-vad")

    # 5 seconds of audio = 160,000 bytes
    huge_data = _generate_silence(5.0)

    # Feed chunk
    vad.feed_chunk(huge_data)

    max_expected = int(16000 * 2 * 3.0)  # 96,000 bytes
    assert len(vad._state.raw_buffer) < vad._frame_size_bytes, "All complete frames should have been processed"

    # Also test incomplete frames left over do not exceed buffer limit
    vad._state.raw_buffer.extend(b"\x00" * 120000)
    # Trigger buffer bound by feeding 1 byte
    vad.feed_chunk(b"\x00")
    assert len(vad._state.raw_buffer) <= max_expected


def test_vad_real_continuous_probabilities():
    """Verify all 3 engines output authentic continuous probabilities (0.0 to 1.0) and VADResult unpacking."""
    from backend_cpp.vad.engines import VADEngineFactory, VADResult
    import numpy as np

    speech_pcm = _load_test_speech(start_sec=1.0, duration_sec=0.5)
    speech_int16 = np.frombuffer(speech_pcm, dtype=np.int16)
    speech_float32 = speech_int16.astype(np.float32) / 32768.0

    for engine_name in ["silero-vad", "fsmn-vad", "firered-vad"]:
        engine = VADEngineFactory.get_engine(engine_name)
        state = engine.create_initial_state()
        frame_samples = engine.native_frame_samples

        # Test a speech frame
        frame = speech_float32[:frame_samples]
        res = engine.is_speech(frame, state, threshold=0.5)

        assert isinstance(res, VADResult)
        assert len(res) == 3

        # Test tuple unpacking
        is_speech, prob, event = res
        assert isinstance(is_speech, (bool, np.bool_))
        assert isinstance(prob, (float, np.floating))
        assert 0.0 <= prob <= 1.0

        # Feed silence frame
        silence_frame = np.zeros(frame_samples, dtype=np.float32)
        res_silence = engine.is_speech(silence_frame, state, threshold=0.5)
        assert 0.0 <= res_silence.probability <= 1.0


def test_vad_per_frame_monotonic_timestamps():
    """Verify that speech chunks output timestamps with per-frame offset precision."""
    received_timestamps = []

    vad = VADProcessor(
        vad_engine="silero-vad",
        on_speech_chunk=lambda chunk, ts: received_timestamps.append(ts),
    )

    speech = _load_test_speech(start_sec=1.0, duration_sec=0.5)
    base_ts = 1500.0
    vad.feed_chunk(speech, capture_timestamp=base_ts)

    assert len(received_timestamps) > 1
    # Check that timestamps strictly increase
    for i in range(len(received_timestamps) - 1):
        assert received_timestamps[i + 1] >= received_timestamps[i]
    # Check that timestamp range reflects chunk duration (within 0.5s)
    assert received_timestamps[0] >= base_ts
    assert received_timestamps[-1] < base_ts + 1.0


def test_vad_stream_state_clean_reset():
    """Verify VADStreamState.reset() cleans all state attributes comprehensively."""
    from backend_cpp.vad.engines import VADEngineFactory
    import numpy as np

    for engine_name in ["silero-vad", "fsmn-vad", "firered-vad"]:
        engine = VADEngineFactory.get_engine(engine_name)
        state = engine.create_initial_state()

        # Simulate active state
        state.is_speech = True
        state.silence_samples = 800
        state.raw_buffer.extend(b"\x01\x02\x03\x04")
        state.pre_speech_ring.append((b"\x00" * 800, 1.0))

        # Feed one frame to populate internal caches
        silence_frame = np.zeros(engine.native_frame_samples, dtype=np.float32)
        engine.is_speech(silence_frame, state, threshold=0.5)


        # Call reset
        state.reset()

        assert state.is_speech is False
        assert state.silence_samples == 0
        assert len(state.raw_buffer) == 0
        assert len(state.pre_speech_ring) == 0
        assert state.firered_caches is None
        assert state.fsmn_cache is None
        if state.silero_probe:
            assert state.silero_probe.last_prob == 0.0


def test_vad_processor_metrics_and_stats():
    """Verify VADProcessor operational metrics and get_stats diagnostics."""
    vad = VADProcessor(
        vad_engine="silero-vad",
        on_speech_chunk=lambda chunk, ts: None,
    )

    stats0 = vad.get_stats()
    assert stats0["engine"] == "silero-vad"
    assert stats0["overflow_count"] == 0
    assert stats0["total_samples_processed"] == 0

    # Feed 0.5s speech
    speech = _load_test_speech(start_sec=1.0, duration_sec=0.5)
    vad.feed_chunk(speech, capture_timestamp=1.0)

    stats1 = vad.get_stats()
    assert stats1["total_samples_processed"] > 0
    assert stats1["audio_seconds_processed"] > 0.0
    assert stats1["callback_count"] > 0

    # Simulate buffer overflow
    vad._state.raw_buffer.extend(b"\x00" * 100000)
    vad.feed_chunk(b"\x00")
    stats2 = vad.get_stats()
    assert stats2["overflow_count"] >= 1




