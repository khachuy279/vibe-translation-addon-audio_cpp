"""Unit tests verifying all fixes from code_review_asr.md."""

import concurrent.futures
import time
import numpy as np
import pytest

from backend_cpp.asr import (
    AudioBufferManager,
    BaseASREngine,
    CommitDeduplicator,
    ModelRegistry,
    TranscribeEngine,
)
from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE
from backend_cpp.config import TranslationConfig


def test_audio_buffer_manager():
    mgr = AudioBufferManager(sample_rate=16000)
    assert mgr.is_empty
    assert mgr.duration_sec == 0.0

    # Feed 0.5s audio (8000 samples int16 = 16000 bytes)
    samples = (np.ones(8000, dtype=np.int16) * 1000).tobytes()
    dur = mgr.feed_bytes(samples)
    assert dur == pytest.approx(0.5, abs=0.01)
    assert not mgr.is_empty

    # Snapshot test
    snapshot = mgr.get_snapshot()
    assert snapshot is not None
    assert snapshot.sample_count == 8000
    assert snapshot.duration_sec == pytest.approx(0.5, abs=0.01)
    # Buffer should not be cleared by snapshot
    assert not mgr.is_empty

    # Slice after 4000 samples
    mgr.slice_after(4000)
    assert mgr.duration_sec == pytest.approx(0.25, abs=0.01)

    # Pop all
    combined, frame_state, pop_dur = mgr.pop_all()
    assert combined is not None
    assert len(combined) == 4000
    assert pop_dur == pytest.approx(0.25, abs=0.01)
    assert mgr.is_empty
    assert mgr.duration_sec == 0.0


def test_audio_buffer_get_snapshot_if_newer():
    mgr = AudioBufferManager(sample_rate=16000)
    # Empty buffer returns None
    assert mgr.get_snapshot_if_newer(0) is None

    # Feed 4000 samples (0.25s)
    chunk1 = (np.ones(4000, dtype=np.int16) * 500).tobytes()
    mgr.feed_bytes(chunk1)

    res1 = mgr.get_snapshot_if_newer(0)
    assert res1 is not None
    assert res1.sample_count == 4000
    assert res1.duration_sec == pytest.approx(0.25, abs=0.01)

    # Calling with same sample count returns None (no new audio)
    assert mgr.get_snapshot_if_newer(res1.sample_count) is None

    # Feed another 4000 samples (total 8000 samples)
    chunk2 = (np.ones(4000, dtype=np.int16) * 500).tobytes()
    mgr.feed_bytes(chunk2)

    # Now get_snapshot_if_newer(4000) must return the new snapshot with 8000 samples
    res2 = mgr.get_snapshot_if_newer(res1.sample_count)
    assert res2 is not None
    assert res2.sample_count == 8000
    assert res2.duration_sec == pytest.approx(0.5, abs=0.01)


def test_commit_deduplicator():
    dedup = CommitDeduplicator(cache_ttl_sec=0.5)

    assert not dedup.is_duplicate("Hello world")
    dedup.record_commit("Hello world", "hello world")

    # Exact match
    assert dedup.is_duplicate("Hello world")
    assert dedup.is_duplicate("hello world!")

    # Substring containment with ratio >= 0.65
    assert dedup.is_duplicate("Oh hello world")

    # Different sentence passes
    assert not dedup.is_duplicate("Something completely different")

    # TTL expiration
    time.sleep(0.55)
    dedup.prune_expired()
    # Expired, so "Hello world" should no longer be a duplicate
    assert not dedup.is_duplicate("Hello world")


def test_model_registry_thread_safety():
    results = []

    def fetch_instance():
        return ModelRegistry.get_instance()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_instance) for _ in range(50)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    first = results[0]
    for inst in results:
        assert inst is first


def test_translation_config_defaults():
    t_cfg = TranslationConfig(base="tencent")
    assert "Hy-MT2" in t_cfg.model
    assert "Hy-MT2" in t_cfg.gguf_file
    assert t_cfg.temperature == 0.7
    assert t_cfg.top_p == 0.6
    assert t_cfg.top_k == 20
    assert t_cfg.repetition_penalty == 1.05

    x_cfg = TranslationConfig(base="xiaomi")
    assert x_cfg.model == "mradermacher/MiLMMT-46-4B-v1.0-GGUF"
    assert x_cfg.gguf_file == "MiLMMT-46-4B-v1.0.Q4_K_M.gguf"
    assert x_cfg.temperature == 0.0
    assert x_cfg.top_p == 1.0
    assert x_cfg.top_k == 1
    assert x_cfg.repetition_penalty == 1.0


def test_base_asr_engine_lsp():
    class DummyEngine(BaseASREngine):
        def feed_audio(self, pcm_bytes: bytes, timestamp: float = 0.0) -> None:
            pass

        def on_speech_start(self) -> None:
            pass

        def on_speech_end(self, reason: str = "VAD_SILENCE") -> None:
            self.last_reason = reason

        async def stream_tokens(self):
            yield {}

        def set_language(self, language: str) -> None:
            pass

        async def cleanup(self) -> None:
            pass

    dummy = DummyEngine()
    dummy.on_speech_end()
    assert dummy.last_reason == "VAD_SILENCE"

    dummy.on_speech_end(reason="MAX_DURATION")
    assert dummy.last_reason == "MAX_DURATION"


def test_audio_buffer_slice_after_preserves_concurrent_chunks():
    mgr = AudioBufferManager(sample_rate=16000)
    # Feed chunk 1: 4000 samples (0.25s)
    chunk1 = (np.ones(4000, dtype=np.int16) * 100).tobytes()
    mgr.feed_bytes(chunk1)

    # Feed chunk 2: 4000 samples (0.25s)
    chunk2 = (np.ones(4000, dtype=np.int16) * 200).tobytes()
    mgr.feed_bytes(chunk2)

    # Now slice_after 2000 samples -> remaining from chunk1+2 is 6000 samples (0.375s)
    mgr.slice_after(2000)
    assert mgr.duration_sec == pytest.approx(0.375, abs=0.01)

    # Feed chunk 3: 8000 samples (0.5s)
    chunk3 = (np.ones(8000, dtype=np.int16) * 300).tobytes()
    mgr.feed_bytes(chunk3)
    assert mgr.duration_sec == pytest.approx(0.875, abs=0.01)

    combined, frame_state, dur = mgr.pop_all()
    assert len(combined) == 14000  # 6000 + 8000
    assert dur == pytest.approx(0.875, abs=0.01)
    assert mgr.is_empty


def test_sentence_segmenter_token_counting():
    from backend_cpp.asr.sentence_segmenter import count_content_tokens
    assert count_content_tokens("") == 0
    assert count_content_tokens("Hello world, this is a test!") == 6
    assert count_content_tokens("What's up-to-date?") == 2
    assert count_content_tokens("你好世界") == 4


def test_audio_buffer_versioning_and_concurrency():
    mgr = AudioBufferManager(sample_rate=16000)
    chunk = (np.ones(4000, dtype=np.int16) * 100).tobytes()
    mgr.feed_bytes(chunk)

    snap = mgr.get_snapshot_with_version()
    assert snap is not None
    assert snap.sample_count == 4000
    assert snap.version == 0
    ver = snap.version

    # Pop all occurs while inference was running
    popped, frame_state, pop_dur = mgr.pop_all()
    assert len(popped) == 4000
    assert mgr.is_empty
    assert mgr.version == 1

    # Stale slice_after must NOT revive cleared audio
    mgr.slice_after(2000, expected_version=ver)
    assert mgr.is_empty
    assert mgr.duration_sec == 0.0


@pytest.mark.asyncio
async def test_transcribe_engine_cleanup_unblocks_consumer():
    engine = TranscribeEngine()

    consumed = []
    async def consumer():
        async for token in engine.stream_tokens():
            consumed.append(token)

    import asyncio
    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)

    # Calling cleanup must unblock consumer via _SENTINEL
    await engine.cleanup()
    await asyncio.wait_for(task, timeout=1.0)
    assert task.done()


def test_check_model_supports_streaming():
    from backend_cpp.asr import check_model_supports_streaming
    assert check_model_supports_streaming("nemotron-3.5-streaming") is True
    assert check_model_supports_streaming("qwen3-asr-1.7b") is False
    assert check_model_supports_streaming("sensevoice-small") is False


def test_streaming_inference_slot_isolation():
    from unittest.mock import MagicMock
    engine = TranscribeEngine(model_key="nemotron-3.5-streaming")
    engine._language = "en-US"

    mock_model = MagicMock()
    mock_model.capabilities.supports_streaming = True
    mock_session = MagicMock()
    mock_session._model = mock_model

    mock_stream = MagicMock()
    mock_stream.text.return_value.full = ""
    mock_session.stream.return_value.__enter__.return_value = mock_stream
    mock_model.session.return_value = mock_session

    TranscribeEngine._shared_model = mock_model
    TranscribeEngine._shared_model_key = "nemotron-3.5-streaming"
    TranscribeEngine._shared_supports_streaming = True
    TranscribeEngine._shared_session = mock_session

    try:
        pcm = np.zeros(16000, dtype=np.float32)
        res = engine._run_inference(pcm, is_commit=True)
        assert res == ""
        assert mock_session.stream.called
        # session.run must NOT be called when session.stream succeeded
        assert not mock_session.run.called
    finally:
        # Clean up
        TranscribeEngine._shared_model = None
        TranscribeEngine._shared_model_key = None
        TranscribeEngine._shared_session = None


def test_normalize_speech_purity_and_limiter():
    # 1. Pure function test: caller buffer must NOT be mutated in-place
    original_pcm = np.array([0.01, -0.01, 0.02, -0.02] * 2000, dtype=np.float32)
    pcm_copy = original_pcm.copy()
    normalized = TranscribeEngine.normalize_speech(original_pcm)

    np.testing.assert_array_equal(original_pcm, pcm_copy)
    assert normalized is not original_pcm
    assert np.max(np.abs(normalized)) > np.max(np.abs(original_pcm))

    # 2. Peak limiter test: peaks exceeding target_peak must be safely compressed
    spike_pcm = np.array([0.05, -0.05, 0.99, -0.99] * 2000, dtype=np.float32)
    norm_spikes = TranscribeEngine.normalize_speech(spike_pcm, target_peak=0.95)
    assert np.max(np.abs(norm_spikes)) <= 0.95001


def test_commit_waiting_no_leak_on_exception():
    from unittest.mock import MagicMock
    engine = TranscribeEngine()

    mock_model = MagicMock()
    mock_session = MagicMock()
    mock_session.run.side_effect = RuntimeError("Simulated C++ engine crash")
    TranscribeEngine._shared_model = mock_model
    TranscribeEngine._shared_session = mock_session
    TranscribeEngine._shared_supports_streaming = False

    try:
        pcm = np.ones(16000, dtype=np.float32) * 0.1
        # Before inference: commit_waiting must be 0
        assert TranscribeEngine._commit_waiting == 0

        res = engine._run_inference(pcm, is_commit=True)
        assert res == ""

        # After exception: commit_waiting MUST be safely decremented back to 0!
        assert TranscribeEngine._commit_waiting == 0
    finally:
        TranscribeEngine._shared_model = None
        TranscribeEngine._shared_session = None
        TranscribeEngine._commit_waiting = 0


def test_clean_transcript_text_multiline():
    from backend_cpp.asr.transcribe_engine import clean_transcript_text
    raw = "<|startoftranscript|>\nsystem:\nassistant: Xin chào bạn <|endoftext|>"
    cleaned = clean_transcript_text(raw)
    assert cleaned == "Xin chào bạn"



