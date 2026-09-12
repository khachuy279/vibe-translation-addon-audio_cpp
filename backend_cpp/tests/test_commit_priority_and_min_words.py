"""Unit tests verifying Commit Priority locking and min_words_to_commit filtering."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend_cpp.asr.sentence_segmenter import SentenceSegmenter, count_content_tokens
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.ws.session_state import SessionState
from backend_cpp.ws.ws_handler import _stream_asr_tokens


def test_count_content_tokens():
    assert count_content_tokens("Um") == 1
    assert count_content_tokens("Yes, okay") == 2
    assert count_content_tokens("Hello world here") == 3
    assert count_content_tokens("Chào bạn nhé") == 3
    assert count_content_tokens("こんにちは世界") == 7  # CJK chars


def test_segmenter_is_text_filtered_with_min_words():
    segmenter = SentenceSegmenter(min_words_to_commit=3)
    assert segmenter.is_text_filtered("Um") is True
    assert segmenter.is_text_filtered("Okay so") is True
    assert segmenter.is_text_filtered("This is fine") is False


def test_segmenter_is_text_filtered_with_zero():
    """Verify min_words_to_commit = 0 disables filtering completely."""
    segmenter = SentenceSegmenter(min_words_to_commit=0)
    assert segmenter.is_text_filtered("Um") is False
    assert segmenter.is_text_filtered("A") is False
    assert segmenter.is_text_filtered("Hello world") is False


def test_commit_priority_lock_state():
    """Verify preview yields when a commit is marked waiting."""
    with TranscribeEngine._commit_lock:
        TranscribeEngine._commit_waiting = 1

    engine = TranscribeEngine.__new__(TranscribeEngine)
    engine._audio_buffer_mgr = MagicMock()
    engine._language = "en"
    engine.model_info = {}

    pcm = np.zeros(16000, dtype=np.float32)
    # Preview should return empty immediately because commit_waiting > 0
    res = engine._run_inference(pcm, is_commit=False)
    assert res == ""

    # Clean up
    with TranscribeEngine._commit_lock:
        TranscribeEngine._commit_waiting = 0


@pytest.mark.asyncio
async def test_ws_handler_drops_short_utterance_for_translation():
    """Verify _stream_asr_tokens does not put short utterance (< min_words) into translation queue."""
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()
    mock_ws.send_bytes = AsyncMock()
    session = SessionState(mock_ws)
    session.init_components()
    session.config["min_words_to_commit"] = 3

    # Mock engine stream yielding a short final utterance ("Yeah")
    async def mock_stream():
        yield {
            "type": "utterance_update",
            "utterance_id": "utt_short",
            "text": "Yeah",
            "is_final": True,
            "language": "en",
            "epoch": 0,
        }
        yield {
            "type": "utterance_update",
            "utterance_id": "utt_valid",
            "text": "Hello how are you",
            "is_final": True,
            "language": "en",
            "epoch": 0,
        }

    session.asr_engine.stream_tokens = mock_stream

    # Run _stream_asr_tokens until mock stream finishes
    await _stream_asr_tokens(session)

    # Queue should only contain the valid utterance ("Hello how are you"), "Yeah" was dropped
    assert session.translation_queue.qsize() == 1
    queued_item = session.translation_queue.get_nowait()
    assert queued_item["text"] == "Hello how are you"
