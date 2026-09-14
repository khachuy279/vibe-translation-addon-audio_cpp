"""Exhaustive unit test suite for SentenceConfig parameters and boundary mechanics.

Covers:
1. Default configuration invariants and field constraints.
2. Engine & Segmenter initialization synchronization.
3. Dynamic runtime update (hot-reload) via update_sentence_config and SessionState.apply_config.
4. VAD-paced soft boundary state machine:
   - NORMAL -> FORCED_PENDING transition at max_duration_sec.
   - Acoustic silence detection -> MAX_DURATION_SAFE commit.
   - Grace period expiration -> MAX_DURATION_EMERGENCY commit.
   - Fallback direct cut when max_duration_require_silence=False.
5. Stability split accumulator (duration & poll threshold).
6. Emission gating for short turns and punctuation filtering under min_words_to_emit_final=1.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend_cpp.asr.audio_buffer import VAD_STATE_NON_SPEECH, VAD_STATE_SPEECH
from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter, count_content_tokens
from backend_cpp.asr.transcribe_engine import BoundaryState, TranscribeEngine
from backend_cpp.config import SentenceConfig, config
from backend_cpp.ws.session_state import SessionState


def test_sentence_config_defaults():
    """Verify production default values in SentenceConfig."""
    cfg = SentenceConfig()
    assert cfg.max_chars == 150
    assert cfg.max_duration_sec == 15.0
    assert cfg.max_duration_grace_sec == 2.0
    assert cfg.max_duration_require_silence is True
    assert cfg.boundary_candidate_silence_ms == 80
    assert cfg.min_words_to_commit == 1
    assert cfg.min_words_to_emit_final == 1
    assert cfg.split_on_stability is True
    assert cfg.stability_duration_sec == 2.0
    assert cfg.stability_threshold_polls == 2


def test_engine_initialization_syncs_with_sentence_config():
    """Verify TranscribeEngine properly copies SentenceConfig to itself and segmenter."""
    engine = TranscribeEngine(session_id="test_init_sync")
    try:
        assert engine.sentence_config.max_chars == config.sentence.max_chars
        assert engine.sentence_config.max_duration_sec == config.sentence.max_duration_sec
        assert engine.sentence_config.min_words_to_commit == config.sentence.min_words_to_commit
        assert engine.sentence_config.min_words_to_emit_final == config.sentence.min_words_to_emit_final
        assert engine.sentence_config.split_on_stability == config.sentence.split_on_stability
        assert engine.sentence_config.stability_duration_sec == config.sentence.stability_duration_sec
        assert engine.sentence_config.stability_threshold_polls == config.sentence.stability_threshold_polls

        # Internal segmenter
        assert engine._segmenter.max_chars == config.sentence.max_chars
        assert engine._segmenter.max_duration_sec == config.sentence.max_duration_sec
        assert engine._segmenter.min_words_to_commit == config.sentence.min_words_to_commit
        assert engine._segmenter.min_words_to_emit_final == config.sentence.min_words_to_emit_final
        assert engine._segmenter.split_on_stability == config.sentence.split_on_stability
        assert engine._segmenter.stability_duration_sec == config.sentence.stability_duration_sec
        assert engine._segmenter.stability_threshold_polls == config.sentence.stability_threshold_polls
    finally:
        asyncio.run(engine.cleanup())


def test_update_sentence_config_hot_reload():
    """Verify update_sentence_config dynamically reconfigures both engine and segmenter."""
    engine = TranscribeEngine(session_id="test_hot_reload")
    try:
        engine.update_sentence_config(
            split_on_stability=False,
            stability_duration_sec=1.5,
            max_duration_sec=12.0,
            max_duration_grace_sec=3.0,
            max_duration_require_silence=False,
            boundary_candidate_silence_ms=120,
            max_chars=100,
            min_words_to_commit=2,
            min_words_to_emit_final=1,
        )

        assert engine.sentence_config.split_on_stability is False
        assert engine.sentence_config.stability_duration_sec == 1.5
        assert engine.sentence_config.max_duration_sec == 12.0
        assert engine.sentence_config.max_duration_grace_sec == 3.0
        assert engine.sentence_config.max_duration_require_silence is False
        assert engine.sentence_config.boundary_candidate_silence_ms == 120
        assert engine.sentence_config.max_chars == 100
        assert engine.sentence_config.min_words_to_commit == 2
        assert engine.sentence_config.min_words_to_emit_final == 1

        # Segmenter mirror check
        assert engine._segmenter.split_on_stability is False
        assert engine._segmenter.stability_duration_sec == 1.5
        assert engine._segmenter.max_chars == 100
        assert engine._segmenter.min_words_to_commit == 2
        assert engine._segmenter.min_words_to_emit_final == 1
    finally:
        asyncio.run(engine.cleanup())


def test_session_state_apply_config_sentence_updates():
    """Verify SessionState.apply_config routes camelCase payload to engine."""
    session = SessionState(MagicMock())
    session.init_components()

    payload = {
        "maxChars": 120,
        "maxDurationSec": 10.0,
        "maxDurationGraceSec": 2.5,
        "maxDurationRequireSilence": True,
        "boundaryCandidateSilenceMs": 100,
        "minWordsToCommit": 3,
        "splitOnStability": True,
        "stabilityDurationSec": 1.8,
    }
    session.apply_config(payload)

    sc = session.asr_engine.sentence_config
    assert sc.max_chars == 120
    assert sc.max_duration_sec == 10.0
    assert sc.max_duration_grace_sec == 2.5
    assert sc.max_duration_require_silence is True
    assert sc.boundary_candidate_silence_ms == 100
    assert sc.min_words_to_commit == 3
    assert sc.split_on_stability is True
    assert sc.stability_duration_sec == 1.8


def test_vad_paced_boundary_state_transitions():
    """Test the soft boundary state machine under synthetic audio feed."""
    engine = TranscribeEngine(session_id="test_state_machine")
    engine.sentence_config.max_duration_sec = 2.0
    engine.sentence_config.max_duration_grace_sec = 1.0
    engine.sentence_config.boundary_candidate_silence_ms = 80
    engine.sentence_config.max_duration_require_silence = True

    committed_reasons = []
    orig_on_speech_end = engine.on_speech_end
    def _mock_speech_end(reason="unknown", **kw):
        committed_reasons.append(reason)
        orig_on_speech_end(reason=reason, **kw)

    with patch.object(engine, "on_speech_end", side_effect=_mock_speech_end):
        engine.on_speech_start()
        assert engine._boundary_state == BoundaryState.NORMAL

        # Feed 1.5s speech -> stays NORMAL (< 2.0s)
        chunk_1s = bytes(int(DEFAULT_SAMPLE_RATE * 1.5 * 2))
        engine.feed_audio(chunk_1s, vad_state=VAD_STATE_SPEECH)
        assert engine._boundary_state == BoundaryState.NORMAL
        assert len(committed_reasons) == 0

        # Feed 0.6s speech (total 2.1s >= 2.0s) -> transitions to FORCED_PENDING
        chunk_06s = bytes(int(DEFAULT_SAMPLE_RATE * 0.6 * 2))
        engine.feed_audio(chunk_06s, vad_state=VAD_STATE_SPEECH)
        assert engine._boundary_state == BoundaryState.FORCED_PENDING
        assert len(committed_reasons) == 0

        # Feed 40ms silence (< 80ms) -> remains FORCED_PENDING
        silence_40ms = bytes(int(DEFAULT_SAMPLE_RATE * 0.040 * 2))
        engine.feed_audio(silence_40ms, vad_state=VAD_STATE_NON_SPEECH)
        assert engine._boundary_state == BoundaryState.FORCED_PENDING
        assert len(committed_reasons) == 0

        # Feed another 50ms silence (cumulative 90ms >= 80ms) -> triggers MAX_DURATION_SAFE commit
        silence_50ms = bytes(int(DEFAULT_SAMPLE_RATE * 0.050 * 2))
        engine.feed_audio(silence_50ms, vad_state=VAD_STATE_NON_SPEECH)
        assert "MAX_DURATION_SAFE" in committed_reasons
        assert engine._boundary_state == BoundaryState.NORMAL


def test_vad_paced_boundary_emergency_cut():
    """Test that continuous speech exceeding max_duration + grace triggers EMERGENCY commit."""
    engine = TranscribeEngine(session_id="test_emergency_cut")
    engine.sentence_config.max_duration_sec = 2.0
    engine.sentence_config.max_duration_grace_sec = 1.0
    engine.sentence_config.boundary_candidate_silence_ms = 80
    engine.sentence_config.max_duration_require_silence = True

    committed_reasons = []
    orig_on_speech_end = engine.on_speech_end
    def _mock_speech_end(reason="unknown", **kw):
        committed_reasons.append(reason)
        orig_on_speech_end(reason=reason, **kw)

    with patch.object(engine, "on_speech_end", side_effect=_mock_speech_end):
        engine.on_speech_start()

        # Feed 2.1s speech -> enters FORCED_PENDING
        chunk_2_1s = bytes(int(DEFAULT_SAMPLE_RATE * 2.1 * 2))
        engine.feed_audio(chunk_2_1s, vad_state=VAD_STATE_SPEECH)
        assert engine._boundary_state == BoundaryState.FORCED_PENDING

        # Continuous uninterrupted speech for 1.0s more (total 3.1s >= 2.0 + 1.0)
        chunk_1_0s = bytes(int(DEFAULT_SAMPLE_RATE * 1.0 * 2))
        engine.feed_audio(chunk_1_0s, vad_state=VAD_STATE_SPEECH)

        assert "MAX_DURATION_EMERGENCY" in committed_reasons
        assert engine._boundary_state == BoundaryState.NORMAL


def test_max_duration_require_silence_false_direct_cut():
    """When max_duration_require_silence is False, cut directly at max_duration_sec without grace."""
    engine = TranscribeEngine(session_id="test_direct_cut")
    engine.sentence_config.max_duration_sec = 2.0
    engine.sentence_config.max_duration_grace_sec = 1.0
    engine.sentence_config.max_duration_require_silence = False

    committed_reasons = []
    orig_on_speech_end = engine.on_speech_end
    def _mock_speech_end(reason="unknown", **kw):
        committed_reasons.append(reason)
        orig_on_speech_end(reason=reason, **kw)

    with patch.object(engine, "on_speech_end", side_effect=_mock_speech_end):
        engine.on_speech_start()
        # Feed 2.1s speech -> direct cut immediately
        chunk_2_1s = bytes(int(DEFAULT_SAMPLE_RATE * 2.1 * 2))
        engine.feed_audio(chunk_2_1s, vad_state=VAD_STATE_SPEECH)

        assert "MAX_DURATION_SAFE" in committed_reasons


def test_stability_accumulator_respects_duration_and_poll_count():
    """Verify SentenceSegmenter.check_stability logic."""
    seg = SentenceSegmenter(
        split_on_stability=True,
        stability_duration_sec=0.10,
        stability_threshold_polls=3,
        min_words_to_commit=1,
    )

    # Initial text
    assert seg.check_stability("東京特許許可局") is False
    assert seg._stable_poll_count == 1

    # Poll 2 within 30ms -> stable poll 2, but elapsed < 0.10s and count < 3
    time.sleep(0.03)
    assert seg.check_stability("東京特許許可局") is False
    assert seg._stable_poll_count == 2

    # Poll 3 after duration elapsed -> poll 3 and duration elapsed -> True
    time.sleep(0.08)
    assert seg.check_stability("東京特許許可局") is True

    # If text subsequently changes, resets immediately
    assert seg.check_stability("東京特許許可局局長") is False
    assert seg._stable_poll_count == 1


def test_cjk_and_latin_short_turn_emission():
    """Verify natural short turns pass under min_words_to_emit_final=1 while empty/junk is dropped."""
    seg = SentenceSegmenter(min_words_to_commit=1, min_words_to_emit_final=1)

    # 1-token and short dialogue turns must NOT be filtered
    assert seg.is_final_too_short("はい") is False
    assert seg.is_final_too_short("うん。") is False
    assert seg.is_final_too_short("Yes.") is False
    assert seg.is_final_too_short("OK") is False
    assert seg.is_final_too_short("え？") is False

    # Empty and punctuation-only strings MUST be filtered
    assert seg.is_final_too_short("") is True
    assert seg.is_final_too_short("   ") is True
    assert seg.is_final_too_short("。。。") is True
    assert seg.is_final_too_short("?!") is True
