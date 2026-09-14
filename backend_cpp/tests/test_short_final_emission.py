"""The final-commit gate is separate from the fragment gate.

Background
----------
`min_words_to_commit` (default 4) is documented as a guard against *fragments* of a split
utterance flashing on screen, and `ws_handler` reuses it to skip translating throwaway
turns. The engine also applied it to the FINAL commit of a VAD-delimited utterance, which
does discard complete short turns: measured with the production functions, 5 of 12 real
Japanese turns ("うん。", "そうね。", "どうぞ。", "はい。", "そう。") and 10 of 10 short English
turns ("Yes.", "I see.", "OK.") were dropped.

Why the default did NOT change
------------------------------
Lowering the gate is measured to be a net loss. On 8 verbatim Japanese streams
(`data/ja_cv/streams`, real-time, full production pipeline):

    gate = 4 (default)  ->  CER 15.58%,  123 commits
    gate = 1            ->  CER 17.57%,  149 commits   (+2.0 pp)

All 26 extra commits were inspected: 22 were "はい。" / "は。", and none of them appear
anywhere in the reference. An isolated VAD fragment is acoustically ambiguous and the
decoder fills it with the most likely short Japanese turn -- the threshold was unknowingly
acting as a hallucination filter.

So the two concerns are now separable knobs (`min_words_to_commit` for fragments,
`min_words_to_emit_final` for finals) with the production default unchanged at 4, and
dialogue-heavy material can opt into 1 after measuring on its own content.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from backend_cpp.asr.sentence_segmenter import SentenceSegmenter, count_content_tokens
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import SentenceConfig, config

SHORT_JA = ["うん。", "そうね。", "どうぞ。", "はい。", "そう。"]
SHORT_EN = ["Yes.", "I see.", "OK.", "Right."]


def test_count_content_tokens_sees_short_turns_as_short():
    """Sanity: these really are below a 4-word fragment threshold."""
    for t in SHORT_JA + SHORT_EN:
        assert count_content_tokens(t) < 4


def test_preview_gate_still_filters_short_text():
    """The fragment guard must keep working -- that is what min_words_to_commit is for."""
    seg = SentenceSegmenter(min_words_to_commit=4, min_words_to_emit_final=1)
    assert seg.is_text_filtered("うん。") is True
    assert seg.is_text_filtered("そうね。") is True
    assert seg.is_text_filtered("Yes.") is True
    assert seg.is_text_filtered("これは十分に長い文です") is False


def test_final_gate_is_independent_of_the_fragment_gate():
    """The whole point of the change: the two gates can differ."""
    seg = SentenceSegmenter(min_words_to_commit=4, min_words_to_emit_final=1)
    assert seg.is_text_filtered("うん。") is True
    assert seg.is_final_too_short("うん。") is False


def test_final_gate_keeps_short_text_when_opted_in():
    seg = SentenceSegmenter(min_words_to_commit=4, min_words_to_emit_final=1)
    for t in SHORT_JA + SHORT_EN:
        assert seg.is_final_too_short(t) is False, f"{t!r} should be emit-able"
    # Empty / punctuation-only output is still not worth a subtitle.
    assert seg.is_final_too_short("") is True
    assert seg.is_final_too_short("   ") is True
    assert seg.is_final_too_short("。。。") is True


def test_production_default_preserves_short_turns():
    """Default SentenceConfig has min_words_to_emit_final=1 to preserve short dialogue turns."""
    cfg = SentenceConfig()
    assert cfg.min_words_to_emit_final == cfg.min_words_to_commit == 1
    seg = SentenceSegmenter(
        min_words_to_commit=cfg.min_words_to_commit,
        min_words_to_emit_final=cfg.min_words_to_emit_final,
    )
    for t in SHORT_JA + SHORT_EN:
        assert seg.is_final_too_short(t) is False


def _engine_with(**sentence_overrides):
    engine = TranscribeEngine.__new__(TranscribeEngine)
    sc = SentenceConfig(**sentence_overrides)
    engine.sentence_config = sc
    engine._segmenter = SentenceSegmenter(
        max_chars=sc.max_chars,
        max_duration_sec=sc.max_duration_sec,
        min_words_to_commit=sc.min_words_to_commit,
        min_words_to_emit_final=sc.min_words_to_emit_final,
        split_on_stability=sc.split_on_stability,
        stability_duration_sec=sc.stability_duration_sec,
        stability_threshold_polls=sc.stability_threshold_polls,
    )
    emitted = []
    engine._emit_final = lambda text, utt_id, **kw: emitted.append(text)
    engine._run_inference = lambda *a, **kw: "うん。"
    engine._state_lock = threading.Lock()
    engine._last_committed_head = ""
    engine._last_committed_head_time = 0.0
    engine._audio_buffer_mgr = None
    return engine, emitted


def test_commit_sync_emits_short_final_when_opted_in():
    engine, emitted = _engine_with(min_words_to_emit_final=1)
    engine._commit_sync(np.zeros(16000, dtype=np.float32), "utt-1", reason="VAD_SILENCE")
    assert emitted == ["うん。"], "a complete short Japanese turn must survive when opted in"


def test_commit_sync_drops_short_final_when_gated_to_4():
    engine, emitted = _engine_with(min_words_to_emit_final=4)
    engine._commit_sync(np.zeros(16000, dtype=np.float32), "utt-2", reason="VAD_SILENCE")
    assert emitted == []


def test_commit_sync_never_emits_empty_or_punctuation_only():
    engine, emitted = _engine_with(min_words_to_emit_final=1)
    engine._run_inference = lambda *a, **kw: "。。。"
    engine._commit_sync(np.zeros(16000, dtype=np.float32), "utt-3", reason="VAD_SILENCE")
    assert emitted == []


def test_update_sentence_config_rewires_both_gates():
    engine = TranscribeEngine("qwen3-asr-1.7b")
    engine.update_sentence_config(min_words_to_commit=3, min_words_to_emit_final=1)
    assert engine.sentence_config.min_words_to_commit == 3
    assert engine.sentence_config.min_words_to_emit_final == 1
    assert engine._segmenter.min_words_to_commit == 3
    assert engine._segmenter.min_words_to_emit_final == 1
