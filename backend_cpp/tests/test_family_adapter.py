"""Unit tests for family_adapter in backend_cpp."""

import pytest
import transcribe_cpp

from backend_cpp.asr.family_adapter import (
    build_family_options,
    normalize_language_for_family,
)


def test_normalize_language_nemotron():
    # 2-letter codes mapped to BCP-47 locale
    assert normalize_language_for_family("en", "nemotron") == "en-US"
    assert normalize_language_for_family("en-us", "nemotron") == "en-US"
    assert normalize_language_for_family("zh", "nemotron") == "zh-CN"
    assert normalize_language_for_family("ja", "nemotron") == "ja-JP"
    assert normalize_language_for_family("ko", "nemotron") == "ko-KR"
    assert normalize_language_for_family("de", "nemotron") == "de-DE"
    assert normalize_language_for_family("fr", "nemotron") == "fr-FR"
    assert normalize_language_for_family("vi", "nemotron") == "vi-VN"

    # Auto / None
    assert normalize_language_for_family("auto", "nemotron") is None
    assert normalize_language_for_family("", "nemotron") is None
    assert normalize_language_for_family(None, "nemotron") is None


def test_normalize_language_whisper():
    # Locales stripped to ISO 639-1
    assert normalize_language_for_family("en-US", "whisper") == "en"
    assert normalize_language_for_family("vi-VN", "whisper") == "vi"
    assert normalize_language_for_family("zh-CN", "whisper") == "zh"
    assert normalize_language_for_family("ja", "whisper") == "ja"

    # Auto / None
    assert normalize_language_for_family("auto", "whisper") is None
    assert normalize_language_for_family(None, "whisper") is None


def test_normalize_language_sensevoice():
    # Supported: zh, en, ja, ko, yue
    assert normalize_language_for_family("zh", "sensevoice") == "zh"
    assert normalize_language_for_family("en-US", "sensevoice") == "en"
    assert normalize_language_for_family("ja", "sensevoice") == "ja"
    # Unsupported language falls back to None (auto)
    assert normalize_language_for_family("vi", "sensevoice") is None
    assert normalize_language_for_family("fr", "sensevoice") is None


def test_build_family_options_nemotron():
    opts = build_family_options("nemotron", {"att_context_right": 13})
    assert isinstance(opts, transcribe_cpp.ParakeetStreamOptions)
    assert opts.att_context_right == 13


def test_build_family_options_whisper():
    opts = build_family_options("whisper", {"condition_on_prev_tokens": False})
    assert isinstance(opts, transcribe_cpp.WhisperRunOptions)


def test_build_family_options_voxtral():
    opts = build_family_options("voxtral", {"num_delay_tokens": 2, "min_decode_interval_ms": 100})
    assert isinstance(opts, transcribe_cpp.VoxtralRealtimeStreamOptions)


def test_build_family_options_generic():
    # Models without special FamilyExtension (like Qwen3) return None
    assert build_family_options("qwen3_asr") is None
    assert build_family_options("sensevoice") is None
    assert build_family_options("unknown_family") is None


def test_build_family_options_slot_filtering():
    # Parakeet/Nemotron is a 'stream'-slot extension
    assert build_family_options("nemotron", {"att_context_right": 13}, slot="stream") is not None
    assert build_family_options("nemotron", {"att_context_right": 13}, slot="run") is None

    # Whisper is a 'run'-slot extension
    assert build_family_options("whisper", {"condition_on_prev_tokens": False}, slot="run") is not None
    assert build_family_options("whisper", {"condition_on_prev_tokens": False}, slot="stream") is None

    # Voxtral is a 'stream'-slot extension
    assert build_family_options("voxtral", {}, slot="stream") is not None
    assert build_family_options("voxtral", {}, slot="run") is None

