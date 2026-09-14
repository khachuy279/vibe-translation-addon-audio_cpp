"""Unit tests for per-VAD engine optimal profiles and dynamic switching."""

import pytest
from unittest.mock import MagicMock
from backend_cpp.config import config, DEFAULT_ENGINE_PROFILES
from backend_cpp.vad.vad_processor import VADProcessor
from backend_cpp.ws.session_state import SessionState


def test_default_engine_profiles_configured():
    """Verify that all three production VAD engines have tuned sweet spots."""
    assert "fsmn-vad" in config.vad.engine_profiles
    assert "firered-vad" in config.vad.engine_profiles
    assert "silero-vad" in config.vad.engine_profiles

    fsmn_prof = config.vad.get_engine_profile("fsmn-vad")
    assert fsmn_prof.threshold == 0.45
    assert fsmn_prof.silence_duration_ms == 500
    assert fsmn_prof.hangover_ms == 300
    assert fsmn_prof.pre_speech_buffer_ms == 120

    firered_prof = config.vad.get_engine_profile("firered-vad")
    assert firered_prof.threshold == 0.45
    assert firered_prof.silence_duration_ms == 500
    assert firered_prof.hangover_ms == 300
    assert firered_prof.pre_speech_buffer_ms == 100

    silero_prof = config.vad.get_engine_profile("silero-vad")
    assert silero_prof.threshold == 0.50
    assert silero_prof.silence_duration_ms == 500
    assert silero_prof.hangover_ms == 288
    assert silero_prof.pre_speech_buffer_ms == 96


def test_apply_engine_profile_updates_vad_config():
    """Verify apply_engine_profile mutates active VADConfig cleanly."""
    original_engine = config.vad.vad_engine
    try:
        prof = config.vad.apply_engine_profile("firered-vad")
        assert config.vad.vad_engine == "firered-vad"
        assert config.vad.threshold == 0.45
        assert config.vad.silence_duration_ms == 500
        assert config.vad.hangover_ms == 300
        assert config.vad.pre_speech_buffer_ms == 100

        # Switch back to fsmn-vad
        prof2 = config.vad.apply_engine_profile("fsmn-vad")
        assert config.vad.vad_engine == "fsmn-vad"
        assert config.vad.threshold == 0.45
        assert config.vad.silence_duration_ms == 500
        assert config.vad.hangover_ms == 300
        assert config.vad.pre_speech_buffer_ms == 120
    finally:
        config.vad.apply_engine_profile(original_engine)


def test_session_state_auto_adapts_vad_profile_on_switch():
    """Verify that when SessionState.apply_config receives a vad_engine change, it adopts the target profile."""
    session = SessionState(MagicMock())
    session.init_components()

    # Session starts with default fsmn-vad
    assert session.config["vad_engine"] == "fsmn-vad"
    assert session.config["vad_threshold"] == 0.45
    assert session.config["silence_duration_ms"] == 500

    # User switches to firered-vad in popup without specifying custom sliders
    session.apply_config({"vad_engine": "firered-vad"})

    assert session.config["vad_engine"] == "firered-vad"
    assert session.config["vad_threshold"] == 0.45
    assert session.config["silence_duration_ms"] == 500
    assert session.config["hangover_ms"] == 300
    assert session.config["pre_speech_buffer_ms"] == 100
    assert session.vad_processor.threshold == 0.45
    assert session.vad_processor.pre_speech_buffer_ms == 100

    # User switches to silero-vad
    session.apply_config({"vad_engine": "silero-vad"})

    assert session.config["vad_engine"] == "silero-vad"
    assert session.config["vad_threshold"] == 0.50
    assert session.config["silence_duration_ms"] == 500
    assert session.config["hangover_ms"] == 288
    assert session.config["pre_speech_buffer_ms"] == 96
    assert session.vad_processor.threshold == 0.50
    assert session.vad_processor.hangover_ms == 288
    assert session.vad_processor.pre_speech_buffer_ms == 96


def test_session_state_preserves_explicit_user_slider_overrides():
    """If the user explicitly sends custom threshold and silence with the switch, preserve them."""
    session = SessionState(MagicMock())
    session.init_components()

    session.apply_config({
        "vad_engine": "silero-vad",
        "vad_threshold": 0.65,
        "silence_duration_ms": 750,
    })

    assert session.config["vad_engine"] == "silero-vad"
    assert session.config["vad_threshold"] == 0.65
    assert session.config["silence_duration_ms"] == 750
    assert session.config["hangover_ms"] == 288
    assert session.vad_processor.threshold == 0.65
    assert session.vad_processor.silence_duration_ms == 750
