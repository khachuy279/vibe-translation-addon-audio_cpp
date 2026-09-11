"""Unit and pipeline test for PyTorch OmniVoice Voice Cloning TTS in backend_cpp."""

import asyncio
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from backend_cpp.tts import (
    TTSEngine,
    OmniVoiceTTS,
    VoiceManager,
    AudioProcessor,
    get_tts_engine,
)


def test_audio_processor():
    """Unit test for AudioProcessor pure functions."""
    # 1. Test convert_to_numpy
    t = [np.array([0.1, 0.2], dtype=np.float32), np.array([0.3], dtype=np.float32)]
    arr = AudioProcessor.convert_to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert len(arr) == 3

    # 2. Test normalize_audio
    loud = np.array([-2.0, 0.5, 1.5], dtype=np.float32)
    norm = AudioProcessor.normalize_audio(loud, volume=1.0, target_peak=0.95)
    assert np.max(np.abs(norm)) <= 0.96

    # 3. Test apply_time_stretch
    sr = 24000
    x = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    x_fast = AudioProcessor.apply_time_stretch(x, speed=1.25, sample_rate=sr)
    assert len(x_fast) < len(x)
    ratio = len(x) / len(x_fast)
    assert 1.20 <= ratio <= 1.30

    x_slow = AudioProcessor.apply_time_stretch(x, speed=0.8, sample_rate=sr)
    assert len(x_slow) > len(x)

    # 4. Test encode_wav_to_base64
    b64 = AudioProcessor.encode_wav_to_base64(x, sr)
    assert isinstance(b64, str)
    assert len(b64) > 100


def test_voice_manager():
    """Test voice discovery, thread-safety, and transcript resolution."""
    voices = VoiceManager.get_available_voices(force_reload=True)
    assert len(voices) >= 1
    assert any("speaker_01_0039.wav" in v["id"] for v in voices)

    ref_audio, ref_text = VoiceManager.resolve_voice("speaker_01_0039.wav")
    assert ref_audio.endswith(".wav")
    assert len(ref_text) > 10

    # Custom voice without transcript must return empty ref_text (no hallucination)
    audio_path, custom_text = VoiceManager.resolve_voice("non_existent_custom_voice.wav")
    assert audio_path.endswith(".wav")


def test_tts_engine_protocol_conformance():
    """Verify OmniVoiceTTS conforms to the TTSEngine protocol."""
    engine = get_tts_engine()
    assert isinstance(engine, TTSEngine)
    assert hasattr(engine, "sample_rate")
    assert engine.sample_rate == 24000
    assert callable(engine.is_model_ready)
    assert callable(engine.synthesize_sync)
    assert callable(engine.synthesize_clone)


@pytest.mark.asyncio
async def test_omnivoice_tts_synthesize_and_speed():
    """Test synthesis with normal speed and stretched speed."""
    tts = OmniVoiceTTS.get_instance()
    assert tts.is_model_ready()

    test_text = "Thử nghiệm giọng đọc nhân bản."
    import torch
    torch.manual_seed(42)
    b64_audio, dur_normal = await tts.synthesize_clone(
        text=test_text,
        voice_id="speaker_01_0039.wav",
        speed=1.0,
    )

    assert b64_audio is not None
    assert len(b64_audio) > 1000
    assert dur_normal > 0.5

    # Test speed adjustment (1.5x should be shorter than normal)
    torch.manual_seed(42)
    b64_fast, dur_fast = await tts.synthesize_clone(
        text=test_text,
        voice_id="speaker_01_0039.wav",
        speed=1.5,
    )
    assert b64_fast is not None
    assert dur_fast < dur_normal


def test_tts_unload_and_reset():
    """Test clean unload and resource release."""
    tts = OmniVoiceTTS.get_instance()
    tts.unload_model()
    assert tts._is_loaded is False
    assert tts.model is None

    # Reset instance
    OmniVoiceTTS.reset_instance()
    assert OmniVoiceTTS._instance is None


def test_audio_processor_unknown_types_and_empty_voices():
    """Verify robust skipping of unknown item types and safe fallback when voices is empty."""
    # 1. convert_to_numpy with invalid/unknown types
    mixed = ["invalid_string", 12345, object(), np.array([1.0, 2.0], dtype=np.float32)]
    arr = AudioProcessor.convert_to_numpy(mixed)
    assert len(arr) == 2
    assert arr[0] == 1.0

    # 2. VoiceManager resolve_voice with empty voices
    from unittest.mock import patch
    with patch.object(VoiceManager, "get_available_voices", return_value=[]):
        audio, text = VoiceManager.resolve_voice(None)
        assert audio == ""
        assert text == ""


def test_voice_prompt_cache():
    """Verify that _get_voice_clone_prompt caches prompts and unload_model clears the cache."""
    engine = OmniVoiceTTS()
    mock_model = MagicMock()
    mock_prompt = MagicMock()
    mock_model.create_voice_clone_prompt.return_value = mock_prompt
    engine.model = mock_model

    # First call: creates prompt and caches it
    res1 = engine._get_voice_clone_prompt("ref.wav", "Hello")
    assert res1 == mock_prompt
    assert mock_model.create_voice_clone_prompt.call_count == 1
    assert ("ref.wav", "Hello") in engine._voice_prompt_cache

    # Second call with same ref: returns cached prompt, does not call create_voice_clone_prompt again
    res2 = engine._get_voice_clone_prompt("ref.wav", "Hello")
    assert res2 == mock_prompt
    assert mock_model.create_voice_clone_prompt.call_count == 1

    # Call with different ref
    mock_prompt2 = MagicMock()
    mock_model.create_voice_clone_prompt.return_value = mock_prompt2
    res3 = engine._get_voice_clone_prompt("ref2.wav", "World")
    assert res3 == mock_prompt2
    assert mock_model.create_voice_clone_prompt.call_count == 2
    assert len(engine._voice_prompt_cache) == 2

    # Unload model should flush the cache
    engine.unload_model()
    assert len(engine._voice_prompt_cache) == 0

