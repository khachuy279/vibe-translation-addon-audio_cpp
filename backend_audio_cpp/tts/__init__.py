"""TTS Module for backend_audio_cpp based on OmniVoice-GGUF via audio.cpp."""

from backend_audio_cpp.tts.omnivoice_engine import (
    OmniVoiceTTSEngine,
    TTSConfig,
    TTSSynthesisResult,
)
from backend_audio_cpp.tts.voice_manager import VoiceManager, VoiceProfile

__all__ = [
    "OmniVoiceTTSEngine",
    "TTSConfig",
    "TTSSynthesisResult",
    "VoiceManager",
    "VoiceProfile",
]
