"""TTS Package for backend_cpp – Native PyTorch OmniVoice Voice Cloning (~0.5s)."""

from backend_cpp.tts.base import TTSEngine
from backend_cpp.tts.audio_processor import AudioProcessor
from backend_cpp.tts.omnivoice_engine import OmniVoiceTTS
from backend_cpp.tts.voice_manager import VoiceManager


def get_tts_engine() -> TTSEngine:
    """Return the singleton PyTorch OmniVoice TTS engine instance implementing TTSEngine."""
    return OmniVoiceTTS.get_instance()


__all__ = ["TTSEngine", "AudioProcessor", "OmniVoiceTTS", "VoiceManager", "get_tts_engine"]
