"""Package Text-to-Speech (TTS) cho Backend - Native PyTorch OmniVoice Voice Cloning."""

from backend.tts.base import BaseTTSEngine
from backend.tts.audio_processor import AudioProcessor
from backend.tts.voice_manager import VoiceManager
from backend.tts.dedup import TTSDedupState
from backend.tts.engine import OmniVoiceTTS


def get_tts_engine() -> BaseTTSEngine:
    """Trả về Singleton instance của OmniVoice TTS Engine."""
    return OmniVoiceTTS.get_instance()


__all__ = [
    "BaseTTSEngine",
    "AudioProcessor",
    "VoiceManager",
    "TTSDedupState",
    "OmniVoiceTTS",
    "get_tts_engine",
]
