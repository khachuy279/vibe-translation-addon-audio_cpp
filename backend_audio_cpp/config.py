"""Global configuration dataclasses for backend_audio_cpp."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend_audio_cpp"
_MODELS_DIR = _BACKEND_DIR / "models"
_VOICES_DIR = _BACKEND_DIR / "voices"

SUPPORTED_LANGUAGES = [
    {"code": "auto", "name": "Auto-detect (Tự động)"},
    {"code": "en", "name": "English (Tiếng Anh)"},
    {"code": "zh", "name": "Chinese (Tiếng Trung)"},
    {"code": "ja", "name": "Japanese (Tiếng Nhật)"},
    {"code": "ko", "name": "Korean (Tiếng Hàn)"},
    {"code": "ru", "name": "Russian (Tiếng Nga)"},
    {"code": "fr", "name": "French (Tiếng Pháp)"},
    {"code": "de", "name": "German (Tiếng Đức)"},
    {"code": "es", "name": "Spanish (Tiếng Tây Ban Nha)"},
    {"code": "vi", "name": "Vietnamese (Tiếng Việt)"},
]

SUPPORTED_VAD_ENGINES = ["silero-vad"]


@dataclass
class WSConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    ping_interval: float = 20.0
    ping_timeout: float = 20.0
    ssl_enabled: bool = True


@dataclass
class VADConfig:
    vad_engine: str = "silero-vad"
    threshold: float = 0.50
    silence_duration_ms: int = 450
    hangover_ms: int = 200
    pre_speech_buffer_ms: int = 300
    min_speech_duration_sec: float = 0.20
    max_speech_duration_sec: float = 8.00
    sample_rate: int = 16000
    enabled: bool = True


@dataclass
class ASRConfig:
    default_model: str = "qwen3-asr-1.7b"
    server_url: str = "http://127.0.0.1:8089"
    sample_rate: int = 16000
    language: str = "auto"
    enable_normalization: bool = True
    poll_interval_sec: float = 0.30


@dataclass
class SentenceConfig:
    min_words_to_commit: int = 2
    max_chars: int = 150
    max_duration_sec: float = 8.0
    split_on_stability: bool = True
    stability_duration_sec: float = 0.8
    stability_threshold_polls: int = 2


@dataclass
class TranslationConfig:
    base: str = "tencent"
    model: str = "unsloth/Hy-MT2-7B-GGUF"
    gguf_file: str = "Hy-MT2-7B-UD-Q4_K_XL.gguf"
    source_lang: str = "auto"
    target_lang: str = "vi"
    n_gpu_layers: int = -1
    n_ctx: int = 2048
    use_context: bool = True
    context_window: int = 3


@dataclass
class TTSConfig:
    enabled: bool = False
    engine: str = "omnivoice"
    model: str = "omnivoice-q8_0.gguf"
    server_url: str = "http://127.0.0.1:8089"
    default_voice: str = "speaker_01_0039.wav"
    speed: float = 1.0
    sample_rate: int = 24000


@dataclass
class AppConfig:
    ws: WSConfig = field(default_factory=WSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    sentence: SentenceConfig = field(default_factory=SentenceConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


# Global singleton configuration instance
config = AppConfig()
