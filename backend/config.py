"""Module quản lý cấu hình tập trung cho Backend.

Hỗ trợ:
- Định nghĩa kiểu dữ liệu chặt chẽ qua Pydantic v2.
- Nạp mặc định không phụ thuộc .env.
- Cơ chế Hot-Reload động (cập nhật cấu hình runtime không cần khởi động lại server).
- Đầy đủ chú thích tiếng Việt cho từng trường cấu hình.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

# Đường dẫn thư mục gốc
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
MODELS_YAML_PATH = BACKEND_DIR / "models.yaml"
TRANSLATION_MODELS_YAML_PATH = BACKEND_DIR / "translation_models.yaml"
VOICES_DIR = BACKEND_DIR / "voices"

# Danh sách ngôn ngữ ASR được hỗ trợ
SUPPORTED_LANGUAGES = [
    {"code": "auto", "name": "Auto-detect"},
    {"code": "en", "name": "English (en)"},
    {"code": "ja", "name": "Japanese (ja)"},
    {"code": "zh", "name": "Chinese (zh)"},
    {"code": "ko", "name": "Korean (ko)"},
    {"code": "ru", "name": "Russian (ru)"},
]


class WSConfig(BaseModel):
    """Cấu hình WebSocket và Server WSS."""
    host: str = "0.0.0.0"
    port: int = 8765
    ping_interval: float = 20.0
    ping_timeout: float = 30.0
    max_payload_bytes: int = 10 * 1024 * 1024  # 10MB


class FireRedVADConfig(BaseModel):
    """Cấu hình chuyên biệt cho FireRed-VAD (Xiaohongshu DFSMN)."""
    threshold: Optional[float] = None
    smooth_window_size: int = 5
    min_speech_frame: int = 8       # 8 frames (80ms) tối thiểu xác nhận bắt đầu nói
    min_silence_frame: int = 20     # 20 frames (200ms) tối thiểu xác nhận kết thúc nói
    pad_start_frame: int = 5        # 5 frames (50ms) pre-padding


class SileroVADConfig(BaseModel):
    """Cấu hình chuyên biệt cho Silero VAD."""
    threshold: Optional[float] = None
    neg_threshold_offset: float = 0.15  # Negative threshold = threshold - offset
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30


class FsmnVADConfig(BaseModel):
    """Cấu hình chuyên biệt cho FSMN-VAD (Alibaba FunASR)."""
    speech_noise_thres: Optional[float] = None
    max_end_silence_time: int = 800
    speech_to_sil_time_thres: int = 200
    sil_to_speech_time_thres: int = 100

    @property
    def threshold(self) -> Optional[float]:
        return self.speech_noise_thres

    @threshold.setter
    def threshold(self, val: Optional[float]) -> None:
        self.speech_noise_thres = val


class VADConfig(BaseModel):
    """Cấu hình tổng hợp cho Voice Activity Detection."""
    enabled: bool = True
    vad_engine: str = "firered-vad"  # firered-vad, silero-vad, fsmn-vad
    threshold: float = 0.45
    silence_duration_ms: int = 600   # Thời gian im lặng (ms) để kích hoạt ngắt câu
    hangover_ms: int = 400           # Giữ trạng thái nói thêm hangover_ms phòng ngắt quãng
    pre_speech_buffer_ms: int = 300  # Đệm âm thanh trước khi bắt đầu nói để tránh mất phụ âm đầu
    sample_rate: int = 16000

    firered: FireRedVADConfig = Field(default_factory=FireRedVADConfig)
    silero: SileroVADConfig = Field(default_factory=SileroVADConfig)
    fsmn: FsmnVADConfig = Field(default_factory=FsmnVADConfig)

    @model_validator(mode="after")
    def _sync_thresholds(self) -> "VADConfig":
        """Đồng bộ ngưỡng mặc định cho các sub-engine nếu chưa cấu hình riêng."""
        if self.firered.threshold is None:
            self.firered.threshold = self.threshold
        if self.silero.threshold is None:
            self.silero.threshold = self.threshold
        if self.fsmn.speech_noise_thres is None:
            self.fsmn.speech_noise_thres = self.threshold
        return self


class ASRConfig(BaseModel):
    """Cấu hình nhận dạng giọng nói ASR qua transcribe.cpp."""
    active_model: str = "qwen3-asr-1.7b"
    backend: str = "auto"  # auto, vulkan, cuda, cpu
    language: str = "auto"
    threads: int = 4
    min_transcribe_sec: float = 0.6
    poll_interval_ms: int = 350
    models_yaml: str = str(MODELS_YAML_PATH)
    
    # Cấu hình Chuẩn hóa Âm Lượng (Speech Normalization)
    normalize_speech: bool = True
    normalize_target_rms: float = 0.10   # Mục tiêu RMS (~ -20 dBFS)
    normalize_target_peak: float = 0.95  # Trần biên độ cực đại tránh méo tiếng
    normalize_max_gain: float = 3.0      # Hệ số khuếch đại tối đa (không kéo nhiễu nền)
    normalize_min_gain: float = 0.3333   # Hệ số nén tối đa (1.0 / max_gain)
    normalize_knee_start: float = 0.025
    normalize_knee_end: float = 0.050
    normalize_gain_smoothing: bool = True
    normalize_attack_alpha: float = 0.15
    normalize_release_alpha: float = 0.35
    normalize_speech_frame_ms: int = 25
    normalize_min_speech_frames: int = 2
    normalize_log_stats: bool = True


class SentenceConfig(BaseModel):
    """Cấu hình ngắt câu và phân đoạn ngữ nghĩa."""
    max_chars: int = 150
    max_duration_sec: float = 8.0          # Giới hạn tối đa độ dài 1 câu nói liên tục
    min_words_to_commit: int = 2           # Số từ tối thiểu để gửi sang dịch/TTS (lọc tiếng ậm ừ)
    split_on_stability: bool = True        # Tự động ngắt câu khi preview text ổn định
    stability_duration_sec: float = 0.8    # Thời gian (giây) preview text bất biến
    stability_threshold_polls: int = 3     # Số chu kỳ poll tối thiểu xác nhận ổn định
    inactivity_timeout_sec: float = 1.2    # Timeout ép chốt câu nếu không có frame mới


class TranslationConfig(BaseModel):
    """Cấu hình dịch thuật cục bộ GGUF qua Llama.cpp."""
    base: str = "tencent"  # tencent, tencent-1.8b, xiaomi, gemmax
    enabled: bool = True
    model: Optional[str] = None
    gguf_file: Optional[str] = None
    target_lang: str = "vi"
    source_lang: str = "auto"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_tokens: int = 128
    use_context: bool = False
    context_window: int = 3
    prompt_style: Optional[str] = None
    n_gpu_layers: int = -1
    models_yaml: str = str(TRANSLATION_MODELS_YAML_PATH)


class TTSConfig(BaseModel):
    """Cấu hình tổng hợp giọng nói Voice Cloning OmniVoice (C++ GGUF / PyTorch Native)."""
    enabled: bool = True
    engine: str = "omnivoice.cpp"  # omnivoice.cpp (GGUF C++ Native) hoặc omnivoice (PyTorch Native)
    model: str = "omnivoice-base-Q8_0.gguf"
    codec_model: str = "omnivoice-tokenizer-F32.gguf"
    device: str = "cuda:0"
    speed: float = 1.0
    num_inference_steps: int = 2
    default_voice: str = "speaker_01_0039.wav"
    voices_dir: str = str(VOICES_DIR)
    volume: float = 0.8
    sample_rate: int = 24000


class AudioBufferConfig(BaseModel):
    """Cấu hình Circular Ring Buffer cho Audio Ingress."""
    sample_rate: int = 16000
    capacity_sec: float = 60.0             # Dung lượng cố định 60 giây audio (~960,000 samples Float32)
    chunk_size_samples: int = 400          # Kích thước frame chuẩn cho VAD (25ms @ 16kHz)
    max_speech_segment_sec: float = 30.0   # Độ dài tối đa 1 đoạn phát âm


class MetricsConfig(BaseModel):
    """Cấu hình đo lường hiệu năng thời gian thực."""
    enabled: bool = True
    alert_threshold_ms: float = 300.0
    dump_report_on_disconnect: bool = True
    report_file: str = "metrics_report.json"


class AppConfig(BaseModel):
    """Cấu hình gốc toàn hệ thống Backend."""
    ws: WSConfig = Field(default_factory=WSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    sentence: SentenceConfig = Field(default_factory=SentenceConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio_buffer: AudioBufferConfig = Field(default_factory=AudioBufferConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    def hot_reload(self, updates: Dict[str, Any]) -> None:
        """Cập nhật cấu hình runtime nhanh chóng không cần khởi động lại server."""
        for key, value in updates.items():
            if hasattr(self, key) and isinstance(value, dict):
                sub_cfg = getattr(self, key)
                for sub_k, sub_v in value.items():
                    if hasattr(sub_cfg, sub_k):
                        setattr(sub_cfg, sub_k, sub_v)
            elif hasattr(self, key):
                setattr(self, key, value)


def load_config() -> AppConfig:
    """Khởi tạo cấu hình mặc định."""
    return AppConfig()


config = load_config()
