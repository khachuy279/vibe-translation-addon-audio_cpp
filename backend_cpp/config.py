"""Configuration management for backend_cpp.

All configuration defaults are defined directly in Python classes below.
No .env file is used.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator

# Root directory of backend_cpp
BACKEND_CPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_CPP_DIR.parent
MODELS_DIR = BACKEND_CPP_DIR / "models"
MODELS_YAML_PATH = BACKEND_CPP_DIR / "models.yaml"
TRANSLATION_MODELS_YAML_PATH = BACKEND_CPP_DIR / "translation_models.yaml"

# Fallback to backend/models if exists
LEGACY_MODELS_DIR = PROJECT_ROOT / "backend" / "models"

SUPPORTED_LANGUAGES = [
    {"code": "auto", "name": "Auto-detect"},
    {"code": "en", "name": "English (en)"},
    {"code": "ja", "name": "Japanese (ja)"},
    {"code": "zh", "name": "Chinese (zh)"},
    {"code": "ko", "name": "Korean (ko)"},
    {"code": "ru", "name": "Russian (ru)"},
]


class WSConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765
    ping_interval: float = 20.0
    ping_timeout: float = 30.0
    # Product constraint: one active translation session at a time (audit finding P1-04).
    # The Firefox extension elects a single capture owner; this is defense-in-depth
    # against duplicate sessions (stale popup broadcast, second tab, etc.).
    #
    # Policy is NEWEST-WINS: the oldest session is closed to admit a newcomer. Rejecting
    # the newcomer instead would let one stale socket lock the user out of START forever.
    # Set to 0 to disable the limit entirely.
    max_sessions: int = 1


class FireRedVADConfig(BaseModel):
    threshold: Optional[float] = None
    smooth_window_size: int = 5
    min_speech_frame: int = 8       # 8 frames (80ms) min speech to confirm onset
    min_silence_frame: int = 20     # 20 frames (200ms) min silence to confirm offset
    pad_start_frame: int = 5        # 5 frames (50ms) onset pre-padding


class SileroVADConfig(BaseModel):
    threshold: Optional[float] = None
    neg_threshold_offset: float = 0.15  # Negative threshold = threshold - offset (0.35)
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30


class FsmnVADConfig(BaseModel):
    speech_noise_thres: Optional[float] = None
    max_end_silence_time: int = 800
    speech_to_sil_time_thres: int = 150
    sil_to_speech_time_thres: int = 150

    @property
    def threshold(self) -> Optional[float]:
        return self.speech_noise_thres

    @threshold.setter
    def threshold(self, val: Optional[float]) -> None:
        self.speech_noise_thres = val


class VADConfig(BaseModel):
    enabled: bool = True
    # Default engine.
    #
    # Was "firered-vad", which measured 108.7 ms per audio-second -- the MOST expensive of
    # the three engines, and VAD is the largest CPU stage in the pipeline.
    #
    # Measured on 24s of real speech (Harvard sentences), all three engines produced
    # IDENTICAL final transcripts and the same 8 segments / 7 utterances:
    #     firered-vad  108.7 ms/audio-s   (x8.2 silero)
    #     fsmn-vad      57.9 ms/audio-s   (x4.4 silero)   <- default now
    #     silero-vad    13.3 ms/audio-s   (x1.0)
    # fsmn-vad was chosen because it halves this stage with zero measured transcript change
    # and identical segmentation. silero-vad is ~4x cheaper still but trims more audio
    # (67.1% vs 79.6% speech ratio), which risks clipping word onsets on quieter or
    # far-field audio, so it stays opt-in.
    #
    # Re-measure before changing this again: python scratch/compare_vad_quality.py
    vad_engine: str = "fsmn-vad"  # fsmn-vad (default), firered-vad, silero-vad
    threshold: float = 0.4
    # Trailing-silence limit that gates a commit. THE single largest knob on subtitle latency.
    #
    # Full measured budget (real speech, `scratch/analyze_latency_budget.py`), from the moment
    # the speaker stops to the subtitle being delivered:
    #     silence wait        480 ms  (44%)   <- this setting
    #     VAD END -> COMMIT   165 ms  (15%)
    #     COMMIT -> subtitle  447 ms  (41%)   <- translation
    #     total              ~1092 ms
    #
    # The safety net commits after exactly this much silence, quantized to the engine frame
    # (60 ms for fsmn): 450 -> 480 ms, 250 -> 300 ms, 150 -> 180 ms, 100 -> 120 ms.
    #
    # Measured end-to-end at each setting (`scratch/compare_vad_silence.py`): 450/250/150/100 all
    # produced BIT-IDENTICAL transcripts on clean speech. The cost of lowering it is subtitle
    # fragmentation, not lost words: on conversational audio the utterance count went 5 -> 8
    # (silence 250) / 9 (150), because shorter natural pauses start splitting sentences.
    #
    # DECISION (2026-09-11, user): 150 ms. Balanced -- saves ~300 ms of the ~1080 ms budget in
    # exchange for splitting at pauses. Do not lower this further without re-measuring
    # fragmentation; clients can still override per session with `silenceDurationMs`.
    silence_duration_ms: int = 150
    # NOTE: this is a GRACE PERIOD, not added latency. See VADProcessor.feed_chunk(): silent
    # frames are attached to the committed audio for
    # `min(hangover_ms, silence_duration_ms * 0.5)` and then the utterance ends. Raising it
    # therefore does NOT delay `[VAD END]`; only `silence_duration_ms` does.
    hangover_ms: int = 400
    pre_speech_buffer_ms: int = 550
    sample_rate: int = 16000

    firered: FireRedVADConfig = Field(default_factory=FireRedVADConfig)
    silero: SileroVADConfig = Field(default_factory=SileroVADConfig)
    fsmn: FsmnVADConfig = Field(default_factory=FsmnVADConfig)

    @model_validator(mode="after")
    def _sync_thresholds(self) -> "VADConfig":
        """Share unified threshold across sub-configs if not explicitly overridden."""
        if self.firered.threshold is None:
            self.firered.threshold = self.threshold
        if self.silero.threshold is None:
            self.silero.threshold = self.threshold
        if self.fsmn.speech_noise_thres is None:
            self.fsmn.speech_noise_thres = self.threshold
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "threshold":
            if hasattr(self, "firered") and self.firered is not None:
                self.firered.threshold = value
            if hasattr(self, "silero") and self.silero is not None:
                self.silero.threshold = value
            if hasattr(self, "fsmn") and self.fsmn is not None:
                self.fsmn.speech_noise_thres = value


class ASRConfig(BaseModel):
    active_model: str = "qwen3-asr-1.7b"
    backend: str = "vulkan"  # auto, vulkan, cuda, cpu
    language: str = "auto"
    threads: int = 4
    min_transcribe_sec: float = 0.6
    poll_interval_ms: int = 350
    # Preview growth gate.
    #
    # A preview re-transcribes the ENTIRE utterance so far, while the poller wakes every
    # `poll_interval_ms`. That makes the preview path process far more audio than was
    # actually spoken: measured on 24s of real speech, `asr.preview_audio_ms` = 52.6s
    # (+20.6s of commit audio) = 3.05x amplification, and 30 of 38 inferences (79%).
    #
    # This gate skips a preview until at least this much NEW audio has accumulated since the
    # previous preview:
    #   required_new = max(preview_min_growth_ms / 1000, preview_min_growth_ratio * duration)
    # A ratio >= 0.5 amortises the repeated work (preview cost grows logarithmically instead
    # of linearly in the number of polls). The COMMIT path is untouched, so final transcripts
    # are bit-identical; only live preview cadence changes.
    #
    # 0 / 0.0 disables the gate (previous behaviour).
    #
    # DECISION (2026-09-10): the gate is implemented and measured but deliberately left OFF
    # by default. It cuts 52% of the audio the ASR processes with bit-identical transcripts,
    # but it also reduces how often the live preview refreshes, and that is user-visible.
    # Do not flip this default without asking; see plan section 0.5E.
    preview_min_growth_ratio: float = 0.0
    preview_min_growth_ms: int = 0
    models_yaml: str = str(MODELS_YAML_PATH)
    # Bounded ASR token queue. Final (committed) messages are never dropped; preview
    # messages are coalesced (latest-wins) once the queue reaches this size.
    token_queue_maxsize: int = 64
    # Lifecycle barrier timeouts. A model unload/swap waits for in-flight native
    # inference to finish before closing the model; these bound that wait.
    unload_lock_timeout_sec: float = 15.0
    model_load_lock_timeout_sec: float = 30.0
    normalize_speech: bool = True
    normalize_target_rms: float = 0.10  # Default target RMS for speech normalization (~ -20 dBFS); should be benchmarked against target ASR models
    normalize_target_peak: float = 0.95
    normalize_max_gain: float = 3.0
    normalize_min_gain: float = 0.3333  # 1.0 / normalize_max_gain
    normalize_knee_start: float = 0.025
    normalize_knee_end: float = 0.050
    normalize_gain_smoothing: bool = True
    normalize_attack_alpha: float = 0.15   # Slow boost to prevent pulling up noise floor
    normalize_release_alpha: float = 0.35  # Fast attenuation to protect headroom on loud speech
    normalize_speech_frame_ms: int = 25
    normalize_min_speech_frames: int = 2
    normalize_min_rms_to_boost: float = 0.04  # Legacy compatibility threshold
    normalize_log_stats: bool = True  # Log RMS/peak/gain stats on commit



class TranslationConfig(BaseModel):
    base: str = "tencent"  # tencent, xiaomi, gemmax
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

    @model_validator(mode="after")
    def _apply_base_defaults(self) -> "TranslationConfig":
        from backend_cpp.translation.model_registry import TranslationModelRegistry

        registry = TranslationModelRegistry.get_instance()
        resolved_key = registry.resolve_key(self.base)
        d = registry.get_model(resolved_key)
        if d is None:
            resolved_key = registry.default_model_key
            d = registry.get_model(resolved_key) or {}

        self.base = resolved_key
        if self.model is None and "model" in d:
            self.model = d["model"]
        if self.gguf_file is None and "gguf_file" in d:
            self.gguf_file = d["gguf_file"]
        if self.temperature is None and "temperature" in d:
            self.temperature = d["temperature"]
        if self.top_p is None and "top_p" in d:
            self.top_p = d["top_p"]
        if self.top_k is None and "top_k" in d:
            self.top_k = d["top_k"]
        if self.repetition_penalty is None and "repetition_penalty" in d:
            self.repetition_penalty = d["repetition_penalty"]
        if self.prompt_style is None and "prompt_style" in d:
            self.prompt_style = d.get("prompt_style", "tencent")
        return self


class SentenceConfig(BaseModel):
    max_chars: int = 150
    max_duration_sec: float = 8.0
    min_words_to_commit: int = 2
    split_on_stability: bool = True         # Ngắt câu khi preview text ổn định qua nhiều chu kỳ
    stability_duration_sec: float = 0.8     # Thời gian (giây) preview text giữ nguyên để chốt câu
    stability_threshold_polls: int = 2      # Số lần poll tối thiểu giữ nguyên kết quả


class TTSConfig(BaseModel):
    enabled: bool = True
    engine: str = "omnivoice"  # PyTorch native OmniVoice (~0.5s)
    model: str = "splendor1811/omnivoice-vietnamese"
    device: str = "cuda:0"
    speed: float = 1.0
    num_inference_steps: int = 8
    default_voice: str = "speaker_01_0039.wav"
    voices_dir: str = str(BACKEND_CPP_DIR / "voices")
    volume: float = 0.8
    sample_rate: int = 24000


class PerfConfig(BaseModel):
    enabled: bool = False
    alert_threshold_ms: float = 300.0
    dump_report_on_disconnect: bool = True
    report_file: str = "perf_report.json"


class DebugConfig(BaseModel):
    dump_audio: bool = False
    dump_dir: str = str(PROJECT_ROOT / "debug_audio")
    bypass_speech_normalization: bool = False
    log_audio_levels: bool = False  # Enable detailed level logging on every preview
    # Raise instead of silently racing when a lifecycle invariant is violated
    # (e.g. ensure_session() called without holding the ASR inference lock).
    strict_lock_checks: bool = False


class AppConfig(BaseModel):
    ws: WSConfig = Field(default_factory=WSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    sentence: SentenceConfig = Field(default_factory=SentenceConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    perf: PerfConfig = Field(default_factory=PerfConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)



def load_config() -> AppConfig:
    """Load configuration with pure Python defaults."""
    cfg = AppConfig()
    return cfg


config = load_config()
