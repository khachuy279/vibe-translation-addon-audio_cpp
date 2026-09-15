"""Package core chứa các thành phần cốt lõi của pipeline Backend."""

from backend.core.audio_buffer import CircularAudioBuffer
from backend.core.normalizer import SpeechNormalizer, NormalizationResult
from backend.core.metrics import metrics, MetricsCollector
from backend.core.dedup import CommitDeduplicator, normalize_for_dedup
from backend.core.commit_manager import CommitManager, count_content_tokens, is_cjk
from backend.core.pipeline_events import (
    AudioChunk,
    SpeechSegment,
    ASRTranscript,
    TranslationItem,
    TTSItem,
    VADState,
    CommitReason,
)

__all__ = [
    "CircularAudioBuffer",
    "SpeechNormalizer",
    "NormalizationResult",
    "metrics",
    "MetricsCollector",
    "CommitDeduplicator",
    "normalize_for_dedup",
    "CommitManager",
    "count_content_tokens",
    "is_cjk",
    "AudioChunk",
    "SpeechSegment",
    "ASRTranscript",
    "TranslationItem",
    "TTSItem",
    "VADState",
    "CommitReason",
]
