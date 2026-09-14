"""ASR subpackage for backend_cpp."""

from backend_cpp.asr.audio_buffer import AudioBufferManager
from backend_cpp.asr.base_engine import BaseASREngine
from backend_cpp.asr.dedup import CommitDeduplicator
from backend_cpp.asr.family_adapter import build_family_options, normalize_language_for_family
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.model_registry import ModelRegistry
from backend_cpp.asr.namo_detector import NamoTurnDetector
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter
from backend_cpp.asr.transcribe_engine import ASREngineConfig, TranscribeEngine, check_model_supports_streaming

__all__ = [
    "ASREngineConfig",
    "ASRModelManager",
    "AudioBufferManager",
    "BaseASREngine",
    "CommitDeduplicator",
    "ModelRegistry",
    "NamoTurnDetector",
    "SentenceSegmenter",
    "TranscribeEngine",
    "build_family_options",
    "check_model_supports_streaming",
    "normalize_language_for_family",
]
