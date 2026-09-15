"""Package ASR (Speech-to-Text) kết nối transcribe.cpp cho Backend."""

from backend.asr.base import BaseASREngine
from backend.asr.registry import ModelRegistry
from backend.asr.adapters import build_family_options, normalize_language_for_family
from backend.asr.text_cleaner import clean_transcript_text
from backend.asr.engine import TranscribeEngine

__all__ = [
    "BaseASREngine",
    "ModelRegistry",
    "build_family_options",
    "normalize_language_for_family",
    "clean_transcript_text",
    "TranscribeEngine",
]
