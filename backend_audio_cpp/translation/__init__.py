"""Translation module for backend_audio_cpp based on Hunyuan-MT2 (Hy-MT2) 7B."""

from backend_audio_cpp.translation.hy_translator import (
    HyMTTranslator,
    TranslationConfig,
    TranslationResult,
)

__all__ = ["HyMTTranslator", "TranslationConfig", "TranslationResult"]
