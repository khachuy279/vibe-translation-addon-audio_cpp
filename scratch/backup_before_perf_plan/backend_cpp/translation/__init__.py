"""Translation subpackage for backend_cpp."""

from backend_cpp.translation.cleaner import clean_translated_text
from backend_cpp.translation.context_manager import ContextManager
from backend_cpp.translation.interfaces import PromptStrategy, TranslationEngine
from backend_cpp.translation.lang_utils import (
    detect_script,
    resolve_effective_source_lang,
    resolve_lang_name,
)
from backend_cpp.translation.local_translator import LocalGGUFTranslator
from backend_cpp.translation.prompt_strategies import (
    build_translation_prompt,
    get_prompt_strategy,
    register_prompt_strategy,
)
from backend_cpp.translation.translator import translate_sentence

from typing import Optional, Any

def get_translator(cfg: Optional[Any] = None) -> TranslationEngine:
    """Return the active singleton TranslationEngine instance, optionally reconfigured."""
    return LocalGGUFTranslator.get_instance(cfg)


def reset_translator() -> None:
    """Reset and release resources of the singleton TranslationEngine."""
    LocalGGUFTranslator.reset_instance()


__all__ = [
    "ContextManager",
    "LocalGGUFTranslator",
    "TranslationEngine",
    "PromptStrategy",
    "get_translator",
    "reset_translator",
    "register_prompt_strategy",
    "translate_sentence",
    "build_translation_prompt",
    "clean_translated_text",
    "detect_script",
    "resolve_effective_source_lang",
    "resolve_lang_name",
    "get_prompt_strategy",
    "TranslationModelRegistry",
]

from backend_cpp.translation.model_registry import TranslationModelRegistry

