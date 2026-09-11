"""Translation entry point for backend_cpp."""

from typing import Optional, Dict, Any
from backend_cpp.translation.interfaces import TranslationEngine
from backend_cpp.translation.local_translator import LocalGGUFTranslator


async def translate_sentence(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "vi",
    context: str = "",
    timeout: Optional[float] = None,
    translator: Optional[TranslationEngine] = None,
) -> Dict[str, Any]:
    """Translate a sentence using any engine conforming to TranslationEngine protocol."""
    engine = translator or LocalGGUFTranslator.get_instance()
    return await engine.translate(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        context=context,
        timeout=timeout,
    )
