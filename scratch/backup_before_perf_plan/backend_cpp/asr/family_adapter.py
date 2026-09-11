"""Adapter for family-specific runtime options and language normalization in transcribe.cpp."""

import logging
from typing import Any, Dict, Optional
import transcribe_cpp

logger = logging.getLogger(__name__)

# Standard 2-letter to BCP-47 locale map for models requiring full locale (e.g. Nemotron/Parakeet)
_NEMOTRON_LOCALE_MAP: Dict[str, str] = {
    "en": "en-US",
    "zh": "zh-CN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "ru": "ru-RU",
    "nl": "nl-NL",
    "hi": "hi-IN",
    "ar": "ar-AR",
    "pl": "pl-PL",
    "sv": "sv-SE",
    "tr": "tr-TR",
    "uk": "uk-UA",
    "cs": "cs-CZ",
    "id": "id-ID",
    "vi": "vi-VN",
    "th": "th-TH",
    "ro": "ro-RO",
    "hu": "hu-HU",
    "el": "el-GR",
    "da": "da-DK",
    "fi": "fi-FI",
    "no": "no-NO",
}

# SenseVoice supported languages
_SENSEVOICE_LANGUAGES = {"zh", "en", "ja", "ko", "yue"}


def normalize_language_for_family(lang: Optional[str], family: str) -> Optional[str]:
    """Normalize input language code to the format required by the model family.

    Args:
        lang: Language string from client/config (e.g. "en", "en-US", "vi", "auto").
        family: Model family name (e.g. "nemotron", "whisper", "qwen3_asr", "sensevoice").

    Returns:
        Properly formatted language string, or None for automatic language detection.
    """
    if not lang or lang.strip().lower() in ("auto", "none"):
        return None

    clean = lang.strip().replace("_", "-")
    family_norm = (family or "").strip().lower()

    # 1. Nemotron / Parakeet family requires full BCP-47 locale (e.g. en-US, zh-CN)
    if family_norm in ("nemotron", "parakeet"):
        clean_lower = clean.lower()
        if clean_lower in _NEMOTRON_LOCALE_MAP:
            return _NEMOTRON_LOCALE_MAP[clean_lower]
        for k, v in _NEMOTRON_LOCALE_MAP.items():
            if clean_lower == v.lower():
                return v
        if "-" in clean:
            parts = clean.split("-")
            return f"{parts[0].lower()}-{parts[1].upper()}"
        return None

    # 2. SenseVoice only accepts 5 specific codes
    if family_norm == "sensevoice":
        base_code = clean.split("-")[0].lower()
        return base_code if base_code in _SENSEVOICE_LANGUAGES else None

    # 3. Whisper, Qwen3, Cohere, Voxtral: standard 2-letter ISO 639-1 code
    base_code = clean.split("-")[0].lower()
    return base_code


def build_family_options(
    family: str,
    model_info: Optional[Dict[str, Any]] = None,
    model: Optional[transcribe_cpp.Model] = None,
    slot: Optional[str] = None,
) -> Optional[transcribe_cpp.FamilyExtension]:
    """Build the appropriate FamilyExtension options subclass for the model family.

    Args:
        family: Family name (e.g. "nemotron", "parakeet", "whisper", "voxtral").
        model_info: Optional dictionary of model settings from models.yaml.
        model: Optional loaded transcribe_cpp.Model instance to verify model.accepts(options).
        slot: Optional slot name ('stream' or 'run'). If specified, options
            not matching this slot will be filtered out (returning None).

    Returns:
        Configured FamilyExtension instance, or None if not applicable.
    """
    opts: Optional[transcribe_cpp.FamilyExtension] = None
    info = model_info or {}
    family_norm = (family or "").strip().lower()

    try:
        if family_norm in ("nemotron", "parakeet"):
            # att_context_right: 0 (0ms), 3 (240ms), 6 (480ms), 13 (1040ms - default/highest accuracy)
            att_right = int(info.get("att_context_right", 13))
            opts = transcribe_cpp.ParakeetStreamOptions(att_context_right=att_right)

        elif family_norm == "voxtral":
            delay = int(info.get("num_delay_tokens", 2))
            min_interval = int(info.get("min_decode_interval_ms", 100))
            opts = transcribe_cpp.VoxtralRealtimeStreamOptions(
                num_delay_tokens=delay,
                min_decode_interval_ms=min_interval,
            )

        elif family_norm == "whisper":
            cond = bool(info.get("condition_on_prev_tokens", False))
            prompt = info.get("initial_prompt", None)
            opts = transcribe_cpp.WhisperRunOptions(
                condition_on_prev_tokens=cond,
                initial_prompt=prompt,
            )

        elif family_norm == "moonshine":
            min_interval = int(info.get("min_decode_interval_ms", 100))
            opts = transcribe_cpp.MoonshineStreamingOptions(min_decode_interval_ms=min_interval)

    except Exception as e:
        logger.warning(f"Failed to build family options for family '{family}': {e}")
        return None

    if opts is not None:
        if slot is not None:
            opt_slot = getattr(opts, "_slot", "")
            if opt_slot and opt_slot != slot:
                return None

        if model is not None:
            try:
                if not model.accepts(opts):
                    logger.warning(
                        f"Model '{getattr(model, 'arch', '')}' rejected options of type {type(opts).__name__}"
                    )
                    return None
            except Exception:
                pass

    return opts
