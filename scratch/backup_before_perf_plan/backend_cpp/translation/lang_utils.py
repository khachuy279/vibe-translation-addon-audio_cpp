"""Language normalization, script detection, and anti-hallucination utilities."""

import re
from typing import Dict

_LANG_PAIRS = [
    # (code, full_name, zh_name, en_name)
    ("vi", "vietnamese", "越南语", "Vietnamese"),
    ("en", "english", "英语", "English"),
    ("zh", "chinese", "中文", "Chinese"),
    ("ja", "japanese", "日语", "Japanese"),
    ("ko", "korean", "韩语", "Korean"),
    ("es", "spanish", "西班牙语", "Spanish"),
    ("fr", "french", "法语", "French"),
    ("de", "german", "德语", "German"),
    ("ru", "russian", "俄语", "Russian"),
    ("th", "thai", "泰语", "Thai"),
    ("id", "indonesian", "印尼语", "Indonesian"),
    ("pt", "portuguese", "葡萄牙语", "Portuguese"),
    ("it", "italian", "意大利语", "Italian"),
    ("ar", "arabic", "阿拉伯语", "Arabic"),
]

LANG_NAME_MAP_ZH: Dict[str, str] = {
    key: zh for code, name, zh, _ in _LANG_PAIRS for key in (code, name)
}
LANG_NAME_MAP_EN: Dict[str, str] = {
    key: en for code, name, _, en in _LANG_PAIRS for key in (code, name)
}

# Pre-compiled regular expressions for high-frequency script detection
_RE_KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_RE_HANGUL = re.compile(r"[\uac00-\ud7af]")
_RE_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_CYRILLIC = re.compile(r"[\u0400-\u04ff]")


def resolve_lang_name(code_or_name: str, use_chinese: bool = True) -> str:
    """Normalize language code or name to standard full name (Chinese or English)."""
    if not code_or_name:
        return "越南语" if use_chinese else "Vietnamese"
    key = code_or_name.lower().strip()
    if use_chinese:
        return LANG_NAME_MAP_ZH.get(key, code_or_name)
    return LANG_NAME_MAP_EN.get(key, code_or_name.title())


def detect_script(text: str) -> str:
    """Detect dominant language script from text characters using pre-compiled regex."""
    if _RE_KANA.search(text):
        return "Japanese"
    if _RE_HANGUL.search(text):
        return "Korean"
    if _RE_CJK.search(text):
        return "Chinese"
    if _RE_CYRILLIC.search(text):
        return "Russian"
    return "English"


def resolve_effective_source_lang(text: str, configured_source: str, use_chinese: bool = False) -> str:
    """Resolve source language name, validating against text script to prevent hallucination."""
    if not text:
        return resolve_lang_name(configured_source, use_chinese=use_chinese)

    has_hangul = bool(_RE_HANGUL.search(text))
    has_kana = bool(_RE_KANA.search(text))
    has_cyrillic = bool(_RE_CYRILLIC.search(text))

    cfg_lower = (configured_source or "auto").lower().strip()

    # Script-based override if configured source conflicts with unmistakable alphabets
    if has_hangul and not has_kana and cfg_lower in ("ja", "japanese", "zh", "chinese", "en", "english", "auto"):
        return "韩语" if use_chinese else "Korean"
    if has_kana and cfg_lower in ("ko", "korean", "zh", "chinese", "en", "english", "auto"):
        return "日语" if use_chinese else "Japanese"
    if has_cyrillic and cfg_lower not in ("ru", "russian"):
        return "俄语" if use_chinese else "Russian"

    if not configured_source or cfg_lower == "auto":
        detected = detect_script(text)
        return resolve_lang_name(detected, use_chinese=use_chinese)

    return resolve_lang_name(configured_source, use_chinese=use_chinese)
