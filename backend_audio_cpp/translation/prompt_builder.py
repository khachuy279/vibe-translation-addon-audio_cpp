"""Prompt builder and language resolution for Hunyuan-MT2 (Hy-MT2) models."""

import re
from typing import Dict, List, Optional

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

_RE_KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_RE_HANGUL = re.compile(r"[\uac00-\ud7af]")
_RE_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_CYRILLIC = re.compile(r"[\u0400-\u04ff]")

STOP_TOKENS: List[str] = [
    "<|im_end|>",
    "<|im_end",
    "<|im_start|>",
    "<|im_start",
    "<|endoftext|>",
    "</s>",
    "\n\n",
]


def resolve_lang_name(code_or_name: str, use_chinese: bool = True) -> str:
    """Normalize language code or name to standard target language name."""
    if not code_or_name:
        return "越南语" if use_chinese else "Vietnamese"
    key = code_or_name.lower().strip()
    if use_chinese:
        return LANG_NAME_MAP_ZH.get(key, code_or_name)
    return LANG_NAME_MAP_EN.get(key, code_or_name.title())


def detect_script(text: str) -> str:
    """Detect dominant language script from text characters."""
    if _RE_KANA.search(text):
        return "ja"
    if _RE_HANGUL.search(text):
        return "ko"
    if _RE_CJK.search(text):
        return "zh"
    if _RE_CYRILLIC.search(text):
        return "ru"
    return "en"


def build_translation_prompt(
    text: str,
    target_lang: str = "vi",
    source_lang: str = "auto",
    context: str = "",
) -> str:
    """Build official Tencent ChatML translation prompt.

    Hunyuan-MT2 natively detects source language when given target language name.
    """
    clean_text = text.strip()
    tgt_name_zh = resolve_lang_name(target_lang, use_chinese=True)

    if context and context.strip():
        user_content = (
            f"【背景信息】\n"
            f"{context.strip()}\n\n"
            f"请结合背景信息将以下文本翻译为{tgt_name_zh}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
            f"{clean_text}"
        )
    else:
        user_content = (
            f"将以下文本翻译为{tgt_name_zh}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
            f"{clean_text}"
        )

    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
