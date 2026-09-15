"""Bộ công cụ làm sạch văn bản và loại bỏ special tokens từ Audio-LLMs."""

import re

# Biểu thức chính quy loại bỏ token đặc biệt dạng <|tag|> hoặc <tag>
_RE_SPECIAL_TAGS = re.compile(r"<\|.*?\|>|<[^>]+>")
_RE_SYSTEM_PREFIX = re.compile(r"(?m)^(?:system|user|assistant|language\s+\w+)\s*[:]?\s*", flags=re.IGNORECASE)


def clean_transcript_text(raw_text: str) -> str:
    """Làm sạch văn bản đầu ra từ ASR model.
    
    - Loại bỏ các special tokens như <|transcribe|>, <|en|>, <|notimestamps|>.
    - Loại bỏ tiền tố role chat system/user/assistant.
    - Cắt bỏ khoảng trắng thừa hai đầu.
    """
    if not raw_text:
        return ""

    text = _RE_SPECIAL_TAGS.sub("", raw_text)
    text = _RE_SYSTEM_PREFIX.sub("", text)
    return text.strip()
