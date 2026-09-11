"""Text sanitization and post-processing for translation output."""

import re

# Pre-compiled regular expressions for high-frequency cleaning
_RE_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_RE_PREFIXES = re.compile(
    r"^(?:"
    r"翻译结果[：:]|翻译[：:]|译文[：:]|结果[：:]|"
    r"Translation:|Translated\s*text:|Result:|"
    r"Vietnamese:|Tiếng\s*[Vv]iệt[：:]|"
    r"K[eế]t\s*qu[aả][：:]|Nhập:|Input:|"
    r"Dịch\s*câu:|Dịch:|Bản\s*dịch:"
    r")\s*",
    re.IGNORECASE,
)

_CHAT_DELIMITERS = ["<|", "</s>", "<eos>", "<end>", "<end_of_turn>"]


def clean_translated_text(raw_text: str, keep_multiline: bool = False) -> str:
    """Remove thought tags, role tokens, quotes, and unwanted prefixes."""
    if not raw_text:
        return ""
    text = _RE_THINK_TAG.sub("", raw_text).strip()

    # Cut off cleanly at any chat template or special token delimiter (<|, </s>, etc.)
    for token in _CHAT_DELIMITERS:
        if token in text:
            text = text.split(token)[0].strip()

    # Extract first line for single-line subtitle streaming unless keep_multiline is explicitly requested
    if not keep_multiline:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            text = lines[0]

    # Strip prompt leakage prefixes with pre-compiled regex
    text = _RE_PREFIXES.sub("", text).strip()

    # Strip enclosing quotes
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "'" and text[-1] == "'")
        or (text[0] == "“" and text[-1] == "”")
    ):
        text = text[1:-1].strip()
    return text
