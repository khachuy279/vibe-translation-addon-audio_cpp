"""Sentence segmentation and boundary detection utility for backend_cpp.

Provides:
- Stability-Split: Detects stable preview text that has remained unchanged over
  configured duration/poll cycles.
- Token/Word filtering to drop empty or sub-threshold fragments.
- Substring & Prefix Overlap removal for robust ASR streaming.
"""

import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Common punctuation set covering Latin, CJK symbols, quotes, brackets, and dashes
_PUNCT_CHARS = r'[.,!?;:，。！？；：…\'"“”‘’\-—()\[\]{}<>]'
RE_PUNCTUATION = re.compile(_PUNCT_CHARS)
RE_LEADING_PUNCT_OR_SPACE = re.compile(rf'^(?:{_PUNCT_CHARS}|\s)+')
_RE_WORD_TOKENS = re.compile(r"\b[\w'-]+\b")


def is_cjk(char: str) -> bool:
    """Check if character is Chinese, Japanese, or Korean."""
    if not char:
        return False
    codepoint = ord(char)
    return (
        (0x4E00 <= codepoint <= 0x9FFF) or
        (0x3040 <= codepoint <= 0x309F) or
        (0x30A0 <= codepoint <= 0x30FF) or
        (0x1100 <= codepoint <= 0x11FF) or
        (0xAC00 <= codepoint <= 0xD7AF)
    )


def count_content_tokens(text: str) -> int:
    """Count meaningful words/tokens in text, properly supporting both Latin and CJK scripts."""
    if not text or not text.strip():
        return 0

    cjk_count = 0
    for ch in text:
        if is_cjk(ch):
            cjk_count += 1

    if cjk_count == 0:
        return len(_RE_WORD_TOKENS.findall(text))

    non_cjk_text = "".join(ch for ch in text if not is_cjk(ch))
    latin_words = len(_RE_WORD_TOKENS.findall(non_cjk_text)) if non_cjk_text else 0
    return cjk_count + latin_words


def _normalize_word(w: str) -> str:
    """Strip surrounding punctuation and lowercase a word for stability matching."""
    return w.strip(".,!?;:，。！？；：…'\"“”‘’").lower()


class SentenceSegmenter:
    """Manages sentence boundaries via stability tracking and word filtering."""

    def __init__(
        self,
        max_chars: int = 150,
        max_duration_sec: float = 15.0,
        min_words_to_commit: int = 1,
        min_words_to_emit_final: Optional[int] = None,
        split_on_stability: bool = True,
        stability_duration_sec: float = 2.0,
        stability_threshold_polls: int = 2,
    ):
        self.max_chars = max_chars
        self.max_duration_sec = max_duration_sec
        self.min_words_to_commit = min_words_to_commit
        self.min_words_to_emit_final = (
            min_words_to_emit_final if min_words_to_emit_final is not None else min_words_to_commit
        )
        self.split_on_stability = split_on_stability
        self.stability_duration_sec = stability_duration_sec
        self.stability_threshold_polls = stability_threshold_polls

        # Stability state
        self._last_stable_text: str = ""
        self._stability_start_time: Optional[float] = None
        self._stable_poll_count: int = 0

    def reset_stability(self) -> None:
        """Reset internal stability timer and text cache."""
        self._last_stable_text = ""
        self._stability_start_time = None
        self._stable_poll_count = 0

    def reset(self) -> None:
        """Full reset between utterances."""
        self.reset_stability()

    def is_text_filtered(self, text: str) -> bool:
        """Check if preview / intermediate text is too short to commit."""
        if not text or not text.strip():
            return True
        return count_content_tokens(text.strip()) < self.min_words_to_commit

    def is_final_too_short(self, text: str) -> bool:
        """Check if final commit text is too short to emit.
        
        Preserves natural short turns (e.g. 'はい。', 'Yes.', 'OK.') while discarding
        empty or punctuation-only strings.
        """
        if not text or not text.strip():
            return True
        cleaned = RE_PUNCTUATION.sub('', text).strip()
        if not cleaned:
            return True
        return count_content_tokens(text.strip()) < self.min_words_to_emit_final

    def should_commit(self, text: str, audio_duration_sec: float) -> bool:
        """Check if audio or text bounds exceed single sentence limits."""
        if not text:
            return False
        if len(text.strip()) >= self.max_chars:
            return True
        if audio_duration_sec >= self.max_duration_sec:
            return True
        return False

    @staticmethod
    def normalize_for_comparison(text: str) -> str:
        """Strip punctuation and whitespace, lowercase for robust duplicate checks."""
        if not text:
            return ""
        cleaned = RE_PUNCTUATION.sub('', text.lower())
        return " ".join(cleaned.split())

    @classmethod
    def remove_prefix_overlap(cls, prefix: str, text: str) -> str:
        """Remove prefix (or near-match prefix) from text if it starts with it."""
        if not prefix or not text:
            return text

        p_norm = cls.normalize_for_comparison(prefix)
        t_norm = cls.normalize_for_comparison(text)

        if not p_norm or not t_norm:
            return text

        # Exact normalized match
        if p_norm == t_norm:
            return ""

        # Check if text starts with prefix exactly
        clean_text = text.strip()
        clean_prefix = prefix.strip()
        if clean_text.lower().startswith(clean_prefix.lower()):
            remainder = clean_text[len(clean_prefix):].strip()
            return RE_LEADING_PUNCT_OR_SPACE.sub('', remainder).strip()

        # Character-level prefix match for CJK or normalized prefix match
        if t_norm.startswith(p_norm):
            p_len = len(p_norm)
            accum_len = 0
            cut_idx = len(clean_text)
            for i, ch in enumerate(clean_text):
                norm_ch = cls.normalize_for_comparison(ch)
                if norm_ch:
                    accum_len += len(norm_ch)
                    if accum_len >= p_len:
                        cut_idx = i + 1
                        break
            remainder = clean_text[cut_idx:].strip()
            return RE_LEADING_PUNCT_OR_SPACE.sub('', remainder).strip()

        # Word-level prefix removal
        p_words = p_norm.split()
        t_words = t_norm.split()
        if len(t_words) >= len(p_words) and t_words[:len(p_words)] == p_words:
            raw_words = clean_text.split()
            if len(raw_words) >= len(p_words):
                remainder = " ".join(raw_words[len(p_words):]).strip()
                return RE_LEADING_PUNCT_OR_SPACE.sub('', remainder).strip()

        return text

    def check_stability(self, text: str) -> bool:
        """Check if partial preview text has remained stable over stability_duration_sec."""
        if not self.split_on_stability or not text or not text.strip():
            self.reset_stability()
            return False

        clean = text.strip()
        if count_content_tokens(clean) < self.min_words_to_commit:
            return False

        now = time.monotonic()
        is_exact = (clean.lower() == self._last_stable_text.lower())

        clean_words = clean.split()
        last_words = self._last_stable_text.split() if self._last_stable_text else []
        common_len = 0
        for w1, w2 in zip(clean_words, last_words):
            if _normalize_word(w1) == _normalize_word(w2):
                common_len += 1
            else:
                break
        is_prefix_stable = (
            (common_len >= 3 and len(clean_words) == len(last_words))
            or (common_len > 0 and common_len == len(clean_words) == len(last_words))
        )

        if is_exact or is_prefix_stable:
            self._stable_poll_count += 1
            if self._stability_start_time is None:
                self._stability_start_time = now
                return False

            elapsed = now - self._stability_start_time
            if elapsed >= self.stability_duration_sec and self._stable_poll_count >= self.stability_threshold_polls:
                return True
            return False
        else:
            self._last_stable_text = clean
            self._stability_start_time = now
            self._stable_poll_count = 1
            return False
