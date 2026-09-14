"""Sentence Committer and Boundary Manager for backend_audio_cpp.

Manages stability tracking, punctuation splitting, CJK/Latin token filtering,
and structured commit decision logging.
"""

import logging
import re
import time
from typing import List, Optional, Tuple, Union

from .sentence_config import SentenceConfig

logger = logging.getLogger("backend_audio_cpp.commit")

# Punctuation sets covering Latin, CJK, brackets, and quotes
_PUNCT_CHARS = r'[.,!?;:，。！？；：…\'"“”‘’\-—()\[\]{}<>]'
RE_PUNCTUATION = re.compile(_PUNCT_CHARS)
RE_SENTENCE_ENDINGS = re.compile(r'[.!?;。！？；…]+')
_RE_WORD_TOKENS = re.compile(r"\b[\w'-]+\b")


def is_cjk(char: str) -> bool:
    """Check if character is Chinese, Japanese, or Korean."""
    if not char:
        return False
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF) or
        (0x3040 <= cp <= 0x309F) or
        (0x30A0 <= cp <= 0x30FF) or
        (0x1100 <= cp <= 0x11FF) or
        (0xAC00 <= cp <= 0xD7AF)
    )


def count_content_tokens(text: str) -> int:
    """Count content words/characters supporting both Latin and CJK scripts."""
    if not text or not text.strip():
        return 0
    cjk_count = sum(1 for ch in text if is_cjk(ch))
    if cjk_count == 0:
        return len(_RE_WORD_TOKENS.findall(text))
    non_cjk = "".join(ch for ch in text if not is_cjk(ch))
    latin_words = len(_RE_WORD_TOKENS.findall(non_cjk)) if non_cjk else 0
    return cjk_count + latin_words


class SentenceCommitter:
    """Evaluates streaming preview text against SentenceConfig and VAD events to trigger commits."""

    def __init__(self, config: Optional[SentenceConfig] = None):
        self.config = config or SentenceConfig()

        # Stability state
        self._last_preview_text: str = ""
        self._stability_start_time: Optional[float] = None
        self._stable_poll_count: int = 0

        # Committed history for current utterance
        self.committed_sentences: List[str] = []
        self.uncommitted_offset: int = 0

    def reset(self) -> None:
        """Reset state between utterances."""
        self._last_preview_text = ""
        self._stability_start_time = None
        self._stable_poll_count = 0
        self.committed_sentences.clear()
        self.uncommitted_offset = 0

    def check_stability_commit(
        self,
        current_preview: str,
        timestamp_sec: float,
        utt_id: Union[int, str],
    ) -> Optional[Tuple[str, str]]:
        """Check if preview text has stabilized enough to trigger a commit.

        Returns:
            (committed_text, reason) if triggered, else None.
        """
        current_clean = current_preview.strip()
        if not current_clean:
            self._last_preview_text = ""
            self._stability_start_time = None
            self._stable_poll_count = 0
            return None

        # Determine if text is unchanged from last poll
        if current_clean == self._last_preview_text:
            if self._stability_start_time is None:
                self._stability_start_time = timestamp_sec
            self._stable_poll_count += 1
        else:
            self._last_preview_text = current_clean
            self._stability_start_time = timestamp_sec
            self._stable_poll_count = 1

        stable_dur = timestamp_sec - (self._stability_start_time or timestamp_sec)

        # 1. Stability Trigger
        if (
            self.config.split_on_stability
            and self._stable_poll_count >= self.config.stability_threshold_polls
            and stable_dur >= self.config.stability_duration_sec
        ):
            if count_content_tokens(current_clean) >= self.config.min_words_to_commit:
                reason = (
                    f"STABILITY (text stable for {stable_dur:.1f}s over {self._stable_poll_count} polls)"
                )
                self._log_commit(utt_id, current_clean, reason)
                self.reset()
                return current_clean, reason

        # 2. Max Characters Trigger
        if len(current_clean) >= self.config.max_chars:
            # Try to split at natural punctuation
            split_idx = -1
            for m in RE_SENTENCE_ENDINGS.finditer(current_clean):
                if m.end() >= len(current_clean) // 2:
                    split_idx = m.end()
            
            if split_idx > 0:
                to_commit = current_clean[:split_idx].strip()
                remainder = current_clean[split_idx:].strip()
                reason = f"MAX_CHARS_REACHED ({len(current_clean)} >= {self.config.max_chars})"
                self._log_commit(utt_id, to_commit, reason)
                self._last_preview_text = remainder
                self._stability_start_time = timestamp_sec
                self._stable_poll_count = 1
                return to_commit, reason

        return None

    def commit_on_vad_silence(
        self,
        final_text: str,
        silence_dur_sec: float,
        utt_id: Union[int, str],
    ) -> Optional[Tuple[str, str]]:
        """Trigger final commit when VAD detects silence timeout."""
        clean_text = final_text.strip()
        if not clean_text:
            self.reset()
            return None

        if count_content_tokens(clean_text) < self.config.min_words_to_commit:
            logger.debug(f"[COMMIT] [utt_{utt_id}] Ignored short utterance: '{clean_text}'")
            self.reset()
            return None

        reason = f"VAD_SILENCE (speaker paused {silence_dur_sec:.2f}s)"
        self._log_commit(utt_id, clean_text, reason)
        self.reset()
        return clean_text, reason

    def commit_on_max_duration(
        self,
        final_text: str,
        speech_dur_sec: float,
        utt_id: Union[int, str],
    ) -> Optional[Tuple[str, str]]:
        """Trigger commit when max speech duration is reached."""
        clean_text = final_text.strip()
        if not clean_text:
            self.reset()
            return None

        reason = f"MAX_DURATION_REACHED ({speech_dur_sec:.1f}s limit)"
        self._log_commit(utt_id, clean_text, reason)
        self.reset()
        return clean_text, reason

    def commit_on_stream_eof(
        self,
        final_text: str,
        utt_id: Union[int, str],
    ) -> Optional[Tuple[str, str]]:
        """Trigger final commit when stream ends."""
        clean_text = final_text.strip()
        if not clean_text:
            self.reset()
            return None

        reason = "STREAM_EOF"
        self._log_commit(utt_id, clean_text, reason)
        self.reset()
        return clean_text, reason

    def _log_commit(self, utt_id: Union[int, str], text: str, reason: str) -> None:
        """Standardized commit logging format."""
        logger.info(f'[COMMIT] [utt_{utt_id}] >> COMMITTED: "{text}" | Reason: {reason}')
