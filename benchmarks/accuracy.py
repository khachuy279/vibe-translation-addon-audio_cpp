"""Accuracy metrics: Word Error Rate (WER) and Character Error Rate (CER) per language."""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import rapidfuzz.distance.Levenshtein as lev
import jieba

# Silence jieba logs
jieba.setLogLevel(60)


@dataclass
class AccuracyResult:
    """Detailed accuracy assessment comparing reference and hypothesis."""
    raw_reference: str
    normalized_reference: str
    raw_hypothesis: str
    normalized_hypothesis: str
    language: str
    wer: Optional[float]
    cer: Optional[float]
    substitutions: int
    deletions: int
    insertions: int
    ref_length: int
    hyp_length: int
    is_annotated: bool

    def to_dict(self) -> dict:
        return {
            "raw_reference": self.raw_reference,
            "normalized_reference": self.normalized_reference,
            "raw_hypothesis": self.raw_hypothesis,
            "normalized_hypothesis": self.normalized_hypothesis,
            "language": self.language,
            "wer": round(self.wer, 4) if self.wer is not None else None,
            "cer": round(self.cer, 4) if self.cer is not None else None,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "ref_length": self.ref_length,
            "hyp_length": self.hyp_length,
            "is_annotated": self.is_annotated,
        }


def normalize_transcript(text: str, language: str = "en") -> str:
    """Normalize text for ASR accuracy evaluation."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)

    lang_lower = (language or "").lower()
    if lang_lower in ("zh", "chinese", "ja", "japanese"):
        # Strip all CJK & western punctuation and whitespace
        t = re.sub(r"[，。！？、；：“”‘’（）《》【】…—\.,!\?:;\"\'\(\)\[\]\-\r\n\t ]", "", t)
    else:
        # Western / Slavic: lowercase, strip punctuation except internal apostrophes
        t = t.lower()
        t = re.sub(r"[^\w\s\']", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize_words(text: str, language: str = "en") -> List[str]:
    """Tokenize normalized text into words based on language."""
    if not text:
        return []
    lang_lower = (language or "").lower()
    if lang_lower in ("zh", "chinese"):
        return [w for w in jieba.cut(text) if w.strip()]
    elif lang_lower in ("ja", "japanese"):
        # For Japanese character-level or bigram tokenization
        return list(text)
    else:
        # Whitespace tokenization for western languages
        return text.split()


def calculate_cer(ref_norm: str, hyp_norm: str) -> Tuple[Optional[float], int, int, int, int]:
    """Compute Character Error Rate (CER) using Levenshtein distance."""
    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)

    ref_len = len(ref_chars)
    hyp_len = len(hyp_chars)

    if ref_len == 0:
        return None, 0, 0, hyp_len, 0

    # Calculate edit ops
    dist = lev.distance(ref_chars, hyp_chars)
    # Using rapidfuzz edit operations to decompose into S, D, I
    ops = lev.editops(ref_chars, hyp_chars)
    s = sum(1 for op in ops if op.tag == "replace")
    d = sum(1 for op in ops if op.tag == "delete")
    i = sum(1 for op in ops if op.tag == "insert")

    cer = dist / float(ref_len)
    return cer, s, d, i, ref_len


def calculate_wer(ref_norm: str, hyp_norm: str, language: str = "en") -> Tuple[Optional[float], int, int, int, int]:
    """Compute Word Error Rate (WER) using Levenshtein distance over word tokens."""
    ref_words = tokenize_words(ref_norm, language=language)
    hyp_words = tokenize_words(hyp_norm, language=language)

    ref_len = len(ref_words)
    hyp_len = len(hyp_words)

    if ref_len == 0:
        return None, 0, 0, hyp_len, 0

    dist = lev.distance(ref_words, hyp_words)
    ops = lev.editops(ref_words, hyp_words)
    s = sum(1 for op in ops if op.tag == "replace")
    d = sum(1 for op in ops if op.tag == "delete")
    i = sum(1 for op in ops if op.tag == "insert")

    wer = dist / float(ref_len)
    return wer, s, d, i, ref_len


def evaluate_accuracy(
    raw_reference: str,
    raw_hypothesis: str,
    language: str = "en",
) -> AccuracyResult:
    """Evaluate accuracy with appropriate language metrics and normalization."""
    is_annotated = bool(raw_reference and raw_reference.strip())

    norm_ref = normalize_transcript(raw_reference, language=language) if is_annotated else ""
    norm_hyp = normalize_transcript(raw_hypothesis, language=language)

    if not is_annotated:
        return AccuracyResult(
            raw_reference=raw_reference,
            normalized_reference="",
            raw_hypothesis=raw_hypothesis,
            normalized_hypothesis=norm_hyp,
            language=language,
            wer=None,
            cer=None,
            substitutions=0,
            deletions=0,
            insertions=len(norm_hyp),
            ref_length=0,
            hyp_length=len(norm_hyp),
            is_annotated=False,
        )

    # Calculate CER
    cer, c_s, c_d, c_i, c_ref_len = calculate_cer(norm_ref, norm_hyp)

    # Calculate WER
    wer, w_s, w_d, w_i, w_ref_len = calculate_wer(norm_ref, norm_hyp, language=language)

    # Primary metric depending on language
    lang_lower = (language or "").lower()
    if lang_lower in ("zh", "chinese", "ja", "japanese"):
        # For CJK, substitutions, deletions, insertions are character-level
        subs, dels, ins = c_s, c_d, c_i
        ref_len = c_ref_len
        hyp_len = len(norm_hyp)
    else:
        # For Western languages, use word counts
        subs, dels, ins = w_s, w_d, w_i
        ref_len = w_ref_len
        hyp_len = len(tokenize_words(norm_hyp, language=language))

    return AccuracyResult(
        raw_reference=raw_reference,
        normalized_reference=norm_ref,
        raw_hypothesis=raw_hypothesis,
        normalized_hypothesis=norm_hyp,
        language=language,
        wer=wer,
        cer=cer,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        ref_length=ref_len,
        hyp_length=hyp_len,
        is_annotated=True,
    )
