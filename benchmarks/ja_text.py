"""Japanese text normalization for ASR scoring.

The generic normalizer in `benchmarks.accuracy` strips punctuation and then compares
characters. For Japanese that is not enough, because equal-meaning outputs differ
*orthographically* and get scored as substitutions:

    五十円  vs  50円              (kanji numeral vs Arabic)
    １万2345 vs 1万2345           (full-width vs half-width)
    令和三年 vs 令和3年

Measured impact: on `transcribe.cpp/samples/ja.wav`, three otherwise-perfect systems
were each charged 5.88% CER purely for emitting `50円` instead of `五十円`.

This module provides:

* `normalize_ja(text, itn=False)` -- NFKC + punctuation/space strip, optionally with
  kanji-numeral -> Arabic conversion on both sides of the comparison.
* `score_ja(ref, hyp)` -- returns CER under BOTH conventions so a report can never
  hide a regression behind a friendlier metric.

The `strict` figure (itn=False) is the headline number; `itn` is a secondary view
that answers "did the model actually hear the right words?".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple

import rapidfuzz.distance.Levenshtein as lev

# Kanji digits and their positional units.
_KANJI_DIGITS = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_LARGE_UNITS = {"万": 10**4, "億": 10**8, "兆": 10**12}

_KANJI_NUMERAL_CHARS = set(_KANJI_DIGITS) | set(_SMALL_UNITS) | set(_LARGE_UNITS)

# Punctuation / symbols / whitespace to drop. NFKC already folded full-width forms.
_PUNCT_RE = re.compile(
    r"[\s、。，．・！？!?「」『』（）()\[\]{}〈〉《》【】〜～ー―—–\-‐…‥：:；;／/＼\\＠@＃#＄$％%＆&＊*＋+＝=＜<＞>\"'“”‘’､｡｢｣]"
)


@dataclass
class JaScore:
    """CER of one hypothesis under two normalization conventions."""

    cer: Optional[float]           # headline: strict (no ITN folding)
    cer_itn: Optional[float]       # secondary: kanji numerals folded to Arabic
    ref_len: int
    hyp_len: int
    norm_ref: str
    norm_hyp: str
    norm_ref_itn: str
    norm_hyp_itn: str
    substitutions: int
    deletions: int
    insertions: int
    substitutions_itn: int
    deletions_itn: int
    insertions_itn: int

    def to_dict(self) -> dict:
        def r(v):
            return round(v, 4) if v is not None else None

        return {
            "cer": r(self.cer),
            "cer_itn": r(self.cer_itn),
            "ref_len": self.ref_len,
            "hyp_len": self.hyp_len,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "substitutions_itn": self.substitutions_itn,
            "deletions_itn": self.deletions_itn,
            "insertions_itn": self.insertions_itn,
        }


def _kanji_number_to_arabic(token: str) -> str:
    """Convert a kanji numeral token (e.g. 一万二千三百四十五) to Arabic digits."""
    total = 0
    section = 0
    current = 0
    seen_any = False

    for ch in token:
        if ch in _KANJI_DIGITS:
            current = _KANJI_DIGITS[ch]
            seen_any = True
        elif ch in _SMALL_UNITS:
            unit = _SMALL_UNITS[ch]
            # "十" alone means 10, "二十" means 20.
            section += (current or 1) * unit
            current = 0
            seen_any = True
        elif ch in _LARGE_UNITS:
            unit = _LARGE_UNITS[ch]
            section += current
            total += (section or 1) * unit
            section = 0
            current = 0
            seen_any = True
        else:
            return token  # not a pure numeral; leave untouched
    if not seen_any:
        return token
    return str(total + section + current)


def normalize_ja(text: str, itn: bool = False) -> str:
    """Normalize Japanese text for CER.

    Args:
        text: raw transcript.
        itn: when True, additionally fold kanji numerals to Arabic digits so that
            ``五十円`` and ``50円`` compare equal.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _PUNCT_RE.sub("", t)
    # Strip anything still classified as punctuation/symbol/space.
    t = "".join(ch for ch in t if not unicodedata.category(ch).startswith(("P", "Z", "C")))
    if not itn:
        return t

    out = []
    i = 0
    n = len(t)
    while i < n:
        if t[i] in _KANJI_NUMERAL_CHARS:
            j = i
            while j < n and t[j] in _KANJI_NUMERAL_CHARS:
                j += 1
            out.append(_kanji_number_to_arabic(t[i:j]))
            i = j
        else:
            out.append(t[i])
            i += 1
    return "".join(out)


def _cer(norm_ref: str, norm_hyp: str) -> Tuple[Optional[float], int, int, int, int]:
    ref_chars = list(norm_ref)
    hyp_chars = list(norm_hyp)
    if not ref_chars:
        return None, 0, 0, len(hyp_chars), 0
    ops = lev.editops(ref_chars, hyp_chars)
    s = sum(1 for o in ops if o.tag == "replace")
    d = sum(1 for o in ops if o.tag == "delete")
    ins = sum(1 for o in ops if o.tag == "insert")
    return lev.distance(ref_chars, hyp_chars) / float(len(ref_chars)), s, d, ins, len(ref_chars)


def score_ja(reference: str, hypothesis: str) -> JaScore:
    """Score a Japanese hypothesis against a reference under both conventions."""
    nr = normalize_ja(reference, itn=False)
    nh = normalize_ja(hypothesis, itn=False)
    nr_i = normalize_ja(reference, itn=True)
    nh_i = normalize_ja(hypothesis, itn=True)

    cer, s, d, ins, ref_len = _cer(nr, nh)
    cer_i, s_i, d_i, ins_i, _ = _cer(nr_i, nh_i)

    return JaScore(
        cer=cer,
        cer_itn=cer_i,
        ref_len=ref_len,
        hyp_len=len(nh),
        norm_ref=nr,
        norm_hyp=nh,
        norm_ref_itn=nr_i,
        norm_hyp_itn=nh_i,
        substitutions=s,
        deletions=d,
        insertions=ins,
        substitutions_itn=s_i,
        deletions_itn=d_i,
        insertions_itn=ins_i,
    )
