"""Test tầng A cho bộ chấm WER/CER của harness `test_09_wer_ab.py`.

Chấm điểm phải ĐÚNG thì kết luận A/B mới đáng tin, nên bộ chấm được test riêng:
- Chuẩn hoá: bỏ span metadata, bỏ nhãn `Speaker N:`, lowercase, bỏ dấu câu.
- Chọn thang đo: CJK → CER theo ký tự; Latin → WER theo từ.
- Levenshtein: đếm đúng substitution/deletion/insertion trên các ca đã biết.
- Bất biến: hyp rỗng ⇒ error_rate = 1.0; hyp == ref ⇒ 0.0.
"""

import pytest

from backend.tests.test_09_wer_ab import (
    is_cjk,
    levenshtein_counts,
    normalize_text,
    score_pair,
    tokens_for,
)


# ------------------------------------------------------------------- chuẩn hoá
def test_normalize_strips_metadata_and_speaker_labels():
    """`00_ingress_stream.txt` có dạng `[00:08] Speaker 1: ...` — phải bị loại bỏ."""
    raw = "[00:08] Speaker 1: Hello, world! [laughter]"
    assert normalize_text(raw) == "hello world"


def test_normalize_is_case_and_punct_insensitive():
    assert normalize_text("Hello,   WORLD!!") == "hello world"
    assert normalize_text("  What's up?  ") == "what s up"


def test_normalize_handles_none_and_empty():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


# ----------------------------------------------------------------- thang đo
def test_cjk_detection_and_tokens():
    assert is_cjk("今天天气很好") is True
    assert is_cjk("日本語のテキスト") is True
    assert is_cjk("hello world") is False

    # CJK: chấm theo KÝ TỰ (bỏ khoảng trắng)
    assert tokens_for("今天 天气", cjk=True) == ["今", "天", "天", "气"]
    # Latin: chấm theo TỪ
    assert tokens_for("hello world", cjk=False) == ["hello", "world"]


# ------------------------------------------------------------- levenshtein
def test_levenshtein_identical():
    assert levenshtein_counts(["a", "b", "c"], ["a", "b", "c"]) == (0, 0, 0)


def test_levenshtein_substitution():
    assert levenshtein_counts(["a", "b", "c"], ["a", "x", "c"]) == (1, 0, 0)


def test_levenshtein_deletion():
    assert levenshtein_counts(["a", "b", "c"], ["a", "c"]) == (0, 1, 0)


def test_levenshtein_insertion():
    assert levenshtein_counts(["a", "b"], ["a", "x", "b"]) == (0, 0, 1)


def test_levenshtein_empty_sides():
    assert levenshtein_counts([], ["a", "b"]) == (0, 0, 2)
    assert levenshtein_counts(["a", "b"], []) == (0, 2, 0)
    assert levenshtein_counts([], []) == (0, 0, 0)


def test_levenshtein_known_sentence_case():
    """Ca đã biết: 2 từ sai trong 6 từ."""
    ref = "the quick brown fox jumps over".split()
    hyp = "the quik brown fox jump over".split()
    sub, dele, ins = levenshtein_counts(ref, hyp)
    assert sub + dele + ins == 2, f"nhận được sub={sub} del={dele} ins={ins}"


# ------------------------------------------------------------------- chấm điểm
def test_score_identical_is_zero():
    s = score_pair("Hello world", "hello, WORLD!")
    assert s["error_rate"] == pytest.approx(0.0)
    assert s["metric"] == "wer"
    assert s["empty_hypothesis"] is False


def test_score_empty_hypothesis_is_one():
    s = score_pair("hello world", "")
    assert s["error_rate"] == pytest.approx(1.0)
    assert s["empty_hypothesis"] is True


def test_score_half_wrong():
    s = score_pair("a b c d", "a b x y")
    assert s["error_rate"] == pytest.approx(0.5)
    assert s["substitutions"] == 2


def test_score_uses_cer_for_cjk():
    s = score_pair("今天天气很好", "今天天气不错")
    assert s["metric"] == "cer"
    # 6 ký tự, 2 ký tự sai => 2/6
    assert s["error_rate"] == pytest.approx(2 / 6, abs=1e-6)


def test_score_ground_truth_metadata_does_not_inflate_error():
    """Nhãn speaker/timestamp không được tính là lỗi."""
    s = score_pair("[00:08] Speaker 1: hello world", "hello world")
    assert s["error_rate"] == pytest.approx(0.0)


def test_score_is_symmetric_denominator_on_ref():
    """Mẫu số luôn là số đơn vị của REF (đúng định nghĩa WER)."""
    s = score_pair("a b c d e", "a b")
    assert s["ref_units"] == 5
    assert s["error_rate"] == pytest.approx(3 / 5)
