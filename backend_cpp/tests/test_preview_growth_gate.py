"""Guards for the W2.2 preview growth gate.

Measured motivation (24s of real speech, qwen3-asr-1.7b):

    gate OFF: 31 previews /  7 commits, 54.6s preview audio + 17.9s commit audio
              = 2.27x amplification, 81% of all inferences are previews
    gate 0.5: 13 previews /  7 commits, 16.9s preview audio + 17.8s commit audio
              = 0.70x amplification, -49% total ASR inference time
    final transcripts: IDENTICAL

Only the preview path is gated, so the commit path (and therefore the final transcript) must
be unaffected. These tests pin the gate arithmetic AND the amortisation property, because the
first implementation advanced the reference point on every skipped poll and silently collapsed
to ONE preview per utterance.
"""

from __future__ import annotations

from backend_cpp.asr.transcribe_engine import TranscribeEngine


def _engine(ratio: float = 0.0, floor_ms: int = 0) -> TranscribeEngine:
    engine = TranscribeEngine.__new__(TranscribeEngine)
    engine._preview_min_growth_ratio = ratio
    engine._preview_min_growth_sec = floor_ms / 1000.0
    engine._last_preview_duration_sec = 0.0
    engine._last_polled_samples = 0
    return engine


def test_skipping_a_poll_must_not_advance_the_gate_reference_point():
    """The one-line regression that silently disabled the gate for long utterances."""
    engine = _engine(ratio=0.5)
    engine._note_preview_ran(duration_sec=2.0, snapshot_samples=16000)

    engine._note_preview_skipped(snapshot_samples=22000)

    assert engine._last_preview_duration_sec == 2.0, "reference must stay at the last preview"
    assert engine._last_polled_samples == 22000, "but the samples must be recorded as seen"


def _simulate(engine: TranscribeEngine, total_sec: float, step_sec: float = 0.35) -> int:
    """Walk the utterance forward one poll at a time; return the number of previews run.

    Uses the production state-update helpers (not hand-rolled assignments) so that a change
    to the update discipline -- e.g. advancing the gate reference point on a skipped poll --
    is actually detected by these tests.
    """
    previews = 0
    duration = 0.0
    while duration <= total_sec + 1e-9:
        if engine._should_run_preview(duration):
            engine._note_preview_ran(duration, snapshot_samples=0)
            previews += 1
        else:
            engine._note_preview_skipped(snapshot_samples=0)
        duration = round(duration + step_sec, 6)
    return previews


# ---------------------------------------------------------------------------
# Disabled by default: existing behaviour must be preserved exactly
# ---------------------------------------------------------------------------
def test_gate_disabled_always_allows_a_preview():
    engine = _engine(ratio=0.0, floor_ms=0)
    for duration in (0.0, 0.7, 1.0, 12.0):
        assert engine._should_run_preview(duration) is True


def test_gate_disabled_matches_every_poll_becoming_a_preview():
    engine = _engine(ratio=0.0)
    # 3.0s at 350ms polls -> a preview on every poll (the old behaviour). The poll instants
    # are 0, 0.35, ... 2.80 before exceeding 3.0s, so 9 previews.
    assert _simulate(engine, 3.0) == int(3.0 / 0.35) + 1


# ---------------------------------------------------------------------------
# Gate arithmetic
# ---------------------------------------------------------------------------
def test_first_preview_of_an_utterance_always_runs():
    engine = _engine(ratio=0.9)
    assert engine._should_run_preview(0.6) is True


def test_growth_below_threshold_is_skipped_and_above_is_run():
    engine = _engine(ratio=0.5)
    engine._last_preview_duration_sec = 2.0

    # needed = 0.5 * duration; at 2.35s only 0.35s of new audio arrived (< 1.175)
    assert engine._should_run_preview(2.35) is False
    # at 3.0s: 1.0s of new audio vs required 1.5 -> still short
    assert engine._should_run_preview(3.0) is False
    # at 4.0s: 2.0s of new audio vs required 2.0 -> allowed
    assert engine._should_run_preview(4.0) is True


def test_absolute_floor_applies_even_when_ratio_is_zero():
    engine = _engine(ratio=0.0, floor_ms=500)
    engine._last_preview_duration_sec = 4.0

    assert engine._should_run_preview(4.3) is False, "300ms < 500ms floor"
    assert engine._should_run_preview(4.5) is True, "500ms reached"


def test_floor_wins_when_larger_than_ratio_term():
    engine = _engine(ratio=0.01, floor_ms=1000)
    engine._last_preview_duration_sec = 20.0

    # ratio term = 0.2s, floor = 1.0s -> max() keeps the floor
    assert engine._preview_required_growth_sec(20.0) == 1.0
    assert engine._should_run_preview(20.5) is False, "only 0.5s of new audio"
    assert engine._should_run_preview(21.0) is True, "1.0s of new audio"


def test_ratio_term_wins_when_larger_than_floor():
    engine = _engine(ratio=0.1, floor_ms=1000)
    engine._last_preview_duration_sec = 20.0

    # ratio term = 2.0s > floor 1.0s, and it grows with the utterance
    assert engine._preview_required_growth_sec(20.0) == 2.0
    assert engine._should_run_preview(21.5) is False, "required(21.5) = 2.15s"
    assert engine._should_run_preview(22.3) is True, "required(22.3) = 2.23s <= 2.3s"


# ---------------------------------------------------------------------------
# Amortisation: this is the property the first implementation broke
# ---------------------------------------------------------------------------
def test_gate_does_not_degenerate_to_one_preview_per_utterance():
    """A 3s utterance must still get several previews, not one.

    If the reference point is advanced on skipped polls, the required growth keeps exceeding
    one poll's worth of audio and the gate never reopens -- 3s utterances then produce exactly
    one preview. This test fails loudly in that case.
    """
    engine = _engine(ratio=0.5)
    previews = _simulate(engine, 3.0)

    assert previews >= 3, f"expected amortised previews, got {previews}"
    assert previews < 9, f"gate is not throttling at all, got {previews}"


def test_preview_count_grows_logarithmically_not_linearly():
    """Doubling the utterance must NOT double the preview count."""
    short = _engine(ratio=0.5)
    long = _engine(ratio=0.5)

    short_previews = _simulate(short, 4.0)
    long_previews = _simulate(long, 16.0)

    assert long_previews < 2 * short_previews, (
        f"4x the audio should not give 2x the previews: {short_previews} -> {long_previews}"
    )


def test_higher_ratio_never_produces_more_previews():
    counts = [_simulate(_engine(ratio=r), 6.0) for r in (0.0, 0.25, 0.5, 0.9)]
    assert counts == sorted(counts, reverse=True), f"not monotonic: {counts}"


# ---------------------------------------------------------------------------
# Utterance boundaries must reset the reference point
# ---------------------------------------------------------------------------
def test_preview_cadence_resets_on_speech_start():
    engine = TranscribeEngine.__new__(TranscribeEngine)
    engine._preview_min_growth_ratio = 0.5
    engine._preview_min_growth_sec = 0.0
    engine._last_preview_duration_sec = 7.5

    from unittest.mock import MagicMock, patch

    engine._state_lock = MagicMock()
    engine._segmenter = MagicMock()
    engine._last_partial_text = ""
    engine._last_partial_samples = 0
    engine._current_utterance_id = "old"
    engine._last_committed_head = ""
    engine._last_committed_head_time = 0.0
    engine._audio_buffer_mgr = MagicMock()
    engine._is_speech_active = False

    with patch("backend_cpp.asr.transcribe_engine.dump_vad_utterance_f32"):
        engine.on_speech_start()

    assert engine._last_preview_duration_sec == 0.0
    assert engine._last_polled_samples == 0
