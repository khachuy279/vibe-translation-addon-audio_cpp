"""Guards for the W2.1 baseline measurement gate.

The audit marked every performance number as `NEEDS BENCHMARK`, so the baseline suite is
the evidence that Phase 2 work is judged against. These tests protect the *measurement
harness* itself, because several ways of getting it wrong all look like a valid result:

* the perf collector is off by default, so a run can produce an all-zero baseline;
* the synthetic test signal is not detected as speech by FireRed VAD, and even when it
  commits the ASR returns an empty transcript -- both silently yield "nothing happened";
* a scenario can produce no inferences at all and still write a beautifully formatted,
  completely meaningless JSON document.

Each test below pins one of those failure modes.
"""

from pathlib import Path

import pytest

from backend_cpp.tests import perf_baseline as pb

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Validity reporting: an empty run must be flagged, not silently accepted
# ---------------------------------------------------------------------------
def test_validity_flags_run_without_inferences():
    result = pb._validity(0)
    assert result["valid"] is False
    assert "VAD/signal mismatch" in result["reason"]


def test_validity_accepts_run_with_inferences():
    result = pb._validity(42)
    assert result["valid"] is True
    assert result["inferences_observed"] == 42


def test_validity_separates_empty_transcript_from_vad_mismatch():
    """Inferences ran but nothing was delivered -> a DISTINCT, accurate diagnosis."""
    result = pb._validity(14, delivered=0)
    assert result["valid"] is True, "cost metrics are still usable"
    assert result["delivery_measurable"] is False
    assert "empty text" in result["delivery_note"]
    assert "VAD" not in result["delivery_note"]


def test_validity_marks_delivery_measurable_when_messages_arrive():
    result = pb._validity(47, delivered=14)
    assert result["delivery_measurable"] is True
    assert result["delivered_to_client"] == 14


# ---------------------------------------------------------------------------
# Signal/engine contract
# ---------------------------------------------------------------------------
def test_synthetic_signal_requires_the_engine_that_detects_it():
    """Scenarios must request the VAD engine that actually fires on the test signal.

    If this changes (e.g. the generator is replaced with real speech), update
    BASELINE_VAD_ENGINE and the reasoning in perf_baseline.py rather than just the value.
    """
    assert pb.BASELINE_VAD_ENGINE == "fsmn-vad"
    config = pb._baseline_config(ttsEnabled=True)
    assert config["vadEngine"] == pb.BASELINE_VAD_ENGINE
    assert config["ttsEnabled"] is True


def test_vad_benchmark_reports_per_engine_cost_and_speech_frames():
    """The VAD row must expose ms/audio-sec AND how many frames reached ASR."""
    result = pb.benchmark_vad_cost(engines=("fsmn-vad",), duration_sec=0.5, repeats=1)

    assert result["benchmark"] == "vad_cost"
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    for key in ("engine", "ms_per_audio_sec", "rtf", "speech_frames_passed_to_asr", "native_frame_samples"):
        assert key in row, f"VAD row must report {key}"
    assert row["engine"] == "fsmn-vad"
    assert row["ms_per_audio_sec"] > 0
    # fsmn-vad is the engine that detects the synthetic signal; 0 would mean the
    # per-speech-frame bytes() copy path was never exercised.
    assert row["speech_frames_passed_to_asr"] > 0


# ---------------------------------------------------------------------------
# Micro-benchmark shape
# ---------------------------------------------------------------------------
def test_snapshot_normalize_benchmark_shape():
    result = pb.benchmark_snapshot_and_normalize(durations_sec=(1, 2), repeats=1)

    assert result["benchmark"] == "snapshot_and_normalize"
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert row["pre_infer_ms"] == pytest.approx(
            row["snapshot"]["median_ms"] + row["normalize"]["median_ms"], abs=0.01
        )
        assert row["snapshot"]["peak_alloc_bytes"] > 0
    scaling = result["scaling"]
    assert "alloc_scales_with_duration" in scaling
    assert "verdict" in scaling


def test_copy_chain_marks_the_reference_row():
    """The NumPy per-frame row is a reference measurement, not a production call.

    Labelling it is what keeps it from being quoted as if vad_processor.py cost that much.
    """
    result = pb.benchmark_copy_chain(duration_sec=0.5, repeats=1)
    reference = [s for s in result["stages"] if s.get("is_reference_only")]
    production = [s for s in result["stages"] if not s.get("is_reference_only")]
    assert len(reference) == 1
    assert "REFERENCE" in reference[0]["stage"]
    assert len(production) >= 3
    assert "tracemalloc" in result["note"]


# ---------------------------------------------------------------------------
# Raw-sample accessor: the correlation contract the baseline depends on
# ---------------------------------------------------------------------------
def test_perf_get_samples_is_index_aligned():
    """asr.infer_ms and asr.audio_dur_sec must be zippable, or the bucketing lies."""
    from backend_cpp.config import config
    from backend_cpp.utils.perf_profiler import perf

    original = config.perf.enabled
    config.perf.enabled = True
    try:
        perf.reset()
        for index in range(5):
            perf.record_metric("asr", "infer_ms", 100.0 + index)
            perf.record_metric("asr", "audio_dur_sec", float(index))

        infer = perf.get_samples("asr", "infer_ms")
        durations = perf.get_samples("asr", "audio_dur_sec")
        assert len(infer) == len(durations) == 5
        assert infer[0] == 100.0
        assert durations[-1] == 4.0
        # A copy, not the live deque: benchmarks must not mutate collector state.
        infer.append(999.0)
        assert len(perf.get_samples("asr", "infer_ms")) == 5
    finally:
        perf.reset()
        config.perf.enabled = original


def test_vad_engine_tradeoff_reports_cost_AND_coverage():
    """Picking a VAD engine needs both sides: cost alone can change subtitle behaviour.

    A cheaper engine that trims or misses speech is not a win, so the benchmark must report
    detection coverage next to the timing or the comparison is unusable.
    """
    result = pb.benchmark_vad_engine_tradeoff(max_sec=3.0, repeats=1)

    assert result["benchmark"] == "vad_engine_tradeoff"
    available = [r for r in result["rows"] if r.get("available")]
    assert available, "no VAD engine was available for the comparison"

    for row in available:
        assert row["ms_per_audio_sec"] > 0
        assert row["rtf"] > 0
        assert "speech_ratio" in row, "coverage must be reported next to cost"
        assert "segments" in row
        assert "cost_multiple_vs_cheapest" in row
        assert "speech_ratio_delta_vs_cheapest" in row

    cheapest = min(available, key=lambda r: r["ms_per_audio_sec"])
    assert result["cheapest_engine"] == cheapest["engine"]
    assert cheapest["cost_multiple_vs_cheapest"] == 1.0


def test_vad_engine_tradeoff_records_unknown_engine_names():
    """An unknown engine name must not silently pick the most expensive engine.

    `VADEngineFactory.get_engine()` falls back when the name is not in
    ``DEFAULT_THRESHOLDS``. It used to fall back to ``firered-vad`` -- the most expensive
    engine (~8x silero) -- silently, so a typo in the VAD config quietly cost CPU. It now
    falls back to the default (``fsmn-vad``) and logs a warning, and this test keeps the
    measurement honest either way: the row must exist and carry real cost so an unknown name
    can never be mistaken for a supported one in the baseline.
    """
    result = pb.benchmark_vad_engine_tradeoff(
        max_sec=1.0, engines=("silero-vad", "definitely-not-an-engine"), repeats=1
    )

    by_engine = {row["engine"]: row for row in result["rows"]}
    assert "definitely-not-an-engine" in by_engine
    assert by_engine["definitely-not-an-engine"]["available"] is True, (
        "the factory falls back rather than raising, so the row is measurable"
    )
    assert by_engine["definitely-not-an-engine"]["ms_per_audio_sec"] > 0

    # The fallback must land on the current default, NOT on whichever engine happens to be
    # the most expensive one.
    from backend_cpp.config import config as app_config
    from backend_cpp.vad.engines import SUPPORTED_VAD_ENGINES

    assert "definitely-not-an-engine" not in SUPPORTED_VAD_ENGINES
    assert app_config.vad.vad_engine in SUPPORTED_VAD_ENGINES
    assert app_config.vad.vad_engine == "fsmn-vad", (
        "the default must stay the measured-cheapest engine with identical transcripts; "
        "re-run scratch/compare_vad_quality.py before changing it"
    )


# ---------------------------------------------------------------------------
# Reference audio for the real-audio scenario and the W2.2 equivalence gate
# ---------------------------------------------------------------------------
def test_reference_audio_loads_as_16k_mono_pcm():
    from backend_cpp.tests.perf_real_audio import DEFAULT_REFERENCE_AUDIO, load_wav_pcm16

    path = PROJECT_ROOT / DEFAULT_REFERENCE_AUDIO
    assert path.exists(), f"reference audio missing: {path}"

    pcm = load_wav_pcm16(path, max_sec=1.0)
    assert len(pcm) == 16000 * 2, "1 second of 16kHz mono 16-bit PCM"


def test_reference_audio_rejects_unsupported_format():
    """Wrong format must fail loudly with a fix, not be silently resampled."""
    from backend_cpp.tests.perf_real_audio import load_wav_pcm16

    wrong = PROJECT_ROOT / "wav_test" / "OSR_us_000_0010_8k.wav"
    if not wrong.exists():
        pytest.skip("8kHz fixture not present")

    with pytest.raises(ValueError) as excinfo:
        load_wav_pcm16(wrong)
    assert "ffmpeg" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Delivery counting: the BY-DESIGN duplicate final must not look like a defect
# ---------------------------------------------------------------------------
def test_delivery_counting_collapses_backward_compat_final_resend():
    """`_process_translation_item()` re-sends a final `utterance_update` on purpose.

    Counting *messages* reports twice the true sentence count and reads exactly like a
    duplicate-transcript bug. That misreading actually happened during W2.1 and sent the
    investigation into the ASR emission path, which was innocent. Count unique
    `utterance_id` instead, while keeping the raw message count visible.
    """
    from backend_cpp.tests.perf_real_audio import count_delivery

    messages = [
        # 1. the ASR final
        {
            "type": "utterance_update",
            "is_final": True,
            "utterance_id": "aaaa",
            "text": "hello",
            "translated": "...",
        },
        # 2. "for complete backward compatibility" resend for the SAME utterance
        {
            "type": "utterance_update",
            "is_final": True,
            "utterance_id": "aaaa",
            "text": "hello",
            "translated": "xin chào",
        },
        {"type": "translation", "utterance_id": "aaaa", "translated": "xin chào"},
    ]

    delivery = count_delivery(messages)

    assert delivery["final_utterances"] == 1, "one sentence, not two"
    assert delivery["final_messages"] == 2, "the raw count must stay visible"
    assert delivery["translations"] == 1, "translations are not duplicated"
    assert len(delivery["unique_finals"]) == 1


def test_delivery_counting_separates_previews_finals_translations_and_tts():
    from backend_cpp.tests.perf_real_audio import count_delivery

    messages = [
        {"type": "utterance_update", "is_final": False, "utterance_id": "aaaa", "text": "hel"},
        {"type": "utterance_update", "is_final": True, "utterance_id": "aaaa", "text": "hello"},
        {"type": "translation", "utterance_id": "aaaa", "translated": "xin chào"},
        {"type": "tts_audio", "utterance_id": "aaaa", "audio": "..."},
    ]

    delivery = count_delivery(messages)

    assert delivery["preview_updates"] == 1
    assert delivery["final_utterances"] == 1
    assert delivery["translations"] == 1
    assert delivery["tts_audio_messages"] == 1


def test_delivery_counting_keeps_distinct_utterances_distinct():
    """The collapse must not merge genuinely different sentences."""
    from backend_cpp.tests.perf_real_audio import count_delivery

    messages = [
        {"type": "utterance_update", "is_final": True, "utterance_id": uid, "text": text}
        for uid, text in (("aaaa", "one"), ("bbbb", "two"), ("cccc", "three"))
    ]

    delivery = count_delivery(messages)

    assert delivery["final_utterances"] == 3
    assert delivery["final_messages"] == 3


# ---------------------------------------------------------------------------
# Baseline document shape
# ---------------------------------------------------------------------------
def test_build_baseline_document_shape():
    document = pb.build_baseline(micro={"x": 1}, scenarios={"y": 2})
    for key in (
        "schema_version",
        "kind",
        "work_item",
        "generated_at",
        "environment",
        "micro_benchmarks",
        "end_to_end_scenarios",
    ):
        assert key in document, f"baseline document must contain {key}"
    assert document["work_item"] == "W2.1"
    assert document["micro_benchmarks"] == {"x": 1}
    assert document["end_to_end_scenarios"] == {"y": 2}


def test_environment_is_reproducible_metadata():
    env = pb.collect_environment()
    for key in ("python", "cpu_count", "asr_model", "vad_engine", "poll_interval_ms", "max_sessions"):
        assert key in env, f"environment block must record {key}"
