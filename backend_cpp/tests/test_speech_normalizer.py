"""Comprehensive test suite for SpeechNormalizer v2 and AudioBufferManager VAD metadata.

Validates the 8 core requirements:
1. Hard-cliff elimination via smootherstep soft-knee (continuity around knee).
2. BGM resistance: VAD-aware speech RMS ignores loud background music.
3. Anti-pumping: Stateful asymmetric EMA smooths out alternating chunk energies.
4. Sudden loud speech: Fast release attenuation + peak protection (peak <= 0.95).
5. Silence immunity: Never boosts low-energy silence/noise floor.
6. Utterance continuity: Preserves gain state across consecutive utterances in a session.
7. Late-onset VAD pre-roll: Robustly recovers speech energy from active pre-roll frames.
8. Buffer overflow metadata sync: Dropping oldest bytes maintains frame-grid alignment.
"""

import numpy as np
import pytest

from backend_cpp.asr.audio_buffer import (
    AudioBufferManager,
    VAD_STATE_NON_SPEECH,
    VAD_STATE_SPEECH,
    VAD_STATE_PRE_ROLL,
)
from backend_cpp.asr.speech_normalizer import SpeechNormalizer, NormalizationResult


def _make_tone(freq: float, duration_sec: float, rms: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate pure sine tone with calibrated RMS amplitude."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    # Sine wave RMS = peak / sqrt(2) -> peak = rms * sqrt(2)
    tone = (rms * np.sqrt(2.0) * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    return np.clip(tone, -1.0, 1.0)


def test_hard_cliff_smootherstep():
    """Test 1: Quanh ngưỡng knee 0.025 - 0.050 gain biến thiên mượt mà, không có cú nhảy 7dB."""
    normalizer = SpeechNormalizer(
        target_rms=0.10,
        max_gain=3.0,
        knee_start=0.025,
        knee_end=0.050,
        gain_smoothing=False,
    )

    rms_values = [0.038, 0.039, 0.040, 0.041, 0.042]
    gains = [normalizer.compute_desired_gain(r) for r in rms_values]

    # Verify gains decrease monotonically and smoothly
    for i in range(len(gains) - 1):
        delta = gains[i] - gains[i + 1]
        assert delta > 0.0, f"Gain should decrease smoothly as RMS increases: {gains}"
        assert delta < 0.15, f"Discontinuity detected! delta={delta} exceeds smooth step limit"


def test_bgm_resistance_with_vad():
    """Test 2: Speech (0.05 RMS) trộn với BGM (0.12 RMS). Nhờ VAD mask, speech_rms ≈ 0.05."""
    normalizer = SpeechNormalizer(frame_samples=400)

    # 10 frames total: 5 frames speech (rms=0.05), 5 frames loud BGM (rms=0.12)
    speech_pcm = _make_tone(440.0, 0.125, 0.05)  # 2000 samples = 5 frames
    bgm_pcm = _make_tone(100.0, 0.125, 0.12)     # 2000 samples = 5 frames
    combined_pcm = np.concatenate([bgm_pcm, speech_pcm])  # 4000 samples = 10 frames

    # Frame state: 5 frames NON_SPEECH (BGM), 5 frames SPEECH
    frame_state = np.array([VAD_STATE_NON_SPEECH] * 5 + [VAD_STATE_SPEECH] * 5, dtype=np.uint8)

    est_rms = normalizer.estimate_speech_rms(combined_pcm, frame_state=frame_state)

    # Without VAD mask, whole RMS would be ~0.092. With VAD mask, it must isolate ~0.05!
    whole_rms = float(np.sqrt(np.mean(combined_pcm ** 2)))
    assert whole_rms > 0.085
    assert est_rms == pytest.approx(0.05, abs=0.005), f"VAD-aware RMS {est_rms} should isolate speech (~0.05)"


def test_anti_pumping_ema():
    """Test 3: Dao động năng lượng luân phiên (0.02 -> 0.05 -> 0.02) không làm gain giật cục."""
    normalizer = SpeechNormalizer(
        target_rms=0.10,
        max_gain=3.0,
        knee_start=0.025,
        knee_end=0.050,
        gain_smoothing=True,
        attack_alpha=0.15,
        release_alpha=0.35,
    )

    chunk_low = _make_tone(300.0, 0.1, 0.02)  # wants full boost (3.0)
    chunk_high = _make_tone(300.0, 0.1, 0.06)  # wants unity gain (1.0)

    gains = []
    for _ in range(5):
        r1 = normalizer.process(chunk_low, use_smoothing=True)
        gains.append(r1.smoothed_gain)
        r2 = normalizer.process(chunk_high, use_smoothing=True)
        gains.append(r2.smoothed_gain)

    # Verify gain does not oscillate wildly between 1.0 and 3.0 on consecutive chunks
    for i in range(len(gains) - 1):
        step_diff = abs(gains[i + 1] - gains[i])
        assert step_diff < 0.8, f"Pumping detected: gain jumped {step_diff:.2f} between chunks"


def test_sudden_loud_speech_and_peak_protection():
    """Test 4: Âm thanh to đột ngột (0.35 RMS) kích hoạt fast release và peak limiter <= 0.95."""
    normalizer = SpeechNormalizer(
        target_rms=0.10,
        target_peak=0.95,
        max_gain=3.0,
        release_alpha=0.35,
    )

    # Start with quiet speech -> gain increases
    quiet = _make_tone(400.0, 0.2, 0.03)
    normalizer.process(quiet)
    normalizer.process(quiet)
    gain_before = normalizer.current_gain
    assert gain_before > 1.2

    # Sudden loud burst (peak would clip beyond 1.0 without protection)
    loud = _make_tone(400.0, 0.2, 0.35)
    res = normalizer.process(loud)

    # 1. Gain must attenuate rapidly
    assert normalizer.current_gain < gain_before
    # 2. Output peak must not exceed target_peak
    assert res.peak_after <= 0.9501
    assert np.max(np.abs(res.pcm)) <= 0.9501


def test_silence_immunity():
    """Test 5: Silence hoặc năng lượng cực thấp (< 1e-4) không bị boost (gain = 1.0)."""
    normalizer = SpeechNormalizer(target_rms=0.10, max_gain=3.0)
    silence = np.zeros(1600, dtype=np.float32)

    res = normalizer.process(silence, use_smoothing=False)
    assert res.desired_gain == 1.0
    assert np.all(res.pcm == 0.0)


def test_gain_continuity_across_utterances():
    """Test 6: Giữ gain state xuyên suốt các câu trong session (không reset về 1.0 mỗi câu)."""
    normalizer = SpeechNormalizer(target_rms=0.10, max_gain=3.0, attack_alpha=0.2)

    # Utterance 1: quiet speaker (0.03 RMS) -> builds up gain
    utt1 = _make_tone(500.0, 0.3, 0.03)
    normalizer.process(utt1)
    normalizer.process(utt1)
    normalizer.process(utt1)
    gain_after_utt1 = normalizer.current_gain
    assert gain_after_utt1 > 1.5

    # Inter-sentence pause (speech ended, next utterance arrives)
    # Gain state must be preserved into utterance 2
    utt2 = _make_tone(500.0, 0.3, 0.03)
    res2 = normalizer.process(utt2)
    # The applied gain for utterance 2 must start from previous level, not 1.0
    assert res2.smoothed_gain > 1.5


def test_late_vad_pre_roll():
    """Test 7: Speech bắt đầu trong pre-roll buffer trước khi VAD START phát hiện."""
    normalizer = SpeechNormalizer(frame_samples=400, min_speech_frames=2)

    # 4 frames PRE_ROLL with speech energy (0.06 RMS), 0 frames marked SPEECH yet
    pre_speech = _make_tone(440.0, 0.1, 0.06)  # 1600 samples = 4 frames
    frame_state = np.array([VAD_STATE_PRE_ROLL] * 4, dtype=np.uint8)

    est_rms = normalizer.estimate_speech_rms(pre_speech, frame_state=frame_state)
    assert est_rms == pytest.approx(0.06, abs=0.01)


def test_buffer_overflow_metadata_sync():
    """Test 8: AudioBufferManager tràn buffer drop PCM và frame_state đồng bộ."""
    # 0.2s max duration = 3200 samples = 6400 bytes = 8 frames (400 samples each)
    mgr = AudioBufferManager(sample_rate=16000, max_duration_sec=0.2, frame_samples=400)

    # Feed 12 frames (4800 samples = 0.3s) with state SPEECH
    chunk = (np.ones(4800, dtype=np.int16) * 300).tobytes()
    mgr.feed_bytes(chunk, vad_state=VAD_STATE_SPEECH)

    snapshot = mgr.get_snapshot()
    assert snapshot is not None
    # Must be clamped to max duration (whole frame multiples)
    assert snapshot.sample_count <= 3200
    assert snapshot.duration_sec <= 0.201
    # Frame metadata count must exactly match PCM samples // 400
    expected_frames = snapshot.sample_count // 400
    assert len(snapshot.frame_state) == expected_frames
    assert np.all(snapshot.frame_state == VAD_STATE_SPEECH)


def test_nan_inf_sanitization():
    """Defensive test: PCM chứa NaN, Inf, và vượt [-1, 1] được làm sạch an toàn."""
    normalizer = SpeechNormalizer()
    dirty_pcm = np.array([0.05, np.nan, 0.05, np.inf, -np.inf, 2.5, -3.0], dtype=np.float32)

    cleaned = normalizer.sanitize_input(dirty_pcm)
    assert np.all(np.isfinite(cleaned))
    assert np.all(cleaned >= -1.0)
    assert np.all(cleaned <= 1.0)
    assert cleaned[1] == 0.0  # nan -> 0
    assert cleaned[3] == 1.0  # inf -> 1
    assert cleaned[4] == -1.0 # -inf -> -1
    assert cleaned[5] == 1.0  # 2.5 clamped -> 1.0
