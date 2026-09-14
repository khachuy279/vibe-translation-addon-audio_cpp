"""Speech Normalizer - Dynamic RMS Normalization & Peak Limiter for ASR Input.

Implements VAD-aware speech RMS estimation, softer soft-knee transition curve,
and true-peak limiter protection to optimize audio level before ASR recognition.
"""

from dataclasses import dataclass
import logging
from typing import Optional, Union
import numpy as np

logger = logging.getLogger("backend_audio_cpp.asr.normalizer")

VAD_STATE_NON_SPEECH: int = 0
VAD_STATE_SPEECH: int = 1
VAD_STATE_PRE_ROLL: int = 2


@dataclass(slots=True)
class NormalizationResult:
    """Telemetry and output result of speech normalization."""
    pcm: np.ndarray
    speech_rms: float
    desired_gain: float
    smoothed_gain: float
    peak_before: float
    peak_scale: float
    peak_after: float


class SpeechNormalizer:
    """Speech normalizer with soft-knee transition and peak limiter."""

    def __init__(
        self,
        target_rms: Optional[float] = None,
        target_peak: Optional[float] = None,
        max_gain: Optional[float] = None,
        min_gain: Optional[float] = None,
        knee_start: Optional[float] = None,
        knee_end: Optional[float] = None,
        gain_smoothing: Optional[bool] = None,
        attack_alpha: Optional[float] = None,
        release_alpha: Optional[float] = None,
        frame_samples: int = 400,  # 25ms @ 16kHz
        min_speech_frames: int = 2,
    ):
        self.target_rms: float = target_rms if target_rms is not None else 0.10
        self.target_peak: float = target_peak if target_peak is not None else 0.95
        self.max_gain: float = max_gain if max_gain is not None else 3.0
        self.min_gain: float = min_gain if min_gain is not None else (1.0 / max(self.max_gain, 1.0))
        self.knee_start: float = knee_start if knee_start is not None else 0.025
        self.knee_end: float = knee_end if knee_end is not None else 0.050
        self.gain_smoothing: bool = gain_smoothing if gain_smoothing is not None else False
        self.attack_alpha: float = attack_alpha if attack_alpha is not None else 0.15
        self.release_alpha: float = release_alpha if release_alpha is not None else 0.35
        self.frame_samples: int = frame_samples
        self.min_speech_frames: int = min_speech_frames

        self.current_gain: float = 1.0

    def reset_gain(self, initial_gain: float = 1.0) -> None:
        """Reset gain state."""
        self.current_gain = float(max(self.min_gain, min(self.max_gain, initial_gain)))

    @staticmethod
    def sanitize_input(pcm: np.ndarray) -> np.ndarray:
        """Copy, sanitize NaN/Inf, and clamp [-1.0, 1.0]."""
        x = np.asarray(pcm, dtype=np.float32).copy()
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(x, -1.0, 1.0)

    def estimate_speech_rms(
        self,
        pcm: np.ndarray,
        frame_state: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate robust speech RMS, trimming silence and noise spikes."""
        n_samples = len(pcm)
        if n_samples == 0:
            return 0.0

        n_frames = n_samples // self.frame_samples
        if n_frames == 0:
            return float(np.sqrt(np.mean(pcm ** 2) + 1e-12))

        frames = pcm[: n_frames * self.frame_samples].reshape(n_frames, self.frame_samples)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

        # 1. Direct VAD-aware speech estimation
        if frame_state is not None and len(frame_state) > 0:
            aligned_states = frame_state[:n_frames]
            speech_idx = np.where(aligned_states == VAD_STATE_SPEECH)[0]

            if len(speech_idx) >= self.min_speech_frames:
                speech_frame_rms = frame_rms[speech_idx]
                if len(speech_frame_rms) >= 5:
                    p_low = np.percentile(speech_frame_rms, 10)
                    p_high = np.percentile(speech_frame_rms, 95)
                    trimmed = speech_frame_rms[(speech_frame_rms >= p_low) & (speech_frame_rms <= p_high)]
                    if len(trimmed) > 0:
                        return float(np.sqrt(np.mean(trimmed ** 2)))
                return float(np.sqrt(np.mean(speech_frame_rms ** 2)))

        # 2. Robust trimmed frame fallback
        if n_frames >= 4:
            p25 = np.percentile(frame_rms, 25)
            p95 = np.percentile(frame_rms, 95)
            robust = frame_rms[(frame_rms >= p25) & (frame_rms <= p95)]
            if len(robust) > 0:
                return float(np.sqrt(np.mean(robust ** 2)))

        return float(np.sqrt(np.mean(pcm ** 2) + 1e-12))

    def compute_desired_gain(self, speech_rms: float) -> float:
        """Compute target gain via smootherstep soft-knee curve."""
        if speech_rms < 1e-4:
            return 1.0

        if speech_rms <= self.knee_start:
            return float(min(self.target_rms / max(speech_rms, 1e-4), self.max_gain))

        if speech_rms >= self.knee_end:
            return 1.0

        # Smootherstep polynomial interpolation
        full_boost = float(min(self.target_rms / speech_rms, self.max_gain))
        t = (speech_rms - self.knee_start) / (self.knee_end - self.knee_start)
        t = max(0.0, min(1.0, t))
        w = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

        return float(full_boost * (1.0 - w) + 1.0 * w)

    def smooth_gain(self, desired_gain: float) -> float:
        """Apply asymmetric EMA gain smoothing."""
        if not self.gain_smoothing:
            self.current_gain = float(max(self.min_gain, min(self.max_gain, desired_gain)))
            return self.current_gain

        if desired_gain > self.current_gain:
            alpha = self.attack_alpha
        else:
            alpha = self.release_alpha

        self.current_gain = (1.0 - alpha) * self.current_gain + alpha * desired_gain
        self.current_gain = float(max(self.min_gain, min(self.max_gain, self.current_gain)))
        return self.current_gain

    def process(
        self,
        pcm: np.ndarray,
        frame_state: Optional[np.ndarray] = None,
        use_smoothing: bool = False,
    ) -> NormalizationResult:
        """Execute complete speech normalization pipeline."""
        if pcm is None or len(pcm) == 0:
            empty = np.array([], dtype=np.float32)
            return NormalizationResult(
                pcm=empty, speech_rms=0.0, desired_gain=1.0, smoothed_gain=1.0,
                peak_before=0.0, peak_scale=1.0, peak_after=0.0,
            )

        # 1. Sanitize input
        x = self.sanitize_input(pcm)
        peak_before = float(np.max(np.abs(x)))

        # 2. VAD-aware speech RMS
        speech_rms = self.estimate_speech_rms(x, frame_state=frame_state)

        # 3. Soft-knee desired gain
        desired_gain = self.compute_desired_gain(speech_rms)

        # 4. Asymmetric EMA smoothing (or stateless)
        if use_smoothing and self.gain_smoothing:
            applied_gain = self.smooth_gain(desired_gain)
        else:
            applied_gain = float(max(self.min_gain, min(self.max_gain, desired_gain)))

        # 5. Apply gain
        x *= applied_gain

        # 6. Peak protection via global attenuation
        peak_after_gain = float(np.max(np.abs(x)))
        peak_scale = 1.0
        if peak_after_gain > self.target_peak:
            peak_scale = float(self.target_peak / peak_after_gain)
            x *= peak_scale

        # 7. Clamp output
        x = np.clip(x, -1.0, 1.0)
        peak_after = float(np.max(np.abs(x)))

        return NormalizationResult(
            pcm=x,
            speech_rms=speech_rms,
            desired_gain=desired_gain,
            smoothed_gain=applied_gain,
            peak_before=peak_before,
            peak_scale=peak_scale,
            peak_after=peak_after,
        )


def normalize_speech(
    pcm: np.ndarray,
    target_rms: Optional[float] = None,
    target_peak: Optional[float] = None,
    max_gain: Optional[float] = None,
    min_rms_to_boost: Optional[float] = None,
    min_rms: float = 1e-4,
    is_commit: bool = False,
    utt_id: Optional[str] = None,
    frame_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dynamic RMS normalization + Peak Limiter optimized for ASR input.

    Stateless wrapper around SpeechNormalizer for backward compatibility and testing.
    """
    if pcm is None or len(pcm) == 0:
        return np.array([], dtype=np.float32)

    knee_end = min_rms_to_boost if min_rms_to_boost is not None else 0.050
    knee_start = min(knee_end * 0.5, 0.025)

    normalizer = SpeechNormalizer(
        target_rms=target_rms,
        target_peak=target_peak,
        max_gain=max_gain,
        knee_start=knee_start,
        knee_end=knee_end,
        gain_smoothing=False,
    )
    res = normalizer.process(pcm, frame_state=frame_state, use_smoothing=False)
    return res.pcm


def normalize_pcm_bytes(
    pcm_bytes: bytes,
    target_rms: Optional[float] = None,
    target_peak: Optional[float] = None,
    max_gain: Optional[float] = None,
    min_rms_to_boost: Optional[float] = None,
) -> bytes:
    """Helper converting int16 PCM bytes -> normalized float32 -> int16 PCM bytes."""
    if not pcm_bytes:
        return b""
    pcm_float = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    norm_float = normalize_speech(
        pcm_float,
        target_rms=target_rms,
        target_peak=target_peak,
        max_gain=max_gain,
        min_rms_to_boost=min_rms_to_boost,
    )
    norm_int16 = np.clip(norm_float * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return norm_int16.tobytes()
