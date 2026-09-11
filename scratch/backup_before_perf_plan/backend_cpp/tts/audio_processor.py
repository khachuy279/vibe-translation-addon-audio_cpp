"""Audio Processing utilities for TTS engines (SRP - Single Responsibility Principle).

Handles tensor-to-numpy conversion, audio normalization, time-stretching (speed),
and WAV Base64 encoding.
"""

import base64
import io
import logging
from typing import Any, List, Union

import numpy as np
import soundfile as sf
import torch
from scipy import signal

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Provides modular, pure-audio transformation utilities for TTS pipelines."""

    @staticmethod
    def convert_to_numpy(audio_output: Any) -> np.ndarray:
        """Convert OmniVoice/TTS output (list/tuple/ndarray/Tensor) to a 1D float32 numpy array."""
        if audio_output is None:
            return np.zeros((0,), dtype=np.float32)

        if isinstance(audio_output, (list, tuple)):
            items = list(audio_output)
        else:
            items = [audio_output]

        converted: List[np.ndarray] = []
        for item in items:
            if isinstance(item, np.ndarray):
                arr = item.astype(np.float32)
            elif isinstance(item, torch.Tensor):
                arr = item.detach().cpu().numpy().astype(np.float32)
            else:
                logger.debug(f"[AudioProcessor] Skipping unrecognized audio item type: {type(item)}")
                continue

            arr = np.atleast_1d(np.squeeze(arr))
            if arr.ndim > 0 and len(arr) > 0:
                converted.append(arr)

        if not converted:
            return np.zeros((0,), dtype=np.float32)

        return np.concatenate(converted, axis=0)

    @staticmethod
    def apply_time_stretch(
        audio: np.ndarray,
        speed: float,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop: int = 256,
    ) -> np.ndarray:
        """Adjust playback speed (0.5x - 2.0x) via Phase Vocoder without altering pitch."""
        if audio is None or len(audio) == 0:
            return np.zeros((0,), dtype=np.float32)

        rate = float(speed)
        # Skip stretching if speed is effectively 1.0 or invalid
        if abs(rate - 1.0) < 0.02 or rate <= 0:
            return audio

        # Bound rate to reasonable range to preserve audio intelligibility
        rate = max(0.5, min(2.0, rate))

        # Check minimum length for STFT
        if len(audio) < n_fft * 2:
            return audio

        try:
            f, t, spec = signal.stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
            time_steps = np.arange(0, spec.shape[1] - 1, rate)
            if len(time_steps) == 0:
                return audio

            int_t = time_steps.astype(int)
            frac_t = time_steps - int_t
            spec_stretched = (1 - frac_t) * spec[:, int_t] + frac_t * spec[:, int_t + 1]

            phase = np.angle(spec[:, 0])
            spec_out = np.zeros_like(spec_stretched, dtype=complex)
            spec_out[:, 0] = spec_stretched[:, 0]
            expected_phase_advance = 2 * np.pi * hop * np.arange(spec.shape[0]) / n_fft

            for i in range(1, spec_stretched.shape[1]):
                dphase = np.angle(spec_stretched[:, i]) - np.angle(spec_stretched[:, i - 1])
                dphase = dphase - expected_phase_advance * rate
                dphase = dphase - 2 * np.pi * np.round(dphase / (2 * np.pi))
                phase = phase + expected_phase_advance * rate + dphase
                spec_out[:, i] = np.abs(spec_stretched[:, i]) * np.exp(1j * phase)

            _, stretched = signal.istft(spec_out, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
            return stretched.astype(np.float32)
        except Exception as e:
            logger.warning(f"[AudioProcessor] Time stretch failed ({e}), using original audio.")
            return audio

    @staticmethod
    def normalize_audio(
        audio: np.ndarray,
        volume: float = 1.0,
        target_peak: float = 0.95,
    ) -> np.ndarray:
        """Apply peak normalization and volume scaling while preventing clipping."""
        if audio is None or len(audio) == 0:
            return np.zeros((0,), dtype=np.float32)

        max_peak = float(np.max(np.abs(audio)))
        if max_peak > 1e-4:
            vol = float(volume if volume is not None else 1.0)
            target = min(0.98, target_peak * vol)
            normalized = (audio / max_peak) * target
            return np.clip(normalized, -0.99, 0.99).astype(np.float32)
        return audio

    @staticmethod
    def encode_wav_to_base64(audio: np.ndarray, sample_rate: int) -> str:
        """Encode float32 audio array to 16-bit PCM WAV base64 string."""
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return ""

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = wav_buffer.getvalue()
        return base64.b64encode(wav_bytes).decode("ascii")
