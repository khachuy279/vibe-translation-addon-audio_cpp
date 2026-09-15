"""Mock fixtures and helpers for backend tests."""

import numpy as np


def generate_sine_wave(freq_hz: float = 440.0, duration_sec: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    """Tạo sóng sin chuẩn làm mock audio."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def generate_silence(duration_sec: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Tạo đoạn âm thanh im lặng tuyệt đối."""
    return np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
