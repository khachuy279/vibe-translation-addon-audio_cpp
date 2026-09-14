"""Base VAD interface and data contracts for backend_audio_cpp."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class VADConfig:
    """Universal configuration for VAD engines (Silero & FSMN)."""
    model_path: str = "backend_audio_cpp/models/fsmn_vad.onnx"
    sample_rate: int = 16000
    frame_samples: int = 512               # 32ms at 16kHz
    threshold: float = 0.50                # Speech detection probability threshold
    min_silence_duration_sec: float = 0.55 # Silence duration to trigger SPEECH_END
    min_speech_duration_sec: float = 0.05  # Minimum speech duration (filters noise spikes)
    max_speech_duration_sec: float = 8.00  # Hard cutoff to prevent runaway sentences
    debug: bool = True                     # Enable structured INFO logging for VAD events


@dataclass
class VADResult:
    """Result of processing a single frame or chunk."""
    timestamp_sec: float
    is_speech: bool
    probability: float
    event: Optional[str] = None            # "SPEECH_START", "SPEECH_END", or None
    reason: Optional[str] = None           # "SILENCE_TIMEOUT", "MAX_SPEECH_DURATION_REACHED", "STREAM_EOF", etc.
    utterance_id: Optional[int] = None
    duration_sec: float = 0.0              # Valid during SPEECH_END
    speech_start_sec: Optional[float] = None
    speech_end_sec: Optional[float] = None


class BaseVADStreamState:
    """Base session-isolated streaming state."""

    def __init__(self, config: VADConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_samples = config.frame_samples
        self.is_speech_active: bool = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.current_utterance_id: int = 0
        self.silence_start_time: Optional[float] = None
        self.last_timestamp_sec: float = 0.0

    def reset(self) -> None:
        self.is_speech_active = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_start_time = None
        self.last_timestamp_sec = 0.0


class BaseVADEngine(ABC):
    """Abstract Base Class for VAD Engines."""

    @abstractmethod
    def create_state(self) -> BaseVADStreamState:
        """Create a fresh isolated session state."""
        raise NotImplementedError

    @abstractmethod
    def process_frame(
        self,
        frame_pcm16: bytes,
        timestamp_sec: float,
        state: BaseVADStreamState,
    ) -> VADResult:
        """Process exactly 512 samples of 16-bit PCM mono."""
        raise NotImplementedError

    @abstractmethod
    def flush(
        self,
        timestamp_sec: float,
        state: BaseVADStreamState,
        auto_reset: bool = True,
    ) -> Optional[VADResult]:
        """Force flush remaining audio buffer at end of audio stream."""
        raise NotImplementedError
