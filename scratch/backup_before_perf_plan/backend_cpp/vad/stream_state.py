"""VAD Stream State management for official VAD engines.

Encapsulates all session-specific buffers and recurrent states so that
inference engines remain completely thread-safe and session-isolated.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class VADStreamState:
    """Stateful context for a single streaming audio session."""

    # High-level speech state tracking
    is_speech: bool = False
    silence_samples: int = 0
    total_samples_processed: int = 0

    # Raw audio byte buffer for frame slicing (16kHz 16-bit mono PCM)
    raw_buffer: bytearray = field(default_factory=bytearray)

    # Ring buffer for pre-speech frames: (pcm_bytes, capture_timestamp)
    pre_speech_ring: Deque[Tuple[bytes, float]] = field(default_factory=deque)

    # FireRed Stream-VAD session state (fireredvad)
    firered_postprocessor: Optional[Any] = None
    firered_caches: Optional[Any] = None

    # Silero VAD session state (silero_vad)
    silero_iterator: Optional[Any] = None
    silero_model: Optional[Any] = None
    silero_probe: Optional[Any] = None

    # FSMN-VAD session state (funasr)
    fsmn_cache: Optional[Dict[str, Any]] = None
    fsmn_in_speech: bool = False

    def reset(self) -> None:
        """Reset all stream buffers and session states to initial values."""
        self.is_speech = False
        self.silence_samples = 0
        self.total_samples_processed = 0

        self.raw_buffer.clear()
        self.pre_speech_ring.clear()

        # Reset FireRed state
        if self.firered_postprocessor is not None:
            self.firered_postprocessor.reset()
        self.firered_caches = None

        # Reset Silero state
        if self.silero_iterator is not None:
            self.silero_iterator.reset_states()
        if self.silero_probe is not None:
            self.silero_probe.last_prob = 0.0

        # Reset FSMN state
        if self.fsmn_cache is not None:
            self.fsmn_cache.clear()
        self.fsmn_cache = None
        self.fsmn_in_speech = False

