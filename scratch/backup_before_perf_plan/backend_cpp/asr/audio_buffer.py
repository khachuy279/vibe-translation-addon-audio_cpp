import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

# VAD frame state enumeration (aligned per 25ms = 400 samples @ 16kHz)
VAD_STATE_NON_SPEECH: int = 0
VAD_STATE_SPEECH: int = 1
VAD_STATE_PRE_ROLL: int = 2


@dataclass(slots=True)
class AudioSnapshot:
    """Immutable snapshot of audio buffer with frame-aligned VAD metadata."""
    pcm: np.ndarray          # float32 [-1.0, 1.0]
    frame_state: np.ndarray  # uint8, 1 element per 25ms frame (400 samples)
    duration_sec: float
    sample_count: int
    version: int


class AudioBufferManager:
    """High-performance raw PCM audio buffer manager with frame-aligned VAD metadata,
    atomic slicing, versioning, and capacity limits.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_duration_sec: float = 60.0,
        frame_samples: int = 400,  # 25ms @ 16kHz
    ):
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.max_bytes: int = int(self.max_duration_sec * self.sample_rate * 2) if max_duration_sec > 0 else 0
        self.frame_samples: int = frame_samples
        self.frame_bytes: int = self.frame_samples * 2  # 16-bit mono = 2 bytes/sample

        self._bytes_buffer: bytearray = bytearray()
        self._frame_states: bytearray = bytearray()
        self._pending_state: int = VAD_STATE_NON_SPEECH
        self._buffer_lock = threading.Lock()
        self._version: int = 0

    def feed_bytes(self, pcm_bytes: bytes, vad_state: int = VAD_STATE_SPEECH) -> float:
        """Append incoming 16-bit mono PCM bytes and track frame-aligned VAD metadata."""
        if not pcm_bytes:
            with self._buffer_lock:
                return len(self._bytes_buffer) / (2.0 * self.sample_rate)

        with self._buffer_lock:
            self._bytes_buffer.extend(pcm_bytes)
            self._pending_state = vad_state

            total_complete = len(self._bytes_buffer) // self.frame_bytes
            new_frames = total_complete - len(self._frame_states)
            if new_frames > 0:
                self._frame_states.extend([vad_state] * new_frames)

            # Drop oldest whole frames if exceeding max_bytes (maintains frame grid alignment)
            if self.max_bytes > 0 and len(self._bytes_buffer) > self.max_bytes:
                excess_bytes = len(self._bytes_buffer) - self.max_bytes
                excess_frames = (excess_bytes + self.frame_bytes - 1) // self.frame_bytes
                drop_bytes = excess_frames * self.frame_bytes
                del self._bytes_buffer[:drop_bytes]
                del self._frame_states[:excess_frames]
                self._version += 1
                logger.warning(
                    f"AudioBufferManager: Buffer exceeded max duration ({self.max_duration_sec:.1f}s), "
                    f"dropped {excess_frames} frames ({drop_bytes} bytes, {drop_bytes / (2.0 * self.sample_rate):.2f}s)."
                )
            return len(self._bytes_buffer) / (2.0 * self.sample_rate)

    def get_snapshot(self) -> Optional[AudioSnapshot]:
        """Return a snapshot of current audio buffer as AudioSnapshot."""
        return self.get_snapshot_with_version()

    def get_snapshot_with_version(self) -> Optional[AudioSnapshot]:
        """Return a snapshot of current audio buffer along with frame_state and version."""
        with self._buffer_lock:
            if not self._bytes_buffer:
                return None
            raw_bytes = bytes(self._bytes_buffer)
            states = bytearray(self._frame_states)
            pending_state = self._pending_state
            ver = self._version

        samples_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        samples_f32 = samples_int16.astype(np.float32) / 32768.0
        n_samples = len(samples_f32)
        dur = n_samples / float(self.sample_rate)

        n_total_frames = (n_samples + self.frame_samples - 1) // self.frame_samples
        if n_total_frames > len(states):
            states.extend([pending_state] * (n_total_frames - len(states)))
        frame_state = np.frombuffer(bytes(states[:n_total_frames]), dtype=np.uint8)

        return AudioSnapshot(
            pcm=samples_f32,
            frame_state=frame_state,
            duration_sec=dur,
            sample_count=n_samples,
            version=ver,
        )

    def get_snapshot_if_newer(self, last_samples: int = 0) -> Optional[AudioSnapshot]:
        """Return a snapshot of audio only if buffer has new samples beyond last_samples."""
        with self._buffer_lock:
            sample_count = len(self._bytes_buffer) // 2
            if sample_count <= last_samples or not self._bytes_buffer:
                return None
            raw_bytes = bytes(self._bytes_buffer)
            states = bytearray(self._frame_states)
            pending_state = self._pending_state
            ver = self._version

        samples_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        samples_f32 = samples_int16.astype(np.float32) / 32768.0
        n_samples = len(samples_f32)
        dur = n_samples / float(self.sample_rate)

        n_total_frames = (n_samples + self.frame_samples - 1) // self.frame_samples
        if n_total_frames > len(states):
            states.extend([pending_state] * (n_total_frames - len(states)))
        frame_state = np.frombuffer(bytes(states[:n_total_frames]), dtype=np.uint8)

        return AudioSnapshot(
            pcm=samples_f32,
            frame_state=frame_state,
            duration_sec=dur,
            sample_count=n_samples,
            version=ver,
        )

    def pop_all(self) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
        """Flush and return all buffered audio as (pcm_float32, frame_state, duration_sec)."""
        with self._buffer_lock:
            if not self._bytes_buffer:
                return None, np.empty(0, dtype=np.uint8), 0.0
            raw_bytes = bytes(self._bytes_buffer)
            states = bytearray(self._frame_states)
            pending_state = self._pending_state
            self._bytes_buffer.clear()
            self._frame_states.clear()
            self._version += 1

        samples_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        samples_f32 = samples_int16.astype(np.float32) / 32768.0
        n_samples = len(samples_f32)
        dur = n_samples / float(self.sample_rate)

        n_total_frames = (n_samples + self.frame_samples - 1) // self.frame_samples
        if n_total_frames > len(states):
            states.extend([pending_state] * (n_total_frames - len(states)))
        frame_state = np.frombuffer(bytes(states[:n_total_frames]), dtype=np.uint8)

        return samples_f32, frame_state, dur

    def slice_after(self, sample_offset: int, expected_version: Optional[int] = None) -> None:
        """Trim the first `sample_offset` samples from the buffer, maintaining frame grid alignment."""
        with self._buffer_lock:
            if expected_version is not None and self._version != expected_version:
                return

            total_samples = len(self._bytes_buffer) // 2
            if sample_offset >= total_samples:
                self._bytes_buffer.clear()
                self._frame_states.clear()
                self._version += 1
                return

            frames_to_drop = sample_offset // self.frame_samples
            bytes_to_drop = frames_to_drop * self.frame_bytes
            if bytes_to_drop > 0:
                del self._bytes_buffer[:bytes_to_drop]
                del self._frame_states[:frames_to_drop]
                self._version += 1

    def clear(self) -> None:
        """Reset audio buffer to empty state."""
        with self._buffer_lock:
            self._bytes_buffer.clear()
            self._frame_states.clear()
            self._version += 1

    @property
    def duration_sec(self) -> float:
        with self._buffer_lock:
            return len(self._bytes_buffer) / (2.0 * self.sample_rate)

    @property
    def is_empty(self) -> bool:
        with self._buffer_lock:
            return len(self._bytes_buffer) == 0

    @property
    def version(self) -> int:
        with self._buffer_lock:
            return self._version
