"""Voice Activity Detection (VAD) Processor using official VAD libraries.

Supports three official production VAD engines:
1. FireRed-VAD (fireredvad) - SOTA multilingual DFSMN streaming VAD (Xiaohongshu)
2. Silero VAD (silero-vad) - Snakers4 official PyTorch/JIT VAD with hysteresis
3. FSMN-VAD (funasr) - Alibaba DAMO Academy industrial streaming VAD

Architecture:
- Stateless/Thread-Safe Models pooled in VADEngineFactory.
- Session-isolated streaming states via VADStreamState (Safe for multi-client / concurrent sessions).
- Audio Sample Clock for precise, deterministic speech boundary detection.
"""

import collections
import logging
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


import numpy as np

from backend_cpp.vad.engines import (
    DEFAULT_THRESHOLDS,
    BaseVADEngine,
    SileroOfficialVADEngine,
    FsmnOfficialVADEngine,
    FireRedOfficialVADEngine,
    VADEngineFactory,
)
from backend_cpp.vad.stream_state import VADStreamState
from backend_cpp.asr.audio_buffer import (
    VAD_STATE_NON_SPEECH,
    VAD_STATE_SPEECH,
    VAD_STATE_PRE_ROLL,
)

# Maintain backward compatibility for any direct imports
SileroVADEngine = SileroOfficialVADEngine
FsmnVADEngine = FsmnOfficialVADEngine
FireRedVADEngine = FireRedOfficialVADEngine


logger = logging.getLogger(__name__)


class VADProcessor:
    """Stream audio chunk buffer and trigger VAD callbacks.

    State is isolated per VADProcessor instance (session-safe).
    Underlying neural models are shared singletons pooled in VADEngineFactory.

    Dual Hysteresis Note:
    - Primary: Events ('START' / 'END') emitted directly by the official engine
      postprocessor take immediate precedence.
    - Safety Net: Processor hangover (silence_duration_ms + hangover_ms) acts as an outer
      safety net for frames where the engine does not emit an explicit event.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_engine: str = "firered-vad",
        threshold: Optional[float] = None,
        silence_duration_ms: int = 600,
        hangover_ms: int = 400,
        pre_speech_buffer_ms: int = 450,
        enabled: bool = True,
        on_speech_chunk: Optional[Callable[[bytes, float], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.vad_engine = (vad_engine or "firered-vad").lower().strip()
        if self.vad_engine not in DEFAULT_THRESHOLDS:
            self.vad_engine = "firered-vad"

        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLDS.get(self.vad_engine, 0.5)
        self.silence_duration_ms = silence_duration_ms
        self.hangover_ms = hangover_ms
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        self.enabled = enabled

        self.on_speech_chunk = on_speech_chunk
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._lock = threading.Lock()
        self._engine: Optional[BaseVADEngine] = None
        self._state: Optional[VADStreamState] = None

        self._frame_samples: int = 400
        self._frame_size_bytes: int = self._frame_samples * 2

        # Operational metrics
        self._overflow_count: int = 0
        self._callback_count: int = 0
        self._total_callback_time_ms: float = 0.0

        self._ensure_model()


    def _ensure_model(self) -> None:
        """Ensure the engine is loaded and initialize session state."""
        if self._engine is None:
            self._engine = VADEngineFactory.get_engine(self.vad_engine)
            self._frame_samples = getattr(self._engine, "native_frame_samples", 400)
            self._frame_size_bytes = self._frame_samples * 2

        if self._state is None:
            self._state = self._engine.create_initial_state(threshold=self.threshold)
            max_pre_frames = max(1, int((self.pre_speech_buffer_ms / 1000.0) * self.sample_rate / self._frame_samples))
            self._state.pre_speech_ring = collections.deque(maxlen=max_pre_frames)

    def update_config(
        self,
        vad_engine: Optional[str] = None,
        threshold: Optional[float] = None,
        silence_duration_ms: Optional[int] = None,
        hangover_ms: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Update VAD parameters dynamically."""
        with self._lock:
            if vad_engine is not None:
                eng = vad_engine.lower().strip()
                if eng in DEFAULT_THRESHOLDS and eng != self.vad_engine:
                    self.vad_engine = eng
                    self._engine = VADEngineFactory.get_engine(eng)
                    self._frame_samples = getattr(self._engine, "native_frame_samples", 400)
                    self._frame_size_bytes = self._frame_samples * 2
                    self._state = self._engine.create_initial_state(threshold=threshold)
                    max_pre_frames = max(1, int((self.pre_speech_buffer_ms / 1000.0) * self.sample_rate / self._frame_samples))
                    self._state.pre_speech_ring = collections.deque(maxlen=max_pre_frames)
                    if threshold is None:
                        self.threshold = DEFAULT_THRESHOLDS[eng]

            if threshold is not None:
                self.threshold = threshold

            if silence_duration_ms is not None:
                self.silence_duration_ms = silence_duration_ms
            if hangover_ms is not None:
                self.hangover_ms = hangover_ms
            if enabled is not None:
                self.enabled = enabled

    def feed_chunk(self, pcm_data: bytes, capture_timestamp: float = 0.0) -> None:
        """Process incoming 16kHz 16-bit mono PCM bytes.

        Uses deterministic Audio Sample Clock for speech/silence duration tracking.
        """
        if not pcm_data:
            return

        if not self.enabled:
            if self.on_speech_chunk:
                self.on_speech_chunk(pcm_data, capture_timestamp)
            return

        self._ensure_model()
        callbacks: List[Tuple[Callable, tuple]] = []

        with self._lock:
            state = self._state
            raw_buf = state.raw_buffer
            raw_buf.extend(pcm_data)

            # Bounded buffer protection: max 3 seconds audio to prevent unbounded memory growth / lag
            max_buffer_bytes = int(self.sample_rate * 2 * 3.0)
            if len(raw_buf) > max_buffer_bytes:
                overflow_bytes = len(raw_buf) - max_buffer_bytes
                del raw_buf[:overflow_bytes]
                self._overflow_count += 1
                if state.is_speech:
                    logger.warning(
                        f"[VAD Buffer Overflow during SPEECH] raw_buffer exceeded {max_buffer_bytes} bytes. "
                        f"Dropped oldest {overflow_bytes} bytes. Downstream processing is lagging!"
                    )
                else:
                    logger.warning(
                        f"[VAD Buffer Overflow] raw_buffer exceeded {max_buffer_bytes} bytes. Dropped oldest {overflow_bytes} bytes."
                    )

            frame_size = self._frame_size_bytes
            buf_len = len(raw_buf)
            offset = 0

            while buf_len - offset >= frame_size:
                frame_end = offset + frame_size
                frame_ts = capture_timestamp + (offset / (self.sample_rate * 2.0))

                # Convert to float32 normalized [-1.0, 1.0]
                samples_int16 = np.frombuffer(raw_buf, dtype=np.int16, count=self._frame_samples, offset=offset)
                samples_float32 = samples_int16.astype(np.float32) / 32768.0
                del samples_int16


                res = self._engine.is_speech(samples_float32, state, self.threshold)
                if len(res) == 3:
                    is_speech_frame, prob, vad_event = res
                else:
                    is_speech_frame, prob = res[:2]
                    vad_event = None

                state.total_samples_processed += self._frame_samples

                # Evaluate state transition
                if vad_event == "START" or (not state.is_speech and is_speech_frame):
                    # Transition: SILENCE -> SPEECH
                    if not state.is_speech:
                        state.is_speech = True
                        state.silence_samples = 0
                        logger.info(
                            f"[VAD START] Speech onset detected ({self.vad_engine}, threshold={self.threshold:.2f}, prob={prob:.2f})"
                        )
                        if self.on_speech_start:
                            callbacks.append((self.on_speech_start, ()))

                        # Flush pre-speech buffer (tagged as PRE_ROLL)
                        while state.pre_speech_ring:
                            pre_bytes, pre_ts = state.pre_speech_ring.popleft()
                            if self.on_speech_chunk:
                                callbacks.append((self.on_speech_chunk, (pre_bytes, pre_ts, VAD_STATE_PRE_ROLL)))

                    frame_bytes = bytes(raw_buf[offset:frame_end])
                    if self.on_speech_chunk:
                        callbacks.append((self.on_speech_chunk, (frame_bytes, frame_ts, VAD_STATE_SPEECH)))

                elif state.is_speech:
                    if vad_event == "END":
                        # Immediate speech end signaled by engine
                        state.is_speech = False
                        state.silence_samples = 0
                        logger.info(
                            f"[VAD END] Speech end detected by engine ({self.vad_engine})"
                        )
                        if self.on_speech_end:
                            callbacks.append((self.on_speech_end, ()))
                    elif is_speech_frame:
                        state.silence_samples = 0
                        frame_bytes = bytes(raw_buf[offset:frame_end])
                        if self.on_speech_chunk:
                            callbacks.append((self.on_speech_chunk, (frame_bytes, frame_ts, VAD_STATE_SPEECH)))
                    else:
                        # In speech but frame is silent -> accumulate silence
                        state.silence_samples += self._frame_samples
                        silence_elapsed_ms = (state.silence_samples / self.sample_rate) * 1000.0
                        total_silence_limit_ms = float(self.silence_duration_ms)
                        grace_hangover_ms = min(float(self.hangover_ms), total_silence_limit_ms * 0.5)

                        if silence_elapsed_ms <= grace_hangover_ms:
                            frame_bytes = bytes(raw_buf[offset:frame_end])
                            if self.on_speech_chunk:
                                callbacks.append((self.on_speech_chunk, (frame_bytes, frame_ts, VAD_STATE_NON_SPEECH)))
                        elif silence_elapsed_ms >= total_silence_limit_ms:
                            state.is_speech = False
                            state.silence_samples = 0
                            logger.info(
                                f"[VAD END] Speech end detected ({self.vad_engine}, threshold={self.threshold:.2f}) after {silence_elapsed_ms:.0f}ms silence"
                            )
                            if self.on_speech_end:
                                callbacks.append((self.on_speech_end, ()))
                else:
                    # Idle silence: add to pre-speech ring buffer
                    frame_bytes = bytes(raw_buf[offset:frame_end])
                    state.pre_speech_ring.append((frame_bytes, frame_ts))

                offset = frame_end

            if offset > 0:
                del raw_buf[:offset]

        # Execute callbacks outside the lock
        for cb, args in callbacks:
            try:
                t0 = time.perf_counter()
                try:
                    cb(*args)
                except TypeError:
                    if len(args) == 3:
                        cb(args[0], args[1])
                    else:
                        raise
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._callback_count += 1
                self._total_callback_time_ms += elapsed_ms
                if elapsed_ms > 100.0:
                    logger.warning(
                        f"⚠️ Slow VAD callback {getattr(cb, '__name__', str(cb))} took {elapsed_ms:.1f}ms"
                    )
            except Exception as e:
                logger.error(f"Error in VAD callback {getattr(cb, '__name__', str(cb))}: {e}", exc_info=True)

    def force_end(self) -> None:
        """Force speech termination when connection closes."""
        callbacks: List[Tuple[Callable, tuple]] = []
        with self._lock:
            state = self._state
            if state and state.is_speech:
                state.is_speech = False
                state.silence_samples = 0
                logger.info(f"[VAD END] Speech force-ended ({self.vad_engine})")
                while state.pre_speech_ring:
                    pre_bytes, pre_ts = state.pre_speech_ring.popleft()
                    if self.on_speech_chunk:
                        callbacks.append((self.on_speech_chunk, (pre_bytes, pre_ts)))
                if self.on_speech_end:
                    callbacks.append((self.on_speech_end, ()))

            if self._state:
                self._state.reset()

        for cb, args in callbacks:
            try:
                cb(*args)
            except Exception as e:
                logger.error(f"Error in VAD force_end callback: {e}", exc_info=True)

    def reset(self) -> None:
        """Reset internal state between streams."""
        with self._lock:
            if self._state:
                self._state.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Return VAD operational metrics for monitoring and diagnostics."""
        with self._lock:
            total_samples = self._state.total_samples_processed if self._state else 0
            return {
                "engine": self.vad_engine,
                "threshold": self.threshold,
                "is_speech": self._state.is_speech if self._state else False,
                "total_samples_processed": total_samples,
                "audio_seconds_processed": (total_samples / self.sample_rate) if self.sample_rate else 0.0,
                "overflow_count": self._overflow_count,
                "callback_count": self._callback_count,
                "avg_callback_ms": (self._total_callback_time_ms / self._callback_count) if self._callback_count > 0 else 0.0,
            }

