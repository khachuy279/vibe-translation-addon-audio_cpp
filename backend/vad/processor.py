"""VAD Stream Processor: Bộ điều phối luồng VAD thời gian thực.

Tính năng:
- Nhận luồng âm thanh dạng bytes (PCM Int16/Float32) hoặc np.ndarray.
- Chia khung (Frame Slicing) chính xác theo kích thước native của engine.
- Quản lý trạng thái nói/im lặng bằng Đồng hồ Mẫu Âm thanh (Sample Clock).
- Hỗ trợ Pre-Speech Ring Buffer (đệm trước khi nói ~300ms) để không nuốt âm đầu.
- Hỗ trợ Hangover & Hysteresis (bảo vệ ngắt quãng giữa các từ).
- Cơ chế callbacks bất đồng bộ/đồng bộ an toàn cho luồng ASR downstream.
"""

from collections import deque
import logging
import threading
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

from backend.config import config
from backend.core.pipeline_events import VADState
from backend.vad.base import BaseVADEngine, VADResult, VADStreamState
from backend.vad.engines import VADEngineFactory, SUPPORTED_VAD_ENGINES
from backend.utils.logger import logger


class VADStreamProcessor:
    """Bộ xử lý VAD streaming cách ly theo phiên (Session-Safe)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_engine: str = "firered-vad",
        threshold: Optional[float] = None,
        silence_duration_ms: int = 600,
        hangover_ms: int = 400,
        pre_speech_buffer_ms: int = 300,
        enabled: bool = True,
        on_speech_chunk: Optional[Callable[[bytes, float, str], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.vad_engine = (vad_engine or "firered-vad").lower().strip()
        if self.vad_engine not in SUPPORTED_VAD_ENGINES:
            self.vad_engine = "firered-vad"

        self.threshold = threshold if threshold is not None else config.vad.threshold
        self.silence_duration_ms = silence_duration_ms
        self.hangover_ms = hangover_ms
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        self.enabled = enabled

        self.on_speech_chunk = on_speech_chunk
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._lock = threading.RLock()
        self._engine: Optional[BaseVADEngine] = None
        self._state: Optional[VADStreamState] = None

        self._frame_samples: int = 400
        self._frame_size_bytes: int = self._frame_samples * 2  # 16-bit PCM bytes

        self._overflow_count: int = 0
        self._ensure_engine()

    def _ensure_engine(self) -> None:
        """Đảm bảo engine và session state đã được khởi tạo."""
        if self._engine is None:
            self._engine = VADEngineFactory.get_engine(self.vad_engine)
            self._frame_samples = getattr(self._engine, "native_frame_samples", 400)
            self._frame_size_bytes = self._frame_samples * 2

        if self._state is None:
            self._state = self._engine.create_initial_state(threshold=self.threshold)
            max_pre_frames = max(1, int((self.pre_speech_buffer_ms / 1000.0) * self.sample_rate / self._frame_samples))
            self._state.pre_speech_ring = deque(maxlen=max_pre_frames)

    def update_config(
        self,
        vad_engine: Optional[str] = None,
        threshold: Optional[float] = None,
        silence_duration_ms: Optional[int] = None,
        hangover_ms: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Cập nhật cấu hình runtime nhanh chóng."""
        with self._lock:
            changes = []
            if vad_engine is not None:
                eng = vad_engine.lower().strip()
                if eng in SUPPORTED_VAD_ENGINES and eng != self.vad_engine:
                    old_eng = self.vad_engine
                    self.vad_engine = eng
                    self._engine = VADEngineFactory.get_engine(eng)
                    self._frame_samples = getattr(self._engine, "native_frame_samples", 400)
                    self._frame_size_bytes = self._frame_samples * 2
                    self._state = self._engine.create_initial_state(threshold=threshold)
                    max_pre_frames = max(1, int((self.pre_speech_buffer_ms / 1000.0) * self.sample_rate / self._frame_samples))
                    self._state.pre_speech_ring = deque(maxlen=max_pre_frames)
                    changes.append(f"engine: '{old_eng}' -> '{eng}'")

            if threshold is not None and float(threshold) != self.threshold:
                changes.append(f"threshold: {self.threshold:.2f} -> {float(threshold):.2f}")
                self.threshold = float(threshold)
            if silence_duration_ms is not None and int(silence_duration_ms) != self.silence_duration_ms:
                changes.append(f"silence: {self.silence_duration_ms}ms -> {int(silence_duration_ms)}ms")
                self.silence_duration_ms = int(silence_duration_ms)
            if hangover_ms is not None and int(hangover_ms) != self.hangover_ms:
                changes.append(f"hangover: {self.hangover_ms}ms -> {int(hangover_ms)}ms")
                self.hangover_ms = int(hangover_ms)
            if enabled is not None and bool(enabled) != self.enabled:
                changes.append(f"enabled: {self.enabled} -> {bool(enabled)}")
                self.enabled = bool(enabled)

            if changes:
                logger.info(
                    f"[VAD CONFIG] Đồng bộ cấu hình: {', '.join(changes)} (silence_duration_ms={self.silence_duration_ms}ms, threshold={self.threshold:.2f})",
                    extra={"module_tag": "VAD"},
                )

    def feed_chunk(self, audio_data: Union[bytes, np.ndarray], capture_timestamp: float = 0.0) -> None:
        """Xử lý nạp chunk âm thanh thô (bytes Int16/Float32 hoặc np.ndarray)."""
        if audio_data is None or len(audio_data) == 0:
            return

        if not self.enabled:
            if self.on_speech_chunk:
                pcm_bytes = audio_data if isinstance(audio_data, bytes) else audio_data.tobytes()
                self.on_speech_chunk(pcm_bytes, capture_timestamp, VADState.SPEECH.value)
            return

        self._ensure_engine()
        callbacks_to_fire: List[Tuple[Callable, tuple]] = []

        with self._lock:
            state = self._state
            raw_buf = state.raw_buffer

            # Chuyển đổi đầu vào thành bytes 16-bit PCM cho raw_buffer
            if isinstance(audio_data, bytes):
                raw_buf.extend(audio_data)
            elif isinstance(audio_data, np.ndarray):
                if audio_data.dtype == np.float32:
                    int16_arr = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
                    raw_buf.extend(int16_arr.tobytes())
                elif audio_data.dtype == np.int16:
                    raw_buf.extend(audio_data.tobytes())

            # Giới hạn buffer tối đa 3 giây để tránh phình bộ nhớ khi nghẽn
            max_buffer_bytes = int(self.sample_rate * 2 * 3.0)
            if len(raw_buf) > max_buffer_bytes:
                overflow_bytes = len(raw_buf) - max_buffer_bytes
                del raw_buf[:overflow_bytes]
                self._overflow_count += 1

            frame_size = self._frame_size_bytes
            buf_len = len(raw_buf)
            offset = 0

            while buf_len - offset >= frame_size:
                frame_end = offset + frame_size
                frame_ts = capture_timestamp + (offset / (self.sample_rate * 2.0))
                frame_bytes = bytes(raw_buf[offset:frame_end])

                # Với FireRed-VAD, truyền chunk_raw (Int16 PCM) trực tiếp
                # để triệt tiêu chi phí cấp phát và chuyển đổi Float32 -> Int16
                if getattr(self._engine, "name", "") == "firered-vad":
                    samples_float32 = None
                else:
                    samples_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
                    samples_float32 = samples_int16.astype(np.float32) / 32768.0

                res = self._engine.is_speech(
                    samples_float32,
                    state,
                    self.threshold,
                    chunk_raw=frame_bytes,
                )
                is_speech_frame = res.is_speech
                prob = res.probability
                vad_event = res.event

                state.total_samples_processed += self._frame_samples

                # Chuyển trạng thái: SILENCE -> SPEECH
                if vad_event == "START" or (not state.is_speech and is_speech_frame):
                    if not state.is_speech:
                        state.is_speech = True
                        state.silence_samples = 0
                        logger.info(
                            f"Speech START detected (prob={prob:.3f} >= threshold={self.threshold:.3f}, "
                            f"silence_limit={self.silence_duration_ms}ms, "
                            f"sample={state.total_samples_processed}, "
                            f"t={state.total_samples_processed/self.sample_rate:.2f}s)",
                            extra={"module_tag": "VAD"},
                        )
                        if self.on_speech_start:
                            callbacks_to_fire.append((self.on_speech_start, ()))

                        # Xả toàn bộ Pre-speech buffer
                        while state.pre_speech_ring:
                            pre_bytes, pre_ts = state.pre_speech_ring.popleft()
                            if self.on_speech_chunk:
                                callbacks_to_fire.append(
                                    (self.on_speech_chunk, (pre_bytes, pre_ts, VADState.PRE_ROLL.value))
                                )

                    if self.on_speech_chunk:
                        callbacks_to_fire.append(
                            (self.on_speech_chunk, (frame_bytes, frame_ts, VADState.SPEECH.value))
                        )

                elif state.is_speech:
                    if vad_event == "END":
                        state.is_speech = False
                        state.silence_samples = 0
                        logger.info(
                            f"Speech END detected by engine event (prob={prob:.3f} < threshold={self.threshold:.3f}, "
                            f"silence_limit={self.silence_duration_ms}ms, sample={state.total_samples_processed})",
                            extra={"module_tag": "VAD"},
                        )
                        if self.on_speech_end:
                            callbacks_to_fire.append((self.on_speech_end, ()))
                    elif is_speech_frame:
                        state.silence_samples = 0
                        if self.on_speech_chunk:
                            callbacks_to_fire.append(
                                (self.on_speech_chunk, (frame_bytes, frame_ts, VADState.SPEECH.value))
                            )
                    else:
                        # Đang trong câu nói nhưng frame hiện tại là khoảng lặng -> Tích lũy silence
                        state.silence_samples += self._frame_samples
                        silence_elapsed_ms = (state.silence_samples / self.sample_rate) * 1000.0
                        total_silence_limit_ms = float(self.silence_duration_ms)
                        grace_hangover_ms = min(float(self.hangover_ms), total_silence_limit_ms * 0.5)

                        if silence_elapsed_ms <= grace_hangover_ms:
                            # Vẫn nằm trong vùng ân hạn Hangover -> Tiếp tục gửi cho ASR
                            if self.on_speech_chunk:
                                callbacks_to_fire.append(
                                    (self.on_speech_chunk, (frame_bytes, frame_ts, VADState.SPEECH.value))
                                )

                        if silence_elapsed_ms >= total_silence_limit_ms:
                            # Đạt ngưỡng im lặng chốt câu -> SILENCE
                            state.is_speech = False
                            state.silence_samples = 0
                            logger.info(
                                f"Speech END detected by silence timeout ({silence_elapsed_ms:.0f}ms >= "
                                f"silence_limit={total_silence_limit_ms:.0f}ms, "
                                f"prob={prob:.3f} < threshold={self.threshold:.3f})",
                                extra={"module_tag": "VAD"},
                            )
                            if self.on_speech_end:
                                callbacks_to_fire.append((self.on_speech_end, ()))
                else:
                    # SILENCE state -> Lưu frame vào Pre-speech ring buffer
                    state.pre_speech_ring.append((frame_bytes, frame_ts))

                offset = frame_end

            if offset > 0:
                del raw_buf[:offset]

        # Kích hoạt callbacks bên ngoài lock để tránh deadlock
        for fn, args in callbacks_to_fire:
            try:
                fn(*args)
            except Exception as e:
                logger.error(f"VAD callback error: {e}", exc_info=True, extra={"module_tag": "VAD"})

    def force_end(self) -> None:
        """Ép buộc kết thúc câu nói hiện tại và kích hoạt callback chốt câu."""
        with self._lock:
            if self._state and self._state.is_speech:
                self._state.is_speech = False
                self._state.silence_samples = 0
                if self.on_speech_end:
                    try:
                        self.on_speech_end()
                    except Exception as e:
                        logger.error(f"VAD on_speech_end callback error: {e}", extra={"module_tag": "VAD"})

    def reset(self) -> None:
        """Reset trạng thái processor."""
        with self._lock:
            if self._state:
                self._state.reset()
            self._overflow_count = 0



# Alias tương thích ngược
VADProcessor = VADStreamProcessor

