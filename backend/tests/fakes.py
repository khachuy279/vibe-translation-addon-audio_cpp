"""Fake engines cho test tầng A (không cần nạp model thật).

Mục tiêu: kiểm thử LOGIC điều phối của pipeline (phân câu, cửa sổ preview, gộp commit,
ranh giới pre-roll, áp dụng cấu hình) trong vài giây thay vì vài phút.

Cách tiếp cận:
- `FakeInferenceEngine` KẾ THỪA `TranscribeEngine` và chỉ override `_run_inference_sync`.
  Nhờ vậy toàn bộ logic orchestration thật (`stream_tokens`, commit tiers, cửa sổ
  preview, xử lý pre-roll) được test nguyên vẹn, chỉ thay phần suy luận C++.
- `FakeVADEngine` là một engine VAD xác định (deterministic) dựa trên năng lượng RMS,
  để test có thể điều khiển chính xác thời điểm bắt đầu/kết thúc nói.
"""

from collections import deque
import threading
import time
from typing import Any, Callable, List, Optional

import numpy as np

from backend.asr.engine import TranscribeEngine
from backend.vad.base import BaseVADEngine, VADResult, VADStreamState


class FakeInferenceEngine(TranscribeEngine):
    """TranscribeEngine không cần model: `_run_inference_sync` trả text giả lập."""

    def __init__(
        self,
        text_fn: Optional[Callable[[int], str]] = None,
        infer_delay: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.infer_delay = float(infer_delay)
        self.infer_calls: List[int] = []  # độ dài (samples) mỗi lần suy luận
        self.infer_times: List[float] = []
        self.text_fn = text_fn or (lambda n: "alpha beta gamma delta epsilon")

    def _run_inference_sync(self, pcm_audio: np.ndarray) -> str:  # type: ignore[override]
        n = 0 if pcm_audio is None else len(pcm_audio)
        self.infer_calls.append(n)
        self.infer_times.append(time.perf_counter())
        if self.infer_delay > 0:
            time.sleep(self.infer_delay)
        if n == 0:
            return ""
        return self.text_fn(n)

    def _preload_model(self, force_warm: bool = False) -> None:  # type: ignore[override]
        """Tầng A: KHÔNG có model thật để nạp/pre-warm — bỏ qua.

        Nếu không override, `stream_tokens` sẽ gọi `_ensure_model_loaded()` và cố nạp
        GGUF thật, làm test tầng A mất hết ý nghĩa "không cần model".
        Giữ đúng chữ ký `force_warm` để test có thể gọi `prewarm()`.
        """
        return

    # Tiện ích cho test
    @property
    def slice_count(self) -> int:
        return len(self.infer_calls)


class StableTextEngine(FakeInferenceEngine):
    """Trả text chỉ phụ thuộc vào một "cột mốc" để test STABLE_PREFIX.

    `milestones` là danh sách (min_samples, text). Text trả về là text của mốc cao
    nhất mà `n >= min_samples`. Nhờ vậy text "đứng yên" khi audio vẫn đang dài ra,
    mô phỏng việc người nói ngừng nói giữa câu.
    """

    def __init__(self, milestones, **kwargs: Any):
        super().__init__(**kwargs)
        self.milestones = sorted(milestones, key=lambda m: m[0])

    def _run_inference_sync(self, pcm_audio: np.ndarray) -> str:  # type: ignore[override]
        n = 0 if pcm_audio is None else len(pcm_audio)
        self.infer_calls.append(n)
        self.infer_times.append(time.perf_counter())
        if self.infer_delay > 0:
            time.sleep(self.infer_delay)
        text = ""
        for min_samples, value in self.milestones:
            if n >= min_samples:
                text = value
        return text


class FakeVADEngine(BaseVADEngine):
    """VAD xác định theo năng lượng RMS của frame (không cần model)."""

    name = "fake-vad"
    native_frame_samples = 400  # 25ms @ 16kHz
    default_threshold = 0.45

    def __init__(self, rms_threshold: float = 0.02):
        self.rms_threshold = float(rms_threshold)

    def create_initial_state(self, threshold: Optional[float] = None) -> VADStreamState:
        return VADStreamState()

    def is_speech(
        self,
        chunk_float32: Optional[np.ndarray],
        state: VADStreamState,
        threshold: float,
        chunk_raw: Optional[bytes] = None,
    ) -> VADResult:
        if chunk_raw is not None:
            arr = np.frombuffer(chunk_raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif chunk_float32 is not None:
            arr = np.asarray(chunk_float32, dtype=np.float32)
        else:
            arr = np.zeros(0, dtype=np.float32)

        if arr.size == 0:
            rms = 0.0
        else:
            rms = float(np.sqrt(np.dot(arr, arr) / arr.size))

        is_speech = rms > self.rms_threshold
        return VADResult(is_speech=is_speech, probability=min(1.0, rms / max(1e-6, self.rms_threshold)), event=None)


class FakeTranslator:
    """Translator giả: trả bản dịch có tiền tố, không nạp model."""

    def __init__(self, prefix: str = "[vi] ", per_char_delay: float = 0.0):
        self.prefix = prefix
        self.calls: List[str] = []
        self.per_char_delay = per_char_delay

    async def translate(self, text: str, source_lang: str = "auto", target_lang: str = "vi", context: str = "") -> dict:
        self.calls.append(text)
        return {"translated_text": f"{self.prefix}{text}", "elapsed_ms": 1.0}

    async def translate_stream(self, text: str, source_lang: str = "auto", target_lang: str = "vi", context: str = ""):
        self.calls.append(text)
        acc = self.prefix
        yield acc
        for ch in text:
            acc += ch
            if self.per_char_delay:
                import asyncio
                await asyncio.sleep(self.per_char_delay)
            yield acc


def make_speech_pcm(duration_sec: float, sample_rate: int = 16000, amplitude: float = 0.3) -> np.ndarray:
    """Tạo audio "có tiếng nói" xác định (sóng sin biên độ lớn hơn ngưỡng RMS)."""
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def make_silence_pcm(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Tạo im lặng tuyệt đối."""
    return np.zeros(int(sample_rate * duration_sec), dtype=np.float32)


def pcm_to_int16_bytes(pcm: np.ndarray) -> bytes:
    """Chuyển float32 [-1,1] sang bytes PCM16 (đúng định dạng client gửi)."""
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
