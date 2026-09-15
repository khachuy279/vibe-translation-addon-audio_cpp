"""TranscribeEngine: Engine ASR Streaming kết nối transcribe.cpp C++ Backend.

Đặc tính kỹ thuật:
- Tối ưu hóa hiệu năng cao cho 1 session: streaming token generator không độ trễ.
- Hỗ trợ Hot-Switch model an toàn trên GPU: Hủy tác vụ inference đang chạy trước khi nạp model mới.
- Tích hợp Speech Normalization thích ứng trước khi gửi vào Audio-LLMs.
- Tự động nhận diện mô hình streaming (session.stream) hoặc offline (session.run).
- Định dạng và lọc sạch output văn bản theo thời gian thực.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional
import numpy as np

try:
    import transcribe_cpp
except ImportError:
    transcribe_cpp = None

from backend.config import config, SentenceConfig
from backend.core.audio_buffer import CircularAudioBuffer
from backend.core.normalizer import SpeechNormalizer
from backend.core.commit_manager import CommitManager
from backend.asr.base import BaseASREngine
from backend.asr.registry import ModelRegistry
from backend.asr.adapters import build_family_options, normalize_language_for_family
from backend.asr.text_cleaner import clean_transcript_text
from backend.utils.cuda import setup_cuda_dll_paths
from backend.utils.logger import logger

setup_cuda_dll_paths()

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr_worker")


class TranscribeEngine(BaseASREngine):
    """Engine nhận dạng giọng nói ASR streaming qua transcribe.cpp."""

    # Model singleton dùng chung trên GPU
    _shared_model: Optional[Any] = None
    _shared_model_key: Optional[str] = None
    _shared_session: Optional[Any] = None
    _shared_supports_streaming: bool = False
    _shared_lock = threading.RLock()
    _infer_lock = threading.RLock()

    @classmethod
    def shutdown_executors(cls, wait: bool = False) -> None:
        """Dừng các luồng worker executor."""
        try:
            _EXECUTOR.shutdown(wait=wait, cancel_futures=True)
        except Exception:
            pass

    @classmethod
    def unload_shared_model(cls) -> None:
        """Giải phóng hoàn toàn model và session khỏi GPU VRAM."""
        with cls._shared_lock:
            if cls._shared_session is not None:
                try:
                    cls._shared_session.close()
                except Exception:
                    pass
                cls._shared_session = None

            if cls._shared_model is not None:
                try:
                    cls._shared_model.close()
                except Exception:
                    pass
                cls._shared_model = None
                cls._shared_model_key = None
                cls._shared_supports_streaming = False
                logger.info("Đã giải phóng model ASR khỏi GPU VRAM.", extra={"module_tag": "ASR"})

    def __init__(
        self,
        model_key: Optional[str] = None,
        language: str = "auto",
        backend: str = "auto",
        threads: int = 4,
        min_transcribe_sec: float = 0.6,
        poll_interval_ms: int = 350,
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.registry = ModelRegistry.get_instance()
        self.model_key = model_key or self.registry.get_active_model_key()
        self.model_info = self.registry.get_model_info(self.model_key) or {}
        self.language = language or config.asr.language
        self.backend = backend or config.asr.backend
        self.threads = threads or config.asr.threads
        self.min_transcribe_sec = min_transcribe_sec
        self.poll_interval_ms = poll_interval_ms

        self.audio_buffer = CircularAudioBuffer(sample_rate=16000, capacity_sec=60.0)
        self.normalizer = SpeechNormalizer()
        self.commit_manager = CommitManager(sentence_cfg=config.sentence)

        self._active_utterance_id: str = str(uuid.uuid4())[:8]
        self._speech_active: bool = False
        self._speech_start_sample: int = 0
        self._last_committed_sample: int = 0
        from collections import deque
        self._pending_commits: deque = deque()

        self._preview_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._is_running: bool = True
        self._lock = threading.RLock()


    def update_sentence_config(self, **kwargs) -> None:
        """Cập nhật cấu hình phân câu và ngắt đoạn runtime."""
        for k, v in kwargs.items():
            if hasattr(self.commit_manager.cfg, k):
                setattr(self.commit_manager.cfg, k, v)



    def _ensure_model_loaded(self) -> Any:
        """Nạp model GGUF và khởi tạo session transcribe.cpp trên GPU."""
        with self.__class__._shared_lock:
            if (
                self.__class__._shared_model is not None
                and self.__class__._shared_model_key == self.model_key
            ):
                return self.__class__._shared_model

            if transcribe_cpp is None:
                raise RuntimeError("transcribe_cpp package chưa được cài đặt!")

            model_path = self.registry.resolve_model_path(self.model_key)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"File model GGUF không tồn tại: {model_path}")

            info = self.registry.get_model_info(self.model_key) or {}
            family = info.get("family", "")

            logger.info(
                f"Đang nạp mô hình ASR '{self.model_key}' ({family}) từ: {model_path}",
                extra={"module_tag": "ASR"},
            )

            # Giải phóng session cũ nếu có
            if self.__class__._shared_session is not None:
                try:
                    self.__class__._shared_session.close()
                except Exception:
                    pass
                self.__class__._shared_session = None

            if self.__class__._shared_model is not None:
                try:
                    self.__class__._shared_model.close()
                except Exception:
                    pass
                self.__class__._shared_model = None

            load_backend = "auto" if not self.backend or self.backend.lower() in ("auto", "none") else self.backend.lower()
            if load_backend == "auto":
                loaded_model = transcribe_cpp.Model(model_path)
            else:
                loaded_model = transcribe_cpp.Model(model_path, backend=load_backend)

            self.__class__._shared_model = loaded_model
            self.__class__._shared_model_key = self.model_key
            self.__class__._shared_supports_streaming = bool(
                getattr(loaded_model.capabilities, "supports_streaming", False)
            )

            # Tạo session mới
            self.__class__._shared_session = loaded_model.session(n_threads=self.threads)

            logger.info(
                f"Nạp thành công ASR Model '{self.model_key}' trên GPU "
                f"(Arch: {getattr(loaded_model, 'arch', 'unknown')}, Backend: {getattr(loaded_model, 'backend', 'unknown')}, "
                f"Streaming: {self.__class__._shared_supports_streaming})",
                extra={"module_tag": "ASR"},
            )
            return loaded_model

    def prewarm(self) -> None:
        """Prewarm mô hình trên GPU bằng đoạn audio im lặng giả lập."""
        try:
            self._ensure_model_loaded()
            dummy_pcm = np.zeros(int(16000 * 0.5), dtype=np.float32)
            self._run_inference_sync(dummy_pcm)
            logger.info(f"Pre-warm hoàn tất cho ASR model '{self.model_key}'", extra={"module_tag": "ASR"})
        except Exception as e:
            logger.warning(f"Pre-warm ASR warning: {e}", extra={"module_tag": "ASR"})

    def feed_audio(self, audio_data: Any, timestamp: float = 0.0, vad_state: str = "") -> None:
        """Nạp dữ liệu audio (bytes PCM Int16 hoặc ndarray Float32) vào buffer."""
        if audio_data is None:
            return

        if isinstance(audio_data, (bytes, bytearray)):
            if len(audio_data) == 0:
                return
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_float32 = (audio_int16.astype(np.float32) / 32768.0)
            self.audio_buffer.write(audio_float32)
        elif isinstance(audio_data, np.ndarray):
            if len(audio_data) == 0:
                return
            if audio_data.dtype == np.int16:
                audio_float32 = (audio_data.astype(np.float32) / 32768.0)
                self.audio_buffer.write(audio_float32)
            else:
                self.audio_buffer.write(audio_data.astype(np.float32))


    def on_speech_start(self) -> None:
        """Callback khi VAD phát hiện bắt đầu nói."""
        with self._lock:
            self._speech_active = True
            self._active_utterance_id = str(uuid.uuid4())[:8]
            self._speech_start_sample = max(0, self.audio_buffer.total_written - 4800)  # pre-roll 300ms

    def on_speech_end(self, reason: str = "VAD_SILENCE") -> None:
        """Callback khi VAD phát hiện kết thúc nói (kích hoạt chốt câu)."""
        with self._lock:
            if not self._speech_active:
                return
            self._speech_active = False
            end_sample = self.audio_buffer.total_written
            
            # Đưa vào hàng đợi commit câu
            self._pending_commits.append({
                "utterance_id": self._active_utterance_id,
                "start_sample": self._speech_start_sample,
                "end_sample": end_sample,
                "reason": reason,
            })

    def set_language(self, language: str) -> None:
        """Cập nhật ngôn ngữ nhận dạng."""
        self.language = language or "auto"

    def _run_inference_sync(self, pcm_audio: np.ndarray) -> str:
        """Thực hiện suy luận ASR đồng bộ trên luồng worker C++."""
        if pcm_audio is None or len(pcm_audio) < int(16000 * self.min_transcribe_sec):
            return ""

        model = self._ensure_model_loaded()

        # Chuẩn hóa âm lượng nếu bật config
        if config.asr.normalize_speech:
            norm_res = self.normalizer.normalize(pcm_audio)
            audio_to_infer = norm_res.audio
        else:
            audio_to_infer = pcm_audio

        info = self.registry.get_model_info(self.model_key) or {}
        family = info.get("family", "")
        lang = normalize_language_for_family(self.language, family)

        with self.__class__._infer_lock:
            session = self.__class__._shared_session
            if session is None:
                session = model.session(n_threads=self.threads)
                self.__class__._shared_session = session

            raw_text = ""
            stream_success = False

            # 1. Nếu model hỗ trợ native streaming
            if self.__class__._shared_supports_streaming:
                try:
                    stream_opts = build_family_options(family, info, model=model, slot="stream")
                    with session.stream(language=lang, family=stream_opts) as stream:
                        stream.feed(audio_to_infer)
                        stream.finalize()
                        raw_text = stream.text().full
                    stream_success = True
                except Exception as stream_err:
                    logger.debug(f"session.stream fallback to session.run: {stream_err}")

            # 2. Non-streaming hoặc fallback: session.run
            if not stream_success:
                try:
                    run_opts = build_family_options(family, info, model=model, slot="run")
                    res = session.run(audio_to_infer, language=lang, family=run_opts)
                    raw_text = getattr(res, "text", str(res))
                except Exception as run_err:
                    partial = getattr(run_err, "partial_result", None)
                    if partial is not None and hasattr(partial, "text"):
                        raw_text = str(partial.text)
                    else:
                        logger.error(f"Lỗi suy luận transcribe.cpp: {run_err}", exc_info=True, extra={"module_tag": "ASR"})
                        return ""

            return clean_transcript_text(raw_text)

    async def stream_tokens(self) -> AsyncIterator[Dict[str, Any]]:
        """Async generator liên tục thăm dò preview và xuất bản kết quả ASR."""
        poll_interval = self.poll_interval_ms / 1000.0

        while self._is_running:
            # 1. Kiểm tra nếu có pending commit cần xử lý ưu tiên
            commit_req = None
            with self._lock:
                if self._pending_commits:
                    commit_req = self._pending_commits.popleft()

            if commit_req:
                utt_id = commit_req["utterance_id"]
                start_s = commit_req["start_sample"]
                end_s = commit_req["end_sample"]
                reason = commit_req["reason"]

                audio_slice = self.audio_buffer.get_slice(start_s, end_s)
                
                t0 = time.perf_counter()
                loop = asyncio.get_running_loop()
                final_text = await loop.run_in_executor(_EXECUTOR, self._run_inference_sync, audio_slice)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                if final_text:
                    logger.info(
                        f"[ASR COMMIT] [utt={utt_id}] [{self.model_key}] [{reason}] '{final_text}' (infer={infer_ms:.1f}ms)",
                        extra={"module_tag": "ASR_COMMIT"},
                    )
                    yield {
                        "type": "utterance_update",
                        "utterance_id": utt_id,
                        "text": final_text,
                        "is_final": True,
                        "language": self.language,
                        "inference_ms": infer_ms,
                        "commit_reason": reason,
                    }
                continue

            # 2. Nếu đang trong trạng thái nói -> Thăm dò preview text
            if self._speech_active:
                current_total = self.audio_buffer.total_written
                start_s = self._speech_start_sample
                
                if current_total - start_s >= int(16000 * self.min_transcribe_sec):
                    audio_slice = self.audio_buffer.get_slice(start_s, current_total)
                    
                    t0 = time.perf_counter()
                    loop = asyncio.get_running_loop()
                    preview_text = await loop.run_in_executor(_EXECUTOR, self._run_inference_sync, audio_slice)
                    infer_ms = (time.perf_counter() - t0) * 1000.0

                    if preview_text and self._speech_active:
                        yield {
                            "type": "utterance_update",
                            "utterance_id": self._active_utterance_id,
                            "text": preview_text,
                            "is_final": False,
                            "stable_text": preview_text,
                            "unstable_text": "",
                            "language": self.language,
                            "inference_ms": infer_ms,
                        }

            await asyncio.sleep(poll_interval)

    async def cleanup(self) -> None:
        """Dừng generator và giải phóng tài nguyên session."""
        self._is_running = False
        with self._lock:
            self._speech_active = False
            self._pending_commits.clear()
        self.audio_buffer.clear()

