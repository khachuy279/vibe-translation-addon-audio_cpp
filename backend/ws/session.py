"""Module quản lý trạng thái phiên kết nối WebSocket (SessionState).

Đặc điểm tối ưu cho 1 Session duy nhất:
- Cấu hình linh hoạt qua Pydantic aliased model.
- Khởi tạo đầy đủ chu trình: VADProcessor -> TranscribeEngine -> CommitManager -> Translation Queue -> TTS Queue.
- Hàng đợi bất đồng bộ có Back-pressure (maxsize=4) ngăn chặn tràn bộ nhớ.
- Fast Cleanup (< 200ms) giải phóng toàn bộ tài nguyên khi Client ngắt kết nối.
"""

import asyncio
import time
import uuid
from typing import Optional, Dict, Any, Union
from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict, Field

from backend.config import config
from backend.asr.engine import TranscribeEngine
from backend.asr.registry import ModelRegistry
from backend.vad.processor import VADProcessor
from backend.ws.connection import SafeWebSocketConnection
from backend.core.metrics import metrics_collector
from backend.utils.logger import get_logger

logger = get_logger("ws.session")


class SessionConfigPayload(BaseModel):
    """Mô hình phân tích thông điệp cấu hình từ Extension (hỗ trợ cả camelCase và snake_case)."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    model_id: Optional[str] = Field(default=None, alias="modelId")
    asr_engine: Optional[str] = Field(default=None, alias="asrEngine")
    asr_model: Optional[str] = Field(default=None, alias="asrModel")
    source_lang: Optional[str] = Field(default=None, alias="sourceLang")
    target_lang: Optional[str] = Field(default=None, alias="targetLang")
    vad_engine: Optional[str] = Field(default=None, alias="vadEngine")
    vad_threshold: Optional[float] = Field(default=None, alias="vadThreshold")
    threshold: Optional[float] = None
    silence_duration_ms: Optional[int] = Field(default=None, alias="silenceDurationMs")
    hangover_ms: Optional[int] = Field(default=None, alias="hangoverMs")
    vad_enabled: Optional[bool] = Field(default=None, alias="vadEnabled")
    tts_enabled: Optional[bool] = Field(default=None, alias="ttsEnabled")
    tts_voice: Optional[str] = Field(default=None, alias="ttsVoice")
    tts_speed: Optional[float] = Field(default=None, alias="ttsSpeed")
    tts_ref_audio: Optional[str] = Field(default=None, alias="ttsRefAudio")
    tts_ref_text: Optional[str] = Field(default=None, alias="ttsRefText")
    translation_model: Optional[str] = Field(default=None, alias="translationModel")

    # Các thông số phân câu
    split_on_stability: Optional[bool] = Field(default=None, alias="splitOnStability")
    enable_stability_split: Optional[bool] = None
    stability_duration_sec: Optional[float] = Field(default=None, alias="stabilityDurationSec")
    max_duration_sec: Optional[float] = Field(default=None, alias="maxDurationSec")
    max_chars: Optional[int] = Field(default=None, alias="maxChars")
    min_words_to_commit: Optional[int] = Field(default=None, alias="minWordsToCommit")


class SessionConfig:
    """Quản lý cấu hình động theo từng phiên làm việc."""

    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = {
            "source_lang": config.asr.language,
            "target_lang": config.translation.target_lang,
            "translation_model": getattr(config.translation, "base", "tencent"),
            "asr_engine": ModelRegistry.get_instance().get_active_model_key(),
            "vad_engine": config.vad.vad_engine,
            "vad_threshold": config.vad.threshold,
            "silence_duration_ms": config.vad.silence_duration_ms,
            "hangover_ms": config.vad.hangover_ms,
            "vad_enabled": config.vad.enabled,
            "min_words_to_commit": config.sentence.min_words_to_commit,
            "tts_enabled": config.tts.enabled,
            "tts_voice": config.tts.default_voice,
            "tts_speed": config.tts.speed,
            "tts_ref_audio": "",
            "tts_ref_text": "",
        }
        if initial_config:
            self._data.update(initial_config)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def update(self, new_data: Dict[str, Any]) -> None:
        self._data.update(new_data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


class SessionState:
    """Theo dõi trạng thái và các thành phần Pipeline cho 1 kết nối WebSocket."""

    def __init__(self, ws: Union[WebSocket, SafeWebSocketConnection]):
        if isinstance(ws, SafeWebSocketConnection):
            self.connection: SafeWebSocketConnection = ws
        else:
            self.connection: SafeWebSocketConnection = SafeWebSocketConnection(ws)

        self.session_id: str = str(uuid.uuid4())
        self.config: SessionConfig = SessionConfig()
        self.connected_at: float = time.time()
        self.chunk_index: int = 0

        # Các thành phần cốt lõi của Pipeline
        self.asr_engine: Optional[TranscribeEngine] = None
        self.vad_processor: Optional[VADProcessor] = None
        self.translation_queue: Optional[asyncio.Queue] = None
        self.tts_queue: Optional[asyncio.Queue] = None

    @property
    def ws(self) -> WebSocket:
        """Truy cập đối tượng WebSocket bên dưới."""
        return self.connection.raw_ws

    @ws.setter
    def ws(self, value: Union[WebSocket, SafeWebSocketConnection]) -> None:
        if isinstance(value, SafeWebSocketConnection):
            self.connection = value
        else:
            self.connection = SafeWebSocketConnection(value)

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        """Gửi JSON payload qua WebSocket an toàn."""
        return await self.connection.send_json(payload)

    async def send_text(self, text: str) -> bool:
        """Gửi raw text qua WebSocket an toàn."""
        return await self.connection.send_text(text)

    def init_components(self) -> None:
        """Khởi tạo toàn bộ các thành phần ASR, VAD và hàng đợi bất đồng bộ."""
        self.asr_engine = TranscribeEngine(session_id=self.session_id)

        self.translation_queue = asyncio.Queue(maxsize=4)
        self.tts_queue = asyncio.Queue(maxsize=4)

        self.vad_processor = VADProcessor(
            sample_rate=config.vad.sample_rate,
            vad_engine=self.config.get("vad_engine", config.vad.vad_engine),
            threshold=self.config["vad_threshold"],
            silence_duration_ms=self.config["silence_duration_ms"],
            hangover_ms=self.config["hangover_ms"],
            pre_speech_buffer_ms=config.vad.pre_speech_buffer_ms,
            enabled=self.config["vad_enabled"],
            on_speech_chunk=self.asr_engine.feed_audio,
            on_speech_start=self.asr_engine.on_speech_start,
            on_speech_end=self.asr_engine.on_speech_end,
        )

    def record_chunk(self, chunk_idx: Optional[int] = None) -> int:
        """Cập nhật và theo dõi số thứ tự chunk nhận được từ client."""
        if chunk_idx is not None:
            if self.chunk_index > 0 and chunk_idx > self.chunk_index + 1:
                dropped = chunk_idx - (self.chunk_index + 1)
                logger.warning(f"Session {self.session_id[:8]}: Phát hiện mất {dropped} chunk âm thanh!")
            self.chunk_index = chunk_idx
        else:
            self.chunk_index += 1
        return self.chunk_index

    def apply_config(self, raw_data: Dict[str, Any]) -> None:
        """Áp dụng và cập nhật cấu hình runtime ngay lập tức."""
        try:
            parsed = SessionConfigPayload.model_validate(raw_data)
        except Exception as e:
            logger.warning(f"Lỗi validate config payload: {e}")
            return

        updates: Dict[str, Any] = {}
        if parsed.source_lang is not None:
            updates["source_lang"] = parsed.source_lang
        if parsed.target_lang is not None:
            updates["target_lang"] = parsed.target_lang
        if parsed.translation_model is not None:
            updates["translation_model"] = parsed.translation_model

        # VAD settings
        if parsed.vad_engine is not None:
            updates["vad_engine"] = parsed.vad_engine
        vad_th = parsed.vad_threshold if parsed.vad_threshold is not None else parsed.threshold
        if vad_th is not None:
            updates["vad_threshold"] = float(vad_th)
        if parsed.silence_duration_ms is not None:
            updates["silence_duration_ms"] = int(parsed.silence_duration_ms)
        if parsed.hangover_ms is not None:
            updates["hangover_ms"] = int(parsed.hangover_ms)
        if parsed.vad_enabled is not None:
            updates["vad_enabled"] = bool(parsed.vad_enabled)

        # TTS settings
        if parsed.tts_enabled is not None:
            updates["tts_enabled"] = bool(parsed.tts_enabled)
        if parsed.tts_voice is not None:
            updates["tts_voice"] = parsed.tts_voice
        if parsed.tts_speed is not None:
            updates["tts_speed"] = float(parsed.tts_speed)
        if parsed.tts_ref_audio is not None:
            updates["tts_ref_audio"] = parsed.tts_ref_audio
        if parsed.tts_ref_text is not None:
            updates["tts_ref_text"] = parsed.tts_ref_text

        self.config.update(updates)

        # Hot-switch ASR Model nếu có yêu cầu
        target_asr = parsed.asr_model or parsed.asr_engine or parsed.model_id
        if target_asr:
            target_asr = target_asr.lower().strip()
            registry = ModelRegistry.get_instance()
            if target_asr in registry.models and target_asr != registry.get_active_model_key():
                registry.set_active_model_key(target_asr)
                self.config["asr_engine"] = target_asr
                if self.asr_engine:
                    self.asr_engine.model_key = target_asr
                    self.asr_engine.model_info = registry.get_model_info(target_asr) or {}
                logger.info(f"Session {self.session_id[:8]}: Đã chuyển ASR Model sang '{target_asr}'")

        if self.asr_engine and "source_lang" in self.config:
            self.asr_engine.set_language(self.config["source_lang"])

        # Cập nhật phân câu (Sentence segmentation)
        if self.asr_engine:
            sentence_updates = {}
            if parsed.split_on_stability is not None:
                sentence_updates["split_on_stability"] = parsed.split_on_stability
            elif parsed.enable_stability_split is not None:
                sentence_updates["split_on_stability"] = parsed.enable_stability_split
            if parsed.stability_duration_sec is not None:
                sentence_updates["stability_duration_sec"] = parsed.stability_duration_sec
            if parsed.max_duration_sec is not None:
                sentence_updates["max_duration_sec"] = parsed.max_duration_sec
            if parsed.max_chars is not None:
                sentence_updates["max_chars"] = parsed.max_chars
            if parsed.min_words_to_commit is not None:
                self.config["min_words_to_commit"] = parsed.min_words_to_commit
                sentence_updates["min_words_to_commit"] = parsed.min_words_to_commit

            if sentence_updates:
                self.asr_engine.update_sentence_config(**sentence_updates)

        if self.vad_processor:
            self.vad_processor.update_config(
                vad_engine=self.config.get("vad_engine"),
                threshold=self.config.get("vad_threshold"),
                silence_duration_ms=self.config.get("silence_duration_ms"),
                hangover_ms=self.config.get("hangover_ms"),
                enabled=self.config.get("vad_enabled"),
            )

    async def drain_queues(self, timeout: float = 0.2) -> None:
        """Cho phép các tác vụ translation và tts đang xử lý dở được hoàn tất nhanh chóng."""
        drain_tasks = []
        if self.translation_queue and not self.translation_queue.empty():
            drain_tasks.append(self.translation_queue.join())
        if self.tts_queue and not self.tts_queue.empty():
            drain_tasks.append(self.tts_queue.join())
        if drain_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*drain_tasks), timeout=timeout)
            except (asyncio.TimeoutError, Exception):
                pass

    async def cleanup(self) -> None:
        """Giải phóng triệt để tài nguyên phiên làm việc trong < 200ms."""
        t0 = time.perf_counter()
        if self.vad_processor:
            self.vad_processor.force_end()

        if self.asr_engine:
            await self.asr_engine.cleanup()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Session {self.session_id[:8]}: Fast Cleanup hoàn tất trong {elapsed_ms:.2f}ms")
