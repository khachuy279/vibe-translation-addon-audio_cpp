"""Session state per WebSocket connection in backend_cpp."""

import asyncio
import logging
import time
import uuid
from typing import Optional, Dict, Any, Union
from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict, Field

from backend_cpp.config import config
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.asr.model_registry import ModelRegistry
from backend_cpp.vad.vad_processor import VADProcessor
from backend_cpp.ws.connection import SafeWebSocketConnection
from backend_cpp.utils.perf_profiler import perf

logger = logging.getLogger(__name__)


class SessionConfigPayload(BaseModel):
    """Strongly-typed, aliased parser for extension session configuration messages."""
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

    # Sentence segmentation fields
    split_on_stability: Optional[bool] = Field(default=None, alias="splitOnStability")
    enable_stability_split: Optional[bool] = None
    stability_duration_sec: Optional[float] = Field(default=None, alias="stabilityDurationSec")
    max_duration_sec: Optional[float] = Field(default=None, alias="maxDurationSec")
    max_chars: Optional[int] = Field(default=None, alias="maxChars")
    min_words_to_commit: Optional[int] = Field(default=None, alias="minWordsToCommit")


class SessionConfig:
    """Manages session runtime configurations with defaults and dict-like access."""

    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = {
            "source_lang": config.asr.language,
            "target_lang": config.translation.target_lang,
            "translation_model": getattr(config.translation, "base", "gemmax"),
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
    """Tracks state and components for a single WebSocket connection."""

    def __init__(self, ws: Union[WebSocket, SafeWebSocketConnection]):
        if isinstance(ws, SafeWebSocketConnection):
            self.connection: SafeWebSocketConnection = ws
        else:
            self.connection: SafeWebSocketConnection = SafeWebSocketConnection(ws)

        self.session_id: str = str(uuid.uuid4())
        self.config: SessionConfig = SessionConfig()
        self.connected_at: float = time.time()
        self.chunk_index: int = 0

        # Pipeline components
        self.asr_engine: Optional[TranscribeEngine] = None
        self.vad_processor: Optional[VADProcessor] = None
        self.translation_queue: Optional[asyncio.Queue] = None
        self.tts_queue: Optional[asyncio.Queue] = None

    @property
    def ws(self) -> WebSocket:
        """Backward-compatible access to underlying WebSocket instance."""
        return self.connection.raw_ws

    @ws.setter
    def ws(self, value: Union[WebSocket, SafeWebSocketConnection]) -> None:
        """Allow re-assigning mock ws in unit tests."""
        if isinstance(value, SafeWebSocketConnection):
            self.connection = value
        else:
            self.connection = SafeWebSocketConnection(value)

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        """Thread-safe & task-safe send JSON through connection."""
        return await self.connection.send_json(payload)

    async def send_text(self, text: str) -> bool:
        """Thread-safe & task-safe send raw text through connection."""
        return await self.connection.send_text(text)

    def init_components(self) -> None:
        """Initialize ASR engine, VAD processor, and decoupled background queues."""
        self.asr_engine = TranscribeEngine(session_id=self.session_id)

        self.translation_queue = asyncio.Queue(maxsize=20)
        self.tts_queue = asyncio.Queue(maxsize=10)

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
        """Update and validate chunk index sequence, detecting missing chunks."""
        if chunk_idx is not None:
            if self.chunk_index > 0 and chunk_idx > self.chunk_index + 1:
                dropped = chunk_idx - (self.chunk_index + 1)
                perf.increment_counter("ws.audio_chunks_dropped", count=dropped)
                logger.warning(
                    f"Session {self.session_id}: Detected {dropped} dropped audio chunk(s) "
                    f"(expected {self.chunk_index + 1}, got {chunk_idx})"
                )
            self.chunk_index = chunk_idx
        else:
            self.chunk_index += 1
        return self.chunk_index

    def apply_config(self, new_config: Dict[str, Any]) -> None:
        """Dynamically update parameters sent by browser extension with validated Pydantic model."""
        try:
            parsed = SessionConfigPayload.model_validate(new_config)
        except Exception as e:
            logger.warning(f"Session {self.session_id}: Error parsing config payload: {e}")
            return

        # VAD & pipeline config updates
        if parsed.source_lang is not None:
            self.config["source_lang"] = parsed.source_lang
        if parsed.target_lang is not None:
            self.config["target_lang"] = parsed.target_lang
        if parsed.vad_engine is not None:
            self.config["vad_engine"] = parsed.vad_engine
        if parsed.vad_threshold is not None:
            self.config["vad_threshold"] = parsed.vad_threshold
        elif parsed.threshold is not None:
            self.config["vad_threshold"] = parsed.threshold
        if parsed.silence_duration_ms is not None:
            self.config["silence_duration_ms"] = parsed.silence_duration_ms
        if parsed.hangover_ms is not None:
            self.config["hangover_ms"] = parsed.hangover_ms
        if parsed.vad_enabled is not None:
            self.config["vad_enabled"] = parsed.vad_enabled
        if parsed.tts_enabled is not None:
            self.config["tts_enabled"] = parsed.tts_enabled
        if parsed.tts_voice is not None:
            self.config["tts_voice"] = parsed.tts_voice
        if parsed.tts_speed is not None:
            self.config["tts_speed"] = parsed.tts_speed
        if parsed.translation_model is not None:
            tm = parsed.translation_model.lower().strip()
            from backend_cpp.translation.model_registry import TranslationModelRegistry
            tm = TranslationModelRegistry.get_instance().resolve_key(tm)
            if tm != self.config.get("translation_model"):
                if self.chunk_index > 0:
                    logger.warning(
                        f"Session {self.session_id}: Cannot switch translation model while capturing audio! Ignored."
                    )
                else:
                    self.config["translation_model"] = tm
                    # LocalGGUFTranslator is an app-wide singleton designed for single-tenant desktop usage.
                    # Switching here reconfigures the underlying engine globally for the active session.
                    config.translation.base = tm
                    try:
                        from backend_cpp.config import TranslationConfig
                        from backend_cpp.translation.local_translator import LocalGGUFTranslator
                        new_trans_cfg = TranslationConfig(
                            base=tm,
                            target_lang=self.config.get("target_lang", "vi"),
                        )
                        LocalGGUFTranslator.get_instance(new_trans_cfg)
                        logger.info(f"Session {self.session_id}: Dynamically switched translation model to '{tm}'")
                    except Exception as e:
                        logger.error(f"Session {self.session_id}: Error reconfiguring translation model: {e}")

        # ASR Model dynamic switch
        target_asr = parsed.model_id or parsed.asr_engine or parsed.asr_model
        if target_asr:
            target_asr = target_asr.lower().strip()
            registry = ModelRegistry.get_instance()
            if target_asr in registry.models and target_asr != registry.get_active_model_key():
                registry.set_active_model_key(target_asr)
                self.config["asr_engine"] = target_asr
                if self.asr_engine:
                    self.asr_engine.model_key = target_asr
                    self.asr_engine.model_info = registry.get_model_info(target_asr) or {}
                logger.info(f"Session {self.session_id}: Switched ASR model to '{target_asr}' via WS config")

        if self.asr_engine and "source_lang" in self.config:
            self.asr_engine.set_language(self.config["source_lang"])

        # Sentence segmentation updates
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

    async def drain_queues(self, timeout: float = 0.5) -> None:
        """Allow pending translation and TTS queue items to drain before shutting down."""
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
        """Release session resources cleanly."""
        if self.vad_processor:
            self.vad_processor.force_end()

        if self.asr_engine:
            await self.asr_engine.cleanup()
