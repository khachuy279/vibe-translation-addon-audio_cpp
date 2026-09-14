"""Session state per WebSocket connection for backend_audio_cpp."""

import asyncio
import io
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from backend_audio_cpp.config import config
from backend_audio_cpp.vad import (
    BaseVADEngine,
    BaseVADStreamState,
    VADConfig,
    VADFactory,
)
from backend_audio_cpp.asr.asr_engine import AudioCppASREngine
from backend_audio_cpp.asr.model_registry import ModelRegistry
from backend_audio_cpp.commit.sentence_committer import SentenceCommitter, count_content_tokens
from backend_audio_cpp.commit.sentence_config import SentenceConfig
from backend_audio_cpp.ws.connection import SafeWebSocketConnection
from backend_audio_cpp.ws.serializers import make_utterance_update_msg

logger = logging.getLogger("backend_audio_cpp.ws.session")


class SessionConfigPayload(BaseModel):
    """Aliased parser for extension session configuration messages."""
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
    min_words_to_commit: Optional[int] = Field(default=None, alias="minWordsToCommit")
    tts_enabled: Optional[bool] = Field(default=None, alias="ttsEnabled")
    tts_voice: Optional[str] = Field(default=None, alias="ttsVoice")
    tts_speed: Optional[float] = Field(default=None, alias="ttsSpeed")
    translation_model: Optional[str] = Field(default=None, alias="translationModel")


class SessionConfig:
    """Manages session runtime configurations with defaults."""

    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = {
            "source_lang": config.asr.language,
            "target_lang": config.translation.target_lang,
            "translation_model": config.translation.base,
            "asr_engine": ModelRegistry.get_instance().get_active_model_key(),
            "vad_engine": config.vad.vad_engine,
            "vad_threshold": config.vad.threshold,
            "silence_duration_ms": config.vad.silence_duration_ms,
            "min_words_to_commit": config.sentence.min_words_to_commit,
            "tts_enabled": config.tts.enabled,
            "tts_voice": config.tts.default_voice,
            "tts_speed": config.tts.speed,
        }
        if initial:
            self._data.update(initial)

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


class SessionState:
    """Tracks state, isolated VAD stream, ASR queues, and commit logic for a WebSocket connection."""

    def __init__(self, ws: SafeWebSocketConnection):
        self.connection: SafeWebSocketConnection = ws
        self.session_id: str = str(uuid.uuid4())
        self.config: SessionConfig = SessionConfig()
        self.connected_at: float = time.time()
        self.chunk_index: int = 0

        # VAD Engine & Stream State via VADFactory
        self.vad_cfg = VADConfig(
            threshold=float(self.config["vad_threshold"]),
            min_silence_duration_sec=float(self.config["silence_duration_ms"]) / 1000.0,
        )
        self.vad_engine: BaseVADEngine = VADFactory.get_engine(self.config["vad_engine"], self.vad_cfg)
        self.vad_state: BaseVADStreamState = self.vad_engine.create_state()

        # Audio buffering & Seek tracking
        self._frame_buffer: bytearray = bytearray()
        self._speech_buffer: bytearray = bytearray()
        self._active_utterance_id: int = 1
        self._last_partial_poll_time: float = 0.0
        self._last_partial_text: str = ""
        self._last_capture_ts: float = 0.0
        self._last_chunk_idx: Optional[int] = None
        self._stream_time_sec: float = 0.0
        self._last_feed_wall_time: float = 0.0

        # ASR Engine & Commit
        self.asr_engine: AudioCppASREngine = AudioCppASREngine.get_instance()
        self.committer: SentenceCommitter = SentenceCommitter(
            SentenceConfig(min_words_to_commit=int(self.config["min_words_to_commit"]))
        )

        # Queues for decoupled workers
        self.translation_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
        self.tts_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
        self.loop = asyncio.get_event_loop()

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        return await self.connection.send_json(payload)

    def reset_vad_and_buffers(self, reason: str = "seek") -> None:
        """Reset VAD stream state and clear frame/speech buffers (called on video seek or stream reset)."""
        logger.info(f"⏩ [SESSION RESET] Session {self.session_id}: Resetting VAD state & audio buffers (reason: {reason})")
        self.vad_state.reset()
        self._frame_buffer.clear()
        self._speech_buffer.clear()
        self._last_partial_text = ""
        self._last_capture_ts = 0.0
        self._last_chunk_idx = None
        self._stream_time_sec = 0.0
        self._last_feed_wall_time = 0.0

    def apply_config(self, new_config: Dict[str, Any]) -> None:
        """Dynamically update parameters on the fly."""
        try:
            parsed = SessionConfigPayload.model_validate(new_config)
        except Exception as e:
            logger.warning(f"Session {self.session_id}: Error parsing config: {e}")
            return

        if parsed.source_lang is not None:
            self.config["source_lang"] = parsed.source_lang
        if parsed.target_lang is not None:
            self.config["target_lang"] = parsed.target_lang
        if parsed.vad_threshold is not None:
            self.config["vad_threshold"] = parsed.vad_threshold
            self.vad_cfg.threshold = parsed.vad_threshold
        elif parsed.threshold is not None:
            self.config["vad_threshold"] = parsed.threshold
            self.vad_cfg.threshold = parsed.threshold
        if parsed.silence_duration_ms is not None:
            self.config["silence_duration_ms"] = parsed.silence_duration_ms
            self.vad_cfg.min_silence_duration_sec = parsed.silence_duration_ms / 1000.0
        if parsed.vad_engine is not None and parsed.vad_engine != self.config["vad_engine"]:
            self.config["vad_engine"] = parsed.vad_engine
            self.vad_engine = VADFactory.get_engine(parsed.vad_engine, self.vad_cfg)
            self.vad_state = self.vad_engine.create_state()
            logger.info(f"Session {self.session_id}: Switched VAD Engine live -> '{parsed.vad_engine}'")
        if parsed.min_words_to_commit is not None:
            self.config["min_words_to_commit"] = parsed.min_words_to_commit
            self.committer.config.min_words_to_commit = parsed.min_words_to_commit
        if parsed.tts_enabled is not None:
            self.config["tts_enabled"] = parsed.tts_enabled
        if parsed.tts_voice is not None:
            self.config["tts_voice"] = parsed.tts_voice
        if parsed.tts_speed is not None:
            self.config["tts_speed"] = parsed.tts_speed
        if parsed.asr_engine is not None:
            self.config["asr_engine"] = parsed.asr_engine
            try:
                self.asr_engine.switch_model(parsed.asr_engine)
            except Exception as e:
                logger.error(f"Error switching ASR engine in session: {e}")
        if parsed.translation_model is not None:
            self.config["translation_model"] = parsed.translation_model

        logger.info(
            f"Session {self.session_id}: Live config applied -> "
            f"ASR={self.config['asr_engine']} | VAD={self.config['vad_engine']} "
            f"(thresh={self.config['vad_threshold']}, silence={self.config['silence_duration_ms']}ms) | "
            f"min_words={self.config['min_words_to_commit']} | "
            f"lang={self.config['source_lang']}->{self.config['target_lang']} | "
            f"TTS={self.config['tts_enabled']} ({self.config['tts_voice']})"
        )

    def feed_pcm(self, pcm_bytes: bytes, capture_ts: float, chunk_idx: Optional[int]) -> None:
        """Feed incoming PCM chunks into VAD frame slicer and accumulate speech."""
        now_wall = time.perf_counter()

        # 1. Wall-clock gap check (client paused/seeked/buffered for > 0.6s)
        if self._last_feed_wall_time > 0.0 and (now_wall - self._last_feed_wall_time) > 0.6:
            logger.info(
                f"⏩ [PAUSE/SEEK GAP] Gap of {now_wall - self._last_feed_wall_time:.2f}s "
                f"between audio feeds. Resetting VAD stream state."
            )
            self.reset_vad_and_buffers(f"wall_clock_gap_{now_wall - self._last_feed_wall_time:.2f}s")

        self._last_feed_wall_time = now_wall

        # 2. Auto-detect Seek / Audio Discontinuity via capture_ts or chunk_idx
        if capture_ts > 0.0 and self._last_capture_ts > 0.0:
            if capture_ts < (self._last_capture_ts - 0.5):
                self.reset_vad_and_buffers(f"backward seek ({self._last_capture_ts:.2f}s -> {capture_ts:.2f}s)")
            elif (capture_ts - self._last_capture_ts) > 3.0:
                self.reset_vad_and_buffers(f"forward gap ({self._last_capture_ts:.2f}s -> {capture_ts:.2f}s)")

        if capture_ts > 0.0:
            self._last_capture_ts = capture_ts

        if chunk_idx is not None:
            if self._last_chunk_idx is not None and (chunk_idx < self._last_chunk_idx - 5 or (self._last_chunk_idx > 10 and chunk_idx <= 2)):
                self.reset_vad_and_buffers(f"chunk index reset ({self._last_chunk_idx} -> {chunk_idx})")
            self._last_chunk_idx = chunk_idx
            self.chunk_index = chunk_idx
        else:
            self.chunk_index += 1

        self._frame_buffer.extend(pcm_bytes)
        bytes_per_frame = 512 * 2  # 1024 bytes = 512 samples @ 16-bit
        frame_duration_sec = 512 / 16000.0  # 0.032s per 32ms frame

        while len(self._frame_buffer) >= bytes_per_frame:
            frame = bytes(self._frame_buffer[:bytes_per_frame])
            del self._frame_buffer[:bytes_per_frame]

            self._stream_time_sec += frame_duration_sec
            frame_ts = self._stream_time_sec

            res = self.vad_engine.process_frame(frame, frame_ts, self.vad_state)

            if res.event == "SPEECH_START":
                self._speech_buffer.clear()
                self._speech_buffer.extend(frame)
                self._active_utterance_id = res.utterance_id or (self._active_utterance_id + 1)
                self._last_partial_poll_time = time.perf_counter()
                self._last_partial_text = ""

            elif res.is_speech:
                self._speech_buffer.extend(frame)
                # Check for partial preview poll
                now = time.perf_counter()
                if (now - self._last_partial_poll_time) >= config.asr.poll_interval_sec:
                    self._last_partial_poll_time = now
                    # Trigger async partial transcription
                    current_pcm = bytes(self._speech_buffer)
                    utt_id = str(self._active_utterance_id)
                    asyncio.run_coroutine_threadsafe(
                        self._async_partial_transcribe(current_pcm, utt_id), self.loop
                    )

            elif res.event == "SPEECH_END":
                self._speech_buffer.extend(frame)
                final_pcm = bytes(self._speech_buffer)
                utt_id = str(res.utterance_id or self._active_utterance_id)
                self._speech_buffer.clear()
                last_partial = self._last_partial_text
                self._last_partial_text = ""
                # Commit final utterance with last partial preview attached for recovery
                asyncio.run_coroutine_threadsafe(
                    self._async_commit_utterance(
                        final_pcm, utt_id, res.reason or "VAD_SILENCE", last_partial=last_partial
                    ),
                    self.loop,
                )

    async def _async_partial_transcribe(self, pcm_bytes: bytes, utt_id: str) -> None:
        """Run partial ASR on speech segment and emit utterance_update (is_final=False)."""
        if len(pcm_bytes) < 3200:  # < 0.1s
            return
        try:
            text, _ = await asyncio.to_thread(
                self.asr_engine.transcribe_chunk,
                pcm_bytes,
                utt_id,
                True,  # is_partial
                self.config.get("asr_engine"),
            )
            if text and text != self._last_partial_text:
                self._last_partial_text = text
                msg = make_utterance_update_msg(
                    utt_id=utt_id,
                    text=text,
                    translated="",
                    is_final=False,
                    stable_text=text,
                    unstable_text="",
                )
                await self.send_json(msg)
        except Exception as e:
            logger.debug(f"Partial ASR error: {e}")

    async def _async_commit_utterance(
        self,
        pcm_bytes: bytes,
        utt_id: str,
        reason: str,
        last_partial: str = "",
    ) -> None:
        """Transcribe final speech chunk, check commit constraints, and enqueue for Translation."""
        if len(pcm_bytes) < 3200 and not last_partial:
            return
        try:
            text = ""
            if len(pcm_bytes) >= 3200:
                text, _ = await asyncio.to_thread(
                    self.asr_engine.transcribe_chunk,
                    pcm_bytes,
                    utt_id,
                    False,  # is_final
                    self.config.get("asr_engine"),
                )

            # Text Recovery Feature: If COMMIT text is shorter than last PARTIAL preview, recover last PARTIAL
            if last_partial:
                final_tokens = count_content_tokens(text)
                partial_tokens = count_content_tokens(last_partial)
                if partial_tokens > final_tokens:
                    logger.info(
                        f"🔄 [RECOVERY] [utt_{utt_id}] Final COMMIT ('{text}', {final_tokens} tokens) "
                        f"is shorter than last PARTIAL ('{last_partial}', {partial_tokens} tokens). "
                        f"Recovered text from last PARTIAL preview!"
                    )
                    text = last_partial

            if not text:
                return

            # Check min words / tokens condition (supporting both Latin words and CJK characters)
            min_words = int(self.config.get("min_words_to_commit") or 2)
            token_count = count_content_tokens(text)
            if token_count < min_words:
                logger.info(
                    f"🚫 [COMMIT FILTER] Dropped short utterance ({token_count} < {min_words} tokens): '{text}'"
                )
                return

            logger.info(f'[COMMIT] [utt_{utt_id}] >> COMMITTED: "{text}" | Reason: {reason}')

            # Emit final utterance update
            msg = make_utterance_update_msg(
                utt_id=utt_id,
                text=text,
                translated="...",
                is_final=True,
                stable_text=text,
            )
            await self.send_json(msg)

            # Enqueue for Translation worker
            try:
                self.translation_queue.put_nowait({
                    "utterance_id": utt_id,
                    "text": text,
                    "source_lang": self.config.get("source_lang", "auto"),
                    "target_lang": self.config.get("target_lang", "vi"),
                    "_queued_at": time.perf_counter(),
                })
            except asyncio.QueueFull:
                logger.warning(f"Session {self.session_id}: Translation queue full")

        except Exception as e:
            logger.error(f"Commit utterance error: {e}", exc_info=True)
