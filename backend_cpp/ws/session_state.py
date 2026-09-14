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
    pre_speech_buffer_ms: Optional[int] = Field(default=None, alias="preSpeechBufferMs")
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
    max_duration_grace_sec: Optional[float] = Field(default=None, alias="maxDurationGraceSec")
    max_duration_require_silence: Optional[bool] = Field(default=None, alias="maxDurationRequireSilence")
    boundary_candidate_silence_ms: Optional[int] = Field(default=None, alias="boundaryCandidateSilenceMs")
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
            "pre_speech_buffer_ms": config.vad.pre_speech_buffer_ms,
            "vad_enabled": True,
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


from collections import deque
import numpy as np

# Global session tracker for telemetry endpoints
ACTIVE_SESSIONS: Dict[str, "SessionState"] = {}
LAST_DISCONNECTED_REPORT: Dict[str, Any] = {}


class TransportTelemetry:
    """Accurately records audio transport statistics, duplicate/out-of-order rejections, jitter, and active drift."""

    def __init__(self):
        self.total_chunks_received: int = 0
        self.accepted_chunks: int = 0
        self.dropped_chunks: int = 0
        self.duplicate_chunks: int = 0
        self.out_of_order_chunks: int = 0
        self.last_chunk_index: Optional[int] = None

        # Jitter tracking: abs(actual_interval_ms - expected_cadence_ms)
        self.last_arrival_time: float = 0.0
        self.jitter_history: deque = deque(maxlen=1000)

        # Clock drift tracking (active continuous playback window)
        self.total_pcm_bytes: int = 0
        self.active_window_start_wall: float = 0.0
        self.total_active_wall_sec: float = 0.0
        self.is_window_active: bool = False

        # Resets count
        self.resets_count: int = 0

        # Paired Client Render Acknowledgement Tracking
        self.render_acks: Dict[str, Dict[str, Any]] = {}
        self.speech_offset_lags: deque = deque(maxlen=1000)
        self.client_render_costs: deque = deque(maxlen=1000)
        self.commit_to_ack_delays: deque = deque(maxlen=1000)

    def start_window(self, now: Optional[float] = None) -> None:
        if not self.is_window_active:
            self.active_window_start_wall = now if now is not None else time.perf_counter()
            self.is_window_active = True

    def pause_window(self, now: Optional[float] = None) -> None:
        if self.is_window_active:
            cur = now if now is not None else time.perf_counter()
            self.total_active_wall_sec += cur - self.active_window_start_wall
            self.is_window_active = False

    def record_chunk(self, chunk_idx: Optional[int], pcm_len: int, expected_cadence_ms: float = 64.0) -> bool:
        """Validate frame sequence, update jitter, and return True if accepted or False if rejected (pre-VAD drop)."""
        now = time.perf_counter()
        self.total_chunks_received += 1

        # Jitter calculation: | (now - last_arrival) * 1000.0 - expected_cadence_ms |
        if self.last_arrival_time > 0 and expected_cadence_ms > 0:
            interval_ms = (now - self.last_arrival_time) * 1000.0
            jitter_ms = abs(interval_ms - expected_cadence_ms)
            self.jitter_history.append(jitter_ms)
        self.last_arrival_time = now

        if not self.is_window_active:
            self.start_window(now=now)

        # Pre-VAD duplicate & out-of-order rejection
        if chunk_idx is not None:
            if self.last_chunk_index is not None:
                if chunk_idx == self.last_chunk_index:
                    self.duplicate_chunks += 1
                    perf.increment_counter("ws.audio_chunks_duplicate")
                    logger.warning(f"Rejected duplicate audio chunk #{chunk_idx} before VAD")
                    return False
                elif chunk_idx < self.last_chunk_index:
                    self.out_of_order_chunks += 1
                    perf.increment_counter("ws.audio_chunks_out_of_order")
                    logger.warning(f"Rejected out-of-order chunk #{chunk_idx} (last was #{self.last_chunk_index}) before VAD")
                    return False
                elif chunk_idx > self.last_chunk_index + 1:
                    dropped = chunk_idx - (self.last_chunk_index + 1)
                    self.dropped_chunks += dropped
                    perf.increment_counter("ws.audio_chunks_dropped", count=dropped)
                    logger.warning(f"Detected {dropped} dropped audio chunk(s) before VAD")

            self.last_chunk_index = chunk_idx

        self.accepted_chunks += 1
        self.total_pcm_bytes += pcm_len
        return True

    def record_render_ack(
        self,
        epoch: int,
        utterance_id: str,
        render_revision: int = 1,
        media_end_time: float = 0.0,
        video_current_time: float = 0.0,
        speech_offset_to_visible_lag_sec: float = 0.0,
        client_render_cost_ms: float = 0.0,
        asr_commit_wall_time: float = 0.0,
    ) -> bool:
        """Record paired browser render acknowledgement and compute latency percentiles."""
        ack_key = f"{epoch}_{utterance_id}_{render_revision}"
        if ack_key in self.render_acks:
            return False

        server_now = time.time()
        commit_to_ack_ms = (server_now - asr_commit_wall_time) * 1000.0 if asr_commit_wall_time > 0 else 0.0

        ack_data = {
            "epoch": epoch,
            "utterance_id": utterance_id,
            "render_revision": render_revision,
            "speech_offset_lag_sec": round(speech_offset_to_visible_lag_sec, 3),
            "client_render_cost_ms": round(client_render_cost_ms, 2),
            "commit_to_ack_delay_ms": round(commit_to_ack_ms, 1),
            "server_received_at": server_now,
        }
        self.render_acks[ack_key] = ack_data
        self.speech_offset_lags.append(speech_offset_to_visible_lag_sec)
        self.client_render_costs.append(client_render_cost_ms)
        if commit_to_ack_ms > 0:
            self.commit_to_ack_delays.append(commit_to_ack_ms)
        return True

    def get_summary(self) -> Dict[str, Any]:
        accepted_pcm_sec = self.total_pcm_bytes / 32000.0
        wall_sec = self.total_active_wall_sec
        if self.is_window_active and self.active_window_start_wall > 0:
            wall_sec += time.perf_counter() - self.active_window_start_wall

        drift_pct = 0.0
        if wall_sec > 1.0:
            drift_pct = abs(wall_sec - accepted_pcm_sec) / wall_sec * 100.0

        jitters = list(self.jitter_history)
        p50_jitter = float(np.percentile(jitters, 50)) if jitters else 0.0
        p95_jitter = float(np.percentile(jitters, 95)) if jitters else 0.0
        mean_jitter = float(np.mean(jitters)) if jitters else 0.0

        lags = list(self.speech_offset_lags)
        p50_lag = float(np.percentile(lags, 50)) if lags else 0.0
        p95_lag = float(np.percentile(lags, 95)) if lags else 0.0
        mean_lag = float(np.mean(lags)) if lags else 0.0
        max_lag = float(max(lags)) if lags else 0.0

        render_costs = list(self.client_render_costs)
        p50_cost = float(np.percentile(render_costs, 50)) if render_costs else 0.0
        p95_cost = float(np.percentile(render_costs, 95)) if render_costs else 0.0
        mean_cost = float(np.mean(render_costs)) if render_costs else 0.0

        ack_delays = list(self.commit_to_ack_delays)
        p50_ack = float(np.percentile(ack_delays, 50)) if ack_delays else 0.0
        p95_ack = float(np.percentile(ack_delays, 95)) if ack_delays else 0.0
        mean_ack = float(np.mean(ack_delays)) if ack_delays else 0.0
        max_ack = float(max(ack_delays)) if ack_delays else 0.0

        return {
            "total_chunks_received": self.total_chunks_received,
            "accepted_chunks": self.accepted_chunks,
            "dropped_chunks": self.dropped_chunks,
            "duplicate_chunks": self.duplicate_chunks,
            "out_of_order_chunks": self.out_of_order_chunks,
            "p50_jitter_ms": round(p50_jitter, 2),
            "p95_jitter_ms": round(p95_jitter, 2),
            "mean_jitter_ms": round(mean_jitter, 2),
            "accepted_pcm_duration_sec": round(accepted_pcm_sec, 3),
            "active_wall_duration_sec": round(wall_sec, 3),
            "clock_drift_pct": round(drift_pct, 3),
            "resets_count": self.resets_count,
            "paired_render_acks_count": len(self.render_acks),
            "speech_offset_to_visible_lag": {
                "p50_sec": round(p50_lag, 3),
                "p95_sec": round(p95_lag, 3),
                "mean_sec": round(mean_lag, 3),
                "max_sec": round(max_lag, 3),
            },
            "client_render_cost_ms": {
                "p50": round(p50_cost, 2),
                "p95": round(p95_cost, 2),
                "mean": round(mean_cost, 2),
            },
            "commit_to_ack_delay_ms": {
                "p50": round(p50_ack, 1),
                "p95": round(p95_ack, 1),
                "mean": round(mean_ack, 1),
                "max": round(max_ack, 1),
            },
        }


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
        self.current_epoch: int = 0
        self.transport_telemetry: TransportTelemetry = TransportTelemetry()

        # Pipeline components
        self.asr_engine: Optional[TranscribeEngine] = None
        self.vad_processor: Optional[VADProcessor] = None
        self.translation_queue: Optional[asyncio.Queue] = None
        self.tts_queue: Optional[asyncio.Queue] = None

        ACTIVE_SESSIONS[self.session_id] = self

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
            threshold=self.config.get("vad_threshold", config.vad.threshold),
            silence_duration_ms=self.config.get("silence_duration_ms", config.vad.silence_duration_ms),
            hangover_ms=self.config.get("hangover_ms", config.vad.hangover_ms),
            pre_speech_buffer_ms=self.config.get("pre_speech_buffer_ms", config.vad.pre_speech_buffer_ms),
            on_speech_chunk=self.asr_engine.feed_audio,
            on_speech_start=self.asr_engine.on_speech_start,
            on_speech_end=self.asr_engine.on_speech_end,
        )

    def record_chunk(self, chunk_idx: Optional[int] = None, pcm_len: int = 0, expected_cadence_ms: float = 64.0) -> bool:
        """Update transport metrics and enforce pre-VAD duplicate/out-of-order rejection."""
        if chunk_idx is not None:
            self.chunk_index = chunk_idx
        else:
            self.chunk_index += 1
        return self.transport_telemetry.record_chunk(chunk_idx, pcm_len, expected_cadence_ms)

    def handle_stream_reset(self, epoch: int, reason: str = "reset", media_time: float = 0.0) -> None:
        """Generation barrier: reset active ASR buffer, VAD, and drain queues."""
        if epoch <= self.current_epoch:
            logger.warning(
                f"[STREAM RESET IGNORED] Session {self.session_id}: Reset epoch {epoch} <= current {self.current_epoch}. Ignored."
            )
            return

        self.current_epoch = epoch
        self.transport_telemetry.resets_count += 1
        self.transport_telemetry.pause_window()

        if self.vad_processor:
            self.vad_processor.reset()
        if self.asr_engine:
            self.asr_engine.reset_stream(epoch=epoch)

        # Drain translation queue
        drained_trans = 0
        while self.translation_queue and not self.translation_queue.empty():
            try:
                self.translation_queue.get_nowait()
                self.translation_queue.task_done()
                drained_trans += 1
            except Exception:
                break

        # Drain TTS queue
        drained_tts = 0
        while self.tts_queue and not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
                drained_tts += 1
            except Exception:
                break

        logger.info(
            f"🔄 [STREAM RESET] Session {self.session_id}: Barrier epoch={epoch} (reason='{reason}', "
            f"media_time={media_time:.3f}s). Drained {drained_trans} translation & {drained_tts} TTS items."
        )

    def close(self) -> None:
        """Archive report and remove from active sessions."""
        global LAST_DISCONNECTED_REPORT
        summary = self.transport_telemetry.get_summary()
        summary["session_id"] = self.session_id
        summary["duration_sec"] = round(time.time() - self.connected_at, 2)
        LAST_DISCONNECTED_REPORT.clear()
        LAST_DISCONNECTED_REPORT.update(summary)
        ACTIVE_SESSIONS.pop(self.session_id, None)


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
            new_vad = parsed.vad_engine.lower().strip()
            if new_vad in ("none", "off", "disabled"):
                logger.warning(
                    f"Session {self.session_id}: VAD is mandatory and cannot be disabled. Ignoring vad_engine='{new_vad}'."
                )
            else:
                old_vad = self.config.get("vad_engine")
                self.config["vad_engine"] = new_vad
                if new_vad != old_vad:
                    profile = config.vad.get_engine_profile(new_vad)
                    if parsed.vad_threshold is None and parsed.threshold is None:
                        self.config["vad_threshold"] = profile.threshold
                    if parsed.silence_duration_ms is None:
                        self.config["silence_duration_ms"] = profile.silence_duration_ms
                    if parsed.hangover_ms is None:
                        self.config["hangover_ms"] = profile.hangover_ms
                    if parsed.pre_speech_buffer_ms is None:
                        self.config["pre_speech_buffer_ms"] = profile.pre_speech_buffer_ms

        if parsed.vad_threshold is not None:
            self.config["vad_threshold"] = parsed.vad_threshold
        elif parsed.threshold is not None:
            self.config["vad_threshold"] = parsed.threshold
        if parsed.silence_duration_ms is not None:
            self.config["silence_duration_ms"] = parsed.silence_duration_ms
        if parsed.hangover_ms is not None:
            self.config["hangover_ms"] = parsed.hangover_ms
        if parsed.pre_speech_buffer_ms is not None:
            self.config["pre_speech_buffer_ms"] = parsed.pre_speech_buffer_ms
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
            if parsed.max_duration_grace_sec is not None:
                sentence_updates["max_duration_grace_sec"] = parsed.max_duration_grace_sec
            if parsed.max_duration_require_silence is not None:
                sentence_updates["max_duration_require_silence"] = parsed.max_duration_require_silence
            if parsed.boundary_candidate_silence_ms is not None:
                sentence_updates["boundary_candidate_silence_ms"] = parsed.boundary_candidate_silence_ms
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
                pre_speech_buffer_ms=self.config.get("pre_speech_buffer_ms"),
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