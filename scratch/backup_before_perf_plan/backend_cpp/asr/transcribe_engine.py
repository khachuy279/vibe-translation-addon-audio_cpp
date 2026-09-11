"""TranscribeCppEngine connecting transcribe.cpp models with streaming pipeline."""

from abc import ABCMeta
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import math
import re
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

import numpy as np
import transcribe_cpp

from backend_cpp.asr.audio_buffer import (
    AudioBufferManager,
    AudioSnapshot,
    VAD_STATE_NON_SPEECH,
    VAD_STATE_SPEECH,
    VAD_STATE_PRE_ROLL,
)
from backend_cpp.asr.speech_normalizer import SpeechNormalizer, NormalizationResult
from backend_cpp.asr.base_engine import BaseASREngine
from backend_cpp.asr.family_adapter import build_family_options, normalize_language_for_family
from backend_cpp.asr.constants import (
    DEFAULT_SAMPLE_RATE,
    MIN_INFERENCE_AUDIO_SEC,
    PREFIX_STRIP_WINDOW_SEC,
    RECENT_COMMITS_CACHE_SEC,
)
from backend_cpp.asr.dedup import CommitDeduplicator
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.model_registry import ModelRegistry
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter, count_content_tokens
from backend_cpp.config import config, SentenceConfig
from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths
from backend_cpp.utils.perf_profiler import perf
from backend_cpp.utils.audio_dumper import dump_vad_utterance, dump_asr_input


setup_cuda_dll_paths()
logger = logging.getLogger(__name__)

_RE_SPECIAL_TAGS = re.compile(r"<\|.*?\|>|<[^>]+>")
_SENTINEL = object()
_SYNC_COMMIT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="commit_sync")


def _level_db(val: float) -> float:
    """Convert linear audio amplitude/RMS to decibels (dBFS)."""
    return 20.0 * math.log10(max(float(val), 1e-12))


@dataclass
class ASREngineConfig:
    """Configurable options for TranscribeEngine enabling Dependency Injection."""
    model_key: Optional[str] = None
    min_transcribe_sec: float = 0.6
    poll_interval_ms: int = 350
    threads: int = 4
    language: str = "auto"
    backend: str = "auto"
    sentence_config: Optional[SentenceConfig] = None


def clean_transcript_text(raw_text: str) -> str:
    """Clean model special tokens and formatting tags."""
    if not raw_text:
        return ""
    text = _RE_SPECIAL_TAGS.sub("", raw_text)
    text = re.sub(r"(?m)^(?:system|user|assistant|language\s+\w+)\s*[:]?\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def check_model_supports_streaming(model_key: Optional[str] = None) -> bool:
    """Check if model supports native streaming (session.stream) before or without loading it."""
    registry = ModelRegistry.get_instance()
    key = model_key or registry.get_active_model_key()
    return ASRModelManager.supports_streaming(key)


class _TranscribeEngineMeta(ABCMeta):
    """Metaclass forwarding shared class-level model/session state directly to ASRModelManager.

    Ensures a Single Source of Truth while preserving full backward compatibility
    for external code and unit tests referencing TranscribeEngine._shared_*.
    """

    @property
    def _shared_model(cls):
        return ASRModelManager._shared_model

    @_shared_model.setter
    def _shared_model(cls, val):
        ASRModelManager._shared_model = val

    @property
    def _shared_model_key(cls):
        return ASRModelManager._shared_model_key

    @_shared_model_key.setter
    def _shared_model_key(cls, val):
        ASRModelManager._shared_model_key = val

    @property
    def _shared_session(cls):
        return ASRModelManager._shared_session

    @_shared_session.setter
    def _shared_session(cls, val):
        ASRModelManager._shared_session = val

    @property
    def _shared_supports_streaming(cls):
        return ASRModelManager._shared_supports_streaming

    @_shared_supports_streaming.setter
    def _shared_supports_streaming(cls, val):
        ASRModelManager._shared_supports_streaming = val

    @property
    def _shared_lock(cls):
        return ASRModelManager._shared_lock

    @property
    def _shared_infer_lock(cls):
        return ASRModelManager._shared_infer_lock

    @property
    def _commit_lock(cls):
        return ASRModelManager._commit_lock


class TranscribeEngine(BaseASREngine, metaclass=_TranscribeEngineMeta):
    """Streaming ASR Engine powered by transcribe.cpp and GGML.

    Architecture & Concurrency Design:
    -----------------------------------
    1. Single-Tenant Desktop Extension Worker:
       - Designed for single-user desktop streaming from browser extensions (Firefox/Chrome).
       - Shares one loaded GGML model instance in GPU VRAM to minimize memory footprint (~1.5GB).

    2. Lock Hierarchy & Concurrency Boundaries:
       - `_shared_lock`: Protects model lifecycle transitions (load, swap, unload).
       - `_shared_infer_lock`: Serializes C++ inference calls on the active session because
         transcribe_cpp session structures are stateful and non-reentrant.
         * Hierarchy: Always acquire `_shared_lock` BEFORE `_shared_infer_lock`.
         * Never acquire `_shared_lock` while holding `_shared_infer_lock`.
       - `_state_lock`: Instance-level lock guarding utterance ID, partial preview text, and speech flags.
         Kept strictly isolated from inference locks to prevent priority inversion.

    3. Sub-Component Responsibilities:
       - Lifecycle Manager: `prewarm()`, `_ensure_shared_model()`, `unload_shared_model()`
       - Audio Buffer Gateway: `feed_audio()`, `_audio_buffer_mgr` (AudioBufferManager)
       - Inference Runner: `_run_inference()`, `_commit_async()`, `_commit_sync()`
       - Preview Poller: `_partial_preview_poller()` (background polling task)
       - Token Queue: `stream_tokens()`, `_push_message()`
    """

    _commit_waiting: int = 0

    @classmethod
    def shutdown_executors(cls, wait: bool = False) -> None:
        """Shut down the module-level commit thread pool executor."""
        global _SYNC_COMMIT_EXECUTOR
        try:
            _SYNC_COMMIT_EXECUTOR.shutdown(wait=wait, cancel_futures=True)
        except Exception as e:
            logger.debug(f"Error shutting down _SYNC_COMMIT_EXECUTOR: {e}")

    @classmethod
    def unload_shared_model(cls) -> None:
        """Release shared model and cached session from VRAM/RAM via ASRModelManager."""
        ASRModelManager.unload_shared_model()

    def prewarm(self) -> None:
        """Public API to pre-warm model and inference session."""
        self._ensure_shared_model()

    @property
    def supports_streaming(self) -> bool:
        """Check if active model supports native streaming (session.stream)."""
        if ASRModelManager._shared_model is not None and ASRModelManager._shared_model_key == self.model_key:
            return ASRModelManager._shared_supports_streaming
        return self._model_manager.supports_streaming(self.model_key)

    def __init__(
        self,
        model_key: Optional[str] = None,
        engine_config: Optional[ASREngineConfig] = None,
        model_manager: Optional[ASRModelManager] = None,
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id or "default"
        self.registry = ModelRegistry.get_instance()
        self._model_manager = model_manager or ASRModelManager(self.registry)


        active_key = (
            model_key
            or (engine_config.model_key if engine_config else None)
            or self.registry.get_active_model_key()
            or getattr(config.asr, "active_model", None)
        )

        if engine_config is not None:
            self._engine_config = engine_config
        else:
            self._engine_config = ASREngineConfig(
                model_key=active_key,
                min_transcribe_sec=getattr(config.asr, "min_transcribe_sec", 0.6),
                poll_interval_ms=getattr(config.asr, "poll_interval_ms", 350),
                threads=getattr(config.asr, "threads", 4),
                language=getattr(config.asr, "language", "auto"),
                backend=getattr(config.asr, "backend", "auto"),
                sentence_config=config.sentence.model_copy() if hasattr(config, "sentence") else None,
            )

        self.model_key = active_key
        self.model_info = self.registry.get_model_info(self.model_key) or {}

        # Audio buffer manager (float32 PCM @ 16kHz)
        self._audio_buffer_mgr = AudioBufferManager(sample_rate=DEFAULT_SAMPLE_RATE)
        # Session-level speech normalizer (maintains continuous gain state across utterances)
        self._normalizer = SpeechNormalizer()

        # Engine speech and utterance state (protected by _state_lock)
        self._state_lock = threading.Lock()
        self._is_speech_active: bool = False
        self._current_utterance_id: str = str(uuid.uuid4())
        self._language: str = self._engine_config.language

        # Sentence segmentation bounds & toggles
        self.sentence_config: SentenceConfig = (
            self._engine_config.sentence_config
            or (config.sentence.model_copy() if hasattr(config, "sentence") else SentenceConfig())
        )
        self._segmenter = SentenceSegmenter(
            max_chars=self.sentence_config.max_chars,
            max_duration_sec=self.sentence_config.max_duration_sec,
            min_words_to_commit=self.sentence_config.min_words_to_commit,
            split_on_stability=self.sentence_config.split_on_stability,
            stability_duration_sec=self.sentence_config.stability_duration_sec,
            stability_threshold_polls=self.sentence_config.stability_threshold_polls,
        )

        # Polling & queue
        self._min_transcribe_sec: float = self._engine_config.min_transcribe_sec
        self._poll_interval_sec: float = self._engine_config.poll_interval_ms / 1000.0
        self._last_partial_text: str = ""
        self._last_partial_samples: int = 0
        self._last_polled_samples: int = 0

        # Deduplication and prefix tracking
        self._deduplicator = CommitDeduplicator(cache_ttl_sec=RECENT_COMMITS_CACHE_SEC)
        self._last_committed_head: str = ""
        self._last_committed_head_time: float = 0.0

        self._token_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._running: bool = True

    @property
    def _buffer_duration_sec(self) -> float:
        return self._audio_buffer_mgr.duration_sec

    def update_sentence_config(
        self,
        split_on_stability: Optional[bool] = None,
        stability_duration_sec: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
        max_chars: Optional[int] = None,
        min_words_to_commit: Optional[int] = None,
    ) -> None:
        """Update sentence segmentation configuration dynamically."""
        if split_on_stability is not None:
            self.sentence_config.split_on_stability = split_on_stability
            self._segmenter.split_on_stability = split_on_stability
        if stability_duration_sec is not None:
            self.sentence_config.stability_duration_sec = stability_duration_sec
            self._segmenter.stability_duration_sec = stability_duration_sec
        if max_duration_sec is not None:
            self.sentence_config.max_duration_sec = max_duration_sec
            self._segmenter.max_duration_sec = max_duration_sec
        if max_chars is not None:
            self.sentence_config.max_chars = max_chars
            self._segmenter.max_chars = max_chars
        if min_words_to_commit is not None:
            self.sentence_config.min_words_to_commit = min_words_to_commit
            self._segmenter.min_words_to_commit = min_words_to_commit

    def _get_queue(self) -> asyncio.Queue:
        if self._token_queue is None:
            self._token_queue = asyncio.Queue()
        return self._token_queue

    def _ensure_shared_model(self) -> transcribe_cpp.Model:
        """Load and cache the transcribe.cpp model via ASRModelManager."""
        return self._model_manager.ensure_model(self.model_key, backend=self._engine_config.backend)

    def _ensure_session(self, model: transcribe_cpp.Model) -> transcribe_cpp.Session:
        """Reuse cached session or instantiate a new session for inference via ASRModelManager."""
        return self._model_manager.ensure_session(model, threads=self._engine_config.threads)

    def set_language(self, language: str) -> None:
        self._language = language or "auto"

    @staticmethod
    def normalize_speech(
        pcm: np.ndarray,
        target_rms: Optional[float] = None,
        target_peak: Optional[float] = None,
        max_gain: Optional[float] = None,
        min_rms_to_boost: Optional[float] = None,
        min_rms: float = 1e-4,
        is_commit: bool = False,
        utt_id: Optional[str] = None,
        frame_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Dynamic RMS normalization + Peak Limiter optimized for ASR input.

        Stateless wrapper around SpeechNormalizer for backward compatibility and testing.
        """
        if getattr(config.debug, "bypass_speech_normalization", False) or not getattr(config.asr, "normalize_speech", True):
            return np.array(pcm, dtype=np.float32, copy=True)

        if len(pcm) == 0:
            return pcm

        knee_end = min_rms_to_boost if min_rms_to_boost is not None else getattr(config.asr, "normalize_knee_end", 0.050)
        knee_start = min(knee_end * 0.5, getattr(config.asr, "normalize_knee_start", 0.025))

        normalizer = SpeechNormalizer(
            target_rms=target_rms,
            target_peak=target_peak,
            max_gain=max_gain,
            knee_start=knee_start,
            knee_end=knee_end,
            gain_smoothing=False,
        )
        res = normalizer.process(pcm, frame_state=frame_state, use_smoothing=False)
        return res.pcm

    def _run_inference(
        self,
        pcm_float32: np.ndarray,
        frame_state: Optional[np.ndarray] = None,
        is_commit: bool = False,
        utt_id: Optional[str] = None,
    ) -> str:
        """Run transcribe.cpp inference in a worker thread under infer_lock with session reuse."""
        if len(pcm_float32) < int(MIN_INFERENCE_AUDIO_SEC * DEFAULT_SAMPLE_RATE):
            return ""

        # Commit Priority Check: If this is a preview and a commit is waiting, yield immediately
        if not is_commit:
            with TranscribeEngine._commit_lock:
                if TranscribeEngine._commit_waiting > 0:
                    logger.debug("⚡ [ASR PREVIEW] Skipped preview inference because commit is waiting")
                    perf.increment_counter("asr.preview_lock_skipped")
                    return ""

        eff_utt_id = utt_id or self._current_utterance_id

        # Dynamic speech normalization via session SpeechNormalizer (VAD-aware RMS + soft-knee + EMA + peak protection)
        if not getattr(config.debug, "bypass_speech_normalization", False) and getattr(config.asr, "normalize_speech", True):
            norm_res = self._normalizer.process(pcm_float32, frame_state=frame_state, use_smoothing=True)
            pcm_float32 = norm_res.pcm

            # Logging and Telemetry
            log_stats = getattr(config.asr, "normalize_log_stats", True)
            verbose_levels = getattr(config.debug, "log_audio_levels", False)
            if (is_commit and log_stats) or verbose_levels:
                mode_tag = "COMMIT" if is_commit else "PREVIEW"
                clean_utt = (eff_utt_id or "unknown")[:8]
                limit_str = f" (peak_scale x{norm_res.peak_scale:.2f})" if norm_res.peak_scale < 1.0 else ""
                crest_db = _level_db(norm_res.peak_after) - _level_db(norm_res.speech_rms) if norm_res.speech_rms > 0 else 0.0
                logger.debug(
                    f"[ASR LEVEL] [utt={clean_utt}] [{mode_tag}] "
                    f"speech_rms={norm_res.speech_rms:.4f} ({_level_db(norm_res.speech_rms):.1f}dB) -> "
                    f"peak_in={norm_res.peak_before:.4f} -> peak_out={norm_res.peak_after:.4f} ({_level_db(norm_res.peak_after):.1f}dB, crest={crest_db:.1f}dB) | "
                    f"gain_des={norm_res.desired_gain:.2f} -> gain_smooth={norm_res.smoothed_gain:.2f}{limit_str}"
                )

            perf.record_metric("asr", "input_rms_db", _level_db(norm_res.speech_rms))
            perf.record_metric("asr", "output_rms_db", _level_db(norm_res.speech_rms * norm_res.smoothed_gain * norm_res.peak_scale))
            perf.record_metric("asr", "applied_gain", norm_res.smoothed_gain * norm_res.peak_scale)
            if norm_res.smoothed_gain > 1.5:
                perf.increment_counter("asr.gain_boosted")
            if norm_res.peak_scale < 1.0:
                perf.increment_counter("asr.peak_limited")

        dump_asr_input(self.session_id, eff_utt_id, pcm_float32, is_commit=is_commit)

        commit_counted = False

        acquired = False
        inference_started = False

        try:
            # LOCK ORDERING CONVENTION:
            # 1. Always acquire _shared_lock (in _ensure_shared_model) BEFORE _shared_infer_lock.
            # 2. Never acquire _shared_lock while holding _shared_infer_lock.
            # 3. Session inference is single-tenant serialized via _shared_infer_lock.
            model = self._ensure_shared_model()
            family = self.model_info.get("family", "")
            lang = normalize_language_for_family(self._language, family)
            stream_opts = build_family_options(family, self.model_info, model=model, slot="stream")
            run_opts = build_family_options(family, self.model_info, model=model, slot="run")

            t_lock_start = time.perf_counter()
            if is_commit:
                with TranscribeEngine._commit_lock:
                    TranscribeEngine._commit_waiting += 1
                    commit_counted = True
                # Timeout on blocking acquire to avoid permanent hang if C++ engine deadlocks
                acquired = TranscribeEngine._shared_infer_lock.acquire(blocking=True, timeout=10.0)
                if not acquired:
                    logger.error("[ASR LOCK] Timed out waiting for infer lock (>10s)")
                    perf.increment_counter("asr.commit_lock_timeout")
                    return ""
            else:
                # Preview: non-blocking acquisition - yield immediately if infer lock is busy
                acquired = TranscribeEngine._shared_infer_lock.acquire(blocking=False)
                if not acquired:
                    logger.debug("[ASR PREVIEW] Skipped preview inference because infer lock is busy")
                    perf.increment_counter("asr.preview_lock_skipped")
                    return ""

            lock_wait_ms = (time.perf_counter() - t_lock_start) * 1000.0
            perf.record_metric("asr", "lock_wait_ms", lock_wait_ms)
            if is_commit and lock_wait_ms > 20.0:
                logger.debug(f"[ASR LOCK] Acquired lock after {lock_wait_ms:.1f}ms wait")

            # Reuse cached session across inference runs to avoid allocating C structures every cycle
            session = self._ensure_session(model)
            inference_started = True

            t_infer_start = time.perf_counter()
            raw_text = ""
            stream_success = False

            # 1. If model supports native streaming, use session.stream(...)
            if TranscribeEngine._shared_supports_streaming:
                try:
                    with session.stream(language=lang, family=stream_opts) as stream:
                        stream.feed(pcm_float32)
                        stream.finalize()
                        raw_text = stream.text().full
                    stream_success = True
                except Exception as stream_err:
                    if lang:
                        try:
                            with session.stream(language=None, family=stream_opts) as stream:
                                stream.feed(pcm_float32)
                                stream.finalize()
                                raw_text = stream.text().full
                            stream_success = True
                        except Exception:
                            pass
                    if not stream_success:
                        logger.warning(
                            f"session.stream() failed on model '{self.model_key}', falling back to session.run(): {stream_err}"
                        )
                        # Fallback to session.run below

            # 2. Non-streaming model or fallback path: session.run(...)
            if not stream_success:
                try:
                    res = session.run(pcm_float32, language=lang, family=run_opts)
                    raw_text = getattr(res, "text", str(res))
                except Exception as run_err:
                    # transcribe_cpp raises OutputTruncated when generation reaches token budget (e.g. 256 tokens)
                    # It attaches partial_result holding the materialized partial transcript.
                    partial = getattr(run_err, "partial_result", None)
                    if partial is not None and hasattr(partial, "text"):
                        logger.warning(
                            f"[ASR] Output truncated for '{self.model_key}' (hit generation token cap), preserving partial transcript: {run_err}"
                        )
                        raw_text = partial.text
                    elif lang:
                        logger.warning(
                            f"Language '{lang}' failed on model {self.model_key}, falling back to auto: {run_err}"
                        )
                        try:
                            res = session.run(pcm_float32, language=None, family=run_opts)
                            raw_text = getattr(res, "text", str(res))
                        except Exception as auto_err:
                            auto_partial = getattr(auto_err, "partial_result", None)
                            if auto_partial is not None and hasattr(auto_partial, "text"):
                                raw_text = auto_partial.text
                            else:
                                raise auto_err
                    else:
                        raise run_err

            infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            audio_dur = len(pcm_float32) / DEFAULT_SAMPLE_RATE
            rtf = (infer_ms / 1000.0) / max(0.001, audio_dur)

            perf.record_metric("asr", "infer_ms", infer_ms)
            perf.record_metric("asr", "rtf", rtf)
            perf.record_metric("asr", "audio_dur_sec", audio_dur)
            perf.increment_counter("asr.total_inferences")

            cleaned = clean_transcript_text(raw_text)
            # if cleaned:
            #     logger.debug(
            #         f"[ASR] [{'COMMIT' if is_commit else 'PREVIEW'}] ({self.model_key}) Recognized: '{cleaned}'"
            #     )
            if config.perf.enabled and infer_ms > 200.0:
                logger.debug(
                    f"⚡ [PERF_ASR] [{'COMMIT' if is_commit else 'PREVIEW'}] Infer: {infer_ms:.1f}ms | Audio: {audio_dur:.2f}s | "
                    f"RTF: {rtf:.2f} | LockWait: {lock_wait_ms:.1f}ms | Out: '{cleaned[:30]}...'"
                )
            return cleaned

        except Exception as e:
            logger.error(f"transcribe.cpp inference error: {e}", exc_info=True)
            # Reset session on error only if the error happened while holding the lock during inference
            if acquired and inference_started:
                if TranscribeEngine._shared_session is not None:
                    try:
                        TranscribeEngine._shared_session.close()
                    except Exception:
                        pass
                    TranscribeEngine._shared_session = None
                    ASRModelManager._shared_session = None
            return ""
        finally:
            if acquired:
                TranscribeEngine._shared_infer_lock.release()
            if commit_counted:
                with TranscribeEngine._commit_lock:
                    TranscribeEngine._commit_waiting -= 1

    def feed_audio(self, pcm_bytes: bytes, timestamp: float = 0.0, vad_state: int = VAD_STATE_SPEECH) -> None:
        """Feed incoming 16kHz 16-bit mono PCM bytes."""
        if not pcm_bytes:
            return

        dur = self._audio_buffer_mgr.feed_bytes(pcm_bytes, vad_state=vad_state)

        # Auto-commit if single sentence exceeds maximum length
        if dur >= self.sentence_config.max_duration_sec:
            logger.debug(f"Utterance reached max duration ({dur:.1f}s), triggering auto-commit")
            self.on_speech_end(reason="MAX_DURATION")

    def on_speech_start(self) -> None:
        """Called by VAD on speech onset."""
        with self._state_lock:
            self._is_speech_active = True
            self._last_partial_text = ""
            self._last_partial_samples = 0
            self._current_utterance_id = str(uuid.uuid4())
            self._last_committed_head = ""
            self._last_committed_head_time = 0.0
        self._last_polled_samples = 0
        self._segmenter.reset()

    def on_speech_end(self, reason: str = "VAD_SILENCE") -> None:
        """Called by VAD, max_duration, or connection end on speech finish."""
        with self._state_lock:
            pcm_combined, frame_state, _ = self._audio_buffer_mgr.pop_all()
            if pcm_combined is None or len(pcm_combined) == 0:
                if reason != "MAX_DURATION":
                    self._is_speech_active = False
                return

            if reason != "MAX_DURATION":
                self._is_speech_active = False

            utt_id = self._current_utterance_id
            self._current_utterance_id = str(uuid.uuid4())
            cached_text = self._last_partial_text
            last_samples = self._last_partial_samples
            self._last_partial_text = ""
            self._last_partial_samples = 0

        self._last_polled_samples = 0
        self._segmenter.reset()

        # Dump VAD utterance for pipeline fidelity audit
        dump_bytes = (np.clip(pcm_combined, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        dump_vad_utterance(self.session_id, utt_id, dump_bytes, reason=reason)

        # Utterance energy & SNR Proxy measurement
        dur_s = len(pcm_combined) / float(DEFAULT_SAMPLE_RATE)
        utt_rms = float(np.sqrt(np.mean(pcm_combined ** 2) + 1e-12))
        utt_peak = float(np.max(np.abs(pcm_combined)))

        # SNR proxy estimation using 25ms frame energy distribution:
        # p15 frame RMS captures true background noise floor, p85 captures speech core.
        frame_size = int(DEFAULT_SAMPLE_RATE * 0.025)  # 400 samples (25ms)
        n_frames = len(pcm_combined) // frame_size
        if n_frames >= 4:
            frames = pcm_combined[:n_frames * frame_size].reshape(n_frames, frame_size)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
            noise_rms = float(np.percentile(frame_rms, 15))
            speech_rms = float(np.percentile(frame_rms, 85))
            snr_db = 20.0 * math.log10(max(speech_rms, 1e-6) / max(noise_rms, 1e-6))
        else:
            snr_db = 0.0

        # Confidence heuristic: SNR proxy is only trustworthy when utterance is sufficiently long (dur >= 2.0s)
        # and has adequate contrast (snr >= 15dB). Short utterances (< 2s) have high variance due to boundary leakage.
        if dur_s >= 2.0 and snr_db >= 15.0:
            conf_tag = "(reliable)"
        elif dur_s < 2.0:
            conf_tag = "(dur<2s)"
        else:
            conf_tag = "(noisy)"

        clean_utt = (utt_id or "unknown")[:8]
        if getattr(config.asr, "normalize_log_stats", True):
            logger.info(
                f"[ASR UTT LEVEL] [utt={clean_utt}] dur={dur_s:.2f}s "
                f"rms={utt_rms:.4f} ({_level_db(utt_rms):.1f}dB) "
                f"peak={utt_peak:.4f} ({_level_db(utt_peak):.1f}dB) "
                f"snr~={snr_db:.1f}dB {conf_tag} reason={reason}"
            )

        # Check if we can reuse cached preview text
        diff_samples = len(pcm_combined) - last_samples
        # Audio Pipeline Fidelity: Only reuse preview text if NO new audio samples arrived since last preview (diff_samples == 0).
        # If there are any un-inferred trailing samples (diff_samples > 0), ALWAYS run inference on pcm_combined
        # so that trailing Japanese verb conjugations, particles, and endings (e.g. 〜ました, 〜ません) are never truncated!
        reuse_preview = (
            bool(cached_text and cached_text.strip())
            and reason != "MAX_DURATION"
            and diff_samples == 0
        )
        final_cached = cached_text if reuse_preview else None

        # Execute final commit asynchronously without blocking the event loop or VAD
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._commit_async(pcm_combined, utt_id, reason=reason, cached_text=final_cached, frame_state=frame_state),
                self._loop,
            )
        else:
            _SYNC_COMMIT_EXECUTOR.submit(
                self._commit_sync, pcm_combined, utt_id, reason, final_cached, frame_state
            )


    def _emit_final(self, text: str, utt_id: str, reason: str) -> None:
        """Emit finalized sentence to client and translation pipeline with descriptive log."""
        if not text or not text.strip():
            return

        clean_text = text.strip()
        clean_utt = (utt_id or "unknown")[:8]
        if self._deduplicator.is_duplicate(clean_text):
            logger.debug(f"[ASR DEDUP] [utt={clean_utt}] Skipped duplicate commit [{reason}]: '{clean_text}'")
            return

        norm_text = SentenceSegmenter.normalize_for_comparison(clean_text)
        self._deduplicator.record_commit(clean_text, norm_text)

        try:
            logger.info(f"[ASR COMMIT] [utt={clean_utt}] [{self.model_key}] [{reason}] ({self._language}): '{clean_text}'")
        except Exception:
            pass
        msg = {
            "type": "utterance_update",
            "utterance_id": utt_id,
            "text": clean_text,
            "ui_text": clean_text,
            "stable_text": clean_text,
            "unstable_text": "",
            "is_final": True,
            "language": self._language,
            "model": self.model_key,
            "commit_method": reason,
            "filtered": False,
        }
        self._push_message(msg)

    async def _commit_async(
        self,
        pcm_combined: np.ndarray,
        utt_id: str,
        reason: str = "VAD_SILENCE",
        cached_text: Optional[str] = None,
        frame_state: Optional[np.ndarray] = None,
    ) -> None:
        """Asynchronously compute final transcription and push to queue."""
        if cached_text:
            text = cached_text
            perf.increment_counter("asr.cached_preview_reuse")
            logger.debug(f"[ASR COMMIT] Reusing stable preview text")
        else:
            perf.increment_counter("asr.commit_inferences")
            text = await asyncio.to_thread(
                self._run_inference, pcm_combined, frame_state=frame_state, is_commit=True, utt_id=utt_id
            )

        if text and not self._segmenter.is_text_filtered(text):
            if reason == "MAX_DURATION":
                with self._state_lock:
                    self._last_committed_head = text
                    self._last_committed_head_time = time.time()
            self._emit_final(text, utt_id, reason=reason)
        elif text:
            clean_utt = (utt_id or "unknown")[:8]
            logger.info(f"[ASR FILTER] [utt={clean_utt}] Dropped short commit (< {self.sentence_config.min_words_to_commit} words): '{text}'")
            perf.increment_counter("asr.short_commits_filtered")

    def _commit_sync(
        self,
        pcm_combined: np.ndarray,
        utt_id: str,
        reason: str = "VAD_SILENCE",
        cached_text: Optional[str] = None,
        frame_state: Optional[np.ndarray] = None,
    ) -> None:
        if cached_text:
            text = cached_text
            perf.increment_counter("asr.cached_preview_reuse")
            logger.debug(f"[ASR COMMIT] Reusing stable preview text")
        else:
            perf.increment_counter("asr.commit_inferences")
            text = self._run_inference(pcm_combined, frame_state=frame_state, is_commit=True, utt_id=utt_id)

        if text and not self._segmenter.is_text_filtered(text):
            if reason == "MAX_DURATION":
                with self._state_lock:
                    self._last_committed_head = text
                    self._last_committed_head_time = time.time()
            self._emit_final(text, utt_id, reason=reason)
        elif text:
            clean_utt = (utt_id or "unknown")[:8]
            logger.info(f"[ASR FILTER] [utt={clean_utt}] Dropped short commit (< {self.sentence_config.min_words_to_commit} words): '{text}'")
            perf.increment_counter("asr.short_commits_filtered")

    def _push_message(self, msg: Dict[str, Any]) -> None:
        if not self._running:
            return
        if self._loop and self._token_queue:
            def _safe_put():
                try:
                    self._token_queue.put_nowait(msg)
                except asyncio.QueueFull:
                    logger.warning("ASR token queue is full, message dropped.")
                except Exception as e:
                    logger.debug(f"Error putting token message: {e}")

            try:
                self._loop.call_soon_threadsafe(_safe_put)
            except RuntimeError:
                pass

    async def _partial_preview_poller(self) -> None:
        """Background poller running non-blocking preview in worker threads."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval_sec)

                with self._state_lock:
                    if not self._is_speech_active:
                        self._last_polled_samples = 0
                        continue

                # Skip polling if a commit is currently waiting for lock
                with TranscribeEngine._commit_lock:
                    if TranscribeEngine._commit_waiting > 0:
                        continue

                perf.increment_counter("asr.preview_polls")

                # Non-blocking snapshot with buffer version - avoids copy if buffer has not received new samples
                snapshot = self._audio_buffer_mgr.get_snapshot_if_newer(self._last_polled_samples)
                if snapshot is None:
                    continue

                pcm_snapshot = snapshot.pcm
                dur = snapshot.duration_sec
                snapshot_samples = snapshot.sample_count
                snap_ver = snapshot.version
                frame_state = snapshot.frame_state

                if dur < self._min_transcribe_sec:
                    continue

                self._last_polled_samples = snapshot_samples

                with self._state_lock:
                    utt_id = self._current_utterance_id

                # Run in worker thread non-blocking so the event loop NEVER blocks
                perf.increment_counter("asr.preview_infers")
                preview_text = await asyncio.to_thread(
                    self._run_inference, pcm_snapshot, frame_state=frame_state, is_commit=False, utt_id=utt_id
                )

                # Strip prefix if it overlaps with recently committed head (prevents re-transcribing same words)
                with self._state_lock:
                    head = self._last_committed_head
                    head_time = self._last_committed_head_time

                if head and (time.time() - head_time < PREFIX_STRIP_WINDOW_SEC):
                    preview_text = SentenceSegmenter.remove_prefix_overlap(head, preview_text)

                if not preview_text:
                    continue

                # Check stability outside state lock to keep critical section minimal
                is_stable = False
                if self.sentence_config.split_on_stability:
                    is_stable = self._segmenter.check_stability(preview_text) and not self._segmenter.is_text_filtered(preview_text)

                should_emit_partial = False
                did_split = False

                with self._state_lock:
                    if not self._is_speech_active or utt_id != self._current_utterance_id:
                        continue

                    if is_stable:
                        self._current_utterance_id = str(uuid.uuid4())
                        self._last_partial_text = ""
                        did_split = True
                    elif preview_text != self._last_partial_text:
                        self._last_partial_text = preview_text
                        self._last_partial_samples = snapshot_samples
                        # Enforce min_words_to_commit: do not flash short fragments on screen
                        if count_content_tokens(preview_text) >= self.sentence_config.min_words_to_commit:
                            should_emit_partial = True
                        else:
                            should_emit_partial = False

                if did_split:
                    with self._state_lock:
                        self._last_committed_head = preview_text
                        self._last_committed_head_time = time.time()
                    self._emit_final(preview_text, utt_id, reason="STABLE_PREFIX")
                    self._audio_buffer_mgr.slice_after(snapshot_samples, expected_version=snap_ver)
                    self._last_polled_samples = 0
                    self._segmenter.reset_stability()
                    continue

                if should_emit_partial:
                    # logger.debug(f"💬 [ASR PREVIEW] [{utt_id[:8]}]: '{preview_text}'")
                    out_msg = {
                        "type": "utterance_update",
                        "utterance_id": utt_id,
                        "text": preview_text,
                        "ui_text": preview_text,
                        "stable_text": "",
                        "unstable_text": preview_text,
                        "is_final": False,
                        "language": self._language,
                        "model": self.model_key,
                        "filtered": False,
                    }
                    self._push_message(out_msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                try:
                    logger.error(f"Error in _partial_preview_poller: {e}", exc_info=True)
                except Exception:
                    pass
                await asyncio.sleep(0.1)

    async def stream_tokens(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield transcribed tokens to WebSocket streaming loop."""
        self._loop = asyncio.get_running_loop()
        q = self._get_queue()

        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._partial_preview_poller())

        try:
            while self._running:
                msg = await q.get()
                if msg is _SENTINEL:
                    q.task_done()
                    break
                yield msg
                q.task_done()
        except asyncio.CancelledError:
            pass

    async def cleanup(self) -> None:
        """Clean up per-session background tasks and buffer."""
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        # Unblock any consumer waiting on stream_tokens
        if self._token_queue:
            self._token_queue.put_nowait(_SENTINEL)

        self._audio_buffer_mgr.clear()
        self._last_polled_samples = 0
