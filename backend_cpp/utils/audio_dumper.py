"""Asynchronous Audio Dumper for pipeline debugging and fidelity auditing.

Writes 16kHz mono WAV files at three critical checkpoints:
1. Ingress: Continuous stream of raw PCM frames received from the WebSocket client.
2. VAD Utterance: Segmented audio chunks output by VAD on speech boundaries.
3. ASR Input: Float32 audio passed directly into transcribe.cpp inference.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import threading
from typing import Dict, Optional, Tuple
import wave

import numpy as np

from backend_cpp.config import config

logger = logging.getLogger(__name__)

_DUMP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio_dumper")
_INGRESS_FILES: Dict[str, Tuple[wave.Wave_write, threading.Lock]] = {}
_INGRESS_FILES_LOCK = threading.Lock()
_UTT_INDEX_MAP: Dict[str, Dict[str, int]] = {}
_UTT_COUNTER: Dict[str, int] = {}
_PREVIEW_COUNTER: Dict[Tuple[str, str], int] = {}
_UTT_LOCK = threading.Lock()


def _ensure_session_dir(session_id: str) -> Path:
    dump_dir = Path(config.debug.dump_dir) / session_id
    dump_dir.mkdir(parents=True, exist_ok=True)
    return dump_dir


def _get_or_create_utt_idx(session_id: str, utt_id: str) -> int:
    with _UTT_LOCK:
        sess_map = _UTT_INDEX_MAP.setdefault(session_id, {})
        short_id = "".join(c for c in (utt_id or "unknown")[:8] if c.isalnum() or c in "-_") or "unknown"
        if short_id not in sess_map:
            _UTT_COUNTER[session_id] = _UTT_COUNTER.get(session_id, 0) + 1
            sess_map[short_id] = _UTT_COUNTER[session_id]
        return sess_map[short_id]


def _short_utt_id(utt_id: str) -> str:
    return "".join(c for c in (utt_id or "unknown")[:8] if c.isalnum() or c in "-_") or "unknown"


def float32_to_pcm16_bytes(pcm_float32: np.ndarray) -> bytes:
    """Convert float32 [-1.0, 1.0] audio into 16-bit little-endian PCM bytes.

    Kept as a helper so the (relatively expensive) full-array clip/scale/astype can
    run on the dumper worker thread instead of the audio hot path.
    """
    return (np.clip(pcm_float32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _write_wav_16k_mono(path: Path, raw_pcm_bytes: bytes) -> None:
    """Write raw 16 kHz mono 16-bit PCM to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(raw_pcm_bytes)


def dump_ingress_chunk(session_id: str, chunk_index: int, pcm_bytes: bytes) -> None:
    """Append incoming raw PCM bytes to continuous ingress WAV file."""
    if not getattr(config.debug, "dump_audio", False) or not pcm_bytes:
        return

    def _write():
        try:
            with _INGRESS_FILES_LOCK:
                if session_id not in _INGRESS_FILES:
                    sess_dir = _ensure_session_dir(session_id)
                    wav_path = sess_dir / "00_ingress_stream.wav"
                    wf = wave.open(str(wav_path), "wb")
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    _INGRESS_FILES[session_id] = (wf, threading.Lock())

                wf, lk = _INGRESS_FILES[session_id]

            with lk:
                wf.writeframes(pcm_bytes)
        except Exception as e:
            logger.debug(f"AudioDumper ingress write error: {e}")

    _DUMP_EXECUTOR.submit(_write)


def dump_vad_utterance(session_id: str, utt_id: str, pcm_bytes: bytes, reason: str = "VAD") -> None:
    """Save segmented audio utterance output by VAD before ASR queuing."""
    if not getattr(config.debug, "dump_audio", False) or not pcm_bytes:
        return

    idx = _get_or_create_utt_idx(session_id, utt_id)
    short_id = "".join(c for c in (utt_id or "unknown")[:8] if c.isalnum() or c in "-_") or "unknown"
    clean_reason = "".join(c for c in (reason or "VAD") if c.isalnum() or c in "-_") or "VAD"
    filename = f"{idx:02d}_{short_id}_vad_{clean_reason}.wav"

    def _write():
        try:
            sess_dir = _ensure_session_dir(session_id)
            wav_path = sess_dir / filename
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm_bytes)
        except Exception as e:
            logger.debug(f"AudioDumper VAD utterance write error: {e}")

    _DUMP_EXECUTOR.submit(_write)


def dump_vad_utterance_f32(
    session_id: str,
    utt_id: str,
    pcm_float32: np.ndarray,
    reason: str = "VAD",
) -> None:
    """Save a VAD utterance given float32 audio.

    Prefer this over :func:`dump_vad_utterance` on hot paths: the float32 -> int16
    conversion is deferred to the dumper worker thread, so the caller pays **nothing**
    when ``config.debug.dump_audio`` is False (the default) and only a reference copy
    when dumping is enabled.
    """
    if not getattr(config.debug, "dump_audio", False):
        return
    if pcm_float32 is None or len(pcm_float32) == 0:
        return

    idx = _get_or_create_utt_idx(session_id, utt_id)
    short_id = _short_utt_id(utt_id)
    clean_reason = "".join(c for c in (reason or "VAD") if c.isalnum() or c in "-_") or "VAD"
    filename = f"{idx:02d}_{short_id}_vad_{clean_reason}.wav"

    # Snapshot the array reference; the conversion itself runs on the worker thread.
    audio = pcm_float32

    def _write():
        try:
            raw_bytes = float32_to_pcm16_bytes(audio)
            _write_wav_16k_mono(_ensure_session_dir(session_id) / filename, raw_bytes)
        except Exception as e:
            logger.debug(f"AudioDumper VAD utterance write error: {e}")

    _DUMP_EXECUTOR.submit(_write)


def dump_asr_input(session_id: str, utt_id: str, pcm_float32: np.ndarray, is_commit: bool = True) -> None:
    """Save normalized/raw float32 audio right before transcribe.cpp model execution."""
    if not getattr(config.debug, "dump_audio", False) or pcm_float32 is None or len(pcm_float32) == 0:
        return

    idx = _get_or_create_utt_idx(session_id, utt_id)
    short_id = _short_utt_id(utt_id)

    if is_commit:
        filename = f"{idx:02d}_{short_id}_asr_commit.wav"
    else:
        with _UTT_LOCK:
            p_key = (session_id, short_id)
            p_idx = _PREVIEW_COUNTER.get(p_key, 0) + 1
            _PREVIEW_COUNTER[p_key] = p_idx
        filename = f"{idx:02d}_{short_id}_asr_preview_{p_idx:02d}.wav"

    # Defer the float32 -> int16 conversion to the worker thread.
    audio = pcm_float32

    def _write():
        try:
            raw_bytes = float32_to_pcm16_bytes(audio)
            _write_wav_16k_mono(_ensure_session_dir(session_id) / filename, raw_bytes)
        except Exception as e:
            logger.debug(f"AudioDumper ASR input write error: {e}")

    _DUMP_EXECUTOR.submit(_write)


def close_session_dumper(session_id: str) -> None:
    """Flush and close active wave file handles for a session."""
    def _close():
        with _INGRESS_FILES_LOCK:
            item = _INGRESS_FILES.pop(session_id, None)

        with _UTT_LOCK:
            _UTT_INDEX_MAP.pop(session_id, None)
            _UTT_COUNTER.pop(session_id, None)
            for k in list(_PREVIEW_COUNTER.keys()):
                if k[0] == session_id:
                    _PREVIEW_COUNTER.pop(k, None)

        if item:
            wf, lk = item
            with lk:
                try:
                    wf.close()
                except Exception:
                    pass

    fut = _DUMP_EXECUTOR.submit(_close)
    try:
        fut.result(timeout=5.0)
    except Exception as e:
        logger.debug(f"AudioDumper close error: {e}")


def flush_dumper(timeout: float = 5.0) -> None:
    """Wait for all pending dump writes to complete."""
    fut = _DUMP_EXECUTOR.submit(lambda: None)
    try:
        fut.result(timeout=timeout)
    except Exception:
        pass

