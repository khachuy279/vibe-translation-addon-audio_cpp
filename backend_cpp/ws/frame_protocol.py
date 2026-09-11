"""Binary audio framing protocol parser for backend_cpp WebSocket connections."""

import json
import logging
import math
import struct
from typing import Optional

logger = logging.getLogger(__name__)


class ParsedFrame(tuple):
    """Audio frame tuple backward-compatible with 3-tuple (pcm, ts, idx) while exposing rich attributes."""

    def __new__(
        cls,
        pcm: Optional[bytes],
        capture_ts: float = 0.0,
        chunk_idx: Optional[int] = None,
        media_start_time: float = 0.0,
        media_end_time: float = 0.0,
        epoch: int = 0,
        playback_rate: float = 1.0,
        chunk_duration_ms: float = 64.0,
    ):
        obj = super(ParsedFrame, cls).__new__(cls, (pcm, capture_ts, chunk_idx))
        obj.pcm = pcm
        obj.capture_timestamp = capture_ts
        obj.chunk_index = chunk_idx
        obj.media_start_time = media_start_time
        obj.media_end_time = media_end_time
        obj.epoch = epoch
        obj.playback_rate = playback_rate
        obj.chunk_duration_ms = chunk_duration_ms
        return obj


def parse_audio_frame(data: bytes) -> ParsedFrame:
    """Parse incoming WebSocket binary frame containing audio data.

    Supports:
    - Format A: 4-byte uint32 header length + JSON header + 16-bit PCM data.
    - Format B: 8-byte float64 capture timestamp + 16-bit PCM data.

    Returns:
        ParsedFrame: Acts as (pcm_data, capture_timestamp, chunk_index)
        with properties: media_start_time, media_end_time, epoch, playback_rate, chunk_duration_ms.
    """
    if not data or len(data) < 4:
        logger.debug(f"[FRAME] Rejected frame: data is empty or too short ({len(data) if data else 0} bytes)")
        return ParsedFrame(None, 0.0, None)

    # Try Format A first: 4-byte uint32 header length
    header_len = struct.unpack("<I", data[:4])[0]
    # Sanity check header length (JSON header should be compact, typically < 2KB)
    if 0 < header_len < 2048 and (4 + header_len) <= len(data):
        try:
            header_json = data[4:4 + header_len].decode("utf-8")
            header = json.loads(header_json)
            if header.get("type") == "audio_chunk":
                pcm = data[4 + header_len:]
                # Verify 16-bit PCM alignment (must be multiple of 2 bytes)
                if len(pcm) > 0 and len(pcm) % 2 == 0:
                    capture_ts = float(header.get("captureTimestamp", 0.0))
                    chunk_idx = header.get("chunkIndex")
                    media_start = float(header.get("chunkStartMediaTime", header.get("mediaCurrentTime", capture_ts)))
                    chunk_dur_ms = float(header.get("chunkDurationMs", (len(pcm) / 32000.0) * 1000.0))
                    media_end = float(header.get("chunkEndMediaTime", media_start + (chunk_dur_ms / 1000.0)))
                    epoch = int(header.get("epoch", 0))
                    rate = float(header.get("playbackRate", 1.0))
                    return ParsedFrame(
                        pcm=pcm,
                        capture_ts=capture_ts,
                        chunk_idx=chunk_idx,
                        media_start_time=media_start,
                        media_end_time=media_end,
                        epoch=epoch,
                        playback_rate=rate,
                        chunk_duration_ms=chunk_dur_ms,
                    )
                logger.debug(f"[FRAME] Rejected Format A frame: invalid PCM length {len(pcm)} (not 16-bit aligned)")
            else:
                logger.debug(f"[FRAME] Rejected Format A frame: unexpected header type '{header.get('type')}'")
            return ParsedFrame(None, 0.0, None)
        except Exception as e:
            logger.debug(f"[FRAME] Format A JSON header parse error: {e}")
            return ParsedFrame(None, 0.0, None)

    # Format B: 8-byte float64 timestamp header
    if len(data) > 8:
        pcm_len = len(data) - 8
        if pcm_len > 0 and pcm_len % 2 == 0:
            try:
                capture_ts = struct.unpack("<d", data[:8])[0]
                if not math.isnan(capture_ts) and not math.isinf(capture_ts) and capture_ts >= 0.0:
                    dur_ms = (pcm_len / 32000.0) * 1000.0
                    return ParsedFrame(
                        pcm=data[8:],
                        capture_ts=capture_ts,
                        chunk_idx=None,
                        media_start_time=capture_ts,
                        media_end_time=capture_ts + (dur_ms / 1000.0),
                        epoch=0,
                        playback_rate=1.0,
                        chunk_duration_ms=dur_ms,
                    )
            except Exception:
                pass

    return ParsedFrame(None, 0.0, None)
