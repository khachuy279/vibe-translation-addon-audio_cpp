"""Binary audio framing protocol parser for backend_audio_cpp WebSocket connections."""

import json
import logging
import math
import struct
from typing import Optional, Tuple

logger = logging.getLogger("backend_audio_cpp.ws.frame")


def parse_audio_frame(data: bytes) -> Tuple[Optional[bytes], float, Optional[int]]:
    """Parse incoming WebSocket binary frame containing audio data.

    Supports:
    - Format A: 4-byte uint32 header length + JSON header + 16-bit PCM data.
    - Format B: 8-byte float64 capture timestamp + 16-bit PCM data.

    Returns:
        (pcm_data, capture_timestamp, chunk_index)
        Returns (None, 0.0, None) if frame is invalid.
    """
    if not data or len(data) < 4:
        return None, 0.0, None

    # Format A: 4-byte uint32 header length
    header_len = struct.unpack("<I", data[:4])[0]
    if 0 < header_len < 2048 and (4 + header_len) <= len(data):
        try:
            header_json = data[4:4 + header_len].decode("utf-8")
            header = json.loads(header_json)
            if header.get("type") == "audio_chunk":
                pcm = data[4 + header_len:]
                if len(pcm) > 0 and len(pcm) % 2 == 0:
                    capture_ts = float(header.get("captureTimestamp", 0.0))
                    chunk_idx = header.get("chunkIndex")
                    return pcm, capture_ts, chunk_idx
            return None, 0.0, None
        except Exception:
            return None, 0.0, None

    # Format B: 8-byte float64 timestamp header
    if len(data) > 8:
        pcm_len = len(data) - 8
        if pcm_len > 0 and pcm_len % 2 == 0:
            try:
                capture_ts = struct.unpack("<d", data[:8])[0]
                if not math.isnan(capture_ts) and not math.isinf(capture_ts) and capture_ts >= 0.0:
                    return data[8:], capture_ts, None
            except Exception:
                pass

    return None, 0.0, None
