"""Binary audio framing protocol parser for backend_cpp WebSocket connections."""

import json
import logging
import math
import struct
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def parse_audio_frame(data: bytes) -> Tuple[Optional[bytes], float, Optional[int]]:
    """Parse incoming WebSocket binary frame containing audio data.

    Supports:
    - Format A: 4-byte uint32 header length + JSON header + 16-bit PCM data.
    - Format B: 8-byte float64 capture timestamp + 16-bit PCM data.

    Returns:
        (pcm_data, capture_timestamp, chunk_index)
        Returns (None, 0.0, None) if frame cannot be safely parsed or is invalid.
    """
    if not data or len(data) < 4:
        logger.debug(f"[FRAME] Rejected frame: data is empty or too short ({len(data) if data else 0} bytes)")
        return None, 0.0, None

    # Try Format A first: 4-byte uint32 header length
    header_len = struct.unpack("<I", data[:4])[0]
    # Sanity check header length (JSON header should be compact, typically < 1KB)
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
                    return pcm, capture_ts, chunk_idx
                logger.debug(f"[FRAME] Rejected Format A frame: invalid PCM length {len(pcm)} (not 16-bit aligned)")
            else:
                logger.debug(f"[FRAME] Rejected Format A frame: unexpected header type '{header.get('type')}'")
            # If header is explicitly typed but not audio_chunk or corrupted PCM, do not fall back to Format B
            return None, 0.0, None
        except Exception as e:
            logger.debug(f"[FRAME] Format A JSON header parse error: {e}")
            # Header decode/parse failed; if header_len was plausibly within bounds, reject rather than injecting garbage
            return None, 0.0, None

    # Format B: 8-byte float64 timestamp header
    if len(data) > 8:
        pcm_len = len(data) - 8
        if pcm_len > 0 and pcm_len % 2 == 0:
            try:
                capture_ts = struct.unpack("<d", data[:8])[0]
                # Validate reasonable timestamp (not NaN, not inf, >= 0)
                if not math.isnan(capture_ts) and not math.isinf(capture_ts) and capture_ts >= 0.0:
                    return data[8:], capture_ts, None
            except Exception:
                pass

    return None, 0.0, None
