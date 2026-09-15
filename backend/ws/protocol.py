"""Module phân tích giao thức khung nhị phân âm thanh (Audio Binary Framing Protocol).

Hỗ trợ 2 định dạng truyền tải từ Firefox Extension:
- Format A: 4-byte uint32 header length + JSON header (chứa captureTimestamp, chunkIndex) + 16-bit PCM data.
- Format B: 8-byte float64 capture timestamp + 16-bit PCM data.
"""

import json
import math
import struct
from typing import Optional, Tuple

from backend.utils.logger import get_logger

logger = get_logger("ws.protocol")


def parse_audio_frame(data: bytes) -> Tuple[Optional[bytes], float, Optional[int]]:
    """Phân tích khung nhị phân âm thanh gửi từ Client qua WebSocket.

    Returns:
        (pcm_data, capture_timestamp, chunk_index)
        Nếu khung không hợp lệ hoặc lỗi định dạng, trả về (None, 0.0, None).
    """
    if not data or len(data) < 4:
        logger.debug(f"Bỏ qua khung nhị phân rỗng hoặc quá ngắn: {len(data) if data else 0} bytes")
        return None, 0.0, None

    # Thử Format A: 4-byte uint32 header length
    header_len = struct.unpack("<I", data[:4])[0]
    if 0 < header_len < 2048 and (4 + header_len) <= len(data):
        try:
            header_json = data[4:4 + header_len].decode("utf-8")
            header = json.loads(header_json)
            if header.get("type") == "audio_chunk":
                pcm = data[4 + header_len:]
                # Đảm bảo căn chỉnh 16-bit PCM (bội số của 2 bytes)
                if len(pcm) > 0 and len(pcm) % 2 == 0:
                    capture_ts = float(header.get("captureTimestamp", 0.0))
                    chunk_idx = header.get("chunkIndex")
                    return pcm, capture_ts, chunk_idx
                logger.debug(f"Độ dài PCM không căn chỉnh 16-bit: {len(pcm)} bytes")
            else:
                logger.debug(f"Loại header không mong đợi: '{header.get('type')}'")
            return None, 0.0, None
        except Exception as e:
            logger.debug(f"Lỗi phân tích JSON header Format A: {e}")
            return None, 0.0, None

    # Thử Format B: 8-byte float64 timestamp
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
