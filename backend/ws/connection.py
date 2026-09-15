"""Lớp bọc WebSocket đảm bảo an toàn luồng và tác vụ bất đồng bộ (SafeWebSocketConnection).

Ngăn chặn hiện tượng xung đột dữ liệu (race-conditions) khi nhiều coroutine (ASR, Translation, TTS)
cùng ghi đồng thời vào một kênh kết nối ASGI WebSocket.
"""

import asyncio
import json
from typing import Any, Dict
from fastapi import WebSocket, WebSocketDisconnect

from backend.utils.logger import get_logger

logger = get_logger("ws.connection")


class SafeWebSocketConnection:
    """Bao bọc WebSocket với asyncio.Lock để tuần tự hóa việc gửi tin nhắn ra ngoài."""

    def __init__(self, ws: WebSocket):
        self._ws: WebSocket = ws
        self._send_lock: asyncio.Lock = asyncio.Lock()
        self._is_closed: bool = False

    @property
    def raw_ws(self) -> WebSocket:
        """Truy cập đối tượng WebSocket FastAPI gốc."""
        return self._ws

    @property
    def is_closed(self) -> bool:
        """Trạng thái kết nối đã đóng hay chưa."""
        return self._is_closed

    async def accept(self) -> None:
        """Chấp nhận bắt tay WebSocket."""
        await self._ws.accept()

    async def receive(self) -> Dict[str, Any]:
        """Nhận khung thông điệp ASGI thô."""
        return await self._ws.receive()

    async def send_text(self, text: str) -> bool:
        """Gửi chuỗi văn bản tuần tự hóa dưới lock an toàn."""
        if self._is_closed:
            return False

        try:
            async with self._send_lock:
                if self._is_closed:
                    return False
                await self._ws.send_text(text)
            return True
        except (WebSocketDisconnect, RuntimeError):
            self._is_closed = True
            return False
        except Exception as e:
            logger.debug(f"SafeWebSocketConnection send_text error: {e}")
            self._is_closed = True
            return False

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        """Tuần tự hóa payload thành JSON và gửi an toàn."""
        if self._is_closed:
            return False
        try:
            raw_text = json.dumps(payload, ensure_ascii=False)
            return await self.send_text(raw_text)
        except (TypeError, ValueError) as e:
            logger.error(f"Lỗi tuần tự hóa JSON payload: {e}")
            return False

    async def close(self, code: int = 1000) -> None:
        """Đóng kết nối an toàn và giải phóng lock."""
        if self._is_closed:
            return
        self._is_closed = True
        try:
            await self._ws.close(code=code)
        except Exception:
            pass
