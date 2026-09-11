"""WebSocket subpackage for backend_cpp."""

from backend_cpp.ws.connection import SafeWebSocketConnection
from backend_cpp.ws.dedup import SlidingWindowDedup, TranslationDedupState, TTSDedupState
from backend_cpp.ws.frame_protocol import parse_audio_frame
from backend_cpp.ws.serializers import (
    make_pong_msg,
    make_utterance_update_msg,
    make_translation_msg,
    make_tts_audio_msg,
)
from backend_cpp.ws.session_state import SessionState
from backend_cpp.ws.ws_handler import handle_ws

__all__ = [
    "handle_ws",
    "SafeWebSocketConnection",
    "SessionState",
    "SlidingWindowDedup",
    "TranslationDedupState",
    "TTSDedupState",
    "parse_audio_frame",
    "make_pong_msg",
    "make_utterance_update_msg",
    "make_translation_msg",
    "make_tts_audio_msg",
]
