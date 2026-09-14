"""JSON message serializers for Firefox Extension protocol."""

import time
from typing import Any, Dict, Optional


def make_utterance_update_msg(
    utt_id: str,
    text: str,
    translated: str = "",
    is_final: bool = False,
    stable_text: str = "",
    unstable_text: str = "",
) -> Dict[str, Any]:
    """Serialize utterance_update event compatible with extension overlay."""
    return {
        "type": "utterance_update",
        "utterance_id": utt_id,
        "text": text,
        "translated": translated,
        "is_final": is_final,
        "stable_text": stable_text,
        "unstable_text": unstable_text,
        "timestamp": time.time(),
    }


def make_translation_msg(
    utt_id: str,
    translated: str,
    elapsed_ms: float = 0.0,
    target_lang: str = "vi",
) -> Dict[str, Any]:
    """Serialize translation event for committed sentences."""
    return {
        "type": "translation",
        "utterance_id": utt_id,
        "translated": translated,
        "target_lang": target_lang,
        "elapsed_ms": elapsed_ms,
        "timestamp": time.time(),
    }


def make_tts_audio_msg(
    utt_id: str,
    text: str,
    audio_b64: str,
    duration_sec: float,
    sample_rate: int = 24000,
) -> Dict[str, Any]:
    """Serialize tts_audio event delivering cloned audio to extension."""
    return {
        "type": "tts_audio",
        "utterance_id": utt_id,
        "text": text,
        "audio": audio_b64,
        "audio_base64": audio_b64,
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "timestamp": time.time(),
    }


def make_pong_msg(client_timestamp: float = 0.0) -> Dict[str, Any]:
    """Serialize pong heartbeat response."""
    return {
        "type": "pong",
        "client_timestamp": client_timestamp,
        "server_timestamp": time.time(),
    }
