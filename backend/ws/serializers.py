"""Module tuần tự hóa (Serialization) các thông điệp JSON gửi qua WebSocket."""

import time
from typing import Any, Dict


def make_pong_msg(timestamp: Any) -> Dict[str, Any]:
    """Tạo gói tin phản hồi Ping/Pong."""
    now = time.time()
    return {
        "type": "pong",
        "timestamp": timestamp,
        "server_time": now,
        "serverTime": now,
    }


def make_utterance_update_msg(
    utt_id: str,
    text: str,
    translated: str = "",
    is_final: bool = False,
    stable_text: str = "",
    unstable_text: str = "",
    filtered: bool = False,
) -> Dict[str, Any]:
    """Tạo gói tin cập nhật kết quả nhận dạng giọng nói ASR trực tiếp."""
    return {
        "type": "utterance_update",
        "utterance_id": utt_id,
        "utteranceId": utt_id,
        "original": text,
        "ui_text": text,
        "text": text,
        "stable_text": stable_text,
        "stableText": stable_text,
        "unstable_text": unstable_text,
        "unstableText": unstable_text,
        "translated": translated,
        "is_final": is_final,
        "isFinal": is_final,
        "filtered": filtered,
    }



def make_translation_msg(
    utt_id: str,
    translated: str,
    elapsed_ms: int,
    target_lang: str,
) -> Dict[str, Any]:
    """Tạo gói tin phụ đề bản dịch hoàn chỉnh."""
    return {
        "type": "translation",
        "sentence_id": utt_id,
        "sentenceId": utt_id,
        "utterance_id": utt_id,
        "utteranceId": utt_id,
        "translated": translated,
        "text": translated,
        "status": "ok",
        "translate_time_ms": elapsed_ms,
        "translateTimeMs": elapsed_ms,
        "target_lang": target_lang,
        "targetLang": target_lang,
    }


def make_tts_audio_msg(
    utt_id: str,
    text: str,
    audio_b64: str,
    duration_sec: float,
    sample_rate: int,
) -> Dict[str, Any]:
    """Tạo gói tin âm thanh Voice Cloning TTS."""
    return {
        "type": "tts_audio",
        "utterance_id": utt_id,
        "utteranceId": utt_id,
        "sentence_id": utt_id,
        "sentenceId": utt_id,
        "text": text,
        "audio": audio_b64,
        "duration_sec": duration_sec,
        "durationSec": duration_sec,
        "sample_rate": sample_rate,
        "sampleRate": sample_rate,
        "format": "audio/wav",
    }
