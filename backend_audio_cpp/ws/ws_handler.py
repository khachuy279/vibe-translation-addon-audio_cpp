"""WebSocket connection handler for streaming audio -> Silero VAD -> ASR -> HyMT Translation -> OmniVoice TTS."""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect

from backend_audio_cpp.config import config
from backend_audio_cpp.translation.hy_translator import HyMTTranslator
from backend_audio_cpp.tts.omnivoice_engine import OmniVoiceTTSEngine
from backend_audio_cpp.ws.connection import SafeWebSocketConnection
from backend_audio_cpp.ws.dedup import TranslationDedupState, TTSDedupState
from backend_audio_cpp.ws.frame_protocol import parse_audio_frame
from backend_audio_cpp.ws.serializers import (
    make_pong_msg,
    make_translation_msg,
    make_tts_audio_msg,
    make_utterance_update_msg,
)
from backend_audio_cpp.ws.session_state import SessionState

logger = logging.getLogger("backend_audio_cpp.ws.handler")


async def handle_ws(ws: WebSocket) -> None:
    """Main WebSocket entry point managing client connection and pipeline lifecycle."""
    safe_ws = SafeWebSocketConnection(ws)
    await safe_ws.accept()

    session = SessionState(safe_ws)
    logger.info(f"Session {session.session_id}: Connected from Firefox Extension")

    translation_task = asyncio.create_task(
        _translation_worker(session), name=f"trans_{session.session_id}"
    )
    tts_task = asyncio.create_task(
        _tts_worker(session), name=f"tts_{session.session_id}"
    )
    workers = [translation_task, tts_task]

    try:
        while True:
            # Monitor workers for unhandled exceptions
            for w in workers:
                if w.done() and not w.cancelled() and w.exception():
                    logger.error(f"Worker {w.get_name()} crashed: {w.exception()}")
                    raise w.exception()

            message = await safe_ws.receive()
            msg_type = message.get("type", "")

            if msg_type == "websocket.disconnect":
                logger.info(f"Session {session.session_id}: client disconnected cleanly")
                break
            elif "text" in message:
                await _handle_text_message(session, message["text"])
            elif "bytes" in message:
                data = message["bytes"]
                pcm_data, capture_ts, chunk_idx = parse_audio_frame(data)
                if pcm_data is not None:
                    session.feed_pcm(pcm_data, capture_ts, chunk_idx)

    except WebSocketDisconnect:
        logger.info(f"Session {session.session_id}: disconnected")
    except Exception as e:
        logger.debug(f"Session {session.session_id} exception: {e}")
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await safe_ws.close()
        logger.info(f"Session {session.session_id}: closed and cleaned up")


async def _handle_text_message(session: SessionState, text: str) -> None:
    """Handle control JSON messages (set_config, ping)."""
    try:
        msg = json.loads(text)
        action = msg.get("type") or msg.get("action", "")

        if action in ("set_config", "configure"):
            session.apply_config(msg)

        elif action == "ping":
            pong = make_pong_msg(msg.get("timestamp", 0))
            await session.send_json(pong)

    except json.JSONDecodeError:
        pass


async def _translation_worker(session: SessionState) -> None:
    """Sequential worker for GPU translation using Hunyuan-MT2 7B."""
    translator = HyMTTranslator.get_instance()
    dedup = TranslationDedupState()

    while True:
        try:
            item = await session.translation_queue.get()
            try:
                utt_id = item.get("utterance_id", "")
                text = item.get("text", "")
                src_lang = item.get("source_lang", "auto")
                tgt_lang = item.get("target_lang", "vi")

                if not text or dedup.is_duplicate(text):
                    continue

                # Run translation in thread pool
                res = await asyncio.to_thread(
                    translator.translate,
                    text,
                    tgt_lang,
                    src_lang,
                    utt_id,
                )
                translated = res.translated_text

                # 1. Send dedicated translation event
                trans_msg = make_translation_msg(
                    utt_id=utt_id,
                    translated=translated,
                    elapsed_ms=res.latency_ms,
                    target_lang=tgt_lang,
                )
                await session.send_json(trans_msg)

                # 2. Send updated utterance_update event
                update_msg = make_utterance_update_msg(
                    utt_id=utt_id,
                    text=text,
                    translated=translated,
                    is_final=True,
                    stable_text=text,
                )
                await session.send_json(update_msg)

                # 3. Enqueue for Voice Cloning TTS if enabled
                if session.config.get("tts_enabled") and translated:
                    try:
                        session.tts_queue.put_nowait({
                            "utterance_id": utt_id,
                            "text": translated,
                            "voice": session.config.get("tts_voice"),
                            "speed": session.config.get("tts_speed", 1.0),
                            "_queued_at": time.perf_counter(),
                        })
                    except asyncio.QueueFull:
                        logger.warning("TTS queue full, dropping sentence")

            finally:
                session.translation_queue.task_done()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Translation worker error: {e}", exc_info=True)
            await asyncio.sleep(0.05)


async def _tts_worker(session: SessionState) -> None:
    """Sequential worker for OmniVoice Zero-Shot Voice Cloning."""
    tts_engine = OmniVoiceTTSEngine.get_instance()
    dedup = TTSDedupState()

    while True:
        try:
            item = await session.tts_queue.get()
            try:
                if not session.config.get("tts_enabled"):
                    continue

                utt_id = item.get("utterance_id", "")
                text = item.get("text", "")
                voice = item.get("voice") or session.config.get("tts_voice")

                if not text or dedup.is_duplicate(text):
                    continue

                res = await asyncio.to_thread(
                    tts_engine.synthesize,
                    text,
                    voice,
                    utt_id,
                )

                if res.audio_wav_bytes:
                    audio_b64 = base64.b64encode(res.audio_wav_bytes).decode("ascii")
                    msg = make_tts_audio_msg(
                        utt_id=utt_id,
                        text=text,
                        audio_b64=audio_b64,
                        duration_sec=res.duration_sec,
                        sample_rate=res.sample_rate,
                    )
                    await session.send_json(msg)
                    logger.info(
                        f"🔊 [TTS SENT] Sent {res.duration_sec:.2f}s audio to client for utt '{utt_id}' "
                        f"(synth={res.synth_time_ms:.1f}ms, RTF={res.rtf:.2f})"
                    )

            finally:
                session.tts_queue.task_done()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"TTS worker error: {e}", exc_info=True)
            await asyncio.sleep(0.05)
