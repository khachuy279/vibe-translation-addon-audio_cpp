"""WebSocket connection handler for streaming audio -> VAD -> ASR -> Translate -> Firefox Extension."""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

from backend_cpp.config import config
from backend_cpp.translation.context_manager import ContextManager
from backend_cpp.translation.translator import translate_sentence
from backend_cpp.tts import get_tts_engine
from backend_cpp.ws.connection import SafeWebSocketConnection
from backend_cpp.ws.dedup import TranslationDedupState, TTSDedupState
from backend_cpp.ws.frame_protocol import parse_audio_frame
from backend_cpp.ws.serializers import (
    make_pong_msg,
    make_utterance_update_msg,
    make_translation_msg,
    make_tts_audio_msg,
)
from backend_cpp.asr.sentence_segmenter import count_content_tokens
from backend_cpp.ws.session_state import SessionState
from backend_cpp.utils.perf_profiler import perf
from backend_cpp.utils.audio_dumper import dump_ingress_chunk, close_session_dumper


logger = logging.getLogger(__name__)

# Backward compatibility aliases
_make_pong_msg = make_pong_msg
_make_utterance_update_msg = make_utterance_update_msg
_make_translation_msg = make_translation_msg
_make_tts_audio_msg = make_tts_audio_msg


async def handle_ws(ws: WebSocket) -> None:
    """Main WebSocket entry point managing connection lifecycle."""
    safe_ws = SafeWebSocketConnection(ws)
    await safe_ws.accept()

    session = SessionState(safe_ws)
    perf.increment_counter("ws.sessions_connected")
    perf.record_resource_checkpoint(f"session_start_{session.session_id[:8]}")
    logger.info(f"Session {session.session_id}: Connected from extension")

    session.init_components()

    asr_task = asyncio.create_task(_stream_asr_tokens(session), name=f"asr_{session.session_id}")
    translation_task = asyncio.create_task(_translation_worker(session), name=f"trans_{session.session_id}")
    tts_task = asyncio.create_task(_tts_worker(session), name=f"tts_{session.session_id}")
    workers = [asr_task, translation_task, tts_task]

    try:
        while True:
            # Check if any worker task crashed unexpectedly
            for t in workers:
                if t.done() and not t.cancelled() and t.exception():
                    logger.error(f"Session {session.session_id}: Worker {t.get_name()} crashed: {t.exception()}")
                    # pyrefly: ignore [bad-raise]
                    raise t.exception()

            message = await safe_ws.receive()
            msg_type = message.get("type", "")

            if msg_type == "websocket.disconnect":
                logger.info(f"Session {session.session_id}: client disconnected")
                break
            elif "text" in message:
                await _handle_text_message(session, message["text"])
            elif "bytes" in message:
                await _handle_binary_message(session, message["bytes"])

    except WebSocketDisconnect:
        logger.info(f"Session {session.session_id}: client disconnected cleanly")
    except Exception as e:
        logger.debug(f"Session {session.session_id}: {e}")
    finally:
        # Graceful drain: allow pending translation & TTS items to complete briefly
        try:
            await session.drain_queues(timeout=0.5)
        except Exception:
            pass

        for t in workers:
            t.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        await session.cleanup()
        close_session_dumper(session.session_id)
        await safe_ws.close()
        perf.record_resource_checkpoint(f"session_end_{session.session_id[:8]}")
        if config.perf.dump_report_on_disconnect:
            perf.dump_report_file(config.perf.report_file)
        logger.info(f"Session {session.session_id}: closed and cleaned up")


async def _handle_text_message(session: SessionState, text: str) -> None:
    """Handle control JSON messages (config updates, ping/pong)."""
    try:
        msg = json.loads(text)
        action = msg.get("type") or msg.get("action", "")

        if action in ("set_config", "configure"):
            session.apply_config(msg)
            logger.info(
                f"Session {session.session_id}: config updated "
                f"(vad={session.config.get('vad_engine')}, "
                f"threshold={session.config.get('vad_threshold')}, "
                f"silence={session.config.get('silence_duration_ms')}ms, "
                f"min_words={session.config.get('min_words_to_commit')}, "
                f"target={session.config.get('target_lang')}, "
                f"source={session.config.get('source_lang')}, "
                f"tts_enabled={session.config.get('tts_enabled')}, "
                f"tts_voice={session.config.get('tts_voice')})"
            )
            if session.config.get("tts_enabled"):
                prewarm_task = asyncio.create_task(
                    get_tts_engine().prewarm(), name=f"tts_prewarm_{session.session_id}"
                )
                prewarm_task.add_done_callback(
                    lambda t: logger.error(f"TTS prewarm failed: {t.exception()}")
                    if not t.cancelled() and t.exception()
                    else None
                )

        elif action == "ping":
            pong_payload = make_pong_msg(msg.get("timestamp", 0))
            await session.send_json(pong_payload)

    except json.JSONDecodeError:
        pass


def _process_binary_chunk(session: SessionState, data: bytes) -> None:
    """CPU worker parsing audio headers and feeding validated PCM chunks to VAD."""
    perf.increment_counter("ws.audio_chunks_received")
    pcm_data, capture_ts, chunk_idx = parse_audio_frame(data)

    if pcm_data is None:
        return

    current_idx = session.record_chunk(chunk_idx)
    dump_ingress_chunk(session.session_id, current_idx, pcm_data)

    if session.vad_processor:
        session.vad_processor.feed_chunk(pcm_data, capture_timestamp=capture_ts)



async def _handle_binary_message(session: SessionState, data: bytes) -> None:
    """Decode and feed audio chunk asynchronously via thread pool."""
    await asyncio.to_thread(_process_binary_chunk, session, data)


async def _stream_asr_tokens(session: SessionState) -> None:
    """Stream live preview and final ASR text to client."""
    engine = session.asr_engine
    if not engine:
        return

    try:
        async for msg in engine.stream_tokens():
            if msg.get("type") == "utterance_update":
                utt_id = msg.get("utterance_id", "")
                text = msg.get("text", "")
                is_final = msg.get("is_final", False)
                stable_text = msg.get("stable_text", "")
                unstable_text = msg.get("unstable_text", "")

                out_msg = make_utterance_update_msg(
                    utt_id=utt_id,
                    text=text,
                    translated="..." if is_final else "",
                    is_final=is_final,
                    stable_text=stable_text,
                    unstable_text=unstable_text,
                )

                sent = await session.send_json(out_msg)
                if not sent:
                    return

                if is_final and text and session.translation_queue:
                    val = session.config.get("min_words_to_commit")
                    min_words = int(val if val is not None else config.sentence.min_words_to_commit)
                    token_cnt = count_content_tokens(text)
                    if token_cnt < min_words:
                        logger.info(f"🚫 [TRANSLATE FILTER] Dropped short utterance ({token_cnt} < {min_words} words): '{text}'")
                        perf.increment_counter("translation.short_words_filtered")
                    else:
                        target_l = session.config.get("target_lang") or config.translation.target_lang
                        try:
                            session.translation_queue.put_nowait({
                                "utterance_id": utt_id,
                                "text": text,
                                "source_lang": msg.get("language", session.config.get("source_lang", "auto")),
                                "target_lang": target_l,
                                "_queued_at": time.perf_counter(),
                            })
                        except asyncio.QueueFull:
                            logger.warning(f"Session {session.session_id}: Translation queue full")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"ASR stream worker error: {e}", exc_info=True)


async def _process_translation_item(
    session: SessionState,
    ctx: ContextManager,
    item: Dict[str, Any],
    dedup_state: TranslationDedupState,
) -> None:
    """Process a single translation queue item with safe deduplication and dual delivery."""
    text = item.get("text", "")
    utt_id = item.get("utterance_id", "")
    src_lang = item.get("source_lang", "auto")
    tgt_lang = session.config.get("target_lang") or item.get("target_lang", "vi")
    queued_at = item.get("_queued_at", time.perf_counter())

    if not text:
        return

    # Queue delay tracking
    queue_wait_ms = (time.perf_counter() - queued_at) * 1000.0
    perf.record_metric("translation", "queue_wait_ms", queue_wait_ms)

    clean_utt = (utt_id or "unknown")[:8]
    # Deduplication check
    if dedup_state.is_duplicate(text):
        perf.increment_counter("translation.dedup_skipped")
        logger.info(f"🚫 [TRANSLATE DEDUP] [utt={clean_utt}] Skipped duplicate translation request: '{text}'")
        return

    start_t = time.monotonic()
    context_str = ctx.get_context_str() if config.translation.use_context else ""

    res = await translate_sentence(
        text=text,
        source_lang=src_lang,
        target_lang=tgt_lang,
        context=context_str,
    )
    elapsed_ms = int((time.monotonic() - start_t) * 1000)
    translated = res.get("translated_text", text)

    logger.info(f"[TRANSLATE] [utt={clean_utt}] ({src_lang} -> {tgt_lang} in {elapsed_ms}ms): '{text}' => '{translated}'")
    ctx.add(text, translated)

    # 1. Dedicated "translation" message
    trans_msg = make_translation_msg(
        utt_id=utt_id,
        translated=translated,
        elapsed_ms=elapsed_ms,
        target_lang=tgt_lang,
    )

    # 2. Updated "utterance_update" message for complete backward compatibility
    update_msg = make_utterance_update_msg(
        utt_id=utt_id,
        text=text,
        translated=translated,
        is_final=True,
    )

    sent1 = await session.send_json(trans_msg)
    sent2 = await session.send_json(update_msg)
    if not sent1 or not sent2:
        return

    # Measure End-to-End latency from ASR commit -> Subtitle delivery
    e2e_sub_ms = (time.perf_counter() - queued_at) * 1000.0
    perf.record_metric("pipeline", "e2e_asr_to_sub_ms", e2e_sub_ms)
    perf.increment_counter("pipeline.subtitles_delivered")

    if config.perf.enabled and e2e_sub_ms > 400.0:
        logger.debug(f"⏱️ [PERF_E2E] ASR->Sub delivered in {e2e_sub_ms:.1f}ms (queue_wait={queue_wait_ms:.1f}ms, infer={elapsed_ms}ms)")

    # 3. Queue translated text for Voice Cloning TTS if enabled
    if session.config.get("tts_enabled"):
        if session.tts_queue and translated:
            try:
                session.tts_queue.put_nowait({
                    "utterance_id": utt_id,
                    "text": translated,
                    "voice": session.config.get("tts_voice"),
                    "speed": float(session.config.get("tts_speed", 1.0)),
                    "_queued_at": time.perf_counter(),
                })
                logger.info(f"🔊 [TTS QUEUED] Queued for synthesis: '{translated}'")
            except asyncio.QueueFull:
                logger.warning(f"Session {session.session_id}: TTS queue full, dropping sentence")
    # else:
    #     logger.debug(
    #         f"Session {session.session_id}: TTS disabled (tts_enabled={session.config.get('tts_enabled')}), skipped: '{translated}'"
    #     )


async def _translation_worker(session: SessionState) -> None:
    """Sequential worker for local GGUF translation ensuring task_done is always called."""
    ctx = ContextManager(window_size=config.translation.context_window)
    dedup_state = TranslationDedupState()

    while True:
        try:
            item = await session.translation_queue.get()
            try:
                await _process_translation_item(session, ctx, item, dedup_state)
            finally:
                session.translation_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Session {session.session_id}: Translation worker error: {e}", exc_info=True)
            await asyncio.sleep(0.05)


async def _process_tts_item(
    session: SessionState,
    tts: Any,
    item: Dict[str, Any],
    dedup_state: TTSDedupState,
) -> None:
    """Process a single TTS queue item with deduplication and synthesis."""
    text = item.get("text", "")
    utt_id = item.get("utterance_id", "")
    voice = item.get("voice") or session.config.get("tts_voice")
    speed = float(item.get("speed") or session.config.get("tts_speed", 1.0))
    queued_at = item.get("_queued_at", time.perf_counter())

    if not text:
        return

    # Queue delay tracking
    tts_queue_wait_ms = (time.perf_counter() - queued_at) * 1000.0
    perf.record_metric("tts", "queue_wait_ms", tts_queue_wait_ms)

    if dedup_state.is_duplicate(text):
        perf.increment_counter("tts.dedup_skipped")
        logger.info(f"🚫 [TTS DEDUP] Skipped duplicate TTS synthesis: '{text}'")
        return

    try:
        t_tts_start = time.perf_counter()
        audio_b64, duration_sec = await tts.synthesize_clone(
            text=text,
            voice_id=voice,
            speed=speed,
        )
        synthesis_ms = (time.perf_counter() - t_tts_start) * 1000.0
        perf.record_metric("tts", "synthesis_ms", synthesis_ms)
        tts_rtf = (synthesis_ms / 1000.0) / max(0.001, duration_sec)
        perf.record_metric("tts", "rtf", tts_rtf)
        perf.increment_counter("tts.synthesized_utterances")

        if audio_b64:
            out_msg = make_tts_audio_msg(
                utt_id=utt_id,
                text=text,
                audio_b64=audio_b64,
                duration_sec=duration_sec,
                sample_rate=tts.sample_rate,
            )
            sent = await session.send_json(out_msg)
            if sent:
                e2e_tts_ms = (time.perf_counter() - queued_at) * 1000.0
                perf.record_metric("pipeline", "e2e_sub_to_tts_ms", e2e_tts_ms)
                logger.info(f"🔊 [TTS SENT] Sent {duration_sec:.2f}s audio to client for utt '{utt_id}' (synth={synthesis_ms:.1f}ms, RTF={tts_rtf:.2f})")
            else:
                logger.warning(f"Session {session.session_id}: Failed to send TTS audio to client")
    except Exception as e:
        logger.error(f"Session {session.session_id}: TTS synthesis error: {e}", exc_info=True)


async def _tts_worker(session: SessionState) -> None:
    """Sequential worker for OmniVoice Voice Cloning ensuring task_done is always called."""
    tts: Optional[Any] = None
    dedup_state = TTSDedupState()

    while True:
        try:
            item = await session.tts_queue.get()
            try:
                if not session.config.get("tts_enabled"):
                    continue
                if tts is None:
                    tts = get_tts_engine()
                await _process_tts_item(session, tts, item, dedup_state)
            finally:
                session.tts_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as outer_err:
            logger.error(f"Session {session.session_id}: TTS worker loop error: {outer_err}", exc_info=True)
            await asyncio.sleep(0.05)


__all__ = ["handle_ws"]
