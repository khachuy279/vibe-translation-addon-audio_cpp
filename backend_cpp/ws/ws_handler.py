"""WebSocket connection handler for streaming audio -> VAD -> ASR -> Translate -> Firefox Extension."""

import asyncio
import itertools
import json
import logging
import threading
import time
from collections import OrderedDict
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

# Admission control (audit finding P1-04).
#
# ⚠️  DO NOT change this back to "reject the newcomer". See INV-9 in
#     extension_firefox/CAPTURE_FRAME_NOTES.md and the guard test
#     backend_cpp/tests/test_extension_invariants.py::test_backend_admission_guard_is_newest_wins
#
# The extension elects a single capture owner so that exactly one frame starts a
# capture session. This registry is the backend's defense-in-depth guard: it bounds
# how many sessions can exist at once.
#
# Policy is NEWEST-WINS: when the limit is reached, the OLDEST session is closed to make
# room for the newcomer. Rejecting the newcomer instead would let one stale connection
# (a forgotten tab, or the extension's background bridge holding a socket open) lock the
# user out of starting capture forever. Superseding guarantees START always works while
# still keeping concurrency bounded.
_active_sessions: "OrderedDict[int, Any]" = OrderedDict()
_active_sessions_lock = threading.Lock()
_session_sequence = itertools.count(1)


def get_active_session_count() -> int:
    """Return the number of currently admitted WebSocket sessions."""
    with _active_sessions_lock:
        return len(_active_sessions)


async def _supersede_excess_sessions(max_sessions: int) -> int:
    """Close oldest sessions so a newcomer fits within ``max_sessions``.

    Returns the number of sessions that were asked to close.
    """
    victims = []
    with _active_sessions_lock:
        while len(_active_sessions) >= max_sessions:
            _, oldest = _active_sessions.popitem(last=False)
            victims.append(oldest)

    for victim in victims:
        try:
            await victim.close(code=1013, reason="superseded_by_new_session")
        except Exception:
            pass
    return len(victims)


async def handle_ws(ws: WebSocket) -> None:
    """Main WebSocket entry point managing connection lifecycle."""
    safe_ws = SafeWebSocketConnection(ws)
    await safe_ws.accept()

    # Admission control (audit finding P1-04), newest-wins.
    # A stale/superseded session is closed so the newcomer always gets in; see the
    # policy note on _active_sessions above.
    max_sessions = int(getattr(config.ws, "max_sessions", 1))
    slot = next(_session_sequence)

    if max_sessions > 0:
        superseded = await _supersede_excess_sessions(max_sessions)
        if superseded:
            logger.warning(
                f"Superseding {superseded} older WebSocket session(s) to admit a new one "
                f"(max_sessions={max_sessions}). A stale session must never block START."
            )
            perf.increment_counter("ws.sessions_superseded", count=superseded)

    with _active_sessions_lock:
        _active_sessions[slot] = safe_ws

    session = SessionState(safe_ws)
    perf.increment_counter("ws.sessions_connected")
    perf.record_resource_checkpoint(f"session_start_{session.session_id[:8]}")
    logger.info(
        f"Session {session.session_id}: Connected from extension "
        f"(active sessions: {get_active_session_count()})"
    )

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
        try:
            # Graceful drain: allow pending translation & TTS items to complete briefly
            try:
                await session.drain_queues(timeout=0.5)
            except Exception:
                pass

            for t in workers:
                t.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

            await session.cleanup()
            session.close()
            close_session_dumper(session.session_id)
            await safe_ws.close()
            perf.record_resource_checkpoint(f"session_end_{session.session_id[:8]}")
            if config.perf.dump_report_on_disconnect:
                perf.dump_report_file(config.perf.report_file)
            logger.info(f"Session {session.session_id}: closed and cleaned up")
        finally:
            # Always release the admission slot, even if teardown raised; otherwise one
            # failure would permanently occupy the slot.
            with _active_sessions_lock:
                _active_sessions.pop(slot, None)


async def _handle_text_message(session: SessionState, text: str) -> None:
    """Handle control JSON messages (config updates, ping/pong)."""
    try:
        msg = json.loads(text)
        action = msg.get("type") or msg.get("action", "")

        if action in ("set_config", "configure"):
            session.apply_config(msg)
            if "epoch" in msg and msg["epoch"] is not None:
                cfg_epoch = int(msg["epoch"])
                if cfg_epoch > session.current_epoch:
                    logger.info(
                        f"Session {session.session_id}: Synchronizing epoch from {session.current_epoch} to {cfg_epoch} via config"
                    )
                    session.handle_stream_reset(epoch=cfg_epoch, reason="config_sync")
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

        elif action == "stream_reset":
            epoch = int(msg.get("epoch", session.current_epoch + 1))
            reason = str(msg.get("reason", "reset"))
            media_time = float(msg.get("media_time", msg.get("mediaTime", 0.0)))
            session.handle_stream_reset(epoch=epoch, reason=reason, media_time=media_time)

        elif action == "client_render_ack":
            session.transport_telemetry.record_render_ack(
                epoch=int(msg.get("epoch", 0)),
                utterance_id=str(msg.get("utterance_id", "")),
                render_revision=int(msg.get("render_revision", 1)),
                media_end_time=float(msg.get("media_end_time", 0.0)),
                video_current_time=float(msg.get("video_current_time", 0.0)),
                speech_offset_to_visible_lag_sec=float(msg.get("speech_offset_to_visible_lag_sec", 0.0)),
                client_render_cost_ms=float(msg.get("client_render_cost_ms", 0.0)),
                asr_commit_wall_time=float(msg.get("asr_commit_wall_time", 0.0)),
            )

        elif action == "ping":
            pong_payload = make_pong_msg(msg.get("timestamp", 0))
            await session.send_json(pong_payload)

    except json.JSONDecodeError:
        pass


def _process_binary_chunk(session: SessionState, data: bytes) -> None:
    """CPU worker parsing audio headers and feeding validated PCM chunks to VAD."""
    perf.increment_counter("ws.audio_chunks_received")
    frame = parse_audio_frame(data)
    pcm_data = frame.pcm

    if pcm_data is None:
        return

    # Generation barrier: drop frames from older epochs
    if frame.epoch < session.current_epoch:
        logger.debug(f"Dropped frame from older epoch {frame.epoch} (current={session.current_epoch})")
        return

    # Generation barrier fast-forward: if incoming audio frame has a newer epoch than current session,
    # advance session epoch so downstream VAD & ASR emissions stay strictly synchronized.
    if frame.epoch > session.current_epoch:
        logger.info(
            f"Session {session.session_id}: Advancing epoch from {session.current_epoch} to {frame.epoch} from audio frame"
        )
        session.handle_stream_reset(epoch=frame.epoch, reason="frame_epoch_advance", media_time=frame.media_start_time)

    accepted = session.record_chunk(
        chunk_idx=frame.chunk_index,
        pcm_len=len(pcm_data),
        expected_cadence_ms=frame.chunk_duration_ms,
    )
    if not accepted:
        # Dropped before VAD (duplicate or out-of-order)
        return

    dump_ingress_chunk(session.session_id, frame.chunk_index or 0, pcm_data)

    if session.vad_processor:
        session.vad_processor.feed_chunk(
            pcm_data,
            capture_timestamp=frame.capture_timestamp,
            media_start_time=frame.media_start_time,
            media_end_time=frame.media_end_time,
            epoch=frame.epoch,
        )



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
            msg_epoch = msg.get("epoch")
            # Strict generation barrier: reject if epoch is missing or different
            if msg_epoch is None or msg_epoch != session.current_epoch:
                logger.debug(f"Generation barrier: dropped stale ASR msg with epoch={msg_epoch} != current={session.current_epoch}")
                continue

            if msg.get("type") == "utterance_update":
                utt_id = msg.get("utterance_id", "")
                text = msg.get("text", "")
                is_final = msg.get("is_final", False)
                stable_text = msg.get("stable_text", "")
                unstable_text = msg.get("unstable_text", "")
                media_start = msg.get("media_start_time", 0.0)
                media_end = msg.get("media_end_time", 0.0)
                commit_time = msg.get("asr_commit_wall_time", time.time())

                out_msg = make_utterance_update_msg(
                    utt_id=utt_id,
                    text=text,
                    translated="..." if is_final else "",
                    is_final=is_final,
                    stable_text=stable_text,
                    unstable_text=unstable_text,
                    epoch=msg_epoch,
                    media_start_time=media_start,
                    media_end_time=media_end,
                    asr_commit_wall_time=commit_time,
                )

                if msg_epoch is None or msg_epoch != session.current_epoch:
                    continue

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
                                "epoch": msg_epoch,
                                "media_start_time": media_start,
                                "media_end_time": media_end,
                                "asr_commit_wall_time": commit_time,
                                "_queued_at": time.perf_counter(),
                            })
                        except asyncio.QueueFull:
                            logger.warning(f"Session {session.session_id}: Translation queue full")
                            perf.increment_counter("translation.queue_full_dropped")

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
    item_epoch = item.get("epoch")
    if item_epoch is None or item_epoch != session.current_epoch:
        logger.debug(f"Generation barrier: Dropped translation item with epoch={item_epoch} != current={session.current_epoch}")
        return

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

    # Generation barrier: verify session epoch has not advanced during translation inference
    if item_epoch is None or item_epoch != session.current_epoch:
        logger.debug(f"Generation barrier: Discarded completed translation from old epoch {item_epoch} (current={session.current_epoch})")
        return

    # 1. Dedicated "translation" message
    trans_msg = make_translation_msg(
        utt_id=utt_id,
        translated=translated,
        elapsed_ms=elapsed_ms,
        target_lang=tgt_lang,
        epoch=item_epoch,
        media_start_time=item.get("media_start_time", 0.0),
        media_end_time=item.get("media_end_time", 0.0),
        asr_commit_wall_time=item.get("asr_commit_wall_time", 0.0),
    )

    # 2. Updated "utterance_update" message for complete backward compatibility
    update_msg = make_utterance_update_msg(
        utt_id=utt_id,
        text=text,
        translated=translated,
        is_final=True,
        epoch=item_epoch,
        media_start_time=item.get("media_start_time", 0.0),
        media_end_time=item.get("media_end_time", 0.0),
        asr_commit_wall_time=item.get("asr_commit_wall_time", 0.0),
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
                    "epoch": item_epoch,
                    "_queued_at": time.perf_counter(),
                })
                logger.info(f"🔊 [TTS QUEUED] Queued for synthesis: '{translated}'")
            except asyncio.QueueFull:
                logger.warning(f"Session {session.session_id}: TTS queue full, dropping sentence")
                perf.increment_counter("tts.queue_full_dropped")
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
    item_epoch = item.get("epoch")
    if item_epoch is None or item_epoch != session.current_epoch:
        logger.debug(f"Generation barrier: Dropped TTS item with epoch={item_epoch} != current={session.current_epoch}")
        return

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

        # Generation barrier: verify session epoch has not changed during synthesis
        if item_epoch is None or item_epoch != session.current_epoch:
            logger.debug(f"Generation barrier: Discarded completed TTS from old epoch {item_epoch} (current={session.current_epoch})")
            return

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
