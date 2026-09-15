"""Trình xử lý kết nối WebSocket chính (WebSocket Handler) kết nối toàn bộ Pipeline.

Chu trình hoạt động:
Audio Stream -> VAD -> ASR Streaming (transcribe.cpp) -> Commit Manager -> Translation -> TTS -> WebSocket Client
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

from backend.config import config
from backend.translation.context import TranslationContextTracker
from backend.translation.dedup import TranslationDeduplicator
from backend.translation.engine import GGUFTranslationEngine
from backend.tts import get_tts_engine, TTSDedupState
from backend.ws.connection import SafeWebSocketConnection
from backend.ws.protocol import parse_audio_frame
from backend.ws.serializers import (
    make_pong_msg,
    make_utterance_update_msg,
    make_translation_msg,
    make_tts_audio_msg,
)
from backend.core.commit_manager import count_content_tokens
from backend.ws.session import SessionState
from backend.core.metrics import metrics_collector
from backend.utils.logger import get_logger

logger = get_logger("ws.handler")


async def handle_ws(ws: WebSocket) -> None:
    """Điểm nhập kết nối WebSocket chính quản lý toàn bộ vòng đời phiên."""
    safe_ws = SafeWebSocketConnection(ws)
    await safe_ws.accept()

    session = SessionState(safe_ws)
    metrics_collector.increment_counter("ws.sessions_connected")
    metrics_collector.record_checkpoint(f"session_start_{session.session_id[:8]}")
    logger.info(f"🔗 Session {session.session_id[:8]}: Đã kết nối từ client")

    session.init_components()

    asr_task = asyncio.create_task(_stream_asr_tokens(session), name=f"asr_{session.session_id[:8]}")
    translation_task = asyncio.create_task(_translation_worker(session), name=f"trans_{session.session_id[:8]}")
    tts_task = asyncio.create_task(_tts_worker(session), name=f"tts_{session.session_id[:8]}")
    workers = [asr_task, translation_task, tts_task]

    t_cleanup_start = None
    try:
        while True:
            # Kiểm tra xem có worker nào bị crash bất ngờ không
            for t in workers:
                if t.done() and not t.cancelled() and t.exception():
                    logger.error(f"Worker {t.get_name()} gặp sự cố: {t.exception()}")
                    raise t.exception()

            message = await safe_ws.receive()
            msg_type = message.get("type", "")

            if msg_type == "websocket.disconnect":
                logger.info(f"Session {session.session_id[:8]}: Client ngắt kết nối")
                break
            elif "text" in message:
                await _handle_text_message(session, message["text"])
            elif "bytes" in message:
                await _handle_binary_message(session, message["bytes"])

    except WebSocketDisconnect:
        logger.info(f"Session {session.session_id[:8]}: Client ngắt kết nối an toàn")
    except Exception as e:
        logger.debug(f"Session {session.session_id[:8]}: Kết thúc vòng lặp ({e})")
    finally:
        t_cleanup_start = time.perf_counter()

        # Dọn dẹp hàng đợi nhanh (< 200ms)
        try:
            await session.drain_queues(timeout=0.15)
        except Exception:
            pass

        for t in workers:
            t.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        await session.cleanup()
        await safe_ws.close()

        metrics_collector.record_checkpoint(f"session_end_{session.session_id[:8]}")
        cleanup_ms = (time.perf_counter() - t_cleanup_start) * 1000.0 if t_cleanup_start else 0.0
        logger.info(f"🏁 Session {session.session_id[:8]}: Đã đóng và giải phóng tài nguyên hoàn tất ({cleanup_ms:.2f}ms)")


async def _handle_text_message(session: SessionState, text: str) -> None:
    """Xử lý thông điệp cấu hình JSON hoặc kiểm tra kết nối ping/pong."""
    try:
        msg = json.loads(text)
        action = msg.get("type") or msg.get("action", "")

        if action in ("set_config", "configure"):
            session.apply_config(msg)
            logger.info(
                f"⚙️ Session {session.session_id[:8]}: Cập nhật cấu hình "
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
                    get_tts_engine().prewarm(), name=f"tts_prewarm_{session.session_id[:8]}"
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
    """Giải mã tiêu đề khung âm thanh và đưa mẫu PCM vào VADProcessor."""
    metrics_collector.increment_counter("ws.audio_chunks_received")
    pcm_data, capture_ts, chunk_idx = parse_audio_frame(data)

    if pcm_data is None:
        return

    session.record_chunk(chunk_idx)
    if session.vad_processor:
        session.vad_processor.feed_chunk(pcm_data, capture_timestamp=capture_ts)


async def _handle_binary_message(session: SessionState, data: bytes) -> None:
    """Xử lý khung nhị phân âm thanh bất đồng bộ."""
    await asyncio.to_thread(_process_binary_chunk, session, data)


async def _stream_asr_tokens(session: SessionState) -> None:
    """Truyền trực tiếp bản xem trước (live preview) và câu chốt (final) ASR tới Client."""
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

                # 1. Nếu là câu chốt final, kiểm tra điều kiện lọc độ dài (min_words_to_commit)
                if is_final and text:
                    val = session.config.get("min_words_to_commit")
                    min_words = int(val if val is not None else config.sentence.min_words_to_commit)
                    token_cnt = count_content_tokens(text)
                    if token_cnt < min_words:
                        logger.info(f"🚫 [TRANSLATE FILTER] Lọc bỏ câu quá ngắn ({token_cnt} < {min_words} từ): '{text}'")
                        metrics_collector.increment_counter("translation.short_words_filtered")
                        # Gửi gói tin filtered=True để Extension xóa bỏ ngay lập tức subtitle draft trên màn hình
                        out_msg = make_utterance_update_msg(
                            utt_id=utt_id,
                            text=text,
                            translated="",
                            is_final=True,
                            filtered=True,
                        )
                        await session.send_json(out_msg)
                        continue

                # 2. Câu hợp lệ hoặc bản xem trước live preview
                out_msg = make_utterance_update_msg(
                    utt_id=utt_id,
                    text=text,
                    translated="..." if is_final else "",
                    is_final=is_final,
                    stable_text=stable_text,
                    unstable_text=unstable_text,
                    filtered=False,
                )

                sent = await session.send_json(out_msg)
                if not sent:
                    return

                # 3. Đưa vào hàng đợi dịch thuật
                if is_final and text and session.translation_queue:
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
                        logger.warning(f"Session {session.session_id[:8]}: Hàng đợi Translation đầy, bỏ qua")


    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Lỗi worker stream ASR: {e}", exc_info=True)


async def _process_translation_item(
    session: SessionState,
    trans_engine: GGUFTranslationEngine,
    ctx_tracker: TranslationContextTracker,
    item: Dict[str, Any],
    dedup: TranslationDeduplicator,
) -> None:
    """Xử lý 1 bản ghi dịch thuật với bộ lọc chống trùng và gửi kết quả về client."""
    text = item.get("text", "")
    utt_id = item.get("utterance_id", "")
    src_lang = item.get("source_lang", "auto")
    tgt_lang = session.config.get("target_lang") or item.get("target_lang", "vi")
    queued_at = item.get("_queued_at", time.perf_counter())

    if not text:
        return

    queue_wait_ms = (time.perf_counter() - queued_at) * 1000.0
    metrics_collector.record_metric("translation", "queue_wait_ms", queue_wait_ms)

    clean_utt = (utt_id or "unknown")[:8]
    if dedup.is_duplicate(text):
        metrics_collector.increment_counter("translation.dedup_skipped")
        logger.info(f"🚫 [TRANSLATE DEDUP] [utt={clean_utt}] Bỏ qua câu dịch trùng lặp: '{text}'")
        return

    start_t = time.monotonic()
    context_str = ctx_tracker.get_context_str() if config.translation.use_context else ""

    res = await trans_engine.translate(
        text=text,
        source_lang=src_lang,
        target_lang=tgt_lang,
        context=context_str,
    )
    elapsed_ms = int((time.monotonic() - start_t) * 1000)
    translated = res.get("translated_text", text)

    logger.info(f"🌐 [TRANSLATE] [utt={clean_utt}] ({src_lang} -> {tgt_lang} in {elapsed_ms}ms): '{text}' => '{translated}'")
    ctx_tracker.add(text, translated)

    # 1. Gói tin translation chuyên biệt
    trans_msg = make_translation_msg(
        utt_id=utt_id,
        translated=translated,
        elapsed_ms=elapsed_ms,
        target_lang=tgt_lang,
    )

    # 2. Gói tin utterance_update cập nhật trạng thái chốt kèm bản dịch
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

    # Đo đạc End-to-End Latency từ lúc ASR commit đến khi Client nhận phụ đề
    e2e_sub_ms = (time.perf_counter() - queued_at) * 1000.0
    metrics_collector.record_metric("pipeline", "e2e_asr_to_sub_ms", e2e_sub_ms)
    metrics_collector.increment_counter("pipeline.subtitles_delivered")

    # 3. Đưa vào hàng đợi TTS nếu người dùng bật chế độ lồng tiếng
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
                logger.info(f"🔊 [TTS QUEUED] Đã đưa vào hàng đợi lồng tiếng: '{translated}'")
            except asyncio.QueueFull:
                logger.warning(f"Session {session.session_id[:8]}: Hàng đợi TTS đầy, bỏ qua câu này")


async def _translation_worker(session: SessionState) -> None:
    """Worker tuần tự dịch thuật GGUF đảm bảo không tắc nghẽn queue."""
    trans_engine = GGUFTranslationEngine.get_instance()
    ctx_tracker = TranslationContextTracker(window_size=config.translation.context_window)
    dedup = TranslationDeduplicator()

    while True:
        try:
            item = await session.translation_queue.get()
            try:
                await _process_translation_item(session, trans_engine, ctx_tracker, item, dedup)
            finally:
                session.translation_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Lỗi Translation worker: {e}", exc_info=True)
            await asyncio.sleep(0.05)


async def _process_tts_item(
    session: SessionState,
    tts: Any,
    item: Dict[str, Any],
    dedup_state: TTSDedupState,
) -> None:
    """Xử lý 1 bản ghi TTS, thực hiện Voice Cloning và gửi audio base64 về Client."""
    text = item.get("text", "")
    utt_id = item.get("utterance_id", "")
    voice = item.get("voice") or session.config.get("tts_voice")
    speed = float(item.get("speed") or session.config.get("tts_speed", 1.0))
    queued_at = item.get("_queued_at", time.perf_counter())

    if not text:
        return

    tts_queue_wait_ms = (time.perf_counter() - queued_at) * 1000.0
    metrics_collector.record_metric("tts", "queue_wait_ms", tts_queue_wait_ms)

    if dedup_state.is_duplicate(text):
        metrics_collector.increment_counter("tts.dedup_skipped")
        logger.info(f"🚫 [TTS DEDUP] Bỏ qua câu phát âm trùng lặp: '{text}'")
        return

    try:
        t_tts_start = time.perf_counter()
        audio_b64, duration_sec = await tts.synthesize_clone(
            text=text,
            voice_id=voice,
            speed=speed,
        )
        synthesis_ms = (time.perf_counter() - t_tts_start) * 1000.0
        metrics_collector.record_metric("tts", "synthesis_ms", synthesis_ms)
        tts_rtf = (synthesis_ms / 1000.0) / max(0.001, duration_sec)
        metrics_collector.record_metric("tts", "rtf", tts_rtf)
        metrics_collector.increment_counter("tts.synthesized_utterances")

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
                metrics_collector.record_metric("pipeline", "e2e_sub_to_tts_ms", e2e_tts_ms)
                logger.info(f"🔊 [TTS SENT] Gửi {duration_sec:.2f}s audio về client cho utt '{utt_id[:8]}' (synth={synthesis_ms:.1f}ms, RTF={tts_rtf:.2f})")
    except Exception as e:
        logger.error(f"Lỗi TTS synthesis: {e}", exc_info=True)


async def _tts_worker(session: SessionState) -> None:
    """Worker tuần tự OmniVoice TTS đảm bảo an toàn VRAM và task_done."""
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
        except Exception as e:
            logger.error(f"Lỗi TTS worker loop: {e}", exc_info=True)
            await asyncio.sleep(0.05)


__all__ = ["handle_ws"]
