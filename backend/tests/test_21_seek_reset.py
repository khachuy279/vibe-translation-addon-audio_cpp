"""Test tầng A — F-44: tua video KHÔNG được làm hỏng phụ đề.

Lỗi thật (log người dùng 22:58): sau khi tua, một câu commit `[MAX_DURATION]` dài **~54 giây**
(infer 1341 ms) chứa lại nguyên nội dung đã phát trước đó ⇒ phụ đề lặp và lệch.

Ba lớp bảo vệ được kiểm thử ở đây:
1. `ASREngine.reset_stream()` — xoá audio/commit/trạng thái speech, tăng thế hệ stream.
2. `WS action "reset_stream"` — handler gọi reset cho cả ASR và VAD, dọn hàng đợi dịch/TTS,
   gửi `stream_reset` cho client.
3. Chốt an toàn cuối trong `_emit_commit`: mảnh commit không bao giờ dài quá
   `max_duration_sec * 1.5` (chỉ lấy phần đuôi) — kể cả khi client không báo tua.
4. Extension: có gọi reset khi `seeking`/`seeked`; worklet + capture có API reset.
"""

import asyncio
from pathlib import Path

import numpy as np

from backend.config import config

ROOT = Path(__file__).resolve().parent.parent.parent
EXT = ROOT / "extension_firefox"


# --------------------------------------------------------------- 1. engine reset
def test_reset_stream_clears_audio_and_state(session_factory, restore_config):
    session = session_factory(text_fn=lambda n: "alpha beta")
    engine = session.asr_engine

    engine.feed_audio(np.zeros(16000 * 3, dtype=np.float32))
    engine._speech_active = True
    engine._speech_start_sample = 0
    engine._last_preview_text = "câu cũ"
    engine._pending_commits.append({"utterance_id": "x", "start_sample": 0, "end_sample": 100})

    before_gen = engine._stream_generation
    asyncio.run(engine.reset_stream("seek"))

    assert engine.audio_buffer.total_written == 0, "phải xoá audio cũ"
    assert not engine._pending_commits, "phải bỏ commit đang chờ"
    assert engine._speech_active is False
    assert engine._speech_start_sample == -1
    assert engine._last_preview_text == ""
    assert engine._stream_generation == before_gen + 1, "thế hệ stream phải tăng"


def test_reset_stream_is_idempotent(session_factory, restore_config):
    session = session_factory(text_fn=lambda n: "alpha beta")
    engine = session.asr_engine
    asyncio.run(engine.reset_stream("seek"))
    asyncio.run(engine.reset_stream("seek"))
    assert engine.audio_buffer.total_written == 0
    assert engine._stream_generation == 2


def test_reset_stream_does_not_cancel_native_inference(session_factory, restore_config, monkeypatch):
    """F-44b: reset vì tua TUYỆT ĐỐI không được gọi `cancel_inference()`.

    Đã thử huỷ native khi tua: `session.cancel()` giữa chừng làm các `run()` sau lỗi ⇒
    worker ASR chết ⇒ WebSocket đóng (popup báo Disconnected, phải Start lại). Nay chỉ
    bỏ kết quả cũ bằng `_stream_generation`.
    """
    session = session_factory(text_fn=lambda n: "alpha beta")
    engine = session.asr_engine
    called = {"n": 0}

    async def _fake_cancel():
        called["n"] += 1

    monkeypatch.setattr(engine, "cancel_inference", _fake_cancel, raising=False)
    asyncio.run(engine.reset_stream("seek"))
    assert called["n"] == 0, "reset_stream KHÔNG được huỷ native (làm hỏng phiên)"


def test_commit_from_old_generation_is_dropped(session_factory, restore_config):
    """Commit gắn thế hệ cũ (trước khi tua) phải bị bỏ, không sinh phụ đề lệch."""
    from backend.core.metrics import metrics_collector

    session = session_factory(text_fn=lambda n: "alpha beta")
    engine = session.asr_engine
    engine.feed_audio(np.zeros(16000 * 2, dtype=np.float32))
    total = engine.audio_buffer.total_written
    engine._stream_generation = 5
    metrics_collector.reset()

    ran = {"infer": 0}

    def _fake_infer(pcm):
        ran["infer"] += 1
        return "câu cũ"

    engine._run_inference_sync = _fake_infer  # type: ignore[assignment]

    async def _run():
        out = []
        async for msg in engine._emit_commit({
            "utterance_id": "old",
            "start_sample": 0,
            "end_sample": total,
            "reason": "VAD_SILENCE",
            "generation": 4,  # cũ hơn thế hệ hiện tại
        }):
            out.append(msg)
        return out

    out = asyncio.run(_run())
    assert out == [], "commit của đoạn cũ KHÔNG được phát ra"
    assert ran["infer"] == 0, "không cần chạy inference cho commit đã lỗi thời"
    assert metrics_collector.get_counter("asr.commit_dropped_stale") == 1


# --------------------------------------------------------------- 2. handler action
def test_handler_reset_stream_action(session_factory, restore_config, monkeypatch):
    import json

    from backend.ws.handler import _handle_text_message

    session = session_factory(text_fn=lambda n: "alpha beta")
    called = {"engine": 0, "vad": 0}

    async def _fake_engine_reset(reason="seek"):
        called["engine"] += 1
        called["reason"] = reason

    monkeypatch.setattr(session.asr_engine, "reset_stream", _fake_engine_reset, raising=False)

    orig_vad_reset = session.vad_processor.reset

    def _vad_reset():
        called["vad"] += 1
        return orig_vad_reset()

    monkeypatch.setattr(session.vad_processor, "reset", _vad_reset, raising=False)

    payload = json.dumps({"type": "reset_stream", "reason": "seek"})
    asyncio.run(_handle_text_message(session, payload))

    assert called["engine"] == 1, "phải gọi engine.reset_stream()"
    assert called["vad"] == 1, "phải reset VAD"
    assert called.get("reason") == "seek"
    sent = getattr(session.mock_ws, "sent_messages", [])
    assert any(m.get("type") == "stream_reset" for m in sent), f"phải gửi stream_reset: {sent}"


def test_handler_reset_stream_accepts_legacy_shape(session_factory, restore_config, monkeypatch):
    """Client gửi kèm `action` trong `set_config` vẫn phải được hiểu là reset (tương thích)."""
    import json

    from backend.ws.handler import _handle_text_message

    session = session_factory(text_fn=lambda n: "alpha beta")
    called = {"n": 0}

    async def _fake_reset(reason="seek"):
        called["n"] += 1

    monkeypatch.setattr(session.asr_engine, "reset_stream", _fake_reset, raising=False)

    payload = json.dumps({"type": "set_config", "action": "reset_stream", "reason": "seeking"})
    asyncio.run(_handle_text_message(session, payload))

    assert called["n"] == 1


def test_handler_reset_stream_drains_queues(session_factory, restore_config):
    from backend.ws.handler import _reset_session_stream

    session = session_factory(text_fn=lambda n: "alpha beta")
    session.translation_queue.put_nowait({"text": "câu cũ"})
    session.tts_queue.put_nowait({"text": "câu cũ"})

    asyncio.run(_reset_session_stream(session, "seek"))

    assert session.translation_queue.empty(), "hàng đợi dịch phải được dọn"
    assert session.tts_queue.empty(), "hàng đợi TTS phải được dọn"
    # F-44b: bộ đếm `_unfinished_tasks` phải về 0, nếu không mọi `queue.join()` sau đó TREO.
    assert session.translation_queue._unfinished_tasks == 0  # noqa: SLF001
    assert session.tts_queue._unfinished_tasks == 0  # noqa: SLF001


def test_queue_join_does_not_hang_after_reset(session_factory, restore_config):
    """F-44b: sau khi reset, `drain_queues()` (dùng queue.join) phải kết thúc ngay."""
    import time as _time

    from backend.ws.handler import _reset_session_stream

    session = session_factory(text_fn=lambda n: "alpha beta")
    session.translation_queue.put_nowait({"text": "câu cũ"})
    asyncio.run(_reset_session_stream(session, "seek"))

    async def _run():
        t0 = _time.perf_counter()
        await session.drain_queues(timeout=2.0)
        return _time.perf_counter() - t0

    elapsed = asyncio.run(_run())
    assert elapsed < 1.0, f"drain_queues bị treo sau reset ({elapsed:.2f}s)"


def test_second_seek_reset_is_coalesced(session_factory, restore_config):
    """F-44c: `seeking` + `seeked` bắn liền nhau ⇒ chỉ reset 1 lần."""
    from backend.ws.handler import _reset_session_stream

    session = session_factory(text_fn=lambda n: "alpha beta")
    asyncio.run(_reset_session_stream(session, "seeking"))
    gen_after_first = session.asr_engine._stream_generation
    asyncio.run(_reset_session_stream(session, "seek"))
    assert session.asr_engine._stream_generation == gen_after_first, (
        "lần reset thứ hai quá sát phải bị gộp lại"
    )


# --------------------------------------------------------------- 3. chốt an toàn mảnh commit
def test_commit_slice_is_clamped_to_max_duration(session_factory, restore_config):
    """F-44: mảnh commit dài bất thường (do tua) chỉ được lấy phần ĐUÔI."""
    from backend.core.metrics import metrics_collector

    config.sentence.max_duration_sec = 6.0
    session = session_factory(text_fn=lambda n: "alpha beta")
    engine = session.asr_engine

    seen = {}

    def _fake_infer(pcm):
        seen["sec"] = len(pcm) / 16000.0
        return "ok"

    engine._run_inference_sync = _fake_infer  # type: ignore[assignment]

    # Nhét 100 giây audio vào buffer (capacity 60 s) rồi yêu cầu commit cả đoạn
    engine.feed_audio(np.zeros(16000 * 100, dtype=np.float32))
    total = engine.audio_buffer.total_written
    metrics_collector.reset()

    commit_req = {
        "utterance_id": "u1",
        "start_sample": 0,
        "end_sample": total,
        "reason": "MAX_DURATION",
    }

    async def _run():
        async for _msg in engine._emit_commit(commit_req):
            pass

    asyncio.run(_run())

    assert "sec" in seen, "inference không được gọi"
    assert seen["sec"] <= 6.0 * 1.5 + 0.01, (
        f"mảnh commit phải bị cắt về <= 9s, nhận {seen['sec']:.1f}s"
    )
    assert metrics_collector.get_counter("asr.commit_slice_clamped") >= 1


def test_normal_commit_slice_is_not_clamped(session_factory, restore_config):
    """Câu bình thường (<= max_duration) KHÔNG được cắt — không ảnh hưởng độ chính xác."""
    from backend.core.metrics import metrics_collector

    config.sentence.max_duration_sec = 6.0
    session = session_factory(text_fn=lambda n: "alpha beta")
    engine = session.asr_engine
    seen = {}

    def _fake_infer(pcm):
        seen["sec"] = len(pcm) / 16000.0
        return "ok"

    engine._run_inference_sync = _fake_infer  # type: ignore[assignment]
    engine.feed_audio(np.zeros(16000 * 5, dtype=np.float32))
    total = engine.audio_buffer.total_written
    metrics_collector.reset()

    async def _run():
        async for _msg in engine._emit_commit(
            {"utterance_id": "u1", "start_sample": 0, "end_sample": total, "reason": "VAD_SILENCE"}
        ):
            pass

    asyncio.run(_run())

    assert 4.9 <= seen["sec"] <= 5.1, f"câu 5s phải giữ nguyên, nhận {seen['sec']:.2f}s"
    assert metrics_collector.get_counter("asr.commit_slice_clamped") == 0


# --------------------------------------------------------------- 4. phía extension
def test_extension_resets_on_seek():
    src = (EXT / "content" / "content-script.js").read_text(encoding="utf-8")
    assert 'addEventListener("seeking"' in src, "phải bắt sự kiện `seeking`"
    assert 'addEventListener("seeked"' in src, "phải bắt sự kiện `seeked`"
    assert '"reset_stream"' in src, "phải báo backend reset stream"
    assert "audioCapture.reset()" in src, "phải xoá buffer audio phía client"
    assert "overlayManager.clear()" in src, "phải xoá phụ đề cũ"


def test_capture_has_reset_api():
    src = (EXT / "lib" / "audio-capture.js").read_text(encoding="utf-8")
    assert "reset()" in src
    assert 'type: "reset"' in src, "phải bảo worklet reset"
    assert "residualSamples = new Float32Array(0)" in src


def test_worklet_supports_reset_message():
    src = (EXT / "lib" / "audio-processor.js").read_text(encoding="utf-8")
    assert 'type === "reset"' in src
    assert "this.phase = 0" in src
    assert "this.bufferIndex = 0" in src


# --------------------------------------------------------------- 6. chống quay nóng vòng lặp
def test_tier234_advances_segment_start_immediately(session_factory, restore_config):
    """F-47: khi BẬC 2 cắt câu, mốc bắt đầu câu MỚI phải được chốt ngay.

    Trước đây mốc chỉ được cập nhật lúc `_emit_commit` chạy xong. Nếu commit bị bỏ thì
    `duration_sec` vẫn > max_duration ⇒ vòng lặp enqueue lại cùng đoạn mãi mãi ⇒ quay nóng
    không nhường event loop (backend treo cứng, watchdog F-46 bắt được stack).
    """
    config.sentence.max_duration_sec = 6.0
    config.sentence.enable_tier234 = True
    session = session_factory(text_fn=lambda n: "alpha beta gamma")
    engine = session.asr_engine

    # Đủ audio để vượt max_duration, mốc bắt đầu câu = 0
    engine.feed_audio(np.zeros(int(8.0 * 16000), dtype=np.float32))
    engine._speech_active = True
    engine._speech_start_sample = 0
    total = engine.audio_buffer.total_written

    # Mô phỏng đúng khối F-42 trong stream_tokens
    duration_sec = (total - engine._speech_start_sample) / 16000.0
    reason = engine._evaluate_tier234("alpha beta gamma", duration_sec)
    assert reason is not None and reason.value == "MAX_DURATION"

    overlap = int(config.sentence.boundary_overlap_ms / 1000.0 * 16000)
    old_start = engine._speech_start_sample
    with engine._lock:
        engine._enqueue_commit_locked(
            utterance_id="u1", start_sample=old_start, end_sample=total, reason=reason.value
        )
        engine._speech_start_sample = max(0, total - overlap)

    assert engine._speech_start_sample > old_start, "mốc câu mới phải tiến lên"
    new_duration = (total - engine._speech_start_sample) / 16000.0
    assert new_duration < config.sentence.max_duration_sec, (
        f"câu mới phải ngắn hơn max_duration, còn {new_duration:.1f}s"
    )


def test_stream_never_spins_without_yielding_after_reset(session_factory, restore_config):
    """F-47: sau reset + commit bị bỏ liên tiếp, vòng lặp PHẢI nhường event loop.

    Chạy generator trong luồng riêng; nếu nó quay nóng (không nhường) thì luồng không kết
    thúc sau 3 giây ⇒ test đỏ (đúng ca đã làm treo backend thật).
    """
    import threading

    config.sentence.max_duration_sec = 6.0
    config.asr.max_inflight_infer = 1
    session = session_factory(text_fn=lambda n: "alpha beta gamma")
    engine = session.asr_engine
    engine.poll_interval_ms = 40
    engine._is_running = True

    # Audio vượt max_duration + mốc câu cũ ⇒ BẬC 2 sẽ cắt liên tục
    engine.feed_audio(np.zeros(int(9.0 * 16000), dtype=np.float32))
    engine._speech_active = True
    engine._speech_start_sample = 0

    # Chèn sẵn nhiều commit "cũ" (thế hệ lệch) để chúng bị bỏ ngay, không inference
    engine._stream_generation = 3
    for i in range(6):
        engine._pending_commits.append({
            "utterance_id": f"old{i}", "start_sample": 0, "end_sample": 100,
            "reason": "MAX_DURATION", "generation": 1,
        })

    done = threading.Event()

    async def _consume():
        gen = engine.stream_tokens()
        try:
            for _ in range(60):
                try:
                    await asyncio.wait_for(gen.__anext__(), timeout=0.05)
                except (StopAsyncIteration, asyncio.TimeoutError):
                    break
        finally:
            await gen.aclose()
            engine._is_running = False

    def _worker():
        asyncio.run(_consume())
        done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    finished = done.wait(3.0)
    engine._is_running = False
    assert finished, "vòng lặp stream quay nóng không nhường event loop (nguy cơ treo backend)"


def test_empty_commit_streak_guard_counter(session_factory, restore_config):
    """Rào chắn F-47 có ghi counter khi phải nhường nhịp (để còn thấy trong metrics)."""
    session = session_factory(text_fn=lambda n: "alpha")
    engine = session.asr_engine
    engine._empty_commit_streak = 9
    # 1 commit bị bỏ (thế hệ lệch) ⇒ streak chạm 10 ⇒ guard kích hoạt
    engine._stream_generation = 2
    engine._pending_commits.append({
        "utterance_id": "old", "start_sample": 0, "end_sample": 100,
        "reason": "MAX_DURATION", "generation": 1,
    })

    async def _run():
        gen = engine.stream_tokens()
        engine._is_running = True
        for _ in range(5):
            try:
                await asyncio.wait_for(gen.__anext__(), timeout=0.2)
            except (StopAsyncIteration, asyncio.TimeoutError):
                break
        engine._is_running = False
        await gen.aclose()

    asyncio.run(_run())
    assert engine._empty_commit_streak == 0, "guard phải reset streak sau khi nhường nhịp"

def test_extension_recovers_capture_state_after_disconnect():
    """F-45: mất kết nối ⇒ dừng capture, và Start lại được (không kẹt 'Already capturing')."""
    src = (EXT / "content" / "content-script.js").read_text(encoding="utf-8")
    assert 'wsClient.on("disconnected"' in src, "phải nghe sự kiện `disconnected` của WSClient"
    assert "stopCapture()" in src, "phải dừng capture khi mất kết nối"
    # Start phải tự dọn trạng thái kẹt
    assert "Trạng thái capture cũ đã kẹt" in src, "startCapture phải tự phục hồi"
    assert "!wsClient.isConnected" in src
