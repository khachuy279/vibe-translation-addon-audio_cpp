"""Watchdog LUỒNG THẬT (không phải coroutine) cho event loop — F-46.

Vì sao cần: đã hai lần gặp tiến trình backend **treo cứng tại một dòng log** và phải kill
(Ctrl+C không ăn). Kiểu treo này xảy ra khi event loop bị chặn bởi một lời gọi **đồng bộ**
(native) — mọi coroutine/timer ngừng chạy, nên không hàng rào viết bằng asyncio nào báo được.

Cách hoạt động: một luồng daemon đọc `backend.core.heartbeat.stall_sec()` (do coroutine
`heartbeat_loop` cập nhật). Nếu event loop đứng quá `threshold_sec`, luồng này gọi
`faulthandler.dump_traceback()` để in **stack của MỌI thread** (kể cả thread đang giữ GIL),
rồi ghi log ERROR kèm hướng dẫn. Nhờ vậy lần treo sau sẽ tự chỉ ra chỗ kẹt thay vì phải
đoán.

Bật/tắt bằng biến môi trường `STALL_WATCHDOG_SEC` (mặc định 10 giây; 0 = tắt).
"""

import faulthandler
import os
import sys
import threading
import time
from typing import Optional

from backend.core import heartbeat
from backend.utils.logger import get_logger

logger = get_logger("stall_watchdog", level=20)  # logging.INFO

DEFAULT_THRESHOLD_SEC = 10.0
_thread: Optional[threading.Thread] = None
_stop = threading.Event()


def threshold_sec() -> float:
    """Ngưỡng coi là treo (giây). 0 = tắt watchdog."""
    raw = os.environ.get("STALL_WATCHDOG_SEC")
    if raw is None:
        return DEFAULT_THRESHOLD_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_THRESHOLD_SEC


def should_report(stall_sec: float, threshold: float) -> bool:
    """Tách riêng để test được: có cần dump stack hay không."""
    return threshold > 0 and stall_sec >= threshold


def dump_all_threads(reason: str = "") -> None:
    """In stack của MỌI thread (dùng được từ bất kỳ thread nào, kể cả khi loop đang treo).

    Lưu ý: `faulthandler.dump_traceback(file=...)` cần file có `fileno()` thật, nên khi
    `sys.stderr` bị thay bằng bộ đệm trong bộ nhớ (pytest, IDE) thì phải lùi về
    `sys.__stderr__` rồi mới tới mặc định.
    """
    for stream in (sys.stderr, sys.__stderr__):
        try:
            stream.write(
                f"\n===== [STALL WATCHDOG] event loop đứng quá lâu {reason} "
                f"— dump mọi thread =====\n"
            )
            stream.flush()
            break
        except Exception:  # noqa: BLE001
            continue

    dumped = False
    for target in (sys.__stderr__, sys.stderr):
        if target is None:
            continue
        try:
            target.flush()
            faulthandler.dump_traceback(file=target, all_threads=True)
            dumped = True
            break
        except Exception:  # noqa: BLE001
            continue
    if not dumped:  # pragma: no cover - phụ thuộc môi trường
        try:
            faulthandler.dump_traceback(all_threads=True)
        except Exception:  # noqa: BLE001
            pass

    for stream in (sys.stderr, sys.__stderr__):
        try:
            stream.write("===== [STALL WATCHDOG] hết dump =====\n")
            stream.flush()
            break
        except Exception:  # noqa: BLE001
            continue


def start(threshold: Optional[float] = None) -> Optional[threading.Thread]:
    """Chạy watchdog ở luồng nền. Trả None nếu tắt."""
    global _thread
    limit = threshold_sec() if threshold is None else float(threshold)
    if limit <= 0:
        logger.info("Watchdog treo: TẮT (STALL_WATCHDOG_SEC=0).", extra={"module_tag": "MAIN"})
        return None
    if _thread is not None and _thread.is_alive():
        return _thread

    _stop.clear()

    def _watch() -> None:
        while not _stop.is_set():
            time.sleep(0.5)
            stall = heartbeat.stall_sec()
            if should_report(stall, limit):
                logger.error(
                    f"Event loop ĐỨNG {stall:.1f}s (> {limit:.0f}s) — nghi một lời gọi native "
                    f"chặn GIL. Đang dump stack mọi thread ra stderr để tìm chỗ kẹt.",
                    extra={"module_tag": "MAIN"},
                )
                dump_all_threads(f"{stall:.1f}s")
                # Nghỉ một quãng để không spam log
                _stop.wait(max(limit, 5.0))

    _thread = threading.Thread(target=_watch, name="stall_watchdog", daemon=True)
    _thread.start()
    logger.info(
        f"Watchdog treo đã bật (ngưỡng {limit:.0f}s) — treo sẽ tự dump stack mọi thread.",
        extra={"module_tag": "MAIN"},
    )
    return _thread


def stop() -> None:
    _stop.set()
