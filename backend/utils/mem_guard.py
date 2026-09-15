"""Rào chắn RAM CỨNG cho các harness đo đạc (bảo vệ máy, không để tiến trình nuốt RAM).

Bối cảnh thật: lỗi phình bộ nhớ nằm BÊN TRONG native `transcribe.cpp`/Vulkan (xem
`report/audit/05_measurements_and_status.md` §12.6). Hàng rào ở tầng engine chỉ canh được
lúc inference; lúc **nạp model / warm-up** thì không. Hệ quả: một lần chạy harness tầng B
đã để tiến trình phình lên **47 GB** và làm máy hết RAM.

Rào chắn này là tuyến cuối: một luồng nền đo RSS mỗi `interval` giây và **tự kết thúc tiến
trình** khi vượt trần, thay vì để hệ điều hành phải vật lộn với việc thiếu bộ nhớ.

Dùng:
    from backend.utils.mem_guard import start_guard
    start_guard()                     # trần mặc định 4000 MB (đổi bằng MEM_GUARD_MB)
    start_guard(limit_mb=6000)
    start_guard(limit_mb=0)           # tắt (không khuyến khích)
"""

import os
import sys
import threading
import time

DEFAULT_LIMIT_MB: float = 4000.0
DEFAULT_INTERVAL_SEC: float = 2.0
EXIT_CODE: int = 70  # EX_SOFTWARE — để phân biệt với lỗi thật của chương trình


def rss_mb() -> float:
    """RSS của tiến trình (MB). Trả 0 nếu không đọc được."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass
    try:  # pragma: no cover - phụ thuộc nền tảng
        import ctypes
        import ctypes.wintypes as wt

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        fn.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
        fn.restype = wt.BOOL
        if fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def exceeds(rss_now_mb: float, limit_mb: float) -> bool:
    """True nếu RSS vượt trần. Tách riêng để test được mà không phải kill tiến trình."""
    return limit_mb > 0 and rss_now_mb > limit_mb


def start_guard(
    limit_mb: float = None,
    interval: float = DEFAULT_INTERVAL_SEC,
    exit_code: int = EXIT_CODE,
) -> "threading.Thread | None":
    """Bật rào chắn. Trả về thread canh hoặc None nếu bị tắt (trần <= 0)."""
    if limit_mb is None:
        raw = os.environ.get("MEM_GUARD_MB")
        limit_mb = float(raw) if raw not in (None, "") else DEFAULT_LIMIT_MB
    limit_mb = float(limit_mb)
    if limit_mb <= 0:
        return None

    def _watch() -> None:
        while True:
            now = rss_mb()
            if now > 0 and exceeds(now, limit_mb):
                msg = (
                    f"[MEM_GUARD] RSS {now:.0f} MB > trần {limit_mb:.0f} MB — TỰ DỪNG tiến trình "
                    f"để bảo vệ máy (đây là lỗi phình bộ nhớ native, xem báo cáo §12.6)."
                )
                print(msg, flush=True)
                try:
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
                except Exception:  # noqa: BLE001
                    pass
                os._exit(exit_code)
            time.sleep(interval)

    t = threading.Thread(target=_watch, name="mem_guard", daemon=True)
    t.start()
    return t
