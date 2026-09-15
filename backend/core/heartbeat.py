"""Nhịp tim của event loop (F-40) — phát hiện sớm event loop bị CHẶN ĐỨNG.

Vì sao cần: trong điều tra vòng 4-5 đã bắt gặp tiến trình treo **im lặng** — log dừng ở một
dòng VAD, không hàng rào Python nào (kể cả đồng hồ 1 s của `_infer_with_watchdog`) phát
hiện được. Điều đó chỉ xảy ra khi event loop bị chặn bởi một lời gọi đồng bộ, khiến MỌI
timer/coroutine ngừng chạy.

Cơ chế: một task nền (coroutine) cập nhật `tick()` mỗi `INTERVAL_SEC`. Bất kỳ ai (endpoint
`/health`, test, giám sát) đọc `stall_sec()` sẽ biết event loop đã đứng bao lâu. Nếu loop
bị chặn thì giá trị này tăng đều — đây là dấu hiệu DUY NHẤT nhìn thấy được từ bên ngoài.
"""

import time

INTERVAL_SEC: float = 0.25

_last_tick: float = time.monotonic()


def tick() -> None:
    """Gọi từ chính event loop (task nền) mỗi `INTERVAL_SEC`."""
    global _last_tick
    _last_tick = time.monotonic()


def stall_sec() -> float:
    """Số giây kể từ nhịp tim gần nhất. Lớn bất thường = event loop đang bị chặn."""
    return max(0.0, time.monotonic() - _last_tick)


def reset() -> None:
    tick()


async def heartbeat_loop() -> None:
    """Task nền: tick đều đặn cho tới khi bị huỷ."""
    import asyncio

    while True:
        tick()
        await asyncio.sleep(INTERVAL_SEC)
