"""Công cụ đo mức dùng GPU của bộ test — trả lời "test có dùng GPU không?".

BỐI CẢNH (vì sao cần công cụ này):
Người dùng quan sát Task Manager chỉ thấy RAM tăng, không thấy VRAM tăng, và nghi ngờ
GPU không được dùng. Có HAI cái bẫy đo lường khiến kết luận sai:

  1. `nvidia-smi --query-compute-apps` (và danh sách tiến trình GPU) **chỉ liệt kê tiến
     trình CUDA**. Tiến trình chỉ dùng **Vulkan** (transcribe.cpp của dự án này) KHÔNG
     xuất hiện ở đó — dù nó đang giữ VRAM. Đã kiểm chứng: khi tier-B chạy (68 % GPU util,
     +1,4 GB VRAM) thì `--query-compute-apps` **trả về rỗng**.
  2. `nvidia-smi --query-gpu=memory.used` (và "Dedicated GPU memory" của Task Manager)
     mới là số VRAM dùng chung của cả card, tính CẢ allocation Vulkan.
  3. Ảnh Task Manager chụp lúc **không có phiên ASR nào chạy** sẽ hiển thị đúng baseline
     desktop (~1,7–1,9 GB), không phải bằng chứng "test không dùng GPU".

Công cụ này chạy một lệnh test và đo **VRAM theo từng PID** (cùng nguồn dữ liệu Task
Manager dùng: counter GPU Process Memory — Dedicated Usage), nên quy được trách nhiệm
cho đúng tiến trình test.

Chạy:
    python backend/tests/gpu_usage_probe.py                  # mặc định: đo pytest tầng A
    python backend/tests/gpu_usage_probe.py --tier-b          # đo thêm harness có model thật
    python backend/tests/gpu_usage_probe.py --full-suite      # đo cả bộ (tầng A)

Không có hàm `test_*` nên pytest KHÔNG thu thập file này; bộ test mặc định vẫn nhanh.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent


def gpu_total_mb():
    """VRAM dùng chung của card (tính CẢ allocation Vulkan, không chỉ CUDA)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            mem, util = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")[:2]]
            return float(mem), float(util)
    except Exception:
        pass
    return None, None


def cuda_apps():
    """{pid: MB} cho tiến trình CUDA. Tiến trình Vulkan KHÔNG xuất hiện ở đây."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        res = {}
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                res[int(parts[0])] = float(parts[1])
        return res
    except Exception:
        return {}


def per_pid_vram_psh(pid: int) -> float:
    """VRAM dedicated của MỘT pid, cùng nguồn dữ liệu Task Manager dùng."""
    script = (
        "$c = Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue; "
        "$m = 0; foreach ($s in $c.CounterSamples) { "
        f"$i = [int]((($s.InstanceName) -split '_')[1]); "
        f"if ($i -eq {pid} -and $s.CookedValue -gt $m) {{ $m = $s.CookedValue }} }}; "
        "[math]::Round($m/1MB)"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                             capture_output=True, text=True, timeout=25)
        txt = (out.stdout or "").strip().splitlines()
        return float(txt[-1]) if txt else 0.0
    except Exception:
        return 0.0


def measure(label: str, cmd, log_path: Path, poll_psh: bool = True):
    print("=" * 88)
    print(f"[{label}]")
    print("=" * 88)
    base_total, _ = gpu_total_mb()
    print(f"   VRAM card truoc chay : {base_total} MB")
    print(f"   tien trinh CUDA      : {cuda_apps() or '(khong co)'}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    peak_total = {"v": base_total or 0.0}
    peak_util = {"v": 0.0}
    peak_own = {"v": 0.0}
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(_ROOT))
        while proc.poll() is None:
            total, util = gpu_total_mb()
            if total is not None:
                peak_total["v"] = max(peak_total["v"], total)
                peak_util["v"] = max(peak_util["v"], util or 0.0)
            if poll_psh:
                own = per_pid_vram_psh(proc.pid)
                peak_own["v"] = max(peak_own["v"], own)
            time.sleep(0.4)
    elapsed = time.perf_counter() - t0
    after_total, _ = gpu_total_mb()

    print(f"   exit                 : {proc.returncode}")
    print(f"   thoi gian            : {elapsed:.1f}s")
    print(f"   VRAM card dinh       : {peak_total['v']:.0f} MB  => tang {peak_total['v'] - (base_total or 0):+.0f} MB")
    if poll_psh:
        print(f"   VRAM cua PID test    : {peak_own['v']:.0f} MB  <-- so quyet dinh")
    print(f"   VRAM sau khi xong    : {after_total} MB  (da tra lai)")
    print(f"   GPU util dinh        : {peak_util['v']:.0f} %")
    print(f"   log                  : {log_path}")
    print()
    return {
        "label": label,
        "exit": proc.returncode,
        "seconds": round(elapsed, 1),
        "vram_base_mb": base_total,
        "vram_peak_total_mb": peak_total["v"],
        "vram_peak_of_test_pid_mb": peak_own["v"] if poll_psh else None,
        "vram_after_mb": after_total,
        "gpu_util_peak_pct": peak_util["v"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-b", action="store_true", help="đo thêm harness có model thật")
    ap.add_argument("--full-suite", action="store_true", help="đo cả bộ test (tầng A)")
    ap.add_argument("--model", default="qwen3-asr-0.6b")
    args = ap.parse_args()

    results = []
    results.append(measure(
        "pytest (tầng A) — MẶC ĐỊNH",
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        _ROOT / "scratch" / "gpu_probe_tierA.log",
    ))
    if args.tier_b:
        results.append(measure(
            f"tier-B ({args.model}, transcribe.cpp trên Vulkan)",
            [sys.executable, "backend/tests/test_08_streaming_latency.py",
             "--model", args.model, "--seconds", "6", "--speed", "1"],
            _ROOT / "scratch" / "gpu_probe_tierB.log",
        ))

    print("=" * 88)
    print(f"{'Case':46} {'VRAM cua PID test':>18} {'util':>6} {'giay':>6}")
    for r in results:
        own = r["vram_peak_of_test_pid_mb"]
        print(f"{r['label']:46} {('%.0f MB' % own) if own is not None else 'n/a':>18} "
              f"{r['gpu_util_peak_pct']:>5.0f}% {r['seconds']:>5.1f}s")

    out = _ROOT / "report" / "audit" / "13_gpu_usage_probe_by_test_tool.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDa ghi: {out}")
    print()
    print("GHI CHU QUAN TRONG VE CACH DOC:")
    print("  * Cot 'VRAM cua PID test' la so quyet dinh. Neu > 200 MB => test CO dung GPU.")
    print("  * KHONG dung `nvidia-smi --query-compute-apps` de ket luan: no chi liet ke tien")
    print("    trinh CUDA, con transcribe.cpp chay bang VULKAN nen se KHONG xuat hien")
    print("    (da kiem chung: tier-B 68% GPU util nhung danh sach CUDA rong).")
    print("  * Test tang A chi nen ton ~0 MB VRAM. Neu thay vai GB => co test dang nap model")
    print("    that; conftest co guard autouse `_forbid_heavy_model_loads` de bat loi nay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
