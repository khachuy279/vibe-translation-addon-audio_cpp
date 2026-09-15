"""Hệ thống thu thập và phân tích Metrics hiệu năng thời gian thực.

Hỗ trợ:
- Đo lường độ trễ chi tiết từng chặng (VAD, ASR TTFT, ASR Commit, Translation, TTS, E2E Pipeline).
- Tính toán phân vị độ trễ (p50, p90, p95, p99).
- Giám sát số lượng sample drop, số lần lọc dedup trùng, số lần force commit.
- Xuất báo cáo hiệu năng JSON và định dạng bảng Markdown.
"""

from collections import defaultdict, deque
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np


class MetricsCollector:
    """Bộ thu thập số liệu hiệu năng tập trung cho Backend."""

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Singleton accessor."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = MetricsCollector()
            return cls._instance

    def __init__(self, max_history: int = 10000):
        self._latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, int] = defaultdict(int)
        self._checkpoints: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()

    def record_latency(self, stage: str, latency_ms: float) -> None:
        """Ghi nhận thời gian thực thi (ms) cho một công đoạn."""
        if latency_ms < 0:
            return
        with self._lock:
            self._latencies[stage].append(latency_ms)

    def increment_counter(self, name: str, count: int = 1) -> None:
        """Tăng bộ đếm sự kiện (ví dụ sample drop, dedup skip)."""
        with self._lock:
            self._counters[name] += count

    def get_counter(self, name: str) -> int:
        """Lấy giá trị hiện tại của một bộ đếm."""
        with self._lock:
            return self._counters[name]

    def get_stage_stats(self, stage: str) -> Dict[str, float]:
        """Tính toán thống kê chi tiết (count, min, max, avg, p50, p95, p99) cho một stage."""
        with self._lock:
            values = list(self._latencies.get(stage, []))

        if not values:
            return {
                "count": 0,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        arr = np.array(values, dtype=np.float64)
        return {
            "count": int(len(arr)),
            "avg_ms": round(float(np.mean(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p90_ms": round(float(np.percentile(arr, 90)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }

    def generate_report(self) -> Dict[str, Any]:
        """Tạo báo cáo tổng hợp toàn bộ số liệu hiệu năng."""
        uptime_sec = time.time() - self._start_time
        
        stages_summary = {}
        with self._lock:
            stage_keys = list(self._latencies.keys())
            counters_copy = dict(self._counters)

        for stage in stage_keys:
            stages_summary[stage] = self.get_stage_stats(stage)

        return {
            "uptime_sec": round(uptime_sec, 2),
            "counters": counters_copy,
            "stages": stages_summary,
        }

    def dump_json(self, filepath: str) -> None:
        """Lưu báo cáo hiệu năng ra file JSON."""
        data = self.generate_report()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record_metric(self, stage: str, metric_name: str, value: float) -> None:
        """Helper ghi nhận metric."""
        self.record_latency(f"{stage}.{metric_name}", value)

    def record_checkpoint(self, name: str) -> None:
        """Ghi nhận checkpoint thời gian."""
        with self._lock:
            self._checkpoints[name] = time.time()

    def reset(self) -> None:
        """Xóa toàn bộ số liệu đo lường."""
        with self._lock:
            self._latencies.clear()
            self._counters.clear()
            self._checkpoints.clear()
            self._start_time = time.time()


metrics = MetricsCollector.get_instance()
metrics_collector = metrics

