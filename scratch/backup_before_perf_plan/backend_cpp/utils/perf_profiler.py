"""High-performance zero/low-overhead profiler and telemetry for backend_cpp.

Tracks:
- Granular latency across VAD -> ASR -> Translation -> TTS stages.
- Lock contention & wait times (e.g. ASR infer lock between preview & commit).
- Queue wait times and backlog sizes.
- Real-Time Factor (RTF = compute_time / audio_duration).
- Generation speed (LLM tokens/second).
- System resources: RAM (RSS), GPU VRAM, OS threads, Asyncio tasks.
- Resource leak detection across sessions.
- Redundant computation (preview repeats, cache hits, dedup events).
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH_CUDA = False

logger = logging.getLogger("backend_cpp.perf")

def __getattr__(name: str):
    if name == "config":
        from backend_cpp.config import config
        return config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class PerfTimer:
    """Accurate monotonic context timer."""
    __slots__ = ("_start", "elapsed_ms")

    def __init__(self):
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


class MetricsCollector:
    """Thread-safe and async-safe performance metrics collector."""

    _instance: Optional["MetricsCollector"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._lock = threading.Lock()
        try:
            from backend_cpp.config import config
            self._enabled: bool = getattr(config.perf, "enabled", True)
            self._alert_threshold_ms: float = getattr(config.perf, "alert_threshold_ms", 300.0)
        except Exception:
            self._enabled = True
            self._alert_threshold_ms = 300.0

        # Metrics storage: category -> metric_name -> deque of floats
        self._samples: Dict[str, Dict[str, deque]] = {}
        # Events and alerts log
        self._alerts: deque = deque(maxlen=200)
        self._last_alert_log_time: Dict[str, float] = {}
        self._alert_throttle_seconds: float = 5.0
        # Counters
        self._counters: Dict[str, int] = {}
        # Resource snapshots at session boundaries
        self._resource_history: deque = deque(maxlen=50)

        # Baseline resource snapshot
        self._baseline_resources = self.get_resource_snapshot()

    @property
    def enabled(self) -> bool:
        try:
            from backend_cpp.config import config
            return bool(config.perf.enabled)
        except Exception:
            return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = bool(val)

    @property
    def alert_threshold_ms(self) -> float:
        try:
            from backend_cpp.config import config
            return float(config.perf.alert_threshold_ms)
        except Exception:
            return self._alert_threshold_ms

    @alert_threshold_ms.setter
    def alert_threshold_ms(self, val: float) -> None:
        self._alert_threshold_ms = float(val)

    def reset(self) -> None:
        """Reset all collected metrics and counters."""
        with self._lock:
            self._samples.clear()
            self._alerts.clear()
            self._last_alert_log_time.clear()
            self._counters.clear()
            self._resource_history.clear()
            self._baseline_resources = self.get_resource_snapshot()
        logger.info("🧹 [PERF] Metrics collector has been reset.")

    def record_metric(
        self,
        category: str,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single numeric metric sample."""
        if not self.enabled:
            return

        with self._lock:
            cat = self._samples.setdefault(category, {})
            sample_deque = cat.setdefault(name, deque(maxlen=self.max_samples))
            sample_deque.append(value)

        # Bottleneck checks
        if "lock_wait" in name and value > 100.0:
            self._add_alert(
                "BOTTLENECK_LOCK_CONTENTION",
                f"High lock contention on {category}.{name}: {value:.1f}ms waiting for lock!",
                metadata,
            )
        elif "queue_wait" in name and value > self.alert_threshold_ms:
            self._add_alert(
                "BOTTLENECK_QUEUE_BACKPRESSURE",
                f"High queue delay on {category}.{name}: {value:.1f}ms wait time!",
                metadata,
            )
        elif "rtf" in name and value > 1.0:
            self._add_alert(
                "BOTTLENECK_RTF_EXCEEDED",
                f"Inference slower than real-time on {category}.{name}: RTF={value:.2f} (> 1.0)!",
                metadata,
            )

    def increment_counter(self, name: str, count: int = 1) -> None:
        """Increment a discrete counter."""
        if not self.enabled:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + count

    def _add_alert(self, alert_type: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        now = time.time()
        entry = {
            "timestamp": now,
            "type": alert_type,
            "message": message,
            "metadata": metadata or {},
        }
        should_log = False
        with self._lock:
            self._alerts.append(entry)
            last_time = self._last_alert_log_time.get(alert_type, 0.0)
            if now - last_time >= self._alert_throttle_seconds:
                self._last_alert_log_time[alert_type] = now
                should_log = True

        if should_log:
            logger.warning(f"⚠️ [PERF ALERT] [{alert_type}] {message}")

    @contextmanager
    def measure(
        self,
        category: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        log_slow_ms: Optional[float] = None,
    ):
        """Context manager to measure latency of a code block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.record_metric(category, name, elapsed_ms, metadata)
            threshold = log_slow_ms or self.alert_threshold_ms
            if elapsed_ms > threshold:
                logger.info(f"⏱️ [PERF_SLOW] {category}.{name} took {elapsed_ms:.1f}ms (threshold: {threshold:.0f}ms)")

    def get_resource_snapshot(self) -> Dict[str, Any]:
        """Capture current process resource usage (RAM, VRAM, Threads, Tasks)."""
        rss_mb = 0.0
        num_threads = 0
        cpu_percent = 0.0

        if _HAS_PSUTIL:
            try:
                proc = psutil.Process(os.getpid())
                mem_info = proc.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                num_threads = proc.num_threads()
                cpu_percent = proc.cpu_percent(interval=None)
            except Exception:
                pass

        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        if _HAS_TORCH_CUDA:
            try:
                gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            except Exception:
                pass

        asyncio_tasks = 0
        try:
            asyncio_tasks = len(asyncio.all_tasks())
        except RuntimeError:
            pass

        return {
            "timestamp": time.time(),
            "ram_rss_mb": round(rss_mb, 2),
            "cpu_percent": round(cpu_percent, 1),
            "gpu_allocated_mb": round(gpu_allocated_mb, 2),
            "gpu_reserved_mb": round(gpu_reserved_mb, 2),
            "os_threads": num_threads,
            "asyncio_tasks": asyncio_tasks,
        }

    def record_resource_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Record resource checkpoint and calculate delta against baseline."""
        snap = self.get_resource_snapshot()
        if not self.enabled:
            return snap
        snap["checkpoint"] = checkpoint_name

        delta_ram = snap["ram_rss_mb"] - self._baseline_resources["ram_rss_mb"]
        delta_gpu = snap["gpu_allocated_mb"] - self._baseline_resources["gpu_allocated_mb"]
        snap["delta_ram_mb"] = round(delta_ram, 2)
        snap["delta_gpu_mb"] = round(delta_gpu, 2)

        with self._lock:
            self._resource_history.append(snap)

        if delta_ram > 500.0:
            self._add_alert(
                "RESOURCE_LEAK_WARNING",
                f"RAM usage grew by +{delta_ram:.1f}MB above baseline! Current: {snap['ram_rss_mb']}MB",
            )
        return snap

    def compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical distribution (Min, Max, Mean, P50, P90, P99)."""
        if not values:
            return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def _percentile(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            d0 = sorted_vals[f] * (c - k)
            d1 = sorted_vals[c] * (k - f)
            return d0 + d1

        return {
            "count": n,
            "min": round(sorted_vals[0], 2),
            "max": round(sorted_vals[-1], 2),
            "avg": round(sum(sorted_vals) / n, 2),
            "p50": round(_percentile(0.50), 2),
            "p90": round(_percentile(0.90), 2),
            "p99": round(_percentile(0.99), 2),
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate structured performance analysis dictionary."""
        with self._lock:
            metrics_summary: Dict[str, Dict[str, Any]] = {}
            for cat, metrics in self._samples.items():
                metrics_summary[cat] = {}
                for name, values_deque in metrics.items():
                    metrics_summary[cat][name] = self.compute_statistics(list(values_deque))

            alerts_list = list(self._alerts)
            counters_dict = dict(self._counters)
            resources = self.get_resource_snapshot()
            res_history = list(self._resource_history)

        return {
            "status": "ok",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_resources": resources,
            "baseline_resources": self._baseline_resources,
            "resource_history": res_history,
            "counters": counters_dict,
            "metrics": metrics_summary,
            "alerts": alerts_list,
        }

    def generate_markdown_summary(self) -> str:
        """Render a clean, human-readable terminal/markdown performance report."""
        report = self.generate_report()
        lines = [
            "================================================================================",
            "                     BACKEND_CPP PERFORMANCE REPORT",
            "================================================================================",
            f"Generated: {report['generated_at']}",
            "",
            "--- SYSTEM RESOURCES ---",
            f"  * RAM (RSS):       {report['current_resources']['ram_rss_mb']:.1f} MB (Baseline: {report['baseline_resources']['ram_rss_mb']:.1f} MB)",
            f"  * GPU VRAM:        {report['current_resources']['gpu_allocated_mb']:.1f} MB (Reserved: {report['current_resources']['gpu_reserved_mb']:.1f} MB)",
            f"  * OS Threads:      {report['current_resources']['os_threads']}",
            f"  * Asyncio Tasks:   {report['current_resources']['asyncio_tasks']}",
            "",
            "--- STAGE LATENCY & THROUGHPUT (All times in ms unless specified) ---",
        ]

        metrics = report.get("metrics", {})
        if not metrics:
            lines.append("  (No performance metrics recorded yet)")
        else:
            header = f"{'Category / Metric':<36} | {'Count':>6} | {'Avg':>8} | {'P50':>8} | {'P90':>8} | {'P99':>8} | {'Max':>8}"
            lines.append(header)
            lines.append("-" * len(header))

            for cat, submetrics in sorted(metrics.items()):
                for name, stats in sorted(submetrics.items()):
                    cat_name = f"{cat}.{name}"
                    cnt = stats["count"]
                    avg = f"{stats['avg']:.1f}"
                    p50 = f"{stats['p50']:.1f}"
                    p90 = f"{stats['p90']:.1f}"
                    p99 = f"{stats['p99']:.1f}"
                    max_v = f"{stats['max']:.1f}"
                    lines.append(f"{cat_name:<36} | {cnt:>6} | {avg:>8} | {p50:>8} | {p90:>8} | {p99:>8} | {max_v:>8}")

        lines.append("")
        lines.append("--- COUNTERS & EFFICIENCY ---")
        counters = report.get("counters", {})
        if not counters:
            lines.append("  (No counters recorded)")
        else:
            for k, v in sorted(counters.items()):
                lines.append(f"  * {k:<34}: {v}")

        alerts = report.get("alerts", [])
        lines.append("")
        lines.append(f"--- BOTTLENECK & RESOURCE ALERTS ({len(alerts)} alerts) ---")
        if not alerts:
            lines.append("  ✅ No critical bottlenecks detected! Pipeline is running smoothly.")
        else:
            for a in alerts[-10:]:
                t_str = time.strftime("%H:%M:%S", time.localtime(a["timestamp"]))
                lines.append(f"  ⚠️ [{t_str}] [{a['type']}] {a['message']}")

        lines.append("================================================================================")
        return "\n".join(lines)

    def dump_report_file(self, filepath: str) -> None:
        """Dump report to JSON and Markdown files."""
        if not self.enabled or not config.perf.dump_report_on_disconnect:
            return
        report = self.generate_report()
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"📊 [PERF] Exported JSON report to {filepath}")

            md_path = filepath.rsplit(".", 1)[0] + "_summary.txt"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self.generate_markdown_summary())
            logger.info(f"📊 [PERF] Exported text summary to {md_path}")
        except Exception as e:
            logger.error(f"Failed to dump performance report to {filepath}: {e}")


# Global helper singleton instance
perf = MetricsCollector.get_instance()
