"""System and hardware resource monitor: CPU, RAM, GPU/VRAM, and thread telemetry."""

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import psutil
import torch


@dataclass
class ResourceSnapshot:
    """Telemetry snapshot of system resources during benchmark execution."""
    cpu_avg_percent: float
    cpu_peak_percent: float
    process_cpu_peak_percent: float
    ram_start_mb: float
    ram_peak_mb: float
    ram_end_mb: float
    vram_start_mb: Optional[float]
    vram_peak_mb: Optional[float]
    vram_end_mb: Optional[float]
    gpu_utilization_percent: Optional[float]
    thread_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "cpu_avg_percent": round(self.cpu_avg_percent, 1),
            "cpu_peak_percent": round(self.cpu_peak_percent, 1),
            "process_cpu_peak_percent": round(self.process_cpu_peak_percent, 1),
            "ram_start_mb": round(self.ram_start_mb, 1),
            "ram_peak_mb": round(self.ram_peak_mb, 1),
            "ram_end_mb": round(self.ram_end_mb, 1),
            "vram_start_mb": round(self.vram_start_mb, 1) if self.vram_start_mb is not None else "NOT AVAILABLE",
            "vram_peak_mb": round(self.vram_peak_mb, 1) if self.vram_peak_mb is not None else "NOT AVAILABLE",
            "vram_end_mb": round(self.vram_end_mb, 1) if self.vram_end_mb is not None else "NOT AVAILABLE",
            "gpu_utilization_percent": round(self.gpu_utilization_percent, 1) if self.gpu_utilization_percent is not None else "NOT AVAILABLE",
            "thread_count": self.thread_count,
        }


class ResourceMonitor:
    """Asynchronous background resource sampler."""

    def __init__(self, interval_sec: float = 0.05):
        self.interval_sec = interval_sec
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.cpu_samples: List[float] = []
        self.proc_cpu_samples: List[float] = []
        self.ram_samples: List[float] = []
        self.vram_samples: List[float] = []
        self.gpu_util_samples: List[float] = []

        self.has_cuda = torch.cuda.is_available()
        self.ram_start = 0.0
        self.vram_start: Optional[float] = None

    def _sample_vram_mb(self) -> Optional[float]:
        """Query CUDA allocated VRAM or device memory in MB."""
        if not self.has_cuda:
            return None
        try:
            # torch.cuda.memory_allocated returns active torch tensors
            # torch.cuda.memory_reserved returns total cached / reserved
            alloc = torch.cuda.memory_allocated(0) / (1024.0 * 1024.0)
            res = torch.cuda.memory_reserved(0) / (1024.0 * 1024.0)
            return max(alloc, res)
        except Exception:
            return None

    def _sample_gpu_util(self) -> Optional[float]:
        """Sample GPU utilization percentage via NVML if accessible."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(rates.gpu)
        except Exception:
            return None

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                # CPU total & process
                c = psutil.cpu_percent(interval=None)
                self.cpu_samples.append(c)

                pc = self.process.cpu_percent(interval=None)
                self.proc_cpu_samples.append(pc)

                # RAM RSS in MB
                r = self.process.memory_info().rss / (1024.0 * 1024.0)
                self.ram_samples.append(r)

                # VRAM
                v = self._sample_vram_mb()
                if v is not None:
                    self.vram_samples.append(v)

                # GPU Util
                gu = self._sample_gpu_util()
                if gu is not None:
                    self.gpu_util_samples.append(gu)

            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self) -> None:
        """Start background polling thread."""
        self.ram_start = self.process.memory_info().rss / (1024.0 * 1024.0)
        self.vram_start = self._sample_vram_mb()

        # Prime cpu_percent counters
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="res_monitor")
        self._thread.start()

    def stop(self) -> ResourceSnapshot:
        """Stop monitor and return aggregated summary."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        ram_end = self.process.memory_info().rss / (1024.0 * 1024.0)
        vram_end = self._sample_vram_mb()

        cpu_avg = float(sum(self.cpu_samples) / len(self.cpu_samples)) if self.cpu_samples else 0.0
        cpu_peak = float(max(self.cpu_samples)) if self.cpu_samples else 0.0
        proc_cpu_peak = float(max(self.proc_cpu_samples)) if self.proc_cpu_samples else 0.0

        ram_peak = float(max(self.ram_samples)) if self.ram_samples else self.ram_start
        vram_peak = float(max(self.vram_samples)) if self.vram_samples else self.vram_start
        gpu_util_avg = float(sum(self.gpu_util_samples) / len(self.gpu_util_samples)) if self.gpu_util_samples else None

        return ResourceSnapshot(
            cpu_avg_percent=cpu_avg,
            cpu_peak_percent=cpu_peak,
            process_cpu_peak_percent=proc_cpu_peak,
            ram_start_mb=self.ram_start,
            ram_peak_mb=ram_peak,
            ram_end_mb=ram_end,
            vram_start_mb=self.vram_start,
            vram_peak_mb=vram_peak,
            vram_end_mb=vram_end,
            gpu_utilization_percent=gpu_util_avg,
            thread_count=self.process.num_threads(),
        )
