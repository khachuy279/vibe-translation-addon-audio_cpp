"""Guards for the resource-leak detector.

The detector compares RSS against a snapshot taken *before* any model is loaded, while GGUF
weights are memory-mapped only on first use. So the first checkpoint after real work always
jumps by hundreds of MB and the warning fires on every fresh process -- measured on the real
audio scenario as +647.6MB, then +6.9MB, +5.5MB on repeats.

A warning that always fires is worse than no warning: it teaches the reader to ignore it, and
a genuine leak would then pass unnoticed. These tests pin the corrected behaviour:

* the first oversized jump is treated as warm-up and re-arms the baseline;
* growth measured *from* that warm baseline is still reported as a leak;
* ordinary small growth never alerts.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from backend_cpp.utils.perf_profiler import MetricsCollector

WARMUP_JUMP_MB = 650.0


@pytest.fixture
def collector(monkeypatch) -> MetricsCollector:
    """A collector with perf enabled (the singleton reads config, so patch the config)."""
    from backend_cpp.config import config

    monkeypatch.setattr(config.perf, "enabled", True, raising=False)
    return MetricsCollector()


def _stub_rss(collector: MetricsCollector, baseline_mb: float) -> Dict[str, float]:
    """Replace the live RSS snapshot with a controllable one."""
    state = {"rss_mb": baseline_mb}

    def fake_snapshot() -> Dict[str, float]:
        return {
            "timestamp": 0.0,
            "ram_rss_mb": float(state["rss_mb"]),
            "cpu_percent": 0.0,
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "os_threads": 1,
            "asyncio_tasks": 1,
        }

    collector.get_resource_snapshot = fake_snapshot  # type: ignore[method-assign]
    collector._baseline_resources = fake_snapshot()
    collector._leak_baseline_warm = False
    return state


def _alert_types(collector: MetricsCollector) -> List[str]:
    return [str(a.get("type", "")) for a in collector._alerts]


def test_first_model_load_rearms_baseline_instead_of_alerting(collector):
    rss = _stub_rss(collector, baseline_mb=1000.0)

    rss["rss_mb"] = 1000.0 + WARMUP_JUMP_MB
    snap = collector.record_resource_checkpoint("session_start")

    assert "RESOURCE_LEAK_WARNING" not in _alert_types(collector), "warm-up is not a leak"
    assert snap["leak_baseline_rearmed"] is True
    assert collector._baseline_resources["ram_rss_mb"] == 1650.0, "baseline adopted the warm state"


def test_growth_after_warmup_is_still_reported_as_a_leak(collector):
    """The fix must not defang the detector -- it only skips the warm-up step."""
    rss = _stub_rss(collector, baseline_mb=1000.0)

    rss["rss_mb"] = 1000.0 + WARMUP_JUMP_MB
    collector.record_resource_checkpoint("session_start")  # warm-up, re-arms
    assert _alert_types(collector) == []

    rss["rss_mb"] = 1000.0 + WARMUP_JUMP_MB + 600.0
    collector.record_resource_checkpoint("session_end")

    assert "RESOURCE_LEAK_WARNING" in _alert_types(collector)
    assert "grew by" in collector._alerts[-1]["message"]


def test_small_growth_never_alerts(collector):
    rss = _stub_rss(collector, baseline_mb=1000.0)

    rss["rss_mb"] = 1200.0
    snap = collector.record_resource_checkpoint("session_start")

    assert _alert_types(collector) == []
    assert "leak_baseline_rearmed" not in snap


def test_repeat_of_the_same_warmup_jump_alerts_once_at_most(collector):
    """Idempotent: the same reading must not keep re-arming or keep alerting."""
    rss = _stub_rss(collector, baseline_mb=1000.0)
    rss["rss_mb"] = 1000.0 + WARMUP_JUMP_MB

    first = collector.record_resource_checkpoint("session_start")
    second = collector.record_resource_checkpoint("session_end")

    assert first["leak_baseline_rearmed"] is True
    assert "leak_baseline_rearmed" not in second
    assert _alert_types(collector) == [], "delta against the re-armed baseline is ~0"


def test_reset_rearms_the_warmup_guard(collector):
    rss = _stub_rss(collector, baseline_mb=1000.0)
    rss["rss_mb"] = 1000.0 + WARMUP_JUMP_MB
    collector.record_resource_checkpoint("session_start")
    assert collector._leak_baseline_warm is True

    collector.reset()

    assert collector._leak_baseline_warm is False, "a fresh baseline needs a fresh warm-up"


def test_checkpoints_are_kept_in_resource_history(collector):
    rss = _stub_rss(collector, baseline_mb=1000.0)
    rss["rss_mb"] = 1100.0
    collector.record_resource_checkpoint("session_start")

    assert [s["checkpoint"] for s in collector._resource_history] == ["session_start"]
    assert collector._resource_history[0]["delta_ram_mb"] == pytest.approx(100.0)
