"""Unit tests for backend_cpp.utils (cuda_utils, ssl_utils, perf_profiler)."""

import os
import site
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths
import backend_cpp.utils.cuda_utils as cuda_module
from backend_cpp.utils.ssl_utils import ensure_ssl_certificates
from backend_cpp.utils.perf_profiler import MetricsCollector


def test_cuda_utils_none_user_site():
    """Verify setup_cuda_dll_paths handles site.USER_SITE being None without TypeError."""
    cuda_module._cuda_paths_initialized = False

    with patch.object(site, "USER_SITE", None), \
         patch.object(site, "getsitepackages", return_value=["C:\\dummy_site"]):
        # Should not raise TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType
        setup_cuda_dll_paths()
        assert cuda_module._cuda_paths_initialized is True


def test_ssl_utils_atomic_creation(tmp_path):
    """Verify ensure_ssl_certificates creates cert and key files atomically."""
    with patch("backend_cpp.utils.ssl_utils._BACKEND_CPP_DIR", tmp_path):
        cert, key = ensure_ssl_certificates()
        assert Path(cert).exists()
        assert Path(key).exists()
        assert Path(cert).name == "cert.pem"
        assert Path(key).name == "key.pem"

        # Subsequent call returns existing files without re-generating
        mtime_cert = Path(cert).stat().st_mtime
        cert2, key2 = ensure_ssl_certificates()
        assert cert2 == cert
        assert key2 == key
        assert Path(cert2).stat().st_mtime == mtime_cert


def test_perf_profiler_alert_throttling():
    """Verify that alert logs are throttled by alert type while all alerts are saved in history."""
    collector = MetricsCollector(max_samples=50)
    collector.reset()

    with patch("backend_cpp.utils.perf_profiler.config.perf.enabled", True), \
         patch("backend_cpp.utils.perf_profiler.logger.warning") as mock_warn:
        # Trigger 5 rapid alerts of same type
        for _ in range(5):
            collector.record_metric("test_cat", "rtf_test", 1.5)

        # All 5 should be recorded in _alerts
        assert len(collector._alerts) == 5
        # But logger.warning should be called only ONCE due to 5.0s throttle
        assert mock_warn.call_count == 1

        # A different alert type should log immediately
        collector.record_metric("test_cat", "lock_wait_test", 200.0)
        assert len(collector._alerts) == 6
        assert mock_warn.call_count == 2
