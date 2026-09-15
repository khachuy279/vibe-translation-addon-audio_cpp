"""Pytest conftest cho backend/tests."""

import os
import sys
from pathlib import Path
import pytest

# Thêm thư mục gốc dự án vào sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.utils.cuda import setup_cuda_dll_paths
setup_cuda_dll_paths()

WAV_TEST_DIR = _PROJECT_ROOT / "wav_test"
REPORT_DIR = _PROJECT_ROOT / "report"


@pytest.fixture
def wav_test_dir() -> Path:
    return WAV_TEST_DIR


@pytest.fixture
def report_dir() -> Path:
    return REPORT_DIR
