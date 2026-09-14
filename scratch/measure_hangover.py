"""Measure the end timing and delivered tail across hangover values (diagnostic)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend_cpp.tests.test_vad_silence_latency import (  # noqa: E402
    _feed_then_silence,
    _tail_delivered_to_decoder,
)

for h in (0, 100, 250, 400, 600):
    end = _feed_then_silence("fsmn-vad", silence_ms=150, hangover_ms=h)
    tail = _tail_delivered_to_decoder("fsmn-vad", silence_ms=150, hangover_ms=h)
    print(f"hangover={h:>4}ms   end_at_silence={end:7.1f}ms   tail_delivered={tail:7.1f}ms")
