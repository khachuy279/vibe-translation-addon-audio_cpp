"""Spike for W2.2: which ASR models actually support native streaming?

W2.2 (incremental streaming) only pays off if the ACTIVE model exposes
`session.stream()` with persisted state across `feed()` calls. If the active model is
non-streaming, every preview poll re-runs `session.run()` over the whole utterance, and
there is no prefix-continuation API to make that incremental.

Run from the project root:  python scratch/spike_streaming_caps.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend_cpp.asr.model_registry import ModelRegistry  # noqa: E402

registry = ModelRegistry.get_instance()
active = registry.get_active_model_key()
print(f"active model: {active}\n")

header = f"{'model_key':42s} {'registry_streaming':>18s}  {'arch':>12s}  family"
print(header)
print("-" * len(header))

for key in sorted(registry.models.keys()):
    info = registry.models.get(key) or {}
    print(
        f"{key:42s} {str(registry.is_streaming_model(key)):>18s}  "
        f"{str(info.get('architecture_type')):>12s}  {info.get('family')}"
    )
