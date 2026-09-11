"""Spike for W2.2: does the ACTIVE ASR model really support native streaming?

The registry answer is metadata-only (`architecture_type` / `family`). The authoritative
answer is `model.capabilities.supports_streaming`, which is what
`ASRModelManager` stores in `_shared_supports_streaming` and what
`TranscribeEngine._run_inference()` branches on:

    if TranscribeEngine._shared_supports_streaming:  -> session.stream(...)
    else:                                            -> session.run(full utterance)

If the active model is NOT streaming-capable, then every preview poll re-runs the whole
utterance through `session.run()`, and W2.2 "keep a stream open and feed deltas" is
impossible for that model. This script proves which case we are in.

Run from the project root:  python scratch/spike_stream_capability.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import transcribe_cpp  # noqa: E402

from backend_cpp.asr.model_registry import ModelRegistry  # noqa: E402

registry = ModelRegistry.get_instance()
active = registry.get_active_model_key()
path = registry.ensure_model(active)
print(f"active model: {active}\npath: {path}\n")

model = transcribe_cpp.Model(path)
caps = model.capabilities
print(f"model.arch              = {getattr(model, 'arch', None)}")
print(f"model.variant           = {getattr(model, 'variant', None)}")
print(f"model.backend           = {getattr(model, 'backend', None)}")
print(f"capabilities.supports_streaming = {getattr(caps, 'supports_streaming', None)!r}")

# Enumerate the rest of the capability flags so the W2.2 decision is fully informed.
extra = {
    name: getattr(caps, name)
    for name in dir(caps)
    if not name.startswith("_") and not callable(getattr(caps, name, None))
}
print(f"all capabilities        = {extra}")

tone = (0.2 * np.sin(2 * np.pi * 220.0 * np.arange(16000, dtype=np.float32) / 16000.0)).astype(
    np.float32
)
del tone  # placeholder; real speech is not needed to test stream() availability

pcm = np.zeros(16000, dtype=np.float32)

with model.session() as session:
    try:
        with session.stream(language=None) as stream:
            stream.feed(pcm[:8000])
            mid = stream.text()
            stream.feed(pcm[8000:])
            end_update = stream.finalize()
            text = stream.text()
        print("\n[session.stream()] AVAILABLE")
        print(f"  after 1st feed : committed={mid.committed!r} tentative={mid.tentative!r}")
        print(f"  finalize update: {end_update!r}")
        print(f"  final text     : full={text.full!r} committed={text.committed!r}")
    except Exception as exc:  # noqa: BLE001 - the point is to classify the failure
        print(f"\n[session.stream()] UNAVAILABLE -> {type(exc).__name__}: {exc}")

    result = session.run(pcm, language=None)
    print(f"\n[session.run()] text={getattr(result, 'text', result)!r}")
