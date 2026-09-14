"""Transcribe the captured production session (debug_audio) to identify language/domain."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transcribe_cpp  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SESSION = REPO / "debug_audio" / "3688800d-0f25-48c0-bfde-497dff5a6486"
MODEL = REPO / "backend_cpp" / "models" / "Qwen3-ASR-1.7B-Q8_0.gguf"


def load(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import soxr

        data = soxr.resample(data, in_rate=sr, out_rate=16000, quality="HQ")
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def main() -> None:
    model = transcribe_cpp.Model(str(MODEL))
    session = model.session()

    ingress = SESSION / "00_ingress_stream.wav"
    pcm = load(ingress)
    print(f"=== INGRESS {ingress.name} ({len(pcm)/16000:.2f}s) ===")
    res = session.run(pcm, language=None)
    print("AUTO:", getattr(res, "text", ""))
    res = session.run(pcm, language="ja")
    print("JA  :", getattr(res, "text", ""))
    res = session.run(pcm, language="en")
    print("EN  :", getattr(res, "text", ""))

    print()
    for f in sorted(SESSION.glob("*_vad_*.wav")):
        pcm = load(f)
        if len(pcm) < 1600:
            continue
        res = session.run(pcm, language="ja")
        print(f"--- {f.name} ({len(pcm)/16000:.2f}s)")
        print("   ", getattr(res, "text", ""))

    session.close()
    model.close()


if __name__ == "__main__":
    main()
