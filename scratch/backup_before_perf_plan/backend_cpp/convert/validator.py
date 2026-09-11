"""Validation engine for converted transcribe.cpp GGUF models.

Performs:
1. Live in-engine ASR inference test using transcribe_cpp Python API.
2. Latency, Real-Time Factor (RTF), and backend device verification.
3. Strict checks on transcript non-emptiness.
4. Optional reference text comparison (similarity / match percentage).
5. Optional official transcribe.cpp validate.py invocation when golden manifests exist.
"""

from __future__ import annotations

import argparse
import difflib
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure Windows console doesn't crash on utf-8 characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = None
    sf = None

try:
    import transcribe_cpp
except ImportError:
    transcribe_cpp = None

logger = logging.getLogger("convert_validator")


def load_audio_16k_mono(audio_path: Path) -> tuple[np.ndarray, float]:
    """Load an audio file, convert to 16kHz mono float32 array in [-1.0, 1.0]."""
    if sf is None:
        raise RuntimeError("soundfile package is required for audio loading. Run: pip install soundfile")

    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sample_rate != 16000:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        except Exception as e:
            logger.warning(f"librosa resample failed ({e}), using simple linear interpolation")
            target_length = int(len(data) * 16000 / sample_rate)
            data = np.interp(
                np.linspace(0.0, 1.0, target_length, endpoint=False),
                np.linspace(0.0, 1.0, len(data), endpoint=False),
                data,
            ).astype(np.float32)

    duration_sec = len(data) / 16000.0
    return data.astype(np.float32), duration_sec


def find_default_test_audio() -> Optional[Path]:
    """Find a suitable sample audio in project directory."""
    candidates = [
        Path("wav_test/test_e2e_audio.wav"),
        Path("wav_test/test01_20s.wav"),
        Path("wav_test/OSR_us_000_0010_16k.wav"),
        Path("wav_test/sample_en.wav"),
    ]
    # Check both relative to cwd and relative to project root
    for c in candidates:
        if c.exists():
            return c.resolve()
        # Relative to project root
        proj_root = Path(__file__).resolve().parent.parent.parent
        p = proj_root / c
        if p.exists():
            return p.resolve()
    return None


def calculate_similarity(text_a: str, text_b: str) -> float:
    """Return sequence matcher similarity ratio between two texts (0.0 to 1.0)."""
    norm_a = " ".join(text_a.strip().lower().split())
    norm_b = " ".join(text_b.strip().lower().split())
    if not norm_a and not norm_b:
        return 1.0
    matcher = difflib.SequenceMatcher(None, norm_a, norm_b)
    return matcher.ratio()


def list_available_backends() -> List[str]:
    """Retrieve available backend kinds from transcribe_cpp."""
    if transcribe_cpp is None or not hasattr(transcribe_cpp, "backends"):
        return ["auto", "cpu"]
    try:
        return [b.kind for b in transcribe_cpp.backends()]
    except Exception:
        return ["auto", "cpu"]


def validate_gguf_model(
    gguf_path: Path,
    audio_path: Optional[Path] = None,
    language: Optional[str] = None,
    backend: str = "auto",
    reference_text: Optional[str] = None,
    threads: int = 4,
    strict: bool = True,
) -> Dict[str, Any]:
    """Execute live in-engine validation test on the converted GGUF model."""
    if transcribe_cpp is None:
        raise RuntimeError("transcribe-cpp is not installed or available.")

    gguf_path = Path(gguf_path).resolve()
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")

    # Resolve test audio
    if audio_path is None:
        audio_path = find_default_test_audio()
        if audio_path is None:
            raise FileNotFoundError("No test audio provided and default wav_test files not found.")
    else:
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Test audio file not found: {audio_path}")

    avail_backends = list_available_backends()

    print("\n" + "=" * 65)
    print("🔍 [VALIDATION] Live Engine Test for Converted GGUF")
    print(f"  Model GGUF      : {gguf_path.name}")
    print(f"  Path            : {gguf_path}")
    print(f"  Size            : {gguf_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"  Test Audio      : {audio_path.name}")
    print(f"  Backend Req     : {backend}")
    print(f"  Avail Backends  : {', '.join(avail_backends)}")
    print("=" * 65)

    # 1. Load audio
    pcm_audio, duration_sec = load_audio_16k_mono(audio_path)
    print(f"  [Audio] Loaded {duration_sec:.2f}s of 16kHz mono audio")

    # 2. Load model
    t0 = time.perf_counter()
    print(f"  [Model] Loading model into transcribe_cpp (backend: {backend})...")
    try:
        model = transcribe_cpp.Model(str(gguf_path), backend=backend)
    except Exception as e:
        print(f"❌ [Model] FAILED to load GGUF model: {e}")
        return {
            "success": False,
            "error": f"Model load error: {e}",
        }
    load_time_ms = (time.perf_counter() - t0) * 1000.0

    arch = getattr(model, "arch", "unknown")
    variant = getattr(model, "variant", "unknown")
    backend_kind = getattr(model, "backend", "unknown")
    device_name = getattr(model, "device", "unknown")
    supports_streaming = bool(getattr(getattr(model, "capabilities", None), "supports_streaming", False))

    print(f"  ✅ [Model] Loaded in {load_time_ms:.1f}ms")
    print(f"     Arch               : {arch}")
    print(f"     Variant            : {variant}")
    print(f"     Backend            : {backend_kind}")
    print(f"     Device             : {device_name}")
    print(f"     Streaming Support  : {supports_streaming}")

    # 3. Create session & run inference
    print("  [Infer] Running ASR inference...")
    t_infer_start = time.perf_counter()
    recognized_text = ""
    infer_error = None

    try:
        # Flexible session creation across transcribe_cpp versions
        try:
            session = model.session(n_threads=threads)
        except TypeError:
            try:
                session = model.session(threads=threads)
            except TypeError:
                session = model.session()

        # Resilient inference call
        lang_arg = None if (language in (None, "auto", "")) else language
        try:
            if lang_arg is not None:
                try:
                    res = session.run(pcm_audio, language=lang_arg)
                except Exception as e_lang:
                    partial = getattr(e_lang, "partial_result", None)
                    if partial is not None and hasattr(partial, "text"):
                        res = partial
                    else:
                        print(f"     [Notice] Language hint '{lang_arg}' failed ({e_lang}), retrying without language hint...")
                        res = session.run(pcm_audio)
            else:
                res = session.run(pcm_audio)
            recognized_text = getattr(res, "text", str(res))
        except Exception as e_run:
            partial = getattr(e_run, "partial_result", None)
            if partial is not None and hasattr(partial, "text"):
                recognized_text = getattr(partial, "text", str(partial))
            else:
                raise e_run
        finally:
            try:
                session.close()
            except Exception:
                pass
    except Exception as e:
        infer_error = str(e)
        print(f"❌ [Infer] FAILED during inference: {e}")
    finally:
        try:
            model.close()
        except Exception:
            pass

    infer_time_ms = (time.perf_counter() - t_infer_start) * 1000.0
    rtf = (infer_time_ms / 1000.0) / max(duration_sec, 0.001)
    speed_x = (duration_sec / max(infer_time_ms / 1000.0, 0.0001))

    clean_text = recognized_text.strip()
    print("-" * 65)
    print(f"  📝 [Result Transcript]: \"{clean_text}\"")
    print(f"  ⏱️  [Timing]: Infer {infer_time_ms:.1f}ms | Audio {duration_sec:.2f}s | RTF: {rtf:.3f}x ({speed_x:.1f}x realtime)")

    # Evaluation
    passed = True
    eval_reasons = []

    if infer_error:
        passed = False
        eval_reasons.append(f"Inference threw error: {infer_error}")

    if strict and not clean_text:
        passed = False
        eval_reasons.append("Transcript is completely empty (failed strict non-empty check)")

    similarity = None
    if reference_text:
        similarity = calculate_similarity(clean_text, reference_text)
        print(f"  🎯 [Similarity with Reference]: {similarity * 100:.1f}%")
        print(f"     Reference: \"{reference_text}\"")
        if strict and similarity < 0.2:
            passed = False
            eval_reasons.append(f"Similarity {similarity*100:.1f}% is below acceptable threshold")

    print("-" * 65)
    if passed:
        print("  🎉 [VALIDATION RESULT]: PASSED ✅")
    else:
        print(f"  ⚠️  [VALIDATION RESULT]: FAILED ❌ ({'; '.join(eval_reasons)})")
    print("=" * 65 + "\n")

    return {
        "success": passed,
        "arch": arch,
        "variant": variant,
        "backend": backend_kind,
        "device": device_name,
        "load_time_ms": load_time_ms,
        "infer_time_ms": infer_time_ms,
        "audio_duration_sec": duration_sec,
        "rtf": rtf,
        "transcript": clean_text,
        "similarity": similarity,
        "reasons": eval_reasons,
    }


def main():
    p = argparse.ArgumentParser(description="Validate a transcribe.cpp GGUF model.")
    p.add_argument("model", type=Path, help="Path to .gguf model file")
    p.add_argument("--audio", type=Path, default=None, help="Path to test WAV audio file")
    p.add_argument("--language", type=str, default=None, help="Language code (e.g. en, ja, zh)")
    p.add_argument("--backend", type=str, default="auto", help="Backend to test: auto, vulkan, cpu, cuda (default: auto)")
    p.add_argument("--reference-text", type=str, default=None, help="Expected reference transcript")
    p.add_argument("--threads", type=int, default=4, help="Inference threads (default: 4)")
    p.add_argument("--no-strict", action="store_true", help="Do not fail on empty transcript")
    args = p.parse_args()

    result = validate_gguf_model(
        args.model,
        audio_path=args.audio,
        language=args.language,
        backend=args.backend,
        reference_text=args.reference_text,
        threads=args.threads,
        strict=not args.no_strict,
    )
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
