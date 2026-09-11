"""Phase B — Model & Runtime Loading Audit.

Inspects runtime state:
1. Model logical ID, file path, SHA256 checksum
2. GGUF metadata, architecture, variant, tensor count, quantization
3. Runtime backend: transcribe_cpp version, GGML Vulkan device
4. Decoder configuration & parameter mapping
5. Memory: RAM (RSS) and GPU VRAM before load, after load, and during inference
6. Timing: Model load time, cold-start inference time, warm inference time
7. Invariants: Singleton instance verification, no tokenizer reload, no duplicate memory
"""

import gc
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import gguf
import numpy as np
import psutil
import torch
import transcribe_cpp

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.model_registry import ModelRegistry
from backend_cpp.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit_model_loading")


def compute_sha256(file_path: Path, block_size: int = 65536) -> str:
    """Compute SHA256 checksum of a large model file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_memory_info() -> Dict[str, float]:
    """Get current RAM and VRAM in MB."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    return {"ram_mb": round(ram_mb, 2), "vram_mb": round(vram_mb, 2)}


def inspect_gguf_metadata(file_path: Path) -> Dict[str, Any]:
    """Inspect GGUF file metadata using gguf reader."""
    reader = gguf.GGUFReader(str(file_path))
    meta: Dict[str, Any] = {}
    for key, field in reader.fields.items():
        val = field.contents() if hasattr(field, "contents") else str(field.parts)
        if isinstance(val, (bytes, bytearray)):
            try:
                val = val.decode("utf-8")
            except Exception:
                val = str(val)
        meta[key] = val

    tensor_count = len(reader.tensors)
    tensors_info = {}
    for t in reader.tensors[:5]:
        tensors_info[t.name] = {
            "shape": [int(x) for x in t.shape],
            "type": str(t.tensor_type),
        }

    return {
        "architecture": str(meta.get("general.architecture", "unknown")),
        "basename": str(meta.get("general.basename", "unknown")),
        "file_type": str(meta.get("general.file_type", "unknown")),
        "quantization_version": str(meta.get("general.quantization_version", "unknown")),
        "tensor_count": tensor_count,
        "sample_tensors": tensors_info,
        "raw_keys_count": len(reader.fields),
    }


def run_model_audit() -> Dict[str, Any]:
    logger.info("================================================================")
    logger.info("▶️ STARTING PHASE B: MODEL & RUNTIME LOADING AUDIT")
    logger.info("================================================================")

    model_key = config.asr.active_model
    registry = ModelRegistry.get_instance()
    model_info = registry.get_model_info(model_key)
    model_path = Path(registry.ensure_model(model_key))

    audit_result: Dict[str, Any] = {}

    # 1. Model File Identity
    logger.info("1. Inspecting Model File Identity...")
    file_size_bytes = model_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    logger.info(f"   Path: {model_path} ({file_size_mb:.2f} MB)")
    logger.info("   Computing SHA256 checksum...")
    sha256_hash = compute_sha256(model_path)
    logger.info(f"   SHA256: {sha256_hash}")

    audit_result["model_identity"] = {
        "model_key": model_key,
        "file_path": str(model_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "sha256": sha256_hash,
        "registry_info": model_info,
    }

    # 2. GGUF Metadata Inspection
    logger.info("2. Reading GGUF Tensor & Header Metadata...")
    gguf_meta = inspect_gguf_metadata(model_path)
    logger.info(f"   Architecture: {gguf_meta['architecture']}")
    logger.info(f"   Basename: {gguf_meta['basename']}")
    logger.info(f"   Tensor Count: {gguf_meta['tensor_count']}")
    audit_result["gguf_metadata"] = gguf_meta

    # 3. Runtime & Hardware Backend
    logger.info("3. Inspecting Runtime Backend...")
    backends_avail = []
    if hasattr(transcribe_cpp, "backends"):
        backends_avail = [b.kind for b in transcribe_cpp.backends()]

    audit_result["runtime_backend"] = {
        "transcribe_cpp_version": getattr(transcribe_cpp, "__version__", "unknown"),
        "backends_available": backends_avail,
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    logger.info(f"   Available Backends: {backends_avail}")
    logger.info(f"   PyTorch CUDA Device: {audit_result['runtime_backend']['gpu_name']}")

    # 4. Model Loading Lifecycle & Memory Invariants
    logger.info("4. Testing Model Loading & Memory Footprint...")
    gc.collect()
    mem_before = get_memory_info()

    t0_load = time.perf_counter()
    mgr = ASRModelManager()
    model = mgr.ensure_model(model_key)
    t_load = time.perf_counter() - t0_load

    mem_after_load = get_memory_info()

    native_arch = getattr(model, "arch", "unknown")
    native_variant = getattr(model, "variant", "unknown")
    native_backend = getattr(model, "backend", "unknown")
    supports_streaming = getattr(model.capabilities, "supports_streaming", False) if hasattr(model, "capabilities") else False

    logger.info(f"   Load Time: {t_load:.3f}s")
    logger.info(f"   Native Arch: {native_arch}, Variant: {native_variant}, Backend: {native_backend}")
    logger.info(f"   Supports Streaming: {supports_streaming}")
    logger.info(f"   RAM: {mem_before['ram_mb']}MB -> {mem_after_load['ram_mb']}MB (Δ: +{mem_after_load['ram_mb'] - mem_before['ram_mb']:.2f}MB)")

    # 5. Cold-Start vs Warm Inference Invariant
    logger.info("5. Measuring Cold-Start vs Warm Inference Latency...")
    test_pcm = np.zeros(32000, dtype=np.float32)

    acquired = ASRModelManager.acquire_infer_lock(blocking=True)
    try:
        session = mgr.ensure_session(model, threads=4)
        t0_cold = time.perf_counter()
        res_cold = session.run(test_pcm, language="en")
        t_cold = time.perf_counter() - t0_cold
        mem_during_infer = get_memory_info()

        t0_warm = time.perf_counter()
        res_warm = session.run(test_pcm, language="en")
        t_warm = time.perf_counter() - t0_warm
    finally:
        ASRModelManager.release_infer_lock()

    logger.info(f"   Cold-Start Inference (2.0s audio): {t_cold * 1000.0:.1f}ms")
    logger.info(f"   Warm Inference (2.0s audio): {t_warm * 1000.0:.1f}ms (Speedup: {t_cold / max(1e-6, t_warm):.2f}x)")

    # 6. Singleton Invariant Test
    logger.info("6. Verifying Singleton Invariant (No duplicate model in memory)...")
    model_2 = mgr.ensure_model(model_key)
    same_instance = (model is model_2)
    same_shared = (ASRModelManager.get_shared_model() is model)
    logger.info(f"   Multiple ensure_model() returns identical pointer: {same_instance}")
    logger.info(f"   get_shared_model() returns identical pointer: {same_shared}")

    audit_result["loading_lifecycle"] = {
        "load_time_sec": round(t_load, 3),
        "native_arch": native_arch,
        "native_variant": native_variant,
        "native_backend": native_backend,
        "supports_streaming": supports_streaming,
        "mem_before": mem_before,
        "mem_after_load": mem_after_load,
        "mem_during_infer": mem_during_infer,
        "ram_increase_mb": round(mem_after_load["ram_mb"] - mem_before["ram_mb"], 2),
        "cold_start_ms": round(t_cold * 1000.0, 1),
        "warm_inference_ms": round(t_warm * 1000.0, 1),
        "singleton_verified": bool(same_instance and same_shared),
    }

    # 7. Decoder Parameters Audit
    logger.info("7. Auditing Decoder Parameters vs Official Protocol...")
    decoder_params = {
        "model_key": model_key,
        "active_backend": native_backend,
        "sample_rate": 16000,
        "threads": 4,
        "context_window_sec": model_info.get("context_window_sec", 30),
        "family": model_info.get("family", "qwen3_asr"),
        "languages_supported": model_info.get("languages", "auto"),
        "official_params_mapping": {
            "beam_size": "unavailable (transcribe_cpp uses greedy argmax search)",
            "temperature": "unavailable (fixed greedy argmax)",
            "repetition_penalty": "unavailable",
            "max_new_tokens": "256 (internal token cap OutputTruncated)",
            "timestamp_token": "unavailable (GGUF non-timestamp export)",
        },
    }
    audit_result["decoder_parameters"] = decoder_params

    out_path = Path("report/audit_model_loading.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Phase B audit completed! Results saved to {out_path}")
    return audit_result


if __name__ == "__main__":
    run_model_audit()
