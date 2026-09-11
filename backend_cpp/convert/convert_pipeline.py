"""Transcribe.cpp ASR Model Conversion Pipeline.

Orchestrates:
1. Family selection (with auto-detection from Hugging Face metadata).
2. Model snapshot downloading to temp folder with sibling-based accurate size estimate.
3. Executing family converter script to produce reference GGUF via flexible adapter layer.
4. Quantization preset selection & running transcribe-quantize (with binary discovery).
5. Live engine validation with real audio playback test (RTF, latency, transcript check).
6. Automatic deployment to backend_cpp/models/ and registration in models.yaml.
7. Temp file cleanup management.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CONVERT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CONVERT_DIR / "scripts"
BIN_DIR = CONVERT_DIR / "bin"
TEMP_DIR = CONVERT_DIR / "temp"
CONFIG_FILE = CONVERT_DIR / "config.json"

BACKEND_CPP_DIR = CONVERT_DIR.parent
PROJECT_ROOT = BACKEND_CPP_DIR.parent
MODELS_DIR = BACKEND_CPP_DIR / "models"

sys.path.insert(0, str(CONVERT_DIR))
from sync_scripts import (
    FAMILY_CATALOG,
    sync_family_script,
    is_family_available,
    check_family_dependencies,
    build_converter_command,
)
from validator import validate_gguf_model, find_default_test_audio
from models_yaml_updater import slugify_model_key, generate_display_name, register_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_pipeline")


def prompt_choice(prompt_text: str, default: str = "") -> str:
    """Prompt user for input, or return default in non-interactive/redirected environments."""
    if not sys.stdin.isatty():
        return default
    try:
        val = input(prompt_text).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        return default


def load_persistent_config() -> Dict[str, Any]:
    """Load config.json from backend_cpp/convert/."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_persistent_config(cfg: Dict[str, Any]) -> None:
    """Save config.json in backend_cpp/convert/."""
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write config.json: {e}")


def find_quantize_binary(custom_path: Optional[str] = None) -> Optional[Path]:
    """Discover transcribe-quantize binary across known search locations."""
    # 1. Custom cli argument
    if custom_path:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p

    # 2. Saved in persistent config
    saved_cfg = load_persistent_config()
    saved_bin = saved_cfg.get("quantize_bin")
    if saved_bin:
        p = Path(saved_bin).resolve()
        if p.is_file():
            return p

    # 3. Standard discovery paths
    candidates = [
        BIN_DIR / "transcribe-quantize.exe",
        BIN_DIR / "transcribe-quantize",
        PROJECT_ROOT / "build" / "bin" / "transcribe-quantize.exe",
        PROJECT_ROOT / "build" / "bin" / "transcribe-quantize",
        PROJECT_ROOT / "transcribe.cpp" / "build" / "bin" / "transcribe-quantize.exe",
        PROJECT_ROOT / "transcribe.cpp" / "build" / "bin" / "transcribe-quantize",
    ]

    for c in candidates:
        if c.is_file():
            return c.resolve()

    # 4. Check system PATH
    which_bin = shutil.which("transcribe-quantize") or shutil.which("transcribe-quantize.exe")
    if which_bin:
        return Path(which_bin).resolve()

    return None


def fetch_hf_model_info(repo_id: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Query Hugging Face API for model metadata (tags, siblings/size, model_type)."""
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "transcribe-converter/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.debug(f"Could not fetch HF info for {repo_id}: {e}")
        return None


def detect_family_from_hf(repo_id: str, hf_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Attempt to automatically determine the model family from HF repo or metadata."""
    lowered = repo_id.lower()
    if "qwen3-asr" in lowered or "qwen3_asr" in lowered:
        return "qwen3_asr"
    if "sensevoice" in lowered:
        return "sensevoice"
    if "whisper" in lowered:
        return "whisper"
    if "parakeet" in lowered or "nemotron" in lowered:
        return "parakeet"
    if "moonshine" in lowered:
        return "moonshine"
    if "voxtral" in lowered:
        return "voxtral"
    if "canary" in lowered:
        return "canary"
    if "gigaam" in lowered:
        return "gigaam"
    if "granite" in lowered:
        return "granite"

    if hf_info:
        config_data = hf_info.get("config", {}) or {}
        m_type = config_data.get("model_type", "").lower()
        if m_type in FAMILY_CATALOG:
            return m_type
        tags = [t.lower() for t in hf_info.get("tags", [])]
        for t in tags:
            if t in FAMILY_CATALOG:
                return t
    return None


def display_family_menu() -> str:
    """Show interactive CLI menu for family selection."""
    print("\n" + "=" * 70)
    print("📋 [SUPPORTED FAMILIES] Supported by transcribe.cpp:")
    print("=" * 70)
    family_keys = list(FAMILY_CATALOG.keys())
    for idx, key in enumerate(family_keys, start=1):
        info = FAMILY_CATALOG[key]
        status_tag = "⭐" if info.get("status") == "stable" else ("⚠️" if info.get("status") == "experimental" else "  ")
        print(f" {status_tag} [{idx:2d}] {key:<20} : {info['name']} ({info['architecture_type']})")
        print(f"       Description: {info['description']}")
    print("=" * 70)

    while True:
        choice = prompt_choice("\n👉 Select family by number or name (e.g. 1 or qwen3_asr): ", default="1")
        if choice.isdigit():
            val = int(choice)
            if 1 <= val <= len(family_keys):
                return family_keys[val - 1]
        elif choice in FAMILY_CATALOG:
            return choice
        print("❌ Invalid selection. Please try again.")


def run_pipeline(
    family: Optional[str] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    quant: str = "Q8_0",
    temp_dir: Optional[Path] = None,
    quantize_bin: Optional[str] = None,
    audio_path: Optional[Path] = None,
    backend: str = "auto",
    skip_validation: bool = False,
    skip_quantize: bool = False,
    keep_temp: Optional[bool] = None,
    dry_run: bool = False,
    force: bool = False,
    offline: bool = False,
    no_download: bool = False,
    variant: Optional[str] = None,
) -> bool:
    """Main pipeline execution workflow."""
    print("\n" + "=" * 70)
    print("🎙️  TRANSCRIBE.CPP ASR MODEL CONVERSION PIPELINE")
    print("=" * 70)

    temp_root = (temp_dir or TEMP_DIR).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prompt / Resolve Repo ID
    if not repo_id:
        repo_id = prompt_choice("\n👉 Enter Hugging Face model repository ID (e.g. neosophie/Qwen3-ASR-1.7B-JA): ")
        while not repo_id:
            repo_id = prompt_choice("👉 Enter Hugging Face model repository ID: ")

    print(f"\n🔎 Querying Hugging Face metadata for '{repo_id}'...")
    hf_info = fetch_hf_model_info(repo_id)

    # Accurate download size estimation from siblings
    estimated_size_mb = 0.0
    if hf_info:
        siblings = hf_info.get("siblings", [])
        total_weight_bytes = 0
        for s in siblings:
            fname = s.get("rfilename", "")
            if any(fname.endswith(ext) for ext in (".safetensors", ".bin", ".nemo", ".pt", ".pth")):
                total_weight_bytes += s.get("size", 0) or 0
        if total_weight_bytes > 0:
            estimated_size_mb = total_weight_bytes / (1024 * 1024)
            print(f"📊 [Model Info] Weight Files Size: ~{estimated_size_mb:.0f} MB ({len(siblings)} files)")
        elif "safetensors" in hf_info:
            st = hf_info["safetensors"]
            total_params = st.get("total", 0)
            if total_params:
                estimated_size_mb = (total_params * 2) / (1024 * 1024)
                print(f"📊 [Model Info] Parameters: ~{total_params / 1e9:.2f}B | Est. Size: ~{estimated_size_mb:.0f} MB")

    # 2. Resolve Family
    detected_family = detect_family_from_hf(repo_id, hf_info)
    if not family:
        if detected_family:
            print(f"💡 Detected family: '{detected_family}' ({FAMILY_CATALOG[detected_family]['name']})")
            confirm = prompt_choice(f"👉 Use '{detected_family}'? [Y/n]: ", default="y").lower()
            if confirm in ("", "y", "yes"):
                family = detected_family
            else:
                family = display_family_menu()
        else:
            family = display_family_menu()

    if family not in FAMILY_CATALOG:
        print(f"❌ Unknown family '{family}'. Available: {list(FAMILY_CATALOG.keys())}")
        return False

    fam_info = FAMILY_CATALOG[family]
    print(f"\n✅ Selected Family: {fam_info['name']} (Key: {family}, Status: {fam_info.get('status', 'supported')})")
    if fam_info.get("status") == "experimental":
        print(f"⚠️  [WARNING] Family '{family}' is marked as experimental upstream in transcribe.cpp.")

    # Check dependencies with version checking
    dep_ok, missing_deps = check_family_dependencies(family)
    if not dep_ok:
        print(f"❌ Missing or incompatible Python packages for {family}:")
        for md in missing_deps:
            print(f"   - {md}")
        print(f"👉 Please install/update them via: pip install {' '.join(missing_deps)}")
        return False

    # 3. Synchronize / Locate Converter Script
    print(f"\n📦 Resolving converter script '{fam_info['script']}'...")
    if not is_family_available(family):
        if offline:
            print(f"❌ Offline mode enabled, but script '{fam_info['script']}' not found in {SCRIPTS_DIR}.")
            return False
        print(f"  Fetching latest '{fam_info['script']}' and shared libraries from transcribe.cpp...")
        script_path = sync_family_script(family)
    else:
        script_path = SCRIPTS_DIR / fam_info["script"]

    if not script_path or not script_path.exists():
        print(f"❌ Converter script '{fam_info['script']}' is not available.")
        return False
    print(f"  Converter script ready: {script_path.name}")

    slug = repo_id.strip().strip("/").split("/")[-1]
    model_temp_dir = temp_root / slug

    # 4. Resolve Quantization Preset
    if skip_quantize:
        quant = "BF16"
    else:
        valid_presets = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "F16", "F32"]
        quant = quant.upper()
        if quant not in valid_presets:
            print(f"\n⚠️  Requested quant '{quant}' not in preset list {valid_presets}. Defaulting to Q8_0.")
            quant = "Q8_0"

    print(f"🎯 Target Quantization Preset: {quant}")

    # Check quantize binary early if block quant is needed
    resolved_quant_bin = None
    if quant not in ("BF16", "F16", "F32"):
        resolved_quant_bin = find_quantize_binary(quantize_bin)
        if resolved_quant_bin:
            print(f"⚙️  [transcribe-quantize] Found tool binary: {resolved_quant_bin}")
            save_persistent_config({"quantize_bin": str(resolved_quant_bin)})
        else:
            print("\n" + "!" * 70)
            print(f"⚠️  [NOTICE] 'transcribe-quantize' binary not found on this machine!")
            print("   To quantize into Q8_0 / Q4_K_M, compile tools/transcribe-quantize from repo")
            print("   or place transcribe-quantize.exe into: backend_cpp/convert/bin/")
            print("!" * 70)
            print("\nOptions:")
            print("  [1] Keep reference BF16 format (Transcribe.cpp runs BF16 directly on Vulkan GPU!)")
            print("  [2] Specify custom path to transcribe-quantize.exe")
            print("  [3] Abort conversion")
            opt = prompt_choice("👉 Choose option [1/2/3] (Default: 1): ", default="1")
            if opt == "2":
                custom_bin_input = prompt_choice("👉 Enter path to transcribe-quantize.exe: ")
                if Path(custom_bin_input).is_file():
                    resolved_quant_bin = Path(custom_bin_input).resolve()
                    save_persistent_config({"quantize_bin": str(resolved_quant_bin)})
                    print(f"✅ Using custom quantize tool: {resolved_quant_bin}")
                else:
                    print(f"❌ File not found: {custom_bin_input}. Falling back to BF16.")
                    quant = "BF16"
            elif opt == "3":
                print("Aborted by user.")
                return False
            else:
                print("Proceeding with unquantized BF16 format.")
                quant = "BF16"

    # Dry-run preview
    ref_gguf_name = f"{slug}-BF16.gguf"
    final_gguf_name = f"{slug}-{quant}.gguf"
    ref_gguf_path = model_temp_dir / ref_gguf_name
    final_gguf_path = (MODELS_DIR / final_gguf_name) if quant in ("BF16", "F16", "F32") else (model_temp_dir / final_gguf_name)

    source_fmt = fam_info.get("source_format", "hf_snapshot")
    should_predownload = (source_fmt == "hf_snapshot" and not no_download)
    model_source = str(model_temp_dir) if should_predownload else repo_id

    # Canonical variant resolution if not supplied
    if not variant:
        lowered_repo = repo_id.lower()
        if family == "qwen3_asr":
            if "0.6b" in lowered_repo:
                variant = "qwen3-asr-0.6b"
            elif "1.7b" in lowered_repo:
                variant = "qwen3-asr-1.7b"
        elif family == "sensevoice":
            variant = "sensevoice-small"

    if dry_run:
        print("\n[DRY RUN] Would perform:")
        if should_predownload:
            print(f"  1. Download '{repo_id}' -> {model_temp_dir}")
        else:
            print(f"  1. Direct source passing to script: '{repo_id}'")
        cmd_preview = build_converter_command(
            family_id=family,
            script_path=script_path,
            model_source=model_source,
            out_path=ref_gguf_path,
            repo_id=repo_id,
            revision=revision,
            variant=variant,
        )
        print(f"  2. Run converter: {' '.join(cmd_preview)}")
        if quant not in ("BF16", "F16", "F32"):
            print(f"  3. Quantize: {resolved_quant_bin} {ref_gguf_path} {final_gguf_path} --quant {quant}")
        print(f"  4. Validate: {final_gguf_path} (backend: {backend})")
        print(f"  5. Copy to: {MODELS_DIR / final_gguf_name}")
        print(f"  6. Register in models.yaml with key '{slugify_model_key(repo_id, quant)}'")
        return True

    # 5. Download Model from Hugging Face (if hf_snapshot format and not bypassed)
    if should_predownload:
        print(f"\n⬇️  [DOWNLOAD] Downloading snapshot '{repo_id}' into '{model_temp_dir}'...")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("❌ huggingface_hub is required. Run: pip install huggingface_hub")
            return False

        model_temp_dir.mkdir(parents=True, exist_ok=True)
        t_dl_start = time.perf_counter()
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(model_temp_dir),
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
            )
        except Exception as e:
            print(f"❌ [DOWNLOAD FAILED]: {e}")
            return False
        print(f"✅ [DOWNLOAD COMPLETED] in {(time.perf_counter() - t_dl_start):.1f}s")
    else:
        print(f"\nℹ️  [SOURCE] Family '{family}' handles checkpoint resolution directly via '{repo_id}'")

    # 6. Run Family Converter Script
    print(f"\n⚙️  [CONVERT] Running {script_path.name}...")
    ref_gguf_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_converter_command(
        family_id=family,
        script_path=script_path,
        model_source=model_source,
        out_path=ref_gguf_path,
        repo_id=repo_id,
        revision=revision,
        variant=variant,
    )
    print(f"   Command: {' '.join(cmd)}")

    t_conv_start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    # Ensure TRANSCRIBE_MODELS_DIR points to temp folder for upstream download_snapshot
    env["TRANSCRIBE_MODELS_DIR"] = str(temp_root)

    proc = subprocess.run(cmd, cwd=str(SCRIPTS_DIR), env=env, capture_output=True, text=True)

    # Save full subprocess output to log file
    log_file = temp_root / "last_convert.log"
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"COMMAND: {' '.join(cmd)}\n")
            lf.write(f"RETURN CODE: {proc.returncode}\n\n")
            if proc.stdout:
                lf.write("=== STDOUT ===\n" + proc.stdout + "\n\n")
            if proc.stderr:
                lf.write("=== STDERR ===\n" + proc.stderr + "\n\n")
    except Exception as e:
        logger.warning(f"Could not write last_convert.log: {e}")

    # Check if target GGUF exists (some scripts like outdir might generate filename internally)
    if not ref_gguf_path.exists():
        # Check if any .gguf was generated in ref_gguf_path.parent
        possible_ggufs = list(ref_gguf_path.parent.glob("*.gguf"))
        if possible_ggufs:
            ref_gguf_path = possible_ggufs[0]

    if proc.returncode != 0 or not ref_gguf_path.exists():
        print(f"\n❌ [CONVERT FAILED] Converter exited with code {proc.returncode}")
        print(f"   Detailed log saved to: {log_file}")
        if proc.stdout:
            stdout_lines = proc.stdout.strip().splitlines()[-30:]
            print("--- Last 30 lines of Standard Output ---")
            for line in stdout_lines:
                print(f"  {line}")
        if proc.stderr:
            stderr_lines = proc.stderr.strip().splitlines()[-30:]
            print("--- Last 30 lines of Standard Error ---")
            for line in stderr_lines:
                print(f"  {line}")
        return False

    print(f"✅ [CONVERT COMPLETED] Reference GGUF produced in {(time.perf_counter() - t_conv_start):.1f}s")
    print(f"   Size: {ref_gguf_path.stat().st_size / (1024 * 1024):.1f} MB -> {ref_gguf_path}")

    # 7. Quantization Step
    deployed_gguf_path = MODELS_DIR / final_gguf_name

    if quant in ("BF16", "F16", "F32") or not resolved_quant_bin:
        # No requantization needed, reference GGUF is our target
        print(f"\n⏩ [QUANTIZE] Skipping requantization (Target is {quant})")
        shutil.copy2(ref_gguf_path, deployed_gguf_path)
    else:
        print(f"\n🔨 [QUANTIZE] Requantizing with transcribe-quantize to {quant}...")
        quant_cmd = [
            str(resolved_quant_bin),
            str(ref_gguf_path),
            str(deployed_gguf_path),
            "--quant",
            quant,
        ]
        print(f"   Command: {' '.join(quant_cmd)}")
        t_q_start = time.perf_counter()
        q_proc = subprocess.run(quant_cmd)
        if q_proc.returncode != 0 or not deployed_gguf_path.exists():
            print(f"❌ [QUANTIZE FAILED] transcribe-quantize exited with code {q_proc.returncode}")
            print(f"   Falling back to unquantized reference GGUF: {ref_gguf_name}")
            quant = "BF16"
            final_gguf_name = ref_gguf_name
            deployed_gguf_path = MODELS_DIR / final_gguf_name
            shutil.copy2(ref_gguf_path, deployed_gguf_path)
        else:
            print(f"✅ [QUANTIZE COMPLETED] in {(time.perf_counter() - t_q_start):.1f}s")
            print(f"   Final GGUF Size: {deployed_gguf_path.stat().st_size / (1024 * 1024):.1f} MB")

    # 8. Validation Step
    if not skip_validation:
        val_audio = audio_path or find_default_test_audio()
        try:
            val_result = validate_gguf_model(
                deployed_gguf_path,
                audio_path=val_audio,
                backend=backend,
                strict=True,
            )
            if not val_result["success"]:
                print(f"⚠️  [VALIDATION WARNING] Live validation did not pass all checks.")
                proceed_anyway = prompt_choice("👉 Continue and register model anyway? [y/N]: ", default="n").lower()
                if proceed_anyway not in ("y", "yes"):
                    print("Registration aborted due to validation failure.")
                    return False
        except Exception as e:
            print(f"⚠️  [VALIDATION ERROR] Could not run live validation: {e}")
            proceed_anyway = prompt_choice("👉 Continue and register model anyway? [y/N]: ", default="n").lower()
            if proceed_anyway not in ("y", "yes"):
                return False
    else:
        print("⏩ [VALIDATION] Skipped validation per flag.")

    # 9. Register in models.yaml
    print("\n📝 [MODELS.YAML] Registering model in models.yaml...")
    model_key = slugify_model_key(repo_id, quant)
    display_name = generate_display_name(repo_id, family, quant)

    # Derive languages
    languages_val = fam_info.get("default_languages", "auto")
    if hf_info:
        card_data = hf_info.get("cardData", {}) or {}
        hf_langs = card_data.get("language")
        if hf_langs:
            languages_val = hf_langs if isinstance(hf_langs, list) else [hf_langs]

    try:
        register_model(
            model_key=model_key,
            name=display_name,
            family=family,
            architecture_type=fam_info["architecture_type"],
            hf_repo=repo_id,
            default_quant=quant,
            file_name=final_gguf_name,
            sample_rate=fam_info["sample_rate"],
            languages=languages_val,
            vram_estimate_mb=fam_info["vram_mb"],
            description=f"{display_name} converted via transcribe.cpp pipeline",
            overwrite=force,
        )
    except Exception as e:
        print(f"❌ Failed to register in models.yaml: {e}")
        return False

    # 10. Cleanup Temp Files
    if keep_temp is None:
        cleanup_choice = prompt_choice(f"\n👉 Delete temporary download folder '{model_temp_dir}' to free space? [Y/n]: ", default="y").lower()
        should_clean = cleanup_choice in ("", "y", "yes")
    else:
        should_clean = not keep_temp

    if should_clean:
        print(f"🧹 Cleaning up temp directory: {model_temp_dir} ...")
        shutil.rmtree(model_temp_dir, ignore_errors=True)
        print("  Temp files removed.")
    else:
        print(f"📁 Kept temporary directory: {model_temp_dir}")

    print("\n" + "=" * 70)
    print("🎉 CONVERSION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Model Key    : {model_key}")
    print(f"  GGUF File    : {deployed_gguf_path}")
    print(f"  Display Name : {display_name}")
    print(f"  Family       : {family} ({fam_info['architecture_type']})")
    print(f"  Quantization : {quant}")
    print("=" * 70 + "\n")
    return True


def main():
    p = argparse.ArgumentParser(description="Transcribe.cpp ASR Model Conversion Pipeline.")
    p.add_argument("--family", type=str, default=None, help="Model family key (e.g. qwen3_asr, sensevoice, whisper)")
    p.add_argument("--repo-id", type=str, default=None, help="Hugging Face repo ID (e.g. neosophie/Qwen3-ASR-1.7B-JA)")
    p.add_argument("--revision", type=str, default=None, help="Hugging Face branch / commit SHA to pin download")
    p.add_argument("--quant", type=str, default="Q8_0", help="Quant preset: Q8_0, Q4_K_M, Q5_K_M, BF16, F16 (default: Q8_0)")
    p.add_argument("--temp-dir", type=Path, default=None, help="Custom temporary folder for downloads")
    p.add_argument("--quantize-bin", type=str, default=None, help="Path to transcribe-quantize executable")
    p.add_argument("--audio", type=Path, default=None, help="Test audio for live validation")
    p.add_argument("--backend", type=str, default="auto", help="Validation backend: auto, vulkan, cpu, cuda (default: auto)")
    p.add_argument("--skip-validation", action="store_true", help="Skip live validation step")
    p.add_argument("--skip-quantize", action="store_true", help="Keep source BF16/F16 without requantizing")
    p.add_argument("--keep-temp", action="store_true", help="Do not delete temporary files after convert")
    p.add_argument("--dry-run", action="store_true", help="Print actions without downloading or converting")
    p.add_argument("--force", action="store_true", help="Overwrite existing model entry in models.yaml")
    p.add_argument("--offline", action="store_true", help="Do not fetch new scripts from GitHub")
    p.add_argument("--no-download", action="store_true", help="Do not pre-download HF snapshot; let converter script handle source")
    p.add_argument("--variant", type=str, default=None, help="Architecture variant (e.g. qwen3-asr-1.7b, sensevoice-small)")
    args = p.parse_args()

    success = run_pipeline(
        family=args.family,
        repo_id=args.repo_id,
        revision=args.revision,
        quant=args.quant,
        temp_dir=args.temp_dir,
        quantize_bin=args.quantize_bin,
        audio_path=args.audio,
        backend=args.backend,
        skip_validation=args.skip_validation,
        skip_quantize=args.skip_quantize,
        keep_temp=True if args.keep_temp else None,
        dry_run=args.dry_run,
        force=args.force,
        offline=args.offline,
        no_download=args.no_download,
        variant=args.variant,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
