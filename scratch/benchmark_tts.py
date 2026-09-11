import os
import sys
import time
import torch
import numpy as np

# Set UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

# Ensure path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Patch torchaudio.load with soundfile to avoid torchcodec issues on Windows
import soundfile as sf
import torchaudio

def _custom_torchaudio_load(path, *args, **kwargs):
    data, sr = sf.read(path, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.T
    return tensor, sr

torchaudio.load = _custom_torchaudio_load


TEST_CASES = [
    {
        "name": "Câu ngắn (3 từ)",
        "text": "Sẵn sàng nhé."
    },
    {
        "name": "Câu vừa (20 từ - thực tế dịch phim)",
        "text": "Đúng là một sự trùng hợp kỳ lạ, khi đầu cô ấy nằm trên đùi tôi, thì đó chính là tương lai."
    },
    {
        "name": "Câu dài (36 từ)",
        "text": "Trời ơi cái giọng nó tự nhiên mà nó mượt mà dã man, nghe không khác gì người thật luôn. Giờ thì tha hồ mà trải nghiệm với kho giọng nói đa dạng đủ mọi sắc thái biểu cảm."
    }
]

REF_AUDIO = os.path.join(ROOT_DIR, "backend_cpp", "voices", "speaker_01_0039.wav")
REF_TEXT = "Suốt quãng đời học sinh, tôi luôn giữ trong lòng một tình cảm đặc biệt dành cho cô bạn ngồi cùng bàn."

def benchmark():
    print("=" * 70)
    print("🚀 BẮT ĐẦU BENCHMARK: OmniVoice (PyTorch CUDA) vs VieNeu-TTS-v3-Turbo")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Ref Audio: {REF_AUDIO}")
    print("-" * 70)

    # 1. Khởi tạo OmniVoice
    print("\n📦 [1/2] Đang nạp mô hình OmniVoice (PyTorch float16 on cuda:0)...")
    t0 = time.perf_counter()
    from backend_cpp.tts.omnivoice_engine import OmniVoiceTTS
    from backend_cpp.config import config
    config.tts.device = "cuda:0"
    config.tts.num_inference_steps = 8
    omni = OmniVoiceTTS.get_instance()
    omni.load_model()
    omni_load_time = time.perf_counter() - t0
    print(f"✅ OmniVoice nạp xong trong {omni_load_time:.2f}s!")

    # 2. Khởi tạo VieNeu-TTS-v3-Turbo
    print("\n📦 [2/2] Đang nạp mô hình VieNeu-TTS-v3-Turbo (vieneu)...")
    t0 = time.perf_counter()
    import vieneu
    vieneu_tts = vieneu.Vieneu()
    vieneu_load_time = time.perf_counter() - t0
    print(f"✅ VieNeu-TTS nạp xong trong {vieneu_load_time:.2f}s!")

    # Warm-up cả 2
    print("\n🔥 Đang warm-up cả 2 mô hình...")
    omni.synthesize_sync("Khởi động.", "speaker_01_0039.wav", speed=1.0)
    vieneu_tts.infer("Khởi động.", ref_audio=REF_AUDIO, denoise=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("✅ Warm-up hoàn tất!\n")

    results = []

    for idx, tc in enumerate(TEST_CASES, 1):
        name = tc["name"]
        text = tc["text"]
        print(f"\n[{idx}/{len(TEST_CASES)}] {name}: '{text}'")

        # --- Test OmniVoice (Steps=8) ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        wav_omni_bytes, omni_dur = omni.synthesize_sync(text, "speaker_01_0039.wav", speed=1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        omni_latency = time.perf_counter() - t_start
        omni_rtf = omni_latency / omni_dur if omni_dur > 0 else 0

        # --- Test VieNeu-TTS Clone (ref_audio) ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        vieneu_audio = vieneu_tts.infer(text, ref_audio=REF_AUDIO, denoise=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vieneu_latency = time.perf_counter() - t_start
        # sample_rate của vieneu v3 turbo là 48000
        vieneu_dur = len(vieneu_audio) / 48000.0 if len(vieneu_audio) > 0 else 0
        vieneu_rtf = vieneu_latency / vieneu_dur if vieneu_dur > 0 else 0

        # --- Test VieNeu-TTS Preset Voice (không cần ref_audio) ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        vieneu_preset_audio = vieneu_tts.infer(text)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vieneu_preset_latency = time.perf_counter() - t_start
        vieneu_preset_dur = len(vieneu_preset_audio) / 48000.0 if len(vieneu_preset_audio) > 0 else 0
        vieneu_preset_rtf = vieneu_preset_latency / vieneu_preset_dur if vieneu_preset_dur > 0 else 0

        print(f"  ├─ OmniVoice (Clone):      Latency = {omni_latency*1000:6.1f} ms | Audio = {omni_dur:.2f}s | RTF = {omni_rtf:.3f}")
        print(f"  ├─ VieNeu-TTS (Clone):     Latency = {vieneu_latency*1000:6.1f} ms | Audio = {vieneu_dur:.2f}s | RTF = {vieneu_rtf:.3f}")
        print(f"  └─ VieNeu-TTS (Preset):    Latency = {vieneu_preset_latency*1000:6.1f} ms | Audio = {vieneu_preset_dur:.2f}s | RTF = {vieneu_preset_rtf:.3f}")

        results.append({
            "case": name,
            "omni_lat": omni_latency,
            "omni_dur": omni_dur,
            "omni_rtf": omni_rtf,
            "vie_clone_lat": vieneu_latency,
            "vie_clone_dur": vieneu_dur,
            "vie_clone_rtf": vieneu_rtf,
            "vie_preset_lat": vieneu_preset_latency,
            "vie_preset_dur": vieneu_preset_dur,
            "vie_preset_rtf": vieneu_preset_rtf,
        })

    print("\n" + "=" * 70)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH:")
    print("=" * 70)
    print(f"{'Trường hợp':<30} | {'OmniVoice Latency':<18} | {'VieNeu Clone':<15} | {'VieNeu Preset':<15}")
    print("-" * 85)
    for r in results:
        print(f"{r['case']:<30} | {r['omni_lat']*1000:6.1f}ms (RTF {r['omni_rtf']:.2f}) | {r['vie_clone_lat']*1000:6.1f}ms ({r['vie_clone_rtf']:.2f}) | {r['vie_preset_lat']*1000:6.1f}ms ({r['vie_preset_rtf']:.2f})")
    print("=" * 70)

if __name__ == "__main__":
    benchmark()
