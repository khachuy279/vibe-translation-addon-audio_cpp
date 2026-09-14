import logging
import sys
import wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend_audio_cpp.asr.asr_engine import AudioCppASREngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    wav_path = "wav_test/Japanese_5s.wav"
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    engine = AudioCppASREngine()
    models = engine.list_available_models()
    print("\nRegistered ASR Models in models.yaml:")
    for m in models:
        print(f"  - {m['id']} ({m['name']}) [active={m['is_active']}]")

    # 1. Qwen3-ASR
    print("\n--- [Step 1] Default Model (Qwen3-ASR 1.7B) ---")
    text1, meta1 = engine.transcribe_chunk(pcm, utt_id="test_qwen", is_partial=False, sample_rate=sr)
    print(f"Output: {text1} | Latency: {meta1['wall_ms']:.1f}ms | RTF: {meta1['rtf']:.3f}")

    # 2. Live switch to Nemotron 3.5 Streaming
    print("\n--- [Step 2] Live Switch to Nemotron 3.5 Streaming ---")
    switched_key = engine.switch_model("nemotron")
    print(f"Active model key is now: {switched_key}")
    text2, meta2 = engine.transcribe_chunk(pcm, utt_id="test_nemotron", is_partial=False, sample_rate=sr)
    print(f"Output: {text2} | Latency: {meta2['wall_ms']:.1f}ms | RTF: {meta2['rtf']:.3f}")

    # 3. Live switch to Voxtral Mini 4B Realtime
    print("\n--- [Step 3] Live Switch to Voxtral Mini 4B Realtime ---")
    switched_key = engine.switch_model("voxtral")
    print(f"Active model key is now: {switched_key}")
    text3, meta3 = engine.transcribe_chunk(pcm, utt_id="test_voxtral", is_partial=False, sample_rate=sr)
    print(f"Output: {text3} | Latency: {meta3['wall_ms']:.1f}ms | RTF: {meta3['rtf']:.3f}")

    # 4. Live switch back to Qwen3
    print("\n--- [Step 4] Live Switch back to Qwen3 ---")
    switched_key = engine.switch_model("qwen3")
    print(f"Active model key is now: {switched_key}")
    text4, meta4 = engine.transcribe_chunk(pcm, utt_id="test_qwen_back", is_partial=False, sample_rate=sr)
    print(f"Output: {text4} | Latency: {meta4['wall_ms']:.1f}ms | RTF: {meta4['rtf']:.3f}")

    print("\nAll live model switching steps succeeded seamlessly without backend restart!")

if __name__ == "__main__":
    main()
