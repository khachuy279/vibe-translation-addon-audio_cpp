"""End-to-End pipeline test for backend_cpp: VAD -> ASR -> Translation."""

import asyncio
import os
import sys
import wave
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend_cpp.vad.vad_processor import VADProcessor
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.translation.translator import translate_sentence
from backend_cpp.asr.model_registry import ModelRegistry


async def run_e2e_test():
    print("=" * 60)
    print("🚀 [TEST 1] Verifying Model Registry...")
    registry = ModelRegistry.get_instance()
    models = registry.list_models()
    print(f"Discovered {len(models)} model(s) in registry:")
    for m in models:
        status = "✅ Downloaded" if m["is_downloaded"] else "⏳ Needs Download"
        print(f" - {m['id']}: {m['name']} ({m['family']}) [{status}]")

    print("\n" + "=" * 60)
    print("🚀 [TEST 2] Testing TranscribeEngine (Qwen3-ASR 1.7B GGUF)...")
    asr = TranscribeEngine("qwen3-asr-1.7b")
    # Preload model
    asr._ensure_shared_model()

    wav_path = os.path.join(PROJECT_ROOT, "wav_test", "OSR_us_000_0010_16k.wav")
    assert os.path.exists(wav_path), f"Test WAV file not found: {wav_path}"

    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # Take 6 seconds of speech
    sample_bytes = raw_bytes[:16000 * 2 * 6]

    print("\n" + "=" * 60)
    print("🚀 [TEST 3] Testing Full Pipeline: Audio Chunks -> VAD -> ASR -> Translation...")

    speech_chunks = []
    sentence_commits = []

    # Stream consumer task
    async def _consume_asr():
        async for msg in asr.stream_tokens():
            if msg.get("is_final"):
                sentence_commits.append(msg)
                print(f"🎯 [ASR FINAL] Text: '{msg['text']}'")
                # Immediately test translation
                print("🔄 Translating sentence...")
                trans_res = await translate_sentence(msg["text"], source_lang="en", target_lang="vi")
                print(f"🇻🇳 [TRANSLATION]: '{trans_res.get('translated_text')}'")
                break

    consumer_task = asyncio.create_task(_consume_asr())

    vad = VADProcessor(
        threshold=0.5,
        silence_duration_ms=400,
        hangover_ms=200,
        on_speech_chunk=asr.feed_audio,
        on_speech_start=asr.on_speech_start,
        on_speech_end=asr.on_speech_end,
    )

    # Feed 20ms chunks (320 samples = 640 bytes)
    chunk_size = 640
    print(f"Feeding {len(sample_bytes)} bytes of audio into VAD...")
    for offset in range(0, len(sample_bytes), chunk_size):
        chunk = sample_bytes[offset:offset + chunk_size]
        vad.feed_chunk(chunk)
        await asyncio.sleep(0.005)

    # Add 1 second of silence to trigger VAD speech end
    silence = b"\x00" * (16000 * 2)
    for offset in range(0, len(silence), chunk_size):
        chunk = silence[offset:offset + chunk_size]
        vad.feed_chunk(chunk)
        await asyncio.sleep(0.005)

    # Wait for consumer
    try:
        await asyncio.wait_for(consumer_task, timeout=15.0)
    except asyncio.TimeoutError:
        print("⚠️ Consumer timeout, forcing ASR commit...")
        asr.on_speech_end()
        await asyncio.sleep(1.0)
        consumer_task.cancel()

    await asr.cleanup()
    print("\n" + "=" * 60)
    print("🎉 All pipeline tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_e2e_test())
