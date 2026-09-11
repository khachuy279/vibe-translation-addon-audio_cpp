"""Diagnostic forensic analysis for Cross_lingual_English_French_Italian_Spanish_6s.

Investigates:
1. What was the exact hypothesis and CER in Phase 3B S4 vs 3C.3?
2. What are the exact VAD boundaries detected by C06?
3. What is the impact of passing language='multi' vs language='auto' vs language=None?
4. What does offline uncut Qwen3-ASR produce on this file?
5. How does normalization/tokenization affect the CER?
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf

from backend_cpp.asr.audio_buffer import VAD_STATE_SPEECH, VAD_STATE_NON_SPEECH
from backend_cpp.asr.family_adapter import normalize_language_for_family
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.simulator import StreamingAudioSimulator

wav_path = Path("wav_test/Cross_lingual_English_French_Italian_Spanish_6s.wav")
txt_path = Path("wav_test/Cross_lingual_English_French_Italian_Spanish_6s.txt")

with open(txt_path, "r", encoding="utf-8") as f:
    ground_truth = f.read().strip()

print("=" * 80)
print("FORENSIC INVESTIGATION: Cross_lingual_English_French_Italian_Spanish_6s")
print(f"Ground Truth ({len(ground_truth)} chars): '{ground_truth}'")
print("=" * 80)

# 1. Inspect Audio & Offline Uncut ASR under different language tags
sim = StreamingAudioSimulator(audio_source=wav_path)
pcm_all = np.frombuffer(sim.pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
total_dur = len(pcm_all) / 16000.0
print(f"\nAudio: {total_dur:.2f}s ({len(pcm_all)} samples)")

mgr = ASRModelManager()
model = mgr.ensure_model("qwen3-asr-1.7b")

ASRModelManager.acquire_infer_lock(blocking=True)
try:
    session = mgr.ensure_session(model)

    print("\n--- TEST 1: Offline Uncut Audio (0..6.23s) under different language params ---")
    for lang_param in ["auto", None, "multi", "en"]:
        norm_lang = normalize_language_for_family(lang_param, "qwen3_asr")
        try:
            res = session.run(pcm_all, language=norm_lang)
            text = getattr(res, "text", str(res)).strip()
        except Exception as e:
            text = f"FAILED: {e}"
        acc = evaluate_accuracy(ground_truth, text, language="en")
        print(f"  param='{lang_param}' -> normalized='{norm_lang}':")
        print(f"     Text: '{text}'")
        print(f"     CER:  {acc.cer:.2%} | WER: {acc.wer:.2%}")
finally:
    ASRModelManager.release_infer_lock()


# 2. Inspect VAD Segmentation by C06
print("\n--- TEST 2: C06 VAD Segmentation Timeline ---")
vad_events = []
def on_start():
    vad_events.append(("START", time.time()))

def on_end():
    vad_events.append(("END", time.time()))

vad = VADProcessor(
    sample_rate=16000,
    vad_engine="fsmn-vad",
    threshold=0.20,
    silence_duration_ms=150,
    hangover_ms=250,
    pre_speech_buffer_ms=800,
    enabled=True,
    on_speech_start=on_start,
    on_speech_end=on_end,
)

frame_samples = 480  # 30ms
frame_bytes = frame_samples * 2
offset = 0
raw_bytes = sim.pcm16_bytes

while offset < len(raw_bytes):
    chunk = raw_bytes[offset : offset + frame_bytes]
    offset += len(chunk)
    vad.feed_chunk(chunk)

# Flush
sil_bytes = bytes(int(16000 * 1.5 * 2))
offset = 0
while offset < len(sil_bytes):
    chunk = sil_bytes[offset : offset + frame_bytes]
    offset += len(chunk)
    vad.feed_chunk(chunk)

print(f"  Total VAD events recorded: {len(vad_events)}")
for ev in vad_events:
    print(f"  VAD {ev[0]}")


# 3. Simulate Streaming with language='auto' vs language='multi'
print("\n--- TEST 3: Streaming Simulation Comparison ---")

async def test_streaming_with_lang(lang_setting: str):
    engine = TranscribeEngine(session_id=f"diag_{lang_setting}")
    engine._language = lang_setting
    
    vad_proc = VADProcessor(
        sample_rate=16000,
        vad_engine="fsmn-vad",
        threshold=0.20,
        silence_duration_ms=150,
        hangover_ms=250,
        pre_speech_buffer_ms=800,
        enabled=True,
        on_speech_chunk=engine.feed_audio,
        on_speech_start=engine.on_speech_start,
        on_speech_end=engine.on_speech_end,
    )
    
    commits = []
    
    async def _consume():
        try:
            async for msg in engine.stream_tokens():
                if msg.get("is_final"):
                    commits.append(msg)
        except asyncio.CancelledError:
            pass
            
    task = asyncio.create_task(_consume())
    
    # Feed chunks
    chunk_samples = int(16000 * 0.032)
    chunk_bytes = chunk_samples * 2
    off = 0
    while off < len(raw_bytes):
        c = raw_bytes[off : off + chunk_bytes]
        off += len(c)
        vad_proc.feed_chunk(c)
        await asyncio.sleep(0.010)
        
    # Flush silence
    sil_bytes = bytes(int(16000 * 1.5 * 2))
    off = 0
    while off < len(sil_bytes):
        c = sil_bytes[off : off + chunk_bytes]
        off += len(c)
        vad_proc.feed_chunk(c)
        await asyncio.sleep(0.010)
        
    await engine.cleanup()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
        
    full_hyp = " ".join([c["text"] for c in commits]).strip()
    acc = evaluate_accuracy(ground_truth, full_hyp, language=lang_setting)
    print(f"\nSetting: engine._language = '{lang_setting}'")
    print(f"  Commits count: {len(commits)}")
    for i, c in enumerate(commits):
        print(f"    Commit {i+1} [{c.get('commit_method')}]: '{c['text']}'")
    print(f"  Full: '{full_hyp}'")
    print(f"  CER:  {acc.cer:.2%} | WER: {acc.wer:.2%}")

asyncio.run(test_streaming_with_lang("auto"))
asyncio.run(test_streaming_with_lang("multi"))
