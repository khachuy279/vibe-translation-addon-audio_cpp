import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import soundfile as sf
import numpy as np
from backend_cpp.vad.vad_processor import VADProcessor

SESSION_DIR = REPO / "debug_audio" / "22d54213-6418-4985-86db-0d1fa892f2dc"
INGRESS_WAV = SESSION_DIR / "00_ingress_stream.wav"

def test_silence(silence_ms, threshold=0.20):
    pcm, sr = sf.read(str(INGRESS_WAV), dtype="float32")
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    
    segments = []
    cur = []
    
    def on_chunk(b, *args):
        cur.append(b)
        
    def on_end(reason=""):
        if cur:
            data = b"".join(cur)
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(arr) >= 1600:
                segments.append((len(arr)/16000.0, reason))
            cur.clear()
            
    vad = VADProcessor(
        vad_engine="fsmn-vad",
        sample_rate=16000,
        threshold=threshold,
        silence_duration_ms=silence_ms,
        hangover_ms=250,
        pre_speech_buffer_ms=800,
        on_speech_chunk=on_chunk,
        on_speech_end=on_end
    )
    
    chunk_size = 512 * 2 # 512 samples int16
    for i in range(0, len(pcm16), chunk_size):
        vad.feed_chunk(pcm16[i:i+chunk_size])
    vad.flush()
    if cur:
        on_end("FLUSH")
        
    durations = [s[0] for s in segments]
    print(f"silence_ms={silence_ms}: {len(segments)} segments, avg_dur={np.mean(durations):.2f}s, min={min(durations):.2f}s, max={max(durations):.2f}s")
    return segments

for sil in [150, 300, 500, 600, 800]:
    test_silence(sil)
