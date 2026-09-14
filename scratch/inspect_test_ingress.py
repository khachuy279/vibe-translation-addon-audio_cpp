from pathlib import Path
import soundfile as sf
import numpy as np
import transcribe_cpp

def main():
    model = transcribe_cpp.Model("backend_cpp/models/Qwen3-ASR-1.7B-Q8_0.gguf")
    sess = model.session()
    p = Path("debug_audio/22d54213-6418-4985-86db-0d1fa892f2dc")
    commits = sorted(p.glob("*_asr_commit.wav"))[:10]
    for f in commits:
        pcm, sr = sf.read(str(f), dtype="float32")
        dur = len(pcm) / 16000.0
        res = sess.run(pcm, language="ja")
        text = getattr(res, "text", "")
        print(f"{f.name} ({dur:.2f}s): {text}")
    sess.close()
    model.close()

if __name__ == "__main__":
    main()
