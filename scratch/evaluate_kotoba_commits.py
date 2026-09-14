import sys
import re
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import soundfile as sf
import transcribe_cpp
from benchmarks.ja_text import score_ja, normalize_ja

SESSION_DIR = REPO / "debug_audio" / "22d54213-6418-4985-86db-0d1fa892f2dc"
GROUND_TRUTH_TXT = SESSION_DIR / "00_ingress_stream.txt"

def load_ground_truth():
    raw = GROUND_TRUTH_TXT.read_text(encoding="utf-8")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    turns = []
    for line in lines:
        m = re.match(r"^\[(\d{2}:\d{2})\]\s*Speaker\s*(\d+):\s*(.*)$", line)
        if m:
            _, _, text = m.groups()
            turns.append(normalize_ja(text))
    return "".join(turns)

def main():
    ref = load_ground_truth()
    commit_wavs = sorted(SESSION_DIR.glob("*_asr_commit.wav"), key=lambda p: int(p.name.split("_")[0]))
    
    # Test Kotoba
    print("Testing Kotoba-whisper-v2.2...")
    model_kotoba = transcribe_cpp.Model(str(REPO / "backend_cpp" / "models" / "kotoba-whisper-v2.2-BF16.gguf"))
    sess_kotoba = model_kotoba.session()
    kotoba_hyps = []
    for f in commit_wavs:
        pcm, _ = sf.read(str(f), dtype="float32")
        if len(pcm) < 800:
            continue
        res = sess_kotoba.run(pcm, language="ja")
        t = getattr(res, "text", "").strip()
        if t:
            kotoba_hyps.append(normalize_ja(t))
    sess_kotoba.close()
    model_kotoba.close()
    
    full_hyp_kotoba = "".join(kotoba_hyps)
    score_k = score_ja(ref, full_hyp_kotoba)
    print(f"Kotoba CER: {score_k.cer*100:.2f}% | ITN: {score_k.cer_itn*100:.2f}% (Sub: {score_k.substitutions}, Del: {score_k.deletions}, Ins: {score_k.insertions})")

if __name__ == "__main__":
    main()
