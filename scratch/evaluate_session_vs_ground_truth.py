import sys
import re
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np
import soundfile as sf
import transcribe_cpp
from benchmarks.ja_text import score_ja, normalize_ja
SESSION_DIR = REPO / "debug_audio" / "22d54213-6418-4985-86db-0d1fa892f2dc"
INGRESS_WAV = SESSION_DIR / "00_ingress_stream.wav"
GROUND_TRUTH_TXT = SESSION_DIR / "00_ingress_stream.txt"

def load_ground_truth():
    raw = GROUND_TRUTH_TXT.read_text(encoding="utf-8")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    turns = []
    for line in lines:
        m = re.match(r"^\[(\d{2}:\d{2})\]\s*Speaker\s*(\d+):\s*(.*)$", line)
        if m:
            time_str, speaker, text = m.groups()
            turns.append({
                "time": time_str,
                "speaker": int(speaker),
                "raw_text": text,
                "norm_text": normalize_ja(text)
            })
    return turns

def main():
    turns = load_ground_truth()
    full_ref_text = "".join(t["norm_text"] for t in turns)
    print(f"Loaded {len(turns)} ground-truth turns, total {len(full_ref_text)} characters.")

    # 1. Evaluate what the 128 production commit WAVs from the session produced
    print("\n--- Evaluating Production Session Commits (Qwen 1.7B) ---")
    model_qwen = transcribe_cpp.Model(str(REPO / "backend_cpp" / "models" / "Qwen3-ASR-1.7B-Q8_0.gguf"))
    sess_qwen = model_qwen.session()
    
    commit_wavs = sorted(SESSION_DIR.glob("*_asr_commit.wav"), key=lambda p: int(p.name.split("_")[0]))
    print(f"Total commit WAVs found in session: {len(commit_wavs)}")
    
    session_hyps = []
    for f in commit_wavs:
        pcm, sr = sf.read(str(f), dtype="float32")
        if len(pcm) < 800:
            continue
        res = sess_qwen.run(pcm, language="ja")
        t = getattr(res, "text", "").strip()
        if t:
            session_hyps.append(t)
            
    sess_qwen.close()
    model_qwen.close()
    
    full_hyp_session = "".join(normalize_ja(h) for h in session_hyps)
    score_session = score_ja(full_ref_text, full_hyp_session)
    print(f"Session Commits Reconstructed Length: {len(full_hyp_session)} chars")
    print(f"Session Commits CER (strict): {score_session.cer*100:.2f}% | CER (ITN): {score_session.cer_itn*100:.2f}%")
    print(f"Substitutions: {score_session.substitutions}, Deletions: {score_session.deletions}, Insertions: {score_session.insertions}")

if __name__ == "__main__":
    main()
