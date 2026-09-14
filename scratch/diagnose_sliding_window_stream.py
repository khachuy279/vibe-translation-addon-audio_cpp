import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import soundfile as sf
import transcribe_cpp
from backend_cpp.asr.model_manager import ASRModelManager
from benchmarks.ingress_benchmark import DEFAULT_WAV, DEFAULT_TXT, parse_ground_truth
from benchmarks.ja_text import normalize_ja, score_ja
from benchmarks.sliding_window_benchmark import run_sliding_window

audio, sr = sf.read(str(DEFAULT_WAV), dtype='float32')
if audio.ndim > 1:
    audio = audio.mean(axis=1)

turns = parse_ground_truth(DEFAULT_TXT)
ref_text = ''.join(t['norm_text'] for t in turns)

mgr = ASRModelManager()
model = mgr.ensure_model('qwen3-asr-1.7b')
session = transcribe_cpp.Session(model)

m = run_sliding_window(
    session=session,
    audio=audio,
    sr=sr,
    ref_text=ref_text,
    window_sec=10.0,
    step_sec=1.5,
    agreement_steps=2,
    min_agreement_chars=2,
)

print(f"CER: {m.cer*100:.2f}% | ITN: {m.cer_itn*100:.2f}% | Commits: {m.num_commits}")
print("Total ref chars:", len(ref_text), "Total hyp chars:", len(m.hyp_text))

# Save comparison to examine
out_file = REPO / "scratch" / "diag_hyp_vs_ref.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(f"REF:\n{ref_text}\n\nHYP:\n{m.hyp_text}\n\nCOMMITS:\n")
    for c in m.commits:
        f.write(f"[{c['stream_time_sec']:5.1f}s] {c['text']}\n")

print(f"Saved diagnostics to {out_file}")
