"""Validate the qwen3-asr-1.7b Japanese failure: leakage in my harness, or the model?

The offline bench reuses one Session across utterances. If the native session leaked
context between calls, the "inserted" text would be an artifact of my harness rather
than a production-relevant model behaviour. This isolates the two:

  A. shared session, sequential          (what the bench does)
  B. fresh Model+Session per utterance   (no possible leakage)
  C. shared session, reverse order       (order sensitivity)
  D. same utterance 3x in a row          (immediate repeat determinism)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transcribe_cpp  # noqa: E402
from benchmarks.ja_text import score_ja  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ja_eval"
MODEL = REPO / "backend_cpp" / "models" / "Qwen3-ASR-1.7B-Q8_0.gguf"
N = 30


def load(path: Path) -> np.ndarray:
    d, sr = sf.read(str(path), dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    return d.astype(np.float32)


def corpus_cer(pairs):
    num = sum((score_ja(r, h).cer or 0.0) * len(r) for r, h in pairs)
    den = sum(len(r) for r, h in pairs)
    return 100.0 * num / den if den else 0.0


def main() -> None:
    rows = [json.loads(l) for l in open(DATA / "utts.jsonl", encoding="utf-8") if l.strip()][:N]
    items = [(r, load(DATA / r["wav"])) for r in rows]

    model = transcribe_cpp.Model(str(MODEL))
    session = model.session()
    session.run(items[0][1], language="ja")  # warm up

    # ---- A: shared session, sequential ----
    a = [(r["text"], session.run(p, language="ja").text.strip()) for r, p in items]
    print(f"A. shared session, sequential      CER={corpus_cer(a):6.2f}%")

    # ---- C: shared session, reverse ----
    c = [(r["text"], session.run(p, language="ja").text.strip()) for r, p in reversed(items)]
    print(f"C. shared session, reverse order   CER={corpus_cer(c):6.2f}%")

    # ---- D: same utterance 3x ----
    r0, p0 = items[0]
    reps = [session.run(p0, language="ja").text.strip() for _ in range(3)]
    print(f"D. same utt x3 identical: {len(set(reps)) == 1}")
    for t in reps:
        print(f"     {t}")

    session.close()
    model.close()

    # ---- B: fresh model+session per utterance (slow, so fewer) ----
    n_fresh = 12
    b = []
    for r, p in items[:n_fresh]:
        m2 = transcribe_cpp.Model(str(MODEL))
        s2 = m2.session()
        hyp = s2.run(p, language="ja").text.strip()
        s2.close()
        m2.close()
        b.append((r["text"], hyp))
    print(f"\nB. fresh model per utt (n={n_fresh})    CER={corpus_cer(b):6.2f}%")
    print(f"A. same {n_fresh} utts, shared session  CER={corpus_cer(a[:n_fresh]):6.2f}%")

    print("\n--- per-utterance A vs B ---")
    for (ref, ha), (_r, hb) in zip(a[:n_fresh], b):
        same = "SAME" if ha == hb else "DIFF"
        print(f"  [{same}] REF : {ref}")
        print(f"          A   : {ha}")
        if ha != hb:
            print(f"          B   : {hb}")

    # ---- E: audio delimited by silence padding (does padding stop continuation?) ----
    pad = np.zeros(int(16000 * 0.3), dtype=np.float32)
    m3 = transcribe_cpp.Model(str(MODEL))
    s3 = m3.session()
    e = []
    for r, p in items[:n_fresh]:
        padded = np.concatenate([pad, p, pad])
        e.append((r["text"], s3.run(padded, language="ja").text.strip()))
    s3.close()
    m3.close()
    print(f"\nE. 300ms silence padding both ends CER={corpus_cer(e):6.2f}%")
    for (ref, _ha), (_r, he) in zip(a[:n_fresh], e):
        print(f"  REF : {ref}")
        print(f"  PAD : {he}")


if __name__ == "__main__":
    main()
