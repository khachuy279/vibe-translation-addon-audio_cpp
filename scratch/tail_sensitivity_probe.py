"""How sensitive is the model to the exact extent of the audio it is given?

The pipeline drops trailing audio beyond a grace window (measured: 144 ms and 504 ms on two
utterances) and drops interior silence runs, which introduced a mid-utterance discontinuity
on another. Before blaming the model's quality, check whether it simply reacts to the audio
extent.

For each utterance, decode:
    full file
    file minus 150 / 300 / 500 ms of trailing audio
    file plus 500 ms of digital silence

If truncation alone changes the transcript, then "keep a generous tail" is a real accuracy
requirement and not a cosmetic one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import transcribe_cpp  # noqa: E402
from backend_cpp.asr.model_manager import ASRModelManager  # noqa: E402
from backend_cpp.config import config  # noqa: E402
from benchmarks.ja_text import score_ja  # noqa: E402

CV = REPO / "data" / "ja_cv"
MODEL = REPO / "backend_cpp" / "models" / "Qwen3-ASR-1.7B-Q8_0.gguf"
N = 10


def corpus_cer(pairs) -> float:
    edits = ref_len = 0
    for ref, hyp in pairs:
        s = score_ja(ref, hyp)
        edits += s.substitutions + s.deletions + s.insertions
        ref_len += s.ref_len
    return 100.0 * edits / max(1, ref_len)


def main() -> None:
    rows = [json.loads(l) for l in open(CV / "utts.jsonl", encoding="utf-8") if l.strip()]
    items = [r for r in rows if 1.2 <= r["duration_sec"] <= 8.0][:N]
    ASRModelManager().ensure_model(config.asr.active_model)

    model = transcribe_cpp.Model(str(MODEL))
    session = model.session()
    session.run(np.zeros(16000, dtype=np.float32), language="ja")

    variants = {
        "full file": lambda p: p,
        "minus 150ms tail": lambda p: p[:-2400],
        "minus 300ms tail": lambda p: p[:-4800],
        "minus 500ms tail": lambda p: p[:-8000],
        "plus 500ms silence": lambda p: np.concatenate([p, np.zeros(8000, dtype=np.float32)]),
        "plus 1000ms silence": lambda p: np.concatenate([p, np.zeros(16000, dtype=np.float32)]),
    }

    refs, pcms = [], []
    for r in items:
        pcm, _sr = sf.read(str(CV / r["wav"]), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        refs.append(r["text"])
        pcms.append(pcm.astype(np.float32))

    print(f"{len(items)} utterances, corpus CER (character-weighted)\n")
    print(f"{'variant':<22}{'CER':>9}{'vs full':>10}   changed transcripts")
    base = None
    results = {}
    for label, fn in variants.items():
        hyps = []
        for pcm in pcms:
            aud = fn(pcm)
            if len(aud) < 1600:
                hyps.append("")
                continue
            hyps.append((session.run(aud, language="ja").text or "").strip())
        cer = corpus_cer(list(zip(refs, hyps)))
        results[label] = hyps
        if base is None:
            base = cer
        changed = sum(1 for a, b in zip(results["full file"], hyps) if a != b)
        print(f"{label:<22}{cer:>8.2f}%{cer - base:>+9.2f}pp{changed:>6}/{len(items)}")

    print(f"\n{'='*100}\nPER-UTTERANCE (only rows where truncation changed the output)\n{'='*100}")
    for i, r in enumerate(items):
        full = results["full file"][i]
        diffs = [lab for lab in variants if results[lab][i] != full]
        if not diffs:
            continue
        print(f"\n{r['id']}  ({r['duration_sec']:.2f}s)   changed by: {', '.join(diffs)}")
        print(f"   REF            : {r['text']}")
        print(f"   full file      : {full}")
        for lab in diffs:
            print(f"   {lab:<15}: {results[lab][i]}")

    session.close()
    model.close()


if __name__ == "__main__":
    main()
