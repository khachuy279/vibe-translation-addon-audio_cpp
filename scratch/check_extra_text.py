"""Check specific utterances across all models: is the extra text in the audio, or invented?"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
data = json.loads((REPO / "report" / "ja_offline_bench.json").read_text(encoding="utf-8"))
by_model = {r["model"]: {u["id"]: u for u in r["per_utt"]} for r in data["results"] if "error" not in r}

IDS = ["rz00004", "rz00005", "rz00006", "rz00071", "rz00075", "rz00035", "rz00090", "rz00161", "rz00080", "rz00198"]
for uid in IDS:
    ref = next(iter(by_model.values()))[uid]["ref"]
    print(f"\n{'='*100}\n[{uid}] REF: {ref}")
    for m, utts in by_model.items():
        u = utts.get(uid)
        if not u:
            continue
        print(f"  {m:<24} CER={u['cer']*100:6.1f}%  I={u['insertions']:>3}  {u['hyp']}")
