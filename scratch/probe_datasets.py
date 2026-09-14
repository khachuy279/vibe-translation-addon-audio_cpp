"""Query HuggingFace for candidate Japanese ASR evaluation datasets."""
import json
import urllib.request

DATASETS = [
    "japanese-asr/ja_asr.reazonspeech_test",
    "japanese-asr/ja_asr.jsut_basic5000",
    "japanese-asr/ja_asr.common_voice_8_0",
    "joujiboi/japanese-anime-speech",
    "google/fleurs",
    "reazon-research/reazonspeech",
]


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "probe"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


for ds in DATASETS:
    try:
        info = api("https://huggingface.co/api/datasets/" + ds + "?full=true")
        sib = info.get("siblings") or []
        size = sum((s.get("size") or 0) for s in sib)
        print(f"{ds}")
        print(f"   downloads={info.get('downloads')} likes={info.get('likes')} gated={info.get('gated')}")
        print(f"   files={len(sib)} total~{size/1e9:.3f} GB")
        names = [s.get("rfilename") for s in sib][:12]
        print(f"   sample files: {names}")
    except Exception as e:
        print(f"{ds}: ERR {type(e).__name__}: {e}")
    print()
