"""Report real file sizes for candidate HF Japanese datasets (tree API)."""
import json
import urllib.request

DATASETS = [
    "japanese-asr/ja_asr.reazonspeech_test",
    "japanese-asr/ja_asr.jsut_basic5000",
    "japanese-asr/ja_asr.common_voice_8_0",
    "joujiboi/japanese-anime-speech",
]


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "probe"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


for ds in DATASETS:
    try:
        tree = api(f"https://huggingface.co/api/datasets/{ds}/tree/main/data?recursive=true")
        total = sum(t.get("size") or 0 for t in tree if t.get("type") == "file")
        print(f"{ds}: {len(tree)} files, {total/1e6:.1f} MB")
        for t in tree[:6]:
            print(f"    {t.get('path')}  {(t.get('size') or 0)/1e6:.1f} MB")
    except Exception as e:
        print(f"{ds}: ERR {type(e).__name__}: {e}")
