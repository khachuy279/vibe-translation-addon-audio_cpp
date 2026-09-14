"""Download the ReazonSpeech Japanese ASR test set and report its schema.

Source: https://huggingface.co/datasets/japanese-asr/ja_asr.reazonspeech_test
(ReazonSpeech = natural Japanese speech from TV broadcasts; matches the
"TV drama / natural dialogue" target domain.)

    python scratch/fetch_ja_eval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ja_eval"
RAW = DATA / "raw"
CACHE = DATA / ".hf_cache"

os.environ.setdefault("HF_HOME", str(CACHE))
# The DSH file sandbox forbids the symlink/temp-touch dance huggingface_hub does when it
# builds a content-addressed cache. Download straight into a plain directory instead:
# `local_dir` writes real files and never creates symlinks.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ID = "japanese-asr/ja_asr.reazonspeech_test"
FILES = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]


def main() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in FILES:
        p = hf_hub_download(
            repo_id=REPO_ID,
            filename=f,
            repo_type="dataset",
            local_dir=str(RAW),
        )
        paths.append(p)
        print(f"downloaded {f} -> {p} ({Path(p).stat().st_size/1e6:.1f} MB)")

    total_rows = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        print(f"\n=== {Path(p).name} ===")
        print("rows:", pf.metadata.num_rows)
        print("schema:")
        print(pf.schema_arrow)
        total_rows += pf.metadata.num_rows
        tbl = pf.read_row_group(0)
        print("first row sample (truncated):")
        d = tbl.slice(0, 1).to_pylist()[0]
        for k, v in d.items():
            s = repr(v)
            print(f"   {k}: {s[:200]}")
    print(f"\nTOTAL rows: {total_rows}")


if __name__ == "__main__":
    main()
