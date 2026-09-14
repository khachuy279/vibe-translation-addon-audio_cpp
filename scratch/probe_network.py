"""Probe outbound network access for model/audio downloads."""
import json
import socket
import ssl
import urllib.request

TARGETS = [
    "https://huggingface.co/api/models?limit=1",
    "https://huggingface.co/datasets?limit=1",
    "https://raw.githubusercontent.com/huggingface/datasets/main/README.md",
]

ctx = ssl.create_default_context()
for url in TARGETS:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "probe"})
        with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
            body = r.read(200)
        print(f"OK   {r.status} {url} ({len(body)}B)")
    except Exception as e:
        print(f"FAIL {url} -> {type(e).__name__}: {e}")

try:
    print("DNS huggingface.co ->", socket.gethostbyname("huggingface.co"))
except Exception as e:
    print("DNS FAIL:", e)
