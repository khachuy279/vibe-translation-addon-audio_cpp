"""Look up GitHub repositories tagged dsh-plugin."""
import json
import urllib.parse
import urllib.request


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "probe", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


QUERIES = [
    "topic:dsh-plugin",
    "dsh-plugin in:name",
    "dsh plugin in:description",
    "topic:dsh",
]

for q in QUERIES:
    try:
        d = api("https://api.github.com/search/repositories?q=" + urllib.parse.quote(q) + "&per_page=30")
        print(f"=== {q}  (total_count={d.get('total_count')}) ===")
        for it in d.get("items", []):
            print(f"  {it['full_name']:<52} stars={it['stargazers_count']:<5} lang={str(it.get('language')):<12} updated={it['updated_at'][:10]}")
            print(f"      {(it.get('description') or '')[:120]}")
        if not d.get("items"):
            print("  (no results)")
    except Exception as e:
        print(f"{q}: ERR {type(e).__name__}: {e}")
    print()
