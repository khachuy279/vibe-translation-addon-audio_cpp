"""Scrape the GitHub topic page for dsh-plugin repository links."""
import re
import urllib.request

URLS = [
    "https://github.com/topics/dsh-plugin",
    "https://github.com/topics/dsh",
]

for url in URLS:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 probe"})
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    except Exception as e:
        print(f"{url}: ERR {type(e).__name__}: {e}")
        continue

    # Repository cards on a topic page look like: href="/owner/repo" ... itemprop="name codeRepository"
    repos = re.findall(r'href="/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"[^>]*>\s*<span[^>]*itemprop="name codeRepository"', html)
    if not repos:
        repos = re.findall(r'itemprop="name codeRepository"[^>]*>\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)', html)
    if not repos:
        # Fallback: any /owner/repo link inside the topic listing region
        cand = re.findall(r'href="/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"', html)
        blocked = {"topics", "collections", "login", "signup", "features", "about", "pricing",
                   "security", "enterprise", "marketplace", "sponsors", "settings", "orgs",
                   "trending", "events", "readme", "explore", "apps", "contact", "site"}
        repos = [
            c for c in dict.fromkeys(cand)
            if c.split("/")[0].lower() not in blocked
            and not c.startswith(("topics/", "collections/", "sponsors/"))
        ]

    print(f"=== {url} (page length {len(html)}) ===")
    if repos:
        for r in repos[:40]:
            print(f"   https://github.com/{r}")
    else:
        print("   (no repo links found)")
        print("   contains 'dsh-plugin'?", "dsh-plugin" in html)
        low = html.lower()
        idx = low.find("dsh-plugin")
        if idx > 0:
            print("   context:", re.sub(r"\s+", " ", html[max(0, idx - 200): idx + 200]))
    print()
