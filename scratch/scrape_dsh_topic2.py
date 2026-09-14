"""Extract the actual repository cards from a GitHub topic page.

The topic page renders repos as cards inside an <article>; a plain "any /owner/repo link"
regex also picks up nav and sponsored links, which is why the first pass returned unrelated
projects such as PicGo and nocobase.
"""
import re
import urllib.request

BLOCKED = {
    "topics", "collections", "login", "signup", "features", "about", "pricing", "security",
    "enterprise", "marketplace", "sponsors", "settings", "orgs", "trending", "events",
    "readme", "explore", "apps", "contact", "site", "github", "deepseek-ai",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 probe"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


for topic in ("dsh-plugin", "dsh"):
    url = f"https://github.com/topics/{topic}"
    html = fetch(url)

    # Repo cards: <h3 ...><a href="/owner">owner</a> / <a href="/owner/repo" ...>repo</a>
    cards = re.findall(
        r'<h3[^>]*>\s*<a[^>]+href="/([^"/]+)"[^>]*>[^<]*</a>\s*/\s*<a[^>]+href="/([^"]+)"',
        html,
    )
    repos = []
    for owner, rest in cards:
        if owner.lower() in BLOCKED:
            continue
        full = rest if "/" in rest else f"{owner}/{rest}"
        if full.count("/") == 1 and full not in repos:
            repos.append(full)

    print(f"=== https://github.com/topics/{topic} : {len(repos)} repo cards ===")
    for r in repos:
        print(f"   https://github.com/{r}")
    if not repos:
        print("   (no repo cards matched; page may not use the expected markup)")
    print()
