"""Sample cited claims, fetch each cited page now, and record what a reader would find there.
For each sample: HTTP status, whether the cited_text occurs on the page (whitespace and case
normalized, first 80 characters), and a text window around the match for the hand check."""
import hashlib, html, json, random, re, urllib.request
from pathlib import Path
PAGES = Path("/home/amit/.cache/vidyalai-scratch/a10/pages"); PAGES.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
cites = [json.loads(l) for l in open("cites.jsonl")]
random.seed(10)
quota = {("one-search", "claude-haiku-5-5"): 15, ("research", "claude-haiku-5-5"): 25,
         ("research", "claude-sonnet-5-5"): 12, ("research-df", "claude-haiku-5-5"): 8}
sample = []
for key, n in quota.items():
    pool = [c for c in cites if (c["arm"], c["model"]) == key]
    sample += random.sample(pool, min(n, len(pool)))

def norm(s):
    s = re.sub(r"[#*_`>\[\]]|\(http[^)]*\)", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip().lower()

def page_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        status, raw = r.status, r.read(5_000_000)
    ctype = r.headers.get("Content-Type", "")
    raw = raw.decode("utf-8", "ignore")
    (PAGES / (hashlib.md5(url.encode()).hexdigest()[:10] + ".html")).write_text(raw)
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
    return status, ctype, norm(re.sub(r"<[^>]+>", " ", t))

cache, out = {}, []
for s in sample:
    url = s["url"]
    if url not in cache:
        try:
            cache[url] = page_text(url)
        except Exception as e:
            cache[url] = (getattr(e, "code", None) or type(e).__name__, "", "")
    status, ctype, text = cache[url]
    ct = norm(s["cited_text"])
    probe = ct[:80].rstrip(". ")
    i = text.find(probe) if probe else -1
    if i < 0:  # try the longest of the cited text's sentences
        parts = sorted(re.split(r"(?<=[.!?])\s", ct), key=len, reverse=True)
        for p in parts[:2]:
            if len(p) > 25 and (j := text.find(p[:60])) >= 0:
                i = j; break
    out.append({**s, "status": status, "page_chars": len(text), "cited_found": i >= 0,
                "window": text[max(0, i - 300): i + 700] if i >= 0 else ""})
Path("audit.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n")
print(len(out), "sampled;", sum(o["cited_found"] for o in out), "cited_text found on the live page;",
      sum(not str(o["status"]).startswith("2") for o in out), "not 2xx")
