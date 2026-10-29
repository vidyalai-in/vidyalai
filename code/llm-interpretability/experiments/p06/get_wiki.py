"""Download random Wikipedia articles (plain-text extracts) for one language via the MediaWiki API.
License of the text: CC BY-SA 4.0 (Wikipedia). Usage: get_wiki.py <lang> <n_articles> <min_chars>"""
import json, sys, time, urllib.request, urllib.parse
lang, n, min_chars = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = f"/home/amit/.cache/vidyalai-scratch/interp/data/p06/wiki-{lang}.jsonl"
UA = {"User-Agent": "vidyalai-interp-research/0.1 (educational, low volume)"}
seen, got = set(), 0
with open(out, "w") as f:
    while got < n:
        q = dict(action="query", generator="random", grnnamespace=0, grnlimit=1, prop="extracts",
                 explaintext=1, exlimit=1, format="json")
        url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q)
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        except Exception as e:
            print("err", e); time.sleep(5); continue
        for p in d["query"]["pages"].values():
            t = p.get("extract", "")
            if p["pageid"] in seen or len(t) < min_chars:
                continue
            seen.add(p["pageid"]); got += 1
            f.write(json.dumps({"pageid": p["pageid"], "title": p["title"], "text": t}, ensure_ascii=False) + "\n")
        time.sleep(0.3)
print(lang, got, "articles ->", out)
