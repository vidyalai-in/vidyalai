"""Random Wikipedia lead sections (plain text), 20 per request, via the MediaWiki API. CC BY-SA 4.0.
Usage: get_wiki_leads.py <lang> <target_chars> <min_chars>"""
import json, sys, time, urllib.request, urllib.parse
lang, target, min_chars = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = f"/home/amit/.cache/vidyalai-scratch/interp/data/p06/wiki-{lang}-leads.jsonl"
UA = {"User-Agent": "vidyalai-interp-research/0.1 (educational, low volume)"}
seen, chars, req = set(), 0, 0
with open(out, "w") as f:
    while chars < target and req < 400:
        q = dict(action="query", generator="random", grnnamespace=0, grnlimit=20, prop="extracts",
                 exintro=1, explaintext=1, exlimit=20, format="json")
        url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q)
        req += 1
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        except Exception as e:
            print("err", e); time.sleep(5); continue
        for p in d["query"]["pages"].values():
            t = p.get("extract", "")
            if p["pageid"] in seen or len(t) < min_chars:
                continue
            seen.add(p["pageid"]); chars += len(t)
            f.write(json.dumps({"pageid": p["pageid"], "title": p["title"], "text": t}, ensure_ascii=False) + "\n")
        time.sleep(0.5)
print(lang, len(seen), "articles", chars, "chars", req, "requests ->", out)
