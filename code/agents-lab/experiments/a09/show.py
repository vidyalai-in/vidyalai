import json, sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    if d["event"] == "step":
        for b in d["content"]:
            if b["type"] == "thinking": print(f"  [think] {b.get('thinking','')[:300]!r}")
            elif b["type"] == "text": print(f"  [text] {b['text'][:400]!r}")
            elif b["type"] == "tool_use": print(f"{d['n']:2d} CALL {b['name']} {json.dumps(b['input'])}")
    elif d["event"] == "tool_results":
        for r in d["results"]:
            c = r.get("content")
            if isinstance(c, list):
                s = " ".join(x.get("text", f"<{x['type']}>")[: int(sys.argv[2]) if len(sys.argv) > 2 else 200] for x in c)
            else: s = str(c)[:300]
            print(f"     -> {'ERR ' if r.get('is_error') else ''}{s!r}")
