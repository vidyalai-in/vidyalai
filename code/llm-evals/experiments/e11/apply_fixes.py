import json, sys
src, fixfiles, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
fix, drop = {}, {}
for ff in fixfiles:
    d = json.load(open(ff))
    for k, v in d.items():
        if v.get("drop"): drop[k] = v["note"]
        else: fix.setdefault(k, {}).update(v)
n = 0
with open(out, "w") as f:
    for l in open(src):
        it = json.loads(l)
        if it["id"] in drop: continue
        for k, v in fix.get(it["id"], {}).items():
            if k != "note": it[k] = v
        f.write(json.dumps(it) + "\n"); n += 1
print(n, "items written;", len(drop), "dropped;", len(fix), "touched")
