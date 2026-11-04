import json, sys, collections, glob
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import longctx as lc
rows = []
for p in sys.argv[1:]:
    rows += [json.loads(l) for l in open(p)]
agg = collections.defaultdict(list); hed = collections.defaultdict(int); cut = collections.defaultdict(int)
for r in rows:
    if r.get("task") in (None, "know") or r.get("event"):
        continue
    if r["task"] == "vt":
        c, rec = lc.grade_vt(r["reply"], r["gold"])
    else:
        c = lc.grade_name(r["reply"], r["gold"]); hed[(r["task"], r["L"])] += lc.hedged(r["reply"])
    agg[(r["task"], r["L"])].append(c)
    cut[(r["task"], r["L"])] += r.get("stop") == "max_tokens"
for k in sorted(agg):
    print(k, "%d/%d" % (sum(agg[k]), len(agg[k])), "hedged", hed[k], "cut", cut[k])
