import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if '"event"' not in l]
know = [r for r in rows if r["task"] == "know"]
print("know", sum(r["correct"] for r in know), "/", len(know))
agg = collections.defaultdict(list)
for r in rows:
    if r["task"] != "know":
        agg[(r["task"], r["L"])].append(r["correct"])
for k in sorted(agg):
    print(k, "%d/%d" % (sum(agg[k]), len(agg[k])))
