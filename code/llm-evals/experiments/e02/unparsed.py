"""Share of Haiku replies that did not start with a letter, and that objected to the question
("I can't ..."), by Redux error type. Per-item means over 3 samples, bootstrap CIs over items."""
import json, sys, collections, re
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import items
data = items.load_redux(); et = np.array([d["error_type"] for d in data])
un = collections.defaultdict(list); obj = collections.defaultdict(list); ex = collections.defaultdict(list)
for l in open("runs/haiku-k3.jsonl"):
    r = json.loads(l); i = int(r["custom_id"][1:5]); t = r["text"]
    u = items.parse_letter(t) is None
    o = u and bool(re.match(r"\s*I (can't|cannot|can’t)", t))
    un[i].append(u); obj[i].append(o)
    if o: ex[et[i]].append(t)
U = np.array([np.mean(un[i]) for i in range(len(data))]); O = np.array([np.mean(obj[i]) for i in range(len(data))])
res = {}
for t in ["ok"] + items.FLAW_TYPES + ["flagged"]:
    m = np.where(et != "ok")[0] if t == "flagged" else np.where(et == t)[0]
    res[t] = dict(n=len(m), unparsed=items.boot_ci(lambda s: U[m[s]].mean(), len(m), 5000),
                  objects=items.boot_ci(lambda s: O[m[s]].mean(), len(m), 5000))
    print(t, len(m), [round(x, 4) for x in res[t]["unparsed"]], [round(x, 4) for x in res[t]["objects"]])
json.dump({k: {"n": v["n"], "unparsed": [float(x) for x in v["unparsed"]], "objects": [float(x) for x in v["objects"]]} for k, v in res.items()},
          open("results/unparsed.json", "w"), indent=1)
for t in ex: print(t, ex[t][:3])
