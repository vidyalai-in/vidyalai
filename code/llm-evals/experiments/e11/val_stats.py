import sys, json, numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import own_eval as oe, stats
items = {json.loads(l)["id"]: json.loads(l) for l in open("data/candidates_fixed.jsonl")}
X = {}
for l in open("runs/val.jsonl"):
    r = json.loads(l); rep, iid = r["id"].split("_", 1)
    X.setdefault(iid, {})[rep] = oe.grade(items[iid], oe.final_answer(r.get("text"))[0])
ids = sorted(items); a = np.array([X[i]["r1"] for i in ids]); b = np.array([X[i]["r2"] for i in ids])
print("acc r1 %.3f r2 %.3f" % (a.mean(), b.mean()), "disagree", int((a != b).sum()))
# planning: assume configs disagree on 5% of items (no item-level data yet) -> MDE at n=195
for disc in (0.02, 0.05, 0.10):
    print("disc", disc, "MDE single run, n=195: %.1f points" % (100 * stats.mde(195, omega2=disc)), "n for 3 pts: %.0f" % stats.sample_size(0.03, disc))
