"""Sonnet 5.5 on the items an HLE-style filter against Haiku keeps (Haiku wrong on its first answer),
for the strata Sonnet was fully scored on: clean (ok) items from its random 370, and wrong-key items."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import items
R = "/home/amit/.cache/vidyalai-scratch/e02"
data = items.load_redux(); gold = np.array([d["answer"] for d in data]); et = np.array([d["error_type"] for d in data])
H = np.load(f"{R}/results/picks-haiku-k3.npy")[:, 1:]; F = H[:, 0] != gold
fix = {}
for l in open(f"{R}/runs/sonnet-k1-reask.jsonl"):
    r = json.loads(l); fix[r["custom_id"]] = json.loads(r["text"])["answer"]
S = {}
for l in open(f"{R}/runs/sonnet-k1.jsonl"):
    r = json.loads(l); L = items.parse_letter(r["text"]) or fix.get(r["custom_id"]); S[int(r["custom_id"][1:5])] = L
out = {}
for t in ["ok", "wrong_groundtruth"]:
    ix = [i for i in S if et[i] == t and F[i]]
    c = np.array([S[i] == items.LETTERS[gold[i]] for i in ix], float)
    out[t] = dict(n=len(ix), sonnet_acc=[round(float(x), 4) for x in items.boot_ci(lambda s: c[s].mean(), len(c), 10000)],
                  haiku_other2=[round(float(x), 4) for x in items.boot_ci(lambda s: (H[ix][:, 1:] == gold[ix][:, None]).mean(1)[s].mean(), len(ix), 10000)])
print(out); json.dump(out, open(f"{R}/results/sonnet-filter.json", "w"), indent=1)
