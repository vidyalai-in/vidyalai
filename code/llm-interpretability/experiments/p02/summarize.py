"""Turn the per-document eval logs into a small summary with 95% bootstrap CIs over documents."""
import json, math, sys, os
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
import torch
from lab.lens import bootstrap_ci

R = "/home/amit/.cache/vidyalai-scratch/p02/runs"
P = "/home/amit/.cache/vidyalai-scratch/p02/post"
LN2 = math.log(2)
out = {}
for key in ["gpt2", "pythia-160m", "qwen2.5-0.5b"]:
    f = f"{R}/{key}-eval.json"
    if not os.path.exists(f):
        continue
    e = json.load(open(f))
    s = {"n_docs": e["n_docs"], "positions_per_doc": e["positions_per_doc"], "eval_sec": e["sec"]}
    s["final_ce_nats"] = bootstrap_ci(e["final_ce"])
    lens_names = [k for k in e if isinstance(e[k], dict)]
    for n in lens_names:
        kl = torch.tensor(e[n]["kl"]) / LN2          # bits
        t1 = torch.tensor(e[n]["top1"]) * 100
        ce = torch.tensor(e[n]["ce"])
        s[n] = {"kl_bits": [bootstrap_ci(kl[:, l]) for l in range(kl.shape[1])],
                "top1_pct": [bootstrap_ci(t1[:, l]) for l in range(t1.shape[1])],
                "ce_nats": [bootstrap_ci(ce[:, l]) for l in range(ce.shape[1])]}
        # paired difference vs the logit lens, per point
        if n != "logit":
            kl0 = torch.tensor(e["logit"]["kl"]) / LN2
            t10 = torch.tensor(e["logit"]["top1"]) * 100
            s[n]["kl_bits_minus_logit"] = [bootstrap_ci(kl[:, l] - kl0[:, l]) for l in range(kl.shape[1])]
            s[n]["top1_minus_logit"] = [bootstrap_ci(t1[:, l] - t10[:, l]) for l in range(t1.shape[1])]
    nm = torch.tensor(e["norm"]); n0 = torch.tensor(e["norm_pos0"])
    s["norm"] = [bootstrap_ci(nm[:, l]) for l in range(nm.shape[1])]
    s["norm_pos0"] = [bootstrap_ci(n0[:, l]) for l in range(n0.shape[1])]
    out[key] = s
for f in sorted(os.listdir(R)):
    if f.endswith(".json") and ("-train-" in f or "sweeps" in f):
        d = json.load(open(f"{R}/{f}"))
        out.setdefault("training", {})[f[:-5]] = d
for f in sorted(os.listdir(R)):
    if f.endswith("-examples.json"):
        out.setdefault("examples", {})[f[:-5]] = json.load(open(f"{R}/{f}"))
os.makedirs(P, exist_ok=True)
json.dump(out, open(f"{P}/summary.json", "w"), indent=1)
print("wrote", f"{P}/summary.json", os.path.getsize(f"{P}/summary.json"), "bytes")
