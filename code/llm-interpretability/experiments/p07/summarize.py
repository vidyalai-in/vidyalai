"""Small, committable summaries: Gemma eval CIs, interpretability ratings with Wilson CI, firing-frequency histograms."""
import json, math
import numpy as np, torch
R = "/home/amit/.cache/vidyalai-scratch/p07"
E = torch.load(f"{R}/runs/gemma_explore.pt")
ev = E["eval"]
a, b = np.array(ev["ce_clean_per_doc"]), np.array(ev["ce_sae_per_doc"])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(a), (5000, len(a)))
d = (b - a)
gem = {"docs": len(a), "tokens_scored": ev["eval_tokens_scored"], "fve": ev["fve"], "l0": ev["l0"],
       "ce_clean": [a.mean(), *np.percentile(a[idx].mean(1), [2.5, 97.5])],
       "ce_sae": [b.mean(), *np.percentile(b[idx].mean(1), [2.5, 97.5])],
       "delta_ce": [d.mean(), *np.percentile(d[idx].mean(1), [2.5, 97.5])],
       "l0_doc_ci": list(np.percentile(np.array(ev["l0_per_doc"])[idx].mean(1), [2.5, 97.5])),
       "explore_tokens": E["n_tokens"], "dead_in_explore": int((E["count"] == 0).sum()),
       "fire_lt_50": int((E["count"] < 50).sum()), "dense_gt_10pct": int((E["count"] / E["n_tokens"] > 0.1).sum())}
rat = json.load(open(f"{R}/notes/ratings.json"))
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [p, c - h, c + h]
T = [r["top"] for r in rat["ratings"]]; RF = [r["rand_fit"] for r in rat["ratings"]]
rs = {"n": len(T), "clear": wilson(sum(t == 2 for t in T), len(T)), "partial": wilson(sum(t == 1 for t in T), len(T)),
      "none": wilson(sum(t == 0 for t in T), len(T)), "rand_fit_ge3": wilson(sum(r >= 3 for r in RF), len(T)),
      "rand_fit_ge2": wilson(sum(r >= 2 for r in RF), len(T))}
X = torch.load(f"{R}/runs/gpt2_eval_extra.pt"); n_tok = 537337
bins = np.arange(-6.5, 0.01, 0.25)
hist = {}
for k in ["topk32_24k", "relu_l1_15_24k", "jb_24k", "oai_topk32_32k", "topk32_3k"]:
    c = X[k]["counts"].numpy()
    f = np.log10(np.maximum(c, 0.5) / n_tok)
    hist[k] = {"dead": int((c == 0).sum()), "hist": np.histogram(f[c > 0], bins)[0].tolist(), "median_log10_freq": float(np.median(f[c > 0]))}
gem["fire_hist"] = np.histogram(np.log10(np.maximum(E["count"].numpy(), 0.5)[E["count"].numpy() > 0] / E["n_tokens"]), bins)[0].tolist()
out = {"gemma": gem, "ratings": rs, "gpt2_fire_hist": {"bins": bins.tolist(), **hist}}
json.dump(out, open(f"{R}/runs/summary.json", "w"), indent=1, default=float)
print(json.dumps({"gemma": {k: v for k, v in gem.items() if k != "fire_hist"}, "ratings": rs,
                  "gpt2": {k: (v["dead"], v["median_log10_freq"]) for k, v in hist.items()}}, default=float, indent=0))
