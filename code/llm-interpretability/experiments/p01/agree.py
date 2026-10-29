"""Agreement between direct logit attribution and ablation, with bootstrap CIs over prompts.
Usage: python agree.py <tag>   (reads runs/<tag>.npz and post/<tag>.json, writes post/<tag>_agree.json)"""
import json, sys
import numpy as np
from scipy.stats import spearmanr

ROOT = "/home/amit/.cache/vidyalai-scratch/p01"
tag = sys.argv[1]
a = np.load(f"{ROOT}/runs/{tag}.npz")
r = json.load(open(f"{ROOT}/post/{tag}.json"))
labs = [l for l in r["labels"] if l.startswith(("attn", "mlp"))]
li = [r["labels"].index(l) for l in labs]
D = a["dla"][li]                                         # [24, n]
M = np.stack([a[f"d_mean_{l}"] for l in labs])           # mean-ablation effect
Z = np.stack([a[f"d_zero_{l}"] for l in labs])
DIR = np.stack([a[f"direct_mean_{l}"] for l in labs])
n = D.shape[1]
rng = np.random.default_rng(0)

def stats(idx):
    d, m, z, dr = D[:, idx].mean(1), M[:, idx].mean(1), Z[:, idx].mean(1), DIR[:, idx].mean(1)
    rho_m = spearmanr(d, -m)[0]
    rho_z = spearmanr(d, -z)[0]
    rho_zm = spearmanr(z, m)[0]
    top = lambda v, k=5: set(np.argsort(-v)[:k])
    overlap = len(top(d) & top(-m))
    # share of the total absolute mean-ablation effect that the direct path accounts for
    direct_share = np.abs(dr).sum() / np.abs(m).sum()
    return rho_m, rho_z, rho_zm, overlap, direct_share

point = stats(np.arange(n))
boots = np.array([stats(rng.integers(0, n, n)) for _ in range(2000)])
names = ["spearman_dla_vs_mean_ablation", "spearman_dla_vs_zero_ablation", "spearman_zero_vs_mean",
         "top5_overlap", "direct_share_of_abs_mean_effect"]
out = {k: [float(point[i]), float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]
       for i, k in enumerate(names)}
d, m, z = D.mean(1), M.mean(1), Z.mean(1)
out["rank_dla"] = [labs[i] for i in np.argsort(-d)[:6]]
out["rank_mean_ablation"] = [labs[i] for i in np.argsort(m)[:6]]
out["rank_zero_ablation"] = [labs[i] for i in np.argsort(z)[:6]]
# per-prompt sign agreement for the components the two methods disagree on most
out["per_prompt"] = {}
for l in ("mlp0", "attn0", "mlp9", "mlp10", "mlp11", "attn9"):
    if l in labs:
        i = labs.index(l)
        out["per_prompt"][l] = {"frac_mean_ablation_hurts": float((M[i] < 0).mean()),
                                "frac_dla_positive": float((D[i] > 0).mean())}
if "d_mean_heads" in a.files:
    dh, mh = a["dla_heads"].mean(-1).ravel(), a["d_mean_heads"].mean(-1).ravel()
    def hstats(idx):
        return spearmanr(a["dla_heads"][:, :, idx].mean(-1).ravel(), -a["d_mean_heads"][:, :, idx].mean(-1).ravel())[0]
    hb = np.array([hstats(rng.integers(0, n, n)) for _ in range(500)])
    out["heads_spearman"] = [float(hstats(np.arange(n))), float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))]
    nh = a["dla_heads"].shape[1]
    big_abl = np.argsort(mh)[:10]
    out["heads_top_ablation"] = [[int(i // nh), int(i % nh), round(float(dh[i]), 3), round(float(mh[i]), 3)] for i in big_abl]
    out["heads_top_dla"] = [[int(i // nh), int(i % nh), round(float(dh[i]), 3), round(float(mh[i]), 3)] for i in np.argsort(-dh)[:10]]
    # heads that matter causally (mean ablation costs > 0.1) but write almost nothing directly (|DLA| < 0.05)
    hidden = [(int(i // nh), int(i % nh), round(float(dh[i]), 3), round(float(mh[i]), 3)) for i in range(len(dh)) if mh[i] < -0.1 and abs(dh[i]) < 0.05]
    out["heads_causal_not_direct"] = hidden
    # per-head CI for the top heads
    out["head_ci"] = {}
    for L, h in [(9, 8), (10, 0), (8, 11), (11, 2), (0, 7)]:
        if L < a["dla_heads"].shape[0] and h < nh:
            x, y = a["dla_heads"][L, h], a["d_mean_heads"][L, h]
            bx = [x[rng.integers(0, n, n)].mean() for _ in range(2000)]
            by = [y[rng.integers(0, n, n)].mean() for _ in range(2000)]
            out["head_ci"][f"{L}.{h}"] = {"dla": [float(x.mean()), *map(float, np.percentile(bx, [2.5, 97.5]))],
                                          "mean_abl": [float(y.mean()), *map(float, np.percentile(by, [2.5, 97.5]))]}
json.dump(out, open(f"{ROOT}/post/{tag}_agree.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items()}, indent=0)[:4000])
