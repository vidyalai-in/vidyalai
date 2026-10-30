"""Paired bootstrap (same windows) of delta-loss differences between SAEs, per text type."""
import json, sys, numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
R = "/home/amit/.cache/vidyalai-scratch/p12/runs"
out = {}
for d in ["fineweb_edu", "wikitext", "german_wiki", "python", "korean_wiki"]:
    a = {s: np.load(f"{R}/sae_{d}_{s}.npz") for s in ["ours_topk_24k", "openai_topk_32k", "bloom_relu_24k"]}
    grp = a["ours_topk_24k"]["grp"]; docs, inv = np.unique(grp, return_inverse=True)
    n = np.bincount(inv, a["ours_topk_24k"]["n"])
    dl = {s: np.bincount(inv, v["ce_sae"] - a["ours_topk_24k"]["ce_clean"]) for s, v in a.items()}
    rng = np.random.default_rng(0); o = {}
    for other in ["openai_topk_32k", "bloom_relu_24k"]:
        diff = dl["ours_topk_24k"] - dl[other]
        bs = [diff[i].sum() / n[i].sum() for i in (rng.integers(0, len(docs), len(docs)) for _ in range(2000))]
        o[f"ours_minus_{other}"] = [float(diff.sum() / n.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    o["n_groups"] = int(len(docs)); o["tokens"] = int(n.sum())
    out[d] = o; print(d, {k: [round(x, 3) for x in v] if isinstance(v, list) else v for k, v in o.items()})
json.dump(out, open(f"{R}/sae_paired.json", "w"), indent=1)
