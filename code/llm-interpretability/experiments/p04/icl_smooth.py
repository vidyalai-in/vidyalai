"""Smoothed ICL gap for GPT-2 from per-token losses: mean loss over text tokens 491-510 minus 41-60,
per window, with an article-level bootstrap 95% CI."""
import json, numpy as np
import sys; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/p04/scripts")
from icl_stats import boot
z = np.load("runs/s4_gpt2_tokens.npz"); art = np.load("runs/wiki_gpt2_article.npy")
out = {}
g = {t: z[t].astype(np.float32)[:, 490:510].mean(1) - z[t].astype(np.float32)[:, 40:60].mean(1) for t in z.files}
for t in z.files:
    out[t] = {"gap": boot(g[t], art), "diff": boot(g[t] - g["clean"], art)}
    print(t, "gap %.3f [%.3f, %.3f]" % tuple(out[t]["gap"]), "diff %.3f [%.3f, %.3f]" % tuple(out[t]["diff"]))
json.dump(out, open("post/icl_smooth_gpt2.json", "w"), indent=1)
