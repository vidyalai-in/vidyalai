"""Split WikiText tokens into 'induction-predictable' (the current token appeared earlier in the
window AND the token that followed it then is the token to predict now) vs the rest; compare losses."""
import json, numpy as np
R = "/home/amit/.cache/vidyalai-scratch/p04/runs/"
z = np.load(R + "s4_gpt2_tokens.npz")
wiki = np.load(R + "wiki_gpt2.npy"); art = np.load(R + "wiki_gpt2_article.npy")
n, T = wiki.shape
mask = np.zeros((n, T - 1), bool)       # mask[w, t]: predicting text token t+1 from token t
for w in range(n):
    seen = set()                         # earlier bigrams (a, b)
    x = wiki[w]
    for t in range(T - 1):
        if t > 0:
            seen.add((x[t - 1], x[t]))   # bigrams that end at or before t
        if (x[t], x[t + 1]) in seen:
            mask[w, t] = True
# token_loss index j predicts token j+1 of [BOS]+text; text token t+1 (0-based t+1) is loss index t+1
def boot(v, g, nb=2000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(g); groups = [np.where(g == a)[0] for a in u]
    m = [np.nanmean(np.concatenate([v[groups[i]] for i in rng.integers(0, len(u), len(u))])) for _ in range(nb)]
    return [float(np.nanmean(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
out = {"frac_predictable": float(mask.mean()), "n_tokens": int(mask.size)}
for tag in z.files:
    L = z[tag].astype(np.float32)[:, 1:]          # drop the BOS->first-token prediction; now L[:, t] predicts text token t+1
    pred = np.where(mask, L, np.nan); rest = np.where(~mask, L, np.nan)
    out[tag] = {"pred": float(np.nanmean(pred)), "rest": float(np.nanmean(rest))}
clean = z["clean"].astype(np.float32)[:, 1:]
for tag in z.files:
    if tag == "clean": continue
    L = z[tag].astype(np.float32)[:, 1:]
    d = L - clean
    dp = np.nanmean(np.where(mask, d, np.nan), 1); dr = np.nanmean(np.where(~mask, d, np.nan), 1)
    out[tag]["d_pred"] = boot(dp, art); out[tag]["d_rest"] = boot(dr, art)
print(json.dumps(out, indent=1))
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/p04/post/s4_gpt2.json", "w"), indent=1)
