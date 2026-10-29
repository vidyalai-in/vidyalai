"""ICL score = loss(token 500) - loss(token 50) (Olsson et al. 2022), with 95% CIs from a
bootstrap that resamples whole articles (windows from one article are not independent)."""
import json, sys, numpy as np
def boot(vals, art, n=2000, seed=0):
    rng = np.random.default_rng(seed); arts = np.unique(art)
    groups = [np.where(art == a)[0] for a in arts]
    ms = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick]); ms.append(vals[idx].mean())
    return float(vals.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
def summarize(path):
    d = json.load(open(path)); art = np.array(d["nat_article"]); nat = d["nat"]
    out = {}
    clean = np.array(nat["clean"]["L500"]) - np.array(nat["clean"]["L50"])
    for tag, v in nat.items():
        s = np.array(v["L500"]) - np.array(v["L50"])
        out[tag] = {"icl": boot(s, art), "diff_vs_clean": boot(s - clean, art), "mean_loss": v["mean_all"],
                    "L50": float(np.mean(v["L50"])), "L500": float(np.mean(v["L500"]))}
    out["n_windows"] = len(art); out["n_articles"] = int(len(np.unique(art)))
    return out
if __name__ == "__main__":
    o = summarize(sys.argv[1])
    for k, v in o.items():
        if isinstance(v, dict):
            print(k, "ICL %.3f [%.3f, %.3f]" % tuple(v["icl"]), "diff %.3f [%.3f, %.3f]" % tuple(v["diff_vs_clean"]), "loss %.3f" % v["mean_loss"], "L50 %.2f L500 %.2f" % (v["L50"], v["L500"]))
        else: print(k, v)
    json.dump(o, open(sys.argv[2], "w"), indent=1)
