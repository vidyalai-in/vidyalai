# Collect per-checkpoint results into post/summary.json (small; used by the charts and the article).
import json, glob, os, numpy as np
R = "/home/amit/.cache/vidyalai-scratch/p10/runs"; P = "/home/amit/.cache/vidyalai-scratch/p10/post"; os.makedirs(P, exist_ok=True)
E = np.load(f"{R}/evalsets.npz")
def ci(x): x = np.asarray(x, float); x = x[~np.isnan(x)]; return [round(float(x.mean()), 4), round(float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), 4)]
out = {"H_uni": float(E["H_uni"]), "H_bi": float(E["H_bi"]), "n_allowed": int(E["n_allowed"]), "runs": {}}
for run in sorted(os.listdir(f"{R}/track")):
    rows = []
    for f in sorted(glob.glob(f"{R}/track/{run}/*.npz")):
        d = np.load(f); ind = d["rep_ind"].mean(0); prev = d["nat_prev"].mean(0); dup = d["rep_dup"].mean(0)
        L, H = ind.shape
        order = np.argsort(ind.ravel())[::-1]
        top3 = [divmod(int(i), H) for i in order[:3]]
        lowprev = prev[:max(1, L - 1)]
        rows.append({"step": int(d["step"]), "tokens": int(d["tokens"]),
                     "ind_top": [[l, h, round(float(ind[l, h]), 4), round(float(1.96 * d["rep_ind"][:, l, h].std(ddof=1) / np.sqrt(d["rep_ind"].shape[0])), 4)] for l, h in top3],
                     "ind_max": round(float(ind.max()), 4), "n_ind_04": int((ind > 0.4).sum()),
                     "prev_max": round(float(prev.max()), 4), "prev_arg": list(divmod(int(prev.argmax()), H)),
                     "dup_max": round(float(dup.max()), 4), "dup_arg": list(divmod(int(dup.argmax()), H)),
                     "first": ci(d["rep_first"]), "second": ci(d["rep_second"]),
                     "icl": ci(d["icl_L500"] - d["icl_L50"]), "icl_s": ci(d["icl_s500"] - d["icl_s50"]), "loss": ci(d["icl_mean"]),
                     "kl_uni": ci(d["kl_uni"]), "kl_bi": ci(d["kl_bi"]), "uni_loss": ci(d["uni_loss"]), "bi_loss": ci(d["bi_loss"]),
                     "ind_grid": np.round(ind, 3).tolist() if run == "main88" else None,
                     "prev_low_max": round(float(prev[:8].max()), 4) if run == "main88" else None})
    out["runs"][run] = rows
for run in out["runs"]:
    lg = json.load(open(f"{R}/log_{run}.json")); out["runs_val_" + run] = {int(k): round(v, 4) for k, v in lg["val"].items()}
if os.path.exists(f"{R}/pythia70m.json"): out["pythia70m"] = json.load(open(f"{R}/pythia70m.json"))
out["probe"] = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/probe/*.json"))]
out["ablate"] = {run: [json.load(open(f)) for f in sorted(glob.glob(f"{R}/ablate/{run}/*.json"))] for run in (os.listdir(f"{R}/ablate") if os.path.exists(f"{R}/ablate") else [])}
json.dump(out, open(f"{P}/summary.json", "w"))
print("size KB", os.path.getsize(f"{P}/summary.json") / 1e3)
