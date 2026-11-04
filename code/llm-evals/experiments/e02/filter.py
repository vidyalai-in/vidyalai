"""What filtering items against a model does. Uses Haiku's 3 samples per item (re-asked where needed).
Filter A: keep items Haiku got wrong on its first sample. Filter B: wrong on all 3 samples.
For each: how many items survive, the share the Redux annotators flagged, Haiku's accuracy on its
other samples (fresh draws), and other models' accuracy (Sonnet on its stratified subset, reweighted;
open models when their runs exist). 95% bootstrap CIs over items."""
import json, sys, os, glob
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import items
R = "/home/amit/.cache/vidyalai-scratch/e02"
data = items.load_redux(); N = len(data)
gold = np.array([d["answer"] for d in data]); flag = np.array([d["error_type"] != "ok" for d in data])
H = np.load(f"{R}/results/picks-haiku-k3.npy")[:, 1:]          # (5700, 3)
Hc = H == gold[:, None]
others = {}
if os.path.exists(f"{R}/results/picks-sonnet-k1.npy"):
    S = np.load(f"{R}/results/picks-sonnet-k1.npy")
    others["Sonnet 5.5"] = (S[:, 0], (S[:, 1] == gold[S[:, 0]]).astype(float))
for f in sorted(glob.glob(f"{R}/runs/open-*.npy")):
    lp = np.load(f); others[os.path.basename(f)[5:-4]] = (np.arange(N), (lp.argmax(1) == gold).astype(float))
w_fl = flag.mean()
def wacc(idx, corr, mask):
    """accuracy over items in mask, reweighting a stratified subset (flagged/ok) to the population."""
    sel = mask[idx]
    if sel.sum() == 0: return np.nan
    f = flag[idx][sel]; c = corr[sel]
    if len(idx) == N: return c.mean()
    pf = mask & flag; po = mask & ~flag                           # population counts in the filtered set
    a_f = c[f].mean() if f.any() else 0; a_o = c[~f].mean() if (~f).any() else 0
    return (a_f * pf.sum() + a_o * po.sum()) / mask.sum()
res = {}
for name, mask, fresh in [("all items", np.ones(N, bool), None),
                          ("Haiku wrong on sample 1", ~Hc[:, 0], Hc[:, 1:]),
                          ("Haiku wrong on all 3", ~Hc.any(1), None)]:
    m = np.where(mask)[0]
    r = dict(items=int(mask.sum()), flagged_share=items.boot_ci(lambda s: flag[m[s]].mean(), len(m), 5000),
             wrong_gt_share=float(np.mean([data[i]["error_type"] == "wrong_groundtruth" for i in m])))
    if fresh is not None:
        r["haiku_fresh_samples_acc"] = items.boot_ci(lambda s: fresh[m[s]].mean(), len(m), 5000)
    r["haiku_all_samples_acc"] = items.boot_ci(lambda s: Hc[m[s]].mean(), len(m), 5000)
    for k, (idx, corr) in others.items():
        def st(s, idx=idx, corr=corr):
            mm = np.zeros(N, bool); np.add.at(mm, m[s], True)   # resampled filtered set (as a set)
            return wacc(idx, corr, mm)
        r[k] = items.boot_ci(st, len(m), 2000)
        r[k + " n_scored"] = int(mask[idx].sum())
    res[name] = {k: ([round(float(x), 4) for x in v] if isinstance(v, tuple) else v) for k, v in r.items()}
    print(name, json.dumps(res[name]))
json.dump(res, open(f"{R}/results/filter.json", "w"), indent=1)
