"""Experiment 1 analysis: score gains and detector AUROCs (CPU only).
Writes runs/results-exp1.json."""
import json, sys
import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import contamination as C
import stats

RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
G = json.load(open(f"{RUNS}/groups.json"))
B0, B1 = json.load(open(f"{RUNS}/eval-base.json")), json.load(open(f"{RUNS}/eval-contam.json"))
items = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e01-arc-challenge.json"))
ids = B0["ids"]
pos = {i: k for k, i in enumerate(ids)}
ans = np.array([items[i]["answer"] for i in ids])


def correct(E, norm=True):
    S, B = np.array(E["S"]), np.array(E["B"])
    sc = S / B if norm else S
    return (sc.argmax(1) == ans).astype(float)


def p_correct(E):
    S = np.array(E["S"])
    S = S - S.max(1, keepdims=True)
    P = np.exp(S) / np.exp(S).sum(1, keepdims=True)
    return P[np.arange(len(ans)), ans]


res = {"groups": {}, "detectors": {}, "shard": {}}
c0, c1 = correct(B0), correct(B1)
q0, q1 = p_correct(B0), p_correct(B1)
names = [g for g in G if g != "_meta"]
d_ctrl = None
for g in names:
    k = np.array([pos[i] for i in G[g]["items"]])
    pd_ = stats.paired_diff(c0[k], c1[k])
    r = dict(n=len(k), before=float(c0[k].mean()), after=float(c1[k].mean()),
             before_ci=stats.wilson(int(c0[k].sum()), len(k)), after_ci=stats.wilson(int(c1[k].sum()), len(k)),
             diff=pd_["diff"], diff_ci=pd_["ci_paired"], mcnemar=stats.mcnemar(c0[k], c1[k]),
             pc_before=float(q0[k].mean()), pc_after=float(q1[k].mean()),
             pc_diff_ci=stats.paired_bootstrap(q0[k], q1[k]))
    res["groups"][g] = r
# gain over the control group's drift (difference in differences), items resampled within groups
rng = np.random.default_rng(0)
k0 = np.array([pos[i] for i in G["g0"]["items"]])
for g in names:
    k = np.array([pos[i] for i in G[g]["items"]])
    dg, d0 = c1[k] - c0[k], c1[k0] - c0[k0]
    boots = [dg[rng.integers(0, len(dg), len(dg))].mean() - d0[rng.integers(0, len(d0), len(d0))].mean()
             for _ in range(10000)]
    res["groups"][g]["did"] = float(dg.mean() - d0.mean())
    res["groups"][g]["did_ci"] = [float(x) for x in np.quantile(boots, [0.025, 0.975])]

# detectors: each seen group (positives) vs the never-seen group g0 (negatives)
lp0 = {i: np.array(x) for i, x in zip(ids, B0["verbatim_lp"])}
lp1 = {i: np.array(x) for i, x in zip(ids, B1["verbatim_lp"])}
for model_tag, E in (("base", B0), ("contam", B1)):
    for kind in ("verbatim", "para"):
        D = {i: dict(d) for i, d in zip(ids, E[kind])}
        if model_tag == "contam" and kind == "verbatim":
            for i in ids:
                D[i]["ref"] = float(lp1[i].mean() - lp0[i].mean())
        for g in names[1:]:
            for det in ("loss", "zlib", "mink", "minkpp", "ref"):
                if det not in D[ids[0]]:
                    continue
                a = C.auroc_ci([D[i][det] for i in G[g]["items"]], [D[i][det] for i in G["g0"]["items"]], n_boot=2000)
                pos_s = np.array([D[i][det] for i in G[g]["items"]]); neg_s = np.array([D[i][det] for i in G["g0"]["items"]])
                thr = np.quantile(neg_s, 0.95)
                res["detectors"][f"{model_tag}/{kind}/{g}/{det}"] = dict(auroc=a[0], ci=[a[1], a[2]],
                                                                       tpr_at_5fpr=float((pos_s > thr).mean()))
for tag, E in (("base", B0), ("contam", B1)):
    for g, r in E["shard"].items():
        res["shard"][f"{tag}/{g}"] = dict(p=r["p"], t=r["t"], mean_diff=r["mean_diff"], n_shards=r["n_shards"],
                                          n_perm=r["n_perm"], shards_p_lt_05=int(sum(p < 0.05 for p in r["shard_p"])))
json.dump(res, open(f"{RUNS}/results-exp1.json", "w"), indent=1)
for g, r in res["groups"].items():
    print(g, "before %.3f after %.3f diff %+.3f [%+.3f, %+.3f]  DiD %+.3f [%+.3f, %+.3f]  pc %.3f->%.3f" % (
        r["before"], r["after"], r["diff"], *r["diff_ci"], r["did"], *r["did_ci"], r["pc_before"], r["pc_after"]))
for k, v in res["detectors"].items():
    print(k, "AUROC %.3f [%.3f, %.3f] TPR@5%%FPR %.2f" % (v["auroc"], *v["ci"], v["tpr_at_5fpr"]))
for k, v in res["shard"].items():
    print("shard", k, v)
