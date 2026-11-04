"""Experiment 1: how well each grader tracks SummEval's expert ratings.
Writes results/summeval.json (small, committable)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import grading as G

D = "/home/amit/.cache/vidyalai-scratch/e06/"
Path(D + "results").mkdir(exist_ok=True)
docs = json.load(open(D + "data/summeval.json"))
DIMS = G.DIMENSIONS
# expert[k][dim] -> (100, 16)
E = {k: {dim: np.array([[d["expert"][s][k][dim] for s in range(16)] for d in docs], float) for dim in DIMS}
     for k in range(3)}
H = {dim: np.mean([E[k][dim] for k in range(3)], 0) for dim in DIMS}

graders = {}
m = json.load(open(D + "runs/metrics.json"))
b = json.load(open(D + "runs/bertscore.json"))
for k, v in m.items():
    graders[k] = np.array(v, float)
for k in ("bertscore_1ref", "bertscore_11ref"):
    graders[k] = np.array(b[k], float)

# LLM judges
judge_meta = {}
for run in ("rubric1", "rubric2", "bare"):
    p = Path(D + f"runs/{run}.jsonl")
    if not p.exists():
        continue
    recs = G.load_runs(p)
    arr = {dim: np.full((100, 16), np.nan) for dim in DIMS}
    bad = 0
    for cid, r in recs.items():
        di, si = (int(x[1:]) for x in cid.split("-"))
        try:
            js = json.loads(r["text"])
            for dim in DIMS:
                v = int(js[dim])
                arr[dim][di, si] = v if 1 <= v <= 5 else np.nan
        except Exception:  # noqa: BLE001
            bad += 1
    judge_meta[run] = dict(n=len(recs), unparsed=bad,
                           missing=int(sum(np.isnan(a).sum() for a in arr.values())))
    for dim in DIMS:
        a = arr[dim]
        a[np.isnan(a)] = 3.0  # impute the scale midpoint for the (reported) few failures
        graders[f"judge_{run}_{dim}"] = a
if "judge_rubric1_coherence" in graders and "judge_rubric2_coherence" in graders:
    for dim in DIMS:
        graders[f"judge_rubric12_{dim}"] = (graders[f"judge_rubric1_{dim}"] + graders[f"judge_rubric2_{dim}"]) / 2

res = {"n_docs": 100, "n_systems": 16, "judge_meta": judge_meta, "corr": {}, "ceiling": {}, "judge_retest": {}}
for g, arr in graders.items():
    for dim in DIMS:
        if g.startswith("judge_") and not g.endswith("_" + dim):
            continue
        key = f"{g}|{dim}"
        res["corr"][key] = {
            "summary_kendall": G.boot_corr(arr, H[dim], G.summary_level, "kendall"),
            "summary_spearman": G.boot_corr(arr, H[dim], G.summary_level, "spearman", n_boot=500),
            "system_kendall": G.boot_corr(arr, H[dim], G.system_level, "kendall"),
            "system_pearson": G.boot_corr(arr, H[dim], G.system_level, "pearson", n_boot=500),
        }

# human ceiling: one expert against the mean of the other two
for dim in DIMS:
    out = {}
    for lvl, fn in (("summary", G.summary_level), ("system", G.system_level)):
        vals = []
        for k in range(3):
            others = np.mean([E[j][dim] for j in range(3) if j != k], 0)
            vals.append(G.boot_corr(E[k][dim], others, fn, "kendall", n_boot=1000, seed=k))
        out[lvl + "_kendall"] = dict(est=float(np.mean([v["est"] for v in vals])),
                                     lo=float(np.mean([v["lo"] for v in vals])),
                                     hi=float(np.mean([v["hi"] for v in vals])), each=vals)
    # exact agreement and kappa between experts (pairs of experts)
    ks = []
    for a in range(3):
        for c in range(a + 1, 3):
            ks.append(G.cohen_kappa(E[a][dim].ravel().astype(int), E[c][dim].ravel().astype(int)))
    out["kappa_pairwise_mean"] = float(np.mean(ks))
    out["share_of_5"] = float((H[dim] == 5).mean())
    res["ceiling"][dim] = out

# judge test-retest
if "judge_rubric2_coherence" in graders:
    for dim in DIMS:
        a, c = graders[f"judge_rubric1_{dim}"].ravel(), graders[f"judge_rubric2_{dim}"].ravel()
        res["judge_retest"][dim] = dict(exact=float((a == c).mean()),
                                        kappa=G.cohen_kappa(a.astype(int), c.astype(int)))

# paired differences: judge vs best reference metric, summary-level Kendall
pairs = {}
if "judge_rubric1_coherence" in graders:
    for dim in DIMS:
        for ref in ("rouge1_1ref", "rouge2_1ref", "rougeL_1ref", "bleu_1ref", "chrf_1ref", "chrf_11ref", "bertscore_1ref", "bertscore_11ref", "rouge1_11ref"):
            pairs[f"{ref}->judge_rubric1|{dim}"] = G.boot_corr_diff(
                graders[ref], graders[f"judge_rubric1_{dim}"], H[dim], G.summary_level, "kendall")
        if "judge_bare_coherence" in graders:
            pairs[f"judge_bare->judge_rubric1|{dim}"] = G.boot_corr_diff(
                graders[f"judge_bare_{dim}"], graders[f"judge_rubric1_{dim}"], H[dim], G.summary_level, "kendall")
res["paired"] = pairs
# score distributions (judge vs human)
res["dist"] = {dim: {"human_mean": float(H[dim].mean()),
                     **({f"{r}_mean": float(graders[f"judge_{r}_{dim}"].mean()) for r in judge_meta})}
               for dim in DIMS}
for r in judge_meta:
    res["dist"][f"{r}_hist"] = {dim: np.bincount(graders[f"judge_{r}_{dim}"].astype(int).ravel(), minlength=6)[1:].tolist()
                                for dim in DIMS}
res["dist"]["human_expert_hist"] = {dim: np.bincount(np.concatenate([E[k][dim].astype(int).ravel() for k in range(3)]),
                                                     minlength=6)[1:].tolist() for dim in DIMS}
json.dump(res, open(D + "results/summeval.json", "w"), indent=1)


def show(key, lvl="summary_kendall"):
    c = res["corr"][key][lvl]
    return f"{c['est']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]"


print("dim        grader                 summary-tau               system-tau")
for dim in DIMS:
    cl = res["ceiling"][dim]
    print(dim, "HUMAN ceiling", f"{cl['summary_kendall']['est']:.3f}", f"{cl['system_kendall']['est']:.3f}",
          "kappa", round(cl["kappa_pairwise_mean"], 3), "share5", round(cl["share_of_5"], 3))
    for g in graders:
        if g.startswith("judge_") and not g.endswith("_" + dim):
            continue
        k = f"{g}|{dim}"
        print(f"  {g:28s} {show(k)}   {show(k, 'system_kendall')}")
print(json.dumps(judge_meta), json.dumps(res["judge_retest"]))
for k, v in pairs.items():
    print(k, f"{v['est']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]")

# fair comparison with the human ceiling: every grader against the mean of two experts
# (averaged over the three ways to pick two), the same target one expert is scored against
fair = {}
for dim in DIMS:
    targets = [np.mean([E[j][dim] for j in range(3) if j != k], 0) for k in range(3)]
    for g in ["rouge1_1ref", "rougeL_1ref", "bleu_1ref", "rouge2_1ref", "chrf_1ref", "bertscore_1ref", "chrf_11ref", "rouge1_11ref", "bertscore_11ref",
              f"judge_rubric1_{dim}", f"judge_bare_{dim}", f"judge_rubric12_{dim}"]:
        if g not in graders:
            continue
        vals = [G.boot_corr(graders[g], t, G.summary_level, "kendall", seed=k) for k, t in enumerate(targets)]
        sys_vals = [G.boot_corr(graders[g], t, G.system_level, "kendall", n_boot=1000, seed=k) for k, t in enumerate(targets)]
        fair[f"{g}|{dim}"] = dict(summary=float(np.mean([v["est"] for v in vals])),
                                  summary_lo=float(np.mean([v["lo"] for v in vals])),
                                  summary_hi=float(np.mean([v["hi"] for v in vals])),
                                  system=float(np.mean([v["est"] for v in sys_vals])),
                                  system_lo=float(np.mean([v["lo"] for v in sys_vals])),
                                  system_hi=float(np.mean([v["hi"] for v in sys_vals])))
res["vs_two_experts"] = fair
json.dump(res, open(D + "results/summeval.json", "w"), indent=1)
for k, v in fair.items():
    print("vs2", k, f"{v['summary']:.3f} [{v['summary_lo']:.3f},{v['summary_hi']:.3f}]  sys {v['system']:.3f} [{v['system_lo']:.3f},{v['system_hi']:.3f}]")
