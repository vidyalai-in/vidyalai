"""Part 11 analysis: pass@1, pass@k, pass^k, run-to-run spread, cost per solved task, paired tests, power."""

import json
import sys
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from evals import pass_at_k, pass_hat_k, suite, task_bootstrap, wilson  # noqa: E402

A11 = Path("/home/amit/.cache/vidyalai-scratch/a11")
RUNS = A11 / "runs"


def load(cfg, max_rep=None):
    recs = [json.loads(l) for l in (RUNS / f"{cfg}.jsonl").read_text().splitlines()]
    if max_rep is not None:
        recs = [r for r in recs if r["rep"] < max_rep]
    return recs


def by_task(recs):
    d = defaultdict(list)
    for r in sorted(recs, key=lambda r: r["rep"]):
        d[r["task"]].append(bool(r["correct"]))
    return dict(d)


def report(cfg, max_rep=None):
    recs = load(cfg, max_rep)
    res = by_task(recs)
    n = min(len(v) for v in res.values())
    res = {t: v[:n] for t, v in res.items()}
    T = len(res)
    runs = T * n
    succ = sum(sum(v) for v in res.values())
    cost = sum(r["cost"] for r in recs)
    print(f"\n=== {cfg}: {T} tasks x {n} runs = {runs} (records {len(recs)})")
    lo, hi = wilson(succ, runs)
    blo, bhi = task_bootstrap(res, lambda r: suite(r, 1, pass_at_k))
    print(f"pass@1 {succ}/{runs} = {succ/runs:.3f}  Wilson {lo:.3f}-{hi:.3f}  task-bootstrap {blo:.3f}-{bhi:.3f}")
    for k in (1, 2, 3, 5, n):
        pa, ph = suite(res, k, pass_at_k), suite(res, k, pass_hat_k)
        a_lo, a_hi = task_bootstrap(res, lambda r: suite(r, k, pass_at_k), n_boot=2000)
        h_lo, h_hi = task_bootstrap(res, lambda r: suite(r, k, pass_hat_k), n_boot=2000)
        p1 = succ / runs
        print(f"k={k:2d} pass@k {pa:.3f} ({a_lo:.3f}-{a_hi:.3f}) naive 1-(1-p)^k {1-(1-p1)**k:.3f} | "
              f"pass^k {ph:.3f} ({h_lo:.3f}-{h_hi:.3f}) naive p^k {p1**k:.3f}")
    print("curve pass@k:", [round(suite(res, k, pass_at_k), 4) for k in range(1, n + 1)])
    print("curve pass^k:", [round(suite(res, k, pass_hat_k), 4) for k in range(1, n + 1)])
    hist = defaultdict(int)
    for v in res.values():
        hist[sum(v)] += 1
    print("tasks by successes:", dict(sorted(hist.items())))
    print("never:", sorted(t for t, v in res.items() if not any(v)))
    print("flaky:", sorted((t, sum(v)) for t, v in res.items() if 0 < sum(v) < n))
    # one "benchmark run" = every task once, at repeat r
    sweep = [np.mean([res[t][r] for t in res]) for r in range(n)]
    p = np.array([np.mean(v) for v in res.values()])
    pred_sd = sqrt(sum(p * (1 - p))) / T
    print(f"sweeps: {[round(s, 3) for s in sweep]}  mean {np.mean(sweep):.3f} sd {np.std(sweep, ddof=1):.4f} "
          f"min {min(sweep):.3f} max {max(sweep):.3f}; predicted sd from per-task p {pred_sd:.4f}")
    print(f"between-task var of p {p.var(ddof=1):.4f}; mean within-task p(1-p) {np.mean(p*(1-p)):.4f}")
    print(f"cost total ${cost:.4f}; per run ${cost/len(recs):.5f}; per solved run ${cost/max(succ,1):.5f}; "
          f"tokens/run {np.mean([r['input']+r['output'] for r in recs]):.0f}; out/run {np.mean([r['output'] for r in recs]):.0f}; "
          f"steps {np.mean([r['steps'] for r in recs]):.1f}; wall {np.mean([r['wall_s'] for r in recs]):.1f}s")
    fails = [r for r in recs if not r["correct"]]
    print("fail stops:", {s: sum(r["stop"] == s for r in fails) for s in set(r["stop"] for r in fails)})
    return res, recs


def paired(res_a, res_b, name):
    tasks = sorted(set(res_a) & set(res_b))
    n = min(min(len(res_a[t]) for t in tasks), min(len(res_b[t]) for t in tasks))
    d = np.array([np.mean(res_b[t][:n]) - np.mean(res_a[t][:n]) for t in tasks])
    se = d.std(ddof=1) / sqrt(len(d))
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # sign-flip permutation test on per-task differences
    flips = rng.choice([-1, 1], size=(20000, len(d)))
    pval = np.mean(np.abs((flips * d).mean(axis=1)) >= abs(d.mean()) - 1e-12)
    better = sum(d > 0); worse = sum(d < 0)
    print(f"\n--- {name}: {len(tasks)} tasks x {n} runs. mean diff {d.mean():+.3f} (SE {se:.3f}, "
          f"bootstrap {lo:+.3f} to {hi:+.3f}); permutation p={pval:.4f}; tasks B better {better}, worse {worse}, same {len(d)-better-worse}")
    return d


def by_rep_table(res_a, res_b, name):
    """With only r repeats and the first T tasks, how often would the comparison look significant?"""
    pass


if __name__ == "__main__":
    out = {}
    for cfg in sys.argv[1:] or ["salt-off", "salt-low", "code-haiku", "code-sonnet"]:
        if (RUNS / f"{cfg}.jsonl").exists():
            out[cfg] = report(cfg)
    if "salt-off" in out and "salt-low" in out:
        paired(out["salt-off"][0], out["salt-low"][0], "salt-off -> salt-low")
    if "code-haiku" in out and "code-sonnet" in out:
        paired(out["code-haiku"][0], out["code-sonnet"][0], "code-haiku -> code-sonnet")
