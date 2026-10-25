"""Summarise the Part 15 arms: accuracy with Wilson and task-bootstrap intervals, cost per run and per
completed task, latency percentiles, cache hit rate. Repeat 0 is the pilot and is left out.

Usage: python analyze.py [arm ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import numpy as np  # noqa: E402
from costlat import pct  # noqa: E402
from evals import task_bootstrap, wilson  # noqa: E402

HERE = Path(__file__).parent
ARMS = ["h-off", "h-low", "h-med", "h-high", "h-med-nocache", "s-med", "team-s", "batch-5m", "batch-1h"]


def load(arm):
    p = HERE / "runs" / f"{arm}.jsonl"
    rs = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    return [r for r in rs if r["repeat"] >= 1]


def latency(r):
    """Model time on the critical path: sum of call times (team: lead plus slowest worker per turn).
    Batch runs: time from the first submit until the run's last batch ended."""
    if r["arm"].startswith("batch"):
        return r["wall_s"]
    return r.get("crit_s", r["model_s"])


def summary(arm):
    rs = load(arm)
    if not rs:
        return None
    n, ok = len(rs), sum(r["correct"] for r in rs)
    lo, hi = wilson(ok, n)
    by = defaultdict(list)
    for r in rs:
        by[r["task"]].append(r["correct"])
    blo, bhi = task_bootstrap(by, lambda d: float(np.mean([np.mean(v) for v in d.values()])), n_boot=4000)
    cost = sum(r["cost"] for r in rs)
    lat = [latency(r) for r in rs]
    inp = sum(r["input"] for r in rs)
    return dict(arm=arm, n=n, ok=ok, acc=ok / n, wlo=lo, whi=hi, blo=blo, bhi=bhi, cost=cost,
                per_run=cost / n, per_ok=cost / ok if ok else float("inf"),
                med=pct(lat, 50), p95=pct(lat, 95), calls=np.mean([r["calls"] for r in rs]),
                inp=inp / n, out=np.mean([r["output"] for r in rs]),
                think=np.mean([r["thinking_tokens"] for r in rs]),
                hit=sum(r["cache_read"] for r in rs) / inp if inp else 0,
                write=sum(r["cache_write"] for r in rs) / inp if inp else 0)


if __name__ == "__main__":
    arms = sys.argv[1:] or ARMS
    print(f"{'arm':14s} {'n':>4s} {'acc':>6s} {'wilson':>12s} {'boot':>12s} {'$/run':>9s} {'$/ok':>9s} "
          f"{'med s':>7s} {'p95 s':>7s} {'calls':>5s} {'in/run':>8s} {'out':>6s} {'think':>6s} {'hit':>5s} {'write':>5s} {'total$':>7s}")
    for a in arms:
        s = summary(a)
        if not s:
            continue
        print(f"{a:14s} {s['n']:4d} {100*s['acc']:5.1f}% {100*s['wlo']:5.1f}-{100*s['whi']:5.1f} "
              f"{100*s['blo']:5.1f}-{100*s['bhi']:5.1f} {s['per_run']:9.5f} {s['per_ok']:9.5f} {s['med']:7.1f} "
              f"{s['p95']:7.1f} {s['calls']:5.1f} {s['inp']:8.0f} {s['out']:6.0f} {s['think']:6.0f} "
              f"{100*s['hit']:4.0f}% {100*s['write']:4.0f}% {s['cost']:7.3f}")


def paired(a, b, iters=10000, seed=0):
    """Accuracy difference b - a in points, with a 95% interval from resampling whole tasks."""
    import random
    ta, tb = defaultdict(list), defaultdict(list)
    for r in load(a):
        ta[r["task"]].append(r["correct"])
    for r in load(b):
        tb[r["task"]].append(r["correct"])
    tasks = sorted(set(ta) & set(tb))
    diff = {t: np.mean(tb[t]) - np.mean(ta[t]) for t in tasks}
    rng = random.Random(seed)
    boots = sorted(sum(diff[rng.choice(tasks)] for _ in tasks) / len(tasks) for _ in range(iters))
    return 100 * np.mean(list(diff.values())), 100 * boots[int(.025 * iters)], 100 * boots[int(.975 * iters)]


def kinds(arm):
    out = {}
    for k in ("chain", "aggregate", "route", "puzzle"):
        rs = [r for r in load(arm) if r["kind"] == k]
        if rs:
            out[k] = (sum(r["correct"] for r in rs), len(rs), np.mean([r["cost"] for r in rs]),
                      pct([latency(r) for r in rs], 50))
    return out


if __name__ == "__main__" and len(sys.argv) == 1:
    print("\npaired vs h-med (points, 95% task bootstrap)")
    for b in ARMS:
        if b != "h-med" and load(b):
            d, lo, hi = paired("h-med", b)
            print(f"  {b:14s} {d:+5.1f} ({lo:+5.1f} to {hi:+5.1f})")
    print("\nper kind: correct/n, $/run, median latency")
    for a in ARMS:
        if load(a):
            print(f"  {a:14s} " + "  ".join(f"{k[:5]} {c}/{n} ${m:.5f} {s:5.1f}s" for k, (c, n, m, s) in kinds(a).items()))
