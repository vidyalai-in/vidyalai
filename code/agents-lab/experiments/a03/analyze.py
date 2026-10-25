"""Summarise runs/*.jsonl: accuracy (Wilson 95% CI), tokens, cost, latency per arm and per task kind.

Usage: python analyze.py [arm ...]
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ORDER = ["plain", "react", "plan", "low", "medium", "high", "xhigh", "max",
         "s-plain", "s-low", "s-medium", "s-high"]
KINDS = ["chain", "aggregate", "route", "puzzle"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(arm):
    p = HERE / "runs" / f"{arm}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def paired(a, b, iters=10000, seed=0):
    """Difference in accuracy (b - a) with a 95% CI from resampling whole tasks."""
    ta, tb = defaultdict(list), defaultdict(list)
    for r in a:
        ta[r["task"]].append(r["correct"])
    for r in b:
        tb[r["task"]].append(r["correct"])
    tasks = sorted(set(ta) & set(tb))
    diff = {t: sum(tb[t]) / len(tb[t]) - sum(ta[t]) / len(ta[t]) for t in tasks}
    rng = random.Random(seed)
    boots = sorted(sum(diff[rng.choice(tasks)] for _ in tasks) / len(tasks) for _ in range(iters))
    return 100 * sum(diff.values()) / len(tasks), 100 * boots[int(.025 * iters)], 100 * boots[int(.975 * iters)]


def main(arms):
    rows = {a: load(a) for a in arms}
    rows = {a: r for a, r in rows.items() if r}
    print(f"{'arm':9s} {'n':>4s} {'acc%':>6s} {'95% CI':>12s} | " + " ".join(f"{k:>9s}" for k in KINDS) +
          f" | {'calls':>5s} {'tools':>5s} {'out':>6s} {'think':>6s} {'in':>7s} {'$/run':>8s} {'s/run':>6s} {'acc/$':>7s} {'$/correct':>9s}")
    for a, r in rows.items():
        n, k = len(r), sum(x["correct"] for x in r)
        lo, hi = wilson(k, n)
        per = []
        for kind in KINDS:
            rr = [x for x in r if x["kind"] == kind]
            per.append(f"{sum(x['correct'] for x in rr):>3d}/{len(rr):<3d}  ")
        mean = lambda f: sum(x[f] for x in r) / n  # noqa: E731
        cost = mean("cost")
        print(f"{a:9s} {n:4d} {100*k/n:6.1f} {lo:5.1f}-{hi:5.1f} | " + " ".join(per) +
              f"| {mean('calls'):5.1f} {mean('tool_calls'):5.1f} {mean('output'):6.0f} {mean('thinking_tokens'):6.0f} "
              f"{mean('input'):7.0f} {cost:8.5f} {mean('model_s'):6.1f} {k/n/cost/1000:7.2f} {cost*n/max(k,1):9.5f}")
    print("\nper kind: mean output tokens / thinking tokens / $ per run / model s")
    for a, r in rows.items():
        cells = []
        for kind in KINDS:
            rr = [x for x in r if x["kind"] == kind]
            m = lambda f: sum(x[f] for x in rr) / len(rr)  # noqa: E731
            cells.append(f"{m('output'):6.0f}/{m('thinking_tokens'):6.0f}/{m('cost'):.5f}/{m('model_s'):5.1f}")
        print(f"{a:9s} " + " | ".join(cells))
    print("\nper kind accuracy with Wilson CI")
    for a, r in rows.items():
        cells = []
        for kind in KINDS:
            rr = [x for x in r if x["kind"] == kind]
            kk = sum(x["correct"] for x in rr)
            lo, hi = wilson(kk, len(rr))
            cells.append(f"{100*kk/len(rr):5.1f} ({lo:4.1f}-{hi:5.1f})")
        print(f"{a:9s} " + " | ".join(cells))
    base = "plain" if "plain" in rows else next(iter(rows))
    print(f"\npaired difference vs {base} (task-resampled 95% CI), percentage points")
    for a, r in rows.items():
        if a != base:
            d, lo, hi = paired(rows[base], r)
            print(f"{a:9s} {d:+6.1f} ({lo:+5.1f} to {hi:+5.1f})")
    print("\nstops and replans")
    for a, r in rows.items():
        stops = defaultdict(int)
        for x in r:
            stops[x["stop"]] += 1
        print(f"{a:9s} {dict(stops)} replanned={sum(x.get('replanned', False) for x in r)} "
              f"thinking-steps share={sum(x['thinking_steps'] for x in r)/max(1,sum(x['calls'] for x in r)):.2f} "
              f"total ${sum(x['cost'] for x in r):.4f}")
    print("\nper task correct counts")
    tasks = sorted({x["task"] for r in rows.values() for x in r}, key=lambda t: (KINDS.index(t.rstrip('0123456789')), t))
    print("task        " + " ".join(f"{a[:7]:>7s}" for a in rows))
    for t in tasks:
        print(f"{t:11s} " + " ".join(f"{sum(x['correct'] for x in r if x['task']==t):>3d}/{sum(1 for x in r if x['task']==t):<3d}" for r in rows.values()))


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)
