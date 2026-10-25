"""Summarize Part 7 runs: accuracy with Wilson intervals, tokens, token multiple, cost, wall time, failures."""
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ARMS = sys.argv[1:] or ["single-full", "team-full", "single-sized", "team-sized", "sonnet-team-full"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h)), round(100 * (c + h))


def load(arm):
    p = HERE / "runs" / f"{arm}.jsonl"
    if not p.exists():
        return []
    return [r for r in map(json.loads, p.read_text().splitlines()) if r["repeat"] < 9]  # 9 = pilot


data = {a: load(a) for a in ARMS}
base_of = {"team-full": "single-full", "sonnet-team-full": "single-full", "team-sized": "single-sized",
           "single-sized": "single-full", "single-full": "single-full"}

for kind in ["all", "wide", "twostage", "chain"]:
    print(f"\n=== {kind}")
    print(f"{'arm':18s} {'n':>3s} {'correct':>14s} {'svc ok':>7s} {'tokens/task':>12s} {'x single':>9s} "
          f"{'peak':>8s} {'$/task':>8s} {'$ x':>6s} {'wall md':>7s} {'model md':>8s} {'workers':>7s} {'reads':>6s} {'dup':>5s}")
    for arm, rs in data.items():
        rs = [r for r in rs if kind == "all" or r["kind"] == kind]
        if not rs:
            continue
        k = sum(r["correct"] for r in rs)
        lo, hi = wilson(k, len(rs))
        base = [r for r in data.get(base_of[arm], []) if kind == "all" or r["kind"] == kind]
        tok = st.mean(r["tokens"] for r in rs)
        btok = st.mean(r["tokens"] for r in base) if base else float("nan")
        cost = st.mean(r["cost"] for r in rs)
        bcost = st.mean(r["cost"] for r in base) if base else float("nan")
        reads = sum(r["reads"] for r in rs)
        uniq = sum(r["unique_reads"] for r in rs)
        print(f"{arm:18s} {len(rs):3d} {k:3d} {100*k/len(rs):4.0f}% ({lo}-{hi}) {sum(r['correct_part'] for r in rs):7d} "
              f"{tok:12,.0f} {tok/btok:9.2f} {st.mean(r['peak_context'] for r in rs):8,.0f} {cost:8.4f} {cost/bcost:6.2f} "
              f"{st.median(r['wall_s'] for r in rs):7.1f} {st.median(r['model_s'] for r in rs):8.1f} "
              f"{st.mean(r['workers'] for r in rs):7.1f} {reads/len(rs):6.1f} {reads/max(uniq,1):5.2f}")

print("\n=== per-task tokens (mean over repeats)")
by = defaultdict(dict)
for arm, rs in data.items():
    g = defaultdict(list)
    for r in rs:
        g[r["task"]].append(r)
    for t, v in g.items():
        by[t][arm] = (st.mean(x["tokens"] for x in v), sum(x["correct"] for x in v), len(v), st.mean(x["cost"] for x in v),
                      st.mean(x["workers"] for x in v), st.mean(x["wall_s"] for x in v))
for t in sorted(by):
    print(t, "  ".join(f"{a}: {v[0]:,.0f} tok {v[1]}/{v[2]} ${v[3]:.4f} w{v[4]:.1f} {v[5]:.0f}s" for a, v in by[t].items()))

print("\n=== failures")
for arm, rs in data.items():
    for r in rs:
        if not r["correct"]:
            print(arm, r["task"], r["repeat"], "ans", repr(r["answer"]), "exp", repr(r["expected"]), r["stop"],
                  "workers", r["workers"], r["worker_stops"], "covered", r["covered"], "/", r["members"],
                  "reads", r["reads"], "uniq", r["unique_reads"])

print("\n=== totals")
for arm, rs in data.items():
    if rs:
        print(f"{arm:18s} runs {len(rs):3d} cost ${sum(r['cost'] for r in rs):.3f}")
