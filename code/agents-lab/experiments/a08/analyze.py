"""Summarize runs/results.jsonl: resolve rate with 95% Wilson CIs, steps, tokens, cost per arm."""

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

A08 = Path("/home/amit/.cache/vidyalai-scratch/a08")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(path=A08 / "runs/results.jsonl"):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def table(rows, key=lambda r: (r["model"], r["effort"], r["arm"])):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    print(f"{'group':42s} {'res':>9s} {'rate':>6s} {'CI':>13s} {'steps':>5s} {'tok/run':>8s} {'out':>6s} "
          f"{'peak':>6s} {'$/run':>8s} {'$/res':>8s} {'wall':>5s} {'edfail':>6s}")
    for k in sorted(g):
        rs = g[k]
        n, res = len(rs), sum(r["resolved"] for r in rs)
        lo, hi = wilson(res, n)
        tok = st.mean(r["input"] + r["cache_read"] + r["cache_write"] + r["output"] for r in rs)
        cost = sum(r["cost"] for r in rs)
        print(f"{str(k):42s} {res:4d}/{n:<4d} {100 * res / n:5.1f}% ({lo:4.1f}-{hi:5.1f}) "
              f"{st.mean(r['steps'] for r in rs):5.1f} {tok:8.0f} {st.mean(r['output'] for r in rs):6.0f} "
              f"{st.mean(r['peak_context'] for r in rs):6.0f} {cost / n:8.5f} {cost / max(res, 1):8.5f} "
              f"{st.mean(r['wall_s'] for r in rs):5.0f} {sum(r.get('edit_fail', 0) for r in rs):6d}")


if __name__ == "__main__":
    rows = load(sys.argv[1] if len(sys.argv) > 1 else A08 / "runs/results.jsonl")
    table(rows)
    print()
    fails = defaultdict(list)
    for r in rows:
        if not r["resolved"]:
            fails[r["task"]].append(f"{r['arm']}/{r['model'].split('-')[1]}/r{r['rep']}")
    for t, v in sorted(fails.items()):
        print(f"{t:30s} {len(v):3d} {' '.join(v)}")
