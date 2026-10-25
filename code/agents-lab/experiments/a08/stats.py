"""Extra statistics for the article: paired comparisons, cutoff split, test use, edit failures."""

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/a08")
from analyze import wilson  # noqa: E402
from tx_analysis import behaviour  # noqa: E402

A08 = Path("/home/amit/.cache/vidyalai-scratch/a08")
rows = [json.loads(l) for l in (A08 / "runs/results.jsonl").read_text().splitlines()]
for r in rows:
    r.update(behaviour(r))
tasks = {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in A08.glob("tasks/*/task.json")}
idx = {(r["model"], r["arm"], r["task"], r["rep"]): r for r in rows}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


H = "claude-haiku-5-5"
print("Paired vs full (Haiku): arm, only-full-resolved, only-arm-resolved, exact McNemar p")
for arm in ["no_tests", "rewrite", "no_search", "map", "map_no_search", "title_only"]:
    b = c = 0
    for (m, a, t, rep), r in idx.items():
        if m == H and a == arm and (H, "full", t, rep) in idx:
            f = idx[(H, "full", t, rep)]["resolved"]
            b += f and not r["resolved"]
            c += r["resolved"] and not f
    print(f"  {arm:14s} {b:3d} {c:3d}  p={mcnemar_exact(b, c):.4f}")

print("\nBy fix date (after the June 2026 training cutoff or not)")
for model in sorted({r["model"] for r in rows}):
    for arm in ["full", "title_only"]:
        for late in (False, True):
            rs = [r for r in rows if r["model"] == model and r["arm"] == arm
                  and (tasks[r["task"]]["fix_date"] > "2026-06-30") == late]
            if rs:
                k = sum(r["resolved"] for r in rs)
                lo, hi = wilson(k, len(rs))
                print(f"  {model:18s} {arm:11s} {'after' if late else 'before'}: {k}/{len(rs)} "
                      f"{100 * k / len(rs):.1f}% ({lo:.1f}-{hi:.1f})")

print("\nTest use per arm: % runs with no run_tests call; resolve rate with / without tests")
for (model, arm) in sorted({(r["model"], r["arm"]) for r in rows}):
    rs = [r for r in rows if r["model"] == model and r["arm"] == arm]
    no = [r for r in rs if r["test_calls"] == 0]
    yes = [r for r in rs if r["test_calls"] > 0]
    fe = [r for r in rs if r["edit_errors"] > 0]
    sr = sum(r["tool_counts"].get("str_replace", 0) for r in rs)
    print(f"  {model.split('-')[1]:7s} {arm:14s} n={len(rs):3d} no-test runs {len(no):3d} "
          f"({100 * len(no) / len(rs):4.1f}%), resolved when no tests {sum(r['resolved'] for r in no)}/{len(no)}, "
          f"with tests {sum(r['resolved'] for r in yes)}/{len(yes)}; runs with a failed edit {len(fe)} "
          f"({100 * len(fe) / len(rs):.1f}%), str_replace calls {sr}, "
          f"repro {sum(r['repro'] > 0 for r in rs)}, median steps {st.median(r['steps'] for r in rs)}, "
          f"max steps {max(r['steps'] for r in rs)}, median cost {st.median(r['cost'] for r in rs):.5f}")

print("\nPer-task resolve counts (full, title_only, rewrite; Haiku)")
for t in sorted(tasks):
    cells = []
    for arm in ["full", "no_tests", "rewrite", "no_search", "map", "map_no_search", "title_only"]:
        rs = [r for r in rows if r["model"] == H and r["arm"] == arm and r["task"] == t]
        cells.append(f"{sum(r['resolved'] for r in rs)}/{len(rs)}")
    ss = [r for r in rows if r["model"] != H and r["task"] == t]
    print(f"  {t:28s} {' '.join(cells)}  sonnet {sum(r['resolved'] for r in ss)}/{len(ss)}")
