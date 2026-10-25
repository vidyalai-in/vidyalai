"""Experiment 1 analysis: measured success vs p^n, per-step error by position, self-conditioning, cost."""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


arm = sys.argv[1] if len(sys.argv) > 1 else "off"
rows = [json.loads(l) for l in open(HERE / "runs" / f"chain-{arm}.jsonl")]
rows = [r for r in rows if r["seed"] < 900]
by_n = defaultdict(list)
for r in rows:
    by_n[r["n"]].append(r)

steps = [s for r in rows for s in r["steps"]]
errs = len(steps) - sum(steps)
p = 1 - errs / len(steps)
plo, phi = wilson(sum(steps), len(steps))
print(f"pooled per-step success p = {p:.4f} ({sum(steps)}/{len(steps)} steps, 95% CI {plo:.4f}-{phi:.4f})")
print(f"{'n':>4} {'runs':>4} {'solved':>6} {'rate':>6} {'95% CI':>13} {'p^n':>6} {'p^n CI':>13} "
      f"{'p_n':>6} {'cost/run':>9} {'tokens in':>9} {'out':>6} {'model s':>7}")
for n in sorted(by_n):
    rs = by_n[n]
    k = sum(r["correct"] for r in rs)
    lo, hi = wilson(k, len(rs))
    st = [s for r in rs for s in r["steps"]]
    pn = sum(st) / len(st)
    print(f"{n:4d} {len(rs):4d} {k:6d} {k/len(rs):6.1%} {lo:6.1%}-{hi:6.1%} {p**n:6.1%} {plo**n:6.1%}-{phi**n:6.1%} "
          f"{pn:6.3f} ${sum(r['cost'] for r in rs)/len(rs):.5f} {sum(r['input'] for r in rs)/len(rs):9.0f} "
          f"{sum(r['output'] for r in rs)/len(rs):6.0f} {sum(r['model_s'] for r in rs)/len(rs):7.1f}")

# per-step error by position
buckets = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 80)]
print("\nerror rate by step position (all runs; conditional on the record the agent was actually on)")
for a, b in buckets:
    st = [r["steps"][i - 1] for r in rows for i in range(a, b + 1) if i <= len(r["steps"])]
    if st:
        e = len(st) - sum(st)
        lo, hi = wilson(e, len(st))
        print(f"  steps {a:2d}-{b:2d}: {e:3d}/{len(st):5d} = {e/len(st):.2%} ({lo:.2%}-{hi:.2%})")

# self-conditioning: error rate on steps after the run's first error vs steps before it
before = after = eb = ea = 0
for r in rows:
    seen = False
    for s in r["steps"]:
        if seen:
            after += 1
            ea += 1 - s
        else:
            before += 1
            eb += 1 - s
        if not s:
            seen = True
print(f"\nself-conditioning: error rate before first error {eb}/{before} = {eb/before:.2%} "
      f"{tuple(round(x*100, 2) for x in wilson(eb, before))}, after first error {ea}/{after} = "
      f"{ea/max(after,1):.2%} {tuple(round(x*100, 2) for x in wilson(ea, after))}")

# other failure modes
odd = [r for r in rows if len(r["steps"]) != r["n"] or not r["start_ok"] or r["stop"] != "end_turn"]
print(f"\nruns with a wrong number of opens, wrong start or not end_turn: {len(odd)}")
for r in odd[:10]:
    print("  ", r["id"], r["stop"], len(r["opened"]), r["n"], r["start_ok"], r["answer"])
# correct despite errors / wrong despite all steps right
print("solved with an error:", sum(1 for r in rows if r["correct"] and not all(r["steps"])),
      " wrong with all steps right:", sum(1 for r in rows if not r["correct"] and all(r["steps"]) and len(r["steps"]) == r["n"]))
print(f"total cost ${sum(r['cost'] for r in rows):.4f} over {len(rows)} runs, {len(steps)} steps")

# Survival model: hazard of the FIRST error at each step position, pooled over all runs (Kaplan-Meier).
maxn = max(r["n"] for r in rows)
at_risk = [0] * (maxn + 1)
first = [0] * (maxn + 1)
for r in rows:
    for i, s in enumerate(r["steps"], start=1):
        at_risk[i] += 1
        if not s:
            first[i] += 1
            break
S, surv = 1.0, {}
for i in range(1, maxn + 1):
    if at_risk[i]:
        S *= 1 - first[i] / at_risk[i]
    surv[i] = S
h0 = sum(first) / sum(at_risk)
print(f"\nfirst-error hazard pooled: {sum(first)}/{sum(at_risk)} = {h0:.4f}; steps 1-2: "
      f"{sum(first[1:3])}/{sum(at_risk[1:3])}; steps 3+: {sum(first[3:])}/{sum(at_risk[3:])} = {sum(first[3:])/sum(at_risk[3:]):.4f}")
for n in sorted(by_n):
    rs = by_n[n]
    print(f"  n={n:3d} measured {sum(r['correct'] for r in rs)/len(rs):.1%}  Kaplan-Meier no-error {surv[n]:.1%}  "
          f"constant first-error hazard {(1-h0)**n:.1%}  pooled p^n {p**n:.1%}")
