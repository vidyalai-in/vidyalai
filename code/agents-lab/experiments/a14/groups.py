"""Part 14 measurement (no API calls): how many of Part 8's tasks would give a GRPO-style learning signal.

A task gives no gradient in group-relative RL when every rollout in its group earns the same reward.
Part 8 ran each of 28 tasks 3 times per configuration; treat each task's 3 runs as a group of 3.
Then compute, for a task the policy solves with probability p, the chance that a group of G
rollouts is mixed (has at least one success and one failure): 1 - p^G - (1-p)^G.
"""
import json
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in
        Path("/home/amit/.cache/vidyalai-scratch/a08/runs/results.jsonl").read_text().splitlines()]
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["arm"], r["task"])].append(bool(r["resolved"]))

print("model, arm: tasks all-pass / mixed / all-fail (groups of 3 runs), resolve rate")
summary = defaultdict(lambda: [0, 0, 0, 0, 0])
for (m, a, t), g in groups.items():
    s = summary[(m, a)]
    s[0 if all(g) else 2 if not any(g) else 1] += 1
    s[3] += sum(g)
    s[4] += len(g)
for (m, a), s in sorted(summary.items()):
    print(f"  {m:18s} {a:14s} {s[0]:2d} / {s[1]:2d} / {s[2]:2d}   {s[3]}/{s[4]}")

print("\nP(group of G rollouts is mixed) for a task solved with probability p")
ps = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
print("  G   " + "  ".join(f"p={p:<5}" for p in ps))
for G in (3, 8, 16, 64, 512):
    print(f"  {G:<4d}" + "  ".join(f"{100 * (1 - p ** G - (1 - p) ** G):6.1f}%" for p in ps))
