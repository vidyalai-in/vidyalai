"""Part 11: how many tasks and repeats separate two configurations? Simulation and Miller's formula.

usage: power.py CONFIG_A CONFIG_B
Uses each task's measured success rate under both configurations as the truth, then simulates
experiments with T tasks (drawn with replacement) and R repeats per task.
"""

import sys

import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from analyze import by_task, load  # noqa: E402
from evals import paired_power  # noqa: E402

a, b = sys.argv[1], sys.argv[2]
ra, rb = by_task(load(a)), by_task(load(b))
tasks = sorted(set(ra) & set(rb))
pa = np.array([np.mean(ra[t]) for t in tasks])
pb = np.array([np.mean(rb[t]) for t in tasks])
delta = pb.mean() - pa.mean()
omega2 = np.var(pb - pa, ddof=1)                      # true per-task differences vary this much
sa2, sb2 = np.mean(pa * (1 - pa)), np.mean(pb * (1 - pb))  # within-task (run-to-run) variance
print(f"{a} -> {b}: {len(tasks)} tasks, delta {delta:+.3f}, omega^2 {omega2:.4f}, "
      f"sigma_A^2 {sa2:.4f}, sigma_B^2 {sb2:.4f}")
z = 1.96 + 0.8416
for K in (1, 3, 5, 10, 1000):
    n = z ** 2 * (omega2 + sa2 / K + sb2 / K) / delta ** 2
    print(f"Miller eq. 9, K={K:4d} repeats: {n:6.1f} tasks for 80% power")
print("\nsimulated power (share of experiments where B beats A at p<0.05, two-sided paired z-test)")
Rs = (1, 3, 5, 10)
print("tasks " + " ".join(f"R={r:<5d}" for r in Rs))
for T in (10, 20, 36, 50, 100, 200):
    print(f"{T:5d} " + " ".join(f"{paired_power(pa, pb, T, r, n_sim=4000, seed=T * 100 + r):.2f}   " for r in Rs))
