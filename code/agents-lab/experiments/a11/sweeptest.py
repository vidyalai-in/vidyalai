"""Is the spread of whole-benchmark scores across repeats larger than independent runs would give?
Shuffle each task's runs across repeat indices (keeps every task's success count), recompute the SD
of the per-repeat scores, and compare with the observed SD."""
import sys
import numpy as np
from analyze import by_task, load
cfg = sys.argv[1]
recs = load(cfg)
res = by_task(recs)
reps = sorted({r["rep"] for r in recs})
M = np.array([[next(r["correct"] for r in recs if r["task"] == t and r["rep"] == k) for k in reps] for t in sorted(res)], float)
obs = M.mean(axis=0).std(ddof=1)
rng = np.random.default_rng(0)
sims = np.array([np.array([rng.permutation(row) for row in M]).mean(axis=0).std(ddof=1) for _ in range(20000)])
print(f"{cfg}: per-repeat scores {np.round(M.mean(axis=0)*100,1)}; observed sd {obs*100:.2f} pts; "
      f"shuffled sd median {np.median(sims)*100:.2f} pts; p(shuffled >= observed) = {np.mean(sims >= obs - 1e-12):.4f}")
