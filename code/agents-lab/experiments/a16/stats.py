"""Paired comparisons for Part 16: difference in seed recall between two configurations on the same
seeds, with a 95% interval from resampling seeds (each seed keeps all its runs), plus the rate at
which each real pre-existing defect was reported per run of its page.

python stats.py <cfgA> <cfgB> [...pairs]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import score

A16 = Path(__file__).parent


def recall_by_seed(cfg):
    _, found, *_ = score.score(cfg)
    return {k: np.mean(v) for k, v in found.items() if v}


def paired(a, b, n_boot=10_000, seed=0):
    ra, rb = recall_by_seed(a), recall_by_seed(b)
    ids = sorted(set(ra) & set(rb))
    d = np.array([rb[i] - ra[i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{a} -> {b}: {len(ids)} seeds, recall {np.mean([ra[i] for i in ids]):.3f} -> "
          f"{np.mean([rb[i] for i in ids]):.3f}, difference {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


def real_rates(cfg):
    runs = [json.loads(p.read_text()) for p in sorted((A16 / "runs" / cfg).glob("*.json"))]
    pages = defaultdict(int)
    for r in runs:
        pages[r["page"]] += 1
    hits = defaultdict(int)
    for r in runs:
        seen = set()
        for f in r["findings"]:
            lab = score.label_of(r["page"], f)
            if lab and lab["label"] == "real":
                seen.add(lab["id"])
        for i in seen:
            hits[i] += 1
    out = {}
    for lab in score.LABELS:
        if lab["label"] == "real" and pages.get(lab["page"]):
            out[lab["id"]] = f"{hits[lab['id']]}/{pages[lab['page']]}"
    cost = sum(r["cost"] for r in runs)
    print(f"{cfg}: real pre-existing defects reported per run of their page: {out}; cost ${cost:.4f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    for a, b in zip(args[::2], args[1::2]):
        paired(a, b)
    for c in dict.fromkeys(args):
        real_rates(c)
