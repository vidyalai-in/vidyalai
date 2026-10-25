"""Part 11: scoring repeated agent runs.

Every task runs n times; c of those runs succeed. From the (n, c) pairs this module computes:

  pass_at_k   chance that at least one of k runs succeeds (unbiased estimator, Chen et al. 2021)
  pass_hat_k  chance that all k runs succeed (pass^k, Yao et al. 2024)
  wilson      95% Wilson interval for a rate
  task_bootstrap   confidence interval that resamples whole tasks (runs of one task are not independent)
  paired_power     how many tasks and repeats separate two configurations, by simulation

Nothing here calls the API.
"""

from math import comb, sqrt

import numpy as np


def pass_at_k(n, c, k):
    """P(at least one of k runs succeeds), estimated from c successes in n runs (needs k <= n)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n, c, k):
    """P(all k runs succeed), estimated from c successes in n runs (needs k <= n)."""
    return comb(c, k) / comb(n, k)


def suite(results, k, metric):
    """Average a per-task metric over tasks. results: {task: [True, False, ...]} with >= k runs each."""
    return float(np.mean([metric(len(r), sum(r), k) for r in results.values()]))


def wilson(successes, n, z=1.96):
    """95% Wilson score interval for a proportion, as (low, high)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def task_bootstrap(results, stat, n_boot=10_000, seed=0):
    """95% interval for stat(results) when whole tasks, with all their runs, are resampled."""
    rng = np.random.default_rng(seed)
    tasks = list(results)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(tasks), len(tasks))
        vals.append(stat({i: results[tasks[j]] for i, j in enumerate(pick)}))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def paired_power(p_a, p_b, n_tasks, repeats, n_sim=4000, seed=0):
    """Share of simulated experiments in which configuration B beats A at the 5% level (two-sided).

    p_a, p_b: per-task success probabilities measured on the same tasks (paired by index).
    Each simulated experiment draws n_tasks tasks with replacement, runs each `repeats` times per
    configuration (Bernoulli draws), and tests the mean per-task difference with a paired z-test.
    """
    rng = np.random.default_rng(seed)
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    wins = 0
    for _ in range(n_sim):
        idx = rng.integers(0, len(p_a), n_tasks)
        a = rng.binomial(repeats, p_a[idx]) / repeats
        b = rng.binomial(repeats, p_b[idx]) / repeats
        d = b - a
        se = d.std(ddof=1) / sqrt(n_tasks)
        if se == 0:
            wins += d.mean() > 0
        else:
            wins += abs(d.mean() / se) > 1.96 and d.mean() > 0
    return wins / n_sim
