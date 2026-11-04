"""Statistics helpers for the Measuring LLMs series (Part 3, "The statistics of evals").

Conventions
-----------
* A score vector ``s`` has one entry per item (question): 0/1 for a single graded answer, or the
  mean of K graded samples of that item (a value in [0, 1]).
* A sample matrix ``X`` has shape (items, K): ``X[i, k]`` is 1 if sample k of item i was right.
* ``clusters`` is a sequence of labels, one per item (e.g. the MMLU subject).
* Two systems are compared on the same items, in the same order (paired by index).
* All intervals are two-sided 95% unless ``z``/``alpha`` say otherwise.

Formulas follow Miller (2024), "Adding Error Bars to Evals", arXiv:2411.00640, equation numbers
noted per function. Nothing here calls an API; numpy and scipy only.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy import stats as _st

Z95 = 1.959963984540054


# ---------------------------------------------------------------- one system, one score --------

def mean_se(s):
    """Mean and its Central-Limit-Theorem standard error, sqrt(Var(s)/n) (Miller eq. 1)."""
    s = np.asarray(s, float)
    return float(s.mean()), float(s.std(ddof=1) / sqrt(len(s)))


def clt_ci(s, z=Z95):
    """(mean, low, high): mean +/- z * SE_CLT (Miller eq. 3). Also called the Wald interval."""
    m, se = mean_se(s)
    return m, m - z * se, m + z * se


def wilson(successes, n, z=Z95):
    """Wilson score interval (low, high) for a proportion of 0/1 outcomes. Better than the CLT
    interval for small n or rates near 0 or 1 (Bowyer et al. 2025, arXiv:2503.01747)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def clustered_se(s, clusters):
    """Cluster-robust standard error of the mean (Miller eq. 4): the CLT variance plus every
    within-cluster cross product (s_i - mean)(s_j - mean), i != j, divided by n^2.

    Equivalent to sqrt(sum_c (sum_{i in c} (s_i - mean))^2) / n up to the (n-1)/n factor that
    eq. 4 keeps in its first term."""
    s = np.asarray(s, float)
    n, m = len(s), s.mean()
    r = s - m
    labels = np.asarray(clusters)
    cross = 0.0
    for c in np.unique(labels):
        rc = r[labels == c]
        cross += rc.sum() ** 2 - (rc ** 2).sum()
    var_clt = s.var(ddof=1) / n
    return float(sqrt(var_clt + cross / n ** 2))


def design_effect(s, clusters):
    """(SE_clustered / SE_CLT)^2: how many times more variance the clustering adds."""
    return (clustered_se(s, clusters) / mean_se(s)[1]) ** 2


def bootstrap_ci(s, clusters=None, stat=np.mean, n_boot=10_000, seed=0, level=0.95):
    """Percentile bootstrap interval (low, high) for stat(s). With ``clusters``, whole clusters are
    resampled (each with all its items), which respects within-cluster correlation."""
    rng = np.random.default_rng(seed)
    s = np.asarray(s, float)
    if clusters is None:
        idx = rng.integers(0, len(s), (n_boot, len(s)))
        vals = np.apply_along_axis(stat, 1, s[idx]) if stat is not np.mean else s[idx].mean(1)
    else:
        labels = np.asarray(clusters)
        groups = [s[labels == c] for c in np.unique(labels)]
        vals = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            vals[b] = stat(np.concatenate([groups[j] for j in pick]))
    a = (1 - level) / 2
    return tuple(float(v) for v in np.quantile(vals, [a, 1 - a]))


# ---------------------------------------------------------------- items vs sampling -------------

def variance_split(X):
    """Split the variance of single-sample scores into item and sampling parts (Miller sec. 3).

    X: (items, K) 0/1 matrix, K >= 2. Returns a dict with
      within  E[sigma_i^2], the average sampling variance of one answer to the same item
              (unbiased: p_i(1 - p_i) * K / (K - 1), averaged over items);
      between Var(x), the variance of the items' true success rates
              (variance of the per-item means minus within / K);
      total   within + between = Var of a single graded answer;
      share_sampling  within / total;
      se_K1, se_K     SE of the benchmark mean with 1 and with K samples per item:
                      sqrt((between + within / K) / n).
    """
    X = np.asarray(X, float)
    n, K = X.shape
    p = X.mean(1)
    within = float(np.mean(p * (1 - p) * K / (K - 1)))
    between = float(p.var(ddof=1) - within / K)
    total = between + within
    return dict(n=n, K=K, within=within, between=between, total=total,
                share_sampling=within / total,
                se_K1=sqrt((between + within) / n), se_K=sqrt((between + within / K) / n))


# ---------------------------------------------------------------- two systems, same items -------

def paired_diff(a, b, clusters=None, z=Z95):
    """Compare system B with system A on the same items (Miller eqs. 5-8).

    a, b: per-item scores (0/1 or K-sample means). Returns the mean difference b - a, its unpaired
    SE (as if the two runs used different items), its paired SE (Miller eq. 7), the item-level
    correlation, the matching 95% intervals and, with ``clusters``, the paired clustered SE (eq. 8).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = len(d)
    se_a, se_b = a.std(ddof=1) / sqrt(n), b.std(ddof=1) / sqrt(n)
    se_unp = sqrt(se_a ** 2 + se_b ** 2)
    se_pair = d.std(ddof=1) / sqrt(n)
    out = dict(n=n, diff=float(d.mean()), se_unpaired=float(se_unp), se_paired=float(se_pair),
               corr=float(np.corrcoef(a, b)[0, 1]),
               ci_unpaired=(float(d.mean() - z * se_unp), float(d.mean() + z * se_unp)),
               ci_paired=(float(d.mean() - z * se_pair), float(d.mean() + z * se_pair)))
    if clusters is not None:
        se_c = clustered_se(d, clusters)
        out.update(se_paired_clustered=se_c,
                   ci_paired_clustered=(float(d.mean() - z * se_c), float(d.mean() + z * se_c)))
    return out


def mcnemar(a, b):
    """McNemar's exact test for two 0/1 vectors on the same items. Only discordant items count:
    n01 = items A got wrong and B got right, n10 = the reverse. Under "no difference" each
    discordant item is a fair coin, so p = two-sided binomial test of n01 out of n01 + n10."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    p = _st.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(n01=n01, n10=n10, p=float(p))


def paired_t(a, b):
    """Paired t-test of mean(b - a) = 0 (scipy.stats.ttest_rel). Returns (t, p)."""
    r = _st.ttest_rel(b, a)
    return float(r.statistic), float(r.pvalue)


def paired_bootstrap(a, b, clusters=None, n_boot=10_000, seed=0):
    """Percentile 95% interval for mean(b - a), resampling items (or whole clusters) so each draw
    keeps A's and B's answers to the same item together."""
    d = np.asarray(b, float) - np.asarray(a, float)
    return bootstrap_ci(d, clusters=clusters, n_boot=n_boot, seed=seed)


def sign_flip_p(a, b, n_perm=20_000, seed=0):
    """Two-sided permutation p-value for mean(b - a): flip the sign of each item's difference at
    random (what "no difference" would allow) and count draws at least as extreme."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, float) - np.asarray(a, float)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], (n_perm, len(d)))
    return float(((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n_perm + 1))


# ---------------------------------------------------------------- power -------------------------

def power_inputs(XA, XB):
    """omega^2, sigma_A^2, sigma_B^2 for Miller's eqs. 9-10 from two (items, K) 0/1 matrices on the
    same items. omega^2 = Var(x_A) + Var(x_B) - 2 Cov(x_A, x_B) is the variance of the true
    per-item differences; sigma^2 is each system's average within-item sampling variance."""
    XA, XB = np.asarray(XA, float), np.asarray(XB, float)
    va, vb = variance_split(XA), variance_split(XB)
    pa, pb = XA.mean(1), XB.mean(1)
    cov = float(np.cov(pa, pb, ddof=1)[0, 1])  # sampling noise of A and B is independent
    omega2 = va["between"] + vb["between"] - 2 * cov
    return dict(omega2=omega2, sigma2_a=va["within"], sigma2_b=vb["within"])


def sample_size(delta, omega2, sigma2_a=0.0, sigma2_b=0.0, k_a=1, k_b=1, alpha=0.05, power=0.8):
    """Items needed to detect a true gap ``delta`` (Miller eq. 9):
    n = (z_{alpha/2} + z_beta)^2 (omega^2 + sigma_A^2/K_A + sigma_B^2/K_B) / delta^2."""
    zz = _st.norm.ppf(1 - alpha / 2) + _st.norm.ppf(power)
    return float(zz ** 2 * (omega2 + sigma2_a / k_a + sigma2_b / k_b) / delta ** 2)


def mde(n, omega2, sigma2_a=0.0, sigma2_b=0.0, k_a=1, k_b=1, alpha=0.05, power=0.8):
    """Minimum detectable effect for n items (Miller eq. 10), the inverse of sample_size."""
    zz = _st.norm.ppf(1 - alpha / 2) + _st.norm.ppf(power)
    return float(zz * sqrt((omega2 + sigma2_a / k_a + sigma2_b / k_b) / n))


def unpaired_sample_size(delta, var_a, var_b, alpha=0.05, power=0.8):
    """Items per system for two independent item sets: (z + z)^2 (Var_A + Var_B) / delta^2, where
    Var is the per-item score variance (p(1 - p) for single 0/1 answers)."""
    zz = _st.norm.ppf(1 - alpha / 2) + _st.norm.ppf(power)
    return float(zz ** 2 * (var_a + var_b) / delta ** 2)


def resampled_power(XA, XB, n, k=1, paired=True, n_sim=4000, seed=0, z=Z95):
    """Power checked by resampling the real answers instead of a formula.

    Each simulated experiment draws n items with replacement (B on the same items if ``paired``,
    on an independent draw if not), takes k of each item's recorded samples per system (with
    replacement), and tests the mean difference with a z-test at the 5% level. Returns the share of
    experiments that find B - A significant in the direction of the full-data difference."""
    rng = np.random.default_rng(seed)
    XA, XB = np.asarray(XA, float), np.asarray(XB, float)
    N, KA = XA.shape
    KB = XB.shape[1]
    sign = np.sign(XB.mean() - XA.mean())
    hits = 0
    for _ in range(n_sim):
        ia = rng.integers(0, N, n)
        ib = ia if paired else rng.integers(0, N, n)
        a = XA[ia[:, None], rng.integers(0, KA, (n, k))].mean(1)
        b = XB[ib[:, None], rng.integers(0, KB, (n, k))].mean(1)
        if paired:
            d = b - a
            se = d.std(ddof=1) / sqrt(n)
        else:
            se = sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
        diff = b.mean() - a.mean()
        hits += se > 0 and diff / se * sign > z
    return hits / n_sim


# ---------------------------------------------------------------- many comparisons --------------

def bonferroni(pvals, alpha=0.05):
    """Reject p_i when p_i <= alpha / m. Controls the chance of any false positive (FWER)."""
    p = np.asarray(pvals, float)
    return p <= alpha / len(p)


def holm(pvals, alpha=0.05):
    """Holm's step-down method: as safe as Bonferroni for the FWER, never less powerful.
    Sort p ascending; reject while p_(j) <= alpha / (m - j) for j = 0, 1, ...; stop at the first
    failure."""
    p = np.asarray(pvals, float)
    m = len(p)
    reject = np.zeros(m, bool)
    for j, i in enumerate(np.argsort(p)):
        if p[i] > alpha / (m - j):
            break
        reject[i] = True
    return reject


def benjamini_hochberg(pvals, q=0.05):
    """Benjamini-Hochberg: controls the expected share of false discoveries among rejections (FDR).
    Reject the k smallest p-values, k the largest j with p_(j) <= q j / m (j from 1)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, m + 1) / m
    reject = np.zeros(m, bool)
    if ok.any():
        reject[order[: np.nonzero(ok)[0].max() + 1]] = True
    return reject
