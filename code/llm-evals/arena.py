"""Arena ratings for the Measuring LLMs series (Part 7, "Arenas and Elo rankings").

Conventions
-----------
* A vote set is three arrays of the same length: ``a`` and ``b`` (integer model ids, the left and
  right model) and ``y``, the score of the left model: 1.0 win, 0.5 tie, 0.0 loss.
* Ratings are on the Elo scale: a gap of ``d`` points means the stronger model wins with
  probability ``1 / (1 + 10 ** (-d / 400))``. Fitted ratings are shifted so their mean is 1000,
  which is also what online Elo started every model at (its updates are zero-sum).
* The Bradley-Terry fit and the style-controlled fit follow LMArena's open code
  (FastChat ``fastchat/serve/monitor/rating_systems.py``): ties count as half a win and half a
  loss, L-BFGS on the log-likelihood, style features ``(f_a - f_b) / (f_a + f_b + 1)`` standardized.

numpy and scipy only; nothing here calls an API.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

ALPHA = math.log(10) / 400  # Elo points -> natural log-odds
OUTCOME = {"model_a": 1.0, "model_b": 0.0, "tie": 0.5, "both_bad": 0.5, "tie (bothbad)": 0.5}


def encode(model_a, model_b, winner, models=None):
    """Names and winner labels -> (a, b, y, models). ``models`` fixes the id order if given."""
    if models is None:
        models = sorted(set(model_a) | set(model_b))
    idx = {m: i for i, m in enumerate(models)}
    a = np.array([idx[m] for m in model_a])
    b = np.array([idx[m] for m in model_b])
    y = np.array([OUTCOME[w] for w in winner], float)
    return a, b, y, list(models)


def win_prob(r_a, r_b):
    """Chance that a model rated r_a beats one rated r_b (Elo scale)."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


# ---------------------------------------------------------------- online Elo --------------------

def online_elo(a, b, y, n_models, k=4.0, order=None, init=1000.0):
    """Classic online Elo: after each vote, both ratings move by k * (result - expected result).
    ``order`` is a permutation of vote indices (default: as given). Returns final ratings."""
    r = np.full(n_models, init)
    idx = np.arange(len(y)) if order is None else order
    for i in idx:
        ia, ib = a[i], b[i]
        e = 1.0 / (1.0 + 10.0 ** ((r[ib] - r[ia]) / 400.0))
        u = k * (y[i] - e)
        r[ia] += u
        r[ib] -= u
    return r


def online_elo_many(a, b, y, n_models, orders, k=4.0, init=1000.0):
    """Online Elo for many vote orders at once. ``orders``: (n_orders, n_votes) index array.
    Returns (n_orders, n_models) final ratings. Same arithmetic as ``online_elo``, vectorized
    across orders (FastChat's ``fit_vectorized_elo``)."""
    orders = np.asarray(orders)
    s = orders.shape[0]
    r = np.full((s, n_models), init)
    rows = np.arange(s)
    for col in orders.T:
        ia, ib = a[col], b[col]
        e = expit(ALPHA * (r[rows, ia] - r[rows, ib]))
        u = k * (y[col] - e)
        r[rows, ia] += u
        r[rows, ib] -= u
    return r


# ---------------------------------------------------------------- Bradley-Terry -----------------

def _collapse(a, b, y, w=None):
    """Unique (a, b, y) rows with summed weights: the BT likelihood only needs the counts."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    key = np.stack([a, b, np.round(y * 2).astype(int)], 1)
    u, inv = np.unique(key, axis=0, return_inverse=True)
    wsum = np.bincount(inv.ravel(), weights=w, minlength=len(u))
    return u[:, 0], u[:, 1], u[:, 2] / 2.0, wsum


def _bt_nll(theta, a, b, y, w):
    z = theta[a] - theta[b]
    nll = -(w * (y * log_expit(z) + (1 - y) * log_expit(-z))).sum()
    g = w * (expit(z) - y)
    grad = np.bincount(a, g, len(theta)) - np.bincount(b, g, len(theta))
    return nll, grad


def fit_bt(a, b, y, n_models, w=None, ties="half"):
    """Bradley-Terry maximum likelihood. ``ties``: "half" (LMArena: a tie is half a win and half
    a loss) or "drop" (ties removed). Returns ratings on the Elo scale, mean 1000."""
    a, b, y = np.asarray(a), np.asarray(b), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if ties == "drop":
        keep = y != 0.5
        a, b, y, w = a[keep], b[keep], y[keep], w[keep]
    ua, ub, uy, uw = _collapse(a, b, y, w)
    res = minimize(_bt_nll, np.zeros(n_models), args=(ua, ub, uy, uw), jac=True,
                   method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-9})
    th = res.x / ALPHA
    return th - th.mean() + 1000.0


def bootstrap_bt(a, b, y, n_models, n_boot=1000, seed=0, ties="half", clusters=None):
    """Percentile-bootstrap BT ratings: resample votes (or whole ``clusters``, e.g. prompts) with
    replacement and refit. Returns (n_boot, n_models)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty((n_boot, n_models))
    if clusters is None:
        # resampling votes = multinomial counts over the distinct (a, b, outcome) rows
        ua, ub, uy, uw = _collapse(np.asarray(a), np.asarray(b), np.asarray(y, float))
        for t in range(n_boot):
            w = rng.multinomial(n, uw / uw.sum())
            out[t] = fit_bt(ua, ub, uy, n_models, w=w, ties=ties)
    else:
        cl, inv = np.unique(clusters, return_inverse=True)
        for t in range(n_boot):
            cw = np.bincount(rng.integers(0, len(cl), len(cl)), minlength=len(cl))
            out[t] = fit_bt(a, b, y, n_models, w=cw[inv], ties=ties)
    return out


def rao_kupper(a, b, y, n_models):
    """Rao-Kupper (1967) Bradley-Terry with an explicit tie parameter theta >= 1:
    P(a beats b) = p_a / (p_a + theta p_b),  P(tie) = (theta^2 - 1) p_a p_b / ((p_a + theta p_b)(theta p_a + p_b)).
    Returns (ratings on the Elo scale with mean 1000, theta)."""
    ua, ub, uy, uw = _collapse(np.asarray(a), np.asarray(b), np.asarray(y, float))

    def nll(p):
        s, lt = p[:-1], math.log1p(math.exp(p[-1]))  # log theta = softplus(...) > 0
        d = s[ua] - s[ub]
        lwin = log_expit(d - lt)          # log p_a / (p_a + theta p_b)
        lloss = log_expit(-d - lt)        # log p_b / (theta p_a + p_b)
        ltie = np.log(np.expm1(2 * lt)) + lwin + lloss
        ll = np.where(uy == 1.0, lwin, np.where(uy == 0.0, lloss, ltie))
        return -(uw * ll).sum()

    res = minimize(nll, np.r_[np.zeros(n_models), 0.0], method="L-BFGS-B",
                   options={"maxiter": 5000})
    s = res.x[:-1] / ALPHA
    return s - s.mean() + 1000.0, math.exp(math.log1p(math.exp(res.x[-1])))


# ---------------------------------------------------------------- style control -----------------

def style_features(fa, fb):
    """LMArena style features: (f_a - f_b) / (f_a + f_b + 1) per column, then standardized.
    fa, fb: (n_votes, n_features) counts (tokens, headers, list items, bold spans)."""
    fa, fb = np.asarray(fa, float), np.asarray(fb, float)
    d = (fa - fb) / (fa + fb + 1.0)
    return (d - d.mean(0)) / d.std(0)


def fit_style_bt(a, b, y, n_models, Z, reg=0.5, w=None):
    """Bradley-Terry plus additive style terms: P(a wins) = sigmoid(theta_a - theta_b + Z gamma).
    L2 penalty reg/2 * |params|^2 on the natural scale, as in LMArena's code.
    Returns (ratings on the Elo scale with mean 1000, gamma in log-odds per standard deviation)."""
    a, b, y, Z = np.asarray(a), np.asarray(b), np.asarray(y, float), np.asarray(Z, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    m = n_models

    def f(p):
        th, ga = p[:m], p[m:]
        z = th[a] - th[b] + Z @ ga
        nll = -(w * (y * log_expit(z) + (1 - y) * log_expit(-z))).sum() + 0.5 * reg * p @ p
        g = w * (expit(z) - y)
        grad = reg * p
        grad[:m] += np.bincount(a, g, m) - np.bincount(b, g, m)
        grad[m:] += Z.T @ g
        return nll, grad

    res = minimize(f, np.zeros(m + Z.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-8})
    th = res.x[:m] / ALPHA
    return th - th.mean() + 1000.0, res.x[m:]


# ---------------------------------------------------------------- ranks and agreement -----------

def ranks(r):
    """Ordinal rank, 1 = highest rating."""
    o = np.argsort(-np.asarray(r))
    rk = np.empty(len(r), int)
    rk[o] = np.arange(1, len(r) + 1)
    return rk


def arena_rank(lo, hi):
    """LMArena-style rank: 1 + the number of models whose lower bound is above this model's
    upper bound. Models whose intervals overlap share a rank."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    return 1 + (lo[None, :] > hi[:, None]).sum(1)


def kendall_tau(x, y):
    """Kendall's tau-b between two score vectors (scipy)."""
    from scipy.stats import kendalltau
    return kendalltau(x, y).statistic


def pair_separated(boot, i, j, level=0.95):
    """True if the bootstrap interval of rating(i) - rating(j) excludes zero."""
    d = boot[:, i] - boot[:, j]
    q = (1 - level) / 2
    lo, hi = np.quantile(d, [q, 1 - q])
    return lo > 0 or hi < 0
