"""Experiment 2: how many votes separate two models N Elo points apart?
(a) formula, (b) head-to-head and against-the-field simulation, (c) parametric simulation from
the fitted ratings on the real vote schedule, (d) real subsamples of the votes."""
import json, time, numpy as np, pandas as pd
from scipy.stats import norm
from common import load, arena
t0 = time.time()
df = load(); a, b, y, models = arena.encode(df.model_a, df.model_b, df.winner)
M, N = len(models), len(y)
tau = float((y == 0.5).mean())
out = {"tie_rate": tau, "N": N}
A = arena.ALPHA
# (a) per-vote variance of the score with ties counted as half, at p = 0.5
var1 = 0.25 * (1 - tau)
c_field = np.sqrt(var1) / 0.25 / A           # SE of one model's rating ~ c_field / sqrt(n_i)
out["c_field"] = c_field
def n_head2head(d, power=0.8):
    p = arena.win_prob(d, 0)
    return ((norm.ppf(0.975) + norm.ppf(power)) ** 2 * var1 / (p - 0.5) ** 2)
def n_field(d, power=0.8):  # votes per model, both models rated against a well-known field
    return (norm.ppf(0.975) + norm.ppf(power)) ** 2 * 2 * c_field ** 2 / d ** 2
gaps = [5, 10, 20, 30, 50, 100]
out["formula"] = {d: {"head2head": n_head2head(d), "field_per_model": n_field(d)} for d in gaps}
# (b) Monte Carlo head-to-head check: power at the formula's n
rng = np.random.default_rng(0)
sim = {}
for d in gaps:
    n = int(round(n_head2head(d))); p = arena.win_prob(d, 0); q = (p - tau / 2) / (1 - tau)
    S = 4000
    ties = rng.binomial(n, tau, S); wins = rng.binomial(n - ties, q)
    s = (wins + ties / 2) / n
    x2 = (wins + ties / 4) / n; se = np.sqrt((x2 - s ** 2) / n)
    sim[d] = {"n": n, "power": float((s - 1.96 * se > 0.5).mean() + (s + 1.96 * se < 0.5).mean())}
out["h2h_sim"] = sim
print("formula/sim", out["formula"], sim, time.time() - t0, flush=True)

# (c)+(d): separation of model pairs vs votes, real subsamples and parametric resimulation
full = arena.fit_bt(a, b, y, M)
iu = np.triu_indices(M, 1)
gap = np.abs(full[iu[0]] - full[iu[1]])
bins = [0, 10, 20, 40, 80, 1000]
def separated(boot):
    d = boot[:, iu[0]] - boot[:, iu[1]]
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return (lo > 0) | (hi < 0), np.sign(np.median(d, 0))
res = []
for frac in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
    for rep in range(3 if frac < 1 else 1):
        idx = rng.choice(N, int(frac * N), replace=False) if frac < 1 else np.arange(N)
        aa, bb, yy = a[idx], b[idx], y[idx]
        nv = np.bincount(aa, minlength=M) + np.bincount(bb, minlength=M)
        se_pred = c_field * np.sqrt(1 / nv[iu[0]] + 1 / nv[iu[1]])
        pw_pred = norm.cdf(gap / se_pred - 1.96)
        # real subsample
        bo = arena.bootstrap_bt(aa, bb, yy, M, n_boot=200, seed=rep)
        sep_real, _ = separated(bo)
        # parametric: same schedule, outcomes simulated from the full-data ratings and tie rate
        p = arena.win_prob(full[aa], full[bb]); q = np.clip((p - tau / 2) / (1 - tau), 0, 1)
        ys = np.where(rng.random(len(p)) < tau, 0.5, (rng.random(len(p)) < q).astype(float))
        bs = arena.bootstrap_bt(aa, bb, ys, M, n_boot=200, seed=rep + 100)
        sep_sim, sgn = separated(bs)
        wrong = sep_sim & (sgn != np.sign(full[iu[0]] - full[iu[1]]))
        for lo_, hi_ in zip(bins[:-1], bins[1:]):
            m = (gap >= lo_) & (gap < hi_)
            res.append({"frac": frac, "rep": rep, "votes": len(idx), "median_votes_per_model": float(np.median(nv)),
                        "bin": f"{lo_}-{hi_}", "pairs": int(m.sum()),
                        "sep_real": float(sep_real[m].mean()), "sep_sim": float(sep_sim[m].mean()),
                        "wrong_sim": float(wrong[m].mean()), "pred": float(pw_pred[m].mean())})
        print(frac, rep, time.time() - t0, flush=True)
out["separation"] = res
# measured SE vs votes per model on the full data (bootstrap from Experiment 1)
boot = np.load("/home/amit/.cache/vidyalai-scratch/e07/runs/exp1_boot.npy")
nv = np.bincount(a, minlength=M) + np.bincount(b, minlength=M)
out["se_full"] = {"votes": nv.tolist(), "se_boot": boot.std(0).tolist(), "se_pred": (c_field / np.sqrt(nv)).tolist()}
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/e07/runs/exp2.json", "w"), default=float)
print("done", time.time() - t0)
