"""Check lab/arena.py's BT fit two ways: the MLE score equations (each model's actual score equals
its expected score) and an independent fit with statsmodels' binomial GLM on a +1/-1 design."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import arena
df = pd.read_csv("data/e07-votes.csv", usecols=["ts","model_a","model_b","winner"]).sort_values("ts").iloc[:30000]
a, b, y, models = arena.encode(df.model_a, df.model_b, df.winner)
M = len(models)
r = arena.fit_bt(a, b, y, M)
p = arena.win_prob(r[a], r[b])
act = np.bincount(a, y, M) + np.bincount(b, 1 - y, M)
exp = np.bincount(a, p, M) + np.bincount(b, 1 - p, M)
print("max |actual - expected score| over models:", np.abs(act - exp).max())
X = np.zeros((len(y), M)); X[np.arange(len(y)), a] = 1; X[np.arange(len(y)), b] = -1
X = X[:, 1:]  # model 0 is the reference
g = sm.GLM(y, X, family=sm.families.Binomial()).fit()
th = np.r_[0, g.params] / arena.ALPHA; th = th - th.mean() + 1000
print("max |arena - statsmodels| (Elo points):", np.abs(th - r).max())
rr, theta = arena.rao_kupper(a, b, y, M)
print("Rao-Kupper theta", theta, "max |RK - BT|", np.abs(rr - r).max())
n = np.bincount(a, minlength=M) + np.bincount(b, minlength=M)
d = np.abs(th - r); o = np.argsort(-d)[:4]
print([(models[i], int(n[i]), round(d[i], 3)) for i in o])
p2 = arena.win_prob(th[a], th[b])
exp2 = np.bincount(a, p2, M) + np.bincount(b, 1 - p2, M)
print("statsmodels: max |actual - expected|:", np.abs(act - exp2).max(), "converged", g.converged)
