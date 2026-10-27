# Part 7: does the last third of the val curve fall below a power law fitted to the middle? (CPU only)
import json, numpy as np
recs = [json.loads(l) for l in open("/home/amit/.cache/vidyalai-scratch/t07/runs/main88/log.jsonl")]
V = np.array([(r["step"], r["val_loss"]) for r in recs if "val_loss" in r])
lr = {r["step"]: r["lr"] for r in recs if "lr" in r}
s, v = V[:, 0], V[:, 1]
f = lambda x, E, A, a: E + A * x ** (-a)
for lo, hi in ((300, 2240), (500, 2240), (500, 1680)):
    m = (s >= lo) & (s <= hi)
    best = None
    for E in np.arange(2.0, min(v[m]) - 0.001, 0.0005):     # grid over the floor E, least squares in log space
        k, c = np.polyfit(np.log(s[m]), np.log(v[m] - E), 1)
        err = ((f(s[m], E, np.exp(c), -k) - v[m]) ** 2).sum()
        if best is None or err < best[0]: best = (err, (E, np.exp(c), -k))
    p = best[1]
    pred = f(3360, *p)
    print(f"fit {lo}-{hi}: E={p[0]:.3f} A={p[1]:.2f} alpha={p[2]:.3f}; predicted at 3360 {pred:.4f}, actual {v[-1]:.4f}, diff {v[-1]-pred:+.4f}; at 2800: pred {f(2800,*p):.4f} actual {v[s==2800][0]:.4f}")
    res = v - f(s, *p)
    print("   residuals (actual - fit) at", {int(k): round(float(r), 4) for k, r in zip(s, res) if k in (1000, 1500, 2000, 2240, 2500, 2800, 3000, 3200, 3360)})
# drop per 100 steps vs lr
for a in (1000, 1500, 2000, 2500, 3000, 3250):
    b = a + 100
    if b in set(s.astype(int)):
        print(f"steps {a}-{b}: val drop {v[s==a][0]-v[s==b][0]:.4f}, lr {lr[a]:.2e}")
