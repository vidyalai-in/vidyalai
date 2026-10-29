# True/false city statements: per-layer probes with city-disjoint 5-fold cross-validation.
# Runs on CPU (1,134 statements). Usage: probe_truth.py gpt2|qwen
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.probes import load_layer, train_probe, mass_mean_direction, bootstrap_ci
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(4)
R = "/home/amit/.cache/vidyalai-scratch/p03/runs"
which = sys.argv[1]
A = f"{R}/acts/{which}"
info = json.load(open(f"{A}/truth_cities/info.json"))
items = json.load(open(f"{R}/truth_data.json"))
y = np.array([it["label"] for it in items])
city = np.array([it["city"] for it in items])
folds = list(GroupKFold(n_splits=5).split(y, y, groups=city))
lp = np.array(json.load(open(f"{A}/truth_logprob.json")))
# does the model itself prefer the true country? (paired, per city)
lp_true = {c: v for c, v, l in zip(city, lp, y) if l == 1}
lp_false = {c: v for c, v, l in zip(city, lp, y) if l == 0}
knows = {c: lp_true[c] > lp_false[c] for c in lp_true}
k_item = np.array([knows[c] for c in city])

def oof(fn):
    pred = np.zeros(len(y), dtype=int)
    for tr, te in folds:
        pred[te] = fn(tr, te)
    return pred

# baselines without the model's activations
texts = [f"The city of {it['city']} is in {it['country']}." for it in items]
def bow(tr, te):
    v = CountVectorizer(token_pattern=r"[^ .]+").fit([texts[i] for i in tr])
    m = LogisticRegression(max_iter=2000).fit(v.transform([texts[i] for i in tr]), y[tr])
    return m.predict(v.transform([texts[i] for i in te]))
def lpbase(tr, te):
    m = LogisticRegression().fit(lp[tr, None], y[tr]); return m.predict(lp[te, None])
base = {"n_statements": len(y), "n_cities": len(set(city)), "n_countries": len({it['true_country'] for it in items}),
        "model_prefers_true_country": float(np.mean(list(knows.values()))),
        "bow": bootstrap_ci(oof(bow) == y, city), "logprob_threshold": bootstrap_ci(oof(lpbase) == y, city)}
print(json.dumps(base), flush=True)

P = dict(epochs=200, batch_size=128, lr=1e-3, weight_decay=1e-2, device="cpu")
out = {"base": base, "layers": {}}
t0 = time.time()
rng = np.random.default_rng(0)
for l in range(info["n_layers"] + 1):
    X = load_layer(f"{A}/truth_cities", l); Xalt = load_layer(f"{A}/truth_alt", l)
    res = {}
    for kind in ["linear", "mlp"]:
        pred = np.zeros(len(y), int); palt = np.zeros(len(y), int); pshuf = np.zeros(len(y), int)
        for k, (tr, te) in enumerate(folds):
            r = train_probe(X[tr], y[tr], np.concatenate([X[te], Xalt[te]]), np.concatenate([y[te], y[te]]),
                            kind=kind, n_classes=2, hidden=256, seed=k, **P)
            pred[te], palt[te] = r["pred"][:len(te)], r["pred"][len(te):]
            ys = y[tr].copy(); rng.shuffle(ys)
            pshuf[te] = train_probe(X[tr], ys, X[te], y[te], kind=kind, n_classes=2, hidden=256, seed=k, **P)["pred"]
        ok = pred == y
        res[kind] = {"acc": bootstrap_ci(ok, city), "alt_template": bootstrap_ci(palt == y, city),
                     "shuffled_labels": bootstrap_ci(pshuf == y, city),
                     "acc_known": float(ok[k_item].mean()), "acc_unknown": float(ok[~k_item].mean())}
    # mass-mean probe: project on (mean true - mean false), threshold at the midpoint of the class means
    pm = np.zeros(len(y), int); pma = np.zeros(len(y), int)
    for tr, te in folds:
        th = mass_mean_direction(X[tr], y[tr]); mid = 0.5 * (X[tr][y[tr] == 1].mean(0) + X[tr][y[tr] == 0].mean(0)) @ th
        pm[te] = (X[te] @ th > mid).astype(int); pma[te] = (Xalt[te] @ th > mid).astype(int)
    res["mass_mean"] = {"acc": bootstrap_ci(pm == y, city), "alt_template": bootstrap_ci(pma == y, city)}
    out["layers"][l] = res
    print(l, " ".join(f"{k} {v['acc'][0]:.3f}" for k, v in res.items()),
          f"| alt lin {res['linear']['alt_template'][0]:.3f} shuf {res['linear']['shuffled_labels'][0]:.3f} | {time.time()-t0:.0f}s", flush=True)
json.dump(out, open(f"{R}/truth_{which}.json", "w"), indent=1)
