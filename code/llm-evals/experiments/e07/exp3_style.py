"""Experiment 3: style control (length and markdown features in the BT model), prompt-mix
subsets, and session-clustered intervals, on the same votes as Experiment 1."""
import json, time, numpy as np, pandas as pd
from common import load, arena
t0 = time.time()
F = ["tok", "hdr", "list", "bold"]
df = load(["session", "order", "lang", "is_code", "turns", "n_criteria"] + [f"{f}_{s}" for f in F for s in "ab"])
a, b, y, models = arena.encode(df.model_a, df.model_b, df.winner)
M, N = len(models), len(y)
fa, fb = df[[f + "_a" for f in F]].values, df[[f + "_b" for f in F]].values
Z = arena.style_features(fa, fb)
out = {"models": models, "N": N}
bt = arena.fit_bt(a, b, y, M); out["bt"] = bt.tolist()
for name, cols in {"both": [0, 1, 2, 3], "length": [0], "markdown": [1, 2, 3]}.items():
    r, g = arena.fit_style_bt(a, b, y, M, Z[:, cols]); out[f"sc_{name}"] = r.tolist(); out[f"gamma_{name}"] = g.tolist()
print("fits", time.time() - t0, flush=True)
# bootstrap the full style model (votes resampled)
rng = np.random.default_rng(0); B = 100
R, G = np.empty((B, M)), np.empty((B, 4))
for t in range(B):
    w = np.bincount(rng.integers(0, N, N), minlength=N)
    R[t], G[t] = arena.fit_style_bt(a, b, y, M, Z, w=w)
out["sc_lo"], out["sc_hi"] = np.percentile(R, 2.5, 0).tolist(), np.percentile(R, 97.5, 0).tolist()
out["gamma_lo"], out["gamma_hi"] = np.percentile(G, 2.5, 0).tolist(), np.percentile(G, 97.5, 0).tolist()
print("style boot", time.time() - t0, flush=True)
# per-model style summary: mean tokens and markdown elements per answer
tok = np.bincount(a, fa[:, 0], M) + np.bincount(b, fb[:, 0], M)
nv = np.bincount(a, minlength=M) + np.bincount(b, minlength=M)
md = np.bincount(a, fa[:, 1:].sum(1), M) + np.bincount(b, fb[:, 1:].sum(1), M)
out["mean_tokens"], out["mean_markdown"] = (tok / nv).tolist(), (md / nv).tolist()
# raw: does the longer answer win more often? (non-tie votes with both answers non-empty)
L = fa[:, 0] - fb[:, 0]; dec = (y != 0.5) & (L != 0)
longer_wins = ((L > 0) & (y == 1)) | ((L < 0) & (y == 0))
out["longer_wins_share"] = [int(longer_wins[dec].sum()), int(dec.sum())]
# prompt mix
out["mix"] = {"lang": df.lang.value_counts().head(10).to_dict(), "code": int(df.is_code.sum()),
              "multi_turn": int((df.turns > 1).sum()), "hard6": int((df.n_criteria >= 6).sum()),
              "sessions": int(df.session.nunique()), "first_in_session": int((df.order == 1).sum())}
subsets = {"english": df.lang == "en", "non_english": df.lang != "en", "polish": df.lang == "pl",
           "code": df.is_code == 1, "no_code": df.is_code == 0, "hard6": df.n_criteria >= 6,
           "first_turn_only": df.turns == 1}
out["subsets"] = {}
for k, m in subsets.items():
    m = m.values
    r = arena.fit_bt(a[m], b[m], y[m], M)
    nvk = np.bincount(a[m], minlength=M) + np.bincount(b[m], minlength=M)
    out["subsets"][k] = {"n": int(m.sum()), "bt": r.tolist(), "votes": nvk.tolist()}
print("subsets", time.time() - t0, flush=True)
# session-clustered bootstrap vs vote bootstrap (Experiment 1 used votes)
bc = arena.bootstrap_bt(a, b, y, M, n_boot=300, seed=0, clusters=df.session.values)
bv = arena.bootstrap_bt(a, b, y, M, n_boot=300, seed=0)
out["se_session"], out["se_vote"] = bc.std(0).tolist(), bv.std(0).tolist()
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/e07/runs/exp3.json", "w"))
print("done", time.time() - t0)
