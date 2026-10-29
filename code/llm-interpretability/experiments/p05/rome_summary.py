"""Hase et al. (2023)-style analysis of the edits (CPU only). Writes post/rome.json."""
import json
import numpy as np

ROOT = "/home/amit/.cache/vidyalai-scratch/p05"
R = np.load(f"{ROOT}/runs/rome.npz")
T = np.load(f"{ROOT}/runs/trace_gpt2.npz")
assert np.array_equal(R["known"], T["gn_ids"])
clean_p = T["clean_m"][0, T["gn_ids"]]                       # [n]
corr_p = T["gn_corrupt"][:, 0]
trace = (T["gn_mlp_max_over_tokens"] - corr_p[:, None]) / (clean_p - corr_p)[:, None]   # [n, L] fractional
LAY = R['layers']
trace = trace[:, LAY]
n, L = trace.shape

def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X]) if X is not None else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()

out = {"n_facts": int(n), "layers": LAY.tolist(), "trace_by_layer": np.round(trace.mean(0), 4).tolist(),
       "trace_peak_layer_hist": np.bincount(trace.argmax(1), minlength=L).tolist()}
layer_onehot = np.eye(L)[np.repeat(np.arange(L)[None], n, 0).ravel()][:, 1:]          # [n*L, L-1]
tr = trace.ravel()
for k in ("rewrite", "para", "neighborhood"):
    y = R[k].ravel()
    ok = ~np.isnan(y)
    rng = np.random.default_rng(0)
    bl = [np.nanmean(R[k][rng.integers(0, n, n)], 0) for _ in range(1000)]
    out[k] = {"by_layer": np.round(np.nanmean(R[k], 0), 4).tolist(),
              "lo": np.round(np.nanpercentile(bl, 2.5, 0), 4).tolist(), "hi": np.round(np.nanpercentile(bl, 97.5, 0), 4).tolist(),
              "r2_layer": float(r2(y[ok], layer_onehot[ok])), "r2_trace": float(r2(y[ok], tr[ok])),
              "r2_both": float(r2(y[ok], np.column_stack([layer_onehot, layer_onehot * tr[:, None], tr])[ok])),
              "corr_within_layer": [float(np.corrcoef(trace[:, l][~np.isnan(R[k][:, l])], R[k][:, l][~np.isnan(R[k][:, l])])[0, 1]) for l in range(L)]}
out["flip_by_layer"] = np.round(R["flip"].mean(0), 3).tolist()
out["delta_ratio_by_layer"] = np.round(np.nanmean(R["delta_ratio"], 0), 3).tolist()
# per fact: is the best edit layer the tracing peak layer?
best = np.nanargmax(R["para"], 1)
out["best_para_layer_equals_trace_peak"] = float((best == trace.argmax(1)).mean())
out["best_para_layer_hist"] = np.bincount(best, minlength=L).tolist()
json.dump(out, open(f"{ROOT}/post/rome.json", "w"))
print(json.dumps({k: v for k, v in out.items()}, indent=None)[:3000])
