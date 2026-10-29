"""Summaries of the IOI runs (CPU only). Writes post/ioi.json."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import patching as P

ROOT = "/home/amit/.cache/vidyalai-scratch/p05"
A = dict(np.load(f"{ROOT}/runs/ioi_A.npz"))
try:
    A.update({k: v for k, v in np.load(f"{ROOT}/runs/ioi_B.npz").items() if k.startswith("path_")})
except FileNotFoundError:
    pass
base = json.load(open(f"{ROOT}/post/ioi_base_A.json"))
c, s, abc, gn = (A[f"base_{k}"] for k in ("clean", "str", "abc", "gn"))     # [3, n]: LD, P(IO), P(S)
n = c.shape[-1]
CLASSES = {"name mover": [(9, 9), (9, 6), (10, 0)], "negative name mover": [(10, 7), (11, 10)],
           "S-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)], "induction": [(5, 5), (6, 9)],
           "duplicate token": [(0, 1), (3, 0)], "previous token": [(2, 2), (4, 11)]}
r3 = lambda x: np.round(np.asarray(x, dtype=float), 3).tolist()
out = {"n": n, "baseline": base["baseline"], "nm_attention": base["nm_attention"], "classes": {k: [f"{a}.{b}" for a, b in v] for k, v in CLASSES.items()}}

def norm(arr, metric, clean, corr, boot=False):
    """arr [..., 3, n] -> normalized effect [...] (and CIs)."""
    x = arr[..., metric, :]
    if boot:
        return P.bootstrap_normalized(x, clean[metric], corr[metric], n_boot=1000)
    return P.normalized(x, clean[metric], corr[metric])

# 1. residual stream heatmaps
for corr_name, corr in (("str", s), ("abc", abc)):
    est, lo, hi = norm(A[f"resid_{corr_name}"], 0, c, corr, boot=True)
    out[f"resid_{corr_name}"] = {"est": r3(est), "lo": r3(lo), "hi": r3(hi)}
out["roles"] = base["roles"]
# 2. attention vs MLP
for where in ("all", "END"):
    est, lo, hi = norm(A[f"layers_{where}"], 0, c, s, boot=True)
    out[f"layers_{where}"] = {"attn": r3(est[0]), "mlp": r3(est[1]), "attn_lo": r3(lo[0]), "attn_hi": r3(hi[0]),
                              "mlp_lo": r3(lo[1]), "mlp_hi": r3(hi[1])}
# 3. heads
corrs = {"den_str": s, "den_gn": gn, "den_abc": abc, "noi_str": s, "den_str_END": s}
out["heads"] = {}
for name, corr in corrs.items():
    arr = A[f"heads_{name}"]
    for mi, mname in ((0, "ld"), (1, "p_io")):
        if name.startswith("noi"):
            est, lo, hi = norm(arr, mi, c, corr, boot=True)
            est, lo, hi = 1 - est, 1 - hi, 1 - lo                       # share of the gap destroyed
        else:
            est, lo, hi = norm(arr, mi, c, corr, boot=True)
        mu, sd = est.mean(), est.std()
        det = [f"{l}.{h}" for l in range(12) for h in range(12) if abs(est[l, h] - mu) > 2 * sd]
        cls = {k: [f"{a}.{b}" for a, b in v if f"{a}.{b}" in det] for k, v in CLASSES.items()}
        out["heads"][f"{name}_{mname}"] = {"est": r3(est), "lo": r3(lo), "hi": r3(hi), "mean": float(mu), "sd": float(sd),
                                           "detected": det, "detected_by_class": cls}
# 4. path patching: relative change in logit difference, (LD_pp - LD_clean) / LD_clean (Wang et al.)
for tgt in ("logits", "nm_q", "si_v"):
    if f"path_{tgt}" not in A:
        continue
    x = A[f"path_{tgt}"][..., 0, :]                                     # [L, H, n]
    rel = (x.mean(-1) - c[0].mean()) / c[0].mean()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, (1000, n))
    bs = (np.nanmean(x[..., idx], -1) - c[0][idx].mean(-1)) / c[0][idx].mean(-1)     # [L, H, B]
    out[f"path_{tgt}"] = {"est": r3(rel), "lo": r3(np.percentile(bs, 2.5, -1)), "hi": r3(np.percentile(bs, 97.5, -1))}
    flat = [(rel[l, h], f"{l}.{h}") for l in range(12) for h in range(12) if not np.isnan(rel[l, h])]
    out[f"path_{tgt}"]["top"] = [(h, round(float(v), 3)) for v, h in sorted(flat, key=lambda t: -abs(t[0]))[:10]]
json.dump(out, open(f"{ROOT}/post/ioi.json", "w"))
# printout
print("baseline clean LD", r3(base["baseline"]["clean"]["ld"]), "STR", r3(base["baseline"]["str"]["ld"]))
print("resid STR (rows=layers, cols=roles)", out["roles"]); print(np.round(np.array(out["resid_str"]["est"]), 2))
print("resid ABC"); print(np.round(np.array(out["resid_abc"]["est"]), 2))
for w in ("all", "END"):
    print(w, "attn", out[f"layers_{w}"]["attn"]); print(w, "mlp ", out[f"layers_{w}"]["mlp"])
for k, v in out["heads"].items():
    e = np.array(v["est"]); top = np.argsort(-np.abs(e).ravel())[:8]
    print(k, "top", [(f"{i//12}.{i%12}", round(e.ravel()[i], 2)) for i in top], "| by class", v["detected_by_class"])
for tgt in ("logits", "nm_q", "si_v"):
    if f"path_{tgt}" in out:
        print("path", tgt, out[f"path_{tgt}"]["top"])
