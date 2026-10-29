import json, numpy as np
from scipy.stats import spearmanr, pearsonr
v = json.load(open("/home/amit/.cache/vidyalai-scratch/p08/post/validate.json"))
rng = np.random.default_rng(0)
out = {}
def boot(x, y, f, n=2000):
    x, y = np.asarray(x), np.asarray(y); vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x)); vals.append(f(x[i], y[i])[0])
    return [f(x, y)[0], float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
allrows = []
for name in ["dallas", "brandenburg", "ioi"]:
    rows = v[name]["nodes"]; allrows += rows
for name, rows in [("dallas", v["dallas"]["nodes"]), ("brandenburg", v["brandenburg"]["nodes"]), ("ioi", v["ioi"]["nodes"]), ("pooled", allrows)]:
    a = np.array([r["actual"] for r in rows]); af = np.array([r["actual_frozen"] for r in rows])
    res = dict(n=len(rows),
        sp_abs_influence=boot(np.abs(a), [r["influence"] for r in rows], spearmanr),
        sp_abs_direct=boot(np.abs(a), [abs(r["pred_direct"]) for r in rows], spearmanr),
        sp_abs_act=boot(np.abs(a), [r["act"] for r in rows], spearmanr),
        pearson_total_vs_actual=boot(a, [r["pred_total"] for r in rows], pearsonr),
        pearson_total_vs_frozen=boot(af, [r["pred_total"] for r in rows], pearsonr),
        pearson_direct_vs_actual=boot(a, [r["pred_direct"] for r in rows], pearsonr),
        sign_agree_total=float(np.mean(np.sign(a) == np.sign([r["pred_total"] for r in rows]))),
        median_abs_actual=float(np.median(np.abs(a))), max_abs_actual=float(np.max(np.abs(a))),
        pearson_frozen_vs_actual=boot(a, af, pearsonr))
    if name != "pooled": res["target"] = v[name]["target"]; res["base"] = v[name]["base_demeaned_logit"]
    out[name] = res
    print(name, json.dumps(res))
s = v["swap"]
for k, x in s.items():
    if isinstance(x, dict) and "p_austin" in x: print(k, round(x["p_austin"], 4), round(x["p_sacramento"], 4), x["top5"][:3])
    elif isinstance(x, dict): print(k, x)
    else: print(k, np.round(x, 3).tolist() if k != "random6_x-2_p_austin" else (np.median(x), np.min(x), np.max(x)))
out["swap"] = s
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/p08/post/validate_summary.json", "w"), indent=1, default=float)
