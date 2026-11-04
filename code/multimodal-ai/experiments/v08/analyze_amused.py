import json, numpy as np
from collections import defaultdict
rows = [json.loads(l) for l in open("runs/amused_clip.jsonl")]
by = defaultdict(list)
for r in rows: by[r["steps"]].append(r)
rng = np.random.default_rng(0)
out = {}
for s in sorted(by):
    rs = by[s]
    P = sorted({r["prompt_id"] for r in rs})
    per = {p: np.mean([r["clip"] for r in rs if r["prompt_id"] == p]) for p in P}
    perm = {p: np.mean([r["clip_mismatch"] for r in rs if r["prompt_id"] == p]) for p in P}
    top1 = np.mean([r["clip_rank"] == 1 for r in rs])
    m = np.mean(list(per.values()))
    boots = [np.mean([per[p] for p in rng.choice(P, len(P))]) for _ in range(4000)]
    # paired difference vs 12 steps, per prompt
    out[s] = {"n": len(rs), "seeds": len({r['seed'] for r in rs}), "clip": m, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
              "mismatch": float(np.mean(list(perm.values()))), "top1": float(top1)}
    # Wilson for top1
    n = len(rs); k = sum(r["clip_rank"] == 1 for r in rs); z = 1.96
    c = (k + z*z/2) / (n + z*z); h = z * np.sqrt(k*(n-k)/n + z*z/4) / (n + z*z)
    out[s]["top1_ci"] = [c - h, c + h]
ref = by[12]
for s in sorted(by):
    P = sorted({r["prompt_id"] for r in by[s]})
    d = {p: np.mean([r["clip"] for r in by[s] if r["prompt_id"] == p]) - np.mean([r["clip"] for r in ref if r["prompt_id"] == p]) for p in P}
    boots = [np.mean([d[p] for p in rng.choice(P, len(P))]) for _ in range(4000)]
    out[s]["diff_vs12"] = [float(np.mean(list(d.values()))), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
for s, v in out.items():
    print(s, v["n"], round(v["clip"], 2), [round(x, 2) for x in v["ci"]], "mis", round(v["mismatch"], 2), "top1", round(v["top1"], 3), [round(x, 3) for x in v["top1_ci"]], "d12", [round(x, 2) for x in v["diff_vs12"]])
json.dump({str(k): v for k, v in out.items()}, open("runs/amused_clip_summary.json", "w"), indent=1)
