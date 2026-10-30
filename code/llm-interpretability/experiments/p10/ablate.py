# Mean-ablate the top-k induction heads (ranked at that checkpoint on a separate selection set) and
# random sets of k heads; measure second-copy loss on repeated tokens and the ICL score on 500 windows.
# Usage: ablate.py RUN step,step,...  (resumable; one JSON per checkpoint)
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import dynamics as D
from lab.heads import top_heads
run = sys.argv[1]; steps = [int(s) for s in sys.argv[2].split(",")]
KS = [int(k) for k in sys.argv[3].split(",")] if len(sys.argv) > 3 else ([1, 3, 5, 8, 12] if run == "main88" else [1, 3, 5])
NR = 10
OUT = f"/home/amit/.cache/vidyalai-scratch/p10/runs/ablate/{run}"; os.makedirs(OUT, exist_ok=True)
E = np.load("/home/amit/.cache/vidyalai-scratch/p10/runs/evalsets.npz")
g = lambda k: torch.as_tensor(E[k], device="cuda")
rep, sel, icl, nat = g("rep"), g("sel"), g("icl")[:500], g("nat")
t0 = time.time()
for step in steps:
    f = f"{OUT}/{step:05d}.json"
    if os.path.exists(f): continue
    m = D.load(f"{D.RUNS[run].path}/model_{step:05d}.pt", D.RUNS[run].cfg)
    cfg = m.cfg; allh = [(l, h) for l in range(cfg.n_layer) for h in range(cfg.n_head)]
    s = D.head_scores(m, sel, half=50)
    ranked = top_heads(s["induction"], len(allh))
    means = D.head_means(m, nat)
    def measure(heads):
        ab = D.ablation(heads, means) if heads else None
        L = D.token_losses(m, icl, ablate=ab)
        R = D.token_losses(m, rep, ablate=ab)
        return {"second": R[:, 51:].mean(1), "icl": L[:, 499] - L[:, 49], "icl_s": L[:, 490:510].mean(1) - L[:, 40:60].mean(1), "loss": L.mean(1)}
    base = measure([])
    out = {"step": step, "ranked": ranked[:12], "ind_scores": [float(s["induction"].mean(0)[l, h]) for l, h in ranked[:12]],
           "base": {k: [float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))] for k, v in base.items()}, "top": {}, "rand": {}}
    rng = np.random.default_rng(step)
    pool = [h for h in allh if h not in ranked[:12]]
    for k in KS:
        if k > len(allh) // 2: continue
        t = measure(ranked[:k])
        out["top"][k] = {kk: [float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                              float((v - base[kk]).mean()), float(1.96 * (v - base[kk]).std(ddof=1) / np.sqrt(len(v)))] for kk, v in t.items()}
        rs = []
        for _ in range(NR):
            pick = [pool[i] for i in rng.choice(len(pool), min(k, len(pool)), replace=False)]
            t = measure(pick); rs.append({kk: float((v - base[kk]).mean()) for kk, v in t.items()} | {"second_abs": float(t["second"].mean())})
        out["rand"][k] = rs
    json.dump(out, open(f, "w"), indent=1)
    print(run, step, "base", {k: round(v[0], 3) for k, v in out["base"].items()},
          {k: (round(v["second"][0], 2), round(v["icl"][2], 3)) for k, v in out["top"].items()},
          {k: (round(float(np.median([r["second_abs"] for r in v])), 2), round(float(np.median([r["icl"] for r in v])), 3)) for k, v in out["rand"].items()},
          f"{time.time()-t0:.0f}s", flush=True)
