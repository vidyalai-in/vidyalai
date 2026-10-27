# Part 7: weight norms across the run's snapshots (CPU only).
import glob, json, math, re, sys
import numpy as np, torch
R = "/home/amit/.cache/vidyalai-scratch/t07/runs/main88"
paths = sorted(glob.glob(f"{R}/model_*.pt"))
out = []
for p in paths:
    step = int(re.search(r"model_(\d+)", p).group(1))
    sd = torch.load(p, map_location="cpu", weights_only=True)
    E = sd["tok_emb.weight"]
    rows = E.norm(dim=1)
    rec = {"step": step, "emb": E.norm().item(), "emb_row_mean": rows.mean().item(), "emb_row_p10": rows.quantile(0.1).item(), "emb_row_p90": rows.quantile(0.9).item()}
    groups = {"wq": [], "wk": [], "wv": [], "wo": [], "gate": [], "up": [], "down": []}
    for k, v in sd.items():
        for g in groups:
            if k.endswith(f".{g}.weight"): groups[g].append(v.norm().item() ** 2)
    for g, sq in groups.items(): rec[g] = math.sqrt(sum(sq))
    rec["matrices_excl_emb"] = math.sqrt(sum(v.norm().item() ** 2 for k, v in sd.items() if v.dim() == 2 and k != "tok_emb.weight"))
    gains = [v for k, v in sd.items() if v.dim() == 1]
    rec["norm_gain_mean"] = torch.cat(gains).mean().item()
    rec["final_norm_gain_mean"] = sd["norm.weight"].mean().item()
    rec["rms_emb_entry"] = (E.pow(2).mean().sqrt()).item()
    out.append(rec); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    del sd
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/t07/post/weight_norms.json", "w"))
# what weight decay alone would do: product of (1 - lr*wd) over the logged schedule
lrs = [json.loads(l)["lr"] for l in open(f"{R}/log.jsonl") if '"loss"' in l]
f = 1.0; fac = {}
for i, lr in enumerate(lrs):
    f *= 1 - lr * 0.1; fac[i + 1] = f
print("pure-decay shrink factor at steps 150, 500, 1000, 2000, 3360:", {s: round(fac[s], 4) for s in (150, 500, 1000, 2000, 3360)}, "sum lr", sum(lrs))
