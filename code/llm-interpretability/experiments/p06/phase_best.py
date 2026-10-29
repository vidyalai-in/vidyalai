"""Recompute phase diagrams from saved weights: outcome of the lowest-loss model per cell (CPU)."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
import torch
from lab import toy
for exp in ("phase2", "phase3"):
    r = json.load(open(f"runs/{exp}.json")); s = torch.load(f"runs/{exp}_W.pt")
    reps = r["reps"]; N = len(r["importances"])
    c = toy.classify(s["W"])[:, -1].view(N * N, reps)
    L = s["loss"].view(N * N, reps)
    best = torch.gather(c, 1, L.argmin(1, keepdim=True))[:, 0]
    agree = (best == torch.tensor(r["majority"])).float().mean().item()
    unanim = (c == c[:, :1]).all(1).float().mean().item()
    r["best"] = best.tolist(); r["agree_best_vs_majority"] = agree; r["unanimous_cells"] = unanim
    json.dump(r, open(f"runs/{exp}.json", "w"))
    print(exp, "agree", round(agree, 3), "unanimous", round(unanim, 3))
    for row in range(N - 1, -1, -1):
        print(f'{1/r["densities"][row]:6.1f} ' + ''.join('.sD'[best[row * N + col]] for col in range(N)))
