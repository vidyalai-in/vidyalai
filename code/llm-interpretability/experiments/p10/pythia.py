# Pythia-70m across public training checkpoints: induction/previous-token scores (Part 4's lab.heads) on
# repeated random tokens, second-copy loss, and the ICL score on Part 4's WikiText-103 windows.
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import heads as H
from transformer_lens import HookedTransformer
OUT = "/home/amit/.cache/vidyalai-scratch/p10/runs/pythia70m.json"
w = np.load("/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_pythia.npy")
ids, c = np.unique(w, return_counts=True); allowed = ids[c >= 5]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
t0 = time.time()
for step in [512, 1000, 2000, 3000, 4000, 8000, 143000]:
    if str(step) in res: continue
    m = HookedTransformer.from_pretrained("pythia-70m", checkpoint_value=None if step == 143000 else step, device="cuda")  # main = step143000
    rep = H.repeated_tokens(m, 200, 50, seed=1, allowed=allowed)
    s = H.head_scores(m, rep, half=50)
    win = torch.as_tensor(w, device="cuda").long()
    L = H.token_loss(m, win, batch=32)
    ind = s["induction"].mean(0)
    res[str(step)] = {"tokens": step * 2_097_152, "induction": ind.tolist(), "prev": s["prev"].mean(0).tolist(),
                      "second": float(s["loss"][:, 51:].mean()), "first": float(s["loss"][:, 1:50].mean()),
                      "icl": float((L[:, 499] - L[:, 49]).mean()), "icl_s": float((L[:, 490:510].mean(1) - L[:, 40:60].mean(1)).mean()),
                      "loss": float(L.mean()), "n_windows": int(len(L))}
    print(step, "ind max", ind.max().round(3), "second", round(res[str(step)]["second"], 2), "icl", round(res[str(step)]["icl"], 3),
          f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(OUT, "w"))
    del m; torch.cuda.empty_cache()
