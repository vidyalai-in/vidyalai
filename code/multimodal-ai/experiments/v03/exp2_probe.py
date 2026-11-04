"""Exp 2: k-shot linear probe on frozen image embeddings vs zero-shot (CPU only).
Train split: first 40,000 CIFAR-10 train images; L2 strength picked per k on the last 10,000."""
import json, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import clip_utils as cu
torch.set_num_threads(4)
OUT = "/home/amit/.cache/vidyalai-scratch/v03"
which = sys.argv[1]
d = torch.load(f"{OUT}/feats-{which}.pt", weights_only=False)
Xtr, ytr = d["train"].float()[:40000], torch.tensor(d["train_y"][:40000])
Xva, yva = d["train"].float()[40000:], torch.tensor(d["train_y"][40000:])
Xte, yte = d["test"].float(), torch.tensor(d["test_y"])
LAMS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def fit(X, y, lam):
    W = torch.zeros(X.shape[1], 10, requires_grad=True); b = torch.zeros(10, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=300, history_size=20, line_search_fn="strong_wolfe")
    Xs = X * 10  # embeddings are unit length; scale so the logits can grow
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(Xs @ W + b, y) + lam * (W ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    return lambda Z: (Z * 10 @ W + b).argmax(1)


res = {"model": which, "k": {}}
for k in [1, 2, 4, 8, 16, 32, 64, 128]:
    accs = []
    for seed in range(10):
        g = np.random.default_rng(seed)
        idx = np.concatenate([g.choice(np.where(ytr.numpy() == c)[0], k, replace=False) for c in range(10)])
        best = max(LAMS, key=lambda l: (fit(Xtr[idx], ytr[idx], l)(Xva) == yva).float().mean().item())
        f = fit(Xtr[idx], ytr[idx], best)
        accs.append((f(Xte) == yte).float().mean().item())
    a = np.array(accs)
    res["k"][k] = {"mean": a.mean(), "sd": a.std(ddof=1), "lo": a.mean() - 2.262 * a.std(ddof=1) / np.sqrt(10),
                   "hi": a.mean() + 2.262 * a.std(ddof=1) / np.sqrt(10), "seeds": accs}
    print(which, k, round(a.mean(), 4), round(a.std(ddof=1), 4), flush=True)
best = max(LAMS, key=lambda l: (fit(Xtr, ytr, l)(Xva) == yva).float().mean().item())
f = fit(Xtr, ytr, best)
c = int((f(Xte) == yte).sum())
res["full"] = {"n_train": 40000, "lam": best, "acc": cu.wilson(c, 10000)}
print(which, "full", res["full"], flush=True)
json.dump(res, open(f"{OUT}/exp2-{which}.json", "w"), indent=1)
