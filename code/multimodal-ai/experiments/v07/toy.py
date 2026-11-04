"""2D toy for Part 7 (CPU only): noise N(0, I) -> two moons (scaled 1.5x, centred at (4, 0) so the paths are easy to see).
Trains (a) a DDPM epsilon-prediction MLP sampled with DDIM, (b) a 1-rectified flow, (c) a 2-rectified flow (one reflow
round on 50,000 (noise, sample) pairs from (b)). Measures path straightness and sliced Wasserstein distance (SWD)
to fresh target samples against the number of sampling steps, 5 seeds each. Writes runs/toy.json and runs/toy_traj.npz.
"""
import json, sys, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import flow

torch.set_num_threads(4)
torch.manual_seed(0)


def moons(n, g=None):
    t = torch.rand(n, generator=g) * math.pi
    up = torch.rand(n, generator=g) < 0.5
    x = torch.where(up, torch.cos(t), 1 - torch.cos(t))
    y = torch.where(up, torch.sin(t), 0.5 - torch.sin(t))
    p = torch.stack([x - 0.5, y - 0.25], 1) * 1.5 + torch.tensor([4.0, 0.0])
    return p + 0.08 * torch.randn(n, 2, generator=g)


class MLP(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 32, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, flow.timestep_embedding(t, 32)], 1))


abar = flow.ddpm_alpha_bar("cpu")


def train(kind, steps=12000, pairs=None, init=None):
    m = MLP()
    if init is not None: m.load_state_dict(init.state_dict())
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        if pairs is None:
            x1 = moons(1024); x0 = torch.randn_like(x1)
        else:
            i = torch.randint(0, len(pairs[0]), (1024,)); x0, x1 = pairs[0][i], pairs[1][i]
        if kind == "ddpm":
            i = torch.randint(0, 1000, (1024,)); a = abar[i][:, None]; eps = torch.randn_like(x1)
            loss = F.mse_loss(m(a.sqrt() * x1 + (1 - a).sqrt() * eps, i.float() / 1000), eps)
        else:
            t = torch.rand(1024); tt = t[:, None]
            loss = F.mse_loss(m((1 - tt) * x0 + tt * x1, t), x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


@torch.no_grad()
def sample(m, kind, z, steps):
    if kind == "ddpm":
        idx = torch.linspace(999, 0, steps).round().long().tolist() if steps > 1 else [999]
        x, traj = z.clone(), [z.clone()]
        for k, i in enumerate(idx):
            a = abar[i]; eps = m(x, torch.full((len(x),), i / 1000))
            x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
            an = abar[idx[k + 1]] if k + 1 < len(idx) else torch.tensor(1.0)
            x = an.sqrt() * x0 + (1 - an).sqrt() * eps; traj.append(x.clone())
        return x, traj
    x, traj = z.clone(), [z.clone()]
    for k in range(steps):
        x = x + m(x, torch.full((len(x),), k / steps)) / steps; traj.append(x.clone())
    return x, traj


def swd(a, b, nproj=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = F.normalize(torch.randn(nproj, 2, generator=g), dim=1)
    pa, pb = (a @ d.T).sort(0).values, (b @ d.T).sort(0).values
    return float(((pa - pb) ** 2).mean().sqrt())


t0 = time.time()
models = {}
models["ddpm"] = train("ddpm"); print("ddpm", time.time() - t0, flush=True)
models["rf1"] = train("flow"); print("rf1", time.time() - t0, flush=True)
z = torch.randn(50000, 2, generator=torch.Generator().manual_seed(5))
x1, _ = sample(models["rf1"], "flow", z, 200)
models["rf2"] = train("flow", pairs=(z, x1), init=models["rf1"]); print("rf2", time.time() - t0, flush=True)

res = {"swd": {}, "straight": {}, "floor": []}
STEPS = [1, 2, 3, 4, 8, 16, 32, 100]
for seed in range(5):
    g = torch.Generator().manual_seed(100 + seed)
    ref = moons(5000, g); ref2 = moons(5000, g)
    res["floor"].append(swd(ref, ref2, seed=seed))
    zz = torch.randn(5000, 2, generator=g)
    for name, m in models.items():
        kind = "ddpm" if name == "ddpm" else "flow"
        for st in STEPS:
            x, _ = sample(m, kind, zz, st)
            res["swd"].setdefault(name, {}).setdefault(str(st), []).append(swd(x, ref, seed=seed))
zz = torch.randn(2000, 2, generator=torch.Generator().manual_seed(9))
save = {}
for name, m in models.items():
    kind = "ddpm" if name == "ddpm" else "flow"
    x, traj = sample(m, kind, zz, 100)
    ps = flow.path_stats(traj)
    res["straight"][name] = {"ratio_mean": float(ps["ratio"].mean()), "ratio_median": float(ps["ratio"].median()),
                             "S_mean": float(ps["S"].mean())}
    save[name] = torch.stack(traj)[:, :300].numpy()           # 101 x 300 x 2
# independent coupling lines for the figure
save["pairs_x0"] = torch.randn(300, 2, generator=torch.Generator().manual_seed(3)).numpy()
save["pairs_x1"] = moons(300, torch.Generator().manual_seed(4)).numpy()
np.savez_compressed("runs/toy_traj.npz", **save)
res["seconds"] = time.time() - t0
json.dump(res, open("runs/toy.json", "w"), indent=1)
for name in models:
    print(name, res["straight"][name], {k: round(float(np.mean(v)), 3) for k, v in res["swd"][name].items()})
print("floor", np.mean(res["floor"]))
