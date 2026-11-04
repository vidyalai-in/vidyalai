"""Experiment 1: DDPM on a 2D spiral, on the CPU. Trains 3 seeds, then sliced Wasserstein distance vs sampling steps."""
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import ddpm

torch.set_num_threads(4)
OUT = "/home/amit/.cache/vidyalai-scratch/v05/runs"
SCALE = None


def spiral(n, g):
    u = torch.rand(n, generator=g)
    r, a = 0.2 + u, 4 * np.pi * u            # two turns, radius 0.2 to 1.2
    x = torch.stack([r * torch.cos(a), r * torch.sin(a)], 1) + 0.03 * torch.randn(n, 2, generator=g)
    return x / 0.62                           # overall std about 1 (scalar, fixed)


def swd(a, b, g, nproj=500):
    d = torch.randn(nproj, 2, generator=g); d = d / d.norm(dim=1, keepdim=True)
    pa, pb = (a @ d.T).sort(0).values, (b @ d.T).sort(0).values
    return float((pa - pb).abs().mean())


CURVE = None


def on_spiral(x, tol=0.1):
    """Share of points within tol of the noise-free spiral curve (data units)."""
    global CURVE
    if CURVE is None:
        u = torch.linspace(0, 1, 20000); r, a = 0.2 + u, 4 * np.pi * u
        CURVE = torch.stack([r * torch.cos(a), r * torch.sin(a)], 1) / 0.62
    d = torch.cdist(x, CURVE).min(1).values
    return float((d < tol).float().mean())


def train(seed, steps=20000, bs=1024):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    s = ddpm.Schedule("linear")
    m = ddpm.MLPDenoiser()
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    data = spiral(100_000, g)
    log = []
    for it in range(steps):
        x0 = data[torch.randint(0, len(data), (bs,), generator=g)]
        loss = ddpm.loss_simple(m, x0, s)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 200 == 0:
            log.append((it, float(loss.detach())))
    return m.eval(), s, log


if __name__ == "__main__":
    STEPS = [1, 2, 5, 10, 20, 50, 100, 250, 1000]
    res = {"swd": {}, "on": {}, "floor": [], "floor_on": [], "loss": {}, "train_s": {}}
    g = torch.Generator().manual_seed(123)
    for k in range(5):
        a_, b_ = spiral(5000, g), spiral(5000, g)
        res["floor"].append(swd(a_, b_, g)); res["floor_on"].append(on_spiral(a_))
    for seed in (0, 1, 2):
        t0 = time.time(); m, s, log = train(seed); res["train_s"][seed] = time.time() - t0
        res["loss"][seed] = log
        ref = spiral(5000, torch.Generator().manual_seed(1000 + seed))
        for meth in ("ddpm", "ddim"):
            for n in STEPS:
                gs = torch.Generator().manual_seed(seed)
                x = ddpm.sample(m, (5000, 2), s, steps=n, method=meth, generator=gs, device="cpu", clip=False)
                res["swd"].setdefault(f"{meth}_{n}", []).append(swd(x, ref, gs))
                res["on"].setdefault(f"{meth}_{n}", []).append(on_spiral(x))
        if seed == 0:
            torch.save(m.state_dict(), f"{OUT}/toy_seed0.pt")
            # pictures: forward snapshots and reverse trajectories
            x0 = spiral(2000, torch.Generator().manual_seed(7))
            fw = {int(t): s.q_sample(x0, torch.full((2000,), t), torch.randn(2000, 2, generator=torch.Generator().manual_seed(8))).tolist()
                  for t in (0, 50, 150, 300, 600, 999)}
            gs = torch.Generator().manual_seed(9)
            _, tr_ddpm = ddpm.sample(m, (300, 2), s, steps=1000, method="ddpm", generator=gs, device="cpu", clip=False, return_traj=True)
            gs = torch.Generator().manual_seed(9)
            _, tr_ddim = ddpm.sample(m, (300, 2), s, steps=1000, method="ddim", generator=gs, device="cpu", clip=False, return_traj=True)
            keep = list(range(0, 1001, 10))
            pics = {"forward": {k: np.round(np.array(v), 3).tolist() for k, v in fw.items()},
                    "traj_ddpm": np.round(torch.stack([tr_ddpm[i] for i in keep])[:, :40].numpy(), 3).tolist(),
                    "traj_ddim": np.round(torch.stack([tr_ddim[i] for i in keep])[:, :40].numpy(), 3).tolist(),
                    "end_ddpm": np.round(tr_ddpm[-1].numpy(), 3).tolist(), "end_ddim": np.round(tr_ddim[-1].numpy(), 3).tolist()}
            json.dump(pics, open(f"{OUT}/toy_pics.json", "w"))
        print(seed, {k: round(v[-1], 4) for k, v in res["swd"].items()}, flush=True)
        print(seed, {k: round(v[-1], 3) for k, v in res["on"].items()}, flush=True)
    json.dump(res, open(f"{OUT}/toy_results.json", "w"), indent=1)
    print("floor", np.mean(res["floor"]), np.std(res["floor"]), np.mean(res["floor_on"]))
