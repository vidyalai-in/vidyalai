"""Sampling + FD-FMNIST evaluation for Part 7.
python evalgen.py steps  --ckpt C --kind ddpm|flow --model unet|dit --steps 1,2,4 --seeds 0,1,2 --n 2000 --out runs/X.jsonl [--save S]
python evalgen.py snaps  --glob 'ckpt/NAME-snap-*.pt' --kind flow --model unet --steps 20 --n 2000 --out runs/X.jsonl
python evalgen.py straight --ckpt C --kind ... --model ... --steps 100 --n 1000 --out runs/X.jsonl
python evalgen.py pairs  --ckpt C --model unet --steps 50 --n 20000 --pairs ckpt/pairs.pt
"""
import argparse, glob, json, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import flow, fmnist_eval as fe

p = argparse.ArgumentParser()
p.add_argument("mode"); p.add_argument("--ckpt"); p.add_argument("--glob"); p.add_argument("--kind", default="flow")
p.add_argument("--model", default="unet"); p.add_argument("--steps", default="20"); p.add_argument("--seeds", default="0")
p.add_argument("--n", type=int, default=2000); p.add_argument("--out"); p.add_argument("--save"); p.add_argument("--pairs")
p.add_argument("--bs", type=int, default=1000); p.add_argument("--tag", default="")
a = p.parse_args()
dev = "cuda"
abar = flow.ddpm_alpha_bar(dev)


def load(path):
    net = (flow.UNet() if a.model == "unet" else flow.DiT()).to(dev)
    s = torch.load(path, map_location=dev)
    net.load_state_dict(s["ema"] if "ema" in s else s); net.eval()
    return net, s.get("step"), s.get("train_time")


def noise(n, seed):
    g = torch.Generator(device=dev).manual_seed(1000 + seed)
    return torch.randn(n, 1, 28, 28, device=dev, generator=g)


def gen(net, z, steps, keep=False):
    outs, trajs = [], []
    for i in range(0, len(z), a.bs):
        zb = z[i:i + a.bs]
        r = flow.ddim_sample(net, zb, steps, abar, keep) if a.kind == "ddpm" else flow.euler_sample(net, zb, steps, keep)
        if keep:
            outs.append(r[0]); trajs.append(r[1])
        else:
            outs.append(r)
    x = torch.cat(outs).clamp(-1, 1)
    if keep:
        return x, [torch.cat([t[k] for t in trajs]) for k in range(len(trajs[0]))]
    return x


out = open(a.out, "a") if a.out else None
def emit(r):
    r["tag"] = a.tag; print(json.dumps(r), flush=True)
    if out: out.write(json.dumps(r) + "\n"); out.flush()

steps_list = [int(s) for s in a.steps.split(",")]
if a.mode == "steps":
    net, st, tt = load(a.ckpt)
    saved = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        z = noise(a.n, seed)
        for steps in steps_list:
            torch.cuda.synchronize(); t0 = time.time()
            x = gen(net, z, steps)
            torch.cuda.synchronize(); dt = time.time() - t0
            r = fe.score(x)
            emit({"ckpt": a.ckpt, "kind": a.kind, "model": a.model, "steps": steps, "seed": seed, "n": a.n,
                  "fd": r["fd"], "mean_conf": r["mean_conf"], "class_hist": r["class_hist"], "gen_sec": dt})
            if seed == 0: saved[f"s{steps}"] = x[:64].cpu().numpy().astype(np.float16)
    if a.save: np.savez_compressed(a.save, **saved)
elif a.mode == "snaps":
    files = sorted(glob.glob(a.glob), key=lambda f: int(f.split("-snap-")[1].split(".")[0]))
    z = noise(a.n, 0)
    for f in files:
        net, st, tt = load(f)
        for steps in steps_list:
            x = gen(net, z, steps)
            r = fe.score(x)
            emit({"ckpt": f, "kind": a.kind, "model": a.model, "train_step": st, "train_time": tt, "steps": steps,
                  "n": a.n, "fd": r["fd"], "mean_conf": r["mean_conf"]})
elif a.mode == "straight":
    net, st, tt = load(a.ckpt)
    for steps in steps_list:
        z = noise(a.n, 7)
        x, traj = gen(net, z, steps, keep=True)
        ps = flow.path_stats(traj)
        ratio, S = ps["ratio"].numpy(), ps["S"].numpy()
        # deviation from the chord: max distance of any trajectory point from the straight segment, relative to chord length
        X = torch.stack([t.flatten(1).double() for t in traj]); c = X[-1] - X[0]
        u = torch.linspace(0, 1, len(traj), dtype=torch.float64)[:, None, None]
        dev_lin = ((X - (X[0] + u * c)).norm(dim=-1).max(0).values / c.norm(dim=-1)).numpy()
        emit({"ckpt": a.ckpt, "kind": a.kind, "model": a.model, "steps": steps, "n": a.n,
              "ratio_mean": float(ratio.mean()), "ratio_sd": float(ratio.std()), "ratio_p95": float(np.percentile(ratio, 95)),
              "S_mean": float(S.mean()), "S_sd": float(S.std()), "maxdev_mean": float(dev_lin.mean()),
              "chord_mean": float(c.norm(dim=-1).mean())})
        # keep 6 trajectories (every 10th frame) for a picture
        if a.save:
            np.savez_compressed(a.save, traj=torch.stack([traj[k][:6] for k in range(0, len(traj), max(1, steps // 10))]).numpy().astype(np.float16))
elif a.mode == "pairs":
    net, st, tt = load(a.ckpt)
    z = noise(a.n, 99)
    torch.cuda.synchronize(); t0 = time.time()
    x = gen(net, z, steps_list[0])
    torch.cuda.synchronize()
    torch.save({"z": z.cpu(), "x": x.cpu()}, a.pairs)
    print(f"pairs {a.n} in {time.time() - t0:.1f}s", flush=True)
