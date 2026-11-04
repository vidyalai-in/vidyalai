"""Experiment 2: a small U-Net DDPM on Fashion-MNIST, trained in resumable chunks.

python train_fm.py --sched linear --minutes 11 --total 30000      # one chunk; rerun to resume
python train_fm.py --pilot                                          # throughput only
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import ddpm, fmnist_eval as fe

p = argparse.ArgumentParser()
p.add_argument("--sched", default="linear"); p.add_argument("--minutes", type=float, default=11)
p.add_argument("--total", type=int, default=30000); p.add_argument("--bs", type=int, default=128)
p.add_argument("--ch", type=int, default=64); p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--nres", type=int, default=2); p.add_argument("--cl", type=int, default=1); p.add_argument("--compile", type=int, default=0)
p.add_argument("--eval_every", type=int, default=2500); p.add_argument("--pilot", action="store_true")
a = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = "cuda"
run = f"/home/amit/.cache/vidyalai-scratch/v05/runs/fm_{a.sched}"
os.makedirs(run, exist_ok=True)
if os.path.exists(f"{run}/total_override"):          # match step counts across runs
    a.total = int(open(f"{run}/total_override").read())
x_train, _ = fe.load("train"); x_train = x_train.to(dev)
sched = ddpm.Schedule(a.sched)


def make():
    torch.manual_seed(0)
    m = ddpm.UNet(ch=a.ch, nres=a.nres).to(dev)
    if a.cl: m = m.to(memory_format=torch.channels_last)
    return m


model = make()
ema = ddpm.EMA(model, 0.9995)
fwd = torch.compile(model) if a.compile else model
efwd = torch.compile(ema.model) if a.compile else ema.model
opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.0, fused=True)
state = {"step": 0, "train_s": 0.0, "log": [], "evals": [], "chunks": []}
ck = f"{run}/ckpt.pt"
if os.path.exists(ck) and not a.pilot:
    c = torch.load(ck, map_location=dev, weights_only=False)
    model.load_state_dict(c["model"]); ema.model.load_state_dict(c["ema"]); opt.load_state_dict(c["opt"])
    state = c["state"]; torch.set_rng_state(c["rng"].cpu()); torch.cuda.set_rng_state(c["cuda_rng"].cpu())
    print("resumed at step", state["step"], flush=True)
else:
    torch.manual_seed(0)


def lr_at(step):
    return a.lr * min(1.0, (step + 1) / 500)


def train_step():
    idx = torch.randint(0, len(x_train), (a.bs,), device=dev)
    x0 = x_train[idx]
    if a.cl: x0 = x0.contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = ddpm.loss_simple(fwd, x0, sched)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); ema.update(model)
    return loss


def evaluate(n=2000, steps=50):
    g = torch.Generator(device=dev).manual_seed(1)
    xs = [ddpm.sample(efwd, (1000, 1, 28, 28), sched, steps=steps, method="ddim", generator=g) for _ in range(n // 1000)]
    x = torch.cat(xs).float().cpu()
    r = fe.score(x)
    g = torch.Generator(device=dev).manual_seed(2)
    grid = ddpm.sample(ema.model, (16, 1, 28, 28), sched, steps=steps, method="ddim", generator=g).float().cpu()
    return r, grid


if a.pilot:
    model.train(); t0 = time.time()
    for _ in range(30): train_step()
    torch.cuda.synchronize(); print(f"warm-up/compile {time.time()-t0:.0f}s")
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(200): train_step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / 200
    print(f"params {sum(q.numel() for q in model.parameters())/1e6:.2f}M  train {1/dt:.1f} it/s  ({dt*1000:.1f} ms/step, bs {a.bs})")
    ema.model.eval(); x = torch.randn(1000, 1, 28, 28, device=dev); t = torch.full((1000,), 500, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t0 = time.time()
        for _ in range(5): efwd(x, t)
        torch.cuda.synchronize(); print(f"sampling warm-up {time.time()-t0:.0f}s"); t0 = time.time()
        for _ in range(20): efwd(x, t)
        torch.cuda.synchronize()
    print(f"forward batch 1000: {(time.time()-t0)/20*1000:.1f} ms; peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    sys.exit()

model.train()
t_end = time.time() + a.minutes * 60
chunk = {"start_step": state["step"], "wall_start": time.time()}
grids = dict(np.load(f"{run}/grids.npz")) if os.path.exists(f"{run}/grids.npz") else {}
while state["step"] < a.total and time.time() < t_end:
    torch.cuda.synchronize(); t0 = time.time()
    k = 0
    while k < 250 and state["step"] < a.total:
        for gp in opt.param_groups: gp["lr"] = lr_at(state["step"])
        loss = train_step(); state["step"] += 1; k += 1
        if state["step"] % 50 == 0:
            state["log"].append((state["step"], float(loss)))
    torch.cuda.synchronize(); state["train_s"] += time.time() - t0
    if state["step"] % a.eval_every == 0 or state["step"] in (500, 1000, 1500) or state["step"] >= a.total:
        if not any(e["step"] == state["step"] for e in state["evals"]):
            t1 = time.time(); r, grid = evaluate()
            state["evals"].append({"step": state["step"], "train_s": state["train_s"], "fd": r["fd"], "mean_conf": r["mean_conf"],
                                   "class_hist": r["class_hist"], "eval_s": time.time() - t1})
            grids[str(state["step"])] = grid.numpy().astype(np.float16)
            np.savez(f"{run}/grids.npz", **grids)
            print(f"step {state['step']} train {state['train_s']:.0f}s loss {float(loss):.4f} FD {r['fd']:.2f} ({time.time()-t1:.0f}s eval)", flush=True)
state["arch"] = {"ch": a.ch, "nres": a.nres}; state["args"] = vars(a)
chunk.update(end_step=state["step"], wall_s=time.time() - chunk["wall_start"]); state["chunks"].append(chunk)
torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(), "state": state,
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, ck + ".tmp")
os.replace(ck + ".tmp", ck)
json.dump(state, open(f"{run}/state.json", "w"))
print("saved at step", state["step"], "train_s", round(state["train_s"]), flush=True)
