"""Resumable training for Part 7. Usage:
python train.py --name NAME --model unet|dit --obj ddpm|flow [--tkind uniform|logit_normal] --steps N --minutes M
  [--init CKPT] [--pairs PAIRS.pt]  (reflow: train on fixed (noise, sample) pairs, starting from --init)
Writes ckpt/NAME.pt (resume state), ckpt/NAME-snap-STEP.pt (EMA weights) and runs/NAME.jsonl (log).
"""
import argparse, json, sys, time, copy
from pathlib import Path
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import flow, fmnist_eval as fe

p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--model", default="unet"); p.add_argument("--obj", default="flow")
p.add_argument("--tkind", default="uniform"); p.add_argument("--steps", type=int, default=6000)
p.add_argument("--minutes", type=float, default=10); p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--snap", type=int, default=1000)
p.add_argument("--seed", type=int, default=0); p.add_argument("--init"); p.add_argument("--pairs")
p.add_argument("--ema", type=float, default=0.999)
a = p.parse_args()

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
CK = Path("ckpt"); CK.mkdir(exist_ok=True); Path("runs").mkdir(exist_ok=True)
state_path = CK / f"{a.name}.pt"
net = (flow.UNet() if a.model == "unet" else flow.DiT()).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
step, train_time = 0, 0.0
g = torch.Generator(device=dev).manual_seed(a.seed)
if state_path.exists():
    s = torch.load(state_path, map_location=dev)
    net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); step, train_time = s["step"], s["train_time"]
    ema = copy.deepcopy(net); ema.load_state_dict(s["ema"])
    torch.set_rng_state(s["cpu_rng"]); torch.cuda.set_rng_state(s["cuda_rng"])
    print(f"resumed at step {step}, train_time {train_time:.0f}s", flush=True)
else:
    torch.manual_seed(a.seed)
    if a.init:
        w = torch.load(a.init, map_location=dev); net.load_state_dict(w["ema"] if "ema" in w else w)
    ema = copy.deepcopy(net)
ema.eval().requires_grad_(False)

if a.pairs:
    P = torch.load(a.pairs, map_location=dev); X0, X1 = P["z"], P["x"]
    N = len(X1)
else:
    X1, _ = fe.load("train"); X1 = X1.to(dev); N = len(X1)
abar = flow.ddpm_alpha_bar(dev)
print(f"{a.name}: {a.model}/{a.obj}/{a.tkind} params {flow.n_params(net):,} N={N}", flush=True)

log = open(f"runs/{a.name}.jsonl", "a")
t_end = time.time() + a.minutes * 60
acc, cnt = 0.0, 0
torch.cuda.synchronize(); t0 = time.time()
while step < a.steps and time.time() < t_end:
    idx = torch.randint(0, N, (a.batch,), device=dev)
    lr = a.lr * min(1.0, (step + 1) / 500)                       # 500-step warm-up, then constant
    for gp in opt.param_groups: gp["lr"] = lr
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if a.obj == "ddpm":
            loss = flow.ddpm_loss(net, X1[idx], abar)
        else:
            loss = flow.flow_loss(net, X1[idx], a.tkind, x0=X0[idx] if a.pairs else None)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    d = min(a.ema, (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()):
            pe.lerp_(pn, 1 - d)
    step += 1; acc += loss.item() if step % 25 == 0 else 0.0; cnt += step % 25 == 0
    if step % 250 == 0 or step == a.steps:
        torch.cuda.synchronize(); now = time.time(); train_time += now - t0; t0 = now
        log.write(json.dumps({"step": step, "loss": acc / max(cnt, 1), "train_time": round(train_time, 2), "lr": lr}) + "\n"); log.flush()
        acc, cnt = 0.0, 0
    if step % a.snap == 0 or step == a.steps:
        torch.save({"ema": ema.state_dict(), "step": step, "train_time": train_time}, CK / f"{a.name}-snap-{step}.pt")
torch.cuda.synchronize(); train_time += time.time() - t0
torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "ema": ema.state_dict(), "step": step, "train_time": train_time,
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, state_path)
print(f"stopped at step {step}, train_time {train_time:.0f}s, steps/s {step / train_time:.2f}", flush=True)
