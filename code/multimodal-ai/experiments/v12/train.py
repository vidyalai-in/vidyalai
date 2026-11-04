"""Resumable training of the tiny video models for Part 12.
python train.py --mode frames|factorized|full --steps 4000 --minutes 11
Writes ckpt/<mode>.pt (resume state + EMA) and runs/train-<mode>.jsonl."""
import argparse, copy, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import video as V, fmnist_eval as fe

p = argparse.ArgumentParser()
p.add_argument("--mode", required=True); p.add_argument("--steps", type=int, default=4000)
p.add_argument("--minutes", type=float, default=11); p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--seed", type=int, default=0)
p.add_argument("--frames", type=int, default=8)
p.add_argument("--name"); p.add_argument("--init_from_image")  # e.g. ckpt/frames.pt: inflate the per-frame model
a = p.parse_args()
dev = "cuda"
CK = Path("ckpt"); CK.mkdir(exist_ok=True)
name = a.name or a.mode
path = CK / f"{name}.pt"
torch.manual_seed(a.seed)
net = V.VideoDiT(a.mode, frames=a.frames).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0, fused=True)
step, train_time = 0, 0.0
gen = torch.Generator(device=dev).manual_seed(1000 + a.seed)
if path.exists():
    s = torch.load(path, map_location=dev)
    net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); step, train_time = s["step"], s["train_time"]
    ema = copy.deepcopy(net); ema.load_state_dict(s["ema"]); gen.set_state(s["gen"].cpu())
    torch.cuda.set_rng_state(s["cuda_rng"].cpu())
    print(f"resumed at {step}", flush=True)
else:
    if a.init_from_image:
        # Inflate: copy every weight of the per-frame ("frames") model into the factorized model. Each block's adaLN
        # projection makes (shift, scale, gate) for [space attention, MLP] there and for [space, time, MLP] here, so its
        # rows are copied around a block of zeros for the new temporal attention, whose gate therefore starts at 0:
        # the inflated model starts out as exactly the image model, as in AnimateDiff and Make-A-Video.
        src = torch.load(a.init_from_image, map_location=dev)["ema"]
        dst = net.state_dict(); d = net.temb[0].out_features
        for k, v in src.items():
            if ".ada.1." in k:
                w = torch.zeros_like(dst[k]); w[:3 * d] = v[:3 * d]; w[6 * d:] = v[3 * d:]; dst[k] = w
            elif ".attn.1." in k:
                raise KeyError(k)
            else:
                dst[k.replace(".attn.0.", ".attn.0.")] = v
        net.load_state_dict(dst)
        print("inflated from", a.init_from_image, flush=True)
    ema = copy.deepcopy(net)
ema.eval().requires_grad_(False)
X, _ = fe.load("train"); X = X.to(dev)          # Fashion-MNIST training split only
cnet = torch.compile(net)
print(f"{a.mode}: params {V.n_params(net):,}", flush=True)
log = open(f"runs/train-{name}.jsonl", "a")
t_end = time.time() + a.minutes * 60
acc = []
torch.cuda.synchronize(); t0 = time.time()
while step < a.steps and time.time() < t_end:
    clips = V.moving_fmnist(X, a.batch, frames=a.frames, generator=gen)["x"]
    for gp in opt.param_groups: gp["lr"] = a.lr * min(1.0, (step + 1) / 250)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = V.flow_loss(cnet, clips)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    dcy = min(0.999, (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()):
            pe.lerp_(pn, 1 - dcy)
    step += 1
    if step % 25 == 0: acc.append(loss.item())
    if step % 250 == 0 or step == a.steps:
        torch.cuda.synchronize(); now = time.time(); train_time += now - t0; t0 = now
        log.write(json.dumps({"step": step, "loss": sum(acc) / max(len(acc), 1), "train_time": round(train_time, 2),
                              "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)}) + "\n"); log.flush(); acc = []
torch.cuda.synchronize(); train_time += time.time() - t0
torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "ema": ema.state_dict(), "step": step,
            "train_time": train_time, "gen": gen.get_state(), "cuda_rng": torch.cuda.get_rng_state(),
            "params": V.n_params(net), "batch": a.batch, "frames": a.frames}, path)
print(f"stopped at {step}, train_time {train_time:.0f}s, {step*a.batch/train_time:.0f} clips/s", flush=True)
