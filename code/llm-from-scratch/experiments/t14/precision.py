"""Does dropping the FP32 master copy hurt? The 22M model trained three ways on the same tokens (Part 14).

  mixed  : train.py's recipe, FP32 weights and AdamW state, BF16 autocast
  bf16   : weights, gradients and AdamW moments all BF16 (8 bytes per parameter instead of 16)
  adam8  : FP32 weights, bitsandbytes AdamW8bit moments (12 bytes per parameter)
Same seed, data order, schedule (train.py's lr_at) and validation tokens.
    python precision.py --mode bf16 --steps 300
"""
import argparse
import json
import time

import torch

from data import ShardLoader
from model import GPT, SMALL
from train import evaluate, get_args, lr_at

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["mixed", "bf16", "adam8"], required=True)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--rerun", default="", help="suffix for the output file, for a repeat of the same run")
a = ap.parse_args()
hp = get_args(["--out", "/tmp/unused", "--steps", str(a.steps), "--warmup", "20", "--lr", "1.5e-3"])

torch.manual_seed(1337)
model = GPT(SMALL).cuda()
if a.mode == "bf16":
    model.to(torch.bfloat16)
decay = [p for p in model.parameters() if p.dim() >= 2]
other = [p for p in model.parameters() if p.dim() < 2]
groups = [{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}]
if a.mode == "adam8":
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(groups, lr=hp.lr, betas=(0.9, 0.95), eps=1e-8)
else:
    opt = torch.optim.AdamW(groups, lr=hp.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
run = torch.compile(model)
autocast = torch.autocast("cuda", dtype=torch.bfloat16)
batches = ShardLoader("train", B=16, T=1024).batches(step=0, accum=a.accum)
val = ShardLoader("val", B=16, T=1024)
log, t0 = [], time.time()
for step in range(a.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step, hp)
    tot = 0.0
    for _ in range(a.accum):
        x, y = (t.pin_memory().cuda(non_blocking=True) for t in next(batches))
        with autocast:
            loss = run(x, y) / a.accum
        loss.backward()
        tot += loss.detach()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 50 == 49 or step == a.steps - 1:
        v = evaluate(run, val, 40, autocast)
        log.append({"step": step + 1, "train": float(tot), "val": v, "elapsed": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
json.dump({"mode": a.mode, "steps": a.steps, "tokens_per_step": 16 * 1024 * a.accum, "log": log,
           "peak_alloc": torch.cuda.max_memory_allocated()},
          open(f"/home/amit/.cache/vidyalai-scratch/t14/runs/precision_{a.mode}{a.rerun}.json", "w"), indent=1)
