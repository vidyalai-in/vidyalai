"""The series' training script (Build an LLM From Scratch, Part 5).

    python train.py --model small --out runs/small --compile --fused       # a fresh run
    python train.py --model small --out runs/small --compile --fused --resume   # same command + --resume

AdamW with weight decay on matrices only, linear warm-up then cosine decay to a floor, gradient
clipping by global norm, BF16 autocast over FP32 weights, gradient accumulation, periodic
validation loss, one JSON line per step in <out>/log.jsonl, and checkpoints that resume exactly.
"""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from data import ShardLoader
from model import GPT, MAIN, SMALL
from sizes import SIZES

CONFIGS = {"main": MAIN, "small": SMALL, **SIZES}   # Part 8 adds the sweep's sizes


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=CONFIGS, default="small")
    p.add_argument("--out", required=True, help="run directory: log.jsonl and ckpt.pt")
    p.add_argument("--micro-batch", type=int, default=16, help="sequences per forward/backward pass")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--accum", type=int, default=32, help="micro-batches per optimizer step")
    p.add_argument("--steps", type=int, default=400, help="length of the learning-rate schedule")
    p.add_argument("--stop-at", type=int, default=None, help="end early, at this step (default: --steps)")
    p.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
    p.add_argument("--warmup", type=int, default=20, help="linear warm-up steps (0 = none)")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine floor, as a fraction of --lr")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0, help="max global gradient norm (0 = no clipping)")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--val-batches", type=int, default=40, help="micro-batches of the val shard per evaluation")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--keep-ckpts", action="store_true", help="also keep model-only snapshots per checkpoint")
    p.add_argument("--snapshot-at", default="", help="extra steps for model-only snapshots, e.g. 0,10,25 (Part 7)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true", help="torch.compile the model (Part 6)")
    p.add_argument("--fused", action="store_true", help="fused AdamW kernel (Part 6)")
    p.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt if it exists")
    p.add_argument("--deterministic", action="store_true", help="bit-reproducible kernels (slower)")
    return p.parse_args(argv)


def lr_at(step, a):
    """Linear warm-up to a.lr over a.warmup steps, then cosine decay to a.lr * a.min_lr_ratio at a.steps."""
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    progress = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    floor = a.lr * a.min_lr_ratio
    return floor + 0.5 * (a.lr - floor) * (1 + math.cos(math.pi * progress))


def make_optimizer(model, a):
    """AdamW with two groups: matrices (every linear layer and the tied embedding) decay; norm gains don't."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": a.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=a.lr, betas=(a.beta1, a.beta2), eps=a.eps,
                             fused=True if a.fused else None)


@torch.no_grad()
def evaluate(model, loader, n_batches, autocast):
    """Mean loss over the first n_batches micro-batches of a split: the same tokens every time."""
    model.eval()
    total = torch.zeros((), device="cuda")
    for i in range(n_batches):
        x, y = (t.cuda(non_blocking=True) for t in loader.batch(i))
        with autocast:
            total += model(x, y)
    model.train()
    return (total / n_batches).item()


def save_checkpoint(path, model, opt, step, a):
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "args": vars(a)}
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)                         # a crash mid-save never corrupts ckpt.pt


def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ckpt_path, log_path = os.path.join(a.out, "ckpt.pt"), os.path.join(a.out, "log.jsonl")
    torch.manual_seed(a.seed)
    if a.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"     # before cuBLAS starts
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True              # FP32 matmuls (fp32 runs only) on tensor cores
    cfg = CONFIGS[a.model]
    raw = GPT(cfg).cuda()                                     # FP32 master weights
    opt = make_optimizer(raw, a)
    step = 0
    if a.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        raw.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        del ckpt
    model = torch.compile(raw) if a.compile else raw
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.dtype == "bf16" else nullcontext()

    train = ShardLoader("train", B=a.micro_batch, T=a.seq_len)
    val = ShardLoader("val", B=a.micro_batch, T=a.seq_len)
    tokens_per_step = a.micro_batch * a.seq_len * a.accum
    stop = a.stop_at if a.stop_at is not None else a.steps
    log = open(log_path, "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()

    write({"event": "resume" if step else "start", "step": step, "args": vars(a), "config": asdict(cfg),
           "params": raw.num_params(), "tokens_per_step": tokens_per_step, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name()})
    batches = train.batches(step=step, accum=a.accum)
    snapshot_at = {int(s) for s in a.snapshot_at.split(",") if s}
    if step == 0 and 0 in snapshot_at:                       # the untrained weights
        torch.save(raw.state_dict(), os.path.join(a.out, "model_00000.pt"))
    start, loss_val, t0 = step, 0.0, time.time()
    while step < stop:
        if a.val_every and step % a.val_every == 0 and (step == 0 or step != start):   # not twice on resume
            write({"step": step, "val_loss": evaluate(model, val, a.val_batches, autocast),
                   "tokens": step * tokens_per_step})
        t_step = time.time()
        lr = lr_at(step, a)
        for group in opt.param_groups:
            group["lr"] = lr
        loss_sum = torch.zeros((), device="cuda")
        for _ in range(a.accum):
            x, y = (t.pin_memory().cuda(non_blocking=True) for t in next(batches))
            with autocast:
                loss = model(x, y) / a.accum                  # so the summed gradients are a mean
            loss.backward()
            loss_sum += loss.detach()
        max_norm = a.grad_clip if a.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), max_norm)   # returns the norm before clipping
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_val = loss_sum.item()                            # waits for the GPU, so the timing is real
        dt = time.time() - t_step
        step += 1
        write({"step": step - 1, "loss": loss_val, "lr": lr, "grad_norm": grad_norm.item(),
               "tokens": step * tokens_per_step, "dt": round(dt, 4), "tok_per_s": round(tokens_per_step / dt),
               "elapsed": round(time.time() - t0, 2)})
        if not math.isfinite(loss_val):
            write({"step": step - 1, "event": "diverged"})
            break
        if step % a.ckpt_every == 0 or step == stop:
            save_checkpoint(ckpt_path, raw, opt, step, a)
            if a.keep_ckpts:
                torch.save(raw.state_dict(), os.path.join(a.out, f"model_{step:05d}.pt"))
        elif step in snapshot_at:
            torch.save(raw.state_dict(), os.path.join(a.out, f"model_{step:05d}.pt"))
    if a.val_every and math.isfinite(loss_val):
        write({"step": step, "val_loss": evaluate(model, val, a.val_batches, autocast),
               "tokens": step * tokens_per_step})
    write({"event": "end", "step": step, "elapsed": round(time.time() - t0, 2),
           "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})


if __name__ == "__main__":
    main()
