"""Experiment 1a: VQ-VAE on Fashion-MNIST, codebook size sweep with and without dead-code restarts.
Logs reconstruction error, codebook usage and FD-FMNIST of 5,000 reconstructed training images.
Usage: python vq_sweep.py <down> <steps> <K,K,...> <restart_every: 0|200> [seed]
Resumable: skips configs already in runs/vq_sweep.jsonl."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import torch

import ar_image as ai
import fmnist_eval as fe

torch.set_float32_matmul_precision("high")
RUNS = Path("/home/amit/.cache/vidyalai-scratch/v08/runs")
CK = Path("/home/amit/.cache/vidyalai-scratch/v08/ckpt")
RUNS.mkdir(exist_ok=True)
CK.mkdir(exist_ok=True)
down, steps = int(sys.argv[1]), int(sys.argv[2])
Ks = [int(k) for k in sys.argv[3].split(",")]
restart = int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
LOG = RUNS / "vq_sweep.jsonl"
done = set()
if LOG.exists():
    for l in LOG.read_text().splitlines():
        r = json.loads(l)
        done.add((r["down"], r["steps"], r["K"], r["restart_every"], r["seed"]))

x, _ = fe.load("train")
xt, _ = fe.load("test")
idx5k = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:5000]
for K in Ks:
    if (down, steps, K, restart, seed) in done:
        continue
    t0 = time.time()
    m, log = ai.train_vqvae(x, K=K, down=down, steps=steps, lr=1e-3, restart_every=restart, seed=seed)
    torch.cuda.synchronize()
    sec = time.time() - t0
    ev, _ = ai.vq_eval(m, xt)
    _, rec = ai.vq_eval(m, x[idx5k])
    fd = fe.score(rec)["fd"]
    r = {"down": down, "steps": steps, "K": K, "restart_every": restart, "seed": seed, "train_sec": sec,
         "test": ev, "fd_recon_train5000": fd, "curve": log}
    with open(LOG, "a") as f:
        f.write(json.dumps(r) + "\n")
    torch.save(m.state_dict(), CK / f"vq_d{down}_K{K}_r{restart}_s{seed}_{steps}.pt")
    print(K, restart, round(sec), json.dumps(ev), round(fd, 2), flush=True)
