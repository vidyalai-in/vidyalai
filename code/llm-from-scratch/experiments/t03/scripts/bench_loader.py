"""ShardLoader throughput (CPU, and optionally with the copy to the GPU) and a resume check."""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from data import ShardLoader

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
loader = ShardLoader("train", B=16, T=1024)
print(f"{len(loader.shards)} shards, {loader.n_tokens:,} tokens, {loader.n_batches:,} micro-batches of 16 x 1024")

def run(step, n):
    it = loader.batches(step=step)
    t0 = time.perf_counter()
    for _ in range(n):
        x, y = next(it)
        if device == "cuda":
            x = x.pin_memory().to("cuda", non_blocking=True); y = y.pin_memory().to("cuda", non_blocking=True)
    if device == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return n * 16 * 1024 / dt, dt

for step in (0, 1000, 3000, 3359):
    tps, dt = run(step, 320)
    print(f"{device}: from step {step}: 320 micro-batches (10 optimizer steps) in {dt:.3f} s = {tps/1e6:.1f}M tokens/s, {tps/28296:.0f}x the 28,296 tok/s training needs")

# Resume check: a fresh loader started at step 2000 gives the same batches as one that ran there.
a = ShardLoader("train"); b = ShardLoader("train")
xa, ya = a.batch(2000 * 32 + 5)
it = b.batches(step=2000); [next(it) for _ in range(5)]; xb, yb = next(it)
print("resume identical:", torch.equal(xa, xb) and torch.equal(ya, yb), "| y is x shifted by one:", torch.equal(xa[0, 1:], ya[0, :-1]))
last = 3360 * 32 - 1
print("last micro-batch of the 1.76B-token run ends at token", (last + 1) * 16 * 1024, "of", loader.n_tokens)
