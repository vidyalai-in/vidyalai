# Gradient accumulation equivalence on a small case: one batch of 8 x 128 vs 4 micro-batches of 2 x 128.
# FP32 on the CPU, the 22M model, seed 0. Run from fromscratch/.
import torch
from data import ShardLoader
from model import GPT, SMALL

torch.manual_seed(0)
model = GPT(SMALL)
x, y = ShardLoader("train", B=8, T=128).batch(0)

def grads(micro, divide=True, targets=y):
    model.zero_grad(set_to_none=True)
    n = 8 // micro
    total = 0.0
    for i in range(n):
        sl = slice(i * micro, (i + 1) * micro)
        loss = model(x[sl], targets[sl])
        (loss / n if divide else loss).backward()
        total += loss.item() / n
    return total, torch.cat([p.grad.flatten() for p in model.parameters()])

l_full, g_full = grads(8)
l_acc, g_acc = grads(2)
l_sum, g_sum = grads(2, divide=False)
print(f"loss: one batch {l_full:.6f}, mean of 4 micro-batches {l_acc:.6f}")
print(f"grad norm: one batch {g_full.norm():.6f}, accumulated /4 {g_acc.norm():.6f}, accumulated without /4 {g_sum.norm():.6f}")
print(f"largest difference, one batch vs accumulated: {(g_full - g_acc).abs().max():.2e} "
      f"(largest gradient entry {g_full.abs().max():.2e}); relative norm {(g_full - g_acc).norm() / g_full.norm():.2e}")

# With masked targets (ignore_index=-1), micro-batches hold different numbers of real targets.
masked = y.clone()
masked[:2, 32:] = -1                               # first micro-batch keeps 64 of its 256 targets
l_full, g_full = grads(8, targets=masked)
l_acc, g_acc = grads(2, targets=masked)
print(f"masked: one batch loss {l_full:.6f}, mean of micro-batch means {l_acc:.6f}, "
      f"relative gradient difference {(g_full - g_acc).norm() / g_full.norm():.2e}")
