# How small are the 16-bit activation gradients in a real training micro-batch, and would FP16 keep them?
# The 22M model, one 16 x 1,024 micro-batch, BF16 autocast, loss divided by 32 as in train.py.
# usage (from fromscratch/): python bf16_grads.py [checkpoint.pt]
import sys
import torch
from data import ShardLoader
from model import GPT, SMALL

torch.manual_seed(1337)
model = GPT(SMALL).cuda()
if len(sys.argv) > 1:
    model.load_state_dict(torch.load(sys.argv[1], map_location="cuda", weights_only=False)["model"])
stats = []

def watch(module, inputs, output):                      # every linear layer's output is BF16 under autocast
    def count(g):
        a = g.detach().abs().flatten()
        nz = a[a > 0].float()
        stats.append((a.numel(), a.numel() - nz.numel(), (nz < 2**-25).sum().item(), (nz < 2**-14).sum().item(),
                      nz.min().item(), nz.max().item(), nz[::64].cpu()))
    output.register_hook(count)

for m in model.modules():
    if isinstance(m, torch.nn.Linear):
        m.register_forward_hook(watch)

x, y = (t.cuda() for t in ShardLoader("train", B=16, T=1024).batch(5000))
for divide in (32, 1):
    stats.clear()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    (loss / divide).backward()
    n = sum(r[0] for r in stats); zeros = sum(r[1] for r in stats); nnz = n - zeros
    sample = torch.cat([r[6] for r in stats])
    print(f"loss/{divide}: {n:,} activation-gradient values in {len(stats)} BF16 tensors; loss {loss.item():.4f}")
    print(f"  exactly zero in BF16: {zeros / n:.4%}; median |g| {sample.median().item():.3e}, "
          f"smallest nonzero {min(r[4] for r in stats):.3e}, largest {max(r[5] for r in stats):.3e}")
    print(f"  below 2^-25 (would round to 0 in FP16): {sum(r[2] for r in stats) / nnz:.2%}; "
          f"below 2^-14 (FP16 subnormal, precision lost): {sum(r[3] for r in stats) / nnz:.2%}")
    pg = torch.cat([p.grad.flatten().abs() for p in model.parameters()])
    print(f"  parameter gradients (FP32): median {pg[pg > 0].median().item():.3e}, "
          f"below 2^-25: {(pg[pg > 0] < 2**-25).float().mean().item():.2%}")
print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
