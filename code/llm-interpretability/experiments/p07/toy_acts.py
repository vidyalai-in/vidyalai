import torch

pre = torch.tensor([3.1, 0.2, -1.0, 1.8, 0.4, 0.0])   # encoder outputs for 6 latents

relu = pre.clamp(min=0)                                # ReLU: keep every positive value
v, i = pre.topk(2)                                     # TopK with k = 2: keep the 2 largest
topk = torch.zeros_like(pre).scatter(0, i, v.clamp(min=0))
jump = torch.where(pre > 1.0, pre, 0.0)                 # JumpReLU with threshold 1.0

for name, f in [("ReLU", relu), ("TopK", topk), ("JumpReLU", jump)]:
    print(f"{name:8s} {[round(x, 1) for x in f.tolist()]}  L0 = {int((f > 0).sum())}")
