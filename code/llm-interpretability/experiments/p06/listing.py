import torch

torch.manual_seed(0)
n, m, k = 5, 2, 20                    # 5 features, 2 hidden dimensions, 20 models trained at once
density = 0.05                        # each feature is nonzero 5% of the time (sparsity S = 0.95)
W = torch.nn.Parameter(torch.randn(k, n, m) * (2 / (n + m)) ** 0.5)
b = torch.nn.Parameter(torch.zeros(k, n))
opt = torch.optim.AdamW([W, b], lr=1e-3)

def run(x):                            # the ReLU output model: x' = ReLU(W^T W x + b)
    h = torch.einsum("bkf,kfh->bkh", x, W)
    return torch.relu(torch.einsum("bkh,kfh->bkf", h, W) + b)

for step in range(10_000):
    x = torch.rand(1024, k, n) * (torch.rand(1024, k, n) < density)
    loss = ((x - run(x)) ** 2).mean(0).sum(-1)     # one loss per model
    opt.zero_grad(); loss.sum().backward(); opt.step()

with torch.no_grad():                  # 2-D toy models get stuck, so keep the best of 20
    x = torch.rand(100_000, k, n) * (torch.rand(100_000, k, n) < density)
    best = ((x - run(x)) ** 2).mean(0).sum(-1).argmin()
Wb = W[best].detach()
print("feature lengths:", Wb.norm(dim=1).numpy().round(2))
angles = torch.atan2(Wb[:, 1], Wb[:, 0]).rad2deg().sort().values
gaps = torch.cat([angles.diff(), 360 + angles[:1] - angles[-1:]])
print("angles between neighbors:", gaps.numpy().round(1))
print("bias:", b[best].detach().numpy().round(2))
