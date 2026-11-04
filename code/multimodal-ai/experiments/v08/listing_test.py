import torch
import torch.nn.functional as F

torch.manual_seed(0)
z = torch.randn(49, 64, requires_grad=True)          # 7 x 7 encoder outputs, 64 numbers each
codebook = torch.randn(512, 64, requires_grad=True)  # 512 learned entries
dist = (z ** 2).sum(1, keepdim=True) - 2 * z @ codebook.T + (codebook ** 2).sum(1)
idx = dist.argmin(1)                                 # the image tokens: 49 integers
zq = codebook[idx]
vq_loss = F.mse_loss(zq, z.detach()) + 0.25 * F.mse_loss(z, zq.detach())  # codebook + commitment
zq = z + (zq - z).detach()                           # straight-through: forward uses zq, gradient reaches z
print(idx.shape, idx[:6].tolist())
