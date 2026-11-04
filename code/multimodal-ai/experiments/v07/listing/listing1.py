import torch
import torch.nn.functional as F

def flow_loss(model, x1):                       # x1: a batch of images in [-1, 1]
    x0 = torch.randn_like(x1)                   # noise
    t = torch.rand(x1.shape[0], device=x1.device)
    tt = t[:, None, None, None]
    xt = (1 - tt) * x0 + tt * x1                # a point on the straight line
    return F.mse_loss(model(xt, t), x1 - x0)    # predict the velocity

@torch.no_grad()
def euler_sample(model, z, steps):              # z: noise, same shape as the images
    x, ts = z, torch.linspace(0, 1, steps + 1, device=z.device)
    for k in range(steps):
        x = x + (ts[k + 1] - ts[k]) * model(x, ts[k].expand(len(x)))
    return x

# test harness (not in the article)
import sys; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab"); import flow as F2
m = F2.UNet()
x1 = torch.rand(4, 1, 28, 28) * 2 - 1
print("loss", flow_loss(m, x1).item())
z = torch.randn(4, 1, 28, 28)
print(euler_sample(m, z, 3).shape)
torch.manual_seed(0); a = flow_loss(m, x1); torch.manual_seed(0); b = F2.flow_loss(m, x1); print("matches lab module", torch.allclose(a, b))
