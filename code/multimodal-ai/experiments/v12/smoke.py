import sys, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import video as V, fmnist_eval as fe
x, y = fe.load("test")
g = torch.Generator().manual_seed(0)
d = V.moving_fmnist(x, 200, frames=8, generator=g, labels=y)
print(d["x"].shape, d["x"].min().item(), d["x"].max().item(), d["py"][0].tolist(), d["px"][0].tolist(), d["v"][0].tolist())
m = V.clip_metrics(d["x"])
print("same", m["same"].mean().item(), "agree", m["agree"].mean().item(), "cos0", m["cos0"].mean().item(), "speed", m["speed"].mean().item(), "accel", m["accel"].mean().item())
lab0 = m["labels"][:,0]; print("frame0 acc vs true", (lab0==d["y"]).float().mean().item())
for mode in ["frames","factorized","full"]:
    net = V.VideoDiT(mode)
    out = net(d["x"][:2], torch.rand(2))
    print(mode, V.n_params(net), out.shape)
# static check: still image of the item each frame
