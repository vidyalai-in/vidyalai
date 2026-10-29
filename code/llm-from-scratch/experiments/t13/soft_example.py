# What one soft target holds: the 88M teacher's next-token distribution after a short prefix, at T = 1 and T = 2 (CPU).
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from snapshots import load
from tokenizer import load_production
tok = load_production()
m = load('/home/amit/.cache/vidyalai-scratch/t07/runs/main88/model_03360.pt', device='cpu')
for text in ["Plants make their own food through a process called", "The heart pumps blood through the"]:
    ids = [0] + tok.encode(text).ids
    with torch.no_grad():
        z = m(torch.tensor([ids]))[0, -1].float()
    for T in (1.0, 2.0):
        p = F.softmax(z / T, -1); v, i = p.topk(6)
        H = -(p * p.clamp_min(1e-30).log()).sum().item()
        print(text, "| T =", T, "| entropy", round(H, 2), "nats |", [(tok.decode([j]), round(x, 3)) for x, j in zip(v.tolist(), i.tolist())],
              "| top-6 mass", round(v.sum().item(), 3), "| tokens with p > 1e-4:", int((p > 1e-4).sum()))
