# Part 7: the never-seen token ids: which tokens, and their final row norms (CPU only).
import sys, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from data import ShardLoader
from tokenizer import load_production
tr = ShardLoader("train"); seen = 3360 * 524288 + 1; cnt = np.zeros(32768, np.int64); pos = 0
for s in tr.shards:
    n = min(len(s), seen - pos)
    if n <= 0: break
    cnt += np.bincount(np.asarray(s[:n]), minlength=32768); pos += n
z = np.where(cnt == 0)[0]; tok = load_production()
E = torch.load("/home/amit/.cache/vidyalai-scratch/t07/runs/main88/model_03360.pt", map_location="cpu", weights_only=True)["tok_emb.weight"].norm(dim=1).numpy()
print(len(z), "mean norm", E[z].mean().round(3), "min", E[z].min().round(3), "max", E[z].max().round(3), "all-rows mean", E.mean().round(3))
print([tok.id_to_token(int(i)) for i in z])
