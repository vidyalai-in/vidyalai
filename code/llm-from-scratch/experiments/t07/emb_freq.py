# Part 7: final embedding row norms against token frequency in the training tokens the run saw; unigram entropy of val (CPU only).
import sys, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from data import ShardLoader
from tokenizer import load_production
tr = ShardLoader("train"); va = ShardLoader("val")
print("train tokens", tr.n_tokens, "val tokens", va.n_tokens)
seen = 3360 * 524288 + 1
cnt = np.zeros(32768, np.int64); pos = 0
for s in tr.shards:
    n = min(len(s), seen - pos)
    if n <= 0: break
    cnt += np.bincount(np.asarray(s[:n]), minlength=32768); pos += n
print("counted", pos, "tokens; never-seen ids:", int((cnt == 0).sum()))
vc = np.bincount(np.asarray(np.concatenate(va.shards)), minlength=32768).astype(float)
p = vc / vc.sum(); H = -(p[p > 0] * np.log(p[p > 0])).sum(); print(f"val unigram entropy {H:.4f} nats")
pt = cnt / cnt.sum(); Hx = -(p[p > 0] * np.log(np.maximum(pt[p > 0], 1e-12))).sum(); print(f"cross-entropy of val under train unigram {Hx:.4f}")
R = "/home/amit/.cache/vidyalai-scratch/t07/runs/main88"
E0 = torch.load(f"{R}/model_00000.pt", map_location="cpu", weights_only=True)["tok_emb.weight"].norm(dim=1).numpy()
E1 = torch.load(f"{R}/model_03360.pt", map_location="cpu", weights_only=True)["tok_emb.weight"].norm(dim=1).numpy()
tok = load_production()
order = np.argsort(-cnt)
for name, sl in (("top 100", order[:100]), ("100-1000", order[100:1000]), ("1000-10000", order[1000:10000]), ("10000-30000", order[10000:30000]), ("rarest 2768", order[30000:])):
    print(f"  rank {name:12s}: count median {np.median(cnt[sl]):>10.0f}  row norm start {E0[sl].mean():.3f} -> end {E1[sl].mean():.3f}")
seen_ids = cnt > 0
print("spearman-ish corr(log count, end norm):", np.corrcoef(np.log1p(cnt), E1)[0,1].round(3))
z = np.where(cnt == 0)[0]
print("never seen:", len(z), "end norms", np.round(E1[z][:20], 3), [tok.id_to_token(int(i)) for i in z[:20]])
print("smallest end norms:", [(tok.id_to_token(int(i)), int(cnt[i]), round(float(E1[i]),3)) for i in np.argsort(E1)[:10]])
print("largest end norms:", [(tok.id_to_token(int(i)), int(cnt[i]), round(float(E1[i]),3)) for i in np.argsort(-E1)[:10]])
print("most frequent:", [(tok.id_to_token(int(i)), int(cnt[i]), round(float(E1[i]),3)) for i in order[:8]])
