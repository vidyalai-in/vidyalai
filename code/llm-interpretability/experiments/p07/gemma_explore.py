"""Gemma 2 2B (bf16, plain transformers, no weight processing) + Gemma Scope layer_12/width_16k/average_l0_82.
1. Evaluate on 500 held-out FineWeb-Edu docs x 256 tokens: L0, FVE, next-token loss clean vs spliced.
2. Explore 8,000 FineWeb-Edu docs x 256 tokens: top-20 activations per latent, firing counts, and every
   activation of 50 randomly chosen latents (for the interpretability sample).
"""
import sys, time, json
import torch, torch.nn.functional as F
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import sae as S

torch.set_grad_enabled(False)
L = 12
D = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gemma-tokens.pt")
t0 = time.time()
g = S.GemmaRunner()
sae = S.load_gemma_scope(L, "16k", 82)
print("loaded in", round(time.time() - t0, 1), "s; GPU GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
g.attach(L)
B = 16

# ---- 1. evaluation
ev = D["eval"]
acc = {"sq_err": 0.0, "sum_x": 0.0, "sum_sq": 0.0, "n": 0, "l0": 0.0}
ce_clean, ce_sae, doc_l0 = [], [], []


def splice(hs):
    x = hs[:, 1:].float()
    f = sae.encode(x)
    xh = sae.decode(f)
    acc["sq_err"] += (x - xh).pow(2).sum().item()
    acc["sum_x"] = acc["sum_x"] + x.sum((0, 1)).double()
    acc["sum_sq"] += x.pow(2).sum().item()
    acc["n"] += x.shape[0] * x.shape[1]
    l0 = (f > 0).sum(-1).float()
    acc["l0"] += l0.sum().item()
    doc_l0.append(l0.mean(1).cpu())
    hs = hs.clone()
    hs[:, 1:] = xh.to(hs.dtype)
    return hs


def ce(logits, t):
    return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), t[:, 1:].flatten(), reduction="none").view(t.shape[0], -1).mean(1)


t1 = time.time()
for i in range(0, len(ev), B):
    t = ev[i:i + B].long().cuda()
    g.edit = None
    ce_clean.append(ce(g.logits(t), t).cpu())
    g.edit = splice
    ce_sae.append(ce(g.logits(t), t).cpu())
g.edit = None
mu = acc["sum_x"] / acc["n"]
tot_dev = acc["sum_sq"] - acc["n"] * mu.pow(2).sum().item()
res = {"eval_docs": len(ev), "eval_tokens_scored": acc["n"], "fve": 1 - acc["sq_err"] / tot_dev,
       "l0": acc["l0"] / acc["n"], "ce_clean_per_doc": torch.cat(ce_clean).tolist(), "ce_sae_per_doc": torch.cat(ce_sae).tolist(),
       "l0_per_doc": torch.cat(doc_l0).tolist(), "eval_s": round(time.time() - t1, 1)}
print({k: v for k, v in res.items() if not isinstance(v, list)}, "clean", torch.cat(ce_clean).mean().item(),
      "sae", torch.cat(ce_sae).mean().item(), flush=True)

# ---- 2. exploration
ex = D["explore"]
n = sae.n
tk = S.TopK(n, 20)
gen = torch.Generator().manual_seed(7)
sample = torch.randperm(n, generator=gen)[:50].cuda()
samp = []                      # (latent slot, global token index, value)
T = ex.shape[1]
t2 = time.time()
for i in range(0, len(ex), B):
    t = ex[i:i + B].long().cuda()
    h = g.resid(t, L).float()           # [B, T, d]
    f = sae.encode(h)
    f[:, 0] = 0                         # never count BOS
    flat = f.reshape(-1, n)
    tk.update(flat, i * T)
    s = flat[:, sample]
    nz = s.nonzero()
    samp.append(torch.stack([nz[:, 1].float(), (nz[:, 0] + i * T).float(), s[nz[:, 0], nz[:, 1]]], 1).cpu())
print("explore", len(ex), "docs in", round(time.time() - t2, 1), "s; peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
torch.save({"eval": res, "top_v": tk.v.cpu(), "top_i": tk.i.cpu(), "count": tk.count.cpu(), "sample": sample.cpu(),
            "sample_acts": torch.cat(samp), "T": T, "n_tokens": len(ex) * (T - 1), "sae": sae.source},
           "/home/amit/.cache/vidyalai-scratch/p07/runs/gemma_explore.pt")
print("total", round(time.time() - t0, 1), "s")
