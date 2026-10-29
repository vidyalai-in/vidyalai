"""Analyze s6_counts.pt (CPU). Writes runs/s6_analysis.json.
A token 'fires' neuron n if its post-GELU activation exceeds half the neuron's max over all 203,200 scored tokens.
Sparse neuron: fires on 20 or more tokens but under 1% of them. Domains spanned: domains holding >= 10% of its firings."""
import json, sys
import torch
from transformers import AutoTokenizer

RUN = "/home/amit/.cache/vidyalai-scratch/p06/runs/"
z = torch.load(RUN + "s6_counts.pt")
win = torch.load(RUN + "s6_windows.pt")
tok = AutoTokenizer.from_pretrained("gpt2")
C = z["C"].float()                                    # [1600, L, H]
NW, T, DOMS = z["NW"], z["T"], z["doms"]
nd, L, H = len(DOMS), C.shape[1], C.shape[2]
C = C.reshape(nd, NW, L * H)                          # domain-major
ntok = nd * NW * T


def stats(F, share_min=0.10, rate_max=0.01, min_fires=20):
    tot = F.sum(0)
    sparse = (tot >= min_fires) & (tot / ntok < rate_max)
    sh = F / tot.clamp_min(1)
    span = (sh >= share_min).sum(0)
    return tot, sparse, span, sh


def per_layer(sparse, span):
    s = sparse.view(L, H); sp = span.view(L, H)
    n = s.sum(1).float()
    return dict(n_sparse=n.tolist(),
                share_2plus=((s & (sp >= 2)).sum(1) / n).tolist(),
                share_3plus=((s & (sp >= 3)).sum(1) / n).tolist(),
                share_4=((s & (sp >= 4)).sum(1) / n).tolist())


F = C.sum(1)                                          # [nd, L*H]
tot, sparse, span, sh = stats(F)
res = dict(ntok=ntok, rule="fires: act > 0.5*max; sparse: >=20 firing tokens and <1% of tokens; span: >=10% of firings",
           layers=per_layer(sparse, span))
s_all = sparse.sum().item()
res["overall"] = dict(n_sparse=s_all, share_2plus=((sparse & (span >= 2)).sum() / s_all).item(),
                      share_3plus=((sparse & (span >= 3)).sum() / s_all).item(),
                      n_dense=(tot / ntok >= 0.01).sum().item(), n_rare=(tot < 20).sum().item())
# single-domain specialists among sparse neurons
dom1 = sh.argmax(0)
res["one_domain_90"] = {d: ((sparse & (sh.max(0).values >= 0.9) & (dom1 == k)).sum().item()) for k, d in enumerate(DOMS)}

# bootstrap over windows (resample 400 windows within each domain)
g = torch.Generator().manual_seed(0)
B = 300
boot = {"share_2plus": [], "share_3plus": [], "layer_2plus": []}
for b in range(B):
    Fb = torch.stack([C[k][torch.randint(0, NW, (NW,), generator=g)].sum(0) for k in range(nd)])
    tb, sb, spb, _ = stats(Fb)
    n = sb.sum().item()
    boot["share_2plus"].append(((sb & (spb >= 2)).sum() / n).item())
    boot["share_3plus"].append(((sb & (spb >= 3)).sum() / n).item())
    boot["layer_2plus"].append(per_layer(sb, spb)["share_2plus"])


def ci(v):
    v = torch.tensor(v)
    return [torch.quantile(v, 0.025, dim=0).tolist(), torch.quantile(v, 0.975, dim=0).tolist()]


res["boot"] = dict(B=B, share_2plus=ci(boot["share_2plus"]), share_3plus=ci(boot["share_3plus"]),
                   layer_2plus=ci(boot["layer_2plus"]))

# control: shuffle domain labels across windows
perm = torch.randperm(nd * NW, generator=g)
Cs = C.reshape(nd * NW, -1)[perm].reshape(nd, NW, -1)
_, ss, sps, _ = stats(Cs.sum(1))
res["shuffled_labels"] = dict(share_2plus=((ss & (sps >= 2)).sum() / ss.sum()).item(),
                              share_4=((ss & (sps >= 4)).sum() / ss.sum()).item())

# robustness to the cut-offs
rob = {}
for share_min in (0.05, 0.10, 0.20):
    for rate_max in (0.005, 0.01, 0.02):
        _, s2, sp2, _ = stats(F, share_min, rate_max)
        rob[f"share>={share_min},rate<{rate_max}"] = ((s2 & (sp2 >= 2)).sum() / s2.sum()).item()
res["robustness_2plus"] = rob

# examples: sparse neurons spanning 3+ domains, ranked by their smallest top-3 domain share
toks = win["tokens"]
bv, bi = z["best_v"], z["best_i"]                     # [nd, L, H, K]


def token_at(idx):
    w, p = divmod(int(idx), T)
    return tok.decode([int(toks[w, p + 1])])


cand = (sparse & (span >= 3)).nonzero().flatten()
order = sh[:, cand].sort(0, descending=True).values[2].argsort(descending=True)
ex = []
for c in cand[order][:40].tolist():
    l, h = divmod(c, H)
    e = dict(neuron=f"L{l}.N{h}", layer=l, idx=h, fire_tokens=int(tot[c]), fire_rate=float(tot[c] / ntok),
             shares={d: round(float(sh[k, c]), 3) for k, d in enumerate(DOMS)}, max=float(z["max"][l, h]), top={})
    for k, d in enumerate(DOMS):
        e["top"][d] = [(token_at(i), round(float(v), 2)) for v, i in zip(bv[k, l, h], bi[k, l, h])]
    ex.append(e)
res["examples"] = ex
json.dump(res, open(RUN + "s6_analysis.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
for e in ex[:12]:
    print(e["neuron"], e["fire_tokens"], e["shares"], {d: [t for t, _ in e["top"][d][:3]] for d in DOMS})
