"""Combine s6 (domain counts) and s7 (token types) into a per-neuron classification. CPU. Writes runs/s7_analysis.json.
Sparse neuron (as s6): >= 20 firing tokens and < 1% of the 203,200 scored tokens.
  one-domain:        >= 90% of firings in one domain... reported separately; here 'span' = domains with >= 10% of firings
  token-specific:    one normalized token type accounts for >= 50% of its firings
  mixed:             spans >= 2 domains, no type reaches 50%, and the top type differs between its two biggest domains"""
import json, math
import torch

RUN = "/home/amit/.cache/vidyalai-scratch/p06/runs/"
z = torch.load(RUN + "s6_counts.pt")
e = torch.load(RUN + "s7_events.pt")
DOMS, NW, T = z["doms"], z["NW"], z["T"]
nd, L, H, V = len(DOMS), e["L"], e["H"], e["V"]
types = e["types"]
ntok = nd * NW * T
keys, cnt = e["keys"], e["counts"].long()
tid = keys % V
nd_key = keys // V                      # neuron*nd + domain
neuron = nd_key // nd
dom = nd_key % nd
N = L * H
# per neuron totals per domain
Fd = torch.zeros(N * nd, dtype=torch.long).index_add_(0, nd_key, cnt).view(N, nd)
tot = Fd.sum(1)
# overall per (neuron, type): aggregate across domains
k2 = neuron * V + tid
u2, inv = k2.unique(return_inverse=True)
c2 = torch.zeros(len(u2), dtype=torch.long).index_add_(0, inv, cnt)
n2 = u2 // V
# top type share per neuron
order = torch.argsort(n2 * (2**40) + (2**31 - c2))   # by neuron, then count descending
n2s, c2s, t2s = n2[order], c2[order], (u2 % V)[order]
first = torch.ones(len(n2s), dtype=torch.bool); first[1:] = n2s[1:] != n2s[:-1]
top_cnt = torch.zeros(N, dtype=torch.long); top_cnt[n2s[first]] = c2s[first]
top_type = torch.full((N,), -1, dtype=torch.long); top_type[n2s[first]] = t2s[first]
ntypes = torch.zeros(N, dtype=torch.long).index_add_(0, n2, torch.ones_like(n2))
# top type per (neuron, domain)
order = torch.argsort(nd_key * (2**40) + (2**31 - cnt))
kds, ts = nd_key[order], tid[order]
first = torch.ones(len(kds), dtype=torch.bool); first[1:] = kds[1:] != kds[:-1]
top_type_d = torch.full((N * nd,), -1, dtype=torch.long); top_type_d[kds[first]] = ts[first]
top_type_d = top_type_d.view(N, nd)

sparse = (tot >= 20) & (tot < 0.01 * ntok)
share = Fd.float() / tot.clamp_min(1)[:, None]
span = (share >= 0.10).sum(1)
topshare = top_cnt.float() / tot.clamp_min(1)
tokspec = topshare >= 0.5
two = share.argsort(1, descending=True)[:, :2]
t_a = torch.gather(top_type_d, 1, two[:, :1])[:, 0]; t_b = torch.gather(top_type_d, 1, two[:, 1:])[:, 0]
mixed = sparse & (span >= 2) & ~tokspec & (t_a != t_b)
multi = sparse & (span >= 2)


def wilson(k, n):
    p = k / n; zz = 1.96
    c = (p + zz * zz / (2 * n)) / (1 + zz * zz / n); h = zz * math.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)) / (1 + zz * zz / n)
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


ns = int(sparse.sum())
res = dict(n_sparse=ns, n_multi=int(multi.sum()),
           multi_share=wilson(int(multi.sum()), ns),
           multi_tokspec_share_of_multi=wilson(int((multi & tokspec).sum()), int(multi.sum())),
           mixed_share_of_sparse=wilson(int(mixed.sum()), ns),
           tokspec_share_of_sparse=wilson(int((sparse & tokspec).sum()), ns),
           median_types_sparse=float(ntypes[sparse].float().median()),
           median_topshare_sparse=float(topshare[sparse].median()))
lay = []
for l in range(L):
    s = slice(l * H, (l + 1) * H)
    n = int(sparse[s].sum())
    lay.append(dict(layer=l, n_sparse=n, multi=round(int(multi[s].sum()) / n, 4), mixed=round(int(mixed[s].sum()) / n, 4),
                    tokspec=round(int((sparse[s] & tokspec[s]).sum()) / n, 4)))
res["layers"] = lay
# examples of mixed neurons: highest firing counts, spread over layers
ex = []
idx = mixed.nonzero().flatten()
idx = idx[tot[idx].argsort(descending=True)]
for g in idx[:60].tolist():
    l, h = divmod(g, H)
    m = keys // V // nd == g
    kk, cc = keys[m], cnt[m]
    per = {}
    for k, d in enumerate(DOMS):
        mm = (kk // V) % nd == k
        if mm.sum() == 0: continue
        o = cc[mm].argsort(descending=True)[:4]
        per[d] = [(types[int(kk[mm][j] % V)], int(cc[mm][j])) for j in o]
    ex.append(dict(neuron=f"{l}.{h}", fires=int(tot[g]), rate=round(float(tot[g]) / ntok, 5),
                   shares={d: round(float(share[g, k]), 3) for k, d in enumerate(DOMS)}, top_share=round(float(topshare[g]), 3),
                   n_types=int(ntypes[g]), top=per))
res["examples_mixed"] = ex
json.dump(res, open(RUN + "s7_analysis.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "examples_mixed"}, indent=1))
for x in ex[:25]:
    print(x["neuron"], x["fires"], x["shares"], x["top_share"], {d: [t for t, _ in v[:3]] for d, v in x["top"].items()})
