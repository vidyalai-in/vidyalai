# Summarize attribution patching vs activation patching (runs/ap_ioi.pt -> post/ap_ioi.json).
import json, torch
from scipy.stats import pearsonr, spearmanr

r = torch.load("/home/amit/.cache/vidyalai-scratch/p08/runs/ap_ioi.pt")
base = torch.cat(r["base"])                      # [n, 2]: clean LD, corrupted LD
n = base.shape[0]
out = dict(n_prompts=n, ld_clean=base[:, 0].mean().item(), ld_corr=base[:, 1].mean().item(),
           timing=r["timing"])
g = torch.Generator().manual_seed(0)
B = 1000
idx_all = torch.randint(0, n, (B, n), generator=g)


def norm(x, idx):
    gap = (base[idx, 0] - base[idx, 1]).mean()
    return x[idx].mean(0) / gap                   # share of the clean-corrupted gap restored


for k in ["heads", "resid", "attn", "mlp"]:
    # templates differ in length for resid, so handle per template there
    acts, attrs = r[k]["act"], r[k]["attr"]
    if k == "resid":
        # align positions from the end (the last tokens " gave a X to" share roles)
        m = min(a.shape[-1] for a in acts)
        acts = [a[..., -m:] for a in acts]; attrs = [a[..., -m:] for a in attrs]
    act, attr = torch.cat(acts), torch.cat(attrs)
    full = torch.arange(n)
    a0, b0 = norm(act, full).flatten().numpy(), norm(attr, full).flatten().numpy()
    res = dict(n_components=len(a0), pearson=pearsonr(a0, b0)[0], spearman=spearmanr(a0, b0)[0])
    ps = []
    for b in range(B):
        a1, b1 = norm(act, idx_all[b]).flatten().numpy(), norm(attr, idx_all[b]).flatten().numpy()
        ps.append(pearsonr(a1, b1)[0])
    ps = torch.tensor(ps)
    res["pearson_ci"] = [ps.quantile(0.025).item(), ps.quantile(0.975).item()]
    if k == "heads":
        A, T = norm(act, full), norm(attr, full)
        top = A.flatten().abs().argsort(descending=True)[:10]
        res["top_by_act"] = [(f"{i // 12}.{i % 12}", round(A.flatten()[i].item(), 3), round(T.flatten()[i].item(), 3)) for i in top.tolist()]
        topa = T.flatten().abs().argsort(descending=True)[:10]
        res["top_by_attr"] = [f"{i // 12}.{i % 12}" for i in topa.tolist()]
        res["top10_overlap"] = len(set(top.tolist()) & set(topa.tolist()))
        res["points"] = [(i // 12, i % 12, round(A.flatten()[i].item(), 4), round(T.flatten()[i].item(), 4)) for i in range(144)]
    if k in ("attn", "mlp"):
        res["per_layer"] = [(l, round(norm(act, full)[l].item(), 3), round(norm(attr, full)[l].item(), 3)) for l in range(12)]
    if k == "resid":
        res["n_pos_aligned"] = m
        A, T = norm(act, full), norm(attr, full)
        res["layer0"] = [(round(A[0, p].item(), 3), round(T[0, p].item(), 3)) for p in range(m)]
        # excluding layer 0
        a2, b2 = A[1:].flatten().numpy(), T[1:].flatten().numpy()
        res["pearson_excl_layer0"] = pearsonr(a2, b2)[0]
    out[k] = res
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/p08/post/ap_ioi.json", "w"), indent=1, default=float)
for k in ["heads", "resid", "attn", "mlp"]:
    print(k, {kk: v for kk, v in out[k].items() if kk not in ("points",)})
print({k: out[k] for k in ["n_prompts", "ld_clean", "ld_corr", "timing"]})
