# CPU analysis for experiment B (model diffing with the base-model Gemma Scope SAE). Writes runs/diff.json
# and runs/diff_contexts.txt (top contexts of the most-changed latents; kept in scratch).
import json, numpy as np, torch
from transformers import AutoTokenizer
from lab import safety as S

D = "/home/amit/.cache/vidyalai-scratch/p11/data/diff"
tok = AutoTokenizer.from_pretrained(S.IT)
C = torch.load(f"{D}/chat_tokens.pt")
T = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gemma-tokens.pt")
seqs = {"plain": [r.long() for r in T["eval"][:300]], "chat": [s.long() for s in C["seqs"]]}
R = {"quality": {}, "latents": {}, "cos": {}}
CATS = {"plain": {"all text": [4]}, "chat": {"all": [1, 2, 3], "template": [1], "user text": [2], "model reply": [3]}}


def boot(P, mask, stat, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    seq = P[:, 0].astype(int); us = np.unique(seq[mask])
    by = {s: np.where((seq == s) & mask)[0] for s in us}
    est = stat(np.where(mask)[0]); bs = []
    for _ in range(n):
        idx = np.concatenate([by[s] for s in rng.choice(us, len(us))])
        bs.append(stat(idx))
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


for which in ["base", "it"]:
    for name in ["plain", "chat"]:
        P = np.load(f"{D}/{which}_{name}_tok.npy")
        lat = torch.load(f"{D}/{which}_{name}_lat.pt")
        for cname, cs in CATS[name].items():
            m = np.isin(P[:, 2], cs)
            # variance about the mean of THIS model's activations on all non-BOS tokens of this set
            k = P[:, 2] > 0
            mean = (lat["sx"].numpy() / k.sum())
            msq = float((mean ** 2).sum())
            fve = lambda idx: 1 - P[idx, 4].sum() / (P[idx, 3].sum() - len(idx) * msq)
            ok = m & ~np.isnan(P[:, 6])
            dce = lambda idx: float(np.nanmean(P[idx, 7] - P[idx, 6]))
            R["quality"][f"{which}/{name}/{cname}"] = {
                "tokens": int(m.sum()), "fve": boot(P, m, fve, n=300), "l0": float(P[m, 5].mean()),
                "norm": float(np.sqrt(P[m, 3]).mean()),
                "ce": float(np.nanmean(P[ok, 6])), "ce_spliced": float(np.nanmean(P[ok, 7])), "dce": boot(P, ok, dce, n=300)}
            q = R["quality"][f"{which}/{name}/{cname}"]
            print(which, name, cname, q["tokens"], "FVE %.3f (%.3f-%.3f)" % q["fve"], "L0 %.1f" % q["l0"], "norm %.0f" % q["norm"],
                  "CE %.3f -> %.3f dCE %.3f (%.3f-%.3f)" % ((q["ce"], q["ce_spliced"]) + q["dce"]))

# latent firing frequencies, base vs IT, same tokens
for name in ["plain", "chat"]:
    lb, li = torch.load(f"{D}/base_{name}_lat.pt"), torch.load(f"{D}/it_{name}_lat.pt")
    cats = CATS[name]["all text" if name == "plain" else "all"]
    Pn = np.load(f"{D}/base_{name}_tok.npy")
    ntok = {c: int((Pn[:, 2] == c).sum()) for c in cats}
    fb = sum(lb["cnt"][c] for c in cats).numpy() / sum(ntok.values())
    fi = sum(li["cnt"][c] for c in cats).numpy() / sum(ntok.values())
    ab = sum(lb["sum"][c] for c in cats).numpy() / sum(ntok.values())
    ai = sum(li["sum"][c] for c in cats).numpy() / sum(ntok.values())
    diff = fi - fb
    corr = float(np.corrcoef(fb, fi)[0, 1])
    lr = np.log10((fi + 1e-5) / (fb + 1e-5))
    R["latents"][name] = {"freq_corr": corr, "n_latents": len(fb),
                          "alive_base": int((fb > 0).sum()), "alive_it": int((fi > 0).sum()),
                          "it_only_10x": int(((fi > 10 * fb) & (fi > 1e-3)).sum()),
                          "base_only_10x": int(((fb > 10 * fi) & (fb > 1e-3)).sum()),
                          "up": [], "down": []}
    for key, order in [("up", np.argsort(-diff)[:15]), ("down", np.argsort(diff)[:15])]:
        for j in order:
            R["latents"][name][key].append({"latent": int(j), "freq_base": float(fb[j]), "freq_it": float(fi[j]),
                                            "mean_base": float(ab[j]), "mean_it": float(ai[j])})
    if name == "chat":   # by category, for the top movers
        for key in ["up", "down"]:
            for e in R["latents"][name][key]:
                j = e["latent"]
                e["by_cat_it"] = {c: float(li["cnt"][c][j] / max(ntok[c], 1)) for c in cats}
                e["by_cat_base"] = {c: float(lb["cnt"][c][j] / max(ntok[c], 1)) for c in cats}
    print(name, "freq corr", round(corr, 3), {k: v for k, v in R["latents"][name].items() if k not in ("up", "down")})

# contexts for the top movers (IT model's strongest activations on the chat set; base model's for "down")
with open("/home/amit/.cache/vidyalai-scratch/p11/runs/diff_contexts.txt", "w") as fh:
    for name in ["chat", "plain"]:
        flat = torch.cat(seqs[name]); starts = np.cumsum([0] + [len(s) for s in seqs[name]])
        for key, src in [("up", "it"), ("down", "base")]:
            lat = torch.load(f"{D}/{src}_{name}_lat.pt")
            for e in R["latents"][name][key][:12]:
                j = e["latent"]
                fh.write(f"\n== {name} {key} latent {j} base {e['freq_base']:.4f} it {e['freq_it']:.4f}\n")
                for v, gi in zip(lat["top_v"][j][:5].tolist(), lat["top_i"][j][:5].tolist()):
                    if gi < 0: continue
                    s = int(np.searchsorted(starts, gi, side="right") - 1)
                    lo = max(starts[s] + 1, gi - 12)
                    ctx = tok.decode(flat[lo:gi].tolist()).replace("\n", "\\n")
                    t = tok.decode(flat[gi:gi + 1].tolist()).replace("\n", "\\n")
                    fh.write(f"  {v:6.1f} ...{ctx[-80:]}[[{t}]]\n")

# how different are the two models' residual streams at layer 12, token by token?
rb = np.load(f"{D}/base_chat_resid.npy", mmap_mode="r"); ri = np.load(f"{D}/it_chat_resid.npy", mmap_mode="r")
P = np.load(f"{D}/base_chat_tok.npy")
cos = np.zeros(len(P)); nr = np.zeros(len(P))
for s in range(0, len(P), 4096):
    a = rb[s:s + 4096].astype(np.float32); b = ri[s:s + 4096].astype(np.float32)
    cos[s:s + 4096] = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)
    nr[s:s + 4096] = np.linalg.norm(b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-6)
for cname, cs in CATS["chat"].items():
    m = np.isin(P[:, 2], cs)
    R["cos"][cname] = {"median_cos": float(np.median(cos[m])), "p10_cos": float(np.percentile(cos[m], 10)),
                       "median_norm_ratio": float(np.median(nr[m]))}
print("cos", R["cos"])
json.dump(R, open("/home/amit/.cache/vidyalai-scratch/p11/runs/diff.json", "w"), indent=1)
