"""Evaluate GPT-2 small layer-8 SAEs on Part 2's 586 held-out FineWeb-Edu documents (4,231 windows of 128 tokens).
Ours (runs/main.saes.pt) and two public ones via sae-lens: gpt2-small-res-jb blocks.8.hook_resid_pre (Bloom 2024)
and gpt2-small-resid-post-v5-32k blocks.7.hook_resid_post (OpenAI, Gao et al. 2024; same tensor).
Also: firing counts per latent and top-20 activating tokens per latent, for our SAEs.
"""
import sys, json, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import sae as S
from transformer_lens import HookedTransformer
from sae_lens import SAE as LensSAE

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
R = "/home/amit/.cache/vidyalai-scratch/p07/runs"
D = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gpt2-tokens.pt")
ev, doc = D["eval"], D["eval_doc"].numpy()
model = HookedTransformer.from_pretrained("gpt2", device="cuda")
L = 8

# mean activation over the eval set (BOS excluded)
s, n = 0, 0
for j in range(0, len(ev), 256):
    h = model(ev[j:j + 256].long().cuda(), stop_at_layer=L)[:, 1:]
    s = s + h.sum((0, 1)); n += h.shape[0] * h.shape[1]
mu = s / n

saes = {}
ck = torch.load(f"{R}/main.saes.pt")
for k, sd in ck["sae"].items():
    c = ck["cfg"]["saes"][k]
    m = S.SAE(768, c["n"], kind=c["kind"], k=c.get("k"), l1=c.get("l1", 0.0)).cuda()
    m.load_state_dict(sd); saes[k] = m
saes["jb_24k"] = LensSAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre", device="cuda")
saes["oai_topk32_32k"] = LensSAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post", device="cuda")

base = None
results, extra = {"trained_steps": ck["step"], "eval_windows": len(ev), "eval_docs": int(len(np.unique(doc)))}, {}
for k, m in saes.items():
    t0 = time.time()
    nl = m.W_dec.shape[0]
    counts = torch.zeros(nl, device="cuda")
    tk = S.TopK(nl, 20) if k in ("topk32_24k", "topk32_3k", "relu_l1_15_24k") else None
    off = {"o": 0}

    def ed(x, m=m):
        f = m.encode(x)
        return m.decode(f), f

    def ex(f, x):
        counts.add_((f > 0).sum(0))
        if tk is not None:
            tk.update(f, off["o"]); off["o"] += f.shape[0]

    r = S.evaluate_gpt2(model, L, ev, ed, mean_act=mu, extra=ex, baselines=(base is None))
    if base is None:
        base = {k: r[k] for k in ("ce_clean", "ce_zero", "ce_mean")}
    r.update(base)
    agg = lambda st: S.doc_bootstrap(r, doc, st)
    out = {
        "fve": agg(lambda d: 1 - d["sq_err"].sum() / d["sq_dev"].sum()),
        "l0": agg(lambda d: d["l0"].sum() / d["n"].sum()),
        "ce_clean": agg(lambda d: d["ce_clean"].sum() / (d["n"].sum() + 0 * d["n"].sum())),
        "ce_sae": agg(lambda d: d["ce_sae"].sum() / d["n"].sum()),
        "delta_ce": agg(lambda d: (d["ce_sae"].sum() - d["ce_clean"].sum()) / d["n"].sum()),
        "ce_zero": agg(lambda d: d["ce_zero"].sum() / d["n"].sum()),
        "ce_mean": agg(lambda d: d["ce_mean"].sum() / d["n"].sum()),
        "frac_recovered_vs_zero": agg(lambda d: (d["ce_zero"].sum() - d["ce_sae"].sum()) / (d["ce_zero"].sum() - d["ce_clean"].sum())),
        "frac_recovered_vs_mean": agg(lambda d: (d["ce_mean"].sum() - d["ce_sae"].sum()) / (d["ce_mean"].sum() - d["ce_clean"].sum())),
        "latents": nl, "dead_on_eval": int((counts == 0).sum()), "dense_over_10pct": int((counts / n > 0.1).sum()),
        "eval_s": round(time.time() - t0, 1),
    }
    results[k] = out
    extra[k] = {"counts": counts.cpu()}
    if tk is not None:
        extra[k].update(top_v=tk.v.cpu(), top_i=tk.i.cpu())
    torch.cuda.empty_cache()
    print(k, {a: (tuple(round(x, 4) for x in b) if isinstance(b, tuple) else b) for a, b in out.items()}, flush=True)

# CE loss per token position is over positions 0..126 predicting 1..127; BOS position never replaced
results["n_tokens_scored"] = int(n)
json.dump(results, open(f"{R}/gpt2_eval.json", "w"), indent=1)
torch.save(extra, f"{R}/gpt2_eval_extra.pt")
