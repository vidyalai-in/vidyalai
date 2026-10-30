"""Audit of Part 7's headline: our TopK SAE (GPT-2 small, layer 8) vs two public SAEs, on text
other than FineWeb-Edu.  -> runs/sae_gpt2.json"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import sae as S, replicate as R
from transformer_lens import HookedTransformer
from sae_lens import SAE as LensSAE

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2", device="cuda")
L, BOS = 8, 50256
fw = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gpt2-tokens.pt")
wiki = np.load("/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_gpt2.npy")
art = np.load("/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_gpt2_article.npy")
wt, wg = R.windows_from(wiki, 127, BOS, group=art)
sel = np.random.default_rng(0).choice(len(wt), 1200, replace=False)
s6 = torch.load("/home/amit/.cache/vidyalai-scratch/p06/runs/s6_windows.pt", weights_only=False)["tokens"]
DATA = {  # name: (tokens [n,128] with BOS first, bootstrap group per window)
    "fineweb_edu": (fw["eval"], fw["eval_doc"].numpy()),
    "wikitext": (wt[sel], wg[sel]),
    "german_wiki": (s6[400:800], np.arange(400)),
    "python": (s6[1200:1600], np.arange(400)),
    "korean_wiki": (s6[800:1200], np.arange(400)),
}
ck = torch.load("/home/amit/.cache/vidyalai-scratch/p07/runs/main.saes.pt")
c = ck["cfg"]["saes"]["topk32_24k"]
ours = S.SAE(768, c["n"], kind=c["kind"], k=c.get("k"), l1=c.get("l1", 0.0)).cuda()
ours.load_state_dict(ck["sae"]["topk32_24k"])
SAES = {"ours_topk_24k": ours,
        "bloom_relu_24k": LensSAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre", device="cuda"),
        "openai_topk_32k": LensSAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post", device="cuda")}
out = {"n_windows": {k: int(len(v[0])) for k, v in DATA.items()}}
for dname, (tok, grp) in DATA.items():
    s, n = 0, 0
    for j in range(0, len(tok), 256):
        h = model(tok[j:j + 256].long().cuda(), stop_at_layer=L)[:, 1:]
        s = s + h.sum((0, 1)); n += h.shape[0] * h.shape[1]
    mu = s / n
    out[dname] = {"tokens_scored": int(n)}
    base = None
    for sname, m in SAES.items():
        t1 = time.time()
        r = S.evaluate_gpt2(model, L, tok, lambda x, m=m: (lambda f: (m.decode(f), f))(m.encode(x)), mean_act=mu,
                            baselines=(base is None))
        if base is None:
            base = {k: r[k] for k in ("ce_clean", "ce_zero", "ce_mean")}
        r.update(base)
        agg = lambda st: S.doc_bootstrap(r, grp, st)
        o = {"fve": agg(lambda d: 1 - d["sq_err"].sum() / d["sq_dev"].sum()),
             "l0": agg(lambda d: d["l0"].sum() / d["n"].sum()),
             "ce_clean": agg(lambda d: d["ce_clean"].sum() / d["n"].sum()),
             "delta_ce": agg(lambda d: (d["ce_sae"].sum() - d["ce_clean"].sum()) / d["n"].sum()),
             "sec": round(time.time() - t1, 1)}
        # paired difference to ours, same windows
        out[dname][sname] = o
        out[dname][sname + "_raw"] = {k: r[k].tolist() for k in ("sq_err", "sq_dev", "ce_sae", "ce_clean", "n")} if False else None
        print(dname, sname, {k: (tuple(round(x, 4) for x in v) if isinstance(v, tuple) else v) for k, v in o.items()}, flush=True)
        np.savez_compressed(f"/home/amit/.cache/vidyalai-scratch/p12/runs/sae_{dname}_{sname}.npz", grp=grp, **r)
        torch.cuda.empty_cache()
out["seconds"] = round(time.time() - t0, 1)
out["gpu_peak_mib"] = torch.cuda.max_memory_allocated() // 2**20
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/p12/runs/sae_gpt2.json", "w"), indent=1)
print("done", out["seconds"], "s", out["gpu_peak_mib"], "MiB")
