"""Stage 3: lighter ablation set (GPT-2 extras; Pythia and Qwen repeats).
Usage: python s3_lite.py <model_key> <section: rep|nat>"""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import heads as H

key, SEC = sys.argv[1], sys.argv[2]
torch.backends.cuda.matmul.allow_tf32 = True
CFG = {"gpt2": ("gpt2", torch.float32, True),
       "pythia": ("pythia-160m", torch.float32, True),
       "qwen": ("Qwen/Qwen2.5-0.5B", torch.bfloat16, False)}
name, dtype, proc = CFG[key]
t0 = time.time()
model = H.load(name, dtype=dtype, processing=proc)
L, NH = model.cfg.n_layers, model.cfg.n_heads
s1 = json.load(open(f"/home/amit/.cache/vidyalai-scratch/p04/runs/s1_{key}.json"))
HALF = 50
ALLOWED = None
if key == "pythia":  # uniform ids include tokens Pythia never learned (losses up to 826 nats)
    _w = np.load("/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_pythia.npy")
    _ids, _c = np.unique(_w, return_counts=True)
    ALLOWED = _ids[_c >= 5]
    print("pythia allowed tokens", len(ALLOWED), flush=True)
ranked = [tuple(x) for x in s1["top_induction_sel"]]
ind_set = set(ranked[:12])
pool = [(l, h) for l in range(L) for h in range(NH) if (l, h) not in ind_set]
rng = np.random.default_rng(1)
out = {"model": name, "ranked": ranked}
BATCH = 16 if key == "qwen" else 32

if SEC == "rep":
    ev_tok = H.repeated_tokens(model, 500, HALF, seed=2, allowed=ALLOWED)
    means = H.mean_z(model, H.repeated_tokens(model, 200, HALF, seed=3, allowed=ALLOWED), batch=BATCH)
    def sh(hooks):
        return H.token_loss(model, ev_tok, fwd_hooks=hooks, batch=BATCH)[:, HALF + 1:].mean(1)
    ci = lambda x: [float(v) for v in H.ci95(x)]
    out["clean"] = ci(sh([]))
    first = H.token_loss(model, ev_tok, batch=BATCH)[:, 1:HALF].mean(1)
    out["first_half"] = ci(first)
    ks = [1, 2, 3, 4, 5, 8, 12]
    if key == "gpt2":
        ks = [4]
    out["top"] = {k: ci(sh(H.ablate_heads_hooks(ranked[:k], means))) for k in ks}
    print("top", out["top"], f"{time.time()-t0:.0f}s", flush=True)
    if key == "gpt2":
        top5 = ranked[:5]
        out["single"] = {f"{l}.{h}": ci(sh(H.ablate_heads_hooks([(l, h)], means))) for l, h in top5}
        out["keep_one"] = {f"{l}.{h}": ci(sh(H.ablate_heads_hooks([x for x in top5 if x != (l, h)], means))) for l, h in top5}
        print("single", out["single"], "keep_one", out["keep_one"], flush=True)
    else:
        out["rand"] = {}
        for k in [5, 12]:
            d = []
            for _ in range(10):
                hs = [pool[i] for i in rng.choice(len(pool), k, replace=False)]
                d.append(float(sh(H.ablate_heads_hooks(hs, means)).mean()))
            d = np.array(d)
            out["rand"][k] = {"mean": float(d.mean()), "p50": float(np.median(d)), "min": float(d.min()),
                              "max": float(d.max()), "draws": d.round(4).tolist()}
            print("rand", k, out["rand"][k], f"{time.time()-t0:.0f}s", flush=True)

if SEC == "nat":
    wiki = np.load(f"/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_{key}.npy")
    art = np.load(f"/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_{key}_article.npy")
    n = int(__import__("os").environ.get("NWIN", len(wiki)))
    idx = np.linspace(0, len(wiki) - 1, n).astype(int)
    bos = model.tokenizer.bos_token_id if model.tokenizer.bos_token_id is not None else model.tokenizer.eos_token_id
    wt = torch.tensor(wiki[idx], dtype=torch.long)
    wt = torch.cat([torch.full((wt.shape[0], 1), bos), wt], 1).to(model.cfg.device)
    nb = 4 if key == "qwen" else 8
    means = H.mean_z(model, wt[::8][:60], batch=nb)
    nat = {}
    def run(tag, hooks):
        l = H.token_loss(model, wt, fwd_hooks=hooks, batch=nb)
        nat[tag] = {"L50": l[:, 49].tolist(), "L500": l[:, 499].tolist(), "curve": l.mean(0).round(4).tolist(),
                    "mean_all": float(l.mean())}
        print("nat", tag, round(float((l[:, 499] - l[:, 49]).mean()), 4), round(float(l.mean()), 4), f"{time.time()-t0:.0f}s", flush=True)
    run("clean", [])
    run("top12", H.ablate_heads_hooks(ranked[:12], means))
    for j in range(int(__import__("os").environ.get("NRAND", "3"))):
        hs = [pool[i] for i in rng.choice(len(pool), 12, replace=False)]
        run(f"rand12_{j}", H.ablate_heads_hooks(hs, means))
    out["nat"] = nat
    out["nat_article"] = art[idx].tolist()
out["seconds"] = time.time() - t0
out["max_mem_mib"] = torch.cuda.max_memory_allocated() // 2**20
json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/p04/runs/s3_{key}_{SEC}.json", "w"))
print("done", f"{out['seconds']:.0f}s", out["max_mem_mib"], "MiB")
