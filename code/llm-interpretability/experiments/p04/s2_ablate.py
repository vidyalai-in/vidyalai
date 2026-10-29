"""Stage 2: ablations and in-context learning score.
Usage: python s2_ablate.py <model_key> <section: rep|nat|dla|kpath>"""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import heads as H

key, SEC = sys.argv[1], sys.argv[2]
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 tensor cores for fp32 matmuls
CFG = {"gpt2": ("gpt2", torch.float32, True),
       "pythia": ("pythia-160m", torch.float32, True),
       "qwen": ("Qwen/Qwen2.5-0.5B", torch.bfloat16, False)}
name, dtype, proc = CFG[key]
t0 = time.time()
model = H.load(name, dtype=dtype, processing=proc)
L, NH = model.cfg.n_layers, model.cfg.n_heads
s1 = json.load(open(f"/home/amit/.cache/vidyalai-scratch/p04/runs/s1_{key}.json"))
HALF = 50
NATDRAW = int(__import__("os").environ.get("NATDRAW", "5"))
ranked = [tuple(x) for x in s1["top_induction_sel"]]        # ranked on the seed-1 selection set
ind_set = set(ranked[:12])
pool = [(l, h) for l in range(L) for h in range(NH) if (l, h) not in ind_set]
rng = np.random.default_rng(0)
out = {"model": name, "ranked": ranked}

# ---------- repeated random tokens ----------
ev_tok = H.repeated_tokens(model, 500, HALF, seed=2)
ref_tok = H.repeated_tokens(model, 200, HALF, seed=3)
means_rep = H.mean_z(model, ref_tok)
def second_half(hooks):
    l = H.token_loss(model, ev_tok, fwd_hooks=hooks)
    return l[:, HALF + 1:].mean(1)                     # tokens 2..50 of the second copy
if SEC == "rep":
  clean = second_half([])
  out["rep_clean"] = [float(x) for x in H.ci95(clean)]
  KS = [1, 2, 3, 5, 8, 12]
  NDRAW = 50
  out["rep_top"], out["rep_rand"] = {}, {}
  for k in KS:
      t = second_half(H.ablate_heads_hooks(ranked[:k], means_rep))
      out["rep_top"][k] = [float(x) for x in H.ci95(t)]
      draws = []
      for _ in range(NDRAW):
          hs = [pool[i] for i in rng.choice(len(pool), k, replace=False)]
          draws.append(float(second_half(H.ablate_heads_hooks(hs, means_rep)).mean()))
      d = np.array(draws)
      out["rep_rand"][k] = {"mean": float(d.mean()), "p5": float(np.percentile(d, 5)),
                            "p50": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}
      print("rep k", k, out["rep_top"][k], out["rep_rand"][k], f"{time.time()-t0:.0f}s", flush=True)
  # zero ablation of the top 5, as a check on the ablation choice
  out["rep_top5_zero"] = [float(x) for x in H.ci95(second_half(H.ablate_heads_hooks(ranked[:5], None)))]

# ---------- natural text: ICL score ----------
if SEC == "nat":
  wiki = np.load(f"/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_{key}.npy")
  art = np.load(f"/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_{key}_article.npy")
  bos = model.tokenizer.bos_token_id if model.tokenizer.bos_token_id is not None else model.tokenizer.eos_token_id
  wt = torch.tensor(wiki, dtype=torch.long)
  wt = torch.cat([torch.full((wt.shape[0], 1), bos), wt], 1).to(model.cfg.device)  # [n, 513]
  ref_idx = np.arange(0, len(wt), 8)[:100]                                           # mean reference windows
  means_nat = H.mean_z(model, wt[ref_idx], batch=8)
  nat = {}
  def run_nat(tag, hooks):
      l = H.token_loss(model, wt, fwd_hooks=hooks, batch=8)          # l[:, t-1] = loss of text token t
      nat[tag] = {"L50": l[:, 49].tolist(), "L500": l[:, 499].tolist(), "curve": l.mean(0).round(4).tolist(),
                  "mean_all": float(l.mean())}
      print("nat", tag, round(float((l[:, 499] - l[:, 49]).mean()), 4), round(float(l.mean()), 4), f"{time.time()-t0:.0f}s", flush=True)
  run_nat("clean", [])
  for k in [5, 12]:
      run_nat(f"top{k}", H.ablate_heads_hooks(ranked[:k], means_nat))
      for j in range(NATDRAW):
          hs = [pool[i] for i in rng.choice(len(pool), k, replace=False)]
          run_nat(f"rand{k}_{j}", H.ablate_heads_hooks(hs, means_nat))
  out["nat"] = nat
  out["nat_article"] = art.tolist()

# ---------- direct logit attribution of each head to the correct next token ----------
if SEC == "dla":
    tok = ev_tok[:200]
    dla2 = torch.zeros(L, NH); dla1 = torch.zeros(L, NH)
    with torch.no_grad():
        for s in range(0, tok.shape[0], 25):
            tb = tok[s:s + 25]
            _, cache = model.run_with_cache(tb, names_filter=lambda n: n.endswith("hook_z") or n == "ln_final.hook_scale")
            scale = cache["ln_final.hook_scale"]                          # [b, pos, 1]
            wu = model.W_U[:, tb[:, 1:]].permute(1, 2, 0)                  # [b, pos-1, d]
            for l in range(L):
                z = cache[f"blocks.{l}.attn.hook_z"]                        # [b, pos, h, dh]
                res = torch.einsum("bphd,hdm->bphm", z, model.W_O[l]) / scale[..., None]
                c = torch.einsum("bphm,bpm->bph", res[:, :-1], wu)           # direct effect on correct logit
                dla2[l] += c[:, HALF + 1:].mean(1).sum(0).cpu()
                dla1[l] += c[:, 1:HALF].mean(1).sum(0).cpu()
    out["dla_second"] = (dla2 / tok.shape[0]).numpy().round(4).tolist()
    out["dla_first"] = (dla1 / tok.shape[0]).numpy().round(4).tolist()

# ---------- K-composition, causally: remove a sender from the induction heads' keys ----------
if SEC == "kpath":
    model.set_use_split_qkv_input(True)
    recv = ranked[:5]
    tok = ev_tok[:200]
    def kpath(senders):
        hooks = H.k_path_ablation_hooks(model, senders, recv, means_rep)
        r = H.head_scores(model, tok, HALF, batch=20, fwd_hooks=hooks)
        ind = np.stack([r["induction"][:, l, h] for l, h in recv], 1).mean(1)
        return ind, r["loss"][:, HALF + 1:].mean(1)
    kp = {}
    for tag, snd in [("none", []), ("4.11", [(4, 11)]), ("4.11+2.2", [(4, 11), (2, 2)])]:
        ind, ls = kpath(snd)
        kp[tag] = {"ind": [float(x) for x in H.ci95(ind)], "loss": [float(x) for x in H.ci95(ls)]}
        print("kpath", tag, kp[tag], flush=True)
    others = [(l, h) for l in range(4) for h in range(NH)] + [(4, h) for h in range(NH) if h != 11]
    single = []
    for snd in others:
        ind, ls = kpath([snd])
        single.append({"head": snd, "ind": float(ind.mean()), "loss": float(ls.mean())})
    kp["single_others"] = single
    out["kpath"] = kp
out["seconds"] = time.time() - t0
out["max_mem_mib"] = torch.cuda.max_memory_allocated() // 2**20
json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/p04/runs/s2_{key}_{SEC}.json", "w"))
print("done", f"{out['seconds']:.0f}s", out["max_mem_mib"], "MiB")
