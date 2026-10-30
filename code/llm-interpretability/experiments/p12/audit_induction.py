"""Audit of Part 4's headline: GPT-2 small's top-5 induction heads, re-measured under new seeds,
frequent tokens, longer sequences and repeated natural text.  -> runs/induction_gpt2.json"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import heads as Hd, replicate as R

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
t0 = time.time()
model = Hd.load("gpt2", dtype=torch.float32, processing=True)
L, NH = model.cfg.n_layers, model.cfg.n_heads
s1 = json.load(open("/home/amit/.cache/vidyalai-scratch/p04/runs/s1_gpt2.json"))   # Part 4's ranking (read only)
ORIG = [tuple(x) for x in s1["top_induction_sel"]]
TOP5, TOP12 = ORIG[:5], set(ORIG[:12])
pool = [(l, h) for l in range(L) for h in range(NH) if (l, h) not in TOP12]
wiki = np.load("/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_gpt2.npy")         # WikiText-103 test+val, GPT-2 tokens
bos = model.tokenizer.bos_token_id
freq = R.frequent_ids(wiki, top=2000)

def rand_rep(n, half, seed, allowed=None):
    return Hd.repeated_tokens(model, n, half, seed=seed, allowed=allowed)
def nat_rep(n, half, seed):
    return R.repeat_natural(wiki, n, half, bos, seed)

CONDS = {   # name: (builder, half, n_eval, seeds (select, eval, means))
    "original": (lambda n, s: rand_rep(n, 50, s), 50, 500, (1, 2, 3)),
    "new_seeds": (lambda n, s: rand_rep(n, 50, s), 50, 500, (101, 102, 103)),
    "frequent_tokens": (lambda n, s: rand_rep(n, 50, s, freq), 50, 500, (201, 202, 203)),
    "long_200": (lambda n, s: rand_rep(n, 200, s), 200, 200, (301, 302, 303)),
    "natural_text": (lambda n, s: nat_rep(n, 50, s), 50, 500, (401, 402, 403)),
}
rng = np.random.default_rng(0)
rand_sets = [[pool[i] for i in rng.choice(len(pool), 5, replace=False)] for _ in range(20)]
out = {"orig_top5": TOP5, "n_random_sets": len(rand_sets)}
for name, (build, half, n_ev, (ss, se, sm)) in CONDS.items():
    t1 = time.time()
    b = 8 if half == 200 else 32
    sel = Hd.head_scores(model, build(200, ss), half, batch=b)
    ev_tok = build(n_ev, se)
    ev = Hd.head_scores(model, ev_tok, half, batch=b)
    means = Hd.mean_z(model, build(200, sm), batch=b)
    new5 = Hd.top_heads(sel["induction"], 5)
    ind = ev["induction"].mean(0)
    second = lambda hooks: Hd.token_loss(model, ev_tok, fwd_hooks=hooks, batch=b)[:, half + 1:].mean(1)
    clean = ev["loss"][:, half + 1:].mean(1)
    first = ev["loss"][:, 1:half].mean(1)
    abl_orig = second(Hd.ablate_heads_hooks(TOP5, means))
    abl_new = abl_orig if set(new5) == set(TOP5) else second(Hd.ablate_heads_hooks(new5, means))
    rnd = np.array([second(Hd.ablate_heads_hooks(s, means)).mean() for s in rand_sets])
    ci = lambda x: [float(v) for v in R.boot_mean(x, n_boot=4000)]
    gap = first - clean
    o = {"half": half, "n_eval": n_ev, "seeds": [ss, se, sm],
         "loss_first": ci(first), "loss_second": ci(clean), "loss_second_top5_orig": ci(abl_orig),
         "loss_second_top5_reranked": ci(abl_new), "reranked_top5": new5,
         "overlap_with_orig_top5": len(set(new5) & set(TOP5)),
         "share_of_repeat_benefit_removed_orig5": ci((abl_orig - clean) / np.maximum(gap, 1e-6)) if False else
             [float((abl_orig.mean() - clean.mean()) / (first.mean() - clean.mean()))],
         "random5_second_loss": {"median": float(np.median(rnd)), "p5": float(np.percentile(rnd, 5)),
                                 "p95": float(np.percentile(rnd, 95)), "max": float(rnd.max())},
         "orig5_induction_scores": {f"{l}.{h}": float(ind[l, h]) for l, h in TOP5},
         "top10_induction_eval": [[f"{l}.{h}", float(ind[l, h])] for l, h in Hd.top_heads(ind, 10)],
         "seconds": round(time.time() - t1, 1)}
    out[name] = o
    print(name, "first", round(o["loss_first"][0], 2), "second", round(o["loss_second"][0], 3),
          "abl orig5", [round(x, 2) for x in o["loss_second_top5_orig"]], "abl new5", round(o["loss_second_top5_reranked"][0], 2),
          "new5", new5, "overlap", o["overlap_with_orig_top5"], "rand med", round(o["random5_second_loss"]["median"], 3),
          "p95", round(o["random5_second_loss"]["p95"], 2), "orig5 scores", {k: round(v, 2) for k, v in o["orig5_induction_scores"].items()},
          f"{o['seconds']}s", flush=True)
out["seconds"] = round(time.time() - t0, 1)
out["gpu_peak_mib"] = torch.cuda.max_memory_allocated() // 2**20
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/p12/runs/induction_gpt2.json", "w"), indent=1)
print("done", out["seconds"], "s", out["gpu_peak_mib"], "MiB")
