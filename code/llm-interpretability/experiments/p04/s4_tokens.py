"""Stage 4: per-token WikiText losses for GPT-2, clean vs top-12 induction heads vs random 12."""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import heads as H
torch.backends.cuda.matmul.allow_tf32 = True
t0 = time.time()
model = H.load("gpt2")
s1 = json.load(open("/home/amit/.cache/vidyalai-scratch/p04/runs/s1_gpt2.json"))
ranked = [tuple(x) for x in s1["top_induction_sel"]]
pool = [(l, h) for l in range(12) for h in range(12) if (l, h) not in set(ranked[:12])]
rng = np.random.default_rng(7)
wiki = np.load("/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_gpt2.npy")
wt = torch.tensor(wiki, dtype=torch.long)
wt = torch.cat([torch.full((wt.shape[0], 1), model.tokenizer.bos_token_id), wt], 1).cuda()
means = H.mean_z(model, wt[np.arange(0, len(wt), 8)[:100]], batch=8)
conds = {"clean": [], "top12": ranked[:12]}
for j in range(2):
    conds[f"rand12_{j}"] = [pool[i] for i in rng.choice(len(pool), 12, replace=False)]
res = {}
for tag, hs in conds.items():
    hooks = H.ablate_heads_hooks(hs, means) if hs else []
    res[tag] = H.token_loss(model, wt, fwd_hooks=hooks, batch=8).astype(np.float16)
    print(tag, float(res[tag].astype(np.float32).mean()), f"{time.time()-t0:.0f}s", flush=True)
np.savez("/home/amit/.cache/vidyalai-scratch/p04/runs/s4_gpt2_tokens.npz", **res)
json.dump({k: [list(x) for x in v] for k, v in conds.items()}, open("/home/amit/.cache/vidyalai-scratch/p04/runs/s4_conds.json", "w"))
print("done", f"{time.time()-t0:.0f}s", torch.cuda.max_memory_allocated() // 2**20, "MiB")
