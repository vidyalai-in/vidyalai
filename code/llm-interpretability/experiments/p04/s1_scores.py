"""Stage 1: head scores on repeated random tokens + natural text, copying and K-composition.
Usage: python s1_scores.py <model_key>   (gpt2 | pythia | qwen)"""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import heads as H

key = sys.argv[1]
CFG = {"gpt2": ("gpt2", torch.float32, True),
       "pythia": ("pythia-160m", torch.float32, True),
       "qwen": ("Qwen/Qwen2.5-0.5B", torch.bfloat16, False)}
name, dtype, proc = CFG[key]
t0 = time.time()
model = H.load(name, dtype=dtype, processing=proc)
print("loaded", name, f"{time.time()-t0:.1f}s", model.cfg.n_layers, model.cfg.n_heads, flush=True)
HALF = 50
ALLOWED = None
if key == "pythia":  # uniform ids include tokens Pythia never learned (losses up to 826 nats)
    _w = np.load("/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_pythia.npy")
    _ids, _c = np.unique(_w, return_counts=True)
    ALLOWED = _ids[_c >= 5]
    print("pythia allowed tokens", len(ALLOWED), flush=True)
out = {"model": name, "dtype": str(dtype), "processing": proc, "half": HALF,
       "n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads}

# selection set (seed 1) and evaluation set (seed 2) of repeated random sequences
sel = H.head_scores(model, H.repeated_tokens(model, 200, HALF, seed=1, allowed=ALLOWED), HALF)
ev = H.head_scores(model, H.repeated_tokens(model, 500, HALF, seed=2, allowed=ALLOWED), HALF)
for k in ["prev", "dup", "induction"]:
    m, c = H.ci95(ev[k])
    out[f"rep_{k}_mean"] = m.round(4).tolist(); out[f"rep_{k}_ci"] = c.round(4).tolist()
out["rep_n_seq"] = 500
out["top_induction_sel"] = H.top_heads(sel["induction"], 20)
out["top_induction_sel_scores"] = [float(sel["induction"].mean(0)[l, h]) for l, h in out["top_induction_sel"]]
loss = ev["loss"]  # [n, 100]; predicting token t+1 from prefix ending at t
first = loss[:, 1:HALF].mean(1)            # predicting tokens 2..50 of first copy
second = loss[:, HALF + 1:].mean(1)        # predicting tokens 2..50 of second copy (skip the first one)
out["rep_loss_first"] = H.ci95(first)
out["rep_loss_second"] = H.ci95(second)
out["rep_loss_by_pos"] = loss.mean(0).round(4).tolist()
out["rep_loss_first"] = [float(x) for x in out["rep_loss_first"]]
out["rep_loss_second"] = [float(x) for x in out["rep_loss_second"]]
print("loss first/second", out["rep_loss_first"], out["rep_loss_second"], flush=True)

# previous-token score on natural text: 200 windows of 256 tokens with BOS prepended
wiki = np.load(f"/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_{key}.npy")
bos = model.tokenizer.bos_token_id if model.tokenizer.bos_token_id is not None else model.tokenizer.eos_token_id
tok = torch.tensor(wiki[:: max(1, len(wiki) // 200)][:200, :255], dtype=torch.long)
tok = torch.cat([torch.full((tok.shape[0], 1), bos), tok], 1).to(model.cfg.device)
nat = H.head_scores(model, tok, None, batch=16)
m, c = H.ci95(nat["prev"])
out["nat_prev_mean"] = m.round(4).tolist(); out["nat_prev_ci"] = c.round(4).tolist(); out["nat_n_seq"] = tok.shape[0]
out["top_prev_nat"] = H.top_heads(nat["prev"], 10)

if proc:  # weight-based analyses need the folded weights
    L, Hh = model.cfg.n_layers, model.cfg.n_heads
    out["copying"] = [[H.copying_score(model, l, h) for h in range(Hh)] for l in range(L)]
    kc, rmean, rstd = H.k_composition(model)
    out["kcomp_random_mean"], out["kcomp_random_std"] = rmean, rstd
    np.save(f"/home/amit/.cache/vidyalai-scratch/p04/runs/kcomp_{key}.npy", kc)
out["seconds"] = time.time() - t0
out["max_mem_mib"] = torch.cuda.max_memory_allocated() // 2**20
json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/p04/runs/s1_{key}.json", "w"))
print("top induction", out["top_induction_sel"][:10], [round(x, 3) for x in out["top_induction_sel_scores"][:10]])
print("top prev (natural)", out["top_prev_nat"][:6])
print("done", f"{out['seconds']:.1f}s", out["max_mem_mib"], "MiB")
