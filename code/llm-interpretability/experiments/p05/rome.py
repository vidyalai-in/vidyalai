"""Part 5, experiment 3: does localization inform editing? (Hase et al. 2023, small scale)
A ROME-style rank-one edit (Meng et al. 2022) of MLP W_out at each layer of GPT-2 small, for each
landmark fact GPT-2 knows, to a new city; compared with that fact's causal-tracing effect per layer.
Needs runs/trace_gpt2.npz. Writes runs/rome.npz and post/rome.json."""
import json, sys, time, random
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/p01/scripts")
from lab import hooks as H
from facts import FACTS, TEMPLATE

ROOT = "/home/amit/.cache/vidyalai-scratch/p05"
LAYERS = [0, 1, 2, 3, 5, 7, 9, 11]
RIDGE = 1e-2                                        # ridge on C, as a share of its mean diagonal
STEPS, LR, CLAMP, WD = 20, 0.5, 4.0, 0.5            # ROME's GPT-2 XL settings for v* (lr, steps, clamp, decay)
COV_TOKENS = 100_000
PARA = ["{} can be found in the city of", "If you visit {}, you will be in the city of"]
t0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = False
model = H.load("gpt2", dtype=torch.float32, processing=True)
for p in model.parameters():
    p.requires_grad_(False)
dev = model.cfg.device
tr = np.load(f"{ROOT}/runs/trace_gpt2.npz")
tok1 = lambda c: model.to_tokens(" " + c, prepend_bos=False)[0]
facts = [(l, c) for l, c in FACTS if len(tok1(c)) == 1 and c not in {"Agra", "Granada", "Krakow", "Pisa", "Mecca"}]
cities = sorted({c for _, c in facts})
known = tr["gn_ids"].tolist()
assert len(facts) == len(tr["known"]) and set(known) == set(np.nonzero(tr["known"])[0].tolist())
rng = random.Random(0)
target = {i: rng.choice([c for c in cities if c != facts[i][1]]) for i in known}

# ---- uncentered second moment C = E[k k^T] of the MLP keys (post-GELU activations), per layer
torch.set_grad_enabled(False)
texts = [json.loads(l)["text"] for l in open("/home/amit/.cache/vidyalai-scratch/interp/data/p02/fineweb-edu-heldout600.jsonl")]
C = torch.zeros(len(LAYERS), 3072, 3072, device=dev, dtype=torch.float64); n_tok = 0
for t in texts:
    toks = model.to_tokens(t)[:, :257]
    if toks.shape[1] < 257:
        continue
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("mlp.hook_post"))
    for a, l in enumerate(LAYERS):
        k = cache[f"blocks.{l}.mlp.hook_post"][0, 1:].double()
        C[a] += k.T @ k
    n_tok += toks.shape[1] - 1
    if n_tok >= COV_TOKENS:
        break
C /= n_tok
eye = torch.eye(3072, device=dev, dtype=torch.float64)
Cinv = torch.stack([torch.linalg.inv(c + RIDGE * c.diagonal().mean() * eye) for c in C]).float()
LI = {l: a for a, l in enumerate(LAYERS)}
print(f"covariance from {n_tok} tokens, {time.time()-t0:.0f}s", flush=True)

# ---- context prefixes, sampled from GPT-2 as ROME does (5 of 10 tokens)
torch.manual_seed(0)
PREF = [""]
for s in ["The", "Therefore", "Because", "I", "You"]:
    out = model.generate(s, max_new_tokens=9, do_sample=True, temperature=1.0, verbose=False, use_past_kv_cache=True)
    PREF.append(out.replace("\n", " ").strip() + ". ")

def batch(prompts):
    """Right-padded tokens and the index of each prompt's last real token."""
    toks = [model.to_tokens(p)[0] for p in prompts]
    T = max(len(t) for t in toks)
    out = torch.full((len(toks), T), model.tokenizer.eos_token_id, device=dev)
    for i, t in enumerate(toks):
        out[i, :len(t)] = t
    return out, torch.tensor([len(t) - 1 for t in toks], device=dev)

def subj_pos(prompts, landmark):
    """Index of the subject's last token in each prompt (prefix + landmark ...)."""
    pos = []
    for p in prompts:
        pre = p[:p.index(landmark) + len(landmark)]
        pos.append(len(model.to_tokens(pre)[0]) - 1)
    return torch.tensor(pos, device=dev)

def probs(prompts, ids, hooks=()):
    toks, last = batch(prompts)
    lg = model.run_with_hooks(toks, fwd_hooks=list(hooks))
    return lg[torch.arange(len(prompts)), last].float().softmax(-1)[:, ids]

def edit_and_eval(i, l):
    landmark, true_c = facts[i]
    new_c = target[i]
    ids = torch.tensor([tok1(true_c).item(), tok1(new_c).item()], device=dev)
    edit_prompts = [p + TEMPLATE.format(landmark) for p in PREF]
    sp = subj_pos(edit_prompts, landmark)
    toks, last = batch(edit_prompts)
    hn = f"blocks.{l}.mlp.hook_post"
    # k*: the key at the subject's last token, averaged over the prefixes
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n in (hn, f"blocks.{l}.hook_mlp_out"))
    kstar = cache[hn][torch.arange(len(sp)), sp].mean(0)
    out0 = cache[f"blocks.{l}.hook_mlp_out"][0, sp[0]]                 # MLP output on the plain prompt
    # v*: optimise a vector added to the MLP output at the subject's last token
    delta = torch.zeros(model.cfg.d_model, device=dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR)
    def add(act, hook):
        return act.index_put((torch.arange(len(sp), device=dev), sp), act[torch.arange(len(sp)), sp] + delta, accumulate=False)
    with torch.enable_grad():
        for step in range(STEPS):
            lg = model.run_with_hooks(toks, fwd_hooks=[(f"blocks.{l}.hook_mlp_out", add)])
            lp = lg[torch.arange(len(sp)), last].log_softmax(-1)[:, ids[1]]
            nll = -lp.mean()
            loss = nll + WD * delta.norm() / out0.norm() ** 2
            if nll.item() < 0.05:
                break
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                mx = CLAMP * out0.norm()
                if delta.norm() > mx:
                    delta.mul_(mx / delta.norm())
    v = (out0 + delta).detach()
    # rank-one update of W_out (row-vector convention: out = k @ W_out + b_out)
    W = model.blocks[l].mlp.W_out
    u = Cinv[LI[l]] @ kstar
    cur = kstar @ W + model.blocks[l].mlp.b_out
    upd = torch.outer(u, v - cur) / (u @ kstar)
    pre = {"edit": probs([TEMPLATE.format(landmark)], ids), "para": probs([p.format(landmark) for p in PARA], ids)}
    nb = [f for j, f in enumerate(facts) if f[1] == true_c and j != i][:4]
    if nb:
        pre["nb"] = probs([TEMPLATE.format(x) for x, _ in nb], ids)
    W += upd
    post = {"edit": probs([TEMPLATE.format(landmark)], ids), "para": probs([p.format(landmark) for p in PARA], ids)}
    if nb:
        post["nb"] = probs([TEMPLATE.format(x) for x, _ in nb], ids)
    W.copy_(W0[l])
    rs = lambda a, b: ((b[:, 1] - a[:, 1]) / (1 - a[:, 1])).mean().item()
    r = {"rewrite": rs(pre["edit"], post["edit"]), "para": rs(pre["para"], post["para"]),
         "nll_steps": step, "delta_ratio": (delta.norm() / out0.norm()).item(),
         "flip": (post["edit"][0, 1] > post["edit"][0, 0]).item()}
    if nb:
        d = (post["nb"][:, 1] - pre["nb"][:, 1]).abs()
        r["neighborhood"] = (1 - d / (0.5 + (pre["nb"][:, 1] - 0.5).abs())).mean().item()
    else:
        r["neighborhood"] = float("nan")
    return r

W0 = {l: model.blocks[l].mlp.W_out.clone() for l in LAYERS}
R = {k: np.full((len(known), len(LAYERS)), np.nan) for k in ("rewrite", "para", "neighborhood", "flip", "delta_ratio")}
t1 = time.time()
for a, i in enumerate(known):
    for b, l in enumerate(LAYERS):
        r = edit_and_eval(i, l)
        for k in R:
            R[k][a, b] = r[k]
    if a % 10 == 0:
        print(f"fact {a+1}/{len(known)} {time.time()-t1:.0f}s rewrite {np.nanmean(R['rewrite'][:a+1], 0).round(2)}", flush=True)
for l in LAYERS:
    assert torch.equal(model.blocks[l].mlp.W_out, W0[l]), "weights not restored"
np.savez_compressed(f"{ROOT}/runs/rome.npz", known=np.array(known), layers=np.array(LAYERS), **R)
json.dump({"prefixes": PREF, "targets": {facts[i][0]: target[i] for i in known}, "cov_tokens": n_tok,
           "wall_s": time.time() - t0, "gpu_peak_mib": torch.cuda.max_memory_allocated() // 2**20,
           "hparams": dict(steps=STEPS, lr=LR, clamp=CLAMP, wd=WD), "paraphrases": PARA},
          open(f"{ROOT}/post/rome_run.json", "w"), indent=1)
print("done", time.time() - t0, flush=True)
