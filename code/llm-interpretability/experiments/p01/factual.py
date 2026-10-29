"""Interpretability Part 1 experiment: three method families on one factual-recall behavior.
Usage: python factual.py gpt2|qwen [--heads]
Writes runs/<model>.npz (per-prompt arrays) and post/<model>.json (summary)."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/p01/scripts")
from lab import hooks as H
from facts import FACTS, TEMPLATE

which = sys.argv[1]
DO_HEADS = "--heads" in sys.argv
ROOT = "/home/amit/.cache/vidyalai-scratch/p01"
cfg = {"gpt2": ("gpt2", True), "qwen": ("Qwen/Qwen2.5-0.5B", False)}[which]
torch.manual_seed(0)
torch.set_grad_enabled(False)
t0 = time.time()
model = H.load(cfg[0], dtype=torch.float32, processing=cfg[1])
print(model.setup_info, f"load {time.time()-t0:.1f}s", flush=True)

tok1 = lambda c: model.to_tokens(" " + c, prepend_bos=False)[0]
facts = [(l, c) for l, c in FACTS if len(tok1(c)) == 1 and c not in {"Mecca"}]
for bad in ("Agra", "Granada", "Krakow", "Pisa", "Mecca"):
    facts = [f for f in facts if f[1] != bad]          # same set for both models
prompts = [TEMPLATE.format(l) for l, _ in facts]
cities = sorted({c for _, c in facts})
cid = torch.tensor([tok1(c).item() for c in cities], device=model.cfg.device)
ans = torch.tensor([cities.index(c) for _, c in facts], device=model.cfg.device)
first = model.to_tokens(prompts[0])[0][:2]
START = 1 if (model.cfg.default_prepend_bos or "--start1" in sys.argv) else 0
print(len(prompts), "prompts,", len(cities), "cities; first tokens", model.to_str_tokens(first), "start", START)

def metrics(logits):
    """logit diff (answer minus mean of the other candidate cities), top-1 over the full vocab,
    top-1 among candidates."""
    cl = logits[:, cid]                                   # [n, n_cities]
    a = cl.gather(1, ans[:, None])[:, 0]
    others = (cl.sum(1) - a) / (len(cities) - 1)
    return (a - others), (logits.argmax(1) == cid[ans]), (cl.argmax(1) == ans)

logits, acts = H.run_cached(model, prompts)
ld, top1, topc = metrics(logits)
print(f"clean: top1 vocab {top1.float().mean():.3f}, top1 among {len(cities)} cities {topc.float().mean():.3f}, "
      f"logit diff {ld.mean():.2f}", flush=True)

# The direction each component would have to push: W_U[answer] - mean W_U[other cities]
WU = model.W_U.float()[:, cid].T                          # [n_cities, d]
dirs = WU[ans] - (WU.sum(0, keepdim=True) - WU[ans]) / (len(cities) - 1)
scale = acts["ln_final.hook_scale"]
labels, comps = H.components(model, acts)
dla = H.dla(model, comps, scale, dirs)                    # [n_comp, n]
# biases and the check that the pieces add up
bU = model.b_U.float()[cid]
bias_term = bU[ans] - (bU.sum() - bU[ans]) / (len(cities) - 1)
lnb = getattr(model.ln_final, "b", None)
if lnb is not None:
    bias_term = bias_term + (lnb.float() * dirs).sum(-1)
recon = dla.sum(0) + bias_term
print(f"decomposition check: max |sum of DLA + biases - logit diff| = {(recon - ld).abs().max():.2e}")
heads_out = H.head_outputs(model, acts)                   # [L, h, n, d]
dla_heads = H.dla(model, heads_out, scale, dirs)          # [L, h, n]

# Means for mean ablation. Default ("position"): one mean per template position, counted from
# the end (" of", " city", ...), and one mean for all subject (landmark) positions, as the IOI
# paper averages per template position. "--pooled": one mean over every position.
POOLED = "--pooled" in sys.argv
NT = len(model.to_tokens(TEMPLATE.format(""), prepend_bos=False)[0])
mean_names = [f"blocks.{l}.{k}" for l in range(model.cfg.n_layers) for k in ("hook_attn_out", "hook_mlp_out", "attn.hook_z")]
def pos_class(T):
    r = torch.arange(T - 1 - START, -1, -1, device=model.cfg.device)     # distance from the end
    return torch.zeros_like(r) if POOLED else r.clamp(max=NT)
sums, counts = {}, torch.zeros(NT + 1, device=model.cfg.device)
def acc_hook(act, hook):
    a = act[:, START:].float().sum(0)                                    # [T-START, ...]
    cls = pos_class(act.shape[1])
    s = torch.zeros((NT + 1,) + a.shape[1:], device=a.device).index_add_(0, cls, a)
    sums[hook.name] = sums.get(hook.name, 0) + s
for idx, toks in H.length_groups(model, prompts):
    model.run_with_hooks(toks, fwd_hooks=[(n, acc_hook) for n in mean_names])
    counts.index_add_(0, pos_class(toks.shape[1]), torch.full((toks.shape[1] - START,), float(len(idx)), device=counts.device))
means = {k: v / counts.clamp(min=1).view(-1, *[1] * (v.dim() - 1)) for k, v in sums.items()}
def mean_fn(name):
    return lambda T: means[name][pos_class(T)]

# direct part of a mean ablation: what removing the piece and adding its mean does through the
# unembedding alone, with the final-norm scale frozen
res = {"setup": model.setup_info, "n_prompts": len(prompts), "n_cities": len(cities),
       "acc_top1_vocab": float(top1.float().mean()), "acc_top1_cities": float(topc.float().mean()),
       "clean_ld": H.bootstrap_ci(ld.cpu()), "components": {}, "labels": labels}
known = topc.cpu().numpy().astype(bool)
res["n_known"] = int(known.sum())
arr = {"ld": ld.cpu().numpy(), "known": known, "dla": dla.cpu().numpy(), "dla_heads": dla_heads.cpu().numpy()}
t1 = time.time()
for li, lab in enumerate(labels):
    entry = {"dla": H.bootstrap_ci(dla[li].cpu().numpy()), "dla_known": H.bootstrap_ci(dla[li].cpu().numpy()[known])}
    if lab.startswith(("attn", "mlp")):
        l = int(lab[4:] if lab.startswith("attn") else lab[3:])
        hname = f"blocks.{l}.hook_attn_out" if lab.startswith("attn") else f"blocks.{l}.hook_mlp_out"
        for mode in ("zero", "mean"):
            lg = H.run_ablated(model, prompts, hname, H.ablate_hook(mode, mean_fn(hname), start=START))
            ld2, _, tc2 = metrics(lg)
            d = (ld2 - ld).cpu().numpy()
            arr[f"d_{mode}_{lab}"] = d
            entry[f"delta_{mode}"] = H.bootstrap_ci(d)
            entry[f"delta_{mode}_known"] = H.bootstrap_ci(d[known])
            entry[f"acc_{mode}"] = float(tc2.float().mean())
            entry[f"acc_{mode}_known"] = float(tc2.cpu().numpy()[known].mean())
        mvec = means[hname][0].expand_as(comps[li])        # mean at the last position
        direct = -(dla[li] - H.dla(model, mvec, scale, dirs)).cpu().numpy()
        arr[f"direct_mean_{lab}"] = direct
        entry["direct_mean"] = H.bootstrap_ci(direct)
        entry["indirect_mean"] = H.bootstrap_ci(arr[f"d_mean_{lab}"] - direct)
    res["components"][lab] = entry
print(f"layer ablations {time.time()-t1:.1f}s", flush=True)

if DO_HEADS:
    t1 = time.time()
    L, nh = model.cfg.n_layers, model.cfg.n_heads
    dh = np.zeros((L, nh, len(prompts)))
    for l in range(L):
        hname = f"blocks.{l}.attn.hook_z"
        for h in range(nh):
            lg = H.run_ablated(model, prompts, hname, H.ablate_hook("mean", mean_fn(hname), head=h, start=START))
            dh[l, h] = (metrics(lg)[0] - ld).cpu().numpy()
    arr["d_mean_heads"] = dh
    res["heads"] = {"dla_mean": dla_heads.cpu().numpy().mean(-1).round(4).tolist(),
                    "dmean_mean": dh.mean(-1).round(4).tolist()}
    print(f"head ablations {time.time()-t1:.1f}s", flush=True)

res["examples"] = []
for i in range(0, len(prompts), 21):
    res["examples"].append({"prompt": prompts[i], "answer": facts[i][1],
                            "top": model.to_string(logits[i].argmax()), "known": bool(known[i])})
res["wall_s"] = time.time() - t0
res["gpu_peak_mib"] = torch.cuda.max_memory_allocated() // 2**20
res["start"] = START
tag = which + ("_start0" if START == 0 else "") + ("_pooled" if POOLED else "")
res["pooled_mean"] = POOLED
np.savez_compressed(f"{ROOT}/runs/{tag}.npz", **arr)
json.dump(res, open(f"{ROOT}/post/{tag}.json", "w"), indent=1)
print("done", f"{res['wall_s']:.1f}s", res["gpu_peak_mib"], "MiB")
