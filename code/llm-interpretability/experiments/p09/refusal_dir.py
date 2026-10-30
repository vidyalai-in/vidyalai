"""Find the refusal direction (Arditi et al. 2024 recipe): difference of means between harmful and
harmless instructions at every layer and each of the last 5 (post-instruction) positions, then select
on validation sets by bypass score (ablation, harmful val), induce score (addition, harmless val), KL (ablation, harmless val).
usage: refusal_dir.py <model_path> <tag> <refusal tokens comma-separated>
Gemma: ablation is disabled (Gemma Prohibited Use Policy), so only induce scores are computed and the direction is
fixed at layer 14, the 'model' token (position -2: the position, at a similar relative depth, that Arditi et al.
selected for the first Gemma 2B IT)."""
import sys, json, time
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.steering import Steerer, diff_means, refusal_metric
torch.set_grad_enabled(False)
path, tag, rtoks = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
S = json.load(open("/home/amit/.cache/vidyalai-scratch/interp/data/p09/splits.json"))
R = "/home/amit/.cache/vidyalai-scratch/p09/runs"
t0 = time.time()
s = Steerer(path)
print("loaded", f"{time.time()-t0:.0f}s", "layers", s.n_layers, "d", s.d_model, "GPU GiB", torch.cuda.max_memory_allocated() / 2**30, flush=True)
ex = s.chat(["Hello"])[0]
ids = s.tok(ex, add_special_tokens=False).input_ids
print("template:", repr(ex), "last5:", s.tok.convert_ids_to_tokens(ids[-5:]), flush=True)
rid = s.tok.convert_tokens_to_ids(rtoks)
print("refusal ids", list(zip(rtoks, rid)))

hp, sp = s.chat(S["harmful_train"]), s.chat(S["harmless_train"])
# first-token distributions
tops = {}
for name, P in [("harmful_train", hp), ("harmless_train", sp)]:
    p = s.next_token_probs(P).mean(0)
    v, i = p.topk(10)
    tops[name] = [(s.tok.convert_ids_to_tokens(int(j)), round(float(x), 4)) for x, j in zip(v, i)]
    print(name, tops[name], flush=True)

A = s.collect(hp); B = s.collect(sp)                  # [L, 5, n, d]
r = diff_means(A, B)                                  # [L, 5, d]
norms = A.norm(dim=-1).mean(-1)                       # mean activation norm [L, 5]
torch.save({"r": r, "harmful_norm": norms, "harmless_norm": B.norm(dim=-1).mean(-1)}, f"{R}/{tag}_dirs.pt")
del A, B
print("diff-means done", f"{time.time()-t0:.0f}s", flush=True)

hv, sv = s.chat(S["harmful_val"]), s.chat(S["harmless_val"])
base_h = refusal_metric(s.next_token_probs(hv), rid)          # unedited model
p_sv = s.next_token_probs(sv)
base_s = refusal_metric(p_sv, rid)
print("baseline refusal metric: harmful val", base_h.mean().item(), "harmless val", base_s.mean().item(), flush=True)
L = s.n_layers
res = []
for l in range(L):
    for i in range(5):
        v = r[l, i]
        bypass = kl = float("nan")
        if not s.is_gemma:
            with s.edits(lambda m: m.ablate(v)):
                bypass = refusal_metric(s.next_token_probs(hv), rid).mean().item()
                p2 = s.next_token_probs(sv)
            kl = (p_sv * (torch.log(p_sv.clamp_min(1e-12)) - torch.log(p2.clamp_min(1e-12)))).sum(-1).mean().item()
        with s.edits(lambda m: m.add(l, v)):
            induce = refusal_metric(s.next_token_probs(sv), rid).mean().item()
        res.append(dict(layer=l, pos=i - 5, bypass=bypass, induce=induce, kl=kl, norm=v.norm().item()))
    print("layer", l, [(round(x["bypass"], 2), round(x["induce"], 2), round(x["kl"], 3)) for x in res[-5:]], flush=True)
if s.is_gemma:
    ok = [x for x in res if x["layer"] == 14 and x["pos"] == -2]
    best = ok[0]
else:
    ok = [x for x in res if x["induce"] > 0 and x["kl"] < 0.1 and x["layer"] < 0.8 * L]
    best = min(ok, key=lambda x: x["bypass"]) if ok else None
print("selected", best, "of", len(ok), "eligible", f"{time.time()-t0:.0f}s", flush=True)
json.dump({"model": path, "tag": tag, "refusal_tokens": list(zip(rtoks, rid)), "first_token_top10": tops,
           "baseline": {"harmful_val": base_h.mean().item(), "harmless_val": base_s.mean().item()},
           "candidates": res, "selected": best, "n_layers": L, "d_model": s.d_model,
           "seconds": time.time() - t0}, open(f"{R}/{tag}_select.json", "w"), indent=1)
