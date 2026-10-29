"""Part 5, experiment 2: causal tracing (Meng et al. 2022) and symmetric-token-replacement patching
on Part 1's landmark prompts. Usage: python trace.py gpt2|gpt2-medium|qwen
Writes runs/trace_<model>.npz (per-prompt arrays) and post/trace_<model>.json."""
import json, sys, time, random
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/p01/scripts")
from lab import hooks as H, patching as P
from facts import FACTS, TEMPLATE

which = sys.argv[1]
ROOT = "/home/amit/.cache/vidyalai-scratch/p05"
name, proc = {"gpt2": ("gpt2", True), "gpt2-medium": ("gpt2-medium", True), "qwen": ("Qwen/Qwen2.5-0.5B", False)}[which]
FULL = which == "gpt2"          # all positions for every kind of restoration (heatmaps, Hase analysis)
NS = 10 if FULL else 5          # noise samples per prompt
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
t0 = time.time()
model = H.load(name, dtype=torch.float32, processing=proc)
L, dev = model.cfg.n_layers, model.cfg.device
W = {12: 3, 24: 5}[L]       # ROME used a window of 10 of GPT-2 XL's 48 layers; about a fifth of the depth here
print(model.setup_info, f"window {W}", flush=True)

tok1 = lambda c: model.to_tokens(" " + c, prepend_bos=False)[0]
facts = [(l, c) for l, c in FACTS if len(tok1(c)) == 1 and c not in {"Agra", "Granada", "Krakow", "Pisa", "Mecca"}]
cities = sorted({c for _, c in facts})
cid = torch.tensor([tok1(c).item() for c in cities], device=dev)
bos = model.tokenizer.bos_token_id if model.cfg.default_prepend_bos else model.tokenizer.eos_token_id
# Every prompt starts with a start token: GPT-2's <|endoftext|> (TransformerLens default); for Qwen,
# which has none, its <|endoftext|> is prepended so that position 0 is never part of the noised subject.
def encode(landmark):
    body = model.to_tokens(TEMPLATE.format(landmark), prepend_bos=False)[0]
    subj = model.to_tokens(landmark, prepend_bos=False)[0]
    assert torch.equal(body[:len(subj)], subj), landmark
    return torch.cat([torch.tensor([bos], device=dev), body]), len(subj)
enc = [encode(l) for l, _ in facts]
ans = torch.tensor([cities.index(c) for _, c in facts], device=dev)
N = len(facts)

def last_logits(toks, hooks=()):
    """Logits at the last position only (skips the unembedding of every other position)."""
    resid = model.run_with_hooks(toks, fwd_hooks=list(hooks), stop_at_layer=L)
    return model.unembed(model.ln_final(resid[:, -1:]))[:, 0]

def metrics(lg, a, alt):
    """[3, b]: P(answer), logit diff vs mean of the other 53 cities, logit diff vs alt city."""
    cl = lg[:, cid].float()
    av = cl.gather(1, a[:, None])[:, 0]
    others = (cl.sum(1) - av) / (len(cities) - 1)
    return torch.stack([P.prob(lg, cid[a]), av - others, av - cl.gather(1, alt[:, None])[:, 0]])

# ---- clean run on all prompts
clean_m = torch.zeros(3, N); known = torch.zeros(N, dtype=torch.bool)
for i, (t, k) in enumerate(enc):
    lg = last_logits(t[None])
    clean_m[:, i] = metrics(lg, ans[i:i + 1], ans[i:i + 1])[:, 0].cpu()
    known[i] = (lg[0, cid].argmax() == ans[i]).item()
kn = known.nonzero()[:, 0].tolist()
print(f"{N} prompts, {len(kn)} known (right city first among {len(cities)}); mean P(city) {clean_m[0].mean():.3f}", flush=True)
res = {"setup": model.setup_info, "n_prompts": N, "n_known": len(kn), "window": W, "noise_samples": NS,
       "clean_p_known": P.bootstrap_mean(clean_m[0, kn].numpy()), "clean_ld_known": P.bootstrap_mean(clean_m[1, kn].numpy())}

ROLES = ["first subject", "middle subject", "last subject", "first after", "further", "last"]
def role_of(T, k):
    """Map positions 1..T-1 (0 is the start token) to roles; returns {role: [positions]}."""
    r = defaultdict(list)
    for p in range(1, T):
        if p <= k:
            r["first subject" if p == 1 else "last subject" if p == k else "middle subject"].append(p)
            if p == 1 and k == 1:
                r["last subject"].append(p)
        elif p == k + 1:
            r["first after"].append(p)
        elif p == T - 1:
            r["last"].append(p)
        else:
            r["further"].append(p)
    return r

def by_role(grid_tp, T, k):
    """grid [..., T] (per position) -> [..., 6] role means (nan when a role is empty)."""
    out = np.full(grid_tp.shape[:-1] + (len(ROLES),), np.nan)
    for j, ro in enumerate(ROLES):
        ps = role_of(T, k).get(ro)
        if ps:
            out[..., j] = grid_tp[..., ps].mean(-1)
    return out

# groups of known prompts with equal length (equal length => equal subject length: same template)
groups = defaultdict(list)
for i in kn:
    groups[len(enc[i][0])].append(i)
RES = ["hook_resid_post", "hook_mlp_out", "hook_attn_out"]
names = [f"blocks.{l}.{r}" for l in range(L) for r in RES]

# ---------------------------------------------------------------- 1. Gaussian-noise causal tracing
sig = model.W_E[torch.cat([enc[i][0][1:1 + enc[i][1]] for i in kn])].float().std().item()
res["sigma_subject_embed"] = sig
gn = {}      # per prompt: dict of arrays
t1 = time.time()
for T, idx in sorted(groups.items()):
    k = enc[idx[0]][1]
    toks = torch.stack([enc[i][0] for i in idx])
    a = ans[idx]
    smask = torch.zeros_like(toks, dtype=torch.bool); smask[:, 1:1 + k] = True
    _, clean = P.cache_acts(model, toks, names)
    rt, rm, noise, _ = P.embedding_noise(model, toks, smask, 3.0, NS, seed=T, sigma=sig)
    ra = a.repeat_interleave(NS)
    rep = {n: v.repeat_interleave(NS, 0) for n, v in clean.items()}
    noise_hook = ("hook_embed", P.add_hook(noise, rm))
    _, corr = P.cache_acts(model, rt, [f"blocks.{l}.{r}" for l in range(L) for r in RES[1:]], fwd_hooks=[noise_hook])
    M = lambda hooks: metrics(last_logits(rt, [noise_hook] + hooks), ra, ra).cpu().numpy().reshape(3, len(idx), NS).mean(-1)
    base = M([])
    def pos_mask(p):
        m = torch.zeros_like(rt, dtype=torch.bool); m[:, p] = True; return m
    positions = list(range(1, T)) if FULL else [k, T - 1]
    out = {"hidden": np.full((L, T, 3, len(idx)), np.nan), "mlp": np.full((L, T, 3, len(idx)), np.nan),
           "attn": np.full((L, T, 3, len(idx)), np.nan)}
    for p in range(1, T):
        pm = pos_mask(p)
        for l in range(L):
            out["hidden"][l, p] = M([(f"blocks.{l}.hook_resid_post", P.replace_hook(rep[f"blocks.{l}.hook_resid_post"], pm))])
            if p in positions:
                for kind in ("mlp", "attn"):
                    hk = [(f"blocks.{w}.hook_{kind}_out", P.replace_hook(rep[f"blocks.{w}.hook_{kind}_out"], pm)) for w in P.window(l, W, L)]
                    out[kind][l, p] = M(hk)
    # single-layer MLP at the last subject token, and severed runs (Meng Fig. 3)
    pm = pos_mask(k)
    out["mlp1_last_subj"] = np.stack([M([(f"blocks.{l}.hook_mlp_out", P.replace_hook(rep[f"blocks.{l}.hook_mlp_out"], pm))]) for l in range(L)])
    for kind in ("mlp", "attn"):
        sev = []
        for l in range(L):
            hk = [(f"blocks.{l}.hook_resid_post", P.replace_hook(rep[f"blocks.{l}.hook_resid_post"], pm))]
            hk += [(f"blocks.{j}.hook_{kind}_out", P.replace_hook(corr[f"blocks.{j}.hook_{kind}_out"], pm)) for j in range(l + 1, L)]
            sev.append(M(hk))
        out[f"severed_{kind}"] = np.stack(sev)
    for j, i in enumerate(idx):
        gn[i] = {"T": T, "k": k, "corrupt": base[:, j],
                 **{f"{kind}_roles": by_role(np.moveaxis(out[kind][..., j], 1, -1), T, k) for kind in ("hidden", "mlp", "attn")},   # [L, 3, 6]
                 "mlp1_last_subj": out["mlp1_last_subj"][..., j], "severed_mlp": out["severed_mlp"][..., j],
                 "severed_attn": out["severed_attn"][..., j],
                 "hidden_last_subj": out["hidden"][:, k, :, j]}
        if FULL:
            gn[i]["mlp_max_over_tokens"] = out["mlp"][:, 1:, 0, j].max(1)        # Hase et al.: P metric, max over tokens
    del clean, rep, corr; torch.cuda.empty_cache()
    print(f"GN group T={T} b={len(idx)} {time.time()-t1:.0f}s", flush=True)

# ---------------------------------------------------------------- 2. symmetric token replacement (STR)
# each known prompt is paired with another known prompt of the same token length and a different city
rng = random.Random(0)
pairs = []
for T, idx in groups.items():
    for i in idx:
        cand = [j for j in idx if facts[j][1] != facts[i][1]]
        if cand:
            pairs.append((i, rng.choice(cand)))
res["n_str_pairs"] = len(pairs)
pg = defaultdict(list)
for i, j in pairs:
    pg[len(enc[i][0])].append((i, j))
st = {}
for T, pl in sorted(pg.items()):
    k = enc[pl[0][0]][1]
    ci = [i for i, _ in pl]; cj = [j for _, j in pl]
    tc = torch.stack([enc[i][0] for i in ci]); tr = torch.stack([enc[j][0] for j in cj])
    a, alt = ans[ci], ans[cj]
    _, clean = P.cache_acts(model, tc, names)
    M = lambda hooks: metrics(last_logits(tr, hooks), a, alt).cpu().numpy()
    base_c = metrics(last_logits(tc), a, alt).cpu().numpy(); base_r = M([])
    def pos_mask(p):
        m = torch.zeros_like(tr, dtype=torch.bool); m[:, p] = True; return m
    positions = list(range(1, T)) if FULL else [k, T - 1]
    out = {kk: np.full((L, T, 3, len(pl)), np.nan) for kk in ("hidden", "mlp", "attn")}
    for p in range(1, T):
        pm = pos_mask(p)
        for l in range(L):
            out["hidden"][l, p] = M([(f"blocks.{l}.hook_resid_post", P.replace_hook(clean[f"blocks.{l}.hook_resid_post"], pm))])
            if p in positions:
                for kind in ("mlp", "attn"):
                    out[kind][l, p] = M([(f"blocks.{w}.hook_{kind}_out", P.replace_hook(clean[f"blocks.{w}.hook_{kind}_out"], pm)) for w in P.window(l, W, L)])
    pm = pos_mask(k)
    mlp1 = np.stack([M([(f"blocks.{l}.hook_mlp_out", P.replace_hook(clean[f"blocks.{l}.hook_mlp_out"], pm))]) for l in range(L)])
    for j, (i, jj) in enumerate(pl):
        st[(i, jj)] = {"T": T, "k": k, "clean": base_c[:, j], "corrupt": base_r[:, j],
                       **{f"{kind}_roles": by_role(np.moveaxis(out[kind][..., j], 1, -1), T, k) for kind in ("hidden", "mlp", "attn")},
                       "mlp1_last_subj": mlp1[..., j], "hidden_last_subj": out["hidden"][:, k, :, j]}
    del clean; torch.cuda.empty_cache()
    print(f"STR group T={T} b={len(pl)} {time.time()-t1:.0f}s", flush=True)

# ---------------------------------------------------------------- 3. Part 1's MLP 0 question, all prompts
# Part 1 mean-ablated MLP 0 at every position after the start token. Split that ablation into the
# subject's positions and the template's positions (same per-position means as Part 1), and noise MLP 0
# at the template positions with activations from a different landmark (resample, STR-style).
allg = defaultdict(list)
for i in range(N):
    allg[len(enc[i][0])].append(i)
NT = len(model.to_tokens(TEMPLATE.format(""), prepend_bos=False)[0])     # template tokens after the subject
sums, cnt = torch.zeros(NT + 1, model.cfg.d_model, device=dev), torch.zeros(NT + 1, device=dev)
mlp0 = {}
for T, idx in allg.items():
    toks = torch.stack([enc[i][0] for i in idx])
    _, c = P.cache_acts(model, toks, ["blocks.0.hook_mlp_out"])
    mlp0[T] = c["blocks.0.hook_mlp_out"]
    cls = torch.arange(T - 2, -1, -1, device=dev).clamp(max=NT)      # distance from the end; subject -> NT
    sums.index_add_(0, cls, mlp0[T][:, 1:].sum(0)); cnt.index_add_(0, cls, torch.full((T - 1,), float(len(idx)), device=dev))
means = sums / cnt[:, None]
abl = {"subject": torch.zeros(N), "template": torch.zeros(N), "both": torch.zeros(N), "resample_template": torch.zeros(N),
       "resample_subject": torch.zeros(N), "clean": clean_m[1].clone()}
for T, idx in allg.items():
    toks = torch.stack([enc[i][0] for i in idx]); a = ans[idx]
    k = enc[idx[0]][1]
    cls = torch.arange(T - 2, -1, -1, device=dev).clamp(max=NT)
    mv = means[cls][None].expand(len(idx), -1, -1)
    full = torch.cat([mlp0[T][:, :1], mv], 1)                            # start token left alone
    sm = torch.zeros_like(toks, dtype=torch.bool); sm[:, 1:1 + k] = True
    tm = torch.zeros_like(toks, dtype=torch.bool); tm[:, 1 + k:] = True
    for nm, mk in (("subject", sm), ("template", tm), ("both", sm | tm)):
        abl[nm][idx] = metrics(last_logits(toks, [("blocks.0.hook_mlp_out", P.replace_hook(full, mk))]), a, a)[1].cpu()
    # resample: MLP 0 output from another prompt of the same length (a different landmark and city)
    if len(idx) > 1:
        perm = [(j + 1) % len(idx) for j in range(len(idx))]
        src = mlp0[T][perm]
        for nm, mk in (("resample_template", tm), ("resample_subject", sm)):
            abl[nm][idx] = metrics(last_logits(toks, [("blocks.0.hook_mlp_out", P.replace_hook(src, mk))]), a, a)[1].cpu()
    else:
        for nm in ("resample_template", "resample_subject"):
            abl[nm][idx] = float("nan")
ok = ~torch.isnan(abl["resample_template"])
res["mlp0_split"] = {nm: P.bootstrap_mean((abl["clean"] - v)[ok if nm.startswith("resample") else slice(None)].numpy())
                     for nm, v in abl.items() if nm != "clean"}
res["mlp0_split_n"] = {"all": N, "resample": int(ok.sum())}
print("MLP0 split (logit diff lost):", {k: np.round(v, 3).tolist() for k, v in res["mlp0_split"].items()}, flush=True)

res["wall_s"] = time.time() - t0
res["gpu_peak_mib"] = torch.cuda.max_memory_allocated() // 2**20
res["roles"] = ROLES
gi = sorted(gn); si = sorted(st)
arr = {"gn_ids": np.array(gi), "gn_corrupt": np.stack([gn[i]["corrupt"] for i in gi]), "clean_m": clean_m.numpy(), "known": known.numpy(),
       "str_pairs": np.array(si), "str_clean": np.stack([st[p]["clean"] for p in si]), "str_corrupt": np.stack([st[p]["corrupt"] for p in si])}
for kk in ("hidden_roles", "mlp_roles", "attn_roles", "mlp1_last_subj", "hidden_last_subj"):
    arr[f"gn_{kk}"] = np.stack([gn[i][kk] for i in gi]); arr[f"str_{kk}"] = np.stack([st[p][kk] for p in si])
for kk in ("severed_mlp", "severed_attn") + (("mlp_max_over_tokens",) if FULL else ()):
    arr[f"gn_{kk}"] = np.stack([gn[i][kk] for i in gi])
for nm, v in abl.items():
    arr[f"mlp0_{nm}"] = v.numpy()
np.savez_compressed(f"{ROOT}/runs/trace_{which}.npz", **arr)
json.dump(res, open(f"{ROOT}/post/trace_{which}.json", "w"), indent=1, default=lambda o: np.asarray(o).tolist())
print("done", f"{res['wall_s']:.0f}s", res["gpu_peak_mib"], "MiB", flush=True)
