"""Part 5, experiment 1: activation patching and path patching on IOI in GPT-2 small.
Writes runs/ioi.npz (per-prompt arrays) and post/ioi.json (summaries with 95% bootstrap CIs)."""
import json, sys, time
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/p05/scripts")
from lab import hooks as H, patching as P
from ioi_data import build, TEMPLATES

ROOT = "/home/amit/.cache/vidyalai-scratch/p05"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
QUICK = "--quick" in sys.argv
PART = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "AB"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
t0 = time.time()
model = H.load("gpt2", dtype=torch.float32, processing=True)
L, NH = model.cfg.n_layers, model.cfg.n_heads
data, info = build(model.tokenizer, N, seed=0)
dev = model.cfg.device

# ---- tokenize and group by length
tok = lambda s: model.to_tokens(s)[0]
groups = defaultdict(list)
for i, d in enumerate(data):
    groups[len(tok(d["clean"]))].append(i)
ROLES = ["IO", "S1", "S1+1", "S2", "S2+1..END-1", "END", "other"]
G = []
for T, idx in sorted(groups.items()):
    g = dict(idx=idx)
    g["clean"] = torch.stack([tok(data[i]["clean"]) for i in idx])
    g["str"] = torch.stack([tok(data[i]["str_s2"]) for i in idx])
    g["abc"] = torch.stack([tok(data[i]["abc"]) for i in idx])
    assert g["str"].shape == g["clean"].shape == g["abc"].shape
    g["io"] = torch.tensor([model.to_single_token(" " + data[i]["io"]) for i in idx], device=dev)
    g["s"] = torch.tensor([model.to_single_token(" " + data[i]["s"]) for i in idx], device=dev)
    b = len(idx)
    masks = {r: torch.zeros(b, T, dtype=torch.bool, device=dev) for r in ROLES}
    for j in range(b):
        c = g["clean"][j]
        io_p = (c == g["io"][j]).nonzero()[:, 0].tolist()
        s_p = (c == g["s"][j]).nonzero()[:, 0].tolist()
        assert len(io_p) == 1 and len(s_p) == 2, (data[idx[j]]["clean"], io_p, s_p)
        masks["IO"][j, io_p[0]] = True; masks["S1"][j, s_p[0]] = True
        masks["S1+1"][j, s_p[0] + 1] = True; masks["S2"][j, s_p[1]] = True
        masks["S2+1..END-1"][j, s_p[1] + 1:T - 1] = True; masks["END"][j, T - 1] = True
    used = torch.zeros_like(masks["IO"])
    for r in ROLES[:-1]:
        used |= masks[r]
    masks["other"] = ~used
    masks["other"][:, 0] = False          # BOS is identical everywhere
    g["masks"] = masks
    G.append(g)
order = np.concatenate([g["idx"] for g in G])
inv = np.argsort(order)
print(f"{N} prompts, {len(G)} length groups, {info}, setup {model.setup_info}", flush=True)

def metric_fn(g):
    def m(logits):
        lg = (logits[:, -1] if logits.dim() == 3 else logits).float()
        return torch.stack([P.logit_diff(lg, g["io"], g["s"]), P.prob(lg, g["io"]), P.prob(lg, g["s"])])
    return m

# Gaussian noise on S2's token embedding (Zhang & Nanda's GN for IOI): 3 x std of the name embeddings
name_ids = torch.tensor(sorted({model.to_single_token(" " + d[k]) for d in data for k in ("io", "s")}), device=dev)
SIGMA = model.W_E[name_ids].float().std().item()
gen = torch.Generator().manual_seed(1)
for g in G:
    b, T = g["clean"].shape
    g["noise"] = (torch.randn(b, T, model.cfg.d_model, generator=gen) * 3 * SIGMA).to(dev)
    g["gn_hook"] = ("hook_embed", P.add_hook(g["noise"], g["masks"]["S2"]))


res = {"n": N, "info": info, "sigma_name_embed": SIGMA, "setup": model.setup_info, "roles": ROLES}
NAMES = [f"blocks.{l}.hook_resid_pre" for l in range(L)] + [f"blocks.{l}.hook_attn_out" for l in range(L)] \
    + [f"blocks.{l}.hook_mlp_out" for l in range(L)] + [f"blocks.{l}.attn.hook_z" for l in range(L)]
ZN = [f"blocks.{l}.attn.hook_z" for l in range(L)]
PN = [f"blocks.{l}.attn.hook_pattern" for l in (9, 10)]
NM = [(9, 9), (9, 6), (10, 0)]
SI = [(7, 3), (7, 9), (8, 6), (8, 10)]
cfgs = {"den_str": ("str", "c_clean", None), "den_gn": ("clean", "c_clean", "gn"),
        "den_abc": ("abc", "c_clean", None), "noi_str": ("clean", "c_str", None)}
if QUICK:
    cfgs = {"den_str": cfgs["den_str"]}
PATHS = ("logits",) if QUICK else tuple(t for t, pt in (("logits", "A"), ("nm_q", "B"), ("si_v", "B")) if pt in PART)


def do_group(g):
    m = metric_fn(g)
    out = {}
    lg, g["c_clean"] = P.cache_acts(model, g["clean"], NAMES + PN); out["base_clean"] = m(lg).cpu().numpy()
    lg, g["c_str"] = P.cache_acts(model, g["str"], ZN + PN); out["base_str"] = m(lg).cpu().numpy()
    lg, g["c_abc"] = P.cache_acts(model, g["abc"], ZN); out["base_abc"] = m(lg).cpu().numpy()
    lg, g["c_gn"] = P.cache_acts(model, g["clean"], PN, fwd_hooks=[g["gn_hook"]]); out["base_gn"] = m(lg).cpu().numpy()
    for k in ("clean", "str", "gn"):     # name movers' attention at END to IO / S1 / S2
        pat = torch.stack([g[f"c_{k}"][f"blocks.{l}.attn.hook_pattern"][:, h] for l, h in NM]).mean(0)[:, -1]
        out[f"att_{k}"] = torch.stack([(pat * g["masks"][r].float()).sum(-1) for r in ("IO", "S1", "S2")]).cpu().numpy()
    if "A" not in PART:
        return do_paths(g, m, out)
    sw = lambda dest, src, items, extra=None: P.sweep(model, g[dest], g[src], items, m,
                                                     extra_hooks=[g["gn_hook"]] if extra == "gn" else ())
    # 1. residual stream by layer x position group (denoising), STR and ABC
    for corr in ("str", "abc"):
        r = sw(corr, "c_clean", [(f"{l}|{ro}", [(f"blocks.{l}.hook_resid_pre", g["masks"][ro], None)]) for l in range(L) for ro in ROLES])
        out[f"resid_{corr}"] = np.stack([[r[f"{l}|{ro}"] for ro in ROLES] for l in range(L)])
    # 2. attention vs MLP output per layer (denoising STR), all positions and END only
    allpos = torch.ones_like(g["masks"]["END"])
    for where, mk in (("all", allpos), ("END", g["masks"]["END"])):
        r = sw("str", "c_clean", [(f"{k}{l}", [(f"blocks.{l}.hook_{k}_out", mk, None)]) for k in ("attn", "mlp") for l in range(L)])
        out[f"layers_{where}"] = np.stack([[r[f"{k}{l}"] for l in range(L)] for k in ("attn", "mlp")])
    # 3. per head, all positions
    for name, (dest, src, extra) in cfgs.items():
        r = sw(dest, src, [(f"{l}.{h}", [(f"blocks.{l}.attn.hook_z", allpos, h)]) for l in range(L) for h in range(NH)], extra)
        out[f"heads_{name}"] = np.stack([[r[f"{l}.{h}"] for h in range(NH)] for l in range(L)])
    r = sw("str", "c_clean", [(f"{l}.{h}", [(f"blocks.{l}.attn.hook_z", g["masks"]["END"], h)]) for l in range(L) for h in range(NH)])
    out["heads_den_str_END"] = np.stack([[r[f"{l}.{h}"] for h in range(NH)] for l in range(L)])
    return do_paths(g, m, out)


def do_paths(g, m, out):
    # 4. path patching (Wang et al.): x_orig = clean, x_new = ABC. Receivers' q/v inputs need
    # split qkv inputs, which triples the cost of a forward pass, so it is switched on only here.
    oz = {l: g["c_clean"][f"blocks.{l}.attn.hook_z"] for l in range(L)}
    nz = {l: g["c_abc"][f"blocks.{l}.attn.hook_z"] for l in range(L)}
    for tgt in PATHS:
        model.set_use_split_qkv_input(tgt != "logits")
        grid = np.full((L, NH, 3, len(g["idx"])), np.nan)
        for l in range(L):
            for h in range(NH):
                if tgt == "logits":
                    grid[l, h] = P.path_patch(model, g["clean"], oz, nz, (l, h), "logits", m, sender_mask=g["masks"]["END"])
                elif tgt == "nm_q" and l < 10:
                    grid[l, h] = P.path_patch(model, g["clean"], oz, nz, (l, h), [(rl, rh, "q", g["masks"]["END"]) for rl, rh in NM if rl > l], m)
                elif tgt == "si_v" and l < 8:
                    grid[l, h] = P.path_patch(model, g["clean"], oz, nz, (l, h), [(rl, rh, "v", g["masks"]["S2"]) for rl, rh in SI if rl > l], m)
        out[f"path_{tgt}"] = grid
    model.set_use_split_qkv_input(False)
    for k in ("c_clean", "c_str", "c_abc", "c_gn"):
        del g[k]
    torch.cuda.empty_cache()
    return out


parts = defaultdict(list)
t1 = time.time()
for gi, g in enumerate(G):
    for k, v in do_group(g).items():
        parts[k].append(v)
    print(f"group {gi+1}/{len(G)} (b={len(g['idx'])}, T={g['clean'].shape[1]}) {time.time()-t1:.0f}s", flush=True)
arrs = {k: np.concatenate(v, -1)[..., inv] for k, v in parts.items()}
res["baseline"] = {}
for k in ("clean", "str", "abc", "gn"):
    v = arrs[f"base_{k}"]
    res["baseline"][k] = {"ld": P.bootstrap_mean(v[0]), "p_io": P.bootstrap_mean(v[1]), "p_s": P.bootstrap_mean(v[2]),
                          "acc_io_over_s": float((v[0] > 0).mean())}
    print(k, {a: np.round(np.array(b, dtype=float), 3).tolist() for a, b in res["baseline"][k].items()})
res["nm_attention"] = {k: dict(zip(("IO", "S1", "S2"), np.round(arrs[f"att_{k}"].mean(-1), 3).tolist())) for k in ("clean", "str", "gn")}
print("name-mover attention at END", res["nm_attention"], flush=True)
res["wall_s"] = time.time() - t0
res["gpu_peak_mib"] = torch.cuda.max_memory_allocated() // 2**20
tag = ("_quick" if QUICK else "") + ("" if PART == "AB" else "_" + PART)
np.savez_compressed(f"{ROOT}/runs/ioi{tag}.npz", **arrs)
json.dump(res, open(f"{ROOT}/post/ioi_base{tag}.json", "w"), indent=1, default=lambda o: np.asarray(o).tolist())
print("done", f"{res['wall_s']:.0f}s", res["gpu_peak_mib"], "MiB", flush=True)
