# Preview of Part 9: does adding the truth probe's direction change what the model SAYS?
# Few-shot TRUE/FALSE prompt as in Marks & Tegmark (2023, sec. 6). For each statement we add the
# mass-mean direction (mean true - mean false, from the other folds' cities) to the residual stream
# at the statement's final "." token at one layer: + for false statements, - for true ones.
# Normalized indirect effect (NIE): 1 = false statements now look as TRUE as real true ones.
# Usage: nudge.py gpt2|qwen
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.probes import load_model, bos_id, load_layer, mass_mean_direction, bootstrap_ci
from sklearn.model_selection import GroupKFold

R = "/home/amit/.cache/vidyalai-scratch/p03/runs"
which = sys.argv[1]
name, dtype = {"gpt2": ("gpt2", torch.float32), "qwen": ("Qwen/Qwen2.5-0.5B", torch.bfloat16)}[which]
model = load_model(name, dtype=dtype); tok = model.tokenizer
L = model.cfg.n_layers
THE = {"United States", "United Kingdom", "United Arab Emirates", "Philippines",
       "Democratic Republic of the Congo", "Dominican Republic"}
cname = lambda c: ("the " + c) if c in THE else c
PREFIX = ("The city of Paris is in France. This statement is: TRUE\n"
          "The city of Lima is in Japan. This statement is: FALSE\n"
          "The city of Sydney is in Canada. This statement is: FALSE\n"
          "The city of Madrid is in Spain. This statement is: TRUE\n")
T_ID = tok(" TRUE", add_special_tokens=False)["input_ids"][0]
F_ID = tok(" FALSE", add_special_tokens=False)["input_ids"][0]
assert T_ID != F_ID
items = json.load(open(f"{R}/truth_data.json"))
y = np.array([it["label"] for it in items]); city = np.array([it["city"] for it in items])
toks, per = [], []
for it in items:
    s = f"The city of {it['city']} is in {cname(it['country'])}."
    a = [bos_id(tok)] + tok(PREFIX + s, add_special_tokens=False)["input_ids"]
    b = tok(" This statement is:", add_special_tokens=False)["input_ids"]
    toks.append(a + b); per.append(len(a) - 1)  # index of the statement's "."
T = max(map(len, toks))
tt = torch.zeros(len(toks), T, dtype=torch.long)
for i, t in enumerate(toks):
    tt[i, :len(t)] = torch.tensor(t)
last = torch.tensor([len(t) - 1 for t in toks]); per_t = torch.tensor(per)
BS = 96

@torch.no_grad()
def run(layer=None, vecs=None, cache_layers=False):
    """Return P(TRUE)-P(FALSE) per item (and optionally the in-context '.' activations)."""
    pd, acts = [], []
    for b in range(0, len(toks), BS):
        sl = slice(b, b + BS); x = tt[sl].cuda(); pp = per_t[sl].cuda(); ll = last[sl].cuda()
        hooks = []
        if layer is not None:
            v = vecs[sl].to(model.cfg.device, dtype=model.cfg.dtype)
            def h(resid, hook):
                resid[torch.arange(len(pp)), pp] += v
                return resid
            hooks = [(f"blocks.{layer}.hook_resid_pre", h)]
        if cache_layers:
            names = [f"blocks.{l}.hook_resid_pre" for l in range(L)] + [f"blocks.{L-1}.hook_resid_post"]
            logits, cache = model.run_with_cache(x, names_filter=lambda n: n in names)
            acts.append(torch.stack([cache[n][torch.arange(len(pp)), pp].float().cpu() for n in names], 1))
        else:
            logits = model.run_with_hooks(x, fwd_hooks=hooks)
        p = logits[torch.arange(len(ll)), ll].float().softmax(-1)
        pd.append((p[:, T_ID] - p[:, F_ID]).cpu())
    pd = torch.cat(pd).numpy()
    return (pd, torch.cat(acts).numpy()) if cache_layers else pd

t0 = time.time()
pd0, ctx = run(cache_layers=True)
base = {"n": len(y), "acc_says_true_when_true": bootstrap_ci((pd0 > 0) == (y == 1), city),
        "PD_true": float(pd0[y == 1].mean()), "PD_false": float(pd0[y == 0].mean())}
print(json.dumps(base), f"{time.time()-t0:.0f}s", flush=True)
folds = list(GroupKFold(n_splits=5).split(y, y, groups=city))
sign = np.where(y == 1, -1.0, 1.0)  # push false toward true, true toward false
rng = np.random.default_rng(0)

def dirs(source, l):
    D = np.zeros((len(y), ctx.shape[2]), np.float32)
    for tr, te in folds:
        X = load_layer(f"{R}/acts/{which}/truth_cities", l) if source == "alone" else ctx[:, l]
        D[te] = mass_mean_direction(X[tr], y[tr])
    return D

def nie_ci(pd, n_boot=1000):
    """95% bootstrap interval of both NIEs, resampling cities."""
    uc, inv = np.unique(city, return_inverse=True); r = np.random.default_rng(1); f, t = [], []
    for _ in range(n_boot):
        k = np.bincount(r.integers(0, len(uc), len(uc)), minlength=len(uc))[inv].astype(float)
        m = lambda a, sel: (a[sel] * k[sel]).sum() / k[sel].sum()
        T1, F1, T0, F0 = m(pd, y == 1), m(pd, y == 0), m(pd0, y == 1), m(pd0, y == 0)
        f.append((F1 - F0) / (T0 - F0)); t.append((T1 - T0) / (F0 - T0))
    return [float(x) for x in np.quantile(f, [0.025, 0.975])] + [float(x) for x in np.quantile(t, [0.025, 0.975])]

def nie(pd):
    f2t = (pd[y == 0].mean() - pd0[y == 0].mean()) / (pd0[y == 1].mean() - pd0[y == 0].mean())
    t2f = (pd[y == 1].mean() - pd0[y == 1].mean()) / (pd0[y == 0].mean() - pd0[y == 1].mean())
    return float(f2t), float(t2f)

out = {"base": base, "layers": {}}
LAYERS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(0, L, 2 if L > 12 else 1))
for l in LAYERS:
    if l == L:
        continue  # hook_resid_pre exists only for l < L
    r = {}
    for source in ["alone", "context"]:
        D = dirs(source, l)
        for alpha in [1.0, 3.0]:
            pd = run(l, torch.tensor(alpha * sign[:, None] * D))
            r[f"{source}_a{alpha:g}"] = nie(pd)
            if len(sys.argv) > 2:
                r[f"{source}_a{alpha:g}_ci"] = nie_ci(pd)
    D = dirs("context", l)
    rnd = rng.standard_normal(D.shape).astype(np.float32)
    rnd *= np.linalg.norm(D, axis=1, keepdims=True) / np.linalg.norm(rnd, axis=1, keepdims=True)
    pd = run(l, torch.tensor(3.0 * sign[:, None] * rnd))
    r["random_a3"] = nie(pd)
    if len(sys.argv) > 2:
        r["random_a3_ci"] = nie_ci(pd)
    r["dir_norm"] = float(np.linalg.norm(D, axis=1).mean())
    r["resid_norm"] = float(np.linalg.norm(ctx[:, l], axis=1).mean())
    out["layers"][l] = r
    print(l, {k: (tuple(round(x, 3) for x in v) if isinstance(v, (tuple, list)) else round(v, 2)) for k, v in r.items()},
          f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open(f"{R}/nudge_{which}{'_fine' if len(sys.argv) > 2 else ''}.json", "w"), indent=1)
print("peak GPU MiB", torch.cuda.max_memory_allocated() // 2**20)
