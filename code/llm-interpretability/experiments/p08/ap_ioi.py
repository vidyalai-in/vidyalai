# Attribution patching vs activation patching on GPT-2 small's IOI task.
# Shared venv: transformer-lens 3.9.0, torch 2.11; GPT-2 small fp32, weight processing on.
# Clean: "When Mary and John went to the store, John gave a drink to" (answer Mary).
# Corrupted: the second subject is flipped to the other name (answer John).
# Denoising patches: clean activation -> corrupted run, one component at a time.
import json, random, sys, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.hooks import load

torch.set_grad_enabled(False)
NAMES = "Mary John Alice Bob Tom James Sarah Kate David Michael Emma Paul Anna Mark Lisa Peter Laura Daniel Rachel Steve Jane Mike Sam Ben Amy Dan Chris Nick Max Joe".split()
PLACES = "store park school beach office restaurant hospital station garden library".split()
OBJECTS = "drink book ball gift letter pen bag cake ring".split()
TEMPLATES = [
    "When {A} and {B} went to the {P}, {S} gave a {O} to",
    "After {A} and {B} went to the {P}, {S} gave a {O} to",
    "While {A} and {B} were working at the {P}, {S} gave a {O} to",
]
PER_TEMPLATE = 40
rng = random.Random(0)

model = load("gpt2", device="cuda")
data = []
for t in TEMPLATES:
    rows = []
    for i in range(PER_TEMPLATE):
        io, s = rng.sample(NAMES, 2)
        p, o = rng.choice(PLACES), rng.choice(OBJECTS)
        first, second = (io, s) if i % 2 == 0 else (s, io)        # ABBA and BABA orders
        clean = t.format(A=first, B=second, P=p, S=s, O=o)
        corr = t.format(A=first, B=second, P=p, S=io, O=o)
        rows.append((clean, corr, " " + io, " " + s))
    data.append(rows)


def ld(logits, a, b):
    return logits[:, -1].gather(1, a[:, None])[:, 0] - logits[:, -1].gather(1, b[:, None])[:, 0]


L, H = model.cfg.n_layers, model.cfg.n_heads
res = dict(heads=dict(act=[], attr=[]), resid=dict(act=[], attr=[]), attn=dict(act=[], attr=[]),
           mlp=dict(act=[], attr=[]), base=[], timing=dict(act=0.0, attr=0.0, n_act_fwd=0))
for rows in data:
    clean = model.to_tokens([r[0] for r in rows]); corr = model.to_tokens([r[1] for r in rows])
    assert clean.shape == corr.shape
    a = torch.tensor([model.to_single_token(r[2]) for r in rows], device="cuda")
    b = torch.tensor([model.to_single_token(r[3]) for r in rows], device="cuda")
    names = lambda n: n.endswith(("hook_z", "hook_resid_pre", "hook_attn_out", "hook_mlp_out"))

    # ---- attribution patching: 2 forward passes + 1 backward pass
    torch.cuda.synchronize(); t0 = time.time()
    clean_logits, cc = model.run_with_cache(clean, names_filter=names)
    with torch.enable_grad():
        grads = {}
        def save_grad(act, hook):
            act.requires_grad_(True)
            act.register_hook(lambda g, n=hook.name: grads.__setitem__(n, g.detach()))
            grads[hook.name + "_act"] = act
            return act
        hooks = [(n, save_grad) for n in cc.keys()]
        corr_logits = model.run_with_hooks(corr, fwd_hooks=hooks)
        ld(corr_logits, a, b).sum().backward()
    attr_heads = torch.stack([((cc[f"blocks.{l}.attn.hook_z"] - grads[f"blocks.{l}.attn.hook_z_act"].detach())
                               * grads[f"blocks.{l}.attn.hook_z"]).sum((1, 3)) for l in range(L)], 1)  # [n, L, H]
    attr_resid = torch.stack([((cc[f"blocks.{l}.hook_resid_pre"] - grads[f"blocks.{l}.hook_resid_pre_act"].detach())
                               * grads[f"blocks.{l}.hook_resid_pre"]).sum(-1) for l in range(L)], 1)  # [n, L, pos]
    attr_attn = torch.stack([((cc[f"blocks.{l}.hook_attn_out"] - grads[f"blocks.{l}.hook_attn_out_act"].detach())
                              * grads[f"blocks.{l}.hook_attn_out"]).sum((1, 2)) for l in range(L)], 1)
    attr_mlp = torch.stack([((cc[f"blocks.{l}.hook_mlp_out"] - grads[f"blocks.{l}.hook_mlp_out_act"].detach())
                             * grads[f"blocks.{l}.hook_mlp_out"]).sum((1, 2)) for l in range(L)], 1)
    torch.cuda.synchronize(); res["timing"]["attr"] += time.time() - t0
    ld_clean, ld_corr = ld(clean_logits, a, b), ld(corr_logits.detach(), a, b)
    res["base"].append(torch.stack([ld_clean, ld_corr], 1).cpu())
    del grads

    # ---- activation patching: one forward pass per component
    torch.cuda.synchronize(); t0 = time.time()
    def patch_run(name, fn):
        out = model.run_with_hooks(corr, fwd_hooks=[(name, fn)])
        res["timing"]["n_act_fwd"] += 1
        return ld(out, a, b) - ld_corr
    act_heads = torch.zeros(len(rows), L, H, device="cuda")
    for l in range(L):
        for h in range(H):
            def f(z, hook, h=h):
                z[:, :, h] = cc[hook.name][:, :, h]; return z
            act_heads[:, l, h] = patch_run(f"blocks.{l}.attn.hook_z", f)
    npos = clean.shape[1]
    act_resid = torch.zeros(len(rows), L, npos, device="cuda")
    for l in range(L):
        for p in range(npos):
            def f(x, hook, p=p):
                x[:, p] = cc[hook.name][:, p]; return x
            act_resid[:, l, p] = patch_run(f"blocks.{l}.hook_resid_pre", f)
    act_attn = torch.zeros(len(rows), L, device="cuda"); act_mlp = torch.zeros(len(rows), L, device="cuda")
    for l in range(L):
        act_attn[:, l] = patch_run(f"blocks.{l}.hook_attn_out", lambda x, hook: cc[hook.name])
        act_mlp[:, l] = patch_run(f"blocks.{l}.hook_mlp_out", lambda x, hook: cc[hook.name])
    torch.cuda.synchronize(); res["timing"]["act"] += time.time() - t0
    for k, act, attr in [("heads", act_heads, attr_heads), ("resid", act_resid, attr_resid),
                         ("attn", act_attn, attr_attn), ("mlp", act_mlp, attr_mlp)]:
        res[k]["act"].append(act.cpu()); res[k]["attr"].append(attr.cpu())
    print("template done", res["timing"], flush=True)

torch.save(res, "/home/amit/.cache/vidyalai-scratch/p08/runs/ap_ioi.pt")
print("saved; tokens per template:", [model.to_tokens(r[0][0]).shape[1] for r in data])
