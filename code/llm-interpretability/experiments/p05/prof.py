import sys, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import hooks as H, patching as P
torch.set_grad_enabled(False)
m = H.load("gpt2", dtype=torch.float32, processing=True)
toks = torch.randint(1000, 40000, (66, 15)).cuda()
mask = torch.zeros(66, 15, dtype=torch.bool, device="cuda"); mask[:, 5] = True
_, c = P.cache_acts(m, toks, ["blocks.5.attn.hook_z"])
def t(fn, n=20):
    fn(); torch.cuda.synchronize(); s = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - s) / n * 1000
hk = [("blocks.5.attn.hook_z", P.replace_hook(c["blocks.5.attn.hook_z"], mask, 3))]
print("plain fwd", t(lambda: m(toks)))
print("hooked fwd", t(lambda: m.run_with_hooks(toks, fwd_hooks=hk)))
print("stop_at_layer", t(lambda: m.run_with_hooks(toks, fwd_hooks=hk, stop_at_layer=12)))
torch.backends.cuda.matmul.allow_tf32 = True
print("tf32 hooked", t(lambda: m.run_with_hooks(toks, fwd_hooks=hk)))
print("tf32 stop", t(lambda: m.run_with_hooks(toks, fwd_hooks=hk, stop_at_layer=12)))
m.set_use_split_qkv_input(True)
print("split tf32 stop", t(lambda: m.run_with_hooks(toks, fwd_hooks=hk, stop_at_layer=12)))
