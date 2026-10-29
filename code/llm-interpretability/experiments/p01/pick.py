import sys, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/p01/scripts")
from facts import FACTS, TEMPLATE
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
m = HookedTransformer.from_pretrained("gpt2", device="cuda")
def z(act, hook): act[:, 1:] = 0; return act
for l, c in FACTS:
    p = TEMPLATE.format(l); lg = m(p)[0, -1]
    if m.to_string(lg.argmax()) != " " + c: continue
    top2 = [m.to_string(t) for t in lg.topk(3).indices]
    ab = m.run_with_hooks(p, fwd_hooks=[("blocks.0.hook_mlp_out", z)])[0, -1]
    print(f"{p!r:60s} top3 {top2} ablated top {m.to_string(ab.argmax())!r}")
