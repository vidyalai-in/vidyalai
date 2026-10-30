import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2", device="cuda")   # fp32, LayerNorm folded
top5 = [(5, 5), (6, 9), (7, 10), (5, 1), (7, 2)]                   # Part 4's induction heads

def repeated(seed, n=200, half=50):                                # [BOS] + r + r, as in Part 4
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(1000, 49257, (n, half), generator=g)
    return torch.cat([torch.full((n, 1), 50256), r, r], 1).cuda()

ref, test = repeated(103), repeated(102)                           # new seeds; Part 4 used 3 and 2
_, cache = model.run_with_cache(ref, names_filter=lambda n: n.endswith("hook_z"))
mean_z = {l: cache[f"blocks.{l}.attn.hook_z"].mean((0, 1)) for l in range(12)}

def ablate(z, hook):                                               # replace each head's output by its mean
    for l, h in top5:
        if hook.layer() == l:
            z[:, :, h] = mean_z[l][h]
    return z

for name, hooks in [("intact", []), ("top-5 ablated", [(f"blocks.{l}.attn.hook_z", ablate) for l in (5, 6, 7)])]:
    logits = model.run_with_hooks(test, fwd_hooks=hooks)
    loss = -logits.log_softmax(-1)[:, :-1].gather(-1, test[:, 1:, None])[..., 0]
    print(f"{name}: first copy {loss[:, 1:50].mean():.2f}, second copy {loss[:, 51:].mean():.2f} nats")
