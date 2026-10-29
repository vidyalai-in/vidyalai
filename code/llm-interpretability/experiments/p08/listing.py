import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device="cuda")  # fp32, LayerNorm folded
clean = model.to_tokens("When Mary and John went to the store, John gave a drink to")
corr = model.to_tokens("When Mary and John went to the store, Mary gave a drink to")
mary, john = model.to_single_token(" Mary"), model.to_single_token(" John")
ld = lambda logits: logits[0, -1, mary] - logits[0, -1, john]   # logit difference

_, clean_cache = model.run_with_cache(clean)
acts, grads = {}, {}
for l in range(12):                                  # cache z and its gradient on the corrupted run
    model.add_hook(f"blocks.{l}.attn.hook_z", lambda x, hook: acts.__setitem__(hook.name, x.detach()))
    model.add_hook(f"blocks.{l}.attn.hook_z", lambda g, hook: grads.__setitem__(hook.name, g.detach()), dir="bwd")
corr_ld = ld(model(corr))
corr_ld.backward()
model.reset_hooks()

# Attribution patching: (clean - corrupted) x gradient, for all 144 heads at once
est = torch.stack([((clean_cache[n] - acts[n]) * grads[n]).sum((0, 1, 3)) for n in sorted(acts, key=lambda n: int(n.split(".")[1]))])
print("logit difference, clean", round(ld(model(clean)).item(), 2), "corrupted", round(corr_ld.item(), 2))
for i in est.flatten().abs().argsort(descending=True)[:4].tolist():
    l, h = divmod(i, 12)
    def patch(z, hook): z[:, :, h] = clean_cache[hook.name][:, :, h]; return z
    with torch.no_grad():
        true = ld(model.run_with_hooks(corr, fwd_hooks=[(f"blocks.{l}.attn.hook_z", patch)])) - corr_ld
    print(f"head {l}.{h}: estimate {est[l, h]:+.2f}, true patch {true.item():+.2f}")
