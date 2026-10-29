import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2", device="cuda")      # fp32, LayerNorm folded
sae = SAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre", device="cuda")

prompt = "The Eiffel Tower is located in the city of"
_, cache = model.run_with_cache(prompt)
x = cache["blocks.8.hook_resid_pre"][0, 1:]       # 11 tokens x 768 numbers, start token dropped
f = sae.encode(x)                                  # 11 tokens x 24,576 latents
x_hat = sae.decode(f)

print("latents active per token:", (f > 0).sum(-1).tolist())
err = (x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
print(f"fraction of variance explained: {1 - err:.3f}")
v, i = f[-1].topk(3)
print("strongest latents on ' of':", i.tolist(), [round(a, 1) for a in v.tolist()])
