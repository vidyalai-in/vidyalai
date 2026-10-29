import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2", device="cuda")  # fp32, LayerNorm folded
prompt = "The Brandenburg Gate is located in the city of"
logits, cache = model.run_with_cache(prompt)
print(model.to_str_tokens(prompt))
print(len(cache), "cached activations;", cache["blocks.9.hook_attn_out"].shape)
print("top guess:", model.to_string(logits[0, -1].argmax()))

# Direct logit attribution: what each piece writes toward " Berlin" rather than " Munich"
ans, alt = model.to_single_token(" Berlin"), model.to_single_token(" Munich")
direction = model.W_U[:, ans] - model.W_U[:, alt]
parts, labels = cache.decompose_resid(layer=-1, apply_ln=True, pos_slice=-1, return_labels=True)
dla = parts[:, 0] @ direction
for i in dla.abs().argsort(descending=True)[:4]:
    print(f"{labels[i]:>8}  {dla[i]:+.2f}")
print("logit diff:", round((logits[0, -1, ans] - logits[0, -1, alt]).item(), 2))

# Ablation: zero MLP 0's output at every position after <|endoftext|>, rerun
def zero_mlp0(act, hook):
    act[:, 1:] = 0
    return act

ablated = model.run_with_hooks(prompt, fwd_hooks=[("blocks.0.hook_mlp_out", zero_mlp0)])
print("ablated logit diff:", round((ablated[0, -1, ans] - ablated[0, -1, alt]).item(), 2))
print("ablated top guess:", model.to_string(ablated[0, -1].argmax()))
