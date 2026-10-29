from functools import partial
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2", device="cuda")   # fp32, LayerNorm folded
clean = "When Mary and John went to the store, John gave a drink to"
corrupt = "When Mary and John went to the store, Mary gave a drink to"
mary, john = model.to_single_token(" Mary"), model.to_single_token(" John")

def logit_diff(logits):                     # Mary minus John, at the last position
    return (logits[0, -1, mary] - logits[0, -1, john]).item()

clean_logits, cache = model.run_with_cache(clean)
ld_clean, ld_corrupt = logit_diff(clean_logits), logit_diff(model(corrupt))
s2 = [i for i, t in enumerate(model.to_str_tokens(clean)) if t == " John"][1]
print(f"clean {ld_clean:+.2f}, corrupted {ld_corrupt:+.2f}, S2 at position {s2}")

def patch(act, hook, pos):                  # paste the clean activation at one position
    act[:, pos] = cache[hook.name][:, pos]
    return act

for layer in (0, 6, 8, 9, 10):
    row = []
    for name, pos in (("S2", s2), ("END", -1)):
        hook = (f"blocks.{layer}.hook_resid_pre", partial(patch, pos=pos))
        ld = logit_diff(model.run_with_hooks(corrupt, fwd_hooks=[hook]))
        row.append(f"{name} {(ld - ld_corrupt) / (ld_clean - ld_corrupt):.2f}")
    print(f"block {layer:2d}:", *row)
