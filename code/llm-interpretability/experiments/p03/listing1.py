import torch
import transformer_lens as tl

model = tl.HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
tokens = model.to_tokens("The old man the boats.")      # prepends <|endoftext|>
_, cache = model.run_with_cache(tokens)
h = cache["blocks.6.hook_resid_pre"][0]                  # layer 6: one 768-number vector per token
print(model.to_str_tokens(tokens), h.shape)

probe = torch.nn.Linear(768, 17)                         # the whole probe: 17 x 768 weights + 17 biases
print(sum(p.numel() for p in probe.parameters()))
scores = probe(h)                                        # one score per tag, for every token
print(scores.shape)                                      # train with cross-entropy against the gold tags
