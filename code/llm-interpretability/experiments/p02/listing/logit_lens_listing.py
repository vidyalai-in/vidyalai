from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")  # fp32, weights as published
prompt = "Neil Armstrong was the first person to walk on the"
logits, cache = model.run_with_cache(prompt)

# The residual stream at 13 points: the embeddings, then after each of the 12 blocks
points = [cache[f"blocks.{l}.hook_resid_pre"] for l in range(12)]
points.append(cache["blocks.11.hook_resid_post"])

for l, h in enumerate(points):
    probs = model.unembed(model.ln_final(h[:, -1:]))[0, -1].softmax(-1)  # the logit lens
    top = probs.argmax()
    print(l, repr(model.to_string(top)), round(probs[top].item(), 3))
