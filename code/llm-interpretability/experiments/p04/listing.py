import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device="cuda")  # fp32, LayerNorm folded
torch.manual_seed(0)
r = torch.randint(1000, 49257, (100, 50))                  # 100 sequences of 50 random tokens
bos = torch.full((100, 1), model.tokenizer.bos_token_id)
tokens = torch.cat([bos, r, r], dim=1).cuda()               # [BOS] + r + r: 101 tokens

logits, cache = model.run_with_cache(tokens)
loss = -logits.log_softmax(-1)[:, :-1].gather(-1, tokens[:, 1:, None])[..., 0]
print(f"loss, first copy {loss[:, 1:50].mean():.2f}, second copy {loss[:, 51:].mean():.2f}")

d = torch.arange(51, 101)                                   # positions in the second copy
ind = torch.stack([cache["pattern", l][:, :, d, d - 49].mean((0, 2)) for l in range(12)])
top = ind.flatten().topk(5)
print([f"{i // 12}.{i % 12}: {v:.2f}" for v, i in zip(top.values.tolist(), top.indices.tolist())])
