# Part 7: how similar the embedding rows of never-seen tokens became (CPU only).
import torch
R = "/home/amit/.cache/vidyalai-scratch/t07/runs/main88"
specials = list(range(32768 - 0, 32768))  # placeholder
import sys; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from tokenizer import load_production
tok = load_production()
ids = [tok.token_to_id(t) for t in ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>"]]
print("special ids", ids)
for step in (0, 3360):
    E = torch.load(f"{R}/model_{step:05d}.pt", map_location="cpu", weights_only=True)["tok_emb.weight"]
    S = torch.nn.functional.normalize(E[ids], dim=1); c = S @ S.T
    off = c[~torch.eye(len(ids), dtype=bool)]
    rnd = torch.nn.functional.normalize(E[torch.randint(0, 32000, (200,), generator=torch.Generator().manual_seed(0))], dim=1)
    cr = (rnd @ rnd.T)[~torch.eye(200, dtype=bool)]
    print(f"step {step}: specials pairwise cosine mean {off.mean():.3f} min {off.min():.3f}; random rows mean |cos| {cr.abs().mean():.3f}")
