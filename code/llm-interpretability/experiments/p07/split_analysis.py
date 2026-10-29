"""Feature splitting between our TopK SAEs (3,072 vs 24,576 latents) on GPT-2 layer 8, CPU only.
For each small-SAE latent: decoder cosine to every large-SAE latent; count children above a threshold;
print snippets (from the held-out eval windows) for chosen latents.
usage: split_analysis.py stats | show <small_latent>
"""
import sys, json
import torch
from transformers import AutoTokenizer

R = "/home/amit/.cache/vidyalai-scratch/p07/runs"
ck = torch.load(f"{R}/main.saes.pt")
ex = torch.load(f"{R}/gpt2_eval_extra.pt")
D = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gpt2-tokens.pt")
ev = D["eval"].to(torch.int32)
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
Ws = ck["sae"]["topk32_3k"]["W_dec"]; Wb = ck["sae"]["topk32_24k"]["W_dec"]
Ws = Ws / Ws.norm(dim=1, keepdim=True); Wb = Wb / Wb.norm(dim=1, keepdim=True)
C = Ws @ Wb.t()                                   # [3072, 24576]


def snip(name, lat, n=6):
    out = []
    for v, i in zip(ex[name]["top_v"][lat][:n].tolist(), ex[name]["top_i"][lat][:n].tolist()):
        if v <= 0:
            continue
        w, p = divmod(int(i), 127); p += 1
        ids = ev[w].tolist()
        out.append(f"{v:5.1f} | {tok.decode(ids[max(1, p - 10):p])}[[{tok.decode([ids[p]])}]]{tok.decode(ids[p + 1:p + 3])}".replace("\n", "\\n"))
    return out


if sys.argv[1] == "stats":
    best = C.max(1).values
    n_child = (C > 0.5).sum(1)
    print("small latents: median best cosine to a large latent", round(best.median().item(), 3),
          "share with best > 0.9", round((best > 0.9).float().mean().item(), 3),
          "share with >=2 large latents above 0.5", round((n_child >= 2).float().mean().item(), 3))
    cnt = ex["topk32_3k"]["counts"]
    cand = [(int(n_child[i]), i) for i in range(len(n_child)) if n_child[i] >= 3 and cnt[i] > 200]
    cand.sort(reverse=True)
    print("candidates (children>0.5, small latent):", cand[:40])
    json.dump({"median_best_cos": best.median().item(), "share_best_gt_0.9": (best > 0.9).float().mean().item(),
               "share_ge2_children_gt_0.5": (n_child >= 2).float().mean().item()}, open(f"{R}/split_stats.json", "w"))
else:
    s = int(sys.argv[2])
    print("SMALL", s, "fires", int(ex["topk32_3k"]["counts"][s]))
    for l in snip("topk32_3k", s, 8):
        print("   ", l)
    vals, idx = C[s].topk(6)
    for c, b in zip(vals.tolist(), idx.tolist()):
        print(f"  LARGE {b} cos {c:.2f} fires {int(ex['topk32_24k']['counts'][b])}")
        for l in snip("topk32_24k", b, 4):
            print("      ", l)
