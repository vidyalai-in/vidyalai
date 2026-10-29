"""Turn runs/gemma_explore.pt into readable top-activation snippets (CPU only).
usage: gemma_snippets.py sample            -> notes/sample_latents.txt (50 random latents: top 10 + 5 random activations)
       gemma_snippets.py latents 123 456   -> print snippets for chosen latents
       gemma_snippets.py search word ...   -> latents whose top-20 max tokens most often are these words
"""
import sys, collections
import torch
from transformers import AutoTokenizer

R = "/home/amit/.cache/vidyalai-scratch/p07"
E = torch.load(f"{R}/runs/gemma_explore.pt")
D = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gemma-tokens.pt")
tok = AutoTokenizer.from_pretrained("/home/amit/.cache/vidyalai-scratch/interp/models/gemma-2-2b-bf16")
X, T = D["explore"], E["T"]


def snip(gidx, val, before=12, after=4):
    d, p = divmod(int(gidx), T)
    ids = X[d].tolist()
    left = tok.decode(ids[max(1, p - before):p])
    cur = tok.decode([ids[p]])
    right = tok.decode(ids[p + 1:p + 1 + after])
    return f"{val:6.1f} | {left}[[{cur}]]{right}".replace("\n", "\\n")


def show(lat, n_top=10, rand=None, out=print):
    c = int(E["count"][lat])
    out(f"=== latent {lat}: fires on {c} of {E['n_tokens']} tokens ({100 * c / E['n_tokens']:.3f}%)")
    for v, i in zip(E["top_v"][lat][:n_top].tolist(), E["top_i"][lat][:n_top].tolist()):
        if v > 0:
            out("  top  " + snip(i, v))
    if rand is not None:
        for v, i in rand:
            out("  rand " + snip(i, v))


if sys.argv[1] == "sample":
    g = torch.Generator().manual_seed(11)
    sa = E["sample_acts"]
    with open(f"{R}/notes/sample_latents.txt", "w") as fh:
        for slot, lat in enumerate(E["sample"].tolist()):
            rows = sa[sa[:, 0] == slot]
            rand = []
            if len(rows):
                pick = torch.randperm(len(rows), generator=g)[:5]
                rand = [(float(rows[j, 2]), int(rows[j, 1])) for j in pick]
            show(lat, rand=rand, out=lambda s: fh.write(s + "\n"))
            fh.write("\n")
    print("wrote sample")
elif sys.argv[1] == "latents":
    for a in sys.argv[2:]:
        show(int(a), n_top=int(20))
elif sys.argv[1] == "search":
    words = set(sys.argv[2:])
    hits = collections.Counter()
    ti = E["top_i"]
    for lat in range(ti.shape[0]):
        for i in ti[lat].tolist():
            if i < 0:
                continue
            d, p = divmod(i, T)
            if tok.decode([int(X[d, p])]).strip().lower() in words:
                hits[lat] += 1
    for lat, c in hits.most_common(15):
        print(lat, c, int(E["count"][lat]))
