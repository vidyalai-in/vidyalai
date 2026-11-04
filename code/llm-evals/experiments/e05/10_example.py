"""Worked example for the article: token log-probabilities of one item seen 16 times, before and
after training, and of one never-seen item, with every detector score (CPU, reads eval JSONs)."""
import json, sys, zlib, math
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import contamination as C

RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
items = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e01-arc-challenge.json"))
G = json.load(open(f"{RUNS}/groups.json"))
B0, B1 = json.load(open(f"{RUNS}/eval-base.json")), json.load(open(f"{RUNS}/eval-contam.json"))
pos = {i: k for k, i in enumerate(B0["ids"])}
pick = sys.argv[1:] or [str(G["g16"]["items"][0]), str(G["g0"]["items"][0])]
for s in pick:
    i = int(s)
    t = f"Question: {items[i]['question']}\nAnswer: {items[i]['choices'][items[i]['answer']]}"
    ids = tok(t, add_special_tokens=False).input_ids
    g = [k for k, v in G.items() if k != "_meta" and i in v["items"]][0]
    print(f"item {i} group {g} tokens {len(ids)}\n{t}")
    for tag, E in (("before", B0), ("after", B1)):
        lp = np.array(E["verbatim_lp"][pos[i]])
        d = E["verbatim"][pos[i]]
        low = np.sort(lp)[: max(1, math.ceil(len(lp) * 0.2))]
        print(f"  {tag}: mean log p {lp.mean():.3f}  lowest-20% tokens {np.round(low, 2).tolist()}  mink {d['mink']:.3f}"
              f"  minkpp {d['minkpp']:.3f}  zlib {d['zlib']:.4f}  total bits {-lp.sum() / math.log(2):.1f}"
              f"  zlib bits {len(zlib.compress(t.encode())) * 8}")
    print("  tokens + after log p:", [(tok.decode([a]), round(float(b), 2)) for a, b in zip(ids[1:], B1["verbatim_lp"][pos[i]])][:40])
