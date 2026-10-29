"""Cut WikiText-103 (test + validation, CC BY-SA 4.0) into 512-token windows that stay
inside one article, for each tokenizer. Saves int32 arrays [n_windows, 512] (no BOS)."""
import re, glob, numpy as np, pandas as pd
from transformers import AutoTokenizer
root = glob.glob("/home/amit/.cache/vidyalai-scratch/hf/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/")[0]
lines = []
for split in ["test", "validation"]:
    lines += list(pd.read_parquet(root + f"{split}-00000-of-00001.parquet")["text"])
arts, cur = [], []
for ln in lines:
    if re.match(r"^ = [^=].* = \n$", ln):
        if cur: arts.append("".join(cur))
        cur = [ln]
    else:
        cur.append(ln)
arts.append("".join(cur))
print("articles", len(arts))
W = 512
for name, hf in [("gpt2", "openai-community/gpt2"), ("pythia", "EleutherAI/pythia-160m"), ("qwen", "Qwen/Qwen2.5-0.5B")]:
    tok = AutoTokenizer.from_pretrained(hf)
    wins, aid = [], []
    for ai, a in enumerate(arts):
        ids = tok(a)["input_ids"]
        for s in range(0, len(ids) - W + 1, W):
            wins.append(ids[s:s + W]); aid.append(ai)
    arr = np.array(wins, dtype=np.int32)
    np.save(f"/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_{name}.npy", arr)
    np.save(f"/home/amit/.cache/vidyalai-scratch/p04/runs/wiki_{name}_article.npy", np.array(aid, dtype=np.int32))
    print(name, arr.shape, "tokens", arr.size)
