"""13-gram check of ARC-Challenge test (question + correct answer) against the 1.76B tokens
From Scratch Part 7's 88M model trained on (CPU). Same rule and hash as t09/scripts/contamination.py."""
import json, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import numpy as np
from data import ShardLoader
from tokenizer import load_production

N, P = 13, np.uint64(0x9E3779B97F4A7C15)
TRAIN_TOKENS = 3360 * 32 * 16 * 1024


def ngram_hashes(t):
    t = np.asarray(t, dtype=np.uint64)
    if len(t) < N:
        return np.zeros(0, np.uint64)
    h = np.zeros(len(t) - N + 1, dtype=np.uint64)
    for j in range(N):
        h = h * P + t[j:len(t) - N + 1 + j]
    return h


tok = load_production(); tok.encode_special_tokens = True
items = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e01-arc-challenge.json"))
hs = [ngram_hashes(tok.encode(f"Question: {it['question']}\nAnswer: {it['choices'][it['answer']]}").ids) for it in items]
allh = np.unique(np.concatenate(hs))
found = np.zeros(len(allh), dtype=bool)
train, done, t0 = ShardLoader("train"), 0, time.time()
for s in train.shards:
    n = min(len(s), TRAIN_TOKENS - done)
    if n <= 0:
        break
    for a in range(0, n, 10_000_000):
        h = ngram_hashes(s[a:min(a + 10_000_000 + N - 1, n)])
        pos = np.searchsorted(allh, h); pos[pos == len(allh)] = 0
        found[pos[allh[pos] == h]] = True
    done += n
hit = set(allh[found].tolist())
flags = [bool(len(h) and any(x in hit for x in h.tolist())) for h in hs]
frac = [float(np.mean([x in hit for x in h.tolist()])) if len(h) else 0.0 for h in hs]
out = dict(n=len(items), flagged=int(sum(flags)), half=int(sum(f >= 0.5 for f in frac)), tokens=done, seconds=round(time.time() - t0))
print(out)
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/e05/runs/ours88-arc-ngram.json", "w"))
