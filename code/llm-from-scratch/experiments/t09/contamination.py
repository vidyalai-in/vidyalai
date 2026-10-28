"""13-gram overlap between benchmark items and the 1.76B tokens the 88M model trained on (CPU only).

An item is flagged if any 13-token sequence of its text (question + correct answer, in our tokens)
appears anywhere in the training token stream, the GPT-3 paper's rule with N = 13.
"""
import json, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import numpy as np
from data import ShardLoader
from evals import load_task, wikitext_docs
from tokenizer import load_production

N, P = 13, np.uint64(0x9E3779B97F4A7C15)
TRAIN_TOKENS = 3360 * 32 * 16 * 1024          # what Part 7's run read: 1,761,607,680


def ngram_hashes(t):
    t = np.asarray(t, dtype=np.uint64)
    if len(t) < N:
        return np.zeros(0, np.uint64)
    h = np.zeros(len(t) - N + 1, dtype=np.uint64)
    for j in range(N):                          # polynomial hash, wraps mod 2^64
        h = h * P + t[j:len(t) - N + 1 + j]
    return h


tok = load_production(); tok.encode_special_tokens = True
sets = {}
for name in ["hellaswag", "arc_easy", "piqa"]:
    items = load_task(name)
    texts = [it["query"] + " " + it["choices"][it["gold"]] for it in items]
    sets[name] = [ngram_hashes(tok.encode(t).ids) for t in texts]
sets["wikitext"] = [ngram_hashes(tok.encode(t).ids) for t in wikitext_docs()]
allh = np.unique(np.concatenate([h for v in sets.values() for h in v]))
print("benchmark 13-grams:", len(allh), flush=True)

train = ShardLoader("train")
found = np.zeros(len(allh), dtype=bool)
t0, done = time.time(), 0
for s in train.shards:
    n = min(len(s), TRAIN_TOKENS - done)
    if n <= 0:
        break
    for a in range(0, n, 25_000_000):            # chunks overlap by N - 1 tokens
        h = ngram_hashes(s[a:min(a + 25_000_000 + N - 1, n)])
        pos = np.searchsorted(allh, h)
        pos[pos == len(allh)] = 0
        found[pos[allh[pos] == h]] = True
    done += n
    print(f"{done:,} tokens, {found.sum():,} benchmark 13-grams seen, {time.time() - t0:.0f} s", flush=True)

hit = set(allh[found].tolist())
out = {}
for name, hs in sets.items():
    flags = [bool(len(h) and any(x in hit for x in h.tolist())) for h in hs]
    frac = [float(np.mean([x in hit for x in h.tolist()])) if len(h) else 0.0 for h in hs]
    out[name] = {"flagged": flags, "share_of_ngrams": frac}
    print(name, "items", len(flags), "flagged", sum(flags), f"{np.mean(flags):.1%}",
          "13-grams seen", f"{sum(f * len(h) for f, h in zip(frac, hs)) / sum(len(h) for h in hs):.2%}")
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/t09/post/contamination.json", "w"))
