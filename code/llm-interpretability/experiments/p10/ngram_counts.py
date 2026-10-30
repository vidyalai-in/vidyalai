# Unigram and bigram counts over the first 100M tokens of the From Scratch training stream
# (train_000.bin, the first shard ShardLoader reads: steps 0 to ~190 of the 88M run).
# Output: runs/ngram.npz with unigram counts (V,) and the bigram table in CSR form
# (crow[V+1], col, count), all from integer counts; no smoothing.
import sys, time
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from data import read_shard

V = 32768
OUT = "/home/amit/.cache/vidyalai-scratch/p10/runs/ngram.npz"
t0 = time.time()
toks = read_shard("/home/amit/.cache/vidyalai-scratch/fromscratch/data/shards/train_000.bin")
N = len(toks)
print("tokens", N, flush=True)
uni = np.bincount(np.asarray(toks), minlength=V).astype(np.int64)
CH = 10_000_000
keys_all, cnt_all = [], []
for s in range(0, N - 1, CH):
    e = min(N - 1, s + CH)
    a = np.asarray(toks[s:e], dtype=np.int64); b = np.asarray(toks[s + 1:e + 1], dtype=np.int64)
    k, c = np.unique(a * V + b, return_counts=True)
    keys_all.append(k); cnt_all.append(c.astype(np.int64))
    print(s, len(k), f"{time.time()-t0:.0f}s", flush=True)
    del a, b
# merge chunk tables pairwise to keep memory small
while len(keys_all) > 1:
    nk, nc = [], []
    for i in range(0, len(keys_all), 2):
        if i + 1 == len(keys_all):
            nk.append(keys_all[i]); nc.append(cnt_all[i]); continue
        k = np.concatenate([keys_all[i], keys_all[i + 1]]); c = np.concatenate([cnt_all[i], cnt_all[i + 1]])
        o = np.argsort(k, kind="stable"); k, c = k[o], c[o]
        u, idx = np.unique(k, return_index=True)
        nk.append(u); nc.append(np.add.reduceat(c, idx))
    keys_all, cnt_all = nk, nc
    print("merged", [len(k) for k in keys_all], f"{time.time()-t0:.0f}s", flush=True)
keys, cnt = keys_all[0], cnt_all[0]
row = (keys // V).astype(np.int32); col = (keys % V).astype(np.int32)
crow = np.zeros(V + 1, dtype=np.int64)
np.add.at(crow, row + 1, 1)
crow = np.cumsum(crow)
np.savez(OUT, uni=uni, crow=crow, col=col, cnt=cnt.astype(np.int32), n_tokens=N)
print("bigrams", len(keys), "unique; seen token types", int((uni > 0).sum()), f"{time.time()-t0:.0f}s")
