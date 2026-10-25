"""Tokenizer throughput on Part 2's held-out sample (26.1 MB): one thread vs rayon threads."""
import os, sys, time, json
threads = sys.argv[1]
os.environ["RAYON_NUM_THREADS"] = threads
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import data
docs = [json.loads(l)["text"] for l in open("/home/amit/.cache/vidyalai-scratch/fromscratch/data/tokenizer-sample/heldout.jsonl")]
nb = sum(len(d.encode()) for d in docs)
tok = data.encoder()
best = []
for rep in range(3):
    t0 = time.perf_counter()
    n = 0
    for i in range(0, len(docs), 1000):
        n += sum(map(len, data.encode_docs(tok, docs[i:i+1000])))
    best.append(time.perf_counter() - t0)
dt = sorted(best)[1]
print(f"threads={threads} docs={len(docs)} bytes={nb} tokens={n} median {dt:.2f}s {nb/dt/1e6:.2f} MB/s {n/dt:,.0f} tok/s bytes/token(incl EOT)={nb/n:.3f} bytes/token(excl)={nb/(n-len(docs)):.3f}")
