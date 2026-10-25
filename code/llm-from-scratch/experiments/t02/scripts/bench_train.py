# Train time on the same 10 MB slice: pure Python vs HF tokenizers. Usage: bench_train.py {py|hf} MB VOCAB
import sys, time, json, pickle, resource
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/t02/scripts")
from slice import slice_docs
mode, MB, V = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
docs, n = slice_docs("/home/amit/.cache/vidyalai-scratch/fromscratch/data/tokenizer-sample/train.jsonl", MB)
out = f"/home/amit/.cache/vidyalai-scratch/t02/tok/bench-{mode}-{MB:g}mb-{V}"
t0 = time.perf_counter()
if mode == "py":
    from tokenizer import BPETokenizer
    tok = BPETokenizer(); tok.train(docs, V)
    dt = time.perf_counter() - t0
    tok.cache = {}
    pickle.dump(tok, open(out + ".pkl", "wb"))
else:
    import train_tokenizer as tt
    tok = tt.train(V, docs=docs, out=out + ".json")
    dt = time.perf_counter() - t0
import os
print("RESULT", json.dumps({"mode": mode, "threads": os.environ.get("RAYON_NUM_THREADS", "all"), "bytes": n, "docs": len(docs), "vocab": V, "train_s": round(dt, 2),
      "peak_rss_GB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}))
