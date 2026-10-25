# Train HF byte-level BPE at one vocab size on the full tokenizer sample; log time and peak RSS.
import sys, time, json, resource
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import train_tokenizer as tt
V = int(sys.argv[1])
out = "/home/amit/.cache/vidyalai-scratch/fromscratch/tokenizer-32k.json" if V == 32768 else f"/home/amit/.cache/vidyalai-scratch/t02/tok/tokenizer-{V}.json"
t0 = time.perf_counter()
tok = tt.train(V, out=out)
dt = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print("RESULT", json.dumps({"vocab": V, "train_s": round(dt, 1), "peak_rss_GB": round(rss, 2), "out": out}))
