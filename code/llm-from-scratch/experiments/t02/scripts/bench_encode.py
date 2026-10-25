# Encode throughput on held-out FineWeb-Edu. Usage: bench_encode.py {py|hf|tt}
import sys, json, time, pickle, os
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
FS = "/home/amit/.cache/vidyalai-scratch/fromscratch"; T2 = "/home/amit/.cache/vidyalai-scratch/t02"
held = [json.loads(l)["text"] for l in open(f"{FS}/data/tokenizer-sample/heldout.jsonl")]
mode = sys.argv[1]
def run(name, docs, fn, reps=3):
    nb = sum(len(d.encode()) for d in docs); best = 1e9; ntok = 0
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(docs); dt = time.perf_counter() - t0; best = min(best, dt)
        ntok = sum(map(len, out))
    print("RESULT", json.dumps({"name": name, "threads": os.environ.get("RAYON_NUM_THREADS", "all"), "MB": round(nb / 1e6, 2), "tokens": ntok, "best_s": round(best, 3), "MB_per_s": round(nb / 1e6 / best, 2)}), flush=True)
if mode == "py":
    tok = pickle.load(open(f"{T2}/tok/bench-py-10mb-4096.pkl", "rb"))
    sub = held[:500]
    def enc(docs):
        tok.cache = {}                     # cold cache each run
        return [tok.encode(d) for d in docs]
    run("pure-python-4k", sub, enc, reps=1)
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(f"{T2}/tok/bench-hf-10mb-4096.json")
    print("agree: py vs hf 4k token counts on 500 docs:", sum(map(len, enc(sub))), sum(len(e.ids) for e in hf.encode_batch(sub, add_special_tokens=False)))
    print("roundtrip py:", sum(tok.decode(tok.encode(d)) != d for d in sub), "failures of", len(sub))
elif mode == "hf":
    from tokenizer import load_production
    tok = load_production()
    run("hf-32k-encode_batch", held, lambda ds: [e.ids for e in tok.encode_batch(ds, add_special_tokens=False)])
else:
    import tiktoken
    e = tiktoken.get_encoding("cl100k_base")
    run("tiktoken-cl100k-batch", held, lambda ds: e.encode_ordinary_batch(ds, num_threads=int(os.environ.get("RAYON_NUM_THREADS", "16"))))
