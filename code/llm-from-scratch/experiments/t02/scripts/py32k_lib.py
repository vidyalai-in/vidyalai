import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from tokenizer import python_from_production, load_production, _gpt2_byte_chars
FS = "/home/amit/.cache/vidyalai-scratch/fromscratch"
py, hf = python_from_production(), load_production()
u2b = _gpt2_byte_chars(); tb = lambda s: bytes(u2b[c] for c in s)
held = [json.loads(l)["text"] for l in open(f"{FS}/data/tokenizer-sample/heldout.jsonl")]
nb = sum(len(d.encode()) for d in held)
for rep in range(2):
    py.cache = {}
    t0 = time.perf_counter(); pyids = [py.encode(d) for d in held]; dpy = time.perf_counter() - t0
    t0 = time.perf_counter(); hfids = [hf.encode(d, add_special_tokens=False).ids for d in held]; dhf = time.perf_counter() - t0
    print(f"py {nb/1e6/dpy:.2f} MB/s, hf loop {nb/1e6/dhf:.2f} MB/s")
same = sum([py.vocab[i] for i in a] == [hf.id_to_token(i) in ("<|endoftext|>",) and b"" or tb(hf.id_to_token(i)) for i in b] for a, b in zip(pyids, hfids))
print("identical", same, len(held))
