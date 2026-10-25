import sys, json, glob, time
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from tokenizer import load_production, SPECIAL_TOKENS
FS = "/home/amit/.cache/vidyalai-scratch/fromscratch"
tok = load_production()
V = tok.get_vocab_size()
# 1. Round trips
held = [json.loads(l)["text"] for l in open(f"{FS}/data/tokenizer-sample/heldout.jsonl")]
code = [open(f, encoding="utf-8").read() for f in sorted(glob.glob("/usr/lib/python3.12/*.py"))]
odd = ["नमस्ते दुनिया", "🙂👍🏽 emoji", "tab\tand\r\nCRLF", "\x00null byte", "  leading and trailing  ", "Zwi​schen", "é vs é"]
for name, docs in [("heldout", held), ("code", code), ("odd", odd)]:
    enc = tok.encode_batch(docs, add_special_tokens=False)
    bad = sum(tok.decode(e.ids) != d for e, d in zip(enc, docs))
    print("roundtrip", name, len(docs), "docs, failures:", bad)
# 2. How often each token appears when encoding the training sample
t0 = time.perf_counter()
counts = np.zeros(V, dtype=np.int64); total = 0; nbytes = 0
batch = []
def flush():
    global total, batch
    for e in tok.encode_batch(batch, add_special_tokens=False):
        a = np.asarray(e.ids, dtype=np.int64); counts[:] += np.bincount(a, minlength=V); total += len(a)
    batch = []
with open(f"{FS}/data/tokenizer-sample/train.jsonl") as f:
    for line in f:
        t = json.loads(line)["text"]; nbytes += len(t.encode()); batch.append(t)
        if len(batch) == 2000: flush()
flush()
dt = time.perf_counter() - t0
np.save("/home/amit/.cache/vidyalai-scratch/t02/train_counts_32k.npy", counts)
ns = len(SPECIAL_TOKENS)
reg = counts[ns:]
print("train sample", nbytes, "bytes", total, "tokens", round(nbytes / total, 3), "bytes/token", round(dt, 1), "s")
for thr in [1, 10, 100, 1000]:
    print(f"tokens seen < {thr} times:", int((reg < thr).sum()))
byte_ids = list(range(ns, ns + 256))
zero_bytes = [tok.id_to_token(i) for i in byte_ids if counts[i] == 0]
print("byte tokens never seen:", len(zero_bytes))
from tokenizers.pre_tokenizers import ByteLevel
# map GPT-2 byte chars back to byte values
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
print("never-seen bytes:", sorted(hex(u2b[c]) for c in zero_bytes))
order = np.argsort(reg)
print("rarest non-byte tokens:")
k = 0
for i in order:
    tid = i + ns
    if tid < ns + 256: continue
    b = bytes(u2b[c] for c in tok.id_to_token(tid))
    print("  ", tid, repr(b.decode("utf-8", "replace")), int(counts[tid]))
    k += 1
    if k == 25: break
# Highest-id tokens (last merges) and their counts
print("last merges:", [(tok.id_to_token(i), int(counts[i])) for i in range(V - 10, V)])
print("median count of tokens", int(np.median(reg)), "top tokens", [(tok.id_to_token(int(i)), int(counts[i])) for i in np.argsort(-counts)[:10]])
# 3. Leading spaces and case
for s in ["the", " the", "The", " The", " THE", "Paris", " Paris", " paris"]:
    print("space/case", repr(s), tok.encode(s).ids, tok.encode(s).tokens)
# 4. Decoding a piece of a character
ids = tok.encode("न").ids
print("partial", ids, [tok.decode([i]) for i in ids], repr(tok.decode(ids)))
