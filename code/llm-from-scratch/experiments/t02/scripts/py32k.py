# Load the production 32K merges into the pure-Python encoder; check identical output and time both.
import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from tokenizer import BPETokenizer, SPECIAL_TOKENS, load_production
FS = "/home/amit/.cache/vidyalai-scratch/fromscratch"
hf = load_production()
j = json.load(open(f"{FS}/tokenizer-32k.json"))
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
tb = lambda s: bytes(u2b[c] for c in s)
py = BPETokenizer(); byid = {py.vocab[i]: i for i in range(256)}
for k, m in enumerate(j["model"]["merges"]):
    a, b = (m.split(" ") if isinstance(m, str) else m)
    new = 256 + k
    py.merges[(byid[tb(a)], byid[tb(b)])] = new
    py.vocab[new] = tb(a) + tb(b); byid[py.vocab[new]] = new
for t in SPECIAL_TOKENS:
    py.special[t] = len(py.vocab); py.vocab[len(py.vocab)] = t.encode()
held = [json.loads(l)["text"] for l in open(f"{FS}/data/tokenizer-sample/heldout.jsonl")]
nb = sum(len(d.encode()) for d in held)
t0 = time.perf_counter(); pyids = [py.encode(d) for d in held]; dpy = time.perf_counter() - t0
t0 = time.perf_counter(); hfids = [hf.encode(d, add_special_tokens=False).ids for d in held]; dhf = time.perf_counter() - t0
same = sum([py.vocab[i] for i in a] == [tb(hf.id_to_token(i)) for i in b] for a, b in zip(pyids, hfids))
print(f"identical docs {same}/{len(held)}; py {nb/1e6/dpy:.2f} MB/s ({dpy:.1f} s), hf single-call loop {nb/1e6/dhf:.2f} MB/s ({dhf:.1f} s), tokens py {sum(map(len,pyids))} hf {sum(map(len,hfids))}")
