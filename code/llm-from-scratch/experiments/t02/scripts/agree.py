import pickle, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from tokenizers import Tokenizer
T2 = "/home/amit/.cache/vidyalai-scratch/t02"
py = pickle.load(open(f"{T2}/tok/bench-py-10mb-4096.pkl", "rb"))
hf = Tokenizer.from_file(f"{T2}/tok/bench-hf-10mb-4096.json")
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
pyset = [py.vocab[i] for i in range(256, 256 + len(py.merges))]
hfset = [bytes(u2b[c] for c in hf.id_to_token(i)) for i in range(264, hf.get_vocab_size())]
print(len(pyset), len(hfset), "common", len(set(pyset) & set(hfset)))
same_order = sum(a == b for a, b in zip(pyset, hfset))
first_diff = next(i for i, (a, b) in enumerate(zip(pyset, hfset)) if a != b)
print("same position", same_order, "first difference at merge", first_diff + 1, pyset[first_diff], hfset[first_diff])
