import sys; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from collections import Counter
import regex as re
from tokenizer import PATTERN, pair_counts, merge
s = "the cat sat on the mat. the cat ate the rat."
for mode in ["regex", "raw"]:
    pieces = re.findall(PATTERN, s) if mode == "regex" else [s]
    chunks = Counter(tuple(p.encode()) for p in pieces)
    vocab = {i: bytes([i]) for i in range(256)}
    tot = lambda: sum(len(k)*n for k, n in chunks.items())
    print(mode, "pieces", pieces, "start tokens", tot(), "distinct bytes", len({b for k in chunks for b in k}))
    for new in range(256, 262):
        c = pair_counts(chunks)
        top = c.most_common(4)
        pair = max(c, key=c.get)
        vocab[new] = vocab[pair[0]] + vocab[pair[1]]
        chunks = Counter({merge(k, pair, new): n for k, n in chunks.items()})
        print(new, pair, vocab[new], c[pair], "tokens now", tot(), "top4", [(vocab[a]+b'|'+vocab[b], n) for (a,b), n in top])
