"""UTF-8 bytes of the whole validation shard and of train.py's 40-micro-batch subset (CPU)."""
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import numpy as np
from data import ShardLoader
from tokenizer import load_production
tok = load_production()
s = np.asarray(ShardLoader("val").shards[0])
def nbytes(a):
    # decode document by document (EOT id 0 decodes to nothing)
    cuts = np.flatnonzero(a == 0)
    tot, start = 0, 0
    for c in list(cuts) + [len(a)]:
        if c > start:
            tot += len(tok.decode(a[start:c].tolist()).encode("utf-8"))
        start = c + 1
    return tot
whole = 915 * 16 * 1024 + 1        # val_loss scores 915 micro-batches: inputs 0..N-1, targets 1..N
sub = 40 * 16 * 1024 + 1
for name, n in (("whole shard (val_loss)", whole), ("40-batch subset", sub)):
    b = nbytes(s[:n]); print(name, "tokens", n - 1, "bytes", b, "bytes/token", b / (n - 1))
print("all", len(s), nbytes(s))
