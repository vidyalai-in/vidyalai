# Part 7: loss on the whole validation shard (and on train.py's 40-micro-batch subset) for given snapshots.
# usage (from fromscratch/): python full_val.py <snapshot.pt> [...]
import sys, time
import torch
from snapshots import load, val_loss
for path in sys.argv[1:]:
    model = load(path)
    t0 = time.time(); sub, nsub = val_loss(model, 40); t1 = time.time()
    full, nfull = val_loss(model); t2 = time.time()
    print(f"{path.split('/')[-1]}: subset {sub:.4f} ({nsub:,} tokens, {t1-t0:.1f} s)  full {full:.4f} "
          f"({nfull:,} tokens, {t2-t1:.1f} s, {nfull/(t2-t1):,.0f} tok/s)", flush=True)
    del model; torch.cuda.empty_cache()
