"""full_val.py <run>...: whole-val-shard loss of each run's final snapshot (and the 40-batch subset as a check)."""
import json, sys, time
from snapshots import load, val_loss
from sizes import SIZES
from train import CONFIGS
D = "/home/amit/.cache/vidyalai-scratch/t08"
for run in sys.argv[1:]:
    start = json.loads(open(f"{D}/runs/{run}/log.jsonl").readline())
    steps = start["args"]["steps"]
    model = load(f"{D}/runs/{run}/model_{steps:05d}.pt", cfg=CONFIGS[start["args"]["model"]])
    t0 = time.time()
    sub, _ = val_loss(model, n_batches=40)
    full, n_tok = val_loss(model)
    rec = {"run": run, "snapshot": f"model_{steps:05d}.pt", "subset40": sub, "whole": full, "tokens": n_tok,
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(rec), flush=True)
    open(f"{D}/post/full_val.jsonl", "a").write(json.dumps(rec) + "\n")
