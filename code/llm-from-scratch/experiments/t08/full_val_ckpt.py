"""full_val_ckpt.py <run>: whole-val-shard loss from a run's final ckpt.pt (runs without --keep-ckpts)."""
import json, sys
from snapshots import load, val_loss
from train import CONFIGS
D = "/home/amit/.cache/vidyalai-scratch/t08"
run = sys.argv[1]
start = json.loads(open(f"{D}/runs/{run}/log.jsonl").readline())
model = load(f"{D}/runs/{run}/ckpt.pt", cfg=CONFIGS[start["args"]["model"]])
sub, _ = val_loss(model, n_batches=40); full, n = val_loss(model)
rec = {"run": run, "snapshot": "ckpt.pt", "subset40": sub, "whole": full, "tokens": n}
print(json.dumps(rec)); open(f"{D}/post/full_val.jsonl", "a").write(json.dumps(rec) + "\n")
