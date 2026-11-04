"""Download lmarena-ai/arena-human-preference-140k (CC-BY-4.0) one shard at a time, keep only
model names, outcome, session/order, timestamp, language/category flags and the style counts
(conv_metadata). No conversation text is kept. Each shard is deleted after extraction."""
import json, os, sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "lmarena-ai/arena-human-preference-140k"
DL = "/home/amit/.cache/vidyalai-scratch/e07/dl"
OUT = "/home/amit/.cache/vidyalai-scratch/e07/data/votes-140k.jsonl"
COLS = ["id", "model_a", "model_b", "winner", "evaluation_session_id", "evaluation_order",
        "conv_metadata", "category_tag", "language", "is_code", "timestamp"]

with open(OUT, "w") as f:
    for i in range(7):
        name = f"data/train-0000{i}-of-00007.parquet"
        p = hf_hub_download(REPO, name, repo_type="dataset", local_dir=DL)
        schema = pq.read_schema(p)
        cols = [c for c in COLS if c in schema.names]
        if i == 0:
            print("columns kept:", cols, "missing:", [c for c in COLS if c not in schema.names])
        t = pq.read_table(p, columns=cols).to_pylist()
        for r in t:
            r["timestamp"] = str(r.get("timestamp"))
            f.write(json.dumps(r, default=str) + "\n")
        print(name, len(t), flush=True)
        del t
        os.remove(p)
