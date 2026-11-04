"""Pick 100 English, single-turn, non-code user prompts from one shard of
lmarena-ai/arena-human-preference-140k (CC-BY-4.0 prompts) for the mini-arena. Text stays in the
scratch folder; it is never committed or quoted."""
import json, os, random
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
DL = "/home/amit/.cache/vidyalai-scratch/e07/dl"
p = hf_hub_download("lmarena-ai/arena-human-preference-140k", "data/train-00003-of-00007.parquet", repo_type="dataset", local_dir=DL)
t = pq.read_table(p, columns=["id", "conversation_a", "language", "is_code", "conv_metadata", "category_tag"]).to_pylist()
os.remove(p)
cand = []
for r in t:
    m = r["conv_metadata"]
    if r["language"] != "en" or r["is_code"] or m["turns"] != 1 or not (8 <= m["sum_user_tokens"] <= 150):
        continue
    u = r["conversation_a"][0]
    if u["role"] != "user": continue
    parts = [c for c in u["content"] if c["type"] == "text"]
    if len(parts) != len(u["content"]) or not parts: continue
    cand.append({"id": r["id"], "prompt": parts[0]["text"], "user_tokens": m["sum_user_tokens"],
                 "n_criteria": sum(bool(v) for v in r["category_tag"]["criteria_v0.1"].values())})
random.Random(0).shuffle(cand)
print("candidates", len(cand))
with open("/home/amit/.cache/vidyalai-scratch/e07/data/e07-arena-prompts.jsonl", "w") as f:
    for c in cand[:400]:
        f.write(json.dumps(c) + "\n")
