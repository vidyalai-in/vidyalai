import json, os
from datasets import load_dataset
out = "/home/amit/.cache/vidyalai-scratch/evals/data"
he = load_dataset("openai/openai_humaneval", split="test")
print("humaneval", len(he), he.column_names)
hp = load_dataset("evalplus/humanevalplus", split="test")
print("humaneval+", len(hp), hp.column_names)
m5 = load_dataset("HuggingFaceH4/MATH-500", split="test")
print("math500", len(m5), m5.column_names)
json.dump([dict(r) for r in he], open(f"{out}/e08-humaneval.json", "w"))
json.dump([dict(r) for r in hp], open(f"{out}/e08-humanevalplus.json", "w"))
json.dump([dict(r) for r in m5], open(f"{out}/e08-math500.json", "w"))
