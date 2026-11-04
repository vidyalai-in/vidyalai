"""Rebuild the shared MMLU subset used across the Measuring LLMs series (evals/data/mmlu-1140.json):
20 test items from each of MMLU's 57 subjects (cais/mmlu, MIT license), drawn with seed 0."""
import json
import random

from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all", split="test")
by_subject = {}
for i, r in enumerate(ds):
    by_subject.setdefault(r["subject"], []).append(i)
rng = random.Random(0)
pick = []
for s in sorted(by_subject):
    pick += sorted(rng.sample(by_subject[s], 20))
items = [{"id": f"mmlu-test-{i}", "row": i, "subject": ds[i]["subject"], "question": ds[i]["question"],
          "choices": ds[i]["choices"], "answer": ds[i]["answer"]} for i in pick]
json.dump(items, open("mmlu-1140.json", "w"))
print(len(items), "items")
