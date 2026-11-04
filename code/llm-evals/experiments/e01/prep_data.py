"""Build e01's item sets: ARC-Easy test, ARC-Challenge test (all items), and a stratified
MMLU-Pro test sample (seed 0). Writes evals/data/e01-*.json with fields id/subject/question/choices/answer."""
import json, random, collections
from datasets import load_dataset

D = "/home/amit/.cache/vidyalai-scratch/evals/data"

def arc(name):
    ds = load_dataset("allenai/ai2_arc", name, split="test")
    out = []
    for r in ds:
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        out.append(dict(id=r["id"], subject=name, question=r["question"],
                        choices=r["choices"]["text"], answer=labels.index(r["answerKey"])))
    return out

for name, fn in [("ARC-Easy", "e01-arc-easy.json"), ("ARC-Challenge", "e01-arc-challenge.json")]:
    items = arc(name)
    json.dump(items, open(f"{D}/{fn}", "w"))
    print(name, len(items), collections.Counter(len(i["choices"]) for i in items))

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
by = collections.defaultdict(list)
for r in ds:
    by[r["category"]].append(r)
rng = random.Random(0)
items = []
for cat in sorted(by):
    for r in rng.sample(by[cat], 60):
        items.append(dict(id=str(r["question_id"]), subject=cat, question=r["question"],
                          choices=list(r["options"]), answer=int(r["answer_index"])))
json.dump(items, open(f"{D}/e01-mmlu-pro-840.json", "w"))
print("MMLU-Pro", len(ds), "->", len(items), collections.Counter(len(i["choices"]) for i in items))
