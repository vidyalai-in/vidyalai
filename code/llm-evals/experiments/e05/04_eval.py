"""Score one model on the 1,170 grouped ARC-Challenge items (GPU; run under the lock).
  python 04_eval.py <tag> <model path or HF id>
Writes runs/eval-<tag>.json: per item cloze log-probs (sum and option bytes), detector scores
on the verbatim text and on the paraphrase text, and the shard (exchangeability) test per group.
"""
import json, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import contamination as C

RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
tag, path = sys.argv[1], sys.argv[2]
n_perm = int(sys.argv[3]) if len(sys.argv) > 3 else 100
t0 = time.time()
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
items = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e01-arc-challenge.json"))
groups = json.load(open(f"{RUNS}/groups.json"))
para = {int(json.loads(l)["id"][1:]): json.loads(l)["text"].strip() for l in open(f"{RUNS}/paraphrase-clean.jsonl")}
ids = sorted(i for g, v in groups.items() if g != "_meta" for i in v["items"])


def text(i, q=None):
    return f"Question: {q or items[i]['question']}\nAnswer: {items[i]['choices'][items[i]['answer']]}"


S, B = C.cloze_scores(model, tok, [items[i] for i in ids])
out = dict(tag=tag, path=path, ids=ids, S=S.tolist(), B=B.tolist())
for kind, texts in (("verbatim", [text(i) for i in ids]), ("para", [text(i, para[i]) for i in ids])):
    st = C.token_stats(model, tok, texts, batch_size=32, add_bos=False)
    out[kind] = [C.detectors(s, t) for s, t in zip(st, texts)]
    out[kind + "_lp"] = [s["lp"].tolist() for s in st]  # kept for the reference-model ratio
print("items done", round(time.time() - t0), "s", flush=True)
out["shard"] = {}
for g, v in groups.items():
    if g == "_meta":
        continue
    pages = [[text(i) for i in pg] for pg in v["pages"]]
    out["shard"][g] = C.shard_test(model, tok, pages, n_perm=n_perm, seed=0, batch_size=16)
    print(g, "shard p", out["shard"][g]["p"], round(time.time() - t0), "s", flush=True)
json.dump(out, open(f"{RUNS}/eval-{tag}.json", "w"))
print("done", round(time.time() - t0), "s")
