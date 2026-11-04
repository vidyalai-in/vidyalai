"""n-gram overlap of every grouped item against the actual training stream (CPU).
For n = 13 (GPT-3) and n = 8 (Llama 3): the share of each item's n-grams (eval text: question and
answer, SmolLM2 tokens) found in the stream, and how many items have any hit / a majority hit.
Writes runs/ngram-stream.json."""
import json, sys
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import contamination as C

RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
items = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e01-arc-challenge.json"))
G = json.load(open(f"{RUNS}/groups.json"))
stream = np.fromfile(f"{RUNS}/stream.u16", dtype=np.uint16).astype(np.int64)
out = {}
for n in (13, 8):
    H = np.unique(np.concatenate([C.hash_ngrams(stream[s:s + 1_000_000 + n - 1], n)
                                  for s in range(0, len(stream), 1_000_000)]))
    for g, v in G.items():
        if g == "_meta":
            continue
        sh = [C.ngram_hits(tok(f"Question: {items[i]['question']}\nAnswer: {items[i]['choices'][items[i]['answer']]}",
                               add_special_tokens=False).input_ids, H, n) for i in v["items"]]
        sh = np.array(sh)
        out[f"{g}/{n}"] = dict(any=int((sh > 0).sum()), half=int((sh >= 0.5).sum()), mean=float(sh.mean()), n=len(sh))
        print(g, n, out[f"{g}/{n}"])
json.dump(out, open(f"{RUNS}/ngram-stream.json", "w"), indent=1)
