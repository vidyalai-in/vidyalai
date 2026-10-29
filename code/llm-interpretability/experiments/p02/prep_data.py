"""Tokenize the FineWeb-Edu subsets for each model.

train: 8,000 documents -> one EOS-separated stream cut into 512-token chunks (for lens training)
eval:  held-out documents with at least 512 tokens -> one chunk per document: EOS + first 511 tokens
"""
import json, hashlib, sys
import torch
from transformers import AutoTokenizer

D = "/home/amit/.cache/vidyalai-scratch/interp/data/p02"
OUT = "/home/amit/.cache/vidyalai-scratch/p02/runs"
T = 512
MODELS = {"gpt2": "openai-community/gpt2", "pythia-160m": "EleutherAI/pythia-160m", "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B"}

train = [json.loads(l)["text"] for l in open(f"{D}/fineweb-edu-train8000.jsonl")]
held = [json.loads(l)["text"] for l in open(f"{D}/fineweb-edu-heldout600.jsonl")]
h = lambda t: hashlib.md5(t.encode()).hexdigest()
overlap = len({h(t) for t in train} & {h(t) for t in held})
print("docs", len(train), len(held), "exact-duplicate docs across splits:", overlap)
hs = {h(t) for t in held}
train = [t for t in train if h(t) not in hs]          # drop the duplicates from the training side
print("train docs after removing duplicates", len(train))
toks = {k: AutoTokenizer.from_pretrained(v) for k, v in MODELS.items()}
keep = [i for i, t in enumerate(held) if all(len(tk(t)["input_ids"]) >= T - 1 for tk in toks.values())]
print("held-out docs with >= 511 tokens under all three tokenizers:", len(keep))

for key, hf in MODELS.items():
    tk = toks[key]
    eos = tk.eos_token_id
    stream = []
    for t in train:
        stream += [eos] + tk(t)["input_ids"]
    n = len(stream) // T
    tr = torch.tensor(stream[: n * T]).view(n, T)
    ev = []
    for i in keep:
        ids = tk(held[i])["input_ids"]
        ev.append([eos] + ids[: T - 1])
    ev = torch.tensor(ev)
    torch.save({"train": tr, "eval": ev, "eos": eos, "eval_doc_index": keep}, f"{OUT}/tokens-{key}.pt")
    print(key, "eos", eos, "train chunks", tuple(tr.shape), "train tokens", tr.numel(), "eval chunks", tuple(ev.shape))
