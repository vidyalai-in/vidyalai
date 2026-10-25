# Part 2's held-out FineWeb-Edu sample, tokenized with the 32K tokenizer: <|endoftext|> (id 0) before each document.
import sys, json, numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from tokenizer import load_production
tok = load_production()
tok.encode_special_tokens = True          # a literal "<|endoftext|>" in web text stays plain text
docs = [json.loads(l)["text"] for l in open("/home/amit/.cache/vidyalai-scratch/fromscratch/data/tokenizer-sample/heldout.jsonl")]
ids = []
for e in tok.encode_batch(docs, add_special_tokens=False):
    ids.append(0); ids.extend(e.ids)
ids = np.array(ids, dtype=np.uint16)
np.save("/home/amit/.cache/vidyalai-scratch/t04/heldout_ids.npy", ids)
print(len(docs), "docs", len(ids), "tokens", "max id", ids.max())
