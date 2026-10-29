# The prompts for sequence-level distillation: the first exchange of Part 10's first 26,000 training
# conversations and of its 2,000 validation conversations, recovered by re-running Part 10's filters
# in order and checked token for token against Part 10's arrays.
import sys, json, glob, hashlib
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sft import chat_tokenizer, fit, DATA
tok = chat_tokenizer()
drop = ("self-oss-instruct", "longalign")
first_user = lambda r: next(m["content"].strip() for m in r["messages"] if m["role"] == "user")
out, val_prompts = {}, set()
for split, pattern, limit in (("val", "test-*", 2000), ("train", "train-*", 26000)):
    ref_ids, ref_off = np.load(f"{DATA}/{split}_ids.npy"), np.load(f"{DATA}/{split}_offsets.npy")
    seen, kept = set(), []
    for f in sorted(glob.glob(f"{DATA}/data/{pattern}.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000):
            for r in b.to_pylist():
                if len(kept) == limit: break
                key = hashlib.md5(json.dumps(r["messages"], sort_keys=True).encode()).digest()
                if r["source"] in drop or key in seen: continue
                if split == "train" and first_user(r) in val_prompts: continue
                bodies = [tok.encode(m["content"].strip()).ids for m in r["messages"]]
                x, y = fit(r["messages"], bodies, 1024)
                if x is None: continue
                n = len(kept)
                assert x == ref_ids[ref_off[n]:ref_off[n + 1]].tolist(), (split, n)   # same conversation as Part 10
                seen.add(key)
                if split == "val": val_prompts.add(first_user(r))
                msgs = r["messages"]
                a = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
                kept.append({"i": n, "source": r["source"], "prompt": msgs[:a], "reply": msgs[a]["content"]})
            if len(kept) == limit: break
        if len(kept) == limit: break
    out[split] = kept
    print(split, len(kept), "verified against Part 10's token arrays")
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t13/data/prompts.json', 'w'))
import collections
for s in out:
    print(s, collections.Counter(len(k["prompt"]) for k in out[s]), collections.Counter(k["prompt"][0]["role"] for k in out[s]))
