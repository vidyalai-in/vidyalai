"""CPU checks: FineWeb slice decode vs. parquet originals, bytes per token, prompt lengths."""
import sys, statistics
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import numpy as np
from evals import fineweb_val_docs, load_task, fewshot_context
from tokenizer import load_production
import data

docs = fineweb_val_docs()
tok = load_production(); tok.encode_special_tokens = True
lens = [len(tok.encode(d).ids) for d in docs]
nbytes = sum(len(d.encode()) for d in docs)
print("fineweb slice docs", flush=True); print("fineweb slice docs", len(docs), "tokens", sum(lens), "bytes", nbytes, "B/tok", nbytes / sum(lens))
print("doc tokens: median", statistics.median(lens), "over 1024:", sum(l > 1024 for l in lens), "max", max(lens))
# compare with original parquet text for first 200 val docs
file_of, start, length = data.load_documents()
val = np.load(data.os.path.join(data.SHARD_DIR, "val_documents.npy"))
import pyarrow.parquet as pq
tables = {}
same = 0
for j, d in enumerate(val[:200]):
    k = file_of[d]
    if k not in tables:
        tables[k] = (pq.ParquetFile(data.parts()[k]), np.flatnonzero(file_of == k)[0])
    pf, first = tables[k]
    r = int(d - first)
    same += pf.read_row_group(r // 1000, columns=["text"]).column(0)[r % 1000].as_py() == docs[j]
print("first 200 decoded docs identical to parquet text:", same)
# whole val shard bytes per token
stream = data.ShardLoader("val").shards[0]
print("val shard tokens", len(stream), "docs", int((stream == 0).sum()))
for name in ["hellaswag", "arc_easy", "piqa"]:
    items = load_task(name)
    train = load_task(name, "train")
    L = [len(tok.encode(it["query"] + " " + it["choices"][it["gold"]]).ids) for it in items]
    S = [len(tok.encode(it["query"] + " " + it["choices"][it["gold"]]).ids) for it in train[:2000]]
    nonascii = sum(any(len(c) != len(c.encode()) for c in it["choices"]) for it in items)
    print(name, len(items), "choices", statistics.mean(len(i["choices"]) for i in items),
          "item tokens mean", round(statistics.mean(L), 1), "max", max(L), "shot mean", round(statistics.mean(S), 1),
          "non-ascii choice items", nonascii)
