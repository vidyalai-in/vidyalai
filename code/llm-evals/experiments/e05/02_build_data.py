"""Build the controlled-contamination training stream for SmolLM2-135M.

ARC-Challenge test (1,172 items) -> 5 random groups of 234 (seed 0):
  g0   never in training (control)
  g1   verbatim, 1 copy      g4  verbatim, 4 copies      g16 verbatim, 16 copies
  gp16 Haiku 5.5 paraphrase of the question (answer text unchanged), 16 copies
Each group, in canonical (dataset) order, is cut into 26 "pages" of 9 items, the way a benchmark
file or a quiz web page would hold them. Every copy of a page is one document, placed at a random
point in about 6.5M tokens of FineWeb-Edu documents. Documents are joined with <|endoftext|>.

Writes runs/stream.u16 (token ids), runs/groups.json, runs/ngram_filler.json.
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import contamination as C

RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
FW = "/home/amit/.cache/vidyalai-scratch/fromscratch/data/fineweb-edu/part-06.parquet"
FILLER_TOKENS = 6_500_000
PAGE = 9
GROUPS = [("g0", 0, False), ("g1", 1, False), ("g4", 4, False), ("g16", 16, False), ("gp16", 16, True)]

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
EOS = tok.eos_token_id
items = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e01-arc-challenge.json"))
para = {}
for line in open(f"{RUNS}/paraphrase-clean.jsonl"):
    r = json.loads(line)
    if r["type"] == "succeeded" and r["stop"] == "end_turn":
        para[int(r["id"][1:])] = r["text"].strip()
print("paraphrases", len(para))


def text(it, q=None):
    return f"Question: {q or it['question']}\nAnswer: {it['choices'][it['answer']]}"


rng = np.random.default_rng(0)
perm = rng.permutation(len(items))
size = len(items) // len(GROUPS)  # 234
groups, docs = {}, []
for g, (name, reps, use_para) in enumerate(GROUPS):
    idx = sorted(int(i) for i in perm[g * size:(g + 1) * size])  # canonical order
    groups[name] = dict(reps=reps, paraphrase=use_para, items=idx)
    pages = [idx[k:k + PAGE] for k in range(0, len(idx), PAGE)]
    groups[name]["pages"] = pages
    for pg in pages:
        body = "\n\n".join(text(items[i], para[i] if use_para else None) for i in pg)
        docs += [(name, body)] * reps
assert all(i in para for i in groups["gp16"]["items"])

# FineWeb-Edu filler, read from the end of one sample file, one row group at a time
f = pq.ParquetFile(FW)
filler, n_tok = [], 0
for rg in range(f.metadata.num_row_groups - 1, -1, -1):
    for t in f.read_row_group(rg, columns=["text"]).column("text").to_pylist():
        ids = np.array(tok(t, add_special_tokens=False).input_ids, np.uint16)
        filler.append(ids)
        n_tok += len(ids) + 1
    if n_tok >= FILLER_TOKENS:
        break
print("filler docs", len(filler), "tokens", n_tok)

# 13-gram check of every item (question + answer, eval format) against the filler alone
grams = np.unique(np.concatenate([C.hash_ngrams(ids, 13) for ids in filler]))
hits = {i: C.ngram_hits(tok(text(it), add_special_tokens=False).input_ids, grams) for i, it in enumerate(items)}
json.dump(dict(flagged=[i for i, h in hits.items() if h > 0], max=max(hits.values()), n=len(items)),
          open(f"{RUNS}/ngram_filler.json", "w"))
print("items with any 13-gram in filler:", sum(h > 0 for h in hits.values()))
del grams

contam = [np.array(tok(b, add_special_tokens=False).input_ids, np.uint16) for _, b in docs]
all_docs = [("fw", ids) for ids in filler] + [(n, ids) for (n, _), ids in zip(docs, contam)]
order = np.random.default_rng(1).permutation(len(all_docs))
stream, pos, at = [], {}, 0
eos = np.array([EOS], np.uint16)
for k in order:
    name, ids = all_docs[k]
    if name != "fw":
        pos.setdefault(name, []).append(at)
    stream += [ids, eos]
    at += len(ids) + 1
arr = np.concatenate(stream)
arr.tofile(f"{RUNS}/stream.u16")
c_tok = sum(len(x) + 1 for x in contam)
for name in pos:
    groups[name]["positions"] = pos[name]
groups["_meta"] = dict(total_tokens=len(arr), filler_tokens=n_tok, contamination_tokens=c_tok,
                       share=c_tok / len(arr), page=PAGE, size=size, seed=0)
json.dump(groups, open(f"{RUNS}/groups.json", "w"))
print(groups["_meta"])
