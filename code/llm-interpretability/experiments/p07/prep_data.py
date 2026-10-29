"""Tokenize FineWeb-Edu (sample-10BT subset already on disk from the From Scratch series) for Part 7.

gpt2 train : documents from part-12.parquet, shuffled, cut into windows of 127 tokens, each prefixed
             with <|endoftext|> (BOS) -> [n, 128] uint16. Documents that also appear in Part 2's
             held-out set are dropped.
gpt2 eval  : Part 2's 600 held-out documents, same windowing -> [n, 128] + document index per window.
gemma explore : documents from part-11.parquet, first 255 tokens of each doc with >= 255 tokens, Gemma BOS
             prefixed -> [n, 256] int32 (for top-activating examples).
gemma eval : Part 2's held-out documents, first 255 tokens -> [n, 256].
usage: prep_data.py gpt2 <max_train_tokens> | gemma <n_explore_docs>
"""
import sys, json, hashlib, random
import numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer

FS = "/home/amit/.cache/vidyalai-scratch/fromscratch/data/fineweb-edu"
P02 = "/home/amit/.cache/vidyalai-scratch/interp/data/p02/fineweb-edu-heldout600.jsonl"
OUT = "/home/amit/.cache/vidyalai-scratch/interp/data/p07"
GEMMA = "/home/amit/.cache/vidyalai-scratch/interp/models/gemma-2-2b-bf16"
md5 = lambda t: hashlib.md5(t.encode()).hexdigest()
held = [json.loads(l)["text"] for l in open(P02)]
held_h = {md5(t) for t in held}


def windows(ids, bos, w):
    n = len(ids) // w
    a = np.asarray(ids[: n * w], dtype=np.int64).reshape(n, w)
    return np.concatenate([np.full((n, 1), bos), a], 1).astype(np.uint16)


def gpt2(max_tokens):
    tk = AutoTokenizer.from_pretrained("openai-community/gpt2")
    bos = tk.eos_token_id
    pf = pq.ParquetFile(f"{FS}/part-12.parquet")
    groups = list(range(pf.metadata.num_row_groups))
    random.Random(0).shuffle(groups)
    out, n_tok, n_docs, dropped = [], 0, 0, 0
    for g in groups:
        texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
        keep = [t for t in texts if md5(t) not in held_h]
        dropped += len(texts) - len(keep)
        for ids in tk(keep)["input_ids"]:
            ws = windows(ids, bos, 127)
            out.append(ws); n_tok += ws.size; n_docs += 1
        if n_tok >= max_tokens:
            break
    arr = torch.from_numpy(np.concatenate(out).astype(np.int32)).to(torch.uint16)
    arr = arr[torch.randperm(len(arr), generator=torch.Generator().manual_seed(0))]
    ev, doc = [], []
    for d, ids in enumerate(tk(held)["input_ids"]):
        ws = windows(ids, bos, 127)
        ev.append(ws); doc += [d] * len(ws)
    torch.save({"train": arr, "eval": torch.from_numpy(np.concatenate(ev).astype(np.int32)).to(torch.uint16), "eval_doc": torch.tensor(doc),
                "bos": bos, "train_docs": n_docs, "dropped_overlap": dropped}, f"{OUT}/gpt2-tokens.pt")
    print("gpt2 train windows", tuple(arr.shape), "tokens", arr.numel(), "docs", n_docs, "dropped (in p02 held-out)", dropped)
    print("gpt2 eval windows", len(doc), "tokens", 128 * len(doc), "docs", len(set(doc)))


def gemma(n_docs):
    tk = AutoTokenizer.from_pretrained(GEMMA)
    bos = tk.bos_token_id
    pf = pq.ParquetFile(f"{FS}/part-11.parquet")
    groups = list(range(pf.metadata.num_row_groups))
    random.Random(1).shuffle(groups)
    ex, src = [], []
    for g in groups:
        tab = pf.read_row_group(g, columns=["text", "id"])
        texts, ids_ = tab.column("text").to_pylist(), tab.column("id").to_pylist()
        enc = tk(texts, add_special_tokens=False)["input_ids"]
        for t, i, ids in zip(texts, ids_, enc):
            if md5(t) in held_h or len(ids) < 255:
                continue
            ex.append([bos] + ids[:255]); src.append(i)
            if len(ex) >= n_docs:
                break
        if len(ex) >= n_docs:
            break
    ev = [[bos] + ids[:255] for ids in tk(held, add_special_tokens=False)["input_ids"] if len(ids) >= 255]
    torch.save({"explore": torch.tensor(ex, dtype=torch.int32), "explore_ids": src,
                "eval": torch.tensor(ev, dtype=torch.int32), "bos": bos}, f"{OUT}/gemma-tokens.pt")
    print("gemma explore", len(ex), "docs x 256 =", 256 * len(ex), "tokens; eval", len(ev), "docs x 256")


if __name__ == "__main__":
    {"gpt2": gpt2, "gemma": gemma}[sys.argv[1]](int(sys.argv[2]))
