"""Does the downloaded sample overlap Part 2's tokenizer sample? Exact (text hash) and near (MinHash LSH)."""
import sys, json, time
import numpy as np, pyarrow.parquet as pq, xxhash
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import data
from concurrent.futures import ProcessPoolExecutor

def hashes(path):
    return np.array([xxhash.xxh3_64_intdigest(t.encode()) for b in data.iter_texts(path) for t in b], dtype=np.uint64)

def sig_list(texts):
    return np.stack([data.minhash(t) for t in texts])

if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(8) as pool:
        ours = np.concatenate(list(pool.map(hashes, data.parts())))
        tokdocs, split = [], []
        for s in ("train", "heldout"):
            for l in open(f"{data.DATA}/tokenizer-sample/{s}.jsonl"):
                tokdocs.append(json.loads(l)["text"]); split.append(s)
        th = np.array([xxhash.xxh3_64_intdigest(t.encode()) for t in tokdocs], dtype=np.uint64)
        chunks = [tokdocs[i:i + 5000] for i in range(0, len(tokdocs), 5000)]
        tsig = np.concatenate(list(pool.map(sig_list, chunks)))
    np.save("/home/amit/.cache/vidyalai-scratch/t03/text_hash.npy", ours)
    split = np.array(split)
    exact = np.flatnonzero(np.isin(ours, th))
    print(f"exact: {len(exact)} of our {len(ours):,} docs appear verbatim in the tokenizer sample "
          f"({np.isin(th[split=='train'], ours).sum()} of its train docs, {np.isin(th[split=='heldout'], ours).sum()} of its held-out docs)")
    import glob
    sigs = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{data.WORK_DIR}/*.minhash.npy"))])
    n = len(sigs)
    pairs = data.lsh_pairs(np.concatenate([sigs, tsig]))
    root = data.clusters(n + len(tsig), pairs)
    tok_roots = set(root[n:][np.isin(np.arange(n, n + len(tsig)), pairs)].tolist())
    near = np.flatnonzero(np.isin(root[:n], list(tok_roots)) & np.isin(np.arange(n), pairs))
    print(f"near (same LSH cluster as a tokenizer-sample doc): {len(near)} of our docs")
    excl = np.union1d(exact, near)
    np.save("/home/amit/.cache/vidyalai-scratch/t03/exclude_from_val.npy", excl)
    json.dump(dict(exact=len(exact), near=len(near), excluded=len(excl), tok_docs=len(tokdocs)), open("/home/amit/.cache/vidyalai-scratch/t03/overlap.json", "w"))
    print(f"{len(excl)} docs excluded from validation, {time.time()-t0:.0f} s")
