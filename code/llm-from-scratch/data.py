"""The pretraining data pipeline (Build an LLM From Scratch, Part 3).

    python data.py download   # every 4th row group of FineWeb-Edu sample-10BT -> data/fineweb-edu/
    python data.py minhash    # MinHash signatures (5-word shingles, 112 hashes), 8 processes
    python data.py dedup      # LSH (14 bands x 8 rows) + union-find -> one copy per cluster
    python data.py overlap    # documents Part 2's tokenizer sample contains: kept out of validation
    python data.py tokenize   # the 32K tokenizer, <|endoftext|> (id 0) after every document
    python data.py pack       # filter, shuffle, split off validation, write 100M-token shards

Every stage writes its outputs per part and skips parts already done, so a crash resumes.
Training reads the shards with ShardLoader: deterministic (x, y) micro-batches from any step.
"""
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW_DIR = os.path.join(DATA, "fineweb-edu")      # sampled parquet parts
WORK_DIR = os.path.join(DATA, "work")            # signatures, tokens, masks
SHARD_DIR = os.path.join(DATA, "shards")         # what training reads
REPO, N_FILES, STRIDE = "HuggingFaceFW/fineweb-edu", 14, 4
COLUMNS = ["text", "id", "dump", "url", "token_count", "score", "int_score", "language_score"]
WORKERS = 8
EOT = 0                                          # <|endoftext|>
MAGIC, VERSION, HEADER_INTS = 20261024, 1, 256   # 1 KB header, the same layout as llm.c's .bin files
SHARD_TOKENS = 100_000_000
MIN_TOKENS = 64                                  # extra filter: drop documents shorter than this
VAL_TOKENS = 15_000_000
SEED = 1337


def parts():
    return sorted(glob.glob(os.path.join(RAW_DIR, "part-*.parquet")))


def part_name(path):
    return os.path.basename(path).removesuffix(".parquet")


# ---------------------------------------------------------------- download

def _read_row_groups(src, groups, local):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(src if local else _open_remote(src))
    return [pf.read_row_group(g, columns=COLUMNS) for g in groups]


def download(indices=None, local_dir=None):
    """Keep every STRIDE-th row group (1,000 documents each) of each sample-10BT file.

    Each file holds only 6 to 12 of the 95 crawls, so whole files would skew the sample. Uses a full local
    copy of a file if one is in `local_dir`, otherwise HTTP range reads of just those row groups.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(RAW_DIR, exist_ok=True)
    for i in indices if indices is not None else range(N_FILES):
        out = os.path.join(RAW_DIR, f"part-{i:02d}.parquet")
        if os.path.exists(out):
            continue
        name = f"sample/10BT/{i:03d}_00000.parquet"
        local = local_dir and os.path.exists(os.path.join(local_dir, name))
        src = os.path.join(local_dir, name) if local else f"datasets/{REPO}/{name}"
        t0 = time.time()
        n_groups = pq.ParquetFile(src if local else _open_remote(src)).metadata.num_row_groups
        groups = [g for g in range(n_groups) if g % STRIDE == 0]
        chunks = [groups[k::WORKERS] for k in range(WORKERS)]
        with ThreadPoolExecutor(WORKERS) as pool:
            tables = sum(pool.map(lambda c: _read_row_groups(src, c, local), chunks), [])
        order = np.argsort(sum(chunks, []))                  # back to file order
        table = pa.concat_tables([tables[k] for k in order])
        pq.write_table(table, out + ".tmp", compression="zstd", row_group_size=1000)
        os.replace(out + ".tmp", out)                        # a finished part is the checkpoint
        print(f"{out}: {len(groups)} row groups, {table.num_rows} docs, "
              f"{'local' if local else 'remote'}, {time.time() - t0:.0f} s", flush=True)


def _open_remote(src):
    from huggingface_hub import HfFileSystem
    return HfFileSystem().open(src, block_size=1 << 20)


def read_column(path, column):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=[column]).column(column)


def iter_texts(path, batch=1000):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=batch, columns=["text"]):
        yield b.column(0).to_pylist()


# ---------------------------------------------------------------- MinHash

N_GRAM, BANDS, ROWS = 5, 14, 8               # FineWeb's settings: 112 hashes, 14 bands of 8
N_HASH = BANDS * ROWS
SEEDS = np.random.default_rng(42).integers(0, 2**63, N_HASH, dtype=np.uint64)  # one per hash function
WORD = re.compile(r"\w+")


def mix64(x):
    """MurmurHash3's 64-bit finalizer: scrambles every input bit into every output bit."""
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xFF51AFD7ED558CCD)                # uint64 arithmetic wraps around mod 2^64
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xC4CEB9FE1A85EC53)
    return x ^ (x >> np.uint64(33))


def shingles(text, n=N_GRAM):
    """The set of n-word shingles of a lower-cased text, each hashed to a 64-bit integer."""
    import xxhash
    words = WORD.findall(text.lower())
    grams = {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}
    return np.fromiter((xxhash.xxh3_64_intdigest(g.encode()) for g in grams), np.uint64, len(grams))


def minhash(text):
    """112 MinHash values: for each hash function, the minimum over the document's shingles."""
    x = shingles(text)
    sig = np.full(N_HASH, np.iinfo(np.uint64).max, dtype=np.uint64)
    for s in range(0, len(x), 2048):                     # blocks keep memory flat on huge docs
        h = mix64(x[None, s:s + 2048] ^ SEEDS[:, None])  # hash function i: mix64(shingle XOR seed_i)
        sig = np.minimum(sig, h.min(axis=1))
    return (sig >> np.uint64(32)).astype(np.uint32)       # keep the top 32 bits


def match_probability(s, bands=BANDS, rows=ROWS):
    """Chance that two documents with Jaccard similarity s share all `rows` values in some band."""
    return 1 - (1 - s ** rows) ** bands


def _minhash_part(path):
    out = os.path.join(WORK_DIR, part_name(path) + ".minhash.npy")
    if not os.path.exists(out):
        sigs = np.stack([minhash(t) for batch in iter_texts(path) for t in batch])
        np.save(out + ".tmp.npy", sigs)
        os.replace(out + ".tmp.npy", out)
    return out


def minhash_all():
    os.makedirs(WORK_DIR, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as pool:
        for out in pool.map(_minhash_part, parts()):
            print(out, f"{time.time() - t0:.0f} s", flush=True)


def lsh_pairs(sigs):
    """Candidate duplicate pairs: documents whose 8 values agree in at least one of the 14 bands."""
    pairs = []
    for b in range(BANDS):
        band = np.ascontiguousarray(sigs[:, b * ROWS:(b + 1) * ROWS])
        keys = band.view(np.dtype((np.void, band.dtype.itemsize * ROWS))).ravel()
        order = np.argsort(keys, kind="stable")
        same = keys[order][1:] == keys[order][:-1]           # neighbours in sorted order collide
        pairs.append(np.stack([order[:-1][same], order[1:][same]], axis=1))
    return np.unique(np.concatenate(pairs), axis=0)


def clusters(n, pairs):
    """Union-find: documents linked by any chain of candidate pairs share one cluster id."""
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def dedup():
    """Keep one randomly chosen document per duplicate cluster, as FineWeb does."""
    sigs = np.concatenate([np.load(p) for p in sorted(glob.glob(os.path.join(WORK_DIR, "*.minhash.npy")))])
    t0 = time.time()
    pairs = lsh_pairs(sigs)
    root = clusters(len(sigs), pairs)
    rank = np.random.default_rng(SEED).permutation(len(sigs))   # random keeper per cluster
    keeper = {}
    for i in np.unique(pairs):
        r = root[i]
        if r not in keeper or rank[i] < rank[keeper[r]]:
            keeper[r] = i
    dup = np.zeros(len(sigs), dtype=bool)
    dup[np.unique(pairs)] = True
    dup[list(keeper.values())] = False
    np.save(os.path.join(WORK_DIR, "duplicate.npy"), dup)
    np.save(os.path.join(WORK_DIR, "lsh_pairs.npy"), pairs)
    print(f"{len(sigs):,} docs, {len(pairs):,} candidate pairs, {len(keeper):,} clusters, "
          f"{dup.sum():,} duplicates removed, {time.time() - t0:.0f} s")


def tokenizer_overlap():
    """Our documents that are copies or near-copies of Part 2's tokenizer sample: kept out of validation."""
    import json
    texts = [json.loads(line)["text"] for split in ("train", "heldout")
             for line in open(os.path.join(DATA, "tokenizer-sample", f"{split}.jsonl"))]
    with ProcessPoolExecutor(WORKERS) as pool:
        theirs = np.stack(list(pool.map(minhash, texts, chunksize=1000)))
    ours = np.concatenate([np.load(p) for p in sorted(glob.glob(os.path.join(WORK_DIR, "*.minhash.npy")))])
    root = clusters(len(ours) + len(theirs), lsh_pairs(np.concatenate([ours, theirs])))
    hit = np.flatnonzero(np.isin(root[:len(ours)], root[len(ours):]))
    np.save(os.path.join(WORK_DIR, "exclude_from_val.npy"), hit)
    print(f"{len(hit):,} documents match the tokenizer sample")


# ---------------------------------------------------------------- tokenize

def encoder():
    """The series tokenizer, set to treat '<|endoftext|>' in web text as plain text."""
    sys.path.insert(0, HERE)
    from tokenizer import load_production
    tok = load_production()
    tok.encode_special_tokens = True
    return tok


def encode_docs(tok, texts):
    """Token ids for a batch of documents, each followed by <|endoftext|> (id 0)."""
    return [enc.ids + [EOT] for enc in tok.encode_batch_fast(texts)]


def _tokenize_part(path):
    os.environ["RAYON_NUM_THREADS"] = "1"                # one thread per worker process
    out = os.path.join(WORK_DIR, part_name(path))
    if os.path.exists(out + ".lengths.npy"):
        return out, 0, 0
    tok, lengths, n_bytes = encoder(), [], 0
    with open(out + ".tokens.tmp", "wb") as f:
        for texts in iter_texts(path):
            n_bytes += sum(len(t.encode("utf-8")) for t in texts)
            for ids in encode_docs(tok, texts):
                f.write(np.array(ids, dtype=np.uint16).tobytes())
                lengths.append(len(ids))
    os.replace(out + ".tokens.tmp", out + ".tokens")
    np.save(out + ".lengths.npy", np.array(lengths, dtype=np.int32))
    return out, n_bytes, sum(lengths)


def tokenize_all():
    os.makedirs(WORK_DIR, exist_ok=True)
    t0, total_b, total_t = time.time(), 0, 0
    with ProcessPoolExecutor(WORKERS) as pool:
        for out, nb, nt in pool.map(_tokenize_part, parts()):
            total_b, total_t = total_b + nb, total_t + nt
            dt = time.time() - t0
            print(f"{out}: {nt:,} tokens | total {total_t:,} tokens, {total_b / 1e6 / dt:.1f} MB/s, "
                  f"{total_t / dt:,.0f} tok/s", flush=True)


# ---------------------------------------------------------------- shuffle and pack

def write_shard(path, tokens):
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[:3] = MAGIC, VERSION, len(tokens)
    with open(path + ".tmp", "wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, dtype=np.uint16).tobytes())
    os.replace(path + ".tmp", path)


def read_shard(path):
    """A shard's tokens as a read-only uint16 memmap (no copy into memory)."""
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == MAGIC and header[1] == VERSION, f"{path}: not a token shard"
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(int(header[2]),))


def load_documents():
    """Every downloaded document's token span: (part file, start offset, length)."""
    files, starts, lengths = [], [], []
    for k, path in enumerate(parts()):
        n = np.load(os.path.join(WORK_DIR, part_name(path) + ".lengths.npy")).astype(np.int64)
        files.append(np.full(len(n), k))
        starts.append(np.concatenate([[0], np.cumsum(n)[:-1]]))
        lengths.append(n)
    return np.concatenate(files), np.concatenate(starts), np.concatenate(lengths)


def pack():
    """Filter, shuffle documents (seeded), split off validation and write uint16 shards."""
    file_of, start, length = load_documents()
    keep = ~np.load(os.path.join(WORK_DIR, "duplicate.npy"))
    keep &= length - 1 >= MIN_TOKENS                       # length counts the trailing EOT
    docs = np.flatnonzero(keep)
    docs = docs[np.random.default_rng(SEED).permutation(len(docs))]
    no_val = np.zeros(len(file_of), dtype=bool)
    no_val[np.load(os.path.join(WORK_DIR, "exclude_from_val.npy"))] = True   # Part 2's tokenizer saw them
    eligible = docs[~no_val[docs]]
    n_val = np.searchsorted(np.cumsum(length[eligible]), VAL_TOKENS) + 1
    val = eligible[:n_val]
    train = docs[~np.isin(docs, val)]
    tokens = [np.memmap(os.path.join(WORK_DIR, part_name(p) + ".tokens"), dtype=np.uint16, mode="r")
              for p in parts()]
    os.makedirs(SHARD_DIR, exist_ok=True)
    for split, ids in (("val", val), ("train", train)):
        buf, n, k = np.empty(SHARD_TOKENS, dtype=np.uint16), 0, 0
        for d in ids:
            piece = tokens[file_of[d]][start[d]:start[d] + length[d]]
            while len(piece):                               # documents may straddle two shards
                take = min(len(piece), SHARD_TOKENS - n)
                buf[n:n + take], n, piece = piece[:take], n + take, piece[take:]
                if n == SHARD_TOKENS:
                    write_shard(os.path.join(SHARD_DIR, f"{split}_{k:03d}.bin"), buf)
                    n, k = 0, k + 1
        if n:
            write_shard(os.path.join(SHARD_DIR, f"{split}_{k:03d}.bin"), buf[:n])
        np.save(os.path.join(SHARD_DIR, f"{split}_documents.npy"), ids)
        print(f"{split}: {len(ids):,} documents, {length[ids].sum():,} tokens", flush=True)


# ---------------------------------------------------------------- loading for training

class ShardLoader:
    """(x, y) micro-batches of B sequences of T tokens, read straight from the shards.

    The shards are one long token stream; micro-batch i is the window starting at token i*B*T,
    so batch(i) is a pure function of i: deterministic, and resumable from any step.
    """

    def __init__(self, split="train", B=16, T=1024, data_dir=SHARD_DIR):
        self.B, self.T = B, T
        self.shards = [read_shard(p) for p in sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))]
        self.ends = np.cumsum([len(s) for s in self.shards])
        self.n_tokens = int(self.ends[-1])
        self.n_batches = (self.n_tokens - 1) // (B * T)

    def window(self, start, n):
        """n tokens of the stream from `start`, stitched across a shard boundary if needed."""
        out, k = [], int(np.searchsorted(self.ends, start, side="right"))
        offset = start - (self.ends[k - 1] if k else 0)
        while n > 0:
            piece = self.shards[k][offset:offset + n]
            out.append(piece)
            n, k, offset = n - len(piece), k + 1, 0
        return np.concatenate(out) if len(out) > 1 else out[0]

    def batch(self, i):
        import torch
        BT = self.B * self.T
        buf = torch.from_numpy(self.window((i % self.n_batches) * BT, BT + 1).astype(np.int64))
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)

    def batches(self, step=0, accum=32):
        """Micro-batches from optimizer step `step` on (accum micro-batches per step)."""
        i = step * accum
        while True:
            yield self.batch(i)
            i += 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "download":
        download(local_dir=sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "minhash":
        minhash_all()
    elif cmd == "dedup":
        dedup()
    elif cmd == "tokenize":
        tokenize_all()
    elif cmd == "overlap":
        tokenizer_overlap()
    elif cmd == "pack":
        pack()
