"""What did MinHash dedup find? Cluster sizes, cross-crawl share, tokens removed, real examples."""
import sys, json, glob, collections
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import data
W = data.WORK_DIR
sigs = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{W}/*.minhash.npy"))])
pairs = np.load(f"{W}/lsh_pairs.npy"); dup = np.load(f"{W}/duplicate.npy")
file_of, start, length = data.load_documents()
th = np.load("/home/amit/.cache/vidyalai-scratch/t03/text_hash.npy")
dumps = np.concatenate([np.array(data.read_column(p, "dump").to_pylist()) for p in data.parts()])
urls = np.concatenate([np.array(data.read_column(p, "url").to_pylist(), dtype=object) for p in data.parts()])
n = len(sigs)
root = data.clusters(n, pairs)
members = collections.defaultdict(list)
for i in np.unique(pairs): members[root[i]].append(i)
sizes = np.array([len(m) for m in members.values()])
cross = sum(len(set(dumps[m])) > 1 for m in members.values())
same_dump_only = len(members) - cross
exact_clusters = sum(len(set(th[m])) == 1 for m in members.values())
est = (sigs[pairs[:, 0]] == sigs[pairs[:, 1]]).mean(axis=1)
pair_cross = (dumps[pairs[:, 0]] != dumps[pairs[:, 1]])
pair_exact = th[pairs[:, 0]] == th[pairs[:, 1]]
rep = dict(docs=n, candidate_pairs=len(pairs), clusters=len(members), docs_in_clusters=int(sizes.sum()),
           removed_docs=int(dup.sum()), removed_frac=float(dup.mean()),
           tokens_total=int(length.sum()), removed_tokens=int(length[dup].sum()), removed_tokens_frac=float(length[dup].sum() / length.sum()),
           cluster_size_counts={int(k): int(v) for k, v in zip(*np.unique(np.minimum(sizes, 10), return_counts=True))},
           largest_clusters=sorted(sizes.tolist())[-5:], cross_dump_clusters=int(cross), same_dump_clusters=int(same_dump_only),
           exact_text_clusters=int(exact_clusters), pairs_cross_dump=int(pair_cross.sum()), pairs_exact=int(pair_exact.sum()),
           est_jaccard_pct=np.percentile(est, [1, 5, 25, 50, 75]).tolist(), pairs_est_below_0_5=int((est < 0.5).sum()))
print(json.dumps(rep, indent=1))
json.dump(rep, open("/home/amit/.cache/vidyalai-scratch/t03/dedup_report.json", "w"), indent=1)
# Largest cluster and some near-duplicate (not exact) cross-crawl pairs, for examples.
first = {k: int(np.flatnonzero(file_of == k)[0]) for k in range(len(data.parts()))}
del sigs
def text(i):
    k = int(file_of[i]); row = int(i) - first[k]
    rg = pq.ParquetFile(data.parts()[k]).read_row_group(row // 1000, columns=["text"])
    return rg.column(0)[row % 1000].as_py()
big = max(members.values(), key=len)
print("\nLARGEST CLUSTER", len(big), collections.Counter(dumps[big]).most_common(5))
for i in big[:4]: print(" ", dumps[i], urls[i], repr(text(i)[:200]))
rng = np.random.default_rng(0)
cand = np.flatnonzero(pair_cross & ~pair_exact & (est > 0.6) & (est < 0.95))
print("\nnear-duplicate cross-crawl pairs:", len(cand))
for k in rng.choice(cand, 8, replace=False):
    i, j = pairs[k]
    a, b = text(i), text(j)
    sa, sb = data.shingles(a), data.shingles(b)
    jac = len(np.intersect1d(sa, sb)) / len(np.union1d(sa, sb))
    print(f"\n--- est {est[k]:.2f} exact J {jac:.3f} | {dumps[i]} {urls[i]} | {dumps[j]} {urls[j]} | lens {len(a)} {len(b)}")
    print("A:", repr(a[:300])); print("B:", repr(b[:300]))
