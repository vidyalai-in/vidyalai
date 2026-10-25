"""Metadata statistics of the downloaded sample: crawls, edu scores, lengths."""
import sys, json, glob, collections
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import data
cols = ["dump", "score", "int_score", "token_count", "language_score"]
dumps, score, iscore, tc, ls, nbytes = collections.Counter(), [], [], [], [], []
for p in data.parts():
    t = pq.read_table(p, columns=cols)
    dumps.update(t.column("dump").to_pylist())
    score.append(t.column("score").to_numpy()); iscore.append(t.column("int_score").to_numpy())
    tc.append(t.column("token_count").to_numpy()); ls.append(t.column("language_score").to_numpy())
    nbytes.append(np.concatenate([pc.binary_length(b.column(0)).to_numpy() for b in pq.ParquetFile(p).iter_batches(batch_size=1000, columns=["text"])]))
score, iscore, tc, ls, nbytes = map(np.concatenate, (score, iscore, tc, ls, nbytes))
np.save("/home/amit/.cache/vidyalai-scratch/t03/meta_score.npy", score)
np.save("/home/amit/.cache/vidyalai-scratch/t03/meta_bytes.npy", nbytes)
np.save("/home/amit/.cache/vidyalai-scratch/t03/meta_gpt2.npy", tc)
ys = collections.Counter()
for d, n in dumps.items(): ys[d[8:12]] += n
out = dict(docs=len(score), gpt2_tokens=int(tc.sum()), text_bytes=int(nbytes.sum()), n_dumps=len(dumps),
           dumps_min=min(dumps.values()), dumps_max=max(dumps.values()), first=min(dumps), last=max(dumps),
           per_year=dict(sorted(ys.items())),
           int_score=np.bincount(iscore).tolist(), score_min=float(score.min()), score_max=float(score.max()),
           score_mean=float(score.mean()), score_pct=np.percentile(score, [10, 25, 50, 75, 90, 99]).tolist(),
           tokens_by_int_score={int(k): int(tc[iscore == k].sum()) for k in np.unique(iscore)},
           gpt2_tc_pct=np.percentile(tc, [1, 5, 25, 50, 75, 95, 99]).tolist(), gpt2_tc_min=int(tc.min()), gpt2_tc_max=int(tc.max()),
           gpt2_tc_mean=float(tc.mean()), lang_min=float(ls.min()), lang_median=float(np.median(ls)),
           bytes_per_gpt2_token=float(nbytes.sum() / tc.sum()))
hist, edges = np.histogram(score, bins=np.arange(2.5, 5.2, 0.0625))
out["score_hist"] = dict(counts=hist.tolist(), edges=edges.tolist())
print(json.dumps(out, indent=1))
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/t03/stats.json", "w"), indent=1)
