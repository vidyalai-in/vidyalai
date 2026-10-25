"""Measure the LSH match probability against the formula 1-(1-s^8)^14.

Take real FineWeb-Edu documents, make edited copies (a random fraction of words replaced),
compute the exact 5-gram Jaccard similarity of each pair, and check whether the MinHash
signatures share all 8 values in at least one of the 14 bands.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import data

rng = np.random.default_rng(0)
docs = [json.loads(l)["text"] for l in open("/home/amit/.cache/vidyalai-scratch/fromscratch/data/tokenizer-sample/heldout.jsonl")]
docs = [d for d in docs if 300 <= len(d.split()) <= 3000][:600]
vocab = sorted({w for d in docs[:50] for w in d.split()})
rows = []
t0 = time.time()
for d in docs:
    words = d.split()
    sa, siga = data.shingles(d), data.minhash(d)
    for q in rng.uniform(0.0, 0.12, 20):           # replace a fraction q of the words
        w = list(words)
        for i in np.flatnonzero(rng.random(len(w)) < q):
            w[i] = vocab[rng.integers(len(vocab))]
        e = " ".join(w)
        sb, sigb = data.shingles(e), data.minhash(e)
        jac = len(np.intersect1d(sa, sb)) / len(np.union1d(sa, sb))
        agree = (siga == sigb)
        band_hit = agree.reshape(data.BANDS, data.ROWS).all(axis=1).any()
        rows.append((jac, agree.mean(), bool(band_hit)))
r = np.array(rows)
print(f"{len(r)} pairs in {time.time()-t0:.0f} s")
bins = np.arange(0.3, 1.0001, 0.05)
out = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (r[:, 0] >= lo) & (r[:, 0] < hi)
    if m.sum() < 20: continue
    s_mid = r[m, 0].mean()
    p = r[m, 2].mean(); n = int(m.sum())
    # Wilson 95% interval
    z = 1.96; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    out.append(dict(lo=lo, hi=hi, s=s_mid, n=n, measured=p, lo95=c-h, hi95=c+h, theory=float(np.mean(data.match_probability(r[m, 0]))),
                    est_err=float(np.mean(np.abs(r[m,1]-r[m,0])))))
    print(f"s in [{lo:.2f},{hi:.2f}) n={n:5d} mean s={s_mid:.3f} measured={p:.3f} [{c-h:.3f},{c+h:.3f}] theory={out[-1]['theory']:.3f} |sig est - s|={out[-1]['est_err']:.3f}")
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/t03/scurve.json", "w"), indent=1)
