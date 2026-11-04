"""Experiment 2 analysis (CPU): original ARC-Challenge text vs its Haiku paraphrase, per model and
detector (paired AUROC, item and its paraphrase resampled together), and the shard tests."""
import json, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import contamination as C
RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
res = {}
for m in ("qwen05", "smol135", "pythia160", "ours88"):
    f = f"{RUNS}/offshelf-items-{m}.json"
    if not os.path.exists(f):
        continue
    E = json.load(open(f))
    for det in ("loss", "zlib", "mink", "minkpp"):
        a = C.auroc_ci([d[det] for d in E["verbatim"]], [d[det] for d in E["para"]], n_boot=2000, paired=True)
        res[f"{m}/{det}"] = dict(auroc=a[0], ci=[a[1], a[2]], n=len(E["ids"]))
        print(m, det, "AUROC %.3f [%.3f, %.3f]" % a)
for f in sorted(glob.glob(f"{RUNS}/offshelf-shard-*.json")):
    r = json.load(open(f))
    res[f"shard/{r['model']}/{r['bench']}"] = {k: r[k] for k in ("p", "t", "mean_diff", "n_shards", "n_perm", "n_examples", "per_shard")}
    res[f"shard/{r['model']}/{r['bench']}"]["shards_p_lt_05"] = int(sum(p < 0.05 for p in r["shard_p"]))
    print("shard", r["model"], r["bench"], res[f"shard/{r['model']}/{r['bench']}"])
json.dump(res, open(f"{RUNS}/results-exp2.json", "w"), indent=1)
