import json, glob, sys
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import video as V
R = {}
import os
for f in sorted(glob.glob("runs/exp2_eval_*.json"), key=os.path.getmtime):
    R.update(json.load(open(f)))          # later files (factorized at 4,000 steps, inflated) overwrite earlier ones
out = {}
for m, r in R.items():
    ps = r["per_seed"]
    fd = [p["fd"] for p in ps]
    cat = lambda k: np.concatenate([p[k] for p in ps])
    same = cat("same"); n = len(same)
    row = {"n_clips": int(n), "seeds": len(ps), "fd_mean": float(np.mean(fd)), "fd_min": float(min(fd)), "fd_max": float(max(fd)),
           "same_label": V.wilson(int(same.sum()), n)}
    for k in ("agree", "cos0", "cosn", "speed", "accel"):
        row[k] = V.boot_ci(cat(k))
    if "timing" in r: row["timing"] = r["timing"]
    out[m] = row
    print(f"{m:11s} FD {row['fd_mean']:.1f} ({row['fd_min']:.1f}-{row['fd_max']:.1f})  same {row['same_label'][0]*100:.1f}% "
          f"[{row['same_label'][1]*100:.1f},{row['same_label'][2]*100:.1f}]  agree {row['agree'][0]:.3f}  cos0 {row['cos0'][0]:.3f} "
          f"[{row['cos0'][1]:.3f},{row['cos0'][2]:.3f}]  speed {row['speed'][0]:.2f}  accel {row['accel'][0]:.2f} [{row['accel'][1]:.2f},{row['accel'][2]:.2f}]")
json.dump(out, open("runs/exp2_summary.json", "w"), indent=1)
