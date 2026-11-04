"""Post-process Part 7 runs into runs/results.json (small, committable)."""
import json, glob, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np

R = Path("runs")
out = {}


def rows(f):
    p = R / f
    return [json.loads(l) for l in open(p)] if p.exists() else []


def name_of(ck):
    return Path(ck).name.split("-snap-")[0]


# FD vs sampling steps (3 seeds x 2,000 samples each)
sw = defaultdict(lambda: defaultdict(list))
gs = defaultdict(lambda: defaultdict(list))
for r in rows("sweep.jsonl"):
    sw[name_of(r["ckpt"])][r["steps"]].append(r["fd"])
    gs[name_of(r["ckpt"])][r["steps"]].append(r["gen_sec"])
out["sweep"] = {m: {s: {"mean": float(np.mean(v)), "min": min(v), "max": max(v), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        "n_seeds": len(v), "gen_sec_2000": float(np.median(gs[m][s]))}
                    for s, v in sorted(d.items())} for m, d in sw.items()}
# FD vs training time
sn = defaultdict(list)
for r in rows("snaps.jsonl"):
    sn[name_of(r["ckpt"])].append({"step": r["train_step"], "train_time_s": round(r["train_time"], 1), "fd": round(r["fd"], 3),
                                   "mean_conf": round(r["mean_conf"], 4)})
out["snaps"] = {k: sorted(v, key=lambda x: x["step"]) for k, v in sn.items()}
# losses
out["loss"] = {}
for f in ["ddpm-unet", "rf-unet", "rf-dit", "reflow-unet"]:
    rr = rows(f + ".jsonl")
    if rr: out["loss"][f] = [{"step": r["step"], "loss": round(r["loss"], 5), "train_time": r["train_time"]} for r in rr]
out["straight"] = {r["tag"]: {k: r[k] for k in ("ratio_mean", "ratio_sd", "ratio_p95", "S_mean", "S_sd", "maxdev_mean", "chord_mean", "n", "steps")}
                   for r in rows("straight.jsonl") if r.get("tag")}
if (R / "timing.json").exists(): out["timing"] = json.load(open(R / "timing.json"))
if (R / "toy.json").exists():
    t = json.load(open(R / "toy.json"))
    out["toy"] = {"straight": t["straight"], "floor_mean": float(np.mean(t["floor"])), "floor_sd": float(np.std(t["floor"], ddof=1)),
                  "swd": {m: {s: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for s, v in d.items()} for m, d in t["swd"].items()}}
json.dump(out, open(R / "results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("loss",)}, indent=1)[:6000])
