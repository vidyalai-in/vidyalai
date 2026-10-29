# Facts for each Part 13 run: steps, tokens, wall time, tokens/s, final val, GPU clocks and power from nvidia-smi logs.
import json, statistics, sys, os
R, L = '/home/amit/.cache/vidyalai-scratch/t13/runs', '/home/amit/.cache/vidyalai-scratch/t13/logs'
def smi(tag):
    rows = [l.split(', ') for l in open(f'{L}/smi_{tag}.csv') if l.count(',') >= 5]
    busy = [r for r in rows if float(r[3]) > 20]               # samples while the GPU was working
    sm, mem, pw, temp = ([float(r[i]) for r in busy] for i in (1, 2, 3, 5))
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    return {"samples": len(rows), "busy": len(busy), "sm_median": statistics.median(sm), "sm_p05": q(sm, .05), "sm_p95": q(sm, .95),
            "mem_clock_counts": {str(int(k)): mem.count(k) for k in set(mem)}, "power_mean": round(statistics.mean(pw), 1),
            "temp_median": statistics.median(temp), "power_limit": rows[0][4]}
out = {}
for run, tags in (("ce_s15", ["ce_s15"]), ("kd_s15_t1", ["kd_s15_t1"]), ("kd_s15_t2a09", ["kd_s15_t2"])):
    p = f'{R}/{run}/log.jsonl'
    if not os.path.exists(p): continue
    recs = [json.loads(l) for l in open(p)]
    steps, vals = {}, {}
    for r in recs:
        if "loss" in r and "step" in r: steps[r["step"]] = r            # last write wins (resumed runs)
        if "val_loss" in r: vals[r["step"]] = (r["val_loss"], r.get("val_kl"))
    ends = [r for r in recs if r.get("event") == "end"]
    walls = [l.split() for l in open(f'{L}/gpu_wall.log') if l.split()[0] in tags]
    out[run] = {"steps": len(steps), "tokens": max(s["tokens"] for s in steps.values()),
                "train_seconds": round(sum(s["dt"] for s in steps.values()), 1),
                "tok_s_median": statistics.median(s["tok_per_s"] for s in steps.values()),
                "job_seconds": sum(int(w[2]) for w in walls), "resumes": sum(r.get("event") == "resume" for r in recs),
                "val": sorted(vals.items()), "final_train": steps[max(steps)], "peak_mem": [e["peak_mem_gib"] for e in ends],
                "smi": smi(tags[0])}
    print(run, {k: v for k, v in out[run].items() if k not in ("val", "final_train", "smi")}, out[run]["val"][-1], out[run]["smi"])
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t13/post/run_facts.json', 'w'), indent=1)
