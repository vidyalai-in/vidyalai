"""analyze.py: per-run summary from runs/<name>/log.jsonl and logs/<name>.smi.csv."""
import csv, json, statistics as st, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
RUNS = ["s11", "s15", "s22", "s33", "s47", "iso_s11", "iso_s22"]
PRED = {"s11": 178000, "s15": 125000, "s22": 105000, "s33": 70000, "s47": 51500, "iso_s11": 178000, "iso_s22": 105000}
out = {}
for r in RUNS:
    recs = [json.loads(l) for l in open(f"runs/{r}/log.jsonl")]
    start = recs[0]; end = [x for x in recs if x.get("event") == "end"][-1]
    steps = [x for x in recs if "loss" in x]
    vals = [(x["step"], x["tokens"], x["val_loss"]) for x in recs if "val_loss" in x]
    a = start["args"]; tps = start["tokens_per_step"]
    step_time = sum(x["dt"] for x in steps)
    tokens = len(steps) * tps
    clipped = sum(x["grad_norm"] > a["grad_clip"] for x in steps if x["step"] >= a["warmup"])
    last_clip = max([x["step"] for x in steps if x["grad_norm"] > a["grad_clip"] and x["step"] >= a["warmup"]] or [None], key=lambda v: v or -1)
    rows = [row for row in csv.reader(open(f"logs/{r}.smi.csv")) if len(row) >= 8]
    busy = [row for row in rows if float(row[3].split()[0]) > 25]
    sm = [int(row[1].split()[0]) for row in busy]; pw = [float(row[3].split()[0]) for row in busy]
    lim = [float(row[4].split()[0]) for row in busy]; tmp = [int(row[5]) for row in busy]
    mem = sorted(set(row[2].strip() for row in busy), key=lambda s: -sum(1 for q in busy if q[2].strip() == s))
    energy_kwh = sum(pw) / 3600 / 1000  # 1 sample per second
    o = dict(run=r, model=a["model"], params=start["params"], tokens=tokens, steps=len(steps), tps_step=tps,
             lr=a["lr"], warmup=a["warmup"], tok_s_steps=tokens / step_time, step_hours=step_time / 3600,
             job_hours=end["elapsed"] / 3600, pred_hours=tokens / PRED[r] / 3600, pred_tok_s=PRED[r],
             final_val=vals[-1][2], vals=vals, peak_mem=end["peak_mem_gib"], clipped_after_warmup=clipped, last_clip=last_clip,
             sm_med=st.median(sm), sm_p5=sorted(sm)[len(sm)//20], sm_p95=sorted(sm)[-len(sm)//20], mem_clk=mem,
             power_mean=st.mean(pw), limit_min=min(lim), limit_max=max(lim), temp_max=max(tmp), temp_med=st.median(tmp),
             energy_kwh=energy_kwh, smi_samples=len(rows))
    out[r] = o
    print(f"{r:8} N={o['params']:>11,} tok={tokens/1e6:8.1f}M steps={len(steps):5} B={tps:7,} lr={a['lr']:<8} "
          f"val={o['final_val']:.4f} tok/s={o['tok_s_steps']:8,.0f} (pred {PRED[r]:,}) step_h={o['step_hours']:.2f} "
          f"job_h={o['job_hours']:.2f} pred_h={o['pred_hours']:.2f} clip>{a['warmup']}={clipped} last={last_clip} "
          f"SM {o['sm_med']:.0f} ({o['sm_p5']}-{o['sm_p95']}) mem {mem} P {o['power_mean']:.1f}W lim {o['limit_min']:.1f}-{o['limit_max']:.1f} "
          f"T {o['temp_med']:.0f}/{o['temp_max']} E {energy_kwh:.3f}kWh peak {o['peak_mem']}")
json.dump(out, open("post/runs.json", "w"), indent=1)
