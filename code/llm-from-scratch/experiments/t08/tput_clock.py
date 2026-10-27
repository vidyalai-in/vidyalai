"""tput_clock.py: tokens/s vs SM clock over time, in 30-minute bins, for the long runs; MFU per run."""
import csv, json, datetime as dt, statistics as st
PEAK = 25.3e12
CFG = {"s11": (240, 5), "s15": (288, 6), "s22": (384, 6), "s33": (480, 7), "s47": (576, 8)}
runs = json.load(open("post/runs.json"))
for r in ["s11", "s15", "s22", "s33", "s47"]:
    o = runs[r]; d, nl = CFG[r]
    fpt = 6 * o["params"] + 12 * nl * d * 1024
    print(f"{r}: FLOPs/token {fpt/1e6:.1f}M (attention {12*nl*d*1024/fpt:.0%}), step tok/s {o['tok_s_steps']:,.0f} -> "
          f"{o['tok_s_steps']*fpt/1e12:.1f} TFLOPS, MFU {o['tok_s_steps']*fpt/PEAK:.1%}; 6N-only MFU {o['tok_s_steps']*6*o['params']/PEAK:.1%}")
for r in ["s33", "s47"]:
    recs = [json.loads(l) for l in open(f"runs/{r}/log.jsonl")]
    t0 = dt.datetime.fromisoformat(open(f"logs/{r}.out").readline().split()[-1])
    steps = [x for x in recs if "loss" in x]
    smi = []
    for row in csv.reader(open(f"logs/{r}.smi.csv")):
        if len(row) < 8: continue
        t = dt.datetime.strptime(row[0].strip()[:19], "%Y/%m/%d %H:%M:%S")
        smi.append((t, int(row[1].split()[0]), float(row[3].split()[0]), int(row[5]), row[2].strip()))
    s0 = smi[0][0]
    bins = {}
    for x in steps:
        b = int(x["elapsed"] // 1800); bins.setdefault(b, []).append(x)
    print(r, "bin(h)  tok/s  SMmed  power  temp")
    for b, xs in sorted(bins.items()):
        tps = sum(len(xs) * 0 + x["tokens"] - (x["tokens"] - recs[0]["tokens_per_step"]) for x in xs) / sum(x["dt"] for x in xs)
        lo, hi = s0 + dt.timedelta(seconds=b * 1800 + 30), s0 + dt.timedelta(seconds=(b + 1) * 1800 + 30)
        w = [q for q in smi if lo <= q[0] < hi and q[2] > 25]
        if w: print(f"  {b/2:4.1f}  {tps:7,.0f}  {st.median(q[1] for q in w):5.0f}  {st.mean(q[2] for q in w):5.1f}  {st.median(q[3] for q in w):3.0f}")
