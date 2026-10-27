# Part 7: GPU clocks, power, temperature and energy over the main run, from nvidia-smi's 1-second log (CPU only).
import sys, json, datetime as dtm
import numpy as np
from collections import Counter
F = "/home/amit/.cache/vidyalai-scratch/t07/logs/main88.smi.csv"
L = "/home/amit/.cache/vidyalai-scratch/t07/runs/main88/log.jsonl"
rows = []
bad = Counter()
for line in open(F):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 8: bad["short"] += 1; continue
    try:
        t = dtm.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        sm, mem = int(p[1].split()[0]), int(p[2].split()[0])
        pw = float(p[3].split()[0]) if "N/A" not in p[3] else np.nan
        lim = float(p[4].split()[0]) if "N/A" not in p[4] else np.nan
        temp, used = int(p[5]), int(p[6].split()[0]); reason = p[7]
    except Exception as e:
        bad[str(e)[:40]] += 1; continue
    rows.append((t, sm, mem, pw, lim, temp, used, reason))
print("parsed", len(rows), "bad", dict(bad))
t, sm, mem, pw, lim, temp, used = (np.array([r[i] for r in rows], dtype=float) for i in range(7))
reason = [r[7] for r in rows]
# run window: process start = first smi row; training is when memory.used > 5000 MiB
recs = [json.loads(l) for l in open(L)]
start_wall = dtm.datetime(2026, 10, 26, 18, 23, 9).timestamp()
print(f"smi span {t[0]-start_wall:.1f} .. {t[-1]-start_wall:.1f} s after 18:23:09")
d = np.diff(t)
print(f"sample interval: median {np.median(d):.3f} s, mean {d.mean():.3f}, max {d.max():.1f}; gaps > 5 s: {(d > 5).sum()}, total {d[d>5].sum():.0f} s")
big = np.argsort(-d)[:8]; print("largest gaps:", [(round(float(d[i]),1), dtm.datetime.fromtimestamp(t[i]).strftime('%H:%M:%S')) for i in big])
bogus = (pw > 200) | np.isnan(pw)
print("bogus power readings:", int(bogus.sum()), pw[pw > 200][:10])
busy = used > 5000
print(f"busy samples {busy.sum()} of {len(t)}")
b = busy & ~bogus
def s(x): return f"median {np.median(x):.0f}, mean {np.mean(x):.1f}, p5 {np.percentile(x,5):.0f}, p95 {np.percentile(x,95):.0f}, min {np.min(x):.0f}, max {np.max(x):.0f}"
print("SM clock (busy):", s(sm[b]))
print("mem clock (busy):", Counter(mem[b].astype(int)).most_common(6))
print("power (busy):", s(pw[b]), f"mean {pw[b].mean():.2f}")
print("power limit (busy):", s(lim[b]), "share at 50 W:", f"{(lim[b] >= 49.99).mean():.3f}", "share >= 49:", f"{(lim[b]>=49).mean():.3f}", "share < 45:", f"{(lim[b]<45).mean():.4f}")
print("limit values:", Counter(np.round(lim[b], 1)).most_common(8))
print("temp (busy):", s(temp[b]), "share 78-83:", f"{((temp[b]>=78)&(temp[b]<=83)).mean():.3f}")
print("temp hist:", sorted(Counter(temp[b].astype(int)).items()))
print("reasons (busy):", Counter(np.array(reason)[b]).most_common(8))
print("mem used (busy) max MiB:", used[b].max())
# energy: trapezoid over valid samples (hold last good value through gaps)
pv = pw.copy(); 
for i in range(len(pv)):
    if bogus[i]: pv[i] = pv[i-1] if i else 0
E = np.sum(0.5 * (pv[1:] + pv[:-1]) * d)    # joules
Eb = np.sum((0.5 * (pv[1:] + pv[:-1]) * d)[busy[1:]])
print(f"energy whole log: {E/3.6e6:.3f} kWh over {(t[-1]-t[0])/3600:.3f} h (mean {E/(t[-1]-t[0]):.2f} W); busy part {Eb/3.6e6:.3f} kWh")
print(f"at 0.20 $/kWh: ${E/3.6e6*0.2:.3f}")
# per-hour summary for the chart
hours = (t - start_wall) / 3600
for h in range(0, 17):
    m = b & (hours >= h) & (hours < h + 1)
    if m.sum(): print(f"  hour {h:2d}: sm {np.median(sm[m]):.0f} power {pw[m].mean():.1f} limit {np.median(lim[m]):.1f} (min {lim[m].min():.1f}) temp {np.median(temp[m]):.0f} n {m.sum()}")
# correlation of per-step tokens/s with the SM clock during that step
train = [r for r in recs if "loss" in r]
t0 = start_wall + (60544 - 60534.08)   # approximate: process t0
ends = np.array([r["elapsed"] for r in train]); dts = np.array([r["dt"] for r in train]); tps = np.array([r["tok_per_s"] for r in train])
# align using wall-clock end: training t0 = end wall time - end elapsed
end_wall = dtm.datetime(2026, 10, 27, 11, 12, 13).timestamp()
T0 = end_wall - 60534.08
smstep = []
for e, dd in zip(ends, dts):
    m = (t >= T0 + e - dd) & (t <= T0 + e)
    smstep.append(np.median(sm[m]) if m.sum() else np.nan)
smstep = np.array(smstep)
ok = ~np.isnan(smstep)
print(f"T0 offset from 18:23:09: {T0-start_wall:.1f} s; steps with clock samples {ok.sum()}")
print(f"corr(tok/s, SM clock) = {np.corrcoef(tps[ok][1:], smstep[ok][1:])[0,1]:.3f}")
r = tps[ok] / (smstep[ok] / 1000)
print(f"tok/s per GHz: median {np.median(r[1:]):.0f}, p5 {np.percentile(r[1:],5):.0f}, p95 {np.percentile(r[1:],95):.0f}")
for lo, hi in ((1500,1700),(1700,1800),(1800,1900),(1900,2000),(2000,2100),(2100,2400)):
    m = ok & (smstep >= lo) & (smstep < hi)
    if m.sum(): print(f"  steps at SM {lo}-{hi}: {m.sum()} median tok/s {np.median(tps[m]):.0f}")
np.save("/home/amit/.cache/vidyalai-scratch/t07/post/smstep.npy", smstep)
# hourly series for chart: 10-minute bins
bins = []
for k in range(0, int(hours[-1] * 6) + 1):
    m = b & (hours >= k/6) & (hours < (k+1)/6)
    if m.sum(): bins.append((round(k/6 + 1/12, 3), int(np.median(sm[m])), round(float(pw[m].mean()), 1), round(float(np.median(lim[m])), 1), int(np.median(temp[m]))))
json.dump(bins, open("/home/amit/.cache/vidyalai-scratch/t07/post/smi_bins.json", "w"))
print("bins", len(bins), bins[:3], bins[-3:])
