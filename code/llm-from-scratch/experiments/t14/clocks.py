"""Median SM clock, memory clock, power and power limit over the busy seconds of each run's nvidia-smi log."""
import csv, glob, json, os, statistics, sys
L = "/home/amit/.cache/vidyalai-scratch/t14/logs"
out = {}
for f in sorted(glob.glob(f"{L}/smi_*.csv")):
    rows = [r for r in csv.reader(open(f)) if len(r) >= 7]
    busy = [r for r in rows if float(r[3]) > 25]          # power draw above 25 W: kernels running
    if not busy:
        continue
    g = lambda i: [float(r[i]) for r in busy if r[i].strip() not in ("[N/A]", "")]
    tag = os.path.basename(f)[4:-4]
    out[tag] = {"busy_s": len(busy), "sm_med": statistics.median(g(1)), "mem_clk": sorted(set(g(2))),
                "power_med": statistics.median(g(3)), "limit": [min(g(4)), max(g(4))],
                "temp": [min(g(5)), max(g(5))], "smi_mem_max": max(float(r[6]) for r in rows)}
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/t14/post/clocks.json", "w"), indent=1)
for k, v in out.items():
    if len(sys.argv) < 2 or k in sys.argv[1:]:
        print(k, v)
