# Clocks and power from nvidia-smi rows logged inside each run's timed window.
import sys, json, statistics as st
from datetime import datetime
D = "/home/amit/.cache/vidyalai-scratch/t04/logs/"
for name in sys.argv[1:]:
    res = json.loads([l for l in open(D + name + ".out") if l.startswith("RESULT")][0][7:])
    rows = []
    for l in open(D + name + ".smi.csv"):
        r = l.strip().split(", ")
        ts = datetime.strptime(r[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        if res.get("t_start", 0) <= ts <= res.get("t_end", 0): rows.append(r)
    if not rows: print(name, "no rows in window"); continue
    sm = [int(r[1].split()[0]) for r in rows]; mem = [int(r[2].split()[0]) for r in rows]
    pw = [float(r[3].split()[0]) for r in rows]; t = [int(r[5]) for r in rows]
    print(f"{name:28s} {res.get('micro_s_median')}s {res.get('tok_s')} tok/s {res.get('peak_mem_GiB')} GiB | n={len(rows)} "
          f"SM med {st.median(sm):.0f} ({min(sm)}-{max(sm)}) mem {sorted(set(mem))} power mean {st.mean(pw):.1f} W "
          f"({min(pw):.1f}-{max(pw):.1f}) limit {sorted(set(r[4] for r in rows))} temp {min(t)}-{max(t)}")
