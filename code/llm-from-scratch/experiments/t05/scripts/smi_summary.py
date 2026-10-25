import sys, statistics as st
for f in sys.argv[1:]:
    rows = [l.split(", ") for l in open(f) if l.strip()]
    sm = [int(r[1].split()[0]) for r in rows]; mem = [int(r[2].split()[0]) for r in rows]
    pw = [float(r[3].split()[0]) for r in rows if "N/A" not in r[3]]; pl = [r[4] for r in rows]
    t = [int(r[5]) for r in rows]
    print(f"{f.split('/')[-1]:32s} n={len(rows):3d} sm med {st.median(sm)} ({min(sm)}-{max(sm)})  mem {sorted(set(mem))}  "
          f"power mean {st.mean(pw):.1f} max {max(pw):.1f}  limit {sorted(set(pl))[:4]}  temp {min(t)}-{max(t)}")
