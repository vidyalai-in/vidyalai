"""Repair rates from a repair run, split by whether the public tests catch the bug."""
import json
import sys
import math


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d * 100, (c + h) / d * 100)

rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"units {len(rows)}, items {len({r['item'] for r in rows})}")
groups = {"all": rows, "public tests catch it": [r for r in rows if not r["public_ok"]],
          "public tests miss it": [r for r in rows if r["public_ok"]]}
for gname, g in groups.items():
    if not g:
        continue
    print(f"\n{gname} (n={len(g)})")
    for s in ("critique r1", "critique r2", "tests", "reflexion", "resample", "oracle"):
        if s.startswith("critique"):
            i = int(s[-1]) - 1
            have = [r for r in g if "critique" in r and len(r["critique"]) > i]
            oks = [r["critique"][i]["ok"] for r in have]
            chg = sum(r["critique"][i]["changed"] for r in have)
            cost = sum(sum(x["cost"] for x in r["critique"][:i + 1]) for r in have)
            extra = f" changed code {chg}/{len(have)}"
        else:
            have = [r for r in g if s in r]
            oks = [r[s]["ok"] for r in have]
            cost = sum(r[s]["cost"] for r in have)
            extra = f" mean calls {sum(r[s]['calls'] for r in have) / max(len(have), 1):.2f}"
        if not have:
            continue
        k = sum(oks)
        lo, hi = wilson(k, len(have))
        print(f"  {s:12s} fixed {k:3d}/{len(have):<3d} {100 * k / len(have):5.1f}%  CI {lo:5.1f}-{hi:5.1f}  "
              f"repair cost/unit ${cost / len(have):.5f}{extra}")
