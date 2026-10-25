"""Summarize a run file: pass rates (Wilson 95% CI), flips, tokens and cost per strategy."""
import json
import math
import sys
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d * 100, (c + h) / d * 100)


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


rows = [json.loads(l) for l in open(sys.argv[1])]
n = len(rows)
print(f"units: {n}, tasks: {len({r['task'] for r in rows})}")


def tok(v):
    return sum(x["tokens"] for x in v) if isinstance(v, list) else v["tokens"]


def cost(v):
    return sum(x["cost"] for x in v) if isinstance(v, list) else v["cost"]


strategies = [("single", None), ("critique r1", ("critique", 0)), ("critique r2", ("critique", 1)),
              ("tests", "tests"), ("reflexion", "reflexion"), ("resample", "resample"), ("oracle", "oracle"),
              ("tool", "tool")]
print(f"{'strategy':12s} {'pass':>9s} {'%':>6s} {'95% CI':>13s} {'c->w':>5s} {'w->c':>5s} {'p(McN)':>7s} "
      f"{'tok/unit':>9s} {'$/unit':>9s} {'$/solved':>9s}")
for name, key in strategies:
    oks, flips_cw, flips_wc, toks, costs = [], 0, 0, [], []
    have = 0
    for r in rows:
        s0 = r["single"]
        if key is None:
            ok = s0["ok"]; extra_t, extra_c = 0, 0
        elif isinstance(key, tuple):
            if key[0] not in r: continue
            rounds = r[key[0]][:key[1] + 1]
            ok = rounds[-1]["ok"]; extra_t = sum(x["tokens"] for x in rounds); extra_c = sum(x["cost"] for x in rounds)
        else:
            if key not in r: continue
            ok = r[key]["ok"]; extra_t, extra_c = r[key]["tokens"], r[key]["cost"]
        have += 1
        oks.append(ok)
        if key is not None and key != "tool":
            flips_cw += s0["ok"] and not ok
            flips_wc += (not s0["ok"]) and ok
        if key == "tool":  # the tool agent writes its own first attempt
            toks.append(extra_t); costs.append(extra_c)
        else:
            toks.append(s0["tokens"] + extra_t); costs.append(s0["cost"] + extra_c)
    if not have:
        continue
    k = sum(oks)
    lo, hi = wilson(k, have)
    p = mcnemar(flips_cw, flips_wc) if key not in (None, "tool") else float("nan")
    print(f"{name:12s} {k:4d}/{have:<4d} {100*k/have:6.1f} {lo:5.1f}-{hi:5.1f} {flips_cw:5d} {flips_wc:5d} {p:7.3f} "
          f"{sum(toks)/have:9.0f} {sum(costs)/have:9.5f} {sum(costs)/max(k,1):9.5f}")

# critique: how often the code changed, and which tasks broke
ch = Counter()
for r in rows:
    if "critique" in r:
        for i, x in enumerate(r["critique"]):
            ch[(i, x["changed"])] += 1
print("critique changed code:", dict(ch))
print("\nper-task failures (single / crit r1 / crit r2 / tests / reflexion / resample / oracle / tool):")
per = defaultdict(lambda: [0] * 8)
for r in rows:
    vals = [r["single"]["ok"], r["critique"][0]["ok"], r["critique"][1]["ok"], r["tests"]["ok"], r["reflexion"]["ok"],
            r["resample"]["ok"], r["oracle"]["ok"], r["tool"]["ok"]] if "tool" in r else [r["single"]["ok"]]
    for i, v in enumerate(vals):
        per[r["task"]][i] += (not v)
for t, v in sorted(per.items()):
    if any(v):
        print(f"  {t:22s} {v}")
if any("tool" in r for r in rows):
    tr = [r["tool"] for r in rows if "tool" in r]
    print("\ntool: mean steps %.2f, mean run_public_tests calls %.2f, runs with 0 test calls %d/%d" % (
        sum(x["steps"] for x in tr) / len(tr), sum(x["tests_called"] for x in tr) / len(tr),
        sum(x["tests_called"] == 0 for x in tr), len(tr)))
# feedback strategies: how many units triggered (public tests failed on attempt 1)
trig = sum(1 for r in rows if not r["single"]["public_ok"])
print(f"attempt 1 failed public tests: {trig}/{n}; failed hidden: {sum(not r['single']['ok'] for r in rows)}/{n}; "
      f"passed public but failed hidden: {sum(r['single']['public_ok'] and not r['single']['ok'] for r in rows)}")
