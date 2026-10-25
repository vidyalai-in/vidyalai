import json, glob, math, collections
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (round(100*(c-h), 1), round(100*(c+h), 1))
runs = [json.load(open(f)) for f in sorted(glob.glob("runs/policy-t*-r*.json"))]
calls = [c for r in runs for c in r["calls"]]
print("runs", len(runs), "calls", len(calls), "cost", round(sum(r["cost"] for r in runs), 4),
      "stops", collections.Counter(r["stop"] for r in runs))
out = {}
for pol in ["ask_all", "manual", "sandboxed", "allow_all"]:
    k = sum(c["decisions"][pol] != "allow" for c in calls)
    runs_hit = sum(any(c["decisions"][pol] != "allow" for c in r["calls"]) for r in runs)
    out[pol] = dict(interrupt_calls=k, pct=round(100*k/len(calls), 1), ci=wilson(k, len(calls)),
                    runs_interrupted=runs_hit, runs_pct_ci=wilson(runs_hit, len(runs)),
                    per_run=round(k/len(runs), 2))
    print(pol, out[pol])
sbx = collections.Counter((c["name"], c["decisions"]["sandboxed"]) for c in calls if c["decisions"]["sandboxed"] != "allow")
print("sandboxed non-allow by tool:", dict(sbx))
print("tools used:", dict(collections.Counter(c["name"] for c in calls)))
json.dump(out, open("runs/policy_summary.json", "w"), indent=1)
