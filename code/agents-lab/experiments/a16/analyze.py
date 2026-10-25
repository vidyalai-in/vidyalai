"""Transcript statistics for Part 16: tool errors by tool, cache share, thinking, per-step latency.
python analyze.py <config> [...]"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

A16 = Path(__file__).parent
for cfg in sys.argv[1:]:
    calls, errs, kinds = Counter(), Counter(), Counter()
    inp = cr = cw = out = 0
    lat = []
    for f in sorted((A16 / "runs" / cfg / "tx").glob("*.jsonl")):
        names = {}
        for line in f.read_text().splitlines():
            d = json.loads(line)
            if d["event"] == "step":
                inp += d["input_tokens"]; cr += d["cache_read"]; cw += d["cache_write"]; out += d["output_tokens"]
                lat.append(d["latency_s"])
                for b in d["content"]:
                    if b["type"] == "tool_use":
                        names[b["id"]] = b["name"]
                        calls[b["name"]] += 1
            if d["event"] == "tool_results":
                for r in d["results"]:
                    if r.get("is_error"):
                        n = names.get(r["tool_use_id"], "?")
                        errs[n] += 1
                        kinds[r["content"].split(":")[0] + (" (old_text match)" if "occurs" in r["content"] else "")] += 1
    tot = inp + cr + cw
    print(f"== {cfg}: calls {sum(calls.values())}, errors {sum(errs.values())}")
    for n in calls:
        print(f"   {n:14s} {calls[n]:4d} calls, {errs[n]:3d} errors ({100 * errs[n] / calls[n]:.1f}%)")
    print("   error kinds", dict(kinds))
    print(f"   input read {tot:,}: uncached {inp:,} ({100*inp/tot:.1f}%), cache read {cr:,} ({100*cr/tot:.1f}%), "
          f"cache write {cw:,} ({100*cw/tot:.1f}%); output {out:,}")
    lat.sort()
    print(f"   model call latency: n={len(lat)} median {statistics.median(lat):.2f}s p95 {lat[int(0.95*(len(lat)-1))]:.2f}s")
