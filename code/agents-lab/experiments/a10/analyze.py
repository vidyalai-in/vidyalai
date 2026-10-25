"""Regrade every run and summarize arms. usage: analyze.py [--answers]"""
import json, math, sys
from collections import defaultdict
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/a10")
from run import Q, grade, RESULTS

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d*100, (c+h)/d*100)

rows = [json.loads(l) for l in RESULTS.read_text().splitlines()]
# Sonnet s02 rep 0 ended on a call to a nonexistent client tool before the loop handled it; rep 1 is the rerun.
rows = [r for r in rows if not (r["model"] == "claude-sonnet-5-5" and r["qid"] == "s02" and r["rep"] == 0)]
for r in rows:
    r["ok"] = grade(Q[r["qid"]], r["answer"])
    u = r["usage"]
    r["s_billed"] = sum(((x.get("server_tool_use") or {}).get("web_search_requests") or 0) for x in u)
    r["f_billed"] = sum(((x.get("server_tool_use") or {}).get("web_fetch_requests") or 0) for x in u)
    r["tok_in"] = sum(x["input_tokens"] + (x.get("cache_read_input_tokens") or 0) + (x.get("cache_creation_input_tokens") or 0) for x in u)
    r["tok_out"] = sum(x["output_tokens"] for x in u)
    r["think"] = sum(((x.get("output_tokens_details") or {}).get("thinking_tokens") or 0) for x in u)
    r["fee"] = r["s_billed"] * 0.01

if "--answers" in sys.argv:
    for r in sorted(rows, key=lambda r: (r["qid"], r["arm"], r["model"], r["rep"])):
        print(f'{r["qid"]} {r["arm"]:11s} {r["model"][7:13]} r{r["rep"]} {"OK " if r["ok"] else "BAD"} | {r["answer"][:110]}')
    sys.exit()

SUB = ["s02","s04","s05","s07","s12","m02","m03","m05","m06","m09","d01","d02","d03","d04","d05"]
def summary(sel, label):
    n = len(sel); k = sum(r["ok"] for r in sel); lo, hi = wilson(k, n)
    m = lambda f: sum(f(r) for r in sel) / n
    cost = m(lambda r: r["cost"])
    print(f'{label:42s} n={n:3d} {k:3d}/{n:<3d} {100*k/n:5.1f}% ({lo:4.1f}-{hi:5.1f})  searches {m(lambda r:r["s_billed"]):.2f}  '
          f'fetches {m(lambda r:r["f_billed"]):.2f}  in {m(lambda r:r["tok_in"]):8,.0f}  out {m(lambda r:r["tok_out"]):6,.0f}  '
          f'${cost:.4f}  fee share {100*m(lambda r:r["fee"])/cost:4.1f}%  lat {m(lambda r:r["latency_s"]):5.1f}s  '
          f'cost/correct ${cost*n/max(k,1):.4f}')

arms = [("one-search","claude-haiku-5-5"),("research","claude-haiku-5-5"),("research-df","claude-haiku-5-5"),("research","claude-sonnet-5-5")]
print("== all questions")
for a, mo in arms:
    sel = [r for r in rows if r["arm"]==a and r["model"]==mo]
    summary(sel, f"{a} {mo}")
    for kind in ("single","multi","deep"):
        summary([r for r in sel if r["kind"]==kind], f"   {kind}")
print("== 15-question subset")
for a, mo in arms:
    summary([r for r in rows if r["arm"]==a and r["model"]==mo and r["qid"] in SUB], f"{a} {mo}")
print("== subset, rep 0 only")
for a, mo in arms:
    summary([r for r in rows if r["arm"]==a and r["model"]==mo and r["qid"] in SUB and r["rep"]==(1 if (mo.endswith('sonnet-5-5') and r["qid"]=="s02") else 0)], f"{a} {mo}")
# latency and api calls
for a, mo in arms:
    sel = [r for r in rows if r["arm"]==a and r["model"]==mo]
    lats = sorted(r["latency_s"] for r in sel)
    print(a, mo, "median latency", lats[len(lats)//2], "max", lats[-1], "api calls>1:", sum(r["api_calls"]>1 for r in sel),
          "stops", {s: sum(r["stop"]==s for r in sel) for s in set(r["stop"] for r in sel)},
          "no-search runs", sum(r["s_billed"]==0 for r in sel), "tool errors", sum(len(r["tool_errors"]) for r in sel),
          "thinking tok mean", round(sum(r["think"] for r in sel)/len(sel)))
