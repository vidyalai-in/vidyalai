"""Harness check: every canonical solution must pass its own test suite, and a stub must fail."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import verify
D = "/home/amit/.cache/vidyalai-scratch/evals/data/"
he = json.load(open(D + "e08-humaneval.json")); hp = json.load(open(D + "e08-humanevalplus.json"))

def run(args):
    suite, t, body = args
    src = t["prompt"] + body + "\n" + t["test"] + f"\ncheck({t['entry_point']})\n"
    t0 = time.time(); st, det = verify.run_program(src, timeout=30); return suite, t["task_id"], st, det, time.time() - t0

jobs = [("base", t, t["canonical_solution"]) for t in he] + [("plus", t, t["canonical_solution"]) for t in hp]
jobs += [("stub-base", t, "    return None\n") for t in he]
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(run, jobs))
out = {}
for s, tid, st, det, dt in res:
    out.setdefault(s, []).append((tid, st, det, round(dt, 2)))
for s, rows in out.items():
    ok = sum(r[1] == "pass" for r in rows)
    print(s, f"{ok}/{len(rows)} pass", "max time", max(r[3] for r in rows))
    for r in rows:
        if (r[1] != "pass") == s.startswith("stub"):
            continue
        print("   ", r)
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/e08/runs/harness_check.json", "w"), indent=0)
