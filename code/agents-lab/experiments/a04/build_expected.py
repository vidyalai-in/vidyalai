import json, sys
sys.path.insert(0, ".")
from tasks import TASKS
from sandbox import run_tests
out = {}
for t in TASKS:
    entry = t["id"]
    res = run_tests(t["ref"], entry, t["tests"], [None] * len(t["tests"]))
    out[entry] = [r["got"] for r in res]
    for r in res:
        print(f"{r['call'][:70]:70s} -> {r['got'][:80]}")
    # self-check: ref passes its own expectations
    chk = run_tests(t["ref"], entry, t["tests"], out[entry])
    assert all(r["ok"] for r in chk), (entry, chk)
json.dump(out, open("expected.json", "w"), indent=0)
print(len(out), "tasks", sum(len(v) for v in out.values()), "tests")
