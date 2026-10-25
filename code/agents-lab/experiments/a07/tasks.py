"""Part 7 task set, built on Part 5's Larkspur incident tracker (incidents.json, 1,000 records).

Three kinds of task, 18 in all, every answer computed here:
  wide   (10)  compare four services in one month: the work splits into four independent lookups
  twostage (4) find one incident in a (service, month) group, then count a second group by its root
               cause: the second half depends on the first
  chain  (4)   Part 5 chains c01, c03, c05, c09: each record names the next, nothing to split

python tasks.py -> writes tasks.json next to this file.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
A05 = Path("/home/amit/.cache/vidyalai-scratch/a05")
INC = json.loads((A05 / "incidents.json").read_text())
R = random.Random(707)
MONTH_NAME = {"2026-06": "June 2026", "2026-07": "July 2026", "2026-08": "August 2026", "2026-09": "September 2026"}

groups = defaultdict(list)
for r in INC.values():
    groups[(r["service"], r["opened"][:7])].append(r)

STATS = {
    "sum_impact": ("the highest total customer impact minutes", lambda rs: sum(r["impact_min"] for r in rs)),
    "sum_robots": ("the highest total number of robots affected", lambda rs: sum(r["robots"] for r in rs)),
    "count_sev12": ("the most SEV1 or SEV2 incidents", lambda rs: sum(r["severity"] in ("SEV1", "SEV2") for r in rs)),
    "count_cause": (None, None),  # filled per task with a cause
}


def wide(i, qkind, used):
    while True:
        month = R.choice(list(MONTH_NAME))
        cands = [s for (s, m), rs in groups.items() if m == month and 9 <= len(rs) <= 20]
        svcs = sorted(R.sample(cands, 4))
        if (tuple(svcs), month) in used:
            continue
        rs = {s: groups[(s, month)] for s in svcs}
        total = sum(len(v) for v in rs.values())
        if not 48 <= total <= 66:
            continue
        mname = MONTH_NAME[month]
        svc_list = ", ".join(svcs[:3]) + " and " + svcs[3]
        if qkind == "max_robots":
            allr = [r for v in rs.values() for r in v]
            top = sorted(allr, key=lambda r: -r["robots"])
            if top[0]["robots"] == top[1]["robots"]:
                continue
            q = (f"Consider every incident on the services {svc_list} opened in {mname}. Which single incident "
                 f"had the most robots affected? Give its incident ID.")
            ans, per = top[0]["id"], {s: max(r["robots"] for r in v) for s, v in rs.items()}
        else:
            if qkind == "count_cause":
                cause = R.choice(["config-change", "dependency-outage", "capacity", "bad-deploy"])
                what = f"the most incidents whose root cause category is {cause}"
                fn = lambda v, c=cause: sum(r["cause"] == c for r in v)  # noqa: E731
            else:
                what, fn = STATS[qkind]
            per = {s: fn(v) for s, v in rs.items()}
            vals = sorted(per.values(), reverse=True)
            if vals[0] == vals[1]:
                continue
            best = max(per, key=per.get)
            q = (f"Compare the services {svc_list} for incidents opened in {mname}. Which of these four services "
                 f"had {what}, and what was that service's figure? End with "
                 f"'ANSWER: <service>, <number>'.")
            ans = f"{best}, {per[best]}"
        used.add((tuple(svcs), month))
        return {"id": f"w{i:02d}", "kind": "wide", "qkind": qkind, "size": total, "answer": ans,
                "per_service": per, "groups": [[s, month] for s in svcs],
                "members": [r["id"] for v in rs.values() for r in v], "question": q}


def twostage(i, used):
    while True:
        (s1, m1), (s2, m2) = R.sample([k for k, v in groups.items() if 10 <= len(v) <= 20], 2)
        g1, g2 = groups[(s1, m1)], groups[(s2, m2)]
        top = sorted(g1, key=lambda r: -r["impact_min"])
        if top[0]["impact_min"] == top[1]["impact_min"]:
            continue
        cause = top[0]["cause"]
        n = sum(r["cause"] == cause for r in g2)
        if n == 0:
            continue
        q = (f"Among the incidents on {s1} opened in {MONTH_NAME[m1]}, find the one with the most customer impact "
             f"minutes. How many incidents on {s2} opened in {MONTH_NAME[m2]} have the same root cause category "
             f"as that incident?")
        return {"id": f"d{i:02d}", "kind": "twostage", "qkind": "count_cause", "size": len(g1) + len(g2),
                "answer": n, "stage1": top[0]["id"], "stage1_cause": cause,
                "groups": [[s1, m1], [s2, m2]], "members": [r["id"] for r in g1 + g2], "question": q}


used = set()
tasks = []
for i, qk in enumerate(["sum_impact", "sum_robots", "count_sev12", "count_cause", "max_robots"] * 2, 1):
    tasks.append(wide(i, qk, used))
for i in range(1, 5):
    tasks.append(twostage(i, used))
a05 = {t["id"]: t for t in json.loads((A05 / "tasks.json").read_text())}
for cid in ["c01", "c03", "c05", "c09"]:
    t = dict(a05[cid])
    tasks.append(t)

(HERE / "tasks.json").write_text(json.dumps(tasks, indent=1))
for t in tasks:
    print(t["id"], t["kind"], t["qkind"], t["size"], t["answer"], t.get("per_service", ""))
