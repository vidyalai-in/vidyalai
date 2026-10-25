"""Part 7 stress set: 8 wide questions over 12 of the 16 services in one month (about 160-200 records each).

python tasks_stress.py -> writes tasks_stress.json next to this file.
"""
import json
import random
from pathlib import Path

from tasks import INC, MONTH_NAME, groups

HERE = Path(__file__).parent
R = random.Random(7070)
out = []
kinds = ["sum_impact", "max_robots", "sum_robots", "count_cause"] * 2
for i, qk in enumerate(kinds, 1):
    while True:
        month = list(MONTH_NAME)[(i - 1) % 4]
        svcs = sorted(R.sample(sorted({s for s, m in groups if m == month}), 12))
        rs = {s: groups[(s, month)] for s in svcs}
        allr = [r for v in rs.values() for r in v]
        svc_list = ", ".join(svcs[:-1]) + " and " + svcs[-1]
        if qk == "max_robots":
            top = sorted(allr, key=lambda r: -r["robots"])
            if top[0]["robots"] == top[1]["robots"]:
                continue
            q = (f"Consider every incident on the services {svc_list} opened in {MONTH_NAME[month]}. Which single "
                 f"incident had the most robots affected? Give its incident ID.")
            ans, per = top[0]["id"], {s: max(r["robots"] for r in v) for s, v in rs.items()}
        else:
            if qk == "count_cause":
                cause = R.choice(["config-change", "dependency-outage", "capacity", "bad-deploy"])
                what = f"the most incidents whose root cause category is {cause}"
                per = {s: sum(r["cause"] == cause for r in v) for s, v in rs.items()}
            elif qk == "sum_impact":
                what = "the highest total customer impact minutes"
                per = {s: sum(r["impact_min"] for r in v) for s, v in rs.items()}
            else:
                what = "the highest total number of robots affected"
                per = {s: sum(r["robots"] for r in v) for s, v in rs.items()}
            vals = sorted(per.values(), reverse=True)
            if vals[0] == vals[1]:
                continue
            best = max(per, key=per.get)
            q = (f"Compare the services {svc_list} for incidents opened in {MONTH_NAME[month]}. Which of these twelve "
                 f"services had {what}, and what was that service's figure? End with 'ANSWER: <service>, <number>'.")
            ans = f"{best}, {per[best]}"
        out.append({"id": f"s{i:02d}", "kind": "wide12", "qkind": qk, "size": len(allr), "answer": ans,
                    "per_service": per, "groups": [[s, month] for s in svcs], "members": [r["id"] for r in allr],
                    "question": q})
        break
(HERE / "tasks_stress.json").write_text(json.dumps(out, indent=1))
for t in out:
    print(t["id"], t["qkind"], t["size"], t["answer"], sorted(t["per_service"].values(), reverse=True)[:3])
