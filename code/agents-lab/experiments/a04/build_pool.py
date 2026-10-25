"""Collect every wrong first attempt made at the main setting (Haiku 5.5, medium, adaptive thinking)."""
import json, sys, glob
sys.path.insert(0, ".")
import reflect as R
T = {t["id"]: t for t in R.TASKS}
pool = []
for runf in ["calib2-medium", "calib4-medium", "calib5-medium", "pilot2", "pool-singles", "main"]:
    replies = {}
    for l in open(f"runs/calls-{runf}.jsonl"):
        c = json.loads(l)
        if c["note"] == "attempt 1":
            replies[c["tag"]] = c
    for tag, c in replies.items():
        task = tag.split("#")[0]
        if task not in T or c["stop"] != "end_turn":
            continue
        code = R.extract_code(c["reply"], task)
        ok, fails = R.grade(T[task], code)
        if not ok:
            pub, _ = R.grade(T[task], code, public_only=True)
            pool.append(dict(source=runf, tag=f"{runf}:{tag}", task=task, reply=c["reply"], public_ok=pub, fails=fails[:3]))
            print(runf, tag, "public_ok" if pub else "public_fail", fails[0][:150])
json.dump(pool, open("runs/pool.json", "w"))
print(len(pool), "wrong attempts")
