import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); sys.path.insert(0, "snippet")
import own_eval as oe, grader as g
items = {json.loads(l)["id"]: json.loads(l) for l in open("data/items.jsonl")}
n = d = 0
for tag in ("main_off", "main_low", "closed"):
    for l in open(f"runs/{tag}.jsonl"):
        r = json.loads(l); iid = r["id"].split("_", 1)[1]; it = items[iid]
        a = oe.grade(it, oe.final_answer(r.get("text"))[0]); b = g.grade(it, r.get("text") or "")
        n += 1
        if a != b: d += 1; print(tag, r["id"], a, b, repr(oe.final_answer(r.get("text"))[0]))
print(n, "replies; snippet differs on", d)
