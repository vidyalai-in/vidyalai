"""Compare grader v1 (Answer: at line start, key number only) with v2 on every reply."""
import sys, json, copy
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import own_eval as oe
items = {json.loads(l)["id"]: json.loads(l) for l in open("data/items.jsonl")}
for tag in sys.argv[1:]:
    n = d = 0; diffs = []
    for l in open(f"runs/{tag}.jsonl"):
        r = json.loads(l); rep, iid = r["id"].split("_", 1)
        it = items[iid]; it1 = {k: v for k, v in it.items() if k != "alt_numbers"}
        g1 = oe.grade_v1(it1, oe.final_answer_v1(r.get("text"))[0]); g2 = oe.grade(it, oe.final_answer(r.get("text"))[0])
        n += 1
        if g1 != g2: d += 1; diffs.append((r["id"], g1, g2, oe.final_answer(r.get("text"))[0]))
    print(tag, n, "replies; v1 != v2 on", d, diffs)
