"""Pick replies for hand-checking the grader: every reply the strict grader marks wrong, every reply
where strict and lenient disagree, and a random 60 of the rest (seed 11)."""
import sys, json, random
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import own_eval as oe
items = {json.loads(l)["id"]: json.loads(l) for l in open("data/items.jsonl")}
tags = sys.argv[1:]
rows = []
for tag in tags:
    for l in open(f"runs/{tag}.jsonl"):
        r = json.loads(l); rep, iid = r["id"].split("_", 1)
        ans, rule = oe.final_answer(r.get("text"))
        rows.append(dict(key=f"{tag}/{r['id']}", tag=tag, id=iid, ans=ans, rule=rule, text=r.get("text"),
                         strict=oe.grade(items[iid], ans), lenient=oe.grade_lenient(items[iid], ans)))
flag = [x for x in rows if x["strict"] == 0 or x["strict"] != x["lenient"]]
rest = [x for x in rows if x not in flag]
random.seed(11); pick = flag + random.sample(rest, min(60, len(rest)))
json.dump([x["key"] for x in pick], open(f"notes/grader_sample_{'_'.join(tags)}.json", "w"))
print(len(rows), "replies;", len(flag), "flagged;", len(pick), "to check")
for x in pick:
    it = items[x["id"]]
    t = (x["text"] or "").replace("\n", " / ")
    print(f"\n{x['key']} [{it['kind']}] KEY={it['answer']!r} {it['aliases']} strict={x['strict']} len={x['lenient']} rule={x['rule']}\n  Q: {it['question'][:160]}\n  R: {t[-260:]}")
