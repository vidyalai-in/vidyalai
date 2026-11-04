"""List validation-pass disagreements for hand checking, plus a random 20% of agreed items."""
import sys, json, random
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import own_eval as oe
items = {json.loads(l)["id"]: json.loads(l) for l in open("data/candidates_fixed.jsonl")}
res = {}
for l in open("runs/val.jsonl"):
    r = json.loads(l); rep, iid = r["id"].split("_", 1); res.setdefault(iid, {})[rep] = r
dis, agree = [], []
for iid, it in items.items():
    rr = res.get(iid, {})
    row = []
    for rep in ("r1", "r2"):
        r = rr.get(rep, {}); ans, rule = oe.final_answer(r.get("text"))
        row.append((ans, rule, oe.grade(it, ans), oe.grade_lenient(it, ans), r.get("stop")))
    if all(x[2] == 1 and x[3] == 1 for x in row): agree.append(iid)
    else: dis.append((iid, row))
random.seed(11); sample = sorted(random.sample(agree, round(0.2 * len(agree))))
print(len(items), "items;", len(dis), "disputed;", len(agree), "agreed; sample", len(sample))
json.dump(dict(disputed=[d[0] for d in dis], sample=sample), open("notes/review_sets.json", "w"))
for iid, row in dis:
    it = items[iid]
    print(f"\n## {iid} [{it['kind']}] Q: {it['question']}\n   KEY: {it['answer']} {it['aliases']}\n   EV: {it['evidence'][:220]}")
    for (ans, rule, g, gl, stop) in row: print(f"   -> {ans!r} ({rule}, strict={g}, lenient={gl}, {stop})")
print("\n#### SAMPLE OF AGREED")
for iid in sample:
    it = items[iid]; ans = oe.final_answer(res[iid]["r1"].get("text"))[0]
    print(f"{iid} [{it['kind']}] Q: {it['question']} | KEY: {it['answer']} | got {ans!r}")
