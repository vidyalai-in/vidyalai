import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import own_eval as oe
arts = oe.load_articles()
drafts = {}
for l in open("runs/drafts.jsonl"):
    r = json.loads(l); drafts[r["part"]] = r  # last draft per part wins
out = []
for n in sorted(drafts):
    for i, it in enumerate(oe.parse_json_array(drafts[n]["text"])):
        it = dict(id=f"e{n:02d}-{i+1:02d}", part=n, kind=it["kind"], question=it["question"], answer=it["answer"],
                  aliases=it.get("aliases") or [], evidence=it.get("evidence", ""), canary=oe.CANARY)
        it["auto_problems"] = oe.check_draft(it, arts[n])
        out.append(it)
with open("data/candidates.jsonl", "w") as f:
    for it in out: f.write(json.dumps(it) + "\n")
print(len(out), "candidates;", sum(bool(it["auto_problems"]) for it in out), "flagged;",
      {k: sum(it["kind"] == k for it in out) for k in ("number", "text", "none")})
for it in out:
    if it["auto_problems"]:
        print(it["id"], it["auto_problems"], "\n  Q:", it["question"], "\n  A:", it["answer"], it["aliases"], "\n  E:", it["evidence"][:300])
