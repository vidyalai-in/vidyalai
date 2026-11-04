"""Draft candidate items for each article with Haiku 5.5 (single calls through the ledger)."""
import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, own_eval as oe
parts = [int(x) for x in sys.argv[1:]]
arts = oe.load_articles()
cl = ledger.client()
for n in parts:
    a = arts[n]
    r = ledger.create(cl, "e11", note=f"draft items part {n}", **oe.draft_request(a))
    text = "".join(c.text for c in r.content if c.type == "text")
    with open("runs/drafts.jsonl", "a") as f:
        f.write(json.dumps({"part": n, "stop": r.stop_reason, "usage": r.usage.model_dump(), "text": text}) + "\n")
    try:
        items = oe.parse_json_array(text)
    except Exception as e:
        print(n, "PARSE FAIL", e); continue
    bad = {i: oe.check_draft(it, a) for i, it in enumerate(items)}
    print(n, r.stop_reason, r.usage.output_tokens, len(items), "kinds", {k: sum(it["kind"] == k for it in items) for k in ("number","text","none")},
          "fail", sum(bool(v) for v in bad.values()))
    for i, v in bad.items():
        if v: print("   ", i, v, items[i]["question"][:80], "|", items[i]["answer"])
