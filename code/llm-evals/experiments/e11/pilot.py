import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, own_eval as oe
cl = ledger.client(); arts = oe.load_articles()
items = [json.loads(l) for l in open("data/candidates_fixed.jsonl")]
pick = [it for it in items if it["part"] == 3][:3] + [it for it in items if it["id"] in ("e03-23", "e03-20")]
f = open("runs/pilot.jsonl", "a")
for cfg in ("off", "low"):
    for it in pick:
        p = oe.answer_request(it, arts[3], cfg)
        r = ledger.create(cl, "e11", note=f"pilot {cfg}", **p)
        text = "".join(c.text for c in r.content if c.type == "text")
        ans, rule = oe.final_answer(text)
        u = r.usage
        print(cfg, it["id"], it["kind"], repr(it["answer"]), "->", repr(ans), rule, oe.grade(it, ans), "| in", u.input_tokens, "cw", u.cache_creation_input_tokens, "cr", u.cache_read_input_tokens, "out", u.output_tokens, r.stop_reason)
        f.write(json.dumps({"cfg": cfg, "id": it["id"], "text": text, "usage": u.model_dump(), "stop": r.stop_reason}) + "\n")
    # closed book
for it in pick[:3]:
    p = oe.answer_request(it, arts[3], "off", closed=True)
    r = ledger.create(cl, "e11", note="pilot closed", **p)
    text = "".join(c.text for c in r.content if c.type == "text"); ans, rule = oe.final_answer(text)
    print("closed", it["id"], repr(it["answer"]), "->", repr(ans), oe.grade(it, ans), r.usage.output_tokens, r.stop_reason)
    f.write(json.dumps({"cfg": "closed", "id": it["id"], "text": text, "usage": r.usage.model_dump(), "stop": r.stop_reason}) + "\n")
