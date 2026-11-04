"""Collect an ended batch whose polling process was killed: check its custom_ids are the
expected rerun jobs, record each succeeded result in the ledger once, write the JSONL."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, validity as V
R = "/home/amit/.cache/vidyalai-scratch/e01/runs"
name, batch_id, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
jobs = {}
for l in open(f"{R}/{name}.jsonl"):
    r = json.loads(l)
    if r["type"] == "succeeded" and r.get("stop") == "max_tokens" and V.extract(r["text"], 10) is None:
        jobs[r["id"]] = {**r["params"], "max_tokens": cap}
cl = ledger.client()
items = list(cl.messages.batches.results(batch_id))
assert {it.custom_id for it in items} == set(jobs), "batch does not match the expected jobs"
assert not any(json.loads(l)["note"] == f"e01 {name} rerun {cap}" for l in open(ledger.LEDGER)), "already recorded"
cost = 0.0
with open(f"{R}/{name}-rerun.jsonl", "a") as f:
    for it in items:
        p = jobs[it.custom_id]
        rec = {"id": it.custom_id, "batch": batch_id, "params": p, "type": it.result.type}
        if it.result.type == "succeeded":
            m = it.result.message
            cost += ledger.record("e01", p["model"], m.usage, note=f"e01 {name} rerun {cap}", batch=True)
            rec.update(text="".join(c.text for c in m.content if c.type == "text"), stop=m.stop_reason,
                       usage=m.usage.model_dump())
        f.write(json.dumps(rec) + "\n")
print({"batch": batch_id, "n": len(items), "cost": cost})
