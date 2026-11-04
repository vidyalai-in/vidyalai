"""Re-download the thinking batch's results with every content block (thinking summaries included). No new cost: results were
already recorded in the ledger by grading.collect; this only reads them again."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger
cl = ledger.client()
bid = json.load(open("/home/amit/.cache/vidyalai-scratch/e10/runs/sandbag_think.jsonl.batch"))["batch"]
with open("/home/amit/.cache/vidyalai-scratch/e10/runs/sandbag_think_full.jsonl", "w") as f:
    for it in cl.messages.batches.results(bid):
        rec = {"id": it.custom_id, "type": it.result.type}
        if it.result.type == "succeeded":
            m = it.result.message
            rec.update(stop=m.stop_reason, usage=m.usage.model_dump(), content=[b.model_dump() for b in m.content])
        f.write(json.dumps(rec) + "\n")
print("ok")
