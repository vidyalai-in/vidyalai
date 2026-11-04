"""Submit Part 4's Claude conditions as one Message Batch; save requests, replies and usage."""
import json, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/e04/scripts")
import ledger
from haiku_common import ITEMS, CONDS, params

model = sys.argv[1]
conds = sys.argv[2].split(",")
idx = range(len(ITEMS)) if len(sys.argv) < 4 else [i for i in range(len(ITEMS)) if i % int(sys.argv[3]) == 0]
tag = sys.argv[4] if len(sys.argv) > 4 else model
out = f"/home/amit/.cache/vidyalai-scratch/e04/runs/batch-{tag}.jsonl"
reqs, meta = [], {}
for c in conds:
    for i in idx:
        p, it = params(c, ITEMS[i], model)
        if model == "claude-sonnet-5-5" and c == "raw":
            p["max_tokens"] = 512  # keeps the worst-case estimate inside the sub-cap
        cid = f"{c}-{i}"
        reqs.append({"custom_id": cid, "params": p})
        meta[cid] = dict(cond=c, i=i, answer=it["answer"], choices=it["choices"], params=p)
est = sum(ledger.estimate_cost(model, len(json.dumps(r["params"])) // 2, r["params"]["max_tokens"]) for r in reqs) * 0.5
print(len(reqs), "requests, worst case $%.4f" % est, flush=True)
ledger.guard("e04", est)
cl = ledger.client()
b = cl.messages.batches.create(requests=reqs)
print("batch", b.id, flush=True)
t0 = time.time()
while True:
    time.sleep(20)
    b = cl.messages.batches.retrieve(b.id)
    print(round(time.time() - t0), b.processing_status, b.request_counts.model_dump(), flush=True)
    if b.processing_status == "ended":
        break
cost = 0.0
with open(out, "w") as f:
    for item in cl.messages.batches.results(b.id):
        m = meta[item.custom_id]
        rec = dict(id=item.custom_id, **m, result=item.result.type)
        if item.result.type == "succeeded":
            msg = item.result.message
            cost += ledger.record("e04", model, msg.usage, note=f"e04 batch {tag}", batch=True)
            rec.update(text="".join(c.text for c in msg.content if c.type == "text"),
                       types=[c.type for c in msg.content], stop=msg.stop_reason, usage=msg.usage.model_dump())
        f.write(json.dumps(rec) + "\n")
print("done, cost $%.4f, %.0f s" % (cost, time.time() - t0), flush=True)
