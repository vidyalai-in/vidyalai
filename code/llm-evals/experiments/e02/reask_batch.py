"""Re-ask every sample whose reply had no leading letter (it started working the problem out and hit
max_tokens), with the same prompt and the reply constrained to {"answer": "A"|"B"|"C"|"D"}.
usage: reask_batch.py RUNNAME   (reads runs/RUNNAME.jsonl, writes runs/RUNNAME-reask.jsonl)"""
import json, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, items
FMT = {"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": list("ABCD")}},
       "required": ["answer"], "additionalProperties": False}}
name = sys.argv[1]
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None   # restrict to these Redux error types
data = items.load_redux()
reqs = []
for line in open(f"runs/{name}.jsonl"):
    r = json.loads(line)
    if only and data[int(r["custom_id"][1:5])]["error_type"] not in only:
        continue
    if r["result_type"] == "succeeded" and items.parse_letter(r["text"]) is None:
        p = dict(r["request"]); p["max_tokens"] = 16; p["output_config"] = {"format": FMT}
        reqs.append({"custom_id": r["custom_id"], "params": p})
model = reqs[0]["params"]["model"]
# structured output adds a system prompt (~300 tokens measured in the pilot): add 400 to the bound
est = sum(ledger.estimate_cost(model, len(json.dumps(r["params"])) // 2 + 400, 16) for r in reqs) * 0.5
if model == "claude-sonnet-5-5":
    # Counted instead of chars/2: Haiku's billed input for the same prompt plus the largest structured-output
    # overhead seen in Haiku's re-asks, x1.1 (Sonnet counts were within 1.022x of Haiku's on 20 prompts).
    main = {}; over = 0
    for line in open("runs/haiku-k3.jsonl"):
        r = json.loads(line); main[r["custom_id"]] = r["usage"]["input_tokens"]
    for line in open("runs/haiku-k3-reask.jsonl"):
        r = json.loads(line); over = max(over, r["usage"]["input_tokens"] - main[r["custom_id"]])
    est = sum(ledger.estimate_cost(model, int((main[r["custom_id"][:5] + "-k0"] + over) * 1.1), 16) for r in reqs) * 0.5
    print("schema overhead", over)
print(f"{len(reqs)} requests, guard estimate ${est:.4f}", flush=True)
ledger.guard("e02", est)
cl = ledger.client()
b = cl.messages.batches.create(requests=reqs); print("batch", b.id, flush=True)
t0 = time.time()
while True:
    time.sleep(20)
    b = cl.messages.batches.retrieve(b.id)
    print(f"{time.time()-t0:.0f}s", b.processing_status, b.request_counts.model_dump(), flush=True)
    if b.processing_status == "ended": break
    if time.time() - t0 > 3600: print("still running after 60 min"); sys.exit(2)
by = {r["custom_id"]: r for r in reqs}; cost = 0
with open(f"runs/{name}-reask.jsonl", "w") as f:
    for it in cl.messages.batches.results(b.id):
        rec = dict(custom_id=it.custom_id, result_type=it.result.type, request=by[it.custom_id]["params"])
        if it.result.type == "succeeded":
            m = it.result.message
            cost += ledger.record("e02", model, m.usage, note=f"e02 {name} reask", batch=True)
            rec.update(text="".join(c.text for c in m.content if c.type == "text"), stop=m.stop_reason, usage=m.usage.model_dump())
        f.write(json.dumps(rec) + "\n")
print(f"done: cost ${cost:.4f}, {time.time()-t0:.0f}s", flush=True)
