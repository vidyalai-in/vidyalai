"""Run a model on MMLU-Redux 2.0 items through the Message Batches API (thinking disabled).

usage: run_batch.py MODEL K OUTNAME [subset.json]
Every request goes in one batch; ledger.guard before submitting, ledger.record for every result.
Requests, responses and usage are saved to e02/runs/OUTNAME.jsonl.
"""
import json, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, items

model, K, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
data = items.load_redux()
idx = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else list(range(len(data)))
MAX_TOKENS = 8
reqs = []
for i in idx:
    THINK_OFF = {"type": "between_tools"} if model == "claude-sonnet-5-5" else {"type": "disabled"}  # Sonnet 5.5 rejects "disabled"
    params = dict(model=model, max_tokens=MAX_TOKENS, thinking=THINK_OFF,
                  messages=[{"role": "user", "content": items.mcq_prompt(data[i])}])
    for k in range(K):
        reqs.append({"custom_id": f"i{i:04d}-k{k}", "params": params})
# Worst case: every input token billed as a cache write, max_tokens out, half price.
# Input tokens bounded by characters / 2 of the whole params JSON.
est = sum(ledger.estimate_cost(model, len(json.dumps(r["params"])) // 2, MAX_TOKENS) for r in reqs) * 0.5
if model == "claude-sonnet-5-5":
    # Counted instead: Haiku's billed input tokens for the same prompts, x1.2 (count_tokens on 20
    # prompts gave Sonnet at most 1.022x Haiku's count).
    hin = {}
    for line in open("/home/amit/.cache/vidyalai-scratch/e02/runs/haiku-k3.jsonl"):
        r = json.loads(line); hin[r["custom_id"][:5]] = r["usage"]["input_tokens"]
    est = sum(ledger.estimate_cost(model, int(hin[r["custom_id"][:5]] * 1.2), MAX_TOKENS) for r in reqs) * 0.5
print(f"{len(reqs)} requests, guard estimate ${est:.4f}", flush=True)
ledger.guard("e02", est)
cl = ledger.client()
b = cl.messages.batches.create(requests=reqs)
print("batch", b.id, flush=True)
t0 = time.time()
while True:
    time.sleep(20)
    b = cl.messages.batches.retrieve(b.id)
    print(f"{time.time()-t0:.0f}s", b.processing_status, b.request_counts.model_dump(), flush=True)
    if b.processing_status == "ended":
        break
    if time.time() - t0 > 3600:
        print("still running after 60 min"); sys.exit(2)
by_id = {r["custom_id"]: r for r in reqs}
cost, n_ok = 0.0, 0
with open(f"/home/amit/.cache/vidyalai-scratch/e02/runs/{name}.jsonl", "w") as f:
    for it in cl.messages.batches.results(b.id):
        rec = dict(custom_id=it.custom_id, result_type=it.result.type, request=by_id[it.custom_id]["params"])
        if it.result.type == "succeeded":
            m = it.result.message
            cost += ledger.record("e02", model, m.usage, note=f"e02 {name}", batch=True)
            n_ok += 1
            rec.update(text="".join(c.text for c in m.content if c.type == "text"),
                       stop=m.stop_reason, usage=m.usage.model_dump())
        else:
            rec.update(error=it.result.model_dump(exclude_none=True))
        f.write(json.dumps(rec) + "\n")
print(f"done: {n_ok}/{len(reqs)} succeeded, cost ${cost:.4f}, {time.time()-t0:.0f}s", flush=True)
