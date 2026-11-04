import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, validity as V
D = "/home/amit/.cache/vidyalai-scratch/evals/data"
cl = ledger.client()
m = V.load(f"{D}/mmlu-1140.json")
for i, kind in []:
    p = V.request(m[i], kind, "claude-haiku-5-5", max_tokens=8)
    print(p["messages"][0]["content"][:400])
    r = ledger.create(cl, "e01", note="e01 pilot2", **p)
    print(r.stop_reason, [c.model_dump() for c in r.content], r.usage.output_tokens)
for i, kind, model in [(40, "choices_only", "claude-sonnet-5-5"), (12, "choices_only", "claude-haiku-5-5"), (12, "full", "claude-sonnet-5-5")]:
    p = V.request(m[i], kind, model)
    r = ledger.create(cl, "e01", note="e01 pilot2", **p)
    print(model, r.stop_reason, [c.type for c in r.content], repr("".join(c.text for c in r.content if c.type=="text")), m[i]["answer"])
