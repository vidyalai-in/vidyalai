"""Pilot: 8 single calls to Haiku 5.5 (thinking off) on Redux items, to size the batch run."""
import json, sys, random
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, items

model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
data = items.load_redux()
random.seed(1)
pick = random.sample(data, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
cl = ledger.client()
out = open(f"/home/amit/.cache/vidyalai-scratch/e02/runs/pilot-{model}.jsonl", "w")
cost = 0
for it in pick:
    params = dict(model=model, max_tokens=8, thinking={"type": "between_tools"} if "sonnet" in model else {"type": "disabled"}, messages=[{"role": "user", "content": items.mcq_prompt(it)}])
    r = ledger.create(cl, "e02", note="e02 pilot", **params)
    text = "".join(b.text for b in r.content if b.type == "text")
    c = ledger.cost_of(model, r.usage)
    cost += c
    out.write(json.dumps(dict(id=it["id"], params=params, text=text, stop=r.stop_reason, usage=r.usage.model_dump(), cost=c)) + "\n")
    print(it["id"], it["error_type"], "gold", items.LETTERS[it["answer"]], "->", repr(text), r.usage.input_tokens, r.usage.output_tokens, r.stop_reason)
print("cost per item (sync)", cost / len(pick))
