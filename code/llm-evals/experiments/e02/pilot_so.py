"""Pilot: re-ask 3 unparsed items with the reply constrained to one letter (structured output)."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, items
FMT = {"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": list("ABCD")}},
       "required": ["answer"], "additionalProperties": False}}
data = items.load_redux(); cl = ledger.client()
out = open("runs/pilot-so.jsonl", "w")
for i in (67, 677, 2549):
    params = dict(model="claude-haiku-5-5", max_tokens=16, thinking={"type": "disabled"}, output_config={"format": FMT},
                  messages=[{"role": "user", "content": items.mcq_prompt(data[i])}])
    r = ledger.create(cl, "e02", note="e02 pilot structured", **params)
    t = "".join(b.text for b in r.content if b.type == "text")
    out.write(json.dumps(dict(i=i, text=t, usage=r.usage.model_dump())) + "\n")
    print(i, data[i]["error_type"], items.LETTERS[data[i]["answer"]], repr(t), r.usage.input_tokens, r.usage.output_tokens, r.stop_reason)
