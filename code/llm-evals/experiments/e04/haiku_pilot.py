import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/e04/scripts")
import ledger
from haiku_common import ITEMS, params
cl = ledger.client()
out = open("/home/amit/.cache/vidyalai-scratch/e04/runs/haiku-pilot3.jsonl", "w")
for cond, i in [("raw", 0), ("raw", 300), ("raw5", 0), ("raw5", 500), ("mmlu5", 500), ("mmlu5", 0)]:
    p, it = params(cond, ITEMS[i])
    r = ledger.create(cl, "e04", note=f"e04 pilot {cond}", **p)
    txt = "".join(b.text for b in r.content if b.type == "text")
    rec = dict(cond=cond, i=i, params=p, answer=it["answer"], text=txt, stop=r.stop_reason, usage=r.usage.model_dump())
    out.write(json.dumps(rec) + "\n")
    print(cond, i, it["answer"], repr(txt), r.usage.input_tokens, r.usage.output_tokens, r.stop_reason)
