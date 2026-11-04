import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals")
import ledger
from lab import safety_evals as S
from lab.validity import load
cl = ledger.client()
items = load("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json")
out = open("/home/amit/.cache/vidyalai-scratch/e10/runs/pilot.jsonl", "a")
for c in ("high_penalized",):
    p = S.mmlu_request(items[40], c, max_tokens=1024, thinking={"type": "adaptive", "display": "summarized"}); p["output_config"] = {"effort": "low"}
    r = ledger.create(cl, "e10", note=f"pilot think {c}", **p)
    blocks = [b.model_dump() for b in r.content]
    out.write(json.dumps({"note": f"pilot think {c}", "params": p, "content": blocks, "usage": r.usage.model_dump()}) + "\n")
    print(c, r.stop_reason, r.usage.output_tokens, [ (b['type'], (b.get('thinking') or b.get('text') or '')[:300]) for b in blocks])
