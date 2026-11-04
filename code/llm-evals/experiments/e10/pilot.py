import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals")
import ledger
from lab import safety_evals as S
from lab.validity import load
cl = ledger.client()
xs = S.load_xstest(); safe = [r for r in xs if r["label"] == "safe"]
out = open("/home/amit/.cache/vidyalai-scratch/e10/runs/pilot.jsonl", "a")
def call(p, note):
    r = ledger.create(cl, "e10", note=note, **p)
    t = "".join(c.text for c in r.content if c.type == "text")
    out.write(json.dumps({"note": note, "params": p, "text": t, "stop": r.stop_reason, "usage": r.usage.model_dump()}) + "\n")
    print(note, r.stop_reason, r.usage.input_tokens, r.usage.output_tokens, repr(t[:150])); return t
reps = [call(S.reply_request(safe[i]["prompt"]), "pilot xstest") for i in (0, 60, 130)]
call(S.judge_request(safe[0]["prompt"], reps[0]), "pilot judge")
call(S.strongreject_request(safe[0]["prompt"], reps[0]), "pilot strongreject")
items = load("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json")
for c in ("neutral", "high_penalized"):
    call(S.mmlu_request(items[3], c), f"pilot mmlu {c}")
