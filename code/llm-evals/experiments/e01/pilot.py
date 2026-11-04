import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, validity as V
D = "/home/amit/.cache/vidyalai-scratch/evals/data"
cl = ledger.client()
sets = {"arc-e": V.load(f"{D}/e01-arc-easy.json"), "mmlu": V.load(f"{D}/mmlu-1140.json"),
        "pro": V.load(f"{D}/e01-mmlu-pro-840.json")}
plan = [("arc-e", 0, "full", "claude-haiku-5-5"), ("mmlu", 5, "full", "claude-haiku-5-5"),
        ("pro", 3, "full", "claude-haiku-5-5"), ("mmlu", 5, "choices_only", "claude-haiku-5-5"),
        ("mmlu", 40, "choices_only", "claude-haiku-5-5"), ("mmlu", 7, "nota_correct", "claude-haiku-5-5"),
        ("pro", 3, "full", "claude-sonnet-5-5"), ("mmlu", 40, "choices_only", "claude-sonnet-5-5")]
with open("/home/amit/.cache/vidyalai-scratch/e01/runs/pilot.jsonl", "a") as f:
    for s, i, kind, model in plan:
        it = sets[s][i]
        p = V.request(it, kind, model)
        r = ledger.create(cl, "e01", note="e01 pilot", **p)
        text = "".join(c.text for c in r.content if c.type == "text")
        pred = V.extract(text, len(it["choices"]))
        print(s, i, kind, model, repr(text), pred, it["answer"], r.usage.input_tokens, r.usage.output_tokens,
              round(ledger.cost_of(model, r.usage), 6))
        f.write(json.dumps({"set": s, "i": i, "kind": kind, "params": p, "text": text, "usage": r.usage.model_dump()}) + "\n")
