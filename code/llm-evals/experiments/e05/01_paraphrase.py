"""Haiku 5.5 rewrites every ARC-Challenge test question (same meaning, new words, no answer).
Used twice: as paraphrase contamination in the training run, and as a fresh control text.
  python 01_paraphrase.py pilot   # 5 single calls
  python 01_paraphrase.py batch   # all 1,172 items in one Message Batch
"""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import contamination as C

ITEMS = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e01-arc-challenge.json"))
RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
SYSTEM = ("You rewrite exam questions. Keep the meaning and every fact needed to answer, but use "
          "different words and sentence structure wherever you can. Do not answer the question, do "
          "not list answer options, and do not add anything. Reply with the rewritten question only.")


def params(it):
    return {"model": "claude-haiku-5-5", "max_tokens": 300, "system": SYSTEM,
            "thinking": C.THINK_OFF["claude-haiku-5-5"],
            "messages": [{"role": "user", "content": it["question"]}]}


if sys.argv[1] == "pilot":
    import ledger
    cl = ledger.client()
    with open(f"{RUNS}/paraphrase-pilot.jsonl", "w") as f:
        for it in ITEMS[:5]:
            r = ledger.create(cl, "e05", note="e05 paraphrase pilot", **params(it))
            t = "".join(c.text for c in r.content if c.type == "text")
            f.write(json.dumps({"id": it["id"], "text": t, "usage": r.usage.model_dump(), "stop": r.stop_reason}) + "\n")
            print(it["question"], "\n ->", t, r.usage.input_tokens, r.usage.output_tokens, "\n")
else:
    jobs = {f"p{i:04d}": params(it) for i, it in enumerate(ITEMS)}
    print(C.run_batch("e05", jobs, f"{RUNS}/paraphrase.jsonl", "e05 paraphrase ARC-Challenge"))
