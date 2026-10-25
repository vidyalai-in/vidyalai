"""Exp 6b: what tool_choice none returns when the question needs a tool (5 repeats each model)."""
from common import ledger, HAIKU, SONNET, dump
from toollib import TOOLS
cl = ledger.client()
for model in (HAIKU, SONNET):
    for k in range(5 if model == HAIKU else 2):
        r = ledger.create(cl, "a02", note="exp6b none", model=model, max_tokens=1024, tools=TOOLS[:2],
                          tool_choice={"type": "none"}, messages=[{"role": "user", "content": "What's the weather in Paris right now?"}])
        rec = {"model": model, "stop": r.stop_reason, "content": [b.model_dump() for b in r.content], "usage": r.usage.model_dump()}
        dump("exp6b_none.jsonl", rec)
        print(model, r.stop_reason, [b.type for b in r.content], r.usage.output_tokens, [getattr(b, "text", "")[:150] for b in r.content])
print(ledger.report())
