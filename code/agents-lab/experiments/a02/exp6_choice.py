"""Exp 6: which tool_choice values each model accepts (one real call each)."""
import json, time
from common import ledger, HAIKU, SONNET, dump
from toollib import TOOLS

cl = ledger.client()
tools = TOOLS[:2]
msg = [{"role": "user", "content": "What's the weather in Paris right now?"}]
for model in (HAIKU, SONNET):
    for ch in ({"type": "auto"}, {"type": "none"}, {"type": "any"}, {"type": "tool", "name": "get_weather"}):
        t0 = time.time()
        try:
            r = ledger.create(cl, "a02", note=f"exp6 tool_choice {ch['type']}", model=model, max_tokens=1024,
                              tools=tools, tool_choice=ch, messages=msg)
            res = {"stop_reason": r.stop_reason, "blocks": [b.type for b in r.content], "usage": r.usage.model_dump(),
                   "content": [b.model_dump() for b in r.content]}
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        res.update(model=model, choice=ch, secs=round(time.time() - t0, 2))
        dump("exp6_choice.jsonl", res)
        print(model, ch["type"], res.get("error") or (res["stop_reason"], res["blocks"], res["usage"]["input_tokens"], res["usage"]["output_tokens"]))
