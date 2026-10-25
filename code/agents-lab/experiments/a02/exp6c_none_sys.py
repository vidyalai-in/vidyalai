"""Exp 6c: tool_choice none plus a system prompt that asks for a text answer."""
from common import ledger, HAIKU, dump
from toollib import TOOLS
cl = ledger.client()
SYS = "Tools are unavailable for this turn. Reply in text: say what you would need to look up."
for k in range(3):
    r = ledger.create(cl, "a02", note="exp6c none+system", model=HAIKU, max_tokens=1024, tools=TOOLS[:2], system=SYS,
                      tool_choice={"type": "none"}, messages=[{"role": "user", "content": "What's the weather in Paris right now?"}])
    dump("exp6c_none_sys.jsonl", {"stop": r.stop_reason, "content": [b.model_dump() for b in r.content], "usage": r.usage.model_dump()})
    print(r.stop_reason, [b.type for b in r.content], r.usage.output_tokens, [getattr(b, "text", "")[:200] for b in r.content])
