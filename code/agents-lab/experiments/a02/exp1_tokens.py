"""Exp 1: input-token overhead of tools, via the free count_tokens endpoint."""
import json, copy
from common import ledger, HAIKU, SONNET, dump
from toollib import TOOLS

cl = ledger.client()
MSG = [{"role": "user", "content": "What's the weather in Paris?"}]

def count(model, tools=None, **kw):
    args = dict(model=model, messages=MSG, **kw)
    if tools is not None:
        args["tools"] = tools
    return cl.messages.count_tokens(**args).input_tokens

out = {}
for model in (HAIKU, SONNET):
    r = {"no_tools": count(model)}
    for n in (1, 2, 5, 10, 20):
        r[f"n{n}"] = count(model, TOOLS[:n])
    r["minimal_tool"] = count(model, [{"name": "t", "input_schema": {"type": "object"}}])
    r["n1_choice_none"] = count(model, TOOLS[:1], tool_choice={"type": "none"})
    r["n1_strict"] = count(model, [dict(TOOLS[0], strict=True, input_schema=dict(TOOLS[0]["input_schema"], additionalProperties=False))])
    r["n1_nostrict_addlfalse"] = count(model, [dict(TOOLS[0], input_schema=dict(TOOLS[0]["input_schema"], additionalProperties=False))])
    for ch in ({"type": "any"}, {"type": "tool", "name": "get_weather"}):
        try:
            r["n1_choice_" + ch["type"]] = count(model, TOOLS[:1], tool_choice=ch)
        except Exception as e:
            r["n1_choice_" + ch["type"]] = f"ERR {type(e).__name__}: {str(e)[:200]}"
    # each tool on its own, to get per-tool definition cost
    r["per_tool"] = {x["name"]: count(model, [x]) - r["minimal_tool"] for x in TOOLS}
    # the tool definition as JSON text, counted as a plain user message
    r["n20_as_text"] = cl.messages.count_tokens(model=model, messages=[{"role": "user", "content": json.dumps(TOOLS)}]).input_tokens
    out[model] = r
    print(model, json.dumps({k: v for k, v in r.items() if k != "per_tool"}, indent=1))
    pt = list(r["per_tool"].values()); print("per-tool (vs minimal tool) min/mean/max", min(pt), sum(pt)/len(pt), max(pt))
dump("exp1_tokens.jsonl", out)
