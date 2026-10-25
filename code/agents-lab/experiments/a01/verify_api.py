"""Cheap checks of API behaviour cited in Part 1 (all calls through ledger.create)."""
import sys; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, anthropic
cl = ledger.client()
tool = {"name": "get_time", "description": "Return the current time.", "input_schema": {"type": "object", "properties": {}}}
msg = [{"role": "user", "content": "What time is it?"}]
def tryit(label, **kw):
    try:
        r = ledger.create(cl, "a01", note=f"verify: {label}", max_tokens=200, messages=msg, **kw)
        print(f"{label}: OK stop={r.stop_reason} blocks={[b.type for b in r.content]} usage_in={r.usage.input_tokens}")
    except anthropic.BadRequestError as e:
        print(f"{label}: 400 -> {e.body.get('error', {}).get('message') if isinstance(e.body, dict) else e}")
for m in ("claude-haiku-5-5", "claude-sonnet-5-5"):
    tryit(f"{m} tool_choice=any", model=m, tools=[tool], tool_choice={"type": "any"})
    tryit(f"{m} temperature=0.5", model=m, extra_body={"temperature": 0.5})
tryit("haiku no tools baseline", model="claude-haiku-5-5")
tryit("haiku tools auto", model="claude-haiku-5-5", tools=[tool])
