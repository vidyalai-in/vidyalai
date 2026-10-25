"""Exp 3b: how strict mode handles a required nullable field, three ways of writing it."""
import json, time
from concurrent.futures import ThreadPoolExecutor
from common import ledger, HAIKU, dump

cl = ledger.client()
VARIANTS = {
    "type_array_with_format": {"type": ["string", "null"], "format": "email"},
    "type_array_no_format": {"type": ["string", "null"]},
    "anyOf_email_or_null": {"anyOf": [{"type": "string", "format": "email"}, {"type": "null"}]},
}
PROMPT = "Minor: the logout button on the web app is misaligned by 2px. Nobody is assigned yet."
SYSTEM = "Act on each request by calling exactly one tool. Do not ask follow-up questions."

def tool(v, strict):
    t = {"name": "file_bug", "description": "File a bug. assignee_email is null when nobody is assigned.",
         "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "assignee_email": VARIANTS[v]},
                          "required": ["title", "assignee_email"], "additionalProperties": False}}
    if strict: t["strict"] = True
    return t

def run(job):
    v, strict, rep = job
    r = ledger.create(cl, "a02", note=f"exp3b {v} strict={strict}", model=HAIKU, max_tokens=2048, system=SYSTEM,
                      tools=[tool(v, strict)], messages=[{"role": "user", "content": PROMPT}])
    inp = next((b.input for b in r.content if b.type == "tool_use"), None)
    rec = {"variant": v, "strict": strict, "rep": rep, "input": inp, "usage": r.usage.model_dump()}
    dump("exp3b_nullable.jsonl", rec)
    return rec

jobs = [(v, s, k) for v in VARIANTS for s in (True, False) for k in range(5)]
with ThreadPoolExecutor(6) as ex:
    for rec in ex.map(run, jobs):
        print(rec["variant"], rec["strict"], json.dumps(rec["input"]))
print(ledger.report())
