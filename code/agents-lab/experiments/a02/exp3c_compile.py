"""Exp 3c: first-use grammar compile latency. A random enum value makes each schema new."""
import copy, time, uuid
from common import ledger, HAIKU, dump
import exp3_schemas as S

cl = ledger.client()
PROMPT = S.PROMPTS[32][1]  # weekly revenue report, Mondays 9am

def tools(strict, nonce):
    t = S.tools(strict)
    t[2]["input_schema"]["properties"]["format"]["enum"] = ["pdf", "csv", "xlsx", f"fmt_{nonce}"]
    return t

for trial in range(6):
    for strict in (True, False):
        nonce = uuid.uuid4().hex[:8]
        for phase in ("first", "repeat", "repeat2"):
            t0 = time.time()
            r = ledger.create(cl, "a02", note=f"exp3c compile strict={strict}", model=HAIKU, max_tokens=2048,
                              system=S.SYSTEM, tools=tools(strict, nonce), messages=[{"role": "user", "content": PROMPT}])
            secs = time.time() - t0
            rec = {"trial": trial, "strict": strict, "phase": phase, "secs": round(secs, 3), "stop": r.stop_reason,
                   "usage": r.usage.model_dump()}
            dump("exp3c_compile.jsonl", rec)
            print(trial, strict, phase, rec["secs"], r.usage.output_tokens)
print(ledger.report())
