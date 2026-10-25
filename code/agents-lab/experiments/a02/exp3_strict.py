"""Exp 3: schema violations in tool calls with strict off vs strict on (Haiku 5.5).

usage: python exp3_strict.py <repeats> [n_prompts]
"""
import sys, time, random, json
from concurrent.futures import ThreadPoolExecutor
from jsonschema import Draft202012Validator, FormatChecker
from common import ledger, HAIKU, dump
import exp3_schemas as S

cl = ledger.client()
VALIDATORS = {t["name"]: Draft202012Validator(t["input_schema"], format_checker=FormatChecker()) for t in S.TOOLS}


def check(name, args):
    """Return (schema_errors, range_errors) for one tool call."""
    if name not in VALIDATORS:
        return [f"unknown tool {name!r}"], []
    errs = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:120]}" for e in VALIDATORS[name].iter_errors(args)]
    rng = []
    for field, (lo, hi) in S.RANGES.get(name, {}).items():
        v = args.get(field) if isinstance(args, dict) else None
        if isinstance(v, int) and not isinstance(v, bool) and not (lo <= v <= hi):
            rng.append(f"{field}={v} outside {lo}-{hi}")
    return errs, rng


def run_one(job):
    strict, idx, rep = job
    expected, prompt = S.PROMPTS[idx]
    t0 = time.time()
    r = ledger.create(cl, "a02", note=f"exp3 strict={strict}", model=HAIKU, max_tokens=4096,
                      system=S.SYSTEM, tools=S.tools(strict), messages=[{"role": "user", "content": prompt}])
    secs = time.time() - t0
    calls = [b for b in r.content if b.type == "tool_use"]
    rec = {"strict": strict, "idx": idx, "rep": rep, "expected": expected, "secs": round(secs, 3),
           "stop_reason": r.stop_reason, "n_calls": len(calls),
           "has_thinking": any(b.type == "thinking" for b in r.content),
           "text": " ".join(b.text for b in r.content if b.type == "text")[:500],
           "usage": r.usage.model_dump(), "calls": []}
    for c in calls:
        errs, rng = check(c.name, c.input)
        rec["calls"].append({"name": c.name, "input": c.input, "schema_errors": errs, "range_errors": rng})
    dump("exp3_strict.jsonl", rec)
    return rec


if __name__ == "__main__":
    reps = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else len(S.PROMPTS)
    jobs = [(s, i, k) for k in range(reps) for i in range(n) for s in (False, True)]
    random.seed(7)
    random.shuffle(jobs)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for rec in ex.map(run_one, jobs):
            bad = [c for c in rec["calls"] if c["schema_errors"]]
            if bad or rec["n_calls"] != 1:
                print("strict" if rec["strict"] else "loose ", rec["idx"], rec["n_calls"], [c["schema_errors"] for c in bad])
    print(ledger.report())
