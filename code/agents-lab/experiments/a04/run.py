"""Driver: python run.py <out.jsonl> <model> <effort> <repeats> <strategies,...> [task ids...]

Writes one JSON row per (task, repeat) with hidden-suite results for each strategy.
Transcripts of every call go to runs/calls-<name>.jsonl.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import reflect as R  # noqa: E402

out_path, model, effort, reps, strategies = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(",")
thinking = not effort.endswith("-nothink")  # e.g. "low-nothink" omits the thinking parameter
effort = effort.replace("-nothink", "")
only = set(sys.argv[6:])
tasks = [t for t in R.TASKS if not only or t["id"] in only]
name = Path(out_path).stem
log = Path(__file__).parent / "runs" / f"calls-{name}.jsonl"
llm = R.Model(model, effort, log, thinking)


def unit(job):
    t, rep = job
    tag = f"{t['id']}#{rep}"
    row = dict(task=t["id"], rep=rep, model=model, effort=effort, thinking=thinking)
    first, tk, c = llm([{"role": "user", "content": R.ask(t)}], "attempt 1", tag)
    code1 = R.extract_code(first, t["id"])
    ok1, _ = R.grade(t, code1)
    pub1, _ = R.grade(t, code1, public_only=True)
    row["single"] = dict(ok=ok1, public_ok=pub1, tokens=tk, cost=c, code=code1)
    if "critique" in strategies:
        rounds = R.critique(llm, t, first, tag)
        row["critique"] = [dict(ok=R.grade(t, r["code"])[0], changed=r["code"].strip() != code1.strip(),
                                tokens=r["tokens"], cost=r["cost"], code=r["code"]) for r in rounds]
    for s in ("tests", "oracle", "reflexion", "resample"):
        if s in strategies:
            if s == "tests":
                code, calls = R.with_tests(llm, t, first, tag)
            elif s == "oracle":
                code, calls = R.with_tests(llm, t, first, tag, oracle=True)
            elif s == "reflexion":
                code, calls = R.reflexion(llm, t, first, tag)
            else:
                code, calls = R.resample(llm, t, first, tag)
            row[s] = dict(ok=R.grade(t, code)[0], calls=len(calls),
                          tokens=sum(x["tokens"] for x in calls), cost=sum(x["cost"] for x in calls),
                          code=code)
    if "tool" in strategies:
        code, info = R.tool_agent(t, tag, model, effort, log.parent)
        row["tool"] = dict(ok=R.grade(t, code)[0], **info, code=code)
    with R._lock, open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    summary = " ".join(f"{s}={'P' if (row[s][-1] if isinstance(row[s], list) else row[s])['ok'] else '.'}"
                       for s in ["single"] + [s for s in strategies if s in row])
    print(f"{tag:28s} {summary}", flush=True)


def safe(job):
    try:
        unit(job)
    except R.ledger.BudgetExceeded as e:
        print("BUDGET", e, flush=True)
    except Exception as e:  # log and keep going
        print(f"ERROR {job[0]['id']}#{job[1]}: {type(e).__name__}: {e}", flush=True)


done = set()
if Path(out_path).exists():  # resume: skip units already written
    done = {(json.loads(l)["task"], json.loads(l)["rep"]) for l in open(out_path)}
jobs = [(t, r) for r in range(reps) for t in tasks if (t["id"], r) not in done]
with ThreadPoolExecutor(int(__import__('os').environ.get('WORKERS', 6))) as ex:
    list(ex.map(safe, jobs))
