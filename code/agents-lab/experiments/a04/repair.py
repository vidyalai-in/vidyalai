"""Repair experiment: start every strategy from a known-WRONG first attempt.

python repair.py <items.json> <out.jsonl> <model> <effort> <reps> <strategies,...>
items.json: list of {task, tag, reply} (wrong attempts from runs, or planted-bug mutants).
The wrong attempt is presented as the model's own previous answer.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import reflect as R  # noqa: E402

items_path, out_path, model, effort, reps, strategies = sys.argv[1:7]
reps, strategies = int(reps), strategies.split(",")
items = json.load(open(items_path))
log = Path(__file__).parent / "runs" / f"calls-{Path(out_path).stem}.jsonl"
llm = R.Model(model, effort, log)
T = {t["id"]: t for t in R.TASKS}


def unit(job):
    it, rep = job
    t, first = T[it["task"]], it["reply"]
    tag = f"{it['tag']}~{rep}"
    code0 = R.extract_code(first, t["id"])
    row = dict(item=it["tag"], task=t["id"], rep=rep, model=model, effort=effort,
               public_ok=R.grade(t, code0, public_only=True)[0])
    assert not R.grade(t, code0)[0], "item must be wrong"
    if "critique" in strategies:
        rounds = R.critique(llm, t, first, tag)
        row["critique"] = [dict(ok=R.grade(t, r["code"])[0], changed=r["code"].strip() != code0.strip(),
                                tokens=r["tokens"], cost=r["cost"], code=r["code"], reply=r["reply"]) for r in rounds]
    if "critique1" in strategies:  # one round only (cheaper, for the Sonnet comparison)
        rounds = R.critique(llm, t, first, tag, rounds=1)
        row["critique"] = [dict(ok=R.grade(t, r["code"])[0], changed=r["code"].strip() != code0.strip(),
                                tokens=r["tokens"], cost=r["cost"], code=r["code"], reply=r["reply"]) for r in rounds]
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
            row[s] = dict(ok=R.grade(t, code)[0], calls=len(calls), tokens=sum(x["tokens"] for x in calls),
                          cost=sum(x["cost"] for x in calls), code=code)
    with R._lock, open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(tag, " ".join(f"{s}={'P' if (row[s][-1] if isinstance(row[s], list) else row[s])['ok'] else '.'}"
                        for s in ("critique", "tests", "oracle", "reflexion", "resample") if s in row), flush=True)


def safe(job):
    try:
        unit(job)
    except R.ledger.BudgetExceeded as e:
        print("BUDGET", e, flush=True)
    except Exception as e:
        print(f"ERROR {job[0]['tag']}~{job[1]}: {type(e).__name__}: {e}", flush=True)


done = set()
if Path(out_path).exists():
    done = {(json.loads(l)["item"], json.loads(l)["rep"]) for l in open(out_path)}
jobs = [(it, r) for r in range(reps) for it in items if (it["tag"], r) not in done]
with ThreadPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex:
    list(ex.map(safe, jobs))
