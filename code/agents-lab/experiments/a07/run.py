"""Run one arm of the Part 7 experiment, resumably.

Usage: python run.py ARM REPEATS [task ids...] [--jobs N] [--first-repeat K]
Arms: single-full, team-full, single-sized, team-sized, sonnet-team-full, sonnet-single-sized
Writes runs/<arm>.jsonl (one summary line per run) and runs/transcripts/<arm>/<task>-r<k>/.
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/a05")
import anthropic  # noqa: E402
import ledger  # noqa: E402
from multiagent import Limited, run_single, run_team  # noqa: E402
from store import NAIVE, SIZED, SYSTEM, final_answer, grade  # noqa: E402

HERE = Path(__file__).parent
PART = "a07"
TASKS = json.loads((HERE / "tasks.json").read_text())
CL = Limited(ledger.client().with_options(max_retries=6), n=3)
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("repeats", type=int)
ap.add_argument("tasks", nargs="*")
ap.add_argument("--jobs", type=int, default=1)
ap.add_argument("--first-repeat", type=int, default=0)
ap.add_argument("--set", default="main")
args = ap.parse_args()
if args.set == "stress":
    TASKS = json.loads((HERE / "tasks_stress.json").read_text())
OUT = HERE / "runs" / (f"{args.arm}.jsonl" if args.set == "main" else f"{args.set}-{args.arm}.jsonl")
OUT.parent.mkdir(exist_ok=True)
DONE = {(json.loads(l)["task"], json.loads(l)["repeat"]) for l in OUT.read_text().splitlines()} if OUT.exists() else set()


def grade_task(task, text):
    if task["kind"] in ("wide", "wide12") and task["qkind"] != "max_robots":
        a = final_answer(text)
        if not a:
            return False, False
        svc, num = task["answer"].rsplit(", ", 1)
        named = {s for s, _ in task["groups"] if re.search(rf"\b{re.escape(s)}\b", a)}
        nums = re.findall(r"\d[\d,]*", a.replace(svc, ""))
        ok_svc = named == {svc}
        return ok_svc and bool(nums) and int(nums[0].replace(",", "")) == int(num), ok_svc
    ok = grade(task, text)
    return ok, ok


def reads(steps):
    ids = []
    for s in steps:
        for c in s.tool_calls:
            if c.startswith("get_incident("):
                ids.append(json.loads(c[len("get_incident("):-1])["incident_id"].strip().upper())
    return ids


def one(task, k):
    if (task["id"], k) in DONE:
        return None
    for attempt in range(4):
        try:
            return _one(task, k)
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            print(f"retry {task['id']} r{k} after {type(e).__name__}", flush=True)
            time.sleep(30 * (attempt + 1))
    print(f"gave up on {task['id']} r{k}", flush=True)


def _one(task, k):
    arm = args.arm
    tools = SIZED if arm.endswith("sized") else NAIVE
    tdir = HERE / "runs" / "transcripts" / (arm if args.set == "main" else f"{args.set}-{arm}") / f"{task['id']}-r{k}"
    tdir.mkdir(parents=True, exist_ok=True)
    for f in tdir.glob("*.jsonl"):
        f.unlink()
    q = task["question"] + f"\n\n(Request ID: {arm}-{task['id']}-r{k})"
    t0 = time.time()
    if "single" in arm:
        model = SONNET if arm.startswith("sonnet") else HAIKU
        r = run_single(q, tools, SYSTEM, part=PART, client=CL, model=model,
                       max_cost=0.25 if model == HAIKU else 1.0, transcript=tdir / "single.jsonl")
    else:
        lead = SONNET if arm.startswith("sonnet") else HAIKU
        r = run_team(q, tools, part=PART, client=CL, lead_model=lead, worker_model=HAIKU,
                     max_cost=0.25 if lead == HAIKU else 0.5, transcript_dir=tdir)
    wall = time.time() - t0
    allsteps = r.all_steps()
    ids = reads(allsteps)
    members = set(task["members"])
    ok, ok_part = grade_task(task, r.answer)
    tok = lambda ss: sum(s.context + s.output_tokens for s in ss)  # noqa: E731
    rec = {"arm": arm, "task": task["id"], "kind": task["kind"], "size": task["size"], "repeat": k,
           "correct": ok, "correct_part": ok_part, "answer": final_answer(r.answer), "expected": task["answer"],
           "stop": r.stop, "workers": len(r.workers), "worker_stops": [w.stop for w in r.workers],
           "lead_steps": len(r.steps), "steps": len(allsteps),
           "tokens": tok(allsteps), "lead_tokens": tok(r.steps),
           "input": sum(s.context for s in allsteps), "output": sum(s.output_tokens for s in allsteps),
           "cache_read": sum(s.cache_read for s in allsteps), "cache_write": sum(s.cache_write for s in allsteps),
           "peak_context": max((s.context for s in allsteps), default=0),
           "lead_peak": max((s.context for s in r.steps), default=0),
           "reads": len(ids), "unique_reads": len(set(ids)), "covered": len(set(ids) & members),
           "members": len(members), "cost": r.cost, "lead_cost": sum(s.cost for s in r.steps),
           "model_s": sum(s.latency_s for s in allsteps), "wall_s": round(wall, 1),
           "briefs": [w.brief for w in r.workers], "reports": [w.answer for w in r.workers]}
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{arm} {task['id']} r{k} {'OK ' if ok else 'BAD'} ans={rec['answer']!r} exp={task['answer']!r} "
          f"workers={rec['workers']} steps={rec['steps']} tok={rec['tokens']:,} peak={rec['peak_context']:,} "
          f"${rec['cost']:.4f} {rec['wall_s']}s", flush=True)
    return rec


tasks = [t for t in TASKS if not args.tasks or t["id"] in args.tasks]
jobs = [(t, k) for k in range(args.first_repeat, args.first_repeat + args.repeats) for t in tasks]
try:
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(lambda j: one(*j), jobs))
except ledger.BudgetExceeded as e:
    print("BUDGET", e)
print(ledger.report())
