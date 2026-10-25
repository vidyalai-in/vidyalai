"""Run one arm of the Part 5 experiment, resumably.

Usage: python run.py ARM REPEATS [--workers N] [--model sonnet] [task ids...]
Arms: naive, trunc, sized, edit, compact, memory.
Writes runs/<tag>.jsonl (one summary line per run) and runs/transcripts/<tag>/<task>-r<k>.jsonl.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import anthropic  # noqa: E402
import ledger  # noqa: E402
from context import run  # noqa: E402
from store import SYSTEM, TASKS, TOOLSETS, final_answer, grade  # noqa: E402

HERE = Path(__file__).parent
PART = "a05"
CL = ledger.client().with_options(max_retries=6)

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("repeats", type=int)
ap.add_argument("tasks", nargs="*")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--model", default="haiku")
ap.add_argument("--first-repeat", type=int, default=0)
ap.add_argument("--nocache", action="store_true")
ap.add_argument("--clock", action="store_true")
args = ap.parse_args()
MODEL = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}[args.model]
TAG = args.arm if args.model == "haiku" else f"{args.model}-{args.arm}"
TAG = ("nocache-" if args.nocache else "") + ("clock-" if args.clock else "") + TAG
OUT = HERE / "runs" / f"{TAG}.jsonl"
OUT.parent.mkdir(exist_ok=True)
DONE = {(json.loads(l)["task"], json.loads(l)["repeat"]) for l in OUT.read_text().splitlines()} if OUT.exists() else set()


def one(task, k):
    if (task["id"], k) in DONE:
        return None
    for attempt in range(5):
        try:
            return _one(task, k)
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            print(f"retry {task['id']} r{k} after {type(e).__name__}", flush=True)
            time.sleep(30 * (attempt + 1))
    print(f"gave up on {task['id']} r{k}", flush=True)


def _one(task, k):
    tdir = HERE / "runs" / "transcripts" / TAG
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"{task['id']}-r{k}.jsonl"
    tpath.unlink(missing_ok=True)
    t0 = time.time()
    r = run(task["question"] + f"\n\n(Request ID: {TAG}-{task['id']}-r{k})", TOOLSETS[args.arm], SYSTEM, args.arm, part=PART, client=CL, model=MODEL,
            memdir=HERE / "memdirs" / TAG / f"{task['id']}-r{k}", transcript=tpath,
            cache=not args.nocache, clock=args.clock,
            max_cost=(0.12 if task["size"] >= 48 else 0.04) * (1 if args.model == "haiku" else 20))
    s = r.steps
    rec = {"arm": args.arm, "model": MODEL, "task": task["id"], "kind": task["kind"], "size": task["size"],
           "repeat": k, "correct": grade(task, r.answer), "answer": final_answer(r.answer),
           "expected": task["answer"], "stop": r.stop, "steps": len(s),
           "tool_calls": sum(len(x.tool_calls) for x in s), "windows": max((x.window for x in s), default=1),
           "peak_context": max((x.context for x in s), default=0),
           "input": sum(x.context for x in s), "uncached_in": sum(x.input_tokens for x in s),
           "cache_read": sum(x.cache_read for x in s), "cache_write": sum(x.cache_write for x in s),
           "output": sum(x.output_tokens for x in s), "cleared": max((x.cleared for x in s), default=0),
           "compactions": sum(x.compacted for x in s), "compaction_in": sum(x.compaction_in for x in s),
           "cost": r.cost, "model_s": sum(x.latency_s for x in s), "wall_s": round(time.time() - t0, 1)}
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{TAG} {task['id']} r{k} {'OK ' if rec['correct'] else 'BAD'} ans={rec['answer']!r} exp={task['answer']} "
          f"steps={rec['steps']} peak={rec['peak_context']:,} ${rec['cost']:.4f}", flush=True)
    return rec


tasks = [t for t in TASKS if not args.tasks or t["id"] in args.tasks]
jobs = [(t, k) for k in range(args.first_repeat, args.first_repeat + args.repeats) for t in tasks]
try:
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda j: one(*j), jobs))
except ledger.BudgetExceeded as e:
    print("BUDGET", e)
print(ledger.report())
