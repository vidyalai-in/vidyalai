"""Run one arm of the Part 15 experiment on the 36 Saltmere tasks, resumably.

Usage: python run.py ARM REPEATS [task ids...] [--jobs N] [--first-repeat K]
Arms:
  h-off        Haiku 5.5, thinking disabled, effort medium, caching on
  h-low/h-med/h-high   Haiku 5.5, adaptive thinking at that effort, caching on
  h-med-nocache        as h-med with no cache breakpoints
  s-med        Sonnet 5.5, adaptive thinking, effort medium, caching on
  s-low        Sonnet 5.5, adaptive thinking, effort low, caching on
  team-s       Sonnet 5.5 lead, Haiku 5.5 workers, effort medium, adaptive thinking, caching on
  team-h       Haiku 5.5 lead and workers, same settings
Writes runs/<arm>.jsonl (one line per run) and runs/transcripts/<arm>/<task>-r<k>[.jsonl|/].
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
from costlat import ADAPTIVE, DISABLED, run_loop, run_team  # noqa: E402
from multiagent import Limited  # noqa: E402
from saltmere import SYSTEM, TASKS, final_answer, grade, make_tools  # noqa: E402

HERE = Path(__file__).parent
PART = "a15"
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
CL = Limited(ledger.client().with_options(max_retries=6), n=3)

LEAD = """You lead a small team that answers questions about the Saltmere Islands, a fictional
archipelago, and solves small puzzles. You cannot use the record tools yourself. Your only tool is
`delegate`, which starts a worker agent with the Saltmere tools (search, lookup, calculate, roads, and
the weigh and guess puzzle tools). A worker starts with an empty context: it knows nothing except the
brief you write, and it cannot see the user's question, your other briefs or other workers' reports.

How to work:
- Split the question into subtasks only when they do not depend on each other, and delegate them in
  the same turn so they run in parallel. Give work that cannot be split (a chain where each fact
  leads to the next, a route search, a puzzle) to one worker.
- The weigh and guess tools have a limited number of uses per task, shared by all workers, so give a
  puzzle to exactly one worker and pass on every rule and limit from the question.
- Write each brief so it stands alone: the exact names, what to find or compute, and the format of
  the report you want back.
- Check the reports against the question before answering.

When you are done, end your reply with one line in the form
ANSWER: <the answer only>"""

WORKER = """You are a worker on the Saltmere Islands records, a fictional archipelago. A lead gave you
the task below. The records are the only source of truth: look facts up with the tools and never guess
them. Use calculate for arithmetic. The weigh and guess tools have a limited number of uses, so make
every use count. Your final reply goes back to the lead, who cannot see your tool calls or their
results, so it must contain everything the lead asked for. Keep it short."""

ARMS = {
    "h-off": dict(model=HAIKU, effort="medium", thinking=DISABLED, cache=True, max_tokens=4000),
    "h-low": dict(model=HAIKU, effort="low", thinking=ADAPTIVE, cache=True, max_tokens=16000),
    "h-med": dict(model=HAIKU, effort="medium", thinking=ADAPTIVE, cache=True, max_tokens=16000),
    "h-high": dict(model=HAIKU, effort="high", thinking=ADAPTIVE, cache=True, max_tokens=16000),
    "h-med-nocache": dict(model=HAIKU, effort="medium", thinking=ADAPTIVE, cache=False, max_tokens=16000),
    "s-med": dict(model=SONNET, effort="medium", thinking=ADAPTIVE, cache=True, max_tokens=16000),
    "s-low": dict(model=SONNET, effort="low", thinking=ADAPTIVE, cache=True, max_tokens=16000),
}

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("repeats", type=int)
ap.add_argument("tasks", nargs="*")
ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--first-repeat", type=int, default=1)
args = ap.parse_args()
OUT = HERE / "runs" / f"{args.arm}.jsonl"
OUT.parent.mkdir(exist_ok=True)
DONE = {(json.loads(l)["task"], json.loads(l)["repeat"]) for l in OUT.read_text().splitlines()} if OUT.exists() else set()


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
    tdir = HERE / "runs" / "transcripts" / arm
    tdir.mkdir(parents=True, exist_ok=True)
    tools = make_tools(task)
    t0 = time.time()
    extra = {}
    if arm.startswith("team"):
        lead = SONNET if arm == "team-s" else HAIKU
        r = run_team(task["question"], tools, LEAD, WORKER, part=PART, client=CL, lead_model=lead,
                     worker_model=HAIKU, effort="medium", max_cost=0.50 if lead == SONNET else 0.10,
                     worker_max_cost=0.05, transcript_dir=tdir / f"{task['id']}-r{k}")
        steps, lead_steps = r.all_steps(), r.steps
        # Critical path of model time: lead calls in sequence, plus, for each lead turn, the slowest of
        # the workers that turn started (they run in parallel). Ledger token counts are excluded.
        crit, i = 0.0, 0
        for st in lead_steps:
            crit += st.latency_s
            m = sum(c.startswith("delegate(") for c in st.tool_calls)
            crit += max((sum(x.latency_s for x in w.steps) for w in r.workers[i:i + m]), default=0.0)
            i += m
        extra = dict(crit_s=round(crit, 2), workers=len(r.workers), lead_steps=len(lead_steps), lead_cost=sum(s.cost for s in lead_steps),
                     worker_cost=r.cost - sum(s.cost for s in lead_steps), briefs=[w.brief for w in r.workers])
        model, effort, thinking = f"{lead}+{HAIKU}", "medium", "adaptive"
    else:
        cfg = ARMS[arm]
        tpath = tdir / f"{task['id']}-r{k}.jsonl"
        tpath.unlink(missing_ok=True)
        r = run_loop(task["question"], tools, SYSTEM, part=PART, client=CL, model=cfg["model"],
                     effort=cfg["effort"], thinking=cfg["thinking"], cache=cfg["cache"],
                     max_tokens=cfg["max_tokens"], max_cost=0.05 if cfg["model"] == HAIKU else 0.60,
                     transcript=tpath, label=f"a15 {arm}")
        steps = r.steps
        model, effort, thinking = cfg["model"], cfg["effort"], cfg["thinking"]["type"]
    wall = time.time() - t0
    rec = {"arm": arm, "model": model, "effort": effort, "thinking": thinking, "task": task["id"],
           "kind": task["kind"], "repeat": k, "correct": grade(task, r.answer), "answer": final_answer(r.answer),
           "expected": task["answer"], "stop": r.stop, "calls": len(steps),
           "tool_calls": sum(len(x.tool_calls) for x in steps),
           "input": sum(x.context for x in steps), "uncached_in": sum(x.input_tokens for x in steps),
           "cache_read": sum(x.cache_read for x in steps), "cache_write": sum(x.cache_write for x in steps),
           "output": sum(x.output_tokens for x in steps),
           "thinking_tokens": sum(getattr(x, "thinking_tokens", 0) for x in steps),
           "cost": r.cost, "model_s": round(sum(x.latency_s for x in steps), 2),
           "call_s": [x.latency_s for x in steps], "wall_s": round(wall, 2), "t_start": t0, **extra}
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{arm:13s} {task['id']:12s} r{k} {'OK ' if rec['correct'] else 'BAD'} {str(rec['answer'])[:18]:18s} "
          f"exp {str(task['answer'])[:12]:12s} calls {rec['calls']:2d} out {rec['output']:6d} "
          f"${rec['cost']:.5f} {rec['wall_s']:.1f}s {rec['stop']}", flush=True)
    return rec


tasks = [t for t in TASKS if not args.tasks or t["id"] in args.tasks]
jobs = [(t, k) for k in range(args.first_repeat, args.first_repeat + args.repeats) for t in tasks]
try:
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(lambda j: one(*j), jobs))
except ledger.BudgetExceeded as e:
    print("BUDGET", e)
recs = [json.loads(l) for l in OUT.read_text().splitlines()]
n = len(recs)
print(f"\n{args.arm}: {sum(r['correct'] for r in recs)}/{n} correct, ${sum(r['cost'] for r in recs):.4f}, "
      f"${sum(r['cost'] for r in recs) / max(n, 1):.5f}/run")
print(ledger.report())
