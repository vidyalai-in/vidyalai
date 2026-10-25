"""Run the Haiku 5.5 medium-effort agent on the Saltmere tasks through the Message Batches API.

Usage: python batch.py NAME CACHE REPEATS [task ids...] [--first-repeat K]
  CACHE is 5m, 1h or off. Every round of the loop is one batch holding the next request of every
  unfinished run. Writes runs/<NAME>.jsonl (one line per run) and runs/<NAME>-rounds.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402
from costlat import ADAPTIVE, run_batch  # noqa: E402
from saltmere import SYSTEM, TASKS, final_answer, grade, make_tools  # noqa: E402

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("cache", choices=["5m", "1h", "off"])
ap.add_argument("repeats", type=int)
ap.add_argument("tasks", nargs="*")
ap.add_argument("--first-repeat", type=int, default=1)
args = ap.parse_args()

tasks = {t["id"]: t for t in TASKS if not args.tasks or t["id"] in args.tasks}
jobs = {f"{tid}-r{k}": (t["question"], make_tools(t))
        for k in range(args.first_repeat, args.first_repeat + args.repeats) for tid, t in tasks.items()}
cache = {"5m": True, "1h": "1h", "off": False}[args.cache]
t0 = time.time()
results, finished, rounds = run_batch(jobs, SYSTEM, part="a15", client=ledger.client().with_options(max_retries=6),
                                      model="claude-haiku-5-5", effort="medium", thinking=ADAPTIVE, cache=cache,
                                      max_tokens=16000, poll_s=10,
                                      log_path=HERE / "runs" / f"{args.name}-transcript.jsonl",
                                      label=f"a15 {args.name}")
with (HERE / "runs" / f"{args.name}-rounds.jsonl").open("w") as f:
    for r in rounds:
        f.write(json.dumps(r) + "\n")
with (HERE / "runs" / f"{args.name}.jsonl").open("a") as f:
    for cid, r in results.items():
        tid, k = cid.rsplit("-r", 1)
        s = r.steps
        rec = {"arm": args.name, "model": "claude-haiku-5-5", "effort": "medium", "thinking": "adaptive",
               "cache": args.cache, "task": tid, "kind": tasks[tid]["kind"], "repeat": int(k),
               "correct": grade(tasks[tid], r.answer), "answer": final_answer(r.answer),
               "expected": tasks[tid]["answer"], "stop": r.stop, "calls": len(s),
               "input": sum(x.context for x in s), "uncached_in": sum(x.input_tokens for x in s),
               "cache_read": sum(x.cache_read for x in s), "cache_write": sum(x.cache_write for x in s),
               "output": sum(x.output_tokens for x in s), "thinking_tokens": sum(x.thinking_tokens for x in s),
               "cost": r.cost, "wall_s": round(finished[cid], 1)}
        f.write(json.dumps(rec) + "\n")
ok = sum(grade(tasks[c.rsplit("-r", 1)[0]], r.answer) for c, r in results.items())
print(f"{args.name}: {ok}/{len(results)} correct, ${sum(r.cost for r in results.values()):.4f}, "
      f"{len(rounds)} rounds, {time.time() - t0:.0f} s total")
print(ledger.report())
