"""Run one arm of the Part 3 experiment.

Usage: python run.py ARM REPEATS [task ids...]
Arms: plain, react, plan, low, medium, high, xhigh, max (Haiku 5.5); s-<arm> for Sonnet 5.5.
Writes runs/<arm>.jsonl (one summary line per run) and runs/transcripts/<arm>/<task>-r<k>.jsonl.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import anthropic  # noqa: E402
import ledger  # noqa: E402
from react import ADAPTIVE, DISABLED, REACT_RULES, plan_and_execute, run_loop  # noqa: E402
from saltmere import SYSTEM, TASKS, final_answer, grade, make_tools  # noqa: E402

HERE = Path(__file__).parent
CL = ledger.client()
PART = "a03"
DONE = set()


def arm_config(arm):
    model = "claude-haiku-5-5"
    if arm.startswith("s-"):
        model, arm = "claude-sonnet-5-5", arm[2:]
    if arm in ("plain", "react", "plan"):
        return model, arm, "medium", DISABLED
    return model, "think", arm.split("-")[0], ADAPTIVE   # arm is the effort level, e.g. xhigh or xhigh-64k


def one(arm, task, k):
    """Run one task once, restarting from scratch (fresh tools) if the API rate-limits us."""
    if (task["id"], k) in DONE:
        return None
    for attempt in range(6):
        try:
            return _one(arm, task, k)
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            print(f"retry {task['id']} r{k} after {type(e).__name__}", flush=True)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"gave up on {task['id']} r{k}")


def _one(arm, task, k):
    model, kind, effort, thinking = arm_config(arm)
    tdir = HERE / "runs" / "transcripts" / arm
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"{task['id']}-r{k}.jsonl"
    tpath.unlink(missing_ok=True)
    tools = make_tools(task)
    cap = (0.15 if arm.endswith("64k") else 0.05) if model.startswith("claude-haiku") else 0.60
    t0 = time.time()
    try:
        if kind == "plan":
            r = plan_and_execute(task["question"], tools, SYSTEM, part=PART, model=model, effort=effort,
                                 max_cost=cap, transcript=tpath, client=CL)
        else:
            system = SYSTEM + (REACT_RULES if kind == "react" else "")
            r = run_loop(task["question"], tools, system, part=PART, model=model, effort=effort,
                         thinking=thinking, max_cost=cap,
                         max_tokens=64000 if arm.endswith("64k") else 20000 if thinking is ADAPTIVE else 4000,
                         timeout=1800 if arm.endswith("64k") else None, transcript=tpath, client=CL)
    except ledger.BudgetExceeded as e:
        print("BUDGET", e)
        raise
    s = r.steps
    rec = {"arm": arm, "model": model, "effort": effort, "thinking": thinking["type"], "task": task["id"],
           "kind": task["kind"], "repeat": k, "correct": grade(task, r.answer),
           "answer": final_answer(r.answer), "expected": task["answer"], "stop": r.stop,
           "calls": len(s), "tool_calls": sum(len(x.tool_calls) for x in s),
           "input": sum(x.context for x in s), "uncached_in": sum(x.input_tokens for x in s),
           "cache_read": sum(x.cache_read for x in s), "cache_write": sum(x.cache_write for x in s),
           "output": sum(x.output_tokens for x in s), "thinking_tokens": sum(getattr(x, "thinking_tokens", 0) for x in s),
           "thinking_steps": sum(1 for x in s if getattr(x, "thinking_tokens", 0)),
           "cost": r.cost, "model_s": sum(x.latency_s for x in s), "wall_s": round(time.time() - t0, 2),
           "replanned": getattr(r, "replanned", False)}
    with open(HERE / "runs" / f"{arm}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{arm:8s} {task['id']:12s} r{k} {'OK ' if rec['correct'] else 'BAD'} {str(rec['answer'])[:20]:20s} "
          f"exp {str(task['answer'])[:12]:12s} calls {rec['calls']:2d} out {rec['output']:6d} "
          f"think {rec['thinking_tokens']:6d} ${rec['cost']:.5f} {rec['model_s']:.1f}s {rec['stop']}", flush=True)
    return rec


if __name__ == "__main__":
    arm, reps = sys.argv[1], int(sys.argv[2])
    ids = set(sys.argv[3:])
    tasks = [t for t in TASKS if not ids or t["id"] in ids]
    jobs = [(t, k) for k in range(1, reps + 1) for t in tasks]
    (HERE / "runs").mkdir(exist_ok=True)
    out = HERE / "runs" / f"{arm}.jsonl"
    if out.exists():  # resume: skip runs already recorded
        DONE.update((r["task"], r["repeat"]) for r in map(json.loads, out.read_text().splitlines()))
    workers = 4
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: one(arm, *j), jobs))
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    n = len(recs)
    print(f"\n{arm}: {sum(r['correct'] for r in recs)}/{n} correct, ${sum(r['cost'] for r in recs):.4f}, "
          f"${sum(r['cost'] for r in recs) / n:.5f}/run")
    print(ledger.report())
