"""Run the coding agent on the mini-benchmark: arms x tasks x repeats, resumable, 3 at a time.

usage: run.py --arms full,no_tests --repeats 2 [--tasks id1,id2] [--model claude-haiku-5-5]
Each run: copy the task's base repo to a throwaway dir, run the agent, grade with the hidden tests,
save the transcript, the diff and one result line in runs/results.jsonl.
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402
from coding import ARMS, NO_RUN, SYSTEM, make_tools, task_message  # noqa: E402
from react import run_loop  # noqa: E402

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/a08/scripts")
from tasklib import grade, load_tasks, pythonpath  # noqa: E402

A08 = Path("/home/amit/.cache/vidyalai-scratch/a08")
RUNS = A08 / "runs"
WORK = A08 / "work"
LOCK = threading.Lock()
THINKING = {"type": "adaptive", "display": "summarized"}


def done_keys(path):
    if not path.exists():
        return set()
    return {(r["task"], r["arm"], r["rep"], r["model"], r["effort"])
            for r in map(json.loads, path.read_text().splitlines())}


def one(task, arm, rep, model, effort, cl, max_cost, max_steps, results):
    tag = f"{task['id']}-{arm}-{model.split('-')[1]}-{effort}-r{rep}"
    ws = WORK / tag
    shutil.rmtree(ws, ignore_errors=True)
    shutil.copytree(A08 / "tasks" / task["id"] / "base", ws / "repo", symlinks=True)
    root = ws / "repo"
    subprocess.run(["git", "init", "-q"], cwd=root)
    subprocess.run(["git", "add", "-A"], cwd=root)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base"], cwd=root)
    kit, _ = ARMS[arm]
    tools = make_tools(root, pythonpath(task, root), kit)
    system = SYSTEM + ("" if "run_tests" in kit else NO_RUN)
    tdir = RUNS / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"{tag}.jsonl"
    tpath.unlink(missing_ok=True)
    t0 = time.time()
    err = None
    try:
        res = run_loop(task_message(task, root, arm), tools, system, part="a08", model=model,
                       effort=effort, thinking=THINKING, max_steps=max_steps, max_cost=max_cost,
                       max_tokens=32000, transcript=tpath, client=cl, timeout=900)
        steps, stop, answer = res.steps, res.stop, res.answer
    except ledger.BudgetExceeded:
        raise
    except Exception as e:  # API error after retries: record as a failed run
        err = f"{type(e).__name__}: {e}"[:300]
        steps, stop, answer = [], "error", ""
    wall = time.time() - t0
    diff = subprocess.run(["git", "diff"], cwd=root, capture_output=True, text=True).stdout
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=root,
                               capture_output=True, text=True).stdout.split()
    (RUNS / "diffs").mkdir(exist_ok=True)
    (RUNS / "diffs" / f"{tag}.diff").write_text(diff)
    g = grade(task, root)
    calls = [c.split("(")[0] for s in steps for c in s.tool_calls]
    rec = dict(task=task["id"], repo=task["repo"], arm=arm, rep=rep, model=model, effort=effort,
               resolved=g["resolved"], f2p_pass=g["f2p_pass"], f2p_total=g["f2p_total"],
               p2p_fail=g["p2p_fail"], stop=stop, error=err, steps=len(steps),
               input=sum(s.input_tokens for s in steps), cache_read=sum(s.cache_read for s in steps),
               cache_write=sum(s.cache_write for s in steps), output=sum(s.output_tokens for s in steps),
               thinking=sum(getattr(s, "thinking_tokens", 0) for s in steps),
               peak_context=max((s.context for s in steps), default=0),
               cost=sum(s.cost for s in steps), wall_s=round(wall, 1),
               tool_counts={k: calls.count(k) for k in sorted(set(calls))},
               diff_lines=sum(1 for l in diff.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")),
               files_changed=[l[6:] for l in diff.splitlines() if l.startswith("+++ b/")],
               untracked=untracked, answer=answer[:600])
    with LOCK:
        with results.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"{tag:55s} {'PASS' if rec['resolved'] else 'fail'} steps {rec['steps']:2d} "
          f"${rec['cost']:.4f} {stop} {wall:.0f}s", flush=True)
    shutil.rmtree(ws, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="full")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--start-rep", type=int, default=0)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-cost", type=float, default=0.20)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--results", default=str(RUNS / "results.jsonl"))
    a = ap.parse_args()
    tasks = load_tasks()
    if a.tasks:
        keep = a.tasks.split(",")
        tasks = [t for t in tasks if t["id"] in keep]
    results = Path(a.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(results)
    cl = ledger.client().with_options(max_retries=8)
    jobs = [(t, arm, rep) for rep in range(a.start_rep, a.start_rep + a.repeats)
            for arm in a.arms.split(",") for t in tasks
            if (t["id"], arm, rep, a.model, a.effort) not in seen]
    print(f"{len(jobs)} runs to do; spent so far for a08: ${ledger.spent('a08'):.4f}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, t, arm, rep, a.model, a.effort, cl, a.max_cost, a.max_steps, results)
                for t, arm, rep in jobs]
        for f in futs:
            try:
                f.result()
            except ledger.BudgetExceeded as e:
                print("BUDGET STOP:", e, flush=True)
                ex.shutdown(cancel_futures=True)
                break
    print(f"a08 spent: ${ledger.spent('a08'):.4f}", flush=True)


if __name__ == "__main__":
    main()
