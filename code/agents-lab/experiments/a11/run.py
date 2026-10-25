"""Part 11: run each task of two small task sets k times per configuration, resumable, 3 at a time.

usage: run.py CONFIG REPEATS [--start R] [--tasks id1,id2]
Configs:
  salt-off      Saltmere questions (Part 3), Haiku 5.5, thinking disabled, effort medium
  salt-low      Saltmere questions, Haiku 5.5, adaptive thinking, effort low
  code-haiku    28 coding bugs (Part 8), title-only issue, Haiku 5.5, adaptive, effort medium
  code-sonnet   same, Sonnet 5.5, adaptive, effort medium
One JSON line per run in runs/<config>.jsonl; transcripts in runs/transcripts/<config>/; diffs for code.
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

LAB = "/home/amit/.cache/vidyalai-scratch/agents/lab"
A11 = Path("/home/amit/.cache/vidyalai-scratch/a11")
sys.path.insert(0, LAB)
sys.path.insert(0, str(A11 / "saltmere"))
sys.path.insert(0, str(A11 / "coding"))
import anthropic  # noqa: E402
import ledger  # noqa: E402
from react import ADAPTIVE, DISABLED, run_loop  # noqa: E402

PART = "a11"
RUNS = A11 / "runs"
LOCK = threading.Lock()
CL = ledger.client().with_options(max_retries=8)

CONFIGS = {
    "salt-off": dict(kind="salt", model="claude-haiku-5-5", effort="medium", thinking=DISABLED,
                     max_tokens=4000, max_cost=0.05, max_steps=20),
    "salt-low": dict(kind="salt", model="claude-haiku-5-5", effort="low", thinking=ADAPTIVE,
                     max_tokens=20000, max_cost=0.05, max_steps=20),
    "code-haiku": dict(kind="code", model="claude-haiku-5-5", effort="medium", thinking=ADAPTIVE,
                       max_tokens=32000, max_cost=0.20, max_steps=50),
    "code-sonnet": dict(kind="code", model="claude-sonnet-5-5", effort="medium", thinking=ADAPTIVE,
                        max_tokens=32000, max_cost=2.00, max_steps=50),
}


def salt_run(cfg, task, tpath):
    from saltmere import SYSTEM, final_answer, grade, make_tools
    r = run_loop(task["question"], make_tools(task), SYSTEM, part=PART, model=cfg["model"],
                 effort=cfg["effort"], thinking=cfg["thinking"], max_cost=cfg["max_cost"],
                 max_steps=cfg["max_steps"], max_tokens=cfg["max_tokens"], transcript=tpath, client=CL)
    return r, dict(correct=grade(task, r.answer), answer=final_answer(r.answer), expected=task["answer"],
                   kind_=task["kind"])


def code_run(cfg, task, tpath, tag):
    from coding import SYSTEM, make_tools, task_message
    from tasklib import grade, pythonpath
    ws = A11 / "work" / tag
    shutil.rmtree(ws, ignore_errors=True)
    shutil.copytree(A11 / "coding" / "tasks" / task["id"] / "base", ws / "repo", symlinks=True)
    root = ws / "repo"
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base"]):
        subprocess.run(cmd, cwd=root)
    kit = ["grep", "glob", "read_file", "str_replace", "create_file", "run_tests"]
    r = run_loop(task_message(task, root, "title_only"), make_tools(root, pythonpath(task, root), kit),
                 SYSTEM, part=PART, model=cfg["model"], effort=cfg["effort"], thinking=cfg["thinking"],
                 max_steps=cfg["max_steps"], max_cost=cfg["max_cost"], max_tokens=cfg["max_tokens"],
                 transcript=tpath, client=CL, timeout=900)
    diff = subprocess.run(["git", "diff"], cwd=root, capture_output=True, text=True).stdout
    (RUNS / "diffs").mkdir(exist_ok=True)
    (RUNS / "diffs" / f"{tag}.diff").write_text(diff)
    g = grade(task, root)
    shutil.rmtree(ws, ignore_errors=True)
    return r, dict(correct=g["resolved"], f2p_pass=g["f2p_pass"], f2p_total=g["f2p_total"],
                   p2p_fail=g["p2p_fail"], diff_lines=sum(1 for l in diff.splitlines()
                                                         if l[:1] in "+-" and l[:3] not in ("+++", "---")))


def one(name, task, rep, out):
    cfg = CONFIGS[name]
    tag = f"{name}-{task['id']}-r{rep}"
    tdir = RUNS / "transcripts" / name
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"{task['id']}-r{rep}.jsonl"
    for attempt in range(4):
        tpath.unlink(missing_ok=True)
        t0 = time.time()
        try:
            r, extra = salt_run(cfg, task, tpath) if cfg["kind"] == "salt" else code_run(cfg, task, tpath, tag)
            break
        except ledger.BudgetExceeded:
            raise
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            print(f"retry {tag} after {type(e).__name__}", flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f"gave up on {tag}")
    s = r.steps
    rec = dict(config=name, task=task["id"], rep=rep, model=cfg["model"], effort=cfg["effort"],
               thinking=cfg["thinking"]["type"], stop=r.stop, steps=len(s),
               input=sum(x.context for x in s), output=sum(x.output_tokens for x in s),
               thinking_tokens=sum(getattr(x, "thinking_tokens", 0) for x in s),
               cost=r.cost, model_s=round(sum(x.latency_s for x in s), 2), wall_s=round(time.time() - t0, 2),
               ts=time.time(), **extra)
    with LOCK, out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{tag:45s} {'OK ' if rec['correct'] else 'BAD'} steps {rec['steps']:2d} ${rec['cost']:.5f} "
          f"{rec['wall_s']:.0f}s {r.stop}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("repeats", type=int)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    if cfg["kind"] == "salt":
        from saltmere import TASKS
        tasks = TASKS
    else:
        from tasklib import load_tasks
        tasks = load_tasks()
    if a.tasks:
        tasks = [t for t in tasks if t["id"] in a.tasks.split(",")]
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{a.config}.jsonl"
    done = {(r["task"], r["rep"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    jobs = [(t, k) for k in range(a.start, a.start + a.repeats) for t in tasks if (t["id"], k) not in done]
    print(f"{len(jobs)} runs to do; a11 spent so far ${ledger.spent(PART):.4f}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, a.config, t, k, out) for t, k in jobs]
        for f in futs:
            try:
                f.result()
            except ledger.BudgetExceeded as e:
                print("BUDGET STOP:", e, flush=True)
                ex.shutdown(cancel_futures=True)
                break
    print(f"a11 spent: ${ledger.spent(PART):.4f}", flush=True)


if __name__ == "__main__":
    main()
