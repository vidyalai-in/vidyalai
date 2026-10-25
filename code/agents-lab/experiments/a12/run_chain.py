"""Experiment 1: compounding error on the chain task.

Usage: python run_chain.py ARM LEVEL N SEEDS...   (LEVEL easy|hard, SEEDS like 1-30)
Arms: off (Haiku 5.5, thinking disabled, effort medium), low (adaptive thinking, effort low)
Writes runs/chain-<arm>.jsonl and runs/tx/chain-<arm>/<id>.jsonl. Skips runs already done.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import anthropic  # noqa: E402
import ledger  # noqa: E402
from reliability import DISABLED, run_durable  # noqa: E402
from chain import RESTATE, SILENT, SYSTEM, ChainState, final, make_task, make_tools, step_record  # noqa: E402

HERE = Path(__file__).parent
CL = anthropic.Anthropic(api_key=ledger.client().api_key, max_retries=6)
ARMS = {"off": dict(effort="medium", thinking=DISABLED, max_tokens=4000),
        "restate": dict(effort="medium", thinking=DISABLED, max_tokens=4000),
        "silent": dict(effort="medium", thinking=DISABLED, max_tokens=4000),
        "low": dict(effort="low", thinking={"type": "adaptive", "display": "summarized"}, max_tokens=16000)}


def one(arm, task, out):
    tdir = HERE / "runs" / "tx" / f"chain-{arm}"
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"{task['id']}.jsonl"
    for attempt in range(4):
        tpath.unlink(missing_ok=True)
        st = ChainState(task)
        try:
            t0 = time.time()
            system = SYSTEM + {"silent": SILENT, "restate": RESTATE}.get(arm, "")
            r = run_durable(task["question"], make_tools(st), system, part="a12", client=CL,
                            max_steps=task["n"] + 15, max_cost=0.25, transcript=tpath,
                            note=f"a12 chain {arm}", **ARMS[arm])
            break
        except (anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.RateLimitError) as e:
            print("retry", task["id"], type(e).__name__, flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        return None
    s = r.steps
    rec = {"arm": arm, "id": task["id"], "seed": task["seed"], "n": task["n"], "level": task["level"], "expected": task["answer"],
           "answer": final(r.answer), "correct": final(r.answer) == task["answer"], "stop": r.stop,
           "calls": len(s), "input": sum(x.context for x in s), "cache_read": sum(x.cache_read for x in s),
           "output": sum(x.output_tokens for x in s), "cost": r.cost, "model_s": sum(x.latency_s for x in s),
           "wall_s": round(time.time() - t0, 1), **step_record(st, r.answer)}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{task['id']:10s} {arm} correct={rec['correct']} steps_ok={sum(rec['steps'])}/{len(rec['steps'])} "
          f"calls={rec['calls']} ${rec['cost']:.4f}", flush=True)
    return rec


def main():
    arm, level, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seeds = []
    for a in sys.argv[4:]:
        lo, _, hi = a.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    out = HERE / "runs" / f"chain-{arm}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    tasks = [make_task(s, n, level) for s in seeds if make_task(s, n, level)["id"] not in done]
    with ThreadPoolExecutor(int(__import__("os").environ.get("CONC", "3"))) as ex:
        list(ex.map(lambda t: one(arm, t, out), tasks))
    print(ledger.report())


if __name__ == "__main__":
    main()
