"""Experiments 2-4 on the refund task.

Usage: python run_refunds.py ARM SEEDS...
  Faults (15% transient on every call; on refunds 15% timeout-after-commit, 10% duplicate delivery):
    bare         the model sees every fault and decides what to do
    retry        the harness retries transient errors and timeouts (3 attempts, backoff + jitter)
    retry_idem   harness retries, and the backend deduplicates on a key = the tool_use id
    model_key    no harness retries; the refund tool takes an idempotency_key the model chooses
  Crash after the first step that issues a refund, then recover (no other faults):
    restart, resume, resume_callid, wal_callid, resume_business
  One order's refund fails k times in a row (no other faults):
    stuck<k>-none, stuck<k>-retry, stuck<k>-rule3   (k = 2, 4 or 1000)
"""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import anthropic  # noqa: E402
import ledger  # noqa: E402
from reliability import Faults, Payments, RepeatedFailureRule, SimulatedCrash, retrying, run_durable  # noqa: E402
from refunds import SYSTEM, grade, make_task, make_tools  # noqa: E402

HERE = Path(__file__).parent
CL = anthropic.Anthropic(api_key=ledger.client().api_key, max_retries=6)
FAULTS = dict(transient=0.15, timeout=0.15, duplicate=0.10)
CRASH_ARMS = {"restart": (None, False, None), "resume": (None, False, "ck"), "resume_callid": ("call_id", False, "ck"),
              "wal_callid": ("call_id", True, "ck"), "resume_business": ("business", False, "ck")}


def setup(arm, task):
    """Backend, tools, run options and fault log for one run."""
    seed, opts = task["seed"], {}
    if arm in ("bare", "retry", "retry_idem", "model_key"):
        f = Faults(seed=seed, **FAULTS)
        key = {"retry_idem": "call_id", "model_key": "model"}.get(arm)
        backend = Payments(task["orders"], dedupe=key is not None)
        sleeps = []

        def wrap(fn, name):
            w = f.wrap(fn, name, key_arg="order_id", effectful=(name == "refund"))
            return retrying(w, rng=random.Random(seed), sleeps=sleeps) if arm.startswith("retry") else w
        return backend, make_tools(backend, key, wrap), opts, {"faults": f, "sleeps": sleeps}
    if arm in CRASH_ARMS:
        key, wal, ck = CRASH_ARMS[arm]
        backend = Payments(task["orders"], dedupe=key is not None)
        return backend, make_tools(backend, key), {"write_ahead": wal}, {"ck": ck}
    if arm.startswith("stuck"):
        k, rule = arm[5:].split("-")
        stuck_id = list(task["orders"])[2]
        f = Faults(seed=seed, stuck={stuck_id: int(k)})
        backend = Payments(task["orders"])
        sleeps = []

        def wrap(fn, name):
            if name != "refund":
                return fn
            w = f.wrap(fn, name, key_arg="order_id", effectful=False)
            return retrying(w, rng=random.Random(seed), sleeps=sleeps) if rule == "retry" else w
        if rule == "rule3":
            opts["rule"] = RepeatedFailureRule(3)
        return backend, make_tools(backend, None, wrap), opts, {"faults": f, "stuck": stuck_id, "sleeps": sleeps}
    raise ValueError(arm)


def one(arm, task, out):
    tdir = HERE / "runs" / "tx" / f"refunds-{arm}"
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"{task['id']}.jsonl"
    for attempt in range(4):
        tpath.unlink(missing_ok=True)
        backend, tools, opts, extra = setup(arm, task)
        ckpath = tdir / f"{task['id']}.ckpt.json"
        ckpath.unlink(missing_ok=True)
        common = dict(part="a12", client=CL, max_steps=40, max_cost=0.10, transcript=tpath,
                      note=f"a12 refunds {arm}")
        try:
            t0, crashed, steps_before = time.time(), False, 0
            if "ck" in extra:  # crash experiment
                first_refund = lambda n, calls: any(c["name"] == "refund" for c in calls)
                ck = ckpath if extra["ck"] else None
                try:
                    r = run_durable(task["question"], tools, SYSTEM, checkpoint=ck, crash_after=first_refund,
                                    **opts, **common)
                except SimulatedCrash:
                    crashed = True
                steps_before = len(json.loads(ckpath.read_text())["steps"]) if ck and ckpath.exists() else 0
                # Recovery: the backend (the payment system) keeps its state; the agent process restarts.
                r0_cost = sum(json.loads(l).get("cost", 0) for l in open(tpath) if '"event": "step"' in l)
                r = run_durable(task["question"], tools, SYSTEM, checkpoint=ck, **opts, **common)
            else:
                r = run_durable(task["question"], tools, SYSTEM, **opts, **common)
            break
        except (anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.RateLimitError) as e:
            print("retry", task["id"], type(e).__name__, flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        return None
    s = r.steps
    # Total cost of the run: every step written to the transcript (before and after any crash).
    steps_all = [json.loads(l) for l in open(tpath) if '"event": "step"' in l]
    rec = {"arm": arm, "id": task["id"], "seed": task["seed"], "stop": r.stop,
           "calls": len(steps_all), "tool_calls": sum(len(x["tool_calls"]) for x in steps_all),
           "refund_calls": sum(sum(c.startswith("refund(") for c in x["tool_calls"]) for x in steps_all),
           "get_calls": sum(sum(c.startswith("get_order(") for c in x["tool_calls"]) for x in steps_all),
           "cost": sum(x["cost"] for x in steps_all), "output": sum(x["output_tokens"] for x in steps_all),
           "wall_s": round(time.time() - t0, 1), **grade(task, backend, r.answer)}
    if "faults" in extra:
        rec["fault_log"] = extra["faults"].log
    if "sleeps" in extra:
        rec["sim_sleep_s"] = round(sum(extra["sleeps"]), 2)
    if "stuck" in extra:
        rec["stuck_id"] = extra["stuck"]
        rec["stuck_attempts"] = extra["faults"].attempts.get(extra["stuck"], 0)
        rec["stuck_ok"] = backend.orders[extra["stuck"]]["refunds"] == [task["orders"][extra["stuck"]]]
        rec["stuck_reported_failed"] = extra["stuck"] not in rec["claimed"]
    if "ck" in extra:
        rec["crashed"] = crashed
        rec["steps_resumed_from"] = steps_before
        rec["cost_before_crash"] = round(r0_cost, 6)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{task['id']:5s} {arm:16s} ok={rec['ok']} extra={rec['extra_refunds']} missed={rec['missed']} "
          f"true_report={rec['report_true']} calls={rec['calls']} ${rec['cost']:.4f}", flush=True)
    return rec


def main():
    arm = sys.argv[1]
    seeds = []
    for a in sys.argv[2:]:
        lo, _, hi = a.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    out = HERE / "runs" / f"refunds-{arm}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    tasks = [make_task(s) for s in seeds if f"t{s}" not in done]
    with ThreadPoolExecutor(int(__import__("os").environ.get("CONC", "3"))) as ex:
        list(ex.map(lambda t: one(arm, t, out), tasks))
    print(ledger.report())


if __name__ == "__main__":
    main()
