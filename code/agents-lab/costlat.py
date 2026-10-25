"""Part 15: cost and latency of agents, one lever at a time.

  run_loop    Part 3's loop with prompt caching as a parameter: on (5-minute or 1-hour entries) or off
  run_team    a lead agent (any model) whose only tool is `delegate`, with Haiku workers (Part 7's design)
  run_batch   the same loop for many tasks at once, each round of model calls sent as one Message Batch
  per_task    cost per completed task, latency percentiles and cache hit rate for a list of run records

Every synchronous call goes through ledger.create. Batches call ledger.guard with a worst-case
estimate before submitting and ledger.record(batch=True) for each result. harness.py, react.py,
multiagent.py and ledger.py are imported, not changed.
"""

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402
from harness import Result, Tool, Transcript, run_tool  # noqa: E402
from multiagent import Run, loop  # noqa: E402
from react import ADAPTIVE, DISABLED, TStep  # noqa: E402,F401


def cache_mark(cache):
    """cache=False: no breakpoints. True or "5m": 5-minute entries. "1h": 1-hour entries (2x write price)."""
    if not cache:
        return None
    return {"type": "ephemeral", "ttl": "1h"} if cache == "1h" else {"type": "ephemeral"}


def build_request(tools, system, *, model, effort, thinking, cache, max_tokens):
    mark = cache_mark(cache)
    request = dict(model=model, max_tokens=max_tokens, tools=[t.schema() for t in tools],
                   system=[{"type": "text", "text": system}], output_config={"effort": effort},
                   thinking=thinking)
    if mark:
        request["system"][0]["cache_control"] = mark  # breakpoint 1: tools + system prompt
        request["cache_control"] = mark               # breakpoint 2: moves forward with the conversation
    return request


def make_step(n, resp, model, latency, batch=False):
    u, calls = resp.usage, [b for b in resp.content if b.type == "tool_use"]
    details = getattr(u, "output_tokens_details", None)
    return TStep(n, u.input_tokens, u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0,
                 u.output_tokens, round(latency, 3), ledger.cost_of(model, u, batch=batch), resp.stop_reason,
                 [f"{b.name}({json.dumps(b.input)})" for b in calls],
                 getattr(details, "thinking_tokens", 0) or 0)


def run_loop(task, tools, system, *, part, client, model="claude-haiku-5-5", effort="medium",
             thinking=ADAPTIVE, cache=True, max_steps=20, max_cost=0.05, max_tokens=16000,
             transcript=None, label="a15"):
    """One agent on one task. `client` is a multiagent.Limited, which times each call in its thread."""
    tmap = {t.name: t for t in tools}
    request = build_request(tools, system, model=model, effort=effort, thinking=thinking, cache=cache,
                            max_tokens=max_tokens)
    log = Transcript(transcript)
    log.write(event="task", task=task, cache=cache, **request)
    messages = [{"role": "user", "content": task}]
    result = Result(answer="", stop="max_steps")
    for n in range(1, max_steps + 1):
        if result.cost >= max_cost:
            result.stop = "max_cost"
            break
        resp = ledger.create(client, part, note=f"{label} step {n}", messages=messages, **request)
        step = make_step(n, resp, model, client.local.last)
        result.steps.append(step)
        log.write(event="step", **step.__dict__, content=[b.model_dump(exclude_none=True) for b in resp.content])
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            result.stop = resp.stop_reason
            result.answer = "".join(b.text for b in resp.content if b.type == "text")
            break
        results = [run_tool(tmap, b) for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "user", "content": results})
        log.write(event="tool_results", n=n, results=results)
    log.write(event="result", answer=result.answer, stop=result.stop, steps=len(result.steps), cost=result.cost)
    return result


# ---- A lead with workers (Part 7's design) ----------------------------------------------------

def run_team(question, worker_tools, lead_system, worker_system, *, part, client,
             lead_model="claude-sonnet-5-5", worker_model="claude-haiku-5-5", effort="medium",
             max_cost=0.50, worker_max_cost=0.05, transcript_dir=None):
    """multiagent.loop for the lead and each worker; the delegate tool starts a worker per call."""
    tdir = Path(transcript_dir) if transcript_dir else None
    if tdir:
        tdir.mkdir(parents=True, exist_ok=True)
    lead = Run("lead")
    count, lock = iter(range(1, 1000)), threading.Lock()

    def delegate(task):
        with lock:
            k = next(count)
            w = Run("worker", brief=task)
            lead.workers.append(w)
        loop(task, worker_tools, worker_system, part=part, role=f"worker{k}", client=client,
             model=worker_model, effort=effort, max_steps=20, max_cost=worker_max_cost,
             transcript=tdir / f"worker{k:02d}.jsonl" if tdir else None, run=w)
        return w.answer if w.stop == "end_turn" else f"[worker stopped early: {w.stop}] {w.answer}".strip()

    tool = Tool("delegate", "Start a worker agent with a fresh context and the Saltmere tools. It sees "
                "only the brief you write here and returns its final report.",
                {"task": {"type": "string", "description": "The complete, self-contained brief for the worker"}},
                delegate)
    return loop(question, [tool], lead_system, part=part, role="lead", client=client, model=lead_model,
                effort=effort, max_steps=12, max_cost=max_cost,
                transcript=tdir / "lead.jsonl" if tdir else None, run=lead)


# ---- The same loop through the Message Batches API ----------------------------------------------

def _plain(block):
    """A response content block as a JSON-ready dict for the next request."""
    return block.model_dump(exclude_none=True)


def run_batch(jobs, system, *, part, client, model="claude-haiku-5-5", effort="medium", thinking=ADAPTIVE,
              cache=True, max_steps=20, max_tokens=16000, poll_s=10, log_path=None, label="a15 batch"):
    """Advance many agent runs in lockstep: each round sends every unfinished run's next request as
    one batch, waits for it to end, runs the tools, and repeats. `jobs` maps a custom_id to
    (task text, tools). Returns {custom_id: Result} and a list of per-round timings.
    `client` is a plain SDK client (batches are not rate-limited per request like messages)."""
    log = Transcript(log_path)
    state = {}
    for cid, (task, tools) in jobs.items():
        state[cid] = dict(tools={t.name: t for t in tools}, messages=[{"role": "user", "content": task}],
                          request=build_request(tools, system, model=model, effort=effort, thinking=thinking,
                                                cache=cache, max_tokens=max_tokens),
                          result=Result(answer="", stop="max_steps"), done=False, finished_at=None)
    rounds, t_start = [], time.time()
    for n in range(1, max_steps + 1):
        active = [cid for cid, s in state.items() if not s["done"]]
        if not active:
            break
        reqs = [{"custom_id": cid, "params": {**state[cid]["request"], "messages": state[cid]["messages"]}}
                for cid in active]
        # Worst case for the whole batch: every input token written to the cache, every request
        # producing max_tokens, at half price. Tokens are bounded by characters / 2 (no count call).
        est = sum(ledger.estimate_cost(model, len(json.dumps(r["params"], default=str)) // 2, max_tokens)
                  for r in reqs) * 0.5
        ledger.guard(part, est)
        t0 = time.time()
        batch = client.messages.batches.create(requests=reqs)
        while True:
            time.sleep(poll_s)
            b = client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
        t1 = time.time()
        rec = dict(round=n, batch_id=batch.id, requests=len(reqs), submitted=t0 - t_start,
                   turnaround_s=round(t1 - t0, 1), counts=b.request_counts.model_dump())
        rounds.append(rec)
        log.write(event="round", **rec)
        for item in client.messages.batches.results(batch.id):
            s = state[item.custom_id]
            if item.result.type != "succeeded":  # errored, expired or canceled: stop this run
                s["done"], s["result"].stop, s["finished_at"] = True, item.result.type, t1 - t_start
                log.write(event="failed", id=item.custom_id, n=n, result=item.result.model_dump(exclude_none=True))
                continue
            msg = item.result.message
            ledger.record(part, model, msg.usage, note=f"{label} round {n}", batch=True)
            step = make_step(n, msg, model, t1 - t0, batch=True)
            s["result"].steps.append(step)
            log.write(event="step", id=item.custom_id, **step.__dict__,
                      content=[_plain(c) for c in msg.content])
            s["messages"].append({"role": "assistant", "content": [_plain(c) for c in msg.content]})
            if msg.stop_reason != "tool_use":
                s["done"], s["result"].stop, s["finished_at"] = True, msg.stop_reason, t1 - t_start
                s["result"].answer = "".join(c.text for c in msg.content if c.type == "text")
                continue
            results = [run_tool(s["tools"], c) for c in msg.content if c.type == "tool_use"]
            s["messages"].append({"role": "user", "content": results})
        print(f"round {n}: {len(reqs)} requests, {rec['turnaround_s']} s", flush=True)
    for s in state.values():
        if not s["done"]:
            s["finished_at"] = time.time() - t_start
    return {cid: s["result"] for cid, s in state.items()}, {cid: s["finished_at"] for cid, s in state.items()}, rounds


# ---- Summaries ------------------------------------------------------------------------------------

def pct(xs, q):
    """Percentile by linear interpolation (q in 0-100)."""
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def per_task(records):
    """records: dicts with correct, cost, wall_s, input, cache_read. Cost per completed task is the
    arm's total spend divided by the runs that got the right answer: what a correct answer costs
    when failed runs are paid for too."""
    n, ok = len(records), sum(r["correct"] for r in records)
    cost = sum(r["cost"] for r in records)
    walls = [r["wall_s"] for r in records]
    inp = sum(r["input"] for r in records)
    return dict(runs=n, correct=ok, cost_per_run=cost / n, cost_per_completed=cost / ok if ok else float("inf"),
                median_s=pct(walls, 50), p95_s=pct(walls, 95),
                cache_hit=sum(r["cache_read"] for r in records) / inp if inp else 0.0)
