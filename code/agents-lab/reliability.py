"""Part 12: long-horizon reliability on top of the lab harness.

  run_durable      the Part 1 loop with what a long run needs: a checkpoint after every step (and,
                   optionally, a write-ahead checkpoint before tools run), resume from it, hard
                   budgets on steps, tokens and cost, a stopping rule for repeated failures, and
                   an optional advisory task budget (beta)
  Faults           simulated tool faults: transient errors, timeouts after the side effect,
                   duplicate deliveries, and errors that persist for k attempts
  retrying         harness-level retries with capped exponential backoff and full jitter
  Payments         a side-effecting backend that can deduplicate on an idempotency key
  call_id()        the tool_use id of the call being executed, for keys derived from it

Everything here is local and simulated: no tool has a real external side effect.
harness.py and ledger.py are imported, not changed.
"""

import contextvars
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402
from harness import Result, Step, Transcript, Tool  # noqa: E402

DISABLED = {"type": "disabled"}
TASK_BUDGET_BETA = "task-budgets-2026-03-13"
_CALL_ID = contextvars.ContextVar("call_id", default=None)


def call_id():
    """The tool_use id of the tool call now running (set by run_durable)."""
    return _CALL_ID.get()


class SimulatedCrash(RuntimeError):
    """Raised by run_durable to imitate the process dying at a chosen point."""


# ---- Faults ------------------------------------------------------------------------------------

class TransientError(RuntimeError):
    pass


class ToolTimeout(RuntimeError):
    pass


@dataclass
class Faults:
    """Seeded fault schedule for one run. Rates are per call; `stuck` maps an argument value to the
    number of attempts that fail before it succeeds (a large number means it never does)."""
    seed: int = 0
    transient: float = 0.0     # fails before doing anything
    timeout: float = 0.0       # does the work, then the caller sees a timeout
    duplicate: float = 0.0     # the request is delivered twice (at-least-once delivery)
    stuck: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.attempts = {}

    def wrap(self, fn, name, key_arg=None, effectful=True):
        def faulty(**kw):
            k = kw.get(key_arg) if key_arg else None
            if k in self.stuck:
                self.attempts[k] = self.attempts.get(k, 0) + 1
                if self.attempts[k] <= self.stuck[k]:
                    self.log.append((name, k, "stuck"))
                    raise TransientError("503 Service Unavailable: the service is temporarily unavailable. "
                                         "No changes were made.")
            r = self.rng.random()
            if r < self.transient:
                self.log.append((name, k, "transient"))
                raise TransientError("503 Service Unavailable: the service is temporarily unavailable. "
                                     "No changes were made.")
            if effectful and r < self.transient + self.timeout:
                fn(**kw)                       # the work happens...
                self.log.append((name, k, "timeout"))
                raise ToolTimeout("Request timed out after 30 s with no response. The request may or "
                                  "may not have been processed.")
            if effectful and r < self.transient + self.timeout + self.duplicate:
                fn(**kw)                       # ...or happens twice, and the caller sees one reply
                self.log.append((name, k, "duplicate"))
            return fn(**kw)
        return faulty


def retrying(fn, attempts=3, base=0.5, cap=8.0, rng=None, on=(TransientError, ToolTimeout), sleeps=None):
    """Retry fn on transient errors and timeouts with capped exponential backoff and full jitter.
    The sleep is recorded in `sleeps` instead of slept, so experiments run on a simulated clock."""
    rng = rng or random.Random(0)

    def wrapped(**kw):
        for i in range(attempts):
            try:
                return fn(**kw)
            except on:
                if i == attempts - 1:
                    raise
                delay = rng.uniform(0, min(cap, base * 2 ** i))  # full jitter
                if sleeps is not None:
                    sleeps.append(delay)
    return wrapped


# ---- A side-effecting backend with optional idempotency ----------------------------------------

class Payments:
    """Orders and refunds in memory. With `dedupe`, a refund carrying an idempotency key that was
    seen before returns the stored reply and changes nothing (the Stripe-style contract)."""

    def __init__(self, orders, dedupe=False):
        self.orders = {k: {"amount": v, "refunds": []} for k, v in orders.items()}
        self.dedupe, self.seen = dedupe, {}

    def get(self, order_id):
        o = self.orders.get(order_id)
        if o is None:
            raise KeyError(f"No order {order_id}")
        done = sum(o["refunds"])
        return (f"Order {order_id}: amount {o['amount']:.2f} EUR, refunded {done:.2f} EUR "
                f"in {len(o['refunds'])} refund(s).")

    def refund(self, order_id, amount, key=None):
        if self.dedupe and key is not None and key in self.seen:
            return self.seen[key]
        o = self.orders.get(order_id)
        if o is None:
            raise KeyError(f"No order {order_id}")
        o["refunds"].append(round(float(amount), 2))
        reply = f"Refunded {float(amount):.2f} EUR on order {order_id} (refund #{len(o['refunds'])})."
        if self.dedupe and key is not None:
            self.seen[key] = reply
        return reply

    def snapshot(self):
        return json.loads(json.dumps({"orders": self.orders, "seen": self.seen}))


# ---- Stopping rule ------------------------------------------------------------------------------

class RepeatedFailureRule:
    """After `k` consecutive failures of the same tool call (name and arguments), the harness
    answers the next identical call itself and tells the model to stop retrying it."""

    def __init__(self, k=3):
        self.k, self.fails = k, {}

    def key(self, block):
        return f"{block['name']}:{json.dumps(block['input'], sort_keys=True)}"

    def blocked(self, block):
        return self.fails.get(self.key(block), 0) >= self.k

    def update(self, block, result):
        k = self.key(block)
        self.fails[k] = self.fails.get(k, 0) + 1 if result.get("is_error") else 0

    def message(self, block):
        return (f"Not run: this exact call has failed {self.k} times in a row. Stop retrying it. "
                f"Finish the rest of the task and report this item as failed.")


# ---- The loop ------------------------------------------------------------------------------------

def _run_tools(tools, blocks, rule):
    out = []
    for b in blocks:
        if rule and rule.blocked(b):
            out.append({"type": "tool_result", "tool_use_id": b["id"], "content": rule.message(b),
                        "is_error": True})
            continue
        token = _CALL_ID.set(b["id"])
        try:
            res = {"type": "tool_result", "tool_use_id": b["id"],
                   "content": str(tools[b["name"]].fn(**b["input"]))}
        except Exception as e:  # the model sees the error and decides what to do
            res = {"type": "tool_result", "tool_use_id": b["id"],
                   "content": f"{type(e).__name__}: {e}", "is_error": True}
        finally:
            _CALL_ID.reset(token)
        if rule:
            rule.update(b, res)
        out.append(res)
    return out


def _create(cl, part, note, task_budget, **req):
    if not task_budget:
        return ledger.create(cl, part, note=note, **req)
    # Beta endpoint: the same guard and record that ledger.create does.
    est_in = cl.messages.count_tokens(**{k: v for k, v in req.items()
                                         if k in ("model", "messages", "system", "tools", "thinking")}).input_tokens
    ledger.guard(part, ledger.estimate_cost(req["model"], est_in, req["max_tokens"]))
    oc = dict(req.pop("output_config", {}), task_budget={"type": "tokens", "total": task_budget})
    resp = cl.beta.messages.create(betas=[TASK_BUDGET_BETA], output_config=oc, **req)
    ledger.record(part, req["model"], resp.usage, note)
    return resp


def run_durable(task, tools, system, *, part, client, model="claude-haiku-5-5", effort="medium",
                thinking=DISABLED, max_tokens=4000, max_steps=60, max_cost=0.10, max_total_tokens=None,
                rule=None, checkpoint=None, write_ahead=False, state=None, crash_after=None,
                task_budget=None, transcript=None, note="a12"):
    """Run the agent loop. With `checkpoint` (a path), the loop saves messages, step records and
    `state()` after every step, and resumes from that file if it exists. With `write_ahead`, it also
    saves the model's tool calls before running them, so a resume re-runs those exact calls (same
    tool_use ids) instead of asking the model again. `crash_after=n` (or a function of the step
    number and its tool calls) raises SimulatedCrash after the tools of that step have run but
    before the step is checkpointed: the worst moment to die."""
    tools = {t.name: t for t in tools}
    request = dict(model=model, max_tokens=max_tokens, tools=[t.schema() for t in tools.values()],
                   system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                   output_config={"effort": effort}, thinking=thinking,
                   cache_control={"type": "ephemeral"})
    log = Transcript(transcript)
    ck = Path(checkpoint) if checkpoint else None
    result = Result(answer="", stop="max_steps")
    messages = [{"role": "user", "content": task}]
    if ck and ck.exists():  # resume
        saved = json.loads(ck.read_text())
        messages = saved["messages"]
        result.steps = [Step(**s) for s in saved["steps"]]
        log.write(event="resume", steps=len(result.steps))
        last = messages[-1]
        if last["role"] == "assistant":  # a write-ahead record: run the saved calls, same ids
            calls = [b for b in last["content"] if b["type"] == "tool_use"]
            if not calls:  # the run had already finished
                result.stop = "end_turn"
                result.answer = "".join(b.get("text", "") for b in last["content"] if b["type"] == "text")
                result.messages = messages
                return result
            messages.append({"role": "user", "content": _run_tools(tools, calls, rule)})
    else:
        log.write(event="task", task=task, **request)

    def save():
        if ck:
            tmp = ck.with_suffix(".tmp")
            tmp.write_text(json.dumps({"messages": messages, "steps": [s.__dict__ for s in result.steps],
                                       "state": state() if state else None}))
            tmp.replace(ck)  # atomic: a crash mid-write leaves the old checkpoint intact

    while len(result.steps) < max_steps:
        if result.cost >= max_cost:
            result.stop = "max_cost"
            break
        if max_total_tokens and sum(s.context + s.output_tokens for s in result.steps) >= max_total_tokens:
            result.stop = "max_total_tokens"
            break
        n = len(result.steps) + 1
        t0 = time.perf_counter()
        resp = _create(client, part, f"{note} step {n}", task_budget, messages=messages, **request)
        u = resp.usage
        calls = [b.model_dump(exclude_none=True) for b in resp.content if b.type == "tool_use"]
        step = Step(n, u.input_tokens, u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0,
                    u.output_tokens, round(time.perf_counter() - t0, 3), ledger.cost_of(model, u),
                    resp.stop_reason, [f"{c['name']}({json.dumps(c['input'])})" for c in calls])
        result.steps.append(step)
        content = [b.model_dump(exclude_none=True) for b in resp.content]
        log.write(event="step", **step.__dict__, content=content)
        messages.append({"role": "assistant", "content": content})
        if resp.stop_reason != "tool_use":
            result.stop = resp.stop_reason
            result.answer = "".join(b.text for b in resp.content if b.type == "text")
            save()
            break
        if write_ahead:
            save()
        results = _run_tools(tools, calls, rule)
        log.write(event="tool_results", n=n, results=results)
        if crash_after == n or (callable(crash_after) and crash_after(n, calls)):
            log.write(event="crash", n=n)
            raise SimulatedCrash(f"crashed after the tools of step {n} ran")
        messages.append({"role": "user", "content": results})
        save()

    result.messages = messages
    log.write(event="result", answer=result.answer, stop=result.stop, steps=len(result.steps),
              cost=result.cost)
    return result
