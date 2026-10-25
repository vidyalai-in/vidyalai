---
title: "Long-Horizon Reliability: Compounding Errors, Retries and Checkpoints"
description: "Why long agent runs fail more often than their steps do, measured on Claude Haiku 5.5, and what retries, idempotency keys, checkpoints and budgets fix."
slug: "agent-reliability"
category: "agents-tool-use"
tags: ["agents", "tool-use", "evals", "cost"]
series: "agentic-ai"
seriesOrder: 12
pubDate: 2026-10-25
---

## TL;DR

A long agent run fails if any of its steps fails, so success falls roughly like `p^n`: per-step success `p` raised to the number of dependent steps `n`. On a chain task of adjustable length, Claude Haiku 5.5 got 97.3% of 3,829 steps right, and success fell from 67% at 5 steps to 30% at 80. One per-step rate got both ends wrong, because 19% of runs failed on the first step and 1.6% per step after that. With faults injected into a refund tool, harness retries without idempotency keys raised the share of tickets with a double refund from 63% to 93%, and a key derived from the tool call's id brought it to 0 of 30. After a simulated crash, resuming from a checkpoint one step stale refunded all 80 orders twice, while a write-ahead checkpoint refunded none twice. Every fix that worked lives in the harness.

## Why it matters

Claude Haiku 5.5 got 97.3% of its steps right on the chain task in this part. Over an 80-step task, that should leave about 1 run in 9 correct; 3 in 10 were. Over 5 steps it should leave 87%; 67% were. Both surprises have the same cause: almost a fifth of the runs went wrong on the very first step, and the steps after that were safer than the average said.

Long runs also meet a misbehaving world: a tool call times out after the payment went through, a queue delivers the same request twice, or the agent's process dies between issuing a refund and recording it. Each becomes wrong money unless the harness handles it. [Detecting and Recovering from Agent Loops](/blog/agent-loop-detection-recovery/) covered loops and escalation, and [Building Tool-Using Agents That Work Reliably](/blog/reliable-tool-using-agents/) did the 0.95^20 arithmetic. This part measures how fast success decays with length, and which harness mechanisms keep a long run correct when tools and processes fail.

## The intuition

Think of assembling flat-pack furniture from a 40-step booklet while your phone keeps ringing.

Each step is easy and can still go quietly wrong: a panel reversed at step 7 only shows at step 38, when the doors won't close. That is compounding error. A bookmark saying which step you finished, for when the phone pulls you away, is a checkpoint. Checking your order history before reordering a screw, after the shop's website timed out on the first order, is an idempotent retry.

The analogy breaks twice. You can look at the furniture; an agent sees only its transcript and tool results, so a silent mistake stays silent. And you know you were interrupted; a crashed agent process knows nothing, so what it needs to resume must be written down before the crash, by the harness.

## Step 1: compounding error, measured

### A task with a length dial

Measuring `p^n` needs dependent steps, a per-step error rate above zero, and silent errors. In the chain task, every three-digit code is a record with a multiplier `m` and an offset `d`, and the next code is the last three digits of `code × m + d`. The agent opens records with one tool, `open_record(code)`, and does the arithmetic itself, with no calculator. With `m` between 101 and 999, each step is a three-by-three-digit multiplication.

Every code exists, so a wrong product lands on a real record and the agent carries on down the wrong chain with no error message. Each hop draws fresh values for `m` and `d`, so a wrong turn rejoins the right path about once in 1,000 steps, and the correct chain never visits a code twice. An earlier version with one fixed table fell into cycles; in one 40-step run the agent noticed "the cycle repeats with period 6" and skipped the remaining records.

Because every code is valid, each step can be graded on its own: was the next code right for the record the agent was actually on? That gives a per-step success rate over every step of every run, including steps after an earlier mistake.

```python
def step_record(state, answer):
    """Per-step correctness, conditional on where the agent actually was: step i is right if the
    code it moved to (the next open, or the final answer) equals f(code it had opened)."""
    seq = [c for c, _, _ in state.opened]
    a = final(answer)
    moved = seq[1:] + ([int(a)] if a is not None else [])
    steps = [int(moved[i] == nxt(*state.opened[i])) for i in range(min(len(seq), len(moved)))]
    return {"opened": seq, "steps": steps, "start_ok": bool(seq) and seq[0] == state.task["start"]}
```

All runs use `claude-haiku-5-5` with thinking disabled and effort `medium`, prompt caching, and the loop from the lab harness extended in `reliability.py`. Each chain length ran on 30 different chains (20 at 80 steps), and rates come with a 95% Wilson interval, a confidence interval for a proportion that stays sensible near 0% and 100%.

### Measured against p^n

| Dependent steps | Runs | Solved (95% CI) | `p^n` with pooled `p` | No error so far, from the measured first-error rate at each step | Input tokens read per run | Cost per run |
|---|---|---|---|---|---|---|
| 5 | 30 | 66.7% (48.8-80.8) | 87.4% | 75.0% | 5,509 | $0.00041 |
| 10 | 30 | 70.0% (52.1-83.3) | 76.3% | 70.6% | 13,429 | $0.00078 |
| 20 | 30 | 56.7% (39.2-72.6) | 58.3% | 55.0% | 38,834 | $0.00166 |
| 40 | 30 | 46.7% (30.2-63.9) | 34.0% | 44.0% | 123,236 | $0.00364 |
| 80 | 20 | 30.0% (14.5-51.9) | 11.5% | 24.0% | 438,630 | $0.00929 |

Of 3,829 graded steps, 3,727 were right: a pooled `p = 0.973` (95% CI 0.968-0.978). Raised to the power `n`, it fits the middle of the table (58.3% predicted at 20 steps, 56.7% measured) and misses both ends: 87.4% predicted at 5 steps against 66.7% solved, and 11.5% at 80 against 30.0%.

The errors were front-loaded. On the very first step, 26 of 140 runs went wrong (18.6%); on the second, 4 of the 114 still on the right path. From the third step on, the chance of a run's first error was 1.6% per step (34 of 2,132). A single `p` averages a risky opening with a steady middle, so it is too hopeful for short tasks and too gloomy for long ones.

![Line chart of tasks solved against the number of dependent steps, from 0 to 80. Measured points with 95% intervals: 66.7% at 5 steps, 70.0% at 10, 56.7% at 20, 46.7% at 40 and 30.0% at 80. A dashed curve, p to the n with p equal to 0.973, starts at 100% and falls to 11.5% at 80 steps, above the measured points at 5 and 10 steps and below them at 40 and 80. A solid curve built from the measured chance of a first error at each step drops steeply over the first two steps to about 79% and then declines slowly to 24% at 80 steps, passing through or near every measured interval. A square marker at 40 steps shows the restate prompt at 60% (12 of 20).](/diagrams/a12-success-vs-steps.svg "Tasks solved against chain length on claude-haiku-5-5 (thinking off, effort medium; 30 chains per length, 20 at 80 steps). The steep start of the solid curve is the first step.")

The solid curve is `p^n` with a separate `p` for each step: the measured chance that a run still on the right path gets that step right. It lands inside every measured interval. Compounding is real, as long as `p` is the chance that step `k` goes right given that every step before it did.

The length at which an agent succeeds half the time is `n50 = ln(0.5) / ln(p)`: 26 steps at the pooled 0.973, and 43 at 0.984 for a run past its first two steps. Length costs mostly in reading. Each request re-reads the whole history, so input tokens grew 80-fold from 5 to 80 steps while the chain grew 16-fold; cost grew 23-fold, because the prompt cache served 87% of that input at 5 steps and 98% at 80, at a tenth of the input price.

### Do mistakes breed mistakes?

Sinha et al. found that models make more mistakes once their context holds their own earlier mistakes, an effect they call self-conditioning. On this task, Haiku didn't: after a run's first error, its per-step error rate was 2.63% (38 of 1,443 steps), against 2.68% before (64 of 2,386). Wrong numbers in the transcript did not make its multiplication worse.

A wrong belief did. In a pilot version that let chains revisit codes, one run slipped at step 3 onto the code it was on and wrote "Record 151 maps to itself, so the chain is stuck in a loop." It stopped computing and opened 151 again and again: 38 wrong steps out of 40. An arithmetic slip cost one step; a slip that changed the agent's picture of the world cost the rest of the run.

### The cheapest way to raise p

The transcripts pointed at a lever. On steps where the agent first copied the record ("Record 186: m = 638, d = 649") and then multiplied, 1.3% of steps after the second went wrong; where it went straight to the arithmetic, 3.2%. That is a correlation, so the test is to ask for it. A second arm added three lines to the system prompt: copy the record, write out the multiplication and the addition, then take the last three digits. Both arms ran the same 20 forty-step chains:

| System prompt | Wrong steps | Wrong first steps | Solved at 40 steps (95% CI) | Cost per run |
|---|---|---|---|---|
| Plain | 26 of 800 (3.3%) | 7 of 20 | 8/20, 40% (21.9-61.3) | $0.00360 |
| Copy the record, then compute | 12 of 800 (1.5%) | 2 of 20 | 12/20, 60% (38.7-78.1) | $0.00389 |

The step error rate halved for 8% more cost, mostly on the first step. The 20-point rise in solved tasks is within noise on 20 chains: 6 chains flipped from wrong to right and 2 the other way, p = 0.29 on an exact paired test.

## Step 2: faults in the tools, and who retries

### The refund task

The second task has side effects. A support ticket lists eight orders and their amounts; the agent has `get_order` (an order and its refunds) and `refund` (moves money) on a simulated payments backend inside the same Python process. Each order must be refunded exactly once, and grading reads the backend: an order passes if it holds exactly one full refund.

A seeded fault injector wraps the tools. Every call has a 15% chance of a transient error (`503 Service Unavailable ... No changes were made.`). Refund calls also have a 15% chance of a timeout after the refund went through (`The request may or may not have been processed.`) and 10% of a duplicate delivery: the backend receives the request twice and the caller sees one normal reply. The last two are the failures that double-charge people.

Four harness designs ran on the same 30 seeded tickets:

- Model only: every fault reaches the model, which decides what to do.
- Harness retries: the harness retries transient errors and timeouts up to 3 times, unseen by the model, with capped exponential backoff (a wait that doubles after each failure, up to 8 seconds) and full jitter (a random wait between zero and that backoff, so many clients' retries spread out).
- Harness retries plus a key: every refund also carries an idempotency key, a label for one intended operation that lets the server recognise a repeat. The key is the tool call's `tool_use` id, which stays the same across the harness's retries and a duplicate delivery. The backend returns the stored first reply for any repeated key, as [Stripe's API](https://docs.stripe.com/api/idempotent_requests) does.
- Model-chosen key: no harness retries; the refund tool takes an `idempotency_key` argument, described as the key to reuse when retrying the same refund.

The retry and the deduplication are a few lines each in `reliability.py`:

```python
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
```

```python
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
```

One ticket with the third design, run from the experiment folder with the lab folder on `PYTHONPATH`:

```python
import random

import ledger
from reliability import Faults, Payments, retrying, run_durable
from refunds import SYSTEM, grade, make_task, make_tools

task = make_task(7)
faults = Faults(seed=7, transient=0.15, timeout=0.15, duplicate=0.10)
backend = Payments(task["orders"], dedupe=True)   # stores the first reply for every key


def wrap(fn, name):  # a flaky network under each tool, then harness retries on top
    flaky = faults.wrap(fn, name, key_arg="order_id", effectful=(name == "refund"))
    return retrying(flaky, rng=random.Random(7))


r = run_durable(task["question"], make_tools(backend, "call_id", wrap), SYSTEM,
                part="a12", client=ledger.client())
print(faults.log)
print({k: v for k, v in grade(task, backend, r.answer).items() if k != "claimed"},
      f"model calls={len(r.steps)} cost=${r.cost:.5f}")
```

```text
[('get_order', 'A-1791', 'transient'), ('get_order', 'A-9779', 'transient'), ('get_order', 'A-2542', 'transient'),
 ('get_order', 'A-6991', 'transient'), ('get_order', 'A-6991', 'transient'), ('refund', 'A-3471', 'transient'),
 ('refund', 'A-3471', 'timeout'), ('refund', 'A-2186', 'duplicate'), ('refund', 'A-2542', 'transient'),
 ('refund', 'A-6991', 'timeout'), ('refund', 'A-6991', 'transient'), ('refund', 'A-6991', 'transient')]
{'ok': 8, 'missed': 0, 'dup_orders': 0, 'extra_refunds': 0, 'wrong': 0, 'all_ok': True, 'report_true': True} model calls=4 cost=$0.00111
```

Twelve faults, and still exactly one refund per order. A-6991 is the subtle case: its first attempt timed out after the money moved, the next two hit 503s, and the model received the last error, "No changes were made." A model retry would have carried a new call id, so a new key; the model checked the order first. When a harness gives up after a timeout, it should report the worst error it saw, not the last.

### The results

Each arm ran 30 tickets (240 orders) with an average of 3.3 to 3.6 transient errors, 1.5 to 1.8 timeouts and 0.9 to 1.0 duplicate deliveries per ticket:

| Design (Haiku 5.5, thinking off) | All 8 refunded exactly once (95% CI) | Tickets with a double refund | Extra refunds, of 240 | Model calls per ticket | Cost per ticket |
|---|---|---|---|---|---|
| Model only | 10/30, 33% (19-51) | 19/30, 63% | 29 | 6.0 | $0.00154 |
| Harness retries, no key | 2/30, 7% (2-21) | 28/30, 93% | 72 | 3.9 | $0.00116 |
| Harness retries, key = `tool_use` id | 30/30, 100% (89-100) | 0/30 | 0 | 3.4 | $0.00096 |
| Model-chosen key, no harness retries | 30/30, 100% (89-100) | 0/30 | 0 | 5.8 | $0.00175 |

The model on its own was a careful retry policy. After every one of the 45 timeouts it saw, its next action on that order was `get_order`, never a blind retry. Of its 29 extra refunds, 28 were duplicate deliveries, which happen below the tool where no care in the model can see them. The 29th was a bookkeeping slip. Seven refunds went out in parallel and two timed out; the model checked both orders, but the check on one of them hit a 503, and it read that 503 as the refund's answer ("The 503s on A-5243 and A-4777 are safe to retry") and refunded again.

Harness retries without a key made things much worse. A retry after a timeout repeats a refund that already happened, so nearly every timeout became a double refund on top of every duplicate delivery: 72 extra refunds, and 2 of 30 tickets clean. The same retries with a key were best on every column, cost included, because the model never saw most faults: 3.4 model calls per ticket against 6.0. The simulated backoff added 1.4 seconds of waiting per ticket.

The model-chosen key also reached 30 of 30, reusing the key in all 7 refund retries, but the model still saw every fault and spent 5.8 calls per ticket checking them.

The backend's reply matters too. In tickets with a double refund, the final report still listed a doubled order as cleanly refunded in 7 of 19 (model only) and 10 of 28 (harness retries). The model caught the rest only because this backend says `refund #2`; an API that returns a bare `200 OK` gives it nothing to catch.

## Step 3: checkpoints and resume

### What a crash loses

A checkpoint is a saved copy of what a run needs to continue: messages, step records and the state of any tools the agent owns. `run_durable` writes one after every step, atomically (a temporary file, then a rename), and resumes from it if it exists:

```python
    def save():
        if ck:
            tmp = ck.with_suffix(".tmp")
            tmp.write_text(json.dumps({"messages": messages, "steps": [s.__dict__ for s in result.steps],
                                       "state": state() if state else None}))
            tmp.replace(ck)  # atomic: a crash mid-write leaves the old checkpoint intact
```

The dangerous moment is between running a tool and saving the checkpoint. The crash test kills the process exactly there, right after the first step that issues refunds; the payments backend keeps its state, as a real one would. With `write_ahead=True`, the loop also saves the model's tool calls, with their ids, before running them, and a resume re-runs those exact calls instead of asking the model again:

```python
        if write_ahead:
            save()
        results = _run_tools(tools, calls, rule)
        log.write(event="tool_results", n=n, results=results)
        if crash_after == n or (callable(crash_after) and crash_after(n, calls)):
            log.write(event="crash", n=n)
            raise SimulatedCrash(f"crashed after the tools of step {n} ran")
        messages.append({"role": "user", "content": results})
        save()
```

This is the same idea as a database's write-ahead log, which records an intended change before making it, and as [Temporal's](https://docs.temporal.io/activity-definition) advice to key an activity by workflow run ID plus activity ID, "consistent across retry attempts but unique among Workflow Executions."

### Five ways to recover

Each recovery ran on 10 tickets with no other faults. Haiku always read all eight orders in its first step and issued all eight refunds in parallel in its second, so the crash came right after eight refunds:

| Recovery after the crash | Idempotency key | Extra refunds, of 80 | Tickets all correct (95% CI) | Model calls per ticket | Cost per ticket |
|---|---|---|---|---|---|
| Restart from scratch | none | 0 | 10/10 (72-100) | 4.0 | $0.00117 |
| Resume from last checkpoint | none | 80 | 0/10 (0-28) | 4.4 | $0.00146 |
| Resume from last checkpoint | `tool_use` id | 80 | 0/10 (0-28) | 4.6 | $0.00155 |
| Resume from write-ahead checkpoint | `tool_use` id | 0 | 10/10 (72-100) | 3.0 | $0.00090 |
| Resume from last checkpoint | `refund:<order id>` | 0 | 10/10 (72-100) | 4.0 | $0.00124 |

Model calls and cost include the calls before the crash.

Restarting from scratch was safe here: the fresh agent read every order first, saw a full refund on each, and wrote "Issuing more refunds now would double-refund the customer, so I'm not calling `refund` on any of them."

Resuming from the last checkpoint was the worst option. That checkpoint ended one step before the crash, with eight `get_order` results saying "refunded 0.00 EUR". The resumed model trusted its context and issued all eight refunds again, in all 10 tickets:

```text
All eight orders match the ticket amounts and have no prior refunds. Issuing the full refunds now.
[resume]
All eight orders match the ticket amounts and have no prior refunds. Issuing the full refunds now.
Each refund reports "refund #2", but my pre-check showed zero prior refunds on every order.
```

A key made from the `tool_use` id didn't help, because the resumed model wrote new calls with new ids. Two fixes worked. The write-ahead checkpoint replayed the original calls with their original ids, which the backend recognised. A business key, `refund:<order id>`, meant "one full refund per order" whatever call produced it; it is simpler, but needs a natural identity for each operation. The write-ahead log works for any tool, and was the cheapest recovery, because nothing was asked twice.

## Step 4: budgets and stopping rules

### Who stops, and when

A stopping rule decides when to give up on a failing call. Here one refund per ticket fails with a 503 for its first `k` attempts, then succeeds: `k = 2`, `k = 4`, or `k = 1000` (never). Ten tickets each ran with the model alone, with the harness's 3-attempt retries, or with a harness rule that answers a call itself after 3 identical failures in a row and tells the model to stop.

| Failing refund recovers after | Model alone: refunded (attempts) | Harness retries: refunded (attempts) | Harness stop rule after 3 |
|---|---|---|---|
| 2 failures | 8/10 (2.8) | 10/10 (3.0) | not run |
| 4 failures | 0/10 (3.0) | 10/10 (5.0) | not run |
| never | 0/10 (2.9) | 0/10 (8.7) | 0/10 (3.0), rule never fired |

Haiku 5.5 brought its own stopping rule: check the order, retry, check again, and after two or three attempts report it as failed. Every report matched the backend, naming the stuck order on its `FAILED:` line when it stayed stuck. The harness rule never fired, because the model always stopped first; it is a backstop for the loops in [the loop-detection post](/blog/agent-loop-detection-recovery/).

The model's caution cost something at `k = 4`: it gave up in all 10 tickets on a dependency that would have recovered. Harness retries fixed that, 10 of 10, with fewer model calls. On a dependency that never recovers, the two layers multiplied: 3 harness attempts inside each of about 3 model attempts gave 8.7 calls to a dead service per ticket, against 2.9. [AWS's Builders' Library](https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter) warns about exactly this: retries at each of five layers raise load on the bottom one 243-fold. Retry in one layer, and tell the others the error was already retried.

### Sizing step and token budgets

A hard budget caps steps, tokens or dollars per task so a runaway run ends. Two measurements say how to size one.

Set step caps under realistic faults. With the harness absorbing faults, every successful refund ticket took 3 to 5 model calls. When the model handled faults itself, successful tickets took 4 to 10, and a cap of 5 would have cut 24 of those 40.

Budget tokens by what's new, not what's re-read. On the chain task, input tokens grew 80-fold from 5 to 80 steps while cost grew 23-fold (Step 1), so a cap on total input tokens would stop long runs well before their cost justified it. Cap dollars, steps, or new tokens (output plus tool results), which grew linearly, about 95 output tokens per step.

Claude's API has an advisory version. A [task budget](https://platform.claude.com/docs/en/build-with-claude/task-budgets) (beta header `task-budgets-2026-03-13`, on Haiku 5.5 and Sonnet 5.5) shows the model a running countdown of tokens for the whole agentic loop, counting thinking, tool calls, tool results and output but not resent history. The docs call it "a soft hint, not a hard cap" and set a 20,000-token minimum, more than any task here spent, so it wasn't measured. Keep `max_tokens` and your own cost cap as the hard limits.

![Flowchart of a durable agent loop. Start the task or load the last checkpoint, then check the budget for steps, tokens and cost; if none is left, stop and report what is done and not done. Otherwise run a model step. A final answer ends the run. Tool calls are first saved in a write-ahead checkpoint with their ids. If the same call has failed three times in a row, the harness answers it and tells the model to stop retrying; otherwise the tool runs with retries using backoff and jitter and an idempotency key equal to the call id. Either way, a checkpoint of messages and tool state is written and the loop returns to the budget check.](/diagrams/a12-durable-loop.svg "The loop with the four mechanisms from this part: budgets, a write-ahead checkpoint, a stopping rule and idempotent retries.")

The model sees none of this machinery, only tool results and, when the stopping rule fires, the harness's note.

The whole part cost $1.20 on the ledger, including pilots and two discarded versions of the chain task. The final chain runs cost $0.38 (140 runs, 3,829 steps) and the restate arm $0.08; the refund experiments cost $0.31 (Step 2 $0.16, Step 3 $0.06, Step 4 $0.08).

## In the real world

- [Sinha et al.](https://arxiv.org/abs/2509.09677) (ICLR 2026) used a task close to the chain here, a running sum of dictionary lookups, with the same horizon formula. They found per-step accuracy falling as tasks went on, partly because models "become more likely to make mistakes when the context contains their errors from prior turns"; thinking models held steady.
- METR's [time-horizon study](https://arxiv.org/abs/2503.14499) (Kwa et al., 2025) measures the task length, in human time, that an agent finishes 50% of the time: about 50 minutes for Claude 3.7 Sonnet, doubling roughly every seven months since 2019. Its 80% horizons were "4-6x shorter," where a constant per-step error predicts 3.1 times (`ln 0.8 / ln 0.5 = 0.32`). Toby Ord's [half-life paper](https://arxiv.org/abs/2505.05115) fits that data with a constant failure rate per minute.
- [MAKER](https://arxiv.org/abs/2511.09030) (Meyerson et al., 2025) raised `p` with single-move microagents for a 20-disk Towers of Hanoi and a vote on each move (first answer 3 votes ahead wins). It ran over one million steps with zero errors on gpt-4.1-mini, whose per-step error rate of 0.22% alone would give a 50% horizon of about 315 steps.
- [Stripe](https://docs.stripe.com/api/idempotent_requests) saves the status code and body of the first request with an idempotency key, returns them for retries, rejects a reused key with different parameters, and may prune keys after 24 hours. [Temporal](https://docs.temporal.io/activity-definition) warns that an activity "may be executed multiple times".
- Claude Code [checkpoints](https://code.claude.com/docs/en/checkpointing) Claude's file edits before each prompt and can rewind them, but files changed only by Bash commands are not tracked, and no rewind undoes a remote side effect. Rewinding leaves the outside world as it is.
- Anthropic's [harness for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) (November 2025) checkpoints coarsely: a progress file, a JSON feature list and git commits that each fresh context window reads first.

## Common misconceptions

- **"Per-step accuracy tells you task success."** Only if errors are spread evenly. Here 97.3% per step predicted 87% success at 5 steps and 11.5% at 80; the measured rates were 67% and 30%, because the first step carried most of the risk.
- **"Retries make a tool call reliable."** Retries without idempotency make writes less reliable. Harness retries raised the share of tickets with a double refund from 63% to 93%; the same retries with a key brought it to 0%.
- **"Resuming from a checkpoint is always safer than restarting."** A checkpoint that misses the last step's side effects tells the model those effects didn't happen. Resuming from one refunded 80 of 80 orders twice; restarting from scratch refunded none twice, because the fresh agent re-read the world.
- **"Agents retry forever."** Haiku 5.5 gave up on a failing refund after two or three attempts and reported it honestly. Its failure was the opposite: it abandoned a dependency that would have recovered on the fifth try.

## Key takeaways

- Success decays roughly like `p^n`, but `p` has to be the chance that a step goes right given that every earlier step did, and it need not be constant. Here the first step failed 19% of the time and later steps 1.6%.
- Put retries in the harness, with an idempotency key the backend checks. Without a key, retries turn timeouts into duplicate writes, and nothing the model does can stop a duplicate delivery.
- Checkpoint before a side effect, not only after. A write-ahead record of the calls and their ids lets a resume replay them instead of repeating them.
- Retry in one layer, size step caps under realistic fault rates, and budget tokens by what is new, not by what is re-read from the cache.

## What's next

Every fault here was simulated inside one Python process, so a wrong refund cost nothing. [Part 13](/blog/agent-sandboxing/) is about what happens when an agent's tools can touch real systems: least privilege, containers, network and secret isolation, and human approval gates, with the cost of each measured.

## Further reading

- [The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs (Sinha et al., 2025)](https://arxiv.org/abs/2509.09677): the horizon formula, per-step accuracy that degrades with length, and self-conditioning.
- [Measuring AI Ability to Complete Long Tasks (Kwa et al., 2025)](https://arxiv.org/abs/2503.14499): METR's 50% time horizon and its seven-month doubling.
- [Is there a half-life for the success rates of AI agents? (Ord, 2025)](https://arxiv.org/abs/2505.05115): the constant-hazard model of agent success over task length.
- [Solving a Million-Step LLM Task with Zero Errors (Meyerson et al., 2025)](https://arxiv.org/abs/2511.09030): extreme decomposition and per-step voting to push `p` toward 1.
- [τ-bench (Yao et al., 2024)](https://arxiv.org/abs/2406.12045): pass^k, the probability that all k runs of a task succeed, as a reliability metric.
- [Task budgets (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/task-budgets): an advisory token budget for a whole agentic loop.
- [Errors (Claude API docs)](https://platform.claude.com/docs/en/api/errors): which API errors are transient, and the SDK's automatic retries with exponential backoff.
- [Idempotent requests (Stripe API reference)](https://docs.stripe.com/api/idempotent_requests): the idempotency-key contract used in Step 2.
- [Timeouts, retries and backoff with jitter (AWS Builders' Library)](https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter): why retries amplify load, and why jitter and idempotency matter.
- [Activity definition (Temporal docs)](https://docs.temporal.io/activity-definition): at-least-once activities and idempotency keys that survive retries.
- [Effective harnesses for long-running agents (Anthropic, 2025)](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents): progress files and git commits as checkpoints across context windows.
- [Detecting and Recovering from Agent Loops](/blog/agent-loop-detection-recovery/): loop detection, budgets and the escalation ladder this part builds on.
