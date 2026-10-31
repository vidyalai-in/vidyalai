---
title: "Detecting and Recovering from Agent Loops"
description: "Classify why an agent loops, add harness guardrails and better tool feedback, and recover with an escalation ladder you can measure."
slug: "agent-loop-detection-recovery"
category: "agents-tool-use"
tags: ["agents", "tool-use", "observability", "evals"]
seriesOrder: 11
pubDate: 2026-01-20
sourceDoc: "1WwZ8yG-cd8f7_IZixvOLmzYPvB9hYJ554S4N4h26QE0"
---

## TL;DR

A loop is a symptom, so find out which kind it is before patching: the same action repeated with no progress, two states bouncing back and forth, or the agent stalling in "thinking". Then fix it in three layers. Hard guardrails in the harness: step, token and wall-clock budgets, plus detection of repeated (tool, args) pairs. Better signal to the model: tool errors it can act on, progress state, and a note when it is repeating itself. A recovery policy: reflect and replan once, then escalate to a stronger model or a human, or end with a partial result. Finally, turn the failing traces into regression evals and track loop rate per 1,000 tasks, because a step cap with nothing else just turns an infinite loop into a slow, expensive failure.

## Requirements to pin down first

- **Which kind of loop is it?** The fix depends on the shape:
  - (a) *Exact repeat*: same tool, same arguments, same result.
  - (b) *Oscillation*: edit A, revert A, edit A.
  - (c) *Semantic churn*: different queries that get no closer to the goal.
  - (d) *Planner loop*: the agent keeps re-planning and never acts.
  - (e) *Multi-agent ping-pong*: agents handing work back and forth.

  Exact repeats are cheap to catch with hashing; semantic churn needs a progress signal; ping-pong needs ownership rules.
- **Is it new?** A loop that started after a model, prompt or tool change points to a regression, so diff the change. A long-tail loop on rare inputs points to missing capability or missing information, so the fix is in tools or data.
- **What does the tool return when it fails?** Tools that return empty strings, a bare `Error`, or truncated output are a frequent cause. The model can't tell the call failed, so it tries again.
- **Is the task actually possible?** Many loops are the agent "trying harder" at something it has no permission or information to do. If so, the fix is an exit path, not a smarter retry.

## Step 1: Instrument and classify before changing anything

Log every step as structured data:

```text
(step, tool, normalized_args_hash, result_hash, tokens, latency, state_digest)
```

Emit these as OpenTelemetry-style spans so a whole trajectory can be replayed step by step.

Then pull 50-100 looping traces and label them by hand using the taxonomy above. One or two causes usually dominate the split. An illustrative breakdown: 60% uninformative tool errors and 25% context overflow, where the agent forgets it already tried X after summarization or truncation dropped the record. Your numbers will differ, and labeling is how you find them.

Track four baseline metrics from day one:

- **Loop rate**: share of trajectories that trip the loop detector.
- **Steps-to-success**: p50 and p95 step count for tasks that succeed.
- **Cost per resolved task**: total spend divided by successful tasks, so caps that waste work show up.
- **Wasted-step fraction**: steps that repeat an earlier (tool, args) pair or make no measurable progress.

## Step 2: Hard guardrails in the harness

These are deterministic and live in the harness, not in the model.

### Budgets

Cap max steps, max tokens per task, max wall-clock time and max spend. Typical starting ranges are 30-50 steps for a coding agent and 8-12 for a support agent. Set the real values from data: put each cap around the p99 of *successful* trajectories, not at an arbitrary 100. Anthropic's agent guidance likewise recommends stopping conditions such as a maximum number of iterations to keep control of autonomous loops.

### Repeat detection

Hash the normalized (tool, args) pair. Normalization means sorting keys, trimming whitespace and lowercasing where case doesn't matter, so reordered arguments still collide. A simple rule: the same call 3 times within a window of 10 steps is a loop.

```python
from collections import deque, Counter
import hashlib, json

def call_key(tool: str, args: dict) -> str:
    norm = json.dumps(args, sort_keys=True, separators=(",", ":")).lower()
    return hashlib.sha256(f"{tool}:{norm}".encode()).hexdigest()

window = deque(maxlen=10)

def is_repeat(tool: str, args: dict, threshold: int = 3) -> bool:
    window.append(call_key(tool, args))
    return Counter(window)[window[-1]] >= threshold
```

For oscillation, track a state digest (files changed, DOM hash, database row versions) and flag A→B→A cycles in the digest sequence.

### No-progress detection

Define a cheap progress signal per domain: tests passing, new information retrieved, or subgoals checked off. "New information" can be measured as embedding novelty: the new observation's similarity to all earlier observations stays below a threshold. If there is no progress in K steps (K≈5 is a reasonable start), intervene.

### Idempotency and caching

Cache read-only tool results within a run. This removes the wasted cost of repeats but leaves the loop itself in place, so treat it as cost control.

## Step 3: Better signal to the model

### Tool results the model can act on

Compare a bare `Error` with this:

```text
File not found: src/utils.py. Did you mean src/util/helpers.py?
Directory listing: ...
```

The second version tells the model what went wrong and what to try next. This is often the highest-leverage fix in a harness. The ReAct paper's error analysis found that non-informative search results made up 23% of its error cases and left the model struggling to recover, and Anthropic's tool-writing guidance recommends prompt-engineering error responses so they communicate specific, actionable improvements. Measure the effect on your own loop rate rather than assuming a fixed reduction.

### Self-awareness injection

When the detector fires, add a system or user message such as:

```text
You have called search(q='refund policy') 3 times with identical results.
Summarize what you have learned, list hypotheses you have NOT tried,
and choose a different action or declare the task blocked.
```

This is a light Reflexion-style intervention: verbal self-reflection that changes the context so the next action can differ.

### Explicit scratchpad and plan state

Keep a persistent "tried / ruled out" list that survives context compaction (summarizing older context when the window fills). Semantic loops in long runs commonly appear right after compaction drops the record of earlier attempts. Anthropic's context-engineering guidance calls this structured note-taking: notes persisted outside the context window and pulled back in when needed.

### A way to exit

Add an `ask_user` or `give_up(reason)` tool and reward its correct use in evals. Agents loop when stopping isn't an available move.

### Decoding: less entropy for arguments, more diversity on retry

At temperature 0, the same context gives the same action every time. On a detected loop, either change the context (the reflection message does this) or raise the temperature, or sample N alternative actions and pick one with a verifier.

## Step 4: Recovery policy as an escalation ladder

![State diagram: the agent acts and loops on itself while it makes progress, finishing when the goal is reached. If the loop detector fires on a repeat, oscillation or lack of progress, the first trigger goes to reflect-and-replan and back to acting, the second trigger forks from the last good checkpoint with a stronger model or new strategy and resumes acting, and the third trigger goes to a graceful exit. Hitting a hard budget also goes straight to graceful exit, which returns a partial result, a failure reason and a human handoff.](/diagrams/q11-loop-recovery.svg "Loop recovery as an escalation ladder, with hard budgets as a backstop.")

Each detector trigger escalates one rung, and the hard budget is a backstop that always ends in a graceful exit.

1. First trigger: reflect and replan. This is cheap and stays in context.
2. Second trigger: fork from the last good checkpoint with a stronger model or a different strategy, for example switching from "edit" to "read more code".
3. Third trigger: end gracefully with a partial result, a structured failure reason and a handoff to a human. Never fail silently.

Search with backtracking (as in Tree of Thoughts) is the general form of rung 2: instead of pushing forward on one greedy path, return to a known-good state and try a different branch.

## Step 5: Fix root causes and prevent regressions

- Fix the tools whose errors were uninformative. Give similar tools distinct names and descriptions so the model doesn't confuse them; overlapping tools are a common source of ping-pong. Anthropic's tool guidance makes the same point: each tool needs a clear, distinct purpose, and namespacing helps the model choose.
- In multi-agent setups, give each subtask one owner, cap handoff depth and make the handoff contract explicit. Step repetition and failure to recognize termination both appear in the MAST failure taxonomy for multi-agent systems.
- Add every labeled looping trace to a regression eval set and gate deploys on loop rate not regressing.

## How to measure it

| Metric | Definition | Example target |
|---|---|---|
| Loop rate | Share of tasks that trip the detector | Down from e.g. 8% to under 1% |
| Hard-cap rate | Share of tasks that hit a budget | ≤ 0.2% |
| Task success rate | Share of tasks completed correctly | Flat or up after the fix |
| Steps-to-success | p50 / p95 steps on successful tasks | p95 down |
| Cost per resolved task | Spend ÷ successful tasks | Down |
| Correct-exit rate | Share of escalations or give-ups with a correct diagnosis | Up, scored by an LLM judge checked against human labels |

The targets above are illustrative; set yours from the baseline you measured in Step 1.

Two checks keep the metrics honest:

- Success rate must not drop. If loop rate falls but success falls with it, the caps are cutting off trajectories that would have succeeded.
- Validate on hard tasks. Run the regression set and a slice of long, legitimately multi-step tasks on every change, so a tighter cap doesn't hurt the hardest work unnoticed.

For the judge that scores exits, see [Building an Eval Suite for a Customer Support Bot](/blog/support-bot-eval-suite/) on calibrating LLM judges against human labels.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Hard step/token caps | Simple, deterministic, bounds cost | Cuts off legitimate long tasks; doesn't fix the cause | Always, as a backstop |
| Repeat/oscillation detector + reflection prompt | Cheap, catches most exact loops, model self-corrects | Misses semantic churn; can over-fire on legitimate polling | Default first intervention |
| Progress-based detection (verifiers, novelty) | Catches semantic loops | Needs a domain-specific progress signal; threshold tuning | Long-horizon agents (coding, research) |
| Escalate to stronger model or human | High recovery rate | Latency, cost, human load | High-value tasks, regulated domains |
| Retrain/fine-tune on loop-recovery trajectories | Fixes behavior at the source | Slow; needs data and RL infrastructure | You own the model and loops are systemic |

- Set budgets from the distribution of successful trajectories, and revisit them after every model upgrade, because newer models often need fewer steps.
- Some repetition is legitimate. Polling a CI job or waiting on async work should be allow-listed or given backoff semantics instead of being flagged.
- Fix the harness before the model. Most loops come from the environment and its tools, not from the model being weak.

## Common mistakes

- **"Just set `max_iterations=10`."** That bounds cost but hides the root cause and lowers the success rate in a way that is easy to miss. Without measurement and diagnosis, you never learn why tasks fail.
- **Ignoring context compaction.** After summarization the agent forgets what it already tried and repeats it. A persistent tried-list or memory fixes this.
- **Exact-match detection only.** Paraphrased queries and reordered arguments slip past. Normalize arguments and add semantic similarity.
- **Retrying without changing anything.** Temperature 0 plus an unchanged context gives deterministic repetition, so a blind retry produces the same loop.
- **No terminal action.** If the agent can't say "I'm blocked", it keeps acting until a cap stops it.
- **Not re-testing long valid trajectories.** A fix that cuts loops but also cuts legitimate long runs is a regression on your hardest tasks.

## Thinking at scale

- Make loop rate a fleet-level SLO. Track it per agent, per tool and per model version, so a regression after a tool change is attributed to the right owner.
- Own the tool contracts. Error-message quality is part of each tool's API. Review it like any other interface, and test it with the agent in the loop.
- Re-baseline budgets on every model migration. Caps tuned for an older model either waste money or cut off work on a newer one.
- Decide when to stop patching. If loops remain systemic after harness and tool fixes, and you own the model, training on loop-recovery trajectories fixes the behavior at the source.

## Further reading

- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629): The thought/action/observation loop most agents build on; its error analysis reports repetitive thought-action loops and the damage done by non-informative search results.
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366): Verbal self-reflection after a failed attempt, the basis of "reflect and replan" on loop detection.
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (Yang et al., 2024)](https://arxiv.org/abs/2405.15793): Shows that interface design (concise feedback, guardrails on edits) strongly affects how often agents get stuck.
- [Building effective agents (Anthropic, 2024)](https://www.anthropic.com/engineering/building-effective-agents): Practical guidance on stopping conditions, tool design and keeping agent loops simple.
- [Writing effective tools for agents (Anthropic, 2025)](https://www.anthropic.com/engineering/writing-tools-for-agents): Actionable error responses and distinct, namespaced tools.
- [Effective context engineering for AI agents (Anthropic, 2025)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): Compaction and structured note-taking, which keep a tried-list alive across long runs.
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)](https://arxiv.org/abs/2305.10601): Search with backtracking as an alternative to greedy single-path loops.
- [Why Do Multi-Agent LLM Systems Fail? (Cemri et al., 2025)](https://arxiv.org/abs/2503.13657): A failure taxonomy that includes step repetition and failure to recognize termination.
