---
title: "Building Tool-Using Agents That Work Reliably"
description: "Tool-use reliability is a systems property: tool design, a guarded control loop, permission gates, and pass^k evals on verifiable end states."
slug: "reliable-tool-using-agents"
category: "agents-tool-use"
tags: ["agents", "tool-use", "structured-output", "evals", "prompt-injection"]
seriesOrder: 10
pubDate: 2026-01-19
sourceDoc: "1wx9vLMMIykYpYxwlDoHQz8GunZXg7raHlJEYPtY-TAI"
---

## TL;DR

Tool-use reliability is mostly a property of the system around the model, not of the prompt. Make the tools hard to misuse: a few well-named tools with strict schemas, informative errors and idempotent side effects. Harden the control loop: schema-constrained calls, validation, bounded retries with error feedback, step and budget limits, and human confirmation for irreversible actions. Measure end-to-end task success with multi-trial pass^k on a realistic eval suite that uses simulated users and real tool backends. The single biggest lever is usually the tool interface itself: consolidating overlapping tools, returning concise high-signal results, and writing descriptions the way you would for a new engineer joining the team.

## Requirements to pin down first

- **What does "reliably" mean here?** There are three different bars:
  - Per-call schema validity is easy. With strict tool use it is guaranteed by construction.
  - Task success is hard, often 50-80% on realistic multi-step tasks.
  - *Consistency* across repeated attempts is harder still. In [τ-bench](https://arxiv.org/abs/2406.12045) (2024), gpt-4o succeeded on fewer than 50% of tasks, and its pass^8 in the retail domain was below 25%.
- **How many tools and steps, and are they read-only or side-effecting?** Ten tools and 3 steps is a different problem from 200 tools and 50 steps.
- **What does a wrong action cost?** Searching a document, issuing a refund and running `rm -rf` need very different safeguards.
- **What is the latency and cost budget per task?** It bounds the number of steps and retries the loop can afford.

## The agent loop and its guards

![A user task enters a model step that either finishes or proposes a tool call. A deterministic guard layer checks budgets and loops, validates the call against schema and business rules, and routes irreversible actions to human confirmation. Valid calls run in the tool layer, whose result or actionable error returns to the model. Invalid calls return an error to the model, with retries bounded before escalating or aborting. When the model reports done, an end-state check verifies success before returning.](/diagrams/q10-agent-loop.svg "A tool-using agent: the model decides, but deterministic guards decide what actually executes.")

The model chooses each step, but every call passes through code that can block, retry, escalate or stop it.

## Layer 1: Tool design (the highest-leverage layer)

- Fewer, higher-level tools. Offer `schedule_meeting(attendees, window)` rather than `list_users`, `get_calendar`, `find_slot` and `create_event`. Every extra tool adds selection errors and context tokens. Once you have more than about 20-30 tools, load them dynamically or let the model search over tool descriptions; Anthropic's [tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) is built for working with thousands of tools this way.
- Clear names, namespacing and descriptions. Include examples, units, formats, and when *not* to use the tool. Namespace by service (`asana_search`, `jira_search`). Use unambiguous parameter names (`user_id`, not `user`) and enums instead of free strings.
- Strict schemas. Use JSON Schema with required fields, enums and `additionalProperties: false`. Turn on the provider's strict mode: with `strict: true` on a Claude tool definition, [grammar-constrained sampling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use) guarantees the tool input matches the schema and the tool name is valid. [Guaranteeing Valid JSON Output from LLMs](/blog/guaranteed-valid-json-output/) covers constrained decoding and schema design.
- Agent-friendly responses. Return concise, semantically meaningful fields (names rather than UUIDs where possible). Paginate and truncate with hints, such as "showing 20 of 340; refine with query". Aim for responses of a few thousand tokens at most. As a hard ceiling for comparison, Claude Code caps tool responses at 25,000 tokens by default ([Anthropic, Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)).
- Actionable errors. Return `"date must be ISO-8601 like 2026-10-15; got '10/15'"`, not a stack trace. Errors are the model's main signal for self-correction.
- Idempotency keys and dry-run modes for side-effecting tools, so a retry can never double-charge a customer.

## Layer 2: The control loop

Use a ReAct-style loop ([ReAct](https://arxiv.org/abs/2210.03629): reason, call a tool, observe the result) with the model deciding the next step, but wrap it in deterministic guards:

- Validate before executing. Check every call against the schema *and* business rules: permissions, value ranges, allowed resources. Strict mode covers the schema; only your code can enforce the business rules. On failure, return the error to the model without executing.
- Bound retries. Allow 2-3 retries per call with the error in context, then escalate or abort.
- Enforce budgets. Cap steps, tokens, cost and wall-clock time (for example, 25 steps and 120 seconds). Detect loops: the same call with the same arguments repeated more than twice. [Detecting and Recovering from Agent Loops](/blog/agent-loop-detection-recovery/) goes deeper on detection and recovery policies.
- Parallelize independent reads to cut latency.
- Manage context on long tasks. Compact old tool outputs, keep a scratchpad or to-do list, and hand large searches to sub-agents so the main context stays clean.
- Prefer workflows where you can. If the path is known, hard-code the sequence (prompt chaining or routing) and use the LLM only for the fuzzy steps. Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) draws the same line between workflows (predefined code paths) and agents (the model directs its own process). Reserve fully autonomous loops for open-ended tasks.

A minimal sketch of the guard layer:

```python
for step in range(MAX_STEPS):
    action = model.next_action(context)
    if action.is_final:
        if verify_end_state(task):
            return action.answer
        context.add_error("End state not reached; keep going or explain why.")
        continue
    if loop_detector.repeated(action, limit=2) or budget.exceeded():
        return escalate(task, reason="loop or budget")
    error = validate(action, schema=True, business_rules=True)
    if error is None and action.tool.irreversible:
        error = require_human_confirmation(action)
    if error:
        retries[action.tool] += 1
        if retries[action.tool] > MAX_RETRIES:
            return escalate(task, reason=error)
        context.add_error(error)
        continue
    result = action.tool.run(action.args, idempotency_key=action.id)
    context.add_result(compact(result))
return escalate(task, reason="step limit")
```

## Layer 3: Safety and permissions

- Classify tools as read-only, reversible write, or irreversible. Require human confirmation or policy approval for irreversible actions.
- Use least-privilege credentials scoped per user and per session. Never give the model raw admin tokens.
- Defend against prompt injection. Treat tool outputs (web pages, emails, documents) as untrusted data. Do not let content from them trigger privileged tools without confirmation. Separate the planner's privileges from the reader's, as in dual-LLM or quarantined-reader patterns. Log and alert on anomalous tool sequences. [Defending Tool-Using Agents Against Prompt Injection by Design](/blog/prompt-injection-defense-tool-agents/) treats this layer in full.
- Sandbox code execution in containers or gVisor, with no network by default and CPU, memory and time limits.

## How to measure it

### Tool-level unit evals

For each tool, write prompts where it should be called and prompts where it should not, and check argument correctness. The [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (BFCL) uses AST-based and executable checks of this kind; its V4 adds multi-turn and agentic categories.

### Task-level evals

Build 100-500 realistic multi-step tasks with *verifiable end states*: the database matches the expected state, or the test suite passes. Judged transcripts alone are not enough. Include simulated users, as τ-bench does, to exercise multi-turn clarification.

### Report pass^k, not only pass@1

pass^k means the agent succeeds on *all* k independent trials of the same task. It exposes inconsistency that pass@1 hides. If trials were independent, an agent with 80% per-trial success would have `0.8^8 ≈ 17%` pass^8. Report pass^k alongside cost, steps and latency per task.

### Failure taxonomy

Review failed transcripts and tag each with a category: wrong tool, wrong arguments, hallucinated tool or argument, premature stop, looping, ignored error, or policy violation. Fix the top category, re-run, and repeat. An agent can do the first pass of this analysis: Anthropic's tool-writing guidance describes feeding eval transcripts to Claude to propose tool and description fixes.

### Production monitoring

Trace every step with OpenTelemetry-style spans. Track task completion, user intervention rate and error rate per tool, and auto-sample transcripts for review.

### Concrete targets

| Metric | Target |
|---|---|
| Schema-valid calls | 100% with strict mode (guaranteed by constrained decoding); at least 99.5% without it |
| Tool-selection accuracy on the unit suite | at least 95% |
| Task success | from about 60% to 85% or more through tool redesign and error-message work |
| Unconfirmed irreversible actions | zero |
| Median steps per task | trending down |

### Why step count matters

Reliability compounds. At 95% per-step success, a 20-step task succeeds `0.95^20 ≈ 36%` of the time. Cutting the task to 8 steps with better tools raises that to `0.95^8 ≈ 66%`, without any improvement in per-step accuracy.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Fixed workflow (chained calls) | Predictable, testable, cheap | Brittle to unexpected inputs; no adaptivity | Known, repeatable processes |
| Autonomous ReAct-style agent | Handles open-ended tasks; adapts to errors | Variance, cost, harder to evaluate | Open-ended, exploratory tasks |
| Many fine-grained tools | Flexible; easy to build from existing APIs | Selection errors, context bloat, long chains | Rarely; prototypes only |
| Few consolidated tools | Higher success, fewer steps | More engineering per tool | Production agents |
| Human-in-the-loop on writes | Safety and user trust | Friction and latency | Irreversible or high-value actions |
| Fine-tune for tool use | Better selection and arguments on your API | Data and maintenance cost; lags base-model upgrades | Very high volume, stable tool set |

- Autonomy vs. control. Add autonomy only where evals show it raises task success. Each extra degree of freedom adds variance.
- Reliability compounds. Reducing the step count with better tools often beats improving per-step accuracy.

## Common mistakes

- **Equating valid JSON with reliability.** Strict mode solves parsing, but tasks fail on choosing the right tool, passing the right arguments and knowing when to stop.
- **Single-run evals.** pass@1 hides inconsistency: an agent can pass a task once and fail it on 40% of reruns. Report pass^k.
- **Verbose tool outputs flooding context.** Dumping 50k tokens of raw API JSON degrades reasoning and raises cost. Summarize, paginate and filter.
- **Non-idempotent retries.** Retrying a timed-out `create_payment` without an idempotency key charges the customer twice.
- **Prompt injection through tool results.** A retrieved web page or email can instruct the agent to exfiltrate data through another tool. The model alone is not a sufficient defense; permissions and confirmation gates are.
- **No stop condition.** Without budgets and an explicit success check, agents loop forever, repeat identical calls, or "finish" without verifying the end state.

## Further reading

- [ReAct (Yao et al., 2022)](https://arxiv.org/abs/2210.03629): interleaving reasoning and acting; the base pattern for tool-using agents.
- [Toolformer (Schick et al., 2023)](https://arxiv.org/abs/2302.04761): self-supervised learning of when and how to call tools.
- [Gorilla (Patil et al., 2023)](https://arxiv.org/abs/2305.15334): LLMs calling large API sets, with retrieval over API docs to reduce hallucinated calls.
- [τ-bench (Yao et al., 2024)](https://arxiv.org/abs/2406.12045): tool-agent-user benchmark with simulated users and the pass^k consistency metric.
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html): standardized function-calling evaluation, now including multi-turn and agentic categories (V4).
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): workflows vs. agents, and keeping agent design simple.
- [Anthropic: Writing effective tools for agents (2025)](https://www.anthropic.com/engineering/writing-tools-for-agents): tool consolidation, namespacing, response design, error messages and eval-driven tool iteration.
- [Claude tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): tool definitions, `tool_choice`, and server tools.
- [Claude strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use): `strict: true` and grammar-constrained schema conformance.
