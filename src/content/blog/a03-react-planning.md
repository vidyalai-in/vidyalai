---
title: "Reasoning and Acting: ReAct, Planning and Thinking, Measured"
description: "ReAct, plan-and-execute and adaptive thinking on one 36-task tool set: accuracy, tokens and cost on Claude Haiku 5.5, and where more reasoning stops paying."
slug: "react-planning"
category: "agents-tool-use"
tags: ["agents", "tool-use", "reasoning", "cost", "latency"]
series: "agentic-ai"
seriesOrder: 3
pubDate: 2026-10-24
---

## TL;DR

An agent can reason about its next move in three ways: a visible thought before every tool call (ReAct), a whole plan before the first call (plan-and-execute), or hidden thinking whenever the model judges a step hard (adaptive thinking, steered by an effort level). This article explains each, then runs all of them on one fixed set of 36 tool-using tasks, 108 runs per setting on Claude Haiku 5.5. Switching on thinking at the lowest effort level raised accuracy from 77% to 88% for 44% more cost. Visible ReAct thoughts and an up-front plan bought nothing measurable. Each step above that cost more per extra correct answer, and on the hardest puzzles `high` effort spent 2.5 times as much as `medium` and solved no more of them.

## Why it matters

Give Claude Haiku 5.5 a road map tool and ask for the shortest route between two towns. With thinking switched off, it explored the roads, listed three candidate routes, and answered 71 km. The right answer was 64 km, through a junction it had already read about and never added up. With thinking on, the same model on the same task got 64 km in all 9 runs at `medium`, `high` and `xhigh` effort.

Reasoning is the part of an agent you pay for twice: once in tokens and once in waiting. On the 36 tasks below, moving from no thinking to `xhigh` effort took accuracy from 77% to 96%, cost per run from $0.0013 to $0.0055, and mean model time from 13.5 to 40.9 seconds. One 4-digit lock puzzle, which `medium` effort solved in all 3 runs for $0.006 each, took one `xhigh` run 83,055 thinking tokens and 5 minutes 24 seconds.

The useful questions are which kind of reasoning to buy, how much, and where the curve flattens. [Part 1](/blog/what-is-an-ai-agent/) built the agent loop and [Part 2](/blog/tool-calling-internals/) opened up the tool call. [Reasoning Models](/blog/reasoning-models/) (Fundamentals Part 14) covers how models learned to think; this part measures what that thinking buys inside a loop.

## The intuition

Picture three ways to drive to an unfamiliar address. You can print turn-by-turn directions before leaving and follow them. You can talk yourself through every junction out loud: "Mill Road, bridge closed, so the next left." Or you can drive normally and only pull over to think when a junction looks confusing.

Printed directions (plan-and-execute) are cheap and fast while the roads match the map, and useless the moment one is closed. Narrating every junction (ReAct) catches the closed road, but you narrate the easy junctions too, and saying a thought aloud doesn't make it a careful one. Pulling over only when it's confusing (adaptive thinking) spends effort where it pays.

The analogy breaks in two places. A driver's thinking is free, while a model's thinking is billed by the token. And a model can't tell in advance how hard a junction is: the effort level only tells it how readily to pull over.

## Step 1: ReAct, thoughts between actions

### The idea

In October 2022, Shunyu Yao and colleagues at Princeton and Google published [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (ICLR 2023). Chain-of-thought prompting made a model reason in writing but couldn't check a fact, so it could reason fluently from a made-up premise. Action generation let a model call tools but gave it no room to think between calls.

ReAct merged them. At each step the model emits either an action, which comes back as an observation, or a thought: text that "does not affect the external environment" and only adds context for later steps. From the paper's first figure:

```text
Thought 1: I need to search Apple Remote and find the program it was originally designed to interact with.
Act 1: Search[Apple Remote]
Obs 1: The Apple Remote is a remote control introduced in October 2005 by Apple ... originally
       designed to control the Front Row media center program ...
Thought 2: Apple Remote was originally designed to control the Front Row media center program.
           I need to search Front Row next and find what other device can control it.
```

The authors prompted PaLM-540B, Google's 540-billion-parameter model, with six hand-written trajectories like this for HotpotQA (multi-hop questions over Wikipedia) and three for FEVER (checking claims against Wikipedia). The tools were deliberately weak: `search[entity]`, `lookup[string]` and `finish[answer]`.

### What it found

| Prompting method (PaLM-540B) | HotpotQA exact match | FEVER accuracy |
|---|---|---|
| Standard (answer directly) | 28.7 | 57.1 |
| Chain of thought (reason only) | 29.4 | 56.3 |
| Act only (tools, no thoughts) | 25.7 | 58.9 |
| ReAct (thoughts and tools) | 27.4 | 60.9 |
| Best ReAct and chain-of-thought combination | 35.1 | 64.6 |

Thoughts beat no thoughts on both tasks. The bigger wins came on interactive tasks: on ALFWorld, a text-based household game, the best ReAct prompt succeeded 71% of the time against 45% for act-only, and on WebShop, a simulated shopping site, 40.0% against 30.1%.

The error analysis explains the mixed scores. The authors hand-labelled 50 correct and 50 wrong HotpotQA trajectories from each method:

| Failure type, share of wrong answers | ReAct | Chain of thought |
|---|---|---|
| Hallucinated facts or reasoning | 0% | 56% |
| Reasoning error, including repeating earlier steps | 47% | 16% |
| Search returned nothing useful | 23% | none (no search) |
| Answer right but didn't match the label | 29% | 28% |

Grounding in observations removed hallucination. In exchange, the rigid thought-action-observation format tripled the share of reasoning errors, including a loop "in which the model repetitively generates the previous thoughts and actions." The modern version of that trade shows up in the measurements below.

## Step 2: plan-and-execute, thinking once up front

ReAct decides one step at a time. The opposite design writes the whole plan first. [Plan-and-Solve prompting](https://arxiv.org/abs/2305.04091) (Wang et al., ACL 2023) showed the idea without tools, replacing "Let's think step by step" with "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step." On GPT-3 (text-davinci-003), accuracy on GSM8K, a set of grade-school math word problems, rose from 56.4% to 58.2%, and to 59.3% with extra instructions to extract variables and calculate carefully.

[ReWOO](https://arxiv.org/abs/2305.18323) (Xu et al., 2023) brought planning to tools. A Planner writes every step up front, with placeholders such as `#E1` for evidence it hasn't seen yet; Workers run the calls and fill them in; a Solver answers. Nothing re-reads a growing transcript after each observation, which is where ReAct spends its tokens. On HotpotQA with gpt-3.5-turbo, ReAct used 9,795 tokens per question for 40.8% accuracy and ReWOO used 1,986 for 42.4%. [LLMCompiler](https://arxiv.org/abs/2312.04511) (Kim et al., ICML 2024) has the planner emit a graph of calls and runs independent ones in parallel, reporting up to 3.7 times lower latency and 6.7 times lower cost than ReAct.

A plan written before any observation can't depend on what the observations say. When the world surprises it, something has to notice and replan.

## Step 3: thinking inside the model

### What the API exposes

ReAct and plan-and-execute were prompting tricks for models that reasoned only when told to. With adaptive thinking, the default on Haiku 5.5 and Sonnet 5.5, the model decides on every request whether to write a thinking block, and how long. From the [steering docs](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost): "A simple factual question might get a direct response with no thinking block at all; a multistep math problem or a tricky debugging task triggers deeper reasoning."

Three settings control it:

- `output_config.effort` is the main dial: `low`, `medium`, `high`, `xhigh` or `max`. Haiku 5.5 defaults to `medium` and Sonnet 5.5 to `high`. Per the [effort docs](https://platform.claude.com/docs/en/build-with-claude/effort), effort is "a behavioral signal, not a strict token budget," and it shapes all output, including how many tool calls the model makes.
- `thinking.type` turns thinking on or off. Haiku 5.5 accepts `{"type": "disabled"}` at `high` effort or below and returns an error at `xhigh` and `max`. Sonnet 5.5 rejects `disabled`; its lowest setting, `between_tools`, only turns off the thinking before the first action.
- `thinking.display` decides what you see. It defaults to `"omitted"`, an empty block, on these models; `"summarized"` returns a summary. The bill is the same either way.

### Interleaved thinking

With adaptive thinking, the model can also think between tool calls: read a result, think about it, then act. The [thinking docs](https://platform.claude.com/docs/en/build-with-claude/thinking) call this interleaved thinking, and on these models it needs no setup. It is ReAct's thought step moved inside the model, written in hidden tokens rather than visible text. Not every step gets it: per the steering docs, follow-up requests "that only process tool results can skip thinking, including at `xhigh` and `max`."

### How thinking is billed

Thinking tokens are output tokens: $0.50 per million on Haiku 5.5 and $10 on Sonnet 5.5, against $0.10 and $2 for uncached input ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)). The visible summary is shorter than the bill, so read `usage.output_tokens_details.thinking_tokens`. Thinking blocks go back unchanged with tool results, and Haiku 5.5 keeps all of them in context, so a block written at step 2 is read again, as input, at every later step. Changing effort mid-conversation invalidates the prompt cache.

![Three stacked loops. ReAct: a visible thought leads to an action (a tool call), which returns an observation (the tool result) that goes back to the model for the next thought. Plan-and-execute: a planner writes the whole plan, an executor runs the next step and loops while results fit the plan; when the plan breaks, the planner replans once and the executor continues. Adaptive thinking: optional hidden thinking, which is billed, leads to a tool call and a tool result that goes back to the model.](/diagrams/a03-reasoning-patterns.svg "Where the reasoning happens in each design: visible text before every action, one plan up front, or hidden thinking the model chooses to do.")

Only adaptive thinking can skip the reasoning on a step: when the model judges a step easy, it writes no thinking block and pays nothing for it.

## Step 4: one task set, eight ways to reason

### The tasks

The test bed is the Saltmere Islands, a fictional archipelago generated by a script that also computes every answer. Its 82 records cover 4 islands, 16 towns, 30 people, 10 companies, 8 ferries and 14 road junctions, with 51 roads between them. The agent gets six tools: `search` and `lookup` (modelled on ReAct's Wikipedia actions), `calculate`, `roads`, and two puzzle tools with a use limit, `weigh` and `guess`. There are 36 tasks, 9 of each kind:

- Chain: a superlative or comparison followed by a 3-4 hop lookup and arithmetic, such as "How many years older is the oldest mayor on Vey than the founder of the company that operates MV Skua?"
- Aggregate: enumerate a whole set, join and compute, such as ranking all 10 companies by revenue per employee.
- Route: the shortest road distance between two towns. In six, the route with the fewest roads is a trap, 8 to 16 km longer than the true shortest; three more must avoid a named town.
- Puzzle: four 12-coin problems (find the odd coin, heavier or lighter, in 3 weighings) and five 4-digit locks (find the code in 8 guesses from "bulls and cows" feedback).

Grading reads the last `ANSWER:` line and compares exactly. A hedge like "2 heavier (3 lighter also fits)" counts as wrong.

### The arms

All runs use `claude-haiku-5-5`, the same system prompt and tools, prompt caching, and effort `medium` unless stated. The code lives in `react.py`, which imports the [Part 1](/blog/what-is-an-ai-agent/) harness and exposes its thinking setting:

```python
ADAPTIVE = {"type": "adaptive", "display": "summarized"}   # model decides when and how much to think
DISABLED = {"type": "disabled"}                            # no thinking blocks at all (Haiku 5.5, effort <= high)

REACT_RULES = """
Work in the ReAct style. Before every tool call, write a short paragraph that starts with "Thought:"
and says what you know so far, what is still missing, and why the next action is the right one.
After each tool result, start your next message with a Thought about what the result means."""
```

- Plain loop: thinking disabled, no instructions about reasoning.
- ReAct: thinking disabled, with `REACT_RULES` appended to the system prompt.
- Plan-and-execute: a planner call (thinking disabled, tools visible but `tool_choice: none`) writes a numbered plan with `#E` placeholders and a rule for each branch. An executor runs it with thinking disabled and may reply `REPLAN: <reason>`; the planner then gets one more try with every observation so far.
- Adaptive thinking at `low`, `medium`, `high` and `xhigh`.

Each arm is one call per run:

```python
from react import DISABLED, REACT_RULES, plan_and_execute, run_loop
from saltmere import SYSTEM, TASKS, grade, make_tools

task = next(t for t in TASKS if t["id"] == "route06")
q = task["question"]
arms = {
    "plain": lambda: run_loop(q, make_tools(task), SYSTEM, part="a03", thinking=DISABLED, max_tokens=4000),
    "react": lambda: run_loop(q, make_tools(task), SYSTEM + REACT_RULES, part="a03", thinking=DISABLED,
                              max_tokens=4000),
    "plan": lambda: plan_and_execute(q, make_tools(task), SYSTEM, part="a03"),
    "high": lambda: run_loop(q, make_tools(task), SYSTEM, part="a03", effort="high", max_tokens=20000),
}
for name, run in arms.items():
    r = run()
    print(f"{name:6s} correct={grade(task, r.answer)}  calls={len(r.steps)}  cost=${r.cost:.5f}")
```

```text
plain  correct=True  calls=5  cost=$0.00124
react  correct=False  calls=7  cost=$0.00136
plan   correct=False  calls=5  cost=$0.00157
high   correct=True  calls=5  cost=$0.00276
```

One run is an anecdote. Each arm below ran all 36 tasks 3 times: 108 runs per arm, 756 in all. Rates come with a 95% Wilson interval, a confidence interval for a proportion that behaves sensibly near 0% and 100%. Differences between arms are tested by resampling whole tasks, since the same tasks repeat.

### The results

| Arm (Haiku 5.5) | Solved (95% CI) | Output tokens per run | of which thinking | Cost per run | Model time per run | Cost per correct answer |
|---|---|---|---|---|---|---|
| Plain loop, thinking off | 76.9% (68.1-83.8) | 1,685 | 0 | $0.00131 | 13.5 s | $0.00171 |
| ReAct prompt, thinking off | 79.6% (71.1-86.1) | 1,982 | 0 | $0.00152 | 14.5 s | $0.00190 |
| Plan-and-execute, thinking off | 77.8% (69.1-84.6) | 2,725 | 0 | $0.00207 | 19.2 s | $0.00267 |
| Adaptive thinking, `low` | 88.0% (80.5-92.8) | 2,642 | 1,588 | $0.00188 | 19.5 s | $0.00214 |
| Adaptive thinking, `medium` | 88.9% (81.6-93.5) | 3,054 | 1,933 | $0.00214 | 21.3 s | $0.00241 |
| Adaptive thinking, `high` | 91.7% (84.9-95.6) | 4,667 | 3,485 | $0.00317 | 27.5 s | $0.00345 |
| Adaptive thinking, `xhigh` | 96.3% (90.9-98.6) | 8,313 | 7,102 | $0.00553 | 40.9 s | $0.00574 |

Only thinking moved the score beyond noise. Against the plain loop, `low` effort added 11.1 points (task-resampled interval +2.8 to +19.4), `medium` 12.0 and `high` 14.8. The ReAct prompt added 2.8 points (-6.5 to +13.0) and plan-and-execute 0.9 (-7.4 to +9.3): both inside the noise, for 16% and 58% more cost.

The `xhigh` row needs a footnote. With `max_tokens` at 20,000, the cap used for every thinking arm, 12 of its 15 lock-puzzle runs stopped at `max_tokens` before answering, and the arm scored 85.2%. The [effort docs](https://platform.claude.com/docs/en/build-with-claude/effort) warn about exactly this: at high effort, `max_tokens` "is a hard limit on total output (thinking plus response text)." Rerunning those 15 with a 64,000-token cap solved all 15, and the table uses those runs. Above about 21,000 tokens the Python SDK refuses a non-streaming request unless you pass a longer `timeout`, which `run_loop` now accepts.

![Scatter plot of accuracy against cost per 1,000 runs on a log scale. The three no-thinking Haiku arms cluster between $1.31 and $2.07 per 1,000 runs at 77 to 80 percent, with intervals from about 68 to 86 percent. Haiku with adaptive thinking rises from 88 percent at $1.88 (low) and 88.9 percent at $2.14 (medium) to 91.7 percent at $3.17 (high) and 96.3 percent at $5.53 (xhigh). Sonnet 5.5 at medium effort sits at 97.2 percent and $21.20, with a wide interval from 86 to 99.5 percent.](/diagrams/a03-accuracy-cost.svg "Tasks solved against cost per 1,000 runs, 36 tasks (claude-haiku-5-5: 108 runs per arm; claude-sonnet-5-5: 36 runs). Switching thinking on costs 57 cents more per 1,000 runs; each later step costs more per point.")

## Step 5: where extra reasoning stops paying

### The marginal cost of a correct answer

Accuracy per dollar always favours the cheapest arm. The useful number is the extra cost of each extra correct answer when you step up:

| Step up | Accuracy change (95% CI) | Cost per run change | Extra cost per extra correct answer |
|---|---|---|---|
| Thinking off to `low` | +11.1 points (+2.8 to +19.4) | +$0.00057 (1.44x) | $0.005 |
| `low` to `medium` | +0.9 (-3.7 to +6.5) | +$0.00026 (1.14x) | $0.028, if real |
| `medium` to `high` | +2.8 (-1.9 to +8.3) | +$0.00102 (1.48x) | $0.037, if real |
| `high` to `xhigh` | +4.6 (+0.9 to +10.2) | +$0.00236 (1.75x) | $0.051 |

The first step costs half a cent per extra solved task. Each later step costs 6 to 10 times as much per extra answer, and the middle two are indistinguishable from zero on 108 runs. Your threshold depends on what a wrong answer costs you; here, only the first step is an easy call.

### Where the thinking went

The gains and the costs came from different tasks:

| Arm | Chain | Aggregate | Route | Puzzle | Puzzle cost per run |
|---|---|---|---|---|---|
| Plain loop | 78% | 93% | 74% | 63% | $0.00131 |
| ReAct | 89% | 93% | 59% | 78% | $0.00159 |
| Plan-and-execute | 89% | 93% | 63% | 67% | $0.00269 |
| `low` | 74% | 100% | 89% | 89% | $0.00295 |
| `medium` | 74% | 96% | 96% | 89% | $0.00344 |
| `high` | 81% | 100% | 100% | 85% | $0.00679 |
| `xhigh` | 89% | 100% | 100% | 96% | $0.01506 |

Each cell is 27 runs, with an interval about 30 points wide, so read the rows rather than single cells. Routes and puzzles are where thinking paid: the route column went from 74% to 100%. Puzzles also took most of the tokens: 62% to 63% of all thinking at `low` and `medium`. On the five lock puzzles alone, `medium` solved 15 of 15 at $0.0039 a run, `high` 14 of 15 at $0.0100, and `xhigh` 15 of 15 at $0.0246 and 155 seconds a run. Past `medium`, the model kept thinking on puzzles it was already solving.

### Thinking between tool calls

The transcripts show where adaptive thinking happens. At `high` effort, every first request had a thinking block, but only 10% of thinking tokens were written there. 75% came on middle steps, after a tool result and before the next call, and 15% before the final answer. At `low` and `medium`, middle steps again took 65%. On tasks where the next move depends on the last result, the work happens between tool calls.

Here is the start of a lock puzzle at `medium` (the bracketed block is the visible summary; the step was billed 4,373 thinking tokens):

```text
guess("0123") -> bulls: 2, cows: 0      guess("4567") -> bulls: 1, cows: 0
[thinking] So the code has exactly two digits from {0,1,2,3} in correct positions, one digit
from {4,5,6,7} in its correct position, and one digit from {8,9}. ...
guess("0827") -> bulls: 2, cows: 1
[thinking] I'm narrowing down the candidates 0187, 0528, 8127, and 8523 by testing how guessing
0187 itself would score ... this guess cleanly separates all four possibilities.
guess("0187") -> bulls: 2, cows: 1
ANSWER: 8127
```

### Why the ReAct prompt did so little

With thinking off, the plain loop already wrote text before 94% of its tool calls, 259 characters on average ("Next, I'll check the roads from each of those three towns."). The ReAct prompt made that 100%, at 366 characters, and labelled it "Thought:". Haiku 5.5 already narrates; the prompt mostly lengthened the narration. The [Haiku 5.5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5) notes the model "sometimes writes reasoning-like text in the reply that users see," more often with thinking off.

### Latency

Mean model time per run went from 13.5 s with thinking off to 21.3 s at `medium` and 40.9 s at `xhigh`; at the 90th percentile, from 20.9 s to 33.1 s and 108 s. Tool calls stayed between 15.5 and 18 per run in every arm, so the extra time was almost all thinking tokens being generated.

### A stronger model instead

As a headline comparison, `claude-sonnet-5-5` ran each task once at `medium` effort with adaptive thinking: 35 of 36 correct (97.2%, interval 85.8 to 99.5), $0.0212 per run, 15.7 s. It thought far less (431 thinking tokens per run against Haiku `xhigh`'s 7,102) and was faster, but cost 3.8 times as much as Haiku at `xhigh` and 9.9 times Haiku at `medium`. It solved all nine chain tasks; its one miss was the 71-versus-64 km route from the opening. With 36 runs, its lead over Haiku at `xhigh` is within noise.

## Step 6: failures from real transcripts

### The plan that couldn't see the roads

Plan-and-execute was worst on routes. The planner wrote a sensible breadth-first search plan, and the executor followed it:

```text
Candidate routes into Rookhaven's neighbours:
- Kettle Combe → Fairholm (21) → Old Toll (15) → Rookhaven (35): 71 km
- Kettle Combe → Oxley Bay (41) → Coldharbour (13) → Rookhaven (19): 73 km
- Kettle Combe → Smithy Cross (22) → Coldharbour (29) → Rookhaven (19): 70 km
```

It had read "Kettle Combe - Port Wren: 6 km" and "Port Wren - Smithy Cross: 8 km" one step earlier, which make a 62 km route, and answered 70, the fewest-roads trap. Of plan-and-execute's 10 wrong route answers, 5 were exactly the trap distance, against 1 of 7 for the plain loop and none with thinking on. Replanning rarely helped: the executor asked for a new plan in 7 of 108 runs, all puzzles, and 2 of those then succeeded.

### The visible thought that checked only one case

In a pilot run with the ReAct settings, a 12-coin puzzle came down to four candidates: 5 or 6 heavy, 2 or 3 light. The agent designed its last weighing in the open:

```text
Thought: ... weigh {5, 2} vs {6, 9}.
- If 5 is heavy: left heavier.
- If 6 is heavy: right heavier.
- If 2 is light: right heavier... that collides with 6 heavy, so I'll use a different design.

Better: weigh {5, 2} vs {6, 3}. 5 heavy → left heavier. 6 heavy → right heavier. 2 light → left
lighter. 3 light → right lighter. Each case gives a distinct result.
```

They aren't distinct. A lighter right pan is the same reading as a heavier left pan, so "5 heavy" and "3 light" collide, exactly the collision it had just caught one line earlier. The scale said left heavier, the agent answered "5 heavier", and coin 3 was the answer. The thought was written out and still wrong: the reasoning error ReAct's authors counted in 47% of their failures.

### The sign that thinking didn't catch

Most chain errors came from three "how many years older" tasks. At `medium` effort:

```text
The oldest mayor of a Toller town is Quinn Ingram of Saltmarsh, born 1974. MV Tern is operated by
Kestrel Mills, founded by Orla Lockhart, born 1985. The difference is 1974 − 1985 = −11, so the
mayor is 11 years younger than the founder.
ANSWER: -11
```

Born earlier means older, so the answer is 11. Thinking at `low` and `medium` got this task wrong in all 6 runs; the ReAct and plan-and-execute arms got it right in all 6, and `high` and `xhigh` in 5 of 6. A guess at why: writing the steps out as text pinned down which year subtracts from which. Thinking fixed searching and puzzle logic, and on this one arithmetic convention it did worse than visible reasoning.

### The answer that never arrived

At `xhigh` with a 20,000-token cap, one lock puzzle ended after three calls: the third request wrote 20,000 tokens of thinking and stopped with `stop_reason: "max_tokens"` and no text, billed in full. Size `max_tokens` for the effort level, and treat a `max_tokens` stop as a failure your loop must handle.

The whole experiment, including pilots, two crashed sweeps and rate-limit retries, cost $3.12 on the ledger. The 756 main Haiku runs cost $1.73, the 15 `xhigh` reruns $0.37 and the 36 Sonnet runs $0.76.

## In the real world

- Anthropic's [think tool post](https://www.anthropic.com/engineering/claude-think-tool) (March 2025) gave Claude 3.7 Sonnet a no-op tool for pausing to reason between tool calls, a ReAct thought as a tool. On the τ-bench airline domain, a customer-service benchmark, it reported a 54% relative improvement with an optimized prompt. A December 2025 update now recommends extended thinking "instead of a dedicated think tool in most cases," the shift this article measured.
- Anthropic's [effort guidance for Haiku 5.5](https://platform.claude.com/docs/en/build-with-claude/effort) recommends starting at `medium`, `low` for "short tool tasks," and `xhigh` or `max` "only where your evals show a quality gain," compared against Sonnet 5.5. Here, `xhigh` beat `medium` by 7.4 points (+1.9 to +14.8) at 2.6 times the cost.
- ReWOO's benchmarks show plan-first designs winning on token count when steps are independent: 64% fewer tokens than ReAct, averaged over six benchmarks. Here, where each step depends on the last, plan-and-execute used more tokens than the plain loop, because the plan (676 output tokens on average) rides along in every executor request.
- The production guardrails around these loops, such as step limits and repeated-action detection, are in [Building Tool-Using Agents That Work Reliably](/blog/reliable-tool-using-agents/) and [Detecting and Recovering From Agent Loops](/blog/agent-loop-detection-recovery/).

## Common misconceptions

- **"ReAct is how you make an agent reason."** On a current model it mostly relabels narration the model already writes. The ReAct prompt moved Haiku 5.5 by 2.8 points, inside the noise, while thinking at `low` moved it by 11.1 for a similar cost.
- **"Planning up front is cheaper than reasoning as you go."** That holds when steps don't depend on each other, as in ReWOO's benchmarks. Here the plan cost 58% more than the plain loop, because it rode along in every step, and it fell for the trap answer on routes it couldn't foresee.
- **"Higher effort is always better."** It's better on average and worse per dollar. From `medium` to `high` the gain was within noise for 48% more cost, and on lock puzzles `high` spent 2.5 times as much and solved no more.
- **"Thinking happens before the agent starts."** In these runs only 10% to 20% of thinking tokens came before the first tool call. Most came between tool calls, reacting to results.
- **"If the summary is short, the thinking was cheap."** A three-line visible summary sat on a step billed 4,373 thinking tokens. Read `output_tokens_details.thinking_tokens` for the real count.

## Key takeaways

- There are three places to put an agent's reasoning: visible text before each action (ReAct), one plan before the first action (plan-and-execute), or hidden thinking the model chooses to do (adaptive thinking).
- On Haiku 5.5, turning thinking on was the step that paid: from 77% to 88% of tasks solved, at half a cent per extra correct answer.
- Each effort level above that cost more per extra answer, and the `medium`-to-`high` step was within noise. Measure your own tasks before raising effort.
- Most thinking happened between tool calls. Interleaved thinking is how a modern model does what ReAct's thoughts were for.
- At `xhigh` and `max`, size `max_tokens` for the thinking or the answer never arrives. 12 of 15 lock puzzles hit a 20,000-token cap.

## What's next

Every arm here reasoned forward and answered once, even when, as in the coin puzzle, a second look at its own reasoning would have caught the mistake. [Part 4](/blog/agent-self-correction/) asks what happens when an agent checks its work: Reflexion, critique loops and verifier tools such as unit tests, and when self-critique fixes an answer versus talks the model out of a right one.

## Further reading

- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629): the thought-action-observation loop, HotpotQA, FEVER, ALFWorld and WebShop results, and the error analysis.
- [Plan-and-Solve Prompting (Wang et al., 2023)](https://arxiv.org/abs/2305.04091): "devise a plan, then carry it out" as a zero-shot prompt.
- [ReWOO: Decoupling Reasoning from Observations (Xu et al., 2023)](https://arxiv.org/abs/2305.18323): planner, worker and solver, and the token savings of planning first.
- [An LLM Compiler for Parallel Function Calling (Kim et al., 2024)](https://arxiv.org/abs/2312.04511): plans as dependency graphs, executed in parallel.
- [Effort (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/effort): effort levels, per-model defaults and recommendations, and effort with tool use.
- [Thinking (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/thinking): adaptive thinking, disabling it per model, interleaved thinking and thinking-block preservation.
- [Steering thinking (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost): how the model decides when to think, and how thinking is billed.
- [Prompting Claude Haiku 5.5 (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5): using effort to control thinking, and its quirks with thinking off.
- [The "think" tool (Anthropic, 2025)](https://www.anthropic.com/engineering/claude-think-tool): a ReAct-style thinking tool, its τ-bench results, and the later advice to prefer extended thinking.
- [Reasoning Models: Chain of Thought, RL and Test-Time Compute](/blog/reasoning-models/): how models learned to think before answering, and what thinking tokens cost.
