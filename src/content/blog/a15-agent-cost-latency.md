---
title: "Cost and Latency of Agents: Caching, Effort, Models and Batch, Measured"
description: "Cost per completed task and time per task on one 36-task set, lever by lever: prompt caching, effort, Haiku vs Sonnet, a Sonnet lead and the Batch API."
slug: "agent-cost-latency"
category: "agents-tool-use"
tags: ["agents", "cost", "latency", "caching", "routing"]
series: "agentic-ai"
seriesOrder: 15
pubDate: 2026-07-20
updatedDate: 2026-10-25
---

## TL;DR

What an agent costs is everything spent across all its steps, failed runs included, divided by the tasks it got right. This part runs one fixed set of 36 tool-using tasks through five levers and records that cost per completed task, plus time per task: prompt caching, effort and thinking, Claude Haiku 5.5 against Claude Sonnet 5.5, a Sonnet lead with Haiku workers, and the Message Batches API. Across 900 graded runs, the cheapest setup was 17 times cheaper per completed task than the dearest. The Batch API halved the bill with no loss of accuracy, at 60 times the waiting. Caching cut cost by 37% and changed neither accuracy nor speed. Turning thinking off was the only lever that cost measurable accuracy. Sonnet cost 8 times as much per completed task and finished sooner than Haiku, with no accuracy gain this set could detect.

## Why it matters

Claude Haiku 5.5 answered these 36 questions for $2.54 per thousand correct answers. Claude Sonnet 5.5 answered them for $20.88 per thousand, and got there faster: 13.6 seconds per task against 18.8. The same Haiku agent sent through the Batch API cost $1.26 per thousand correct answers, and each answer took a median of 18 minutes.

Each lever moves price, waiting time and accuracy by different amounts, and some move accuracy without telling you. [Cutting LLM Cost by 70% Without Losing Quality](/blog/cut-llm-cost-70-percent/) (System Design 6) lays out the levers for any LLM workload, and [Routing Requests Between a Cheap and a Strong Model](/blog/routing-cheap-and-strong-models/) (System Design 9) covers choosing a model per request. This part measures what those levers do to an agent, where one task is many calls, and a failed task is paid for in full.

## The intuition

Think of a courier company working out what a delivery costs. Fuel per kilometre misses a van that takes a longer route and a parcel that comes back undelivered and goes out again tomorrow. The useful number is cost per parcel delivered.

A bigger van is a stronger model. Driving more carefully is effort. Not reloading the same boxes at every stop is the prompt cache. Overnight freight, half price but you get it tomorrow, is the Batch API. A dispatcher in the office who plans each route for a driver is the lead-and-workers design.

The analogy breaks in one place. A courier knows when a parcel came back. An agent that answers wrongly usually sounds as sure as one that answered right, so "cost per completed task" assumes you can tell which runs failed. When you can't, the error rate is a cost of its own.

## Step 1: one task set, one unit

### The task set and the harness

All runs use the Saltmere Islands set from [Part 3](/blog/react-planning/): a fictional archipelago of 82 records, six tools (`search`, `lookup`, `calculate`, `roads`, and two puzzle tools with a use limit, `weigh` and `guess`) and 36 tasks, 9 of each kind. Chains follow facts from record to record, aggregates count or sum across records, routes ask for the shortest road between two towns, and puzzles find a fake coin or a lock code within a limited number of tool uses. A generator computed every answer, and grading is exact. Parts 3 and 11 measured this set before, so its noise is known: in [Part 11](/blog/agent-evals/), whole-set scores of one configuration had a standard deviation of 4.4 to 5.4 points from run to run.

The agent is the lab's loop from [Part 1](/blog/what-is-an-ai-agent/), extended in a new module, `costlat.py`, so caching can be switched off or set to a one-hour lifetime. Here is how one request is built:

```python
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
```

Two cache breakpoints, the same as Part 1: one after the tools and system prompt, which never change, and one that follows the end of the conversation, so each step reads everything before it from the cache and pays the write price only on what is new.

### Cost per completed task

Every arm's cost comes from the lab's ledger, which prices each response's `usage` at the published rates. Cost per completed task is the arm's total spend divided by the runs that got the right answer:

```text
cost per completed task = total spend / correct runs = cost per run / success rate
```

With small numbers: if a run costs $0.002 and succeeds 80% of the time, a correct answer costs $0.002 / 0.8 = $0.0025, because the fifth run that failed was paid for too. [Part 11](/blog/agent-evals/) used the same unit, following Kapoor and colleagues' argument that agent results must be "cost-controlled."

### Latency per task

Latency here is model time on the critical path: the sum of the wall-clock times of a run's model calls, timed around each request on this machine over the public API. Tools run locally in microseconds. The lab's budget guard sends a free token-count request before every call, which added about 2 seconds per step; that is excluded, since a production agent wouldn't make it. For a lead with workers, workers started in the same turn run in parallel, so the critical path is the lead's calls plus the slowest worker of each turn. For the Batch API it is the time from the first submission to the end of the run's last batch.

Times below are medians and 95th percentiles (p95, the time that 1 run in 20 exceeds). All runs happened on October 25, 2026 between 19:05 and 22:05 IST, with at most 3 requests in flight. API latency varies by time of day and load, so the absolute seconds are one evening's sample; the ratios between arms travel better. The synchronous arms ran interleaved, one repeat of each in turn, so a slow hour hit them all.

![A flowchart of one agent step and the levers on it. A task reaches a decision, someone waiting for the answer? Yes leads to a synchronous request taking seconds per step, no leads to a Message Batch at half price taking minutes per round. Both feed a box of levers on every step: the prompt cache, which rereads history at 0.05 to 0.1 times the input price, effort and thinking, which set output tokens per step, and the model or a lead with workers, which sets the price per token. The levers feed one model step. On tool_use the harness runs tools, appends results and returns to the decision; on end_turn the answer is graded, giving cost per completed task.](/diagrams/a15-cost-levers.svg "Where each lever acts in the agent loop. The batch decision repeats on every step, which is why its delay multiplies by the number of steps.")

Every lever acts on every step, so whatever a lever saves or costs per call is multiplied by the number of calls in a run, about six here.

## Step 2: the baseline and the prompt cache

### The baseline

The reference arm is `claude-haiku-5-5` with adaptive thinking (the model decides whether to think before each reply) at effort `medium`, caching on, `max_tokens` 16,000 and a $0.05 cap per run. Each of the 36 tasks ran 3 times: 108 runs. It solved 96 of them, 88.9% (95% Wilson interval 81.6-93.5), matching the 88.9% Part 3 measured with the same settings. A run averaged 5.9 model calls, read 18,402 input tokens and wrote 3,250 output tokens, 2,165 of them thinking.

Output tokens, including thinking, were 72% of the bill. Cache writes were 22%, cache reads 6%. Haiku 5.5 charges $0.50 per million output tokens and $0.01 per million cache reads, a factor of 50, so on short agent runs the model's own writing is the main cost.

### Caching on and off

The same arm ran with no cache breakpoints: 108 more runs.

| Haiku 5.5, `medium` | Correct (95% CI) | Cost per run | Cost per completed task | Cache hit rate | Median time | p95 time |
|---|---|---|---|---|---|---|
| Caching on | 96/108, 88.9% (81.6-93.5) | $0.00226 | $0.00254 | 79% | 18.8 s | 47.1 s |
| Caching off | 97/108, 89.8% (82.7-94.2) | $0.00363 | $0.00404 | 0% | 19.1 s | 47.2 s |

The cache hit rate is the share of all input tokens read from the cache. Caching cut cost per completed task by 37%, changed accuracy by +0.9 points with an interval of -4.6 to +6.5 (resampling whole tasks, since the same tasks repeat), and changed time by nothing measurable.

A 37% saving is smaller than [Part 5](/blog/context-engineering/)'s factor of five on 20-to-48-step chains, and the reason is the bill's shape. Caching discounts input; here input was only 54% of the uncached bill, because runs were short (about six calls and 19,000 input tokens) and output-heavy. The longer the run and the larger the history it rereads, the more of the bill caching can reach.

### Does a cache hit make a call faster?

The prompt caching docs say you "generally see improved time-to-first-token for long documents." The agent calls above read a median of 2,469 tokens and at most 14,920, too short to show it, so a separate test sent one short request after a long fixed prefix: thinking disabled, effort `low`, `max_tokens` 8, so the call is almost all prompt processing. Each prefix carried a random marker so no earlier cache entry could match. After one call wrote the cache, 6 cached and 6 uncached calls alternated.

| Prompt length | Median time, cache read | Median time, no cache | Cost per call, cache read | Cost per call, no cache |
|---|---|---|---|---|
| 4,634 tokens | 1.29 s | 1.13 s | $0.00005 | $0.00047 |
| 32,174 tokens | 1.10 s | 1.05 s | $0.00033 | $0.00322 |
| 123,970 tokens | 1.08 s | 1.06 s | $0.0062 | $0.0620 |

At every length the cache cut the price by a factor of about 10 and the time by nothing. The 124,000-token row also crossed Haiku 5.5's 100,000-token line, where every rate rises fivefold, so one uncached call cost six cents (that row was meant to stay under the line; the prefix came out larger than planned). Twelve calls per row can't rule out a few tens of milliseconds, and a server may reuse computation it doesn't bill as a cache hit. On Haiku 5.5, on this evening, the cache changed the price and left the time alone.

## Step 3: effort, thinking and the model

### Effort and thinking

[Part 3](/blog/react-planning/) mapped effort against accuracy on this set. Repeated here with latency and cost per completed task, 108 runs per row, all Haiku 5.5 with caching on:

| Setting | Correct (95% CI) | Change vs `medium` (95% CI) | Cost per completed task | Median time | p95 time | Thinking tokens per run |
|---|---|---|---|---|---|---|
| Thinking off | 86/108, 79.6% (71.1-86.1) | -9.3 (-17.6 to -1.9) | $0.00173 | 12.8 s | 23.0 s | 0 |
| `low` | 93/108, 86.1% (78.3-91.4) | -2.8 (-13.0 to +5.6) | $0.00228 | 16.9 s | 42.3 s | 1,685 |
| `medium` | 96/108, 88.9% (81.6-93.5) | 0 | $0.00254 | 18.8 s | 47.1 s | 2,165 |
| `high` | 101/108, 93.5% (87.2-96.8) | +4.6 (-0.9 to +11.1) | $0.00327 | 22.7 s | 73.2 s | 3,326 |

"Thinking off" is `thinking: {"type": "disabled"}` at effort `medium`, which Haiku 5.5 accepts at `high` effort or below. It was the only change in this whole article whose accuracy loss was outside the noise: 9.3 points. It still had the lowest synchronous cost per completed task, because a run without thinking is cheap enough that paying for the failures costs less than paying for the thinking. That only holds if you know which answers failed and can rerun them.

Effort moves the tail more than the middle. From thinking off to `high`, the median time rose 1.8 times and the p95 3.2 times. The slow runs were puzzles: at `high` a puzzle took a median 42.7 seconds against 17.9 for a chain, because thinking concentrates where the task is hard. A step's time scales with the tokens it writes, so a 95th-percentile budget for an agent depends on effort far more than its median does.

### Haiku or Sonnet

`claude-sonnet-5-5` ran the same loop at effort `medium` with adaptive thinking, 2 repeats (72 runs), a smaller headline comparison:

| Model | Correct (95% CI) | Cost per run | Cost per completed task | Median time | p95 time | Output tokens per run | Model calls per run |
|---|---|---|---|---|---|---|---|
| Haiku 5.5 | 96/108, 88.9% (81.6-93.5) | $0.00226 | $0.00254 | 18.8 s | 47.1 s | 3,250 | 5.9 |
| Sonnet 5.5 | 68/72, 94.4% (86.6-97.8) | $0.01972 | $0.02088 | 13.6 s | 29.4 s | 1,337 | 5.3 |

Sonnet 5.5's list price is 20 times Haiku 5.5's per token. It cost 8.7 times as much per run, because it wrote 59% fewer output tokens: 380 tokens of thinking per run against 2,165. The difference in accuracy, +5.6 points (-5.1 to +16.2), is inside the noise. Sonnet was also faster, though it generated about 86 output tokens per second of call time against Haiku's 141, because it had less to write.

All four of Sonnet's failures were shortest-route tasks, where its light thinking cut a search short. On `route06` it compared seven routes and answered 71 km, where 64 was right:

```text
I did not look up every town on the map. The routes through Barrow Gate and the other unchecked
towns already started longer than 71 km, so I stopped there.
```

That pruning rule is wrong for shortest paths, where a route that starts long can still finish short. Part 3 found the same kind of error in Haiku with thinking switched off.

### A Sonnet lead with Haiku workers

[Part 7](/blog/multi-agent-systems/)'s design gives a lead agent one tool, `delegate`, which starts a worker agent with a fresh context and the record tools. Here the lead is Sonnet 5.5 and the workers Haiku 5.5, both at effort `medium` with adaptive thinking and caching. The lead's prompt asks it to split only independent subtasks and to give any chain, route or puzzle to a single worker. Two repeats, 72 runs:

| Design | Correct (95% CI) | Cost per completed task | Median time | p95 time | Model calls per run |
|---|---|---|---|---|---|
| Haiku alone | 96/108, 88.9% (81.6-93.5) | $0.00254 | 18.8 s | 47.1 s | 5.9 |
| Sonnet alone | 68/72, 94.4% (86.6-97.8) | $0.02088 | 13.6 s | 29.4 s | 5.3 |
| Sonnet lead, Haiku workers | 70/72, 97.2% (90.4-99.2) | $0.01072 | 25.2 s | 65.9 s | 8.4 |

The lead used one worker in 64 of 72 runs and two in the other 8, so on this set the design is a Sonnet brief, a Haiku agent, and a Sonnet check of the report. The lead cost $0.0078 per task for an average of 2.1 calls, 75% of the bill; the workers cost $0.0026. Paying Sonnet only for planning and checking halved the cost of Sonnet alone, but it was still 4.2 times Haiku alone per completed task, and the slowest of the three, because the lead's first call comes before any work starts and its last after all work ends.

Its accuracy, 97.2%, was the highest measured, and +8.3 points over the Haiku baseline (+0.5 to +17.1). That interval barely clears zero, and Step 4 shows why one interval like that, among eight comparisons, is weak evidence.

## Step 4: the Batch API for work nobody waits for

### An agent loop in batches

The Message Batches API takes many requests at once, processes them asynchronously at 50% of the normal price, and returns results when all have ended; the docs say "most batches finish in less than 1 hour" and a batch expires after 24 hours. An agent can't send its whole run as one batch, because each step depends on the tool results of the last. So `costlat.run_batch` advances many runs in lockstep: each round sends the next request of every unfinished run as one batch, waits for it to end, runs the tools, and builds the next round.

```python
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger
from costlat import run_batch
from saltmere import SYSTEM, TASKS, make_tools

jobs = {f"{t['id']}-r{k}": (t["question"], make_tools(t)) for k in (1, 2, 3) for t in TASKS}
results, finished, rounds = run_batch(jobs, SYSTEM, part="a15", client=ledger.client(),
                                      model="claude-haiku-5-5", effort="medium", cache=True)
```

Before each round, the ledger's guard checks a worst-case estimate for the whole batch, and each result is recorded at the batch price. The 108 runs, the same Haiku `medium` configuration as the baseline, went through twice: with the usual 5-minute cache entries, and with 1-hour entries, which the batch docs suggest "because batches can take longer than 5 minutes to process."

### Half the price, sixty times the wait

| Haiku 5.5, `medium` | Correct (95% CI) | Cost per run | Cost per completed task | Cache hit rate | Median time per task | p95 time per task |
|---|---|---|---|---|---|---|
| Synchronous | 96/108, 88.9% (81.6-93.5) | $0.00226 | $0.00254 | 79% | 18.8 s | 47.1 s |
| Batch, 5-minute cache | 104/108, 96.3% (90.9-98.6) | $0.00122 | $0.00126 | 78% | 18.5 min | 27.1 min |
| Batch, 1-hour cache | 100/108, 92.6% (86.1-96.2) | $0.00138 | $0.00150 | 76% | 15.4 min | 28.1 min |

A batch round took a median of 141 seconds (5-minute run) and 131 seconds (1-hour run), with a range of 77 to 206 seconds, however many requests it held: the first round carried 108 requests, the last one or two. A run that needs six model calls waits six rounds, about a quarter of an hour, for work that takes 19 seconds synchronously. The whole set of 108 runs finished in 35 and 33 minutes. The synchronous arm took about 7 minutes per 36 runs at 3 requests in flight, about 21 minutes for 108. A batch's total time depends on its number of rounds and hardly at all on its number of runs, so for a large non-interactive job it needs no concurrency limit of your own.

The 5-minute cache kept working, at a 78% hit rate, because every round ended inside five minutes and each read refreshes an entry's lifetime. One-hour entries bought nothing and cost 14% more, since a 1-hour write costs 2 times the input price instead of 1.25 times. That would reverse if rounds took longer than five minutes, which the docs warn can happen under load.

### The same agent, four different scores

The two batch arms and the two synchronous Haiku `medium` arms ran the same model, effort, prompts and tasks. They scored 88.9%, 89.8%, 96.3% and 92.6%. The requests were identical, so the 5-minute batch arm's +7.4-point lead over the synchronous baseline (+1.9 to +13.9) is the kind of result that turns up by chance when eight arms are each compared with one baseline. Pooled, the four arms solved 397 of 432 runs, 91.9%. That spread is the noise floor for every accuracy difference in this article, and only the thinking-off result sits clearly below it.

## Which levers cost quality

![A scatter chart of cost per 1,000 completed tasks against median time per task, both on log scales. Synchronous Haiku 5.5 arms cluster between 12.8 and 22.7 seconds and $1.73 to $3.27, with the no-cache arm higher at $4.04. Sonnet 5.5 alone sits at 13.6 seconds and $20.88, and the Sonnet lead with Haiku workers at 25.2 seconds and $10.72. The two batch arms sit far right at 15 to 18 minutes and $1.26 to $1.50.](/diagrams/a15-cost-latency.svg "Cost per 1,000 completed tasks against median time per task, 36 Saltmere tasks (claude-haiku-5-5 and claude-sonnet-5-5, adaptive thinking unless marked; 108 runs per Haiku arm, 72 per Sonnet arm).")

Lower and further left is better: the batch arms pay for the lowest cost with the longest waits, and Sonnet pays for short waits with the highest cost.

All changes are relative to the Haiku 5.5 baseline (`medium`, adaptive thinking, caching on, synchronous):

| Lever | Cost per completed task | Median time | Accuracy change (95% CI) | Costs quality here? |
|---|---|---|---|---|
| Caching off | 1.59x | 1.02x | +0.9 (-4.6 to +6.5) | No |
| Thinking off | 0.68x | 0.68x | -9.3 (-17.6 to -1.9) | Yes |
| Effort `low` | 0.90x | 0.90x | -2.8 (-13.0 to +5.6) | Not measurable |
| Effort `high` | 1.29x | 1.21x | +4.6 (-0.9 to +11.1) | Not measurable |
| Sonnet 5.5 | 8.2x | 0.72x | +5.6 (-5.1 to +16.2) | Not measurable |
| Sonnet lead, Haiku workers | 4.2x | 1.34x | +8.3 (+0.5 to +17.1) | Possibly a gain |
| Batch API, 5-minute cache | 0.50x | 59x | +7.4 (+1.9 to +13.9) | No (same model) |

Caching, batching and the cache lifetime don't change what the model is asked, so they carry no quality risk; they move price and waiting time only. Effort, thinking and model choice change the model's work, and on 108 runs only the largest of their accuracy moves showed. Part 11's power analysis found that even the 10-point gap between thinking off and `low` on this set needs about 100 tasks with 3 runs each to show reliably; a 3-point gap needs far more.

All experiments in this article cost $4.37 on the ledger: $1.33 for the five synchronous Haiku arms, $1.46 for Sonnet alone, $0.77 for the Sonnet-led team, $0.28 for the two batch arms, $0.52 for the prefix-latency test and under $0.01 for the batch pilot. The lab's ledger first priced 1-hour cache writes at the 5-minute rate, which put the 1-hour batch arm at $0.13 instead of $0.15; it now prices them at 2 times input, and the figures here are the corrected ones.

## In the real world

- Anthropic's [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (Hadfield et al., June 13, 2025) reports that "agents typically use about 4× more tokens than chat interactions, and multi-agent systems use about 15× more tokens than chats," and that on its BrowseComp evaluation "token usage by itself explains 80% of the variance" in performance. Its conclusion is economic: multi-agent systems need tasks "where the value of the task is high enough to pay for the increased performance."
- Claude Code's [cost docs](https://code.claude.com/docs/en/costs) put the average at "around $13 per developer per active day and $150-250 per developer per month, with costs remaining below $30 per active day for 90% of users," and say agent teams "use approximately 7x more tokens than standard sessions when teammates run in plan mode." They recommend a smaller model for simple subagent tasks, the arrangement of Step 3's lead with Haiku workers.
- The [Holistic Agent Leaderboard](https://arxiv.org/abs/2510.11977) (Kapoor et al., 2025) ran 21,730 rollouts across 9 models and 9 benchmarks for about $40,000, and found that "in 21 of 36 model-agent-benchmark combinations, increased reasoning effort produces equal or lower accuracy." It left latency for future work, because massively parallel evaluation adds variance to timings, the same caveat as the evening-long timings here.
- [AI Agents That Matter](https://arxiv.org/abs/2407.01502) (Kapoor et al., 2024) argues that agent evaluations "must be cost-controlled" and plots accuracy against cost as a Pareto frontier; on HotPotQA, jointly optimizing the two gave "53% lower variable cost with similar accuracy" for GPT-3.5. Other vendors' models are cited, not measured here.
- The [batch processing docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing) limit a batch to 100,000 requests or 256 MB, keep results for 29 days, and describe cache hits in batches as "best-effort," with typical hit rates of "30 percent to 98 percent." Both batch arms here landed at 76% to 78%.

## Common misconceptions

- **"The cheapest model is the cheapest agent."** Per token, yes. Per completed task, the answer depends on tokens written and failure rate as much as on price: Sonnet 5.5 is 20 times Haiku 5.5's price per token and cost 8.2 times as much per completed task here, because it wrote less. A model that fails more pays for its failures too.
- **"Prompt caching makes agents faster."** On these runs it cut cost by 37% and time by nothing, and a separate test at prompts up to 124,000 tokens found the same: a tenth of the price, the same 1.1 seconds. Treat it as a price lever unless your own timings say otherwise.
- **"A stronger model is slower."** Sonnet 5.5 generated output more slowly than Haiku 5.5 but finished tasks 28% sooner, because at the same effort level it thought much less. Time follows tokens written.
- **"Batch is only for one-shot jobs."** A multi-step agent runs fine through the Batch API: one batch per step, at half the price. The cost is waiting: about two to three minutes per step here, so a six-step task takes a quarter of an hour.
- **"Cost per completed task tells you what an answer costs."** Only if you can detect failures and retry them. Thinking off had the lowest synchronous cost per completed task and the most wrong answers; without a verifier, those wrong answers ship.

## Key takeaways

- Measure cost per completed task and time per task over whole runs, with repeats. Per-call prices hide the number of calls, the tokens each one writes and the runs that fail.
- On short agent runs most of the bill is output and thinking. Caching cut 37% here and up to five times on Part 5's long chains; the longer the history an agent rereads, the more caching saves.
- Caching and batching don't change what the model is asked, so they can't cost accuracy. Batching halves the price and turns seconds per step into minutes per step: use it whenever nobody is waiting.
- Effort, thinking and model choice do change the work. Only thinking off cost measurable accuracy on this set, and four runs of the identical configuration spread from 88.9% to 96.3%, so smaller differences need far more tasks to see.
- A strong lead over cheap workers cost between the two models and ran slower than either. On this set the lead handed nearly every task to a single worker, so it paid Sonnet prices to brief and check one Haiku agent.

## What's next

[Part 16](/blog/agents-in-production/), the series finale, builds one useful agent end to end with everything the series measured: tools and schemas, a loop with the right effort, a verifier, context management, a sandbox and permission policy, retries and budgets, evaluated with repeats and intervals, and priced per real result.

## Further reading

- [Pricing (Claude API docs)](https://platform.claude.com/docs/en/about-claude/pricing): per-token prices for Haiku 5.5 and Sonnet 5.5, cache multipliers, the batch discount and Haiku 5.5's long-prompt rates.
- [Batch processing (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/batch-processing): limits, turnaround, results retention and prompt caching in batches.
- [Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): breakpoints, the 1-hour lifetime and the usage fields behind the hit rates here.
- [Effort (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/effort): what each effort level does and the per-model recommendations for Haiku 5.5 and Sonnet 5.5.
- [AI Agents That Matter (Kapoor et al., 2024)](https://arxiv.org/abs/2407.01502): cost-controlled agent evaluation and accuracy-cost Pareto frontiers.
- [Holistic Agent Leaderboard (Kapoor et al., 2025)](https://arxiv.org/abs/2510.11977): cost, tokens and reasoning effort across 21,730 agent rollouts.
- [How we built our multi-agent research system (Hadfield et al., Anthropic, 2025)](https://www.anthropic.com/engineering/multi-agent-research-system): token multiples of agents and multi-agent systems, and when they pay.
- [Manage costs effectively (Claude Code docs)](https://code.claude.com/docs/en/costs): real per-developer costs of a coding agent and the levers it exposes.
