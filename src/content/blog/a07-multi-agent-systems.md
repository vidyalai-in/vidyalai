---
title: "Multi-Agent Systems: Orchestrators, Sub-agents and A2A, Measured"
description: "One agent or a lead with sub-agents? Measured on 26 tasks with Claude Haiku 5.5: accuracy, token multiples, cost and wall time, plus A2A on the wire."
slug: "multi-agent-systems"
category: "agents-tool-use"
tags: ["agents", "tool-use", "cost", "latency", "architecture"]
series: "agentic-ai"
seriesOrder: 7
pubDate: 2026-10-24
---

## TL;DR

In an orchestrator-workers system, a lead agent writes a brief for each subtask, each worker solves its brief in a fresh context window, and only a short report comes back. Here that design runs against a single agent on 18 tasks with Claude Haiku 5.5, 36 runs each. Both answered every task correctly. The team read 1.15 times the tokens and cost half as much, but only because splitting kept each request under the 100,000-token line where Haiku 5.5's price jumps fivefold; at flat prices it would have cost about 50% more. It was also slower, and its failures came from the handoff: briefs that ruled out shortcuts, a lead that "corrected" a right answer, and second passes nobody needed. The last part runs a worker behind the Agent2Agent (A2A) protocol 1.0 and shows the wire.

## Why it matters

Four Haiku workers answered a question about 55 incident records for $0.025. One Haiku agent answered the same question correctly for $0.083, more than three times as much, while reading fewer tokens. It made no mistake: its context window crossed 100,000 tokens, and Haiku 5.5 bills every request over that line at five times the base rate ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)).

Anthropic reports that its multi-agent research system used "about 15× more tokens than chats" and beat a single agent by 90.2% on its internal research evaluation ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)); a later Claude blog post puts the overhead at "3-10x more tokens than single-agent approaches for equivalent tasks." This part tests that trade on tasks whose answers are known.

[Part 5](/blog/context-engineering/) kept one agent's window small with sized tool results, compaction and memory files. A sub-agent is another answer: give a piece of the work its own window and discard it when the piece is done. Tracing such a system is covered in [Tracing Every Step of a Multi-Agent Workflow](/blog/tracing-multi-agent-workflows/) (System Design 23); this part asks whether to build one.

## The intuition

Picture a manager with four analysts. The manager splits a question into four parts, writes each analyst a one-page brief, and waits. Each analyst works alone at a clean desk, reads the files the brief points to, and hands back a one-page memo. The manager never sees the files, only the memos, and writes the final answer from them.

This works when the parts are independent and the brief says everything that matters. It fails when a brief leaves something out, two analysts do the same work, or the manager second-guesses a correct memo.

The analogy breaks in two places. Real analysts can walk down the hall and ask each other questions; these workers can't see each other at all. And a model rereads its whole desk at every step and pays for it, so four small desks can cost less than one big one, or more, depending on how the reading is priced.

## Step 1: the lead, the workers and the handoff

### Three designs

Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (December 19, 2024) defines the orchestrator-workers pattern: "a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results." The workers here are sub-agents: agents started by another agent, each with its own context window (the tokens one model call can read). Claude Code's [subagent docs](https://code.claude.com/docs/en/sub-agents) put it as "Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions."

Three designs ran on the same tasks:

- Single agent: the [Part 1](/blog/what-is-an-ai-agent/) loop, one model with the tracker tools and one growing window.
- Lead with workers: a lead agent (the orchestrator) whose only tool is `delegate`. Each call starts a worker with a fresh window, the tracker tools and nothing but the brief. Calls made in one turn run in parallel.
- The same team with a `claude-sonnet-5-5` lead, a small headline comparison.

### The delegate tool

The whole mechanism is one tool whose function runs another agent loop. From the lab's `multiagent.py`:

```python
def make_delegate(worker_tools, *, part, client, lead_run, model="claude-haiku-5-5", effort="medium",
                  max_cost=0.06, transcript_dir=None):
    count = iter(range(1, 1000))
    lock = threading.Lock()

    def delegate(task):
        with lock:
            k = next(count)
            w = Run("worker", brief=task)
            lead_run.workers.append(w)
        path = transcript_dir / f"worker{k:02d}.jsonl" if transcript_dir else None
        loop(task, worker_tools, WORKER, part=part, role=f"worker{k}", client=client, model=model,
             effort=effort, max_steps=30, max_cost=max_cost, transcript=path, run=w)
        if w.stop != "end_turn":
            return f"[worker stopped early: {w.stop}] {w.answer}".strip()
        return w.answer

    return Tool("delegate", "Start a worker agent with a fresh context and the incident tracker tools. "
                "It sees only the brief you write here and returns its final report.",
                {"task": {"type": "string", "description": "The complete, self-contained brief for the worker"}},
                delegate)
```

`loop` is Part 1's loop with one change: the tool calls of a turn run in a thread pool, so three `delegate` calls in one turn start three workers at once. The worker's final text is the tool result the lead reads. Everything the worker read stays in its own window and is discarded.

The lead's system prompt says a worker "knows nothing except the brief you write," asks it to delegate independent subtasks "in the same turn so they run in parallel," to keep unsplittable work with one worker, to name exact services, months, fields and report format in every brief, and to "check the reports against each other and the question before answering."

![A flowchart. A user question goes to a lead agent whose only tool is delegate. The lead exchanges a brief and a report with worker 1 and worker 2, each with a fresh context, and an A2A SendMessage request and task artifact with a remote worker behind an A2A server. All three workers call the incident tracker tools. The lead combines the reports into the ANSWER line.](/diagrams/a07-orchestrator-workers.svg "The lead sees briefs and reports, never the records. Workers can run in the same process or behind an A2A server.")

The lead never sees a record. In the runs below, briefs had a median of 899 characters and reports 1,208.

### Measuring the multiple

The token multiple is the total tokens processed by every agent in a run (all input read, cached or not, plus all output, summed over every model call) divided by the single agent's total on the same task. Cost comes from the lab's ledger at the API's prices.

## Step 2: the same 18 tasks, two designs

### The test bed

The tasks use Part 5's incident tracker for Larkspur Robotics, a fictional company: 1,000 generated records of about 2,150 tokens each, with service, severity, open time, follow-up incident, impact minutes, robots affected and root cause. Three kinds of task test three shapes of work:

- Wide (10 tasks): compare four services in one month, for example "Which of these four services had the highest total customer impact minutes, and what was that service's figure?" Each covers 53 to 64 records, and the work splits into four independent lookups.
- Two-stage (4 tasks): find the top incident in one service and month, then count incidents in a second service and month with the same root cause. The second half depends on the first.
- Chain (4 tasks, Part 5's `c01`, `c03`, `c05`, `c09`): follow 20 to 26 records, each naming the next. Nothing can be split.

A generator computes every answer, and grading is exact: for the wide tasks both the service and the number must be right.

Every agent runs `claude-haiku-5-5` at effort `medium` with adaptive thinking (the model decides when to think), prompt caching on, two repeats per task, so 36 runs per design. Workers in one run share a limit of 3 requests in flight.

### Results with full records

| Design | Correct (95% CI) | Tokens per task | Token multiple | Peak window | Cost per task | Median wall time |
|---|---|---|---|---|---|---|
| Single agent | 36/36, 100% (90-100) | 260,654 | 1.00 | 89,350 | $0.0408 | 22.0 s |
| Lead with workers | 36/36, 100% (90-100) | 299,246 | 1.15 | 54,042 | $0.0201 | 33.5 s |

The interval is a 95% Wilson interval, a range for a rate that behaves sensibly at 0% and 100%. "Peak window" is the most one call read, averaged over runs. Accuracy is identical, so the comparison comes down to tokens, money and time.

By kind of task:

| Task kind | Runs per design | Token multiple | Cost, single | Cost, team | Team cost ÷ single | Median wall, single | Median wall, team |
|---|---|---|---|---|---|---|---|
| Wide | 20 | 1.38 | $0.0635 | $0.0243 | 0.38 | 22.0 s | 33.3 s |
| Two-stage | 8 | 1.10 | $0.0111 | $0.0149 | 1.35 | 16.8 s | 26.7 s |
| Chain | 8 | 1.01 | $0.0140 | $0.0148 | 1.06 | 55.7 s | 64.5 s |

On wide tasks the lead did what the prompt asked: four workers, one per service, in one turn (4.1 workers per run on average). On two-stage tasks it usually gave the whole job to one worker, and on every chain it used exactly one. That kept the overhead on unsplittable work to 1% more tokens and 6% more money: the lead's own calls and the worker's second copy of the system prompt.

### Why the team was cheaper

The single agent read all 55 to 64 wide-task records in one parallel batch of tool calls, so its window jumped past 100,000 tokens (109,409 at peak on average) for its last call or two. Haiku 5.5 prices a request "by prompt length," counting cache reads and writes, and a request over the line "pays the higher prices even when part of its prompt is a cache hit." Those 19 calls, out of 292 in all single-agent runs, made the difference. No team call crossed the line; a worker's window peaked at about 45,000 tokens on average.

Repricing every call at Haiku's base rates removes the effect:

| Wide tasks, cost per task | Actual prices | Base rates only |
|---|---|---|
| Single agent | $0.0635 | $0.0161 |
| Lead with workers | $0.0243 | $0.0243 |

Without the price step the team costs 1.51 times as much, in line with its 1.38 times the tokens. Claude Sonnet 5.5 and the other current models bill their whole 1M-token window at one rate, so on them a team saves money only if it reads fewer tokens than one agent would.

### Results with small tool results

Part 5's "sized" tools return about 170 tokens per record instead of 2,150. With them, the single agent never came near the 100,000-token line:

| Design, sized tools | Correct (95% CI) | Tokens per task | Token multiple | Cost per task | Cost multiple | Median wall time |
|---|---|---|---|---|---|---|
| Single agent | 36/36, 100% (90-100) | 51,119 | 1.00 | $0.0035 | 1.00 | 23.6 s |
| Lead with workers | 35/36, 97% (86-100) | 85,013 | 1.66 | $0.0074 | 2.10 | 36.1 s |

On wide tasks the multiple was 2.08 in tokens and 2.38 in cost. Each worker pays for its own system prompt, tool definitions and brief, writes its own report, and the lead reads all the reports; when the records themselves are small, that overhead is a large share of the bill. These multiples sit below the 3-to-10-times range Anthropic quotes, which comes from different and larger tasks.

## Step 3: where each design breaks

All failures below come from the transcripts, `claude-haiku-5-5` at effort `medium`.

### The brief rules out a shortcut

Two wide tasks asked which service had the most SEV1 or SEV2 incidents. Severity is in each record's header line, which the search tool returns, so the single agent answered from four searches without opening a record: about 5,600 tokens and $0.001, correct in all four runs.

The lead's briefs added caution the question never asked for: "Check the opened/created date, not the resolved or updated date," "If the tracker has duplicates or merged/linked records, count each incident once and note this," "Do not guess; if you cannot verify a record, list it separately." Workers read that as an order to verify, and in three of the four team runs they opened 27 to 33 records each run. The team averaged 261,000 tokens on one question and 139,000 on the other, 46 and 23 times the single agent. The lead wrote the plan before anyone had seen the data, and the worker who saw it followed the plan.

### The lead "corrects" a right answer

On chain `c05` with sized tools, the worker followed all 20 records and reported "Total robots affected across the chain: 41,715," with every value listed. That was right. The lead, told to check reports, re-added the list and wrote:

```text
The worker's per-incident counts add up to **42,083**, not the 41,715 it reported. I re-added its figures
by hand and got the same 42,083 both times, so I'm using that total.
```

It was the only wrong answer in 144 main runs. The check the prompt asked for added an arithmetic step, and that step failed: a sum that two models recompute has two chances to go wrong.

### Second passes nobody needed

Duplicated work was rare: across 72 team runs, workers read 1.03 records per unique record on average. The two exceptions were deliberate. In one two-stage run a worker noted that its searches "returned the full set of matches (68 billing-api, 67 dock-controller)," all months included, beside its in-month counts of 18 and 17. The lead sent a second worker to redo everything: "A previous pass gave inconsistent counts, so please verify carefully and completely." Both passes agreed; the run read 70 records for 35 unique ones and took 133 seconds instead of about 27. In a sized wide run the lead gave all four services to one worker, then sent a second to verify the list: 110 reads for 55 records. Anthropic's post reports the same family, from subagents that "duplicated work" to early versions "spawning 50 subagents for simple queries."

### Parallel workers, slower answers

The team was slower on every kind of task: 33.5 seconds median against 22.0 with full records. Its model calls added up to 53 seconds per run, so parallelism hid about a third of that time, but the single agent needed only 19 seconds of model time because it already ran its lookups in parallel, as a batch of tool calls in one step ([Part 2](/blog/tool-calling-internals/) measured that saving). The team pays for the lead's first call before any worker starts and its last call after all finish, and here at most three of four workers ran at once. Parallel sub-agents can only win when each subtask needs many model calls in sequence; when one batch of tool calls would do, they add time.

### A harder set: 12 services at once

Sized tools let one window hold far more records, so a harder set asked eight questions that each compare 12 of the 16 services in one month: 153 to 212 records per question, two repeats per design, same settings.

| Design, 12 services, sized tools | Correct (95% CI) | Tokens per task | Model calls per run | Peak window | Cost per task | Median wall time |
|---|---|---|---|---|---|---|
| Single agent | 16/16, 100% (81-100) | 319,463 | 10.7 | 62,803 | $0.0192 | 74.8 s |
| Lead with workers | 16/16, 100% (81-100) | 352,722 | 32.1 | 32,046 | $0.0288 | 76.1 s |

Two hundred records in one window did not break Haiku 5.5 on lookups like these. The team cost 1.50 times as much for 1.10 times the tokens, finished in about the same time, and split the work differently from run to run, with anywhere from 2 to 13 workers. Duplicate reads rose to 1.15 per unique record.

### A stronger lead

The 10 wide tasks also ran once each with `claude-sonnet-5-5` as the lead (effort `medium`, adaptive thinking) and Haiku workers, on full records. All 10 were correct (95% CI 72-100), at $0.0509 per task. The Sonnet lead read fewer tokens than the Haiku lead (5,109 per run against 8,141) but cost ten times as much ($0.0153 against $0.0015).

On 2 of the 10 tasks Sonnet gave all four services to a single worker. That worker's window crossed 100,000 tokens (137,549 and 141,422 at peak), and those two runs cost $0.094 and $0.107, more than the single Haiku agent. On the other eight it cost $0.0385 per task against $0.0247 for the Haiku-led team. With accuracy already at 100%, the stronger lead had nothing to add here; Anthropic's 90.2% gain came from open-ended research. One run per task is an example, not a measurement.

## Step 4: agents talking across a network with A2A

So far the workers ran in the same Python process as the lead. When the worker is someone else's agent, on another server and maybe another framework, the handoff needs a protocol.

### What A2A is

The [Agent2Agent (A2A) Protocol](https://a2a-protocol.org/latest/specification/) is "an open standard designed to facilitate communication and interoperability between independent, potentially opaque AI agent systems." The specification site lists 1.0.0 as the latest release; the [GitHub releases](https://github.com/a2aproject/A2A/releases) date 1.0.0 to March 12, 2026, with a 1.0.1 fix release on May 26, 2026. Its "Opaque Execution" principle is the brief-and-report boundary from Step 1, made formal: agents collaborate "without needing to share their internal thoughts, plans, or tool implementations."

An agent describes itself in an Agent Card, a JSON document at `/.well-known/agent-card.json` listing its skills, endpoints and security schemes. A client sends a Message made of Parts (text, a file or structured data). The agent answers with a Task, the unit of work, which has an ID and a state (submitted, working, completed, failed, canceled, input required, rejected or auth required), and puts its results in Artifacts.

Operations such as `SendMessage`, `SendStreamingMessage`, `GetTask` and `CancelTask` map onto three bindings: JSON-RPC 2.0 over HTTP, gRPC, and HTTP+JSON. Streaming uses Server-Sent Events, long tasks can push status to a URL the client registers, and clients send an `A2A-Version` header with each request.

A2A and MCP solve different problems. The spec's Appendix B says MCP "focuses on standardizing how AI models and agents connect to and interact with tools, APIs, data sources," while A2A is about "how independent, often opaque, AI agents communicate and collaborate with each other as peers." In this lab the worker is an A2A server to the lead and could reach its tools through MCP ([Part 6](/blog/model-context-protocol/)).

### The worker as an A2A server

The official Python SDK (`a2a-sdk` 1.2.3) wraps the same worker loop. The executor turns an incoming message into a task, runs the loop, and returns the report as an artifact:

```python
class WorkerExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        await updater.update_status(state=TaskState.TASK_STATE_WORKING)
        run = await asyncio.to_thread(loop, get_message_text(context.message), TOOLS, WORKER, part="a07",
                                      role="a2a-worker", client=CL, max_steps=30, max_cost=0.06, run=Run("worker"))
        await updater.add_artifact(parts=[new_text_part(text=run.answer, media_type="text/plain")], name="report")
        await updater.update_status(state=TaskState.TASK_STATE_COMPLETED if run.stop == "end_turn"
                                    else TaskState.TASK_STATE_FAILED)
```

Its Agent Card, fetched with `curl`:

```json
{"name":"Larkspur incident analyst","description":"Answers one self-contained question about Larkspur Robotics' incident tracker.","supportedInterfaces":[{"url":"http://127.0.0.1:9999","protocolBinding":"JSONRPC","protocolVersion":"1.0"}],"version":"1.0.0","capabilities":{"streaming":false},"defaultInputModes":["text/plain"],"defaultOutputModes":["text/plain"],"skills":[{"id":"incident-lookup","name":"Incident lookup","description":"Search incidents by service and month, read records, compute totals.","tags":["incidents"],"examples":["Total customer impact minutes on map-tiles in August 2026"]}]}
```

On the lead's side, `delegate` becomes an A2A client call: build a `SendMessageRequest` from the brief, send it, and read the text parts of the returned task's artifacts.

### On the wire

A logging hook on the client's HTTP connection recorded every exchange. The client fetched the Agent Card, then posted the brief as a `SendMessage` call with the header `A2A-Version: 1.0` (IDs and text shortened):

```text
client→server POST / {"method":"SendMessage","params":{"message":{"messageId":"d1c87864-...","role":"ROLE_USER","parts":[{"text":"You have access to Larkspur Robotics' incident tracker through search and read tools. Task: find every incident for the service \"charge-scheduler\" that was OPENED in June 2026 [...]"}]},"configuration":{}},"id":"26f715aa-...","jsonrpc":"2.0"}
server→client 200 {"result":{"task":{"id":"f1966411-...","contextId":"ea785f0e-...","status":{"state":"TASK_STATE_COMPLETED","timestamp":"2026-10-25T09:09:39.734207Z"},"artifacts":[{"artifactId":"006ec06d-...","name":"report","parts":[{"text":"SERVICE: charge-scheduler\nMATCHING INCIDENT IDS: INC-7832 (3), INC-6575 (139), [...]\nTOTAL CUSTOMER IMPACT MINUTES: 1271\nNOTES: [...]","mediaType":"text/plain"}]}],"history":[{"messageId":"d1c87864-...","role":"ROLE_USER","parts":[{"text":"[the brief again]"}]}]}},"id":"26f715aa-...","jsonrpc":"2.0"}
```

The text part is just `{"text": ...}`: version 1.0 dropped the `"kind": "text"` field that 0.3 used. Without streaming, the lead sees only the finished task, never the `WORKING` state. And the task's `history` repeats the brief, so each round trip carried it twice: requests were 1,105 to 1,606 bytes and responses 2,356 to 3,489.

Errors follow JSON-RPC with A2A codes in the -32001 to -32099 range. Asking for a task that doesn't exist returns `-32001` ("Task not found"), and a request with `A2A-Version: 2.0` gets `-32009`, "A2A version '2.0' is not supported by this handler. Expected version '1.0'." The 0.3-era method name `message/send` now gets the plain JSON-RPC `-32601` "Method not found", because 1.0 renamed methods to PascalCase.

With `delegate` swapped for the A2A client and the worker server using sized tools, the Haiku lead answered three tasks (two wide, one two-stage) correctly with 10 worker calls, each round trip taking 14.7 to 29.9 seconds. The runs cost $0.0093 per task, against $0.0097 for the same tasks in process. On one machine the protocol added bytes and left cost and accuracy where they were. It adds what a handoff between teams needs: discovery, declared security schemes, versioning, and a task that can be polled, streamed or canceled.

All experiments in this article cost $4.05 on the ledger: $2.20 for the full-record runs, $0.39 for the sized runs, $0.77 for the 12-service set, $0.51 for the Sonnet lead, $0.03 for A2A and $0.16 for pilots.

## In the real world

- Anthropic's [research system](https://www.anthropic.com/engineering/multi-agent-research-system) (Hadfield et al., June 13, 2025) runs a Claude Opus 4 lead with Claude Sonnet 4 subagents. The lead "spins up 3-5 subagents in parallel," which with parallel tool calls "cut research time by up to 90% for complex queries," and the prompts scale effort: "Simple fact-finding requires just 1 agent with 3-10 tool calls." Subagents write output to a filesystem to avoid a "game of telephone." These are older models on other tasks, not measured here.
- Claude's [Building multi-agent systems: when and how to use them](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) (Cara Phillips, January 23, 2026) names three cases where several agents pay off (context protection, parallel work and specialization); "outside these situations, the coordination costs typically exceed the benefits." Its advice to take a "context-centric view rather than a problem-centric view when decomposing work" matches the lead's choice to keep chains in one worker.
- Cognition's [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) (Walden Yan, June 12, 2025) argues "Share context, and share full agent traces, not just individual messages" and "Actions carry implicit decisions, and conflicting decisions carry bad results," and prefers a single-threaded agent by default. Briefs written before anyone saw the data are a small case of the first.
- The Claude Platform's [multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) for Managed Agents (beta header `managed-agents-2026-04-01`) runs each agent in its own "context-isolated event stream" sharing one sandbox; delegation "is one level deep," with at most 25 child threads per session and up to 20 listed agents. This article doesn't use it.
- [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) (Cemri et al., 2025) annotated over 1,600 traces from 7 multi-agent frameworks into 14 failure modes in three groups: system design issues, inter-agent misalignment and task verification. The lead's bad re-check here is a verification failure.

## Common misconceptions

- **"Multi-agent systems always cost more."** They read more tokens (1.15 to 2.08 times here), but on Haiku 5.5 the team cost half as much on full records because no worker's prompt crossed the 100,000-token price line. On flat pricing, or with small tool results, it cost 1.5 to 2.4 times as much.
- **"More agents means a better answer."** On these tasks both designs scored 100% on full records, and the team's only wrong answer in 144 main runs came from its own lead re-checking a correct report.
- **"Parallel workers make the answer arrive sooner."** On the main tasks the team was 11 to 13 seconds slower in median, because one agent can already run many lookups as parallel tool calls, and the lead adds a call before and after the workers.
- **"Splitting a task is free context management."** Each brief fixes a plan before anyone has seen the data. Briefs that asked workers to verify dates and duplicates cost 23 to 46 times the single agent's tokens on questions answerable from search results alone.
- **"A2A replaces MCP."** A2A connects agents as opaque peers through tasks and artifacts; MCP connects one agent to tools and data. A worker can be an A2A server and an MCP client at once.

## Key takeaways

- An orchestrator-workers system is one tool that runs another agent loop. The brief down and the report up are the whole interface, and every failure measured here happened there.
- On splittable work the team read 1.38 times the tokens of one agent. Whether that costs more depends on pricing: on Haiku 5.5 it cost 62% less because it stayed under the long-prompt line, and at flat rates it would have cost 51% more.
- Keep unsplittable work in one agent. The lead did, and paid only 1% to 6% for the wrapper.
- Watch the handoff for the failures measured here: briefs that forbid shortcuts, leads that redo arithmetic, and verification passes that double the work.
- A2A 1.0 standardizes the handoff across processes and vendors: an Agent Card for discovery, `SendMessage` for the brief, a task with a state, and an artifact for the report.

## What's next

Anthropic's research team notes that "most coding tasks involve fewer truly parallelizable tasks than research." [Part 8](/blog/coding-agents/) turns to coding agents: how SWE-bench-style tasks get solved with repository search, edit tools and test loops, measured on a mini-benchmark built from real open-source repositories.

## Further reading

- [A2A Protocol Specification, version 1.0](https://a2a-protocol.org/latest/specification/): the data model, operations, JSON-RPC, gRPC and HTTP bindings, error codes and the relationship to MCP.
- [A2A Python SDK](https://github.com/a2aproject/a2a-python) and [samples](https://github.com/a2aproject/a2a-samples): the server and client used in Step 4.
- [How we built our multi-agent research system (Hadfield et al., Anthropic, 2025)](https://www.anthropic.com/engineering/multi-agent-research-system): the lead-and-subagents design, token use, parallelism and prompting lessons.
- [Building effective agents (Anthropic, 2024)](https://www.anthropic.com/engineering/building-effective-agents): workflows versus agents, and the orchestrator-workers and parallelization patterns.
- [Building multi-agent systems: when and how to use them (Phillips, Anthropic, 2026)](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them): when splitting pays, and context-centric decomposition.
- [Subagents (Claude Code docs)](https://code.claude.com/docs/en/sub-agents): a production sub-agent design with its own context, tools and model.
- [Multiagent orchestration (Claude Platform docs)](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration): server-hosted subagents, workflows and their limits.
- [Pricing (Claude API docs)](https://platform.claude.com/docs/en/about-claude/pricing): Haiku 5.5's prices over 100,000 prompt tokens, the effect behind Step 2's cost result.
- [Why Do Multi-Agent LLM Systems Fail? (Cemri et al., 2025)](https://arxiv.org/abs/2503.13657): a taxonomy of 14 failure modes from 1,600+ annotated traces.
- [Don't Build Multi-Agents (Yan, Cognition, 2025)](https://cognition.ai/blog/dont-build-multi-agents): the case for one context-sharing agent.
