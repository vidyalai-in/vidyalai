---
title: "Tracing Every Step of a Multi-Agent Workflow"
description: "Model multi-agent runs as one OpenTelemetry trace: GenAI span conventions, context propagation across queues, private content storage and tail sampling."
slug: "tracing-multi-agent-workflows"
category: "production-engineering"
tags: ["observability", "agents", "tool-use", "privacy", "evals"]
seriesOrder: 23
pubDate: 2026-10-16
sourceDoc: "15jgtoXCDflquvlaEjUw7LYBq2BmcoregHbUPWEtQCxw"
---

## TL;DR

Make each user task a single distributed trace using OpenTelemetry (OTel) and its GenAI semantic conventions. Every agent invocation, LLM call, tool execution, retrieval and handoff becomes a span, joined by W3C trace context that you propagate explicitly across queues and async boundaries. Spans carry small, structured metadata: model, prompt version, token usage, cost, latency, finish reason and tool status. Full prompts and completions go to a separate, access-controlled blob store referenced from the span. Use tail-based sampling so every error, expensive run and flagged run is kept while normal traffic is sampled. The traces are there to answer three questions quickly: why did this run fail, what changed, and where do the cost and latency go.

## Requirements to pin down first

- **What is the topology?** Orchestrator-worker, peer-to-peer handoffs, or an explicit graph (LangGraph-style)? Synchronous calls or queue-based fan-out? Do runs last seconds or hours? Queues and long runs are where trace context gets lost, so they shape the propagation design.
- **What is the scale?** Runs per day and spans per run drive storage cost and the sampling strategy.
- **How sensitive is the data?** Is PII or customer data in prompts? What retention and residency rules apply? Who may view raw content? The answers decide whether content can ever sit in the tracing backend.
- **Who consumes the traces?** On-call engineers debugging, eval and quality teams, finance (cost attribution), or safety (audit trails). Each needs different retention, fidelity and access.

## The span model

![Example span tree for one multi-agent research run. The root span invoke_workflow research_task has one child, invoke_agent orchestrator. The orchestrator has a chat span on a large model for planning, two researcher agents and a synthesizer agent. Researcher 1 has a chat span on a small model, an execute_tool web_search span with a child HTTP GET span, and a retrieval span on a docs index. Researcher 2 has a chat span and an execute_tool code_exec span. The synthesizer's span carries span links to both researcher spans, and has its own chat span on the large model for the final answer.](/diagrams/q23-span-tree.svg "One run, one trace: parent-child spans for delegation, span links for fan-in.")

Parent-child edges show who delegated to whom; span links show which parallel results the synthesizer consumed.

The OTel GenAI conventions define operation names (`chat`, `invoke_agent`, `invoke_workflow`, `execute_tool`, `retrieval`, `embeddings`, `create_agent`) and span names of the form `{operation} {model or agent or tool name}`. They are still marked "Development" stability, so pin the convention version your instrumentation targets.

### Root span

`invoke_workflow <name>`: task ID, hashed user and tenant IDs, entrypoint, workflow version. (Teams that predate the `invoke_workflow` operation often use a custom `workflow.run` span; either works if it is consistent.)

### Agent spans

`invoke_agent <name>`: `gen_ai.agent.name`, `gen_ai.agent.id`, `gen_ai.agent.version`, the agent's role, its prompt template name and version, and its max-steps budget.

### LLM spans

`chat <model>`, with the standard attributes:

- `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`
- `gen_ai.response.finish_reasons`, `gen_ai.request.temperature`
- `gen_ai.response.time_to_first_chunk` for streaming calls
- `gen_ai.prompt.name` and `gen_ai.prompt.version` for the template that produced the prompt
- your derived cost, as a custom attribute

### Tool spans

`execute_tool <name>`: `gen_ai.tool.name`, `gen_ai.tool.call.id`, tool version, a hash of the arguments, status, retry count and a side-effecting flag. Downstream HTTP or database calls appear as child spans through normal auto-instrumentation.

### Retrieval spans

`retrieval <index>`: index version, top-k, and the returned document IDs and scores. The conventions mark query text and retrieved documents as opt-in, because they are content.

### Handoffs and fan-in

Use parent-child spans for delegation. Use **span links** when an aggregator consumes outputs from N parallel workers, or when a resumed run continues an earlier trace.

### Events on spans

Record guardrail verdicts, planner decisions ("chose tool X"), loop-detection triggers and human-approval waits as span events, so they line up on the timeline.

## Context propagation (where most implementations break)

- **In-process.** OTel stores the active context in context variables (`contextvars` in Python). Thread pools and some async task-group patterns drop context unless you wrap the submitted callables to copy it.
- **Across services.** Send `traceparent` and `tracestate` in HTTP or gRPC headers (W3C Trace Context). Auto-instrumentation handles this for common clients.
- **Across queues.** Inject the context into message headers (Kafka, SQS, Pub/Sub) on publish and extract it in the consumer. Nothing does this for a hand-rolled job table, so add it yourself.
- **Long-running and human-in-the-loop runs.** Persist the trace context in the workflow state store (Temporal or another durable executor), so a run resumed hours later stays attached. If the trace would otherwise grow without bound, start a new trace and link it to the old one.
- **Correlation everywhere.** Put the `trace_id` into every log line and return it in the API response (for example as `x-request-id`), so a support ticket maps straight to a trace.

## Content capture and privacy

- **Keep span attributes small and structured.** Store full prompts and completions in an object store, encrypted and keyed by span ID, and put a pointer attribute on the span. The OTel GenAI conventions recommend exactly this pattern for production: content capture is off by default, recording content on spans is suited to pre-production, and external storage with references is the recommended production option because it enables separate access controls. It also keeps the tracing backend cheap and lets content have stricter ACLs and shorter retention (for example 30 days for content, 13 months for metadata).
- **Redact or tokenize PII in the SDK before export.** Never log secrets from tool arguments; allowlist the argument fields you record.
- **Record enough to replay.** Store the exact rendered prompt, model snapshot, sampling parameters and seed, and tool results. A failing run can then be re-executed against a new prompt or model, which turns production traces into eval cases. (LLM sampling is not perfectly deterministic even with a seed, so replay compares behavior rather than bytes.)

## Volume math and sampling

### How much data

For example, at 2M runs per day:

```text
Span metadata: 2M runs × 40 spans × 1.5 KB         ≈ 120 GB/day
Content:       2M runs × 15 LLM spans × 15 KB       ≈ 450 GB/day
```

Content is almost 4x the metadata, which is another reason to keep it out of the tracing backend.

### Head vs tail sampling

**Head sampling** decides at the root span, before anything has happened. It is cheap, but it throws away the rare failures you most need.

**Tail sampling** in the OTel Collector decides after the trace completes. All spans of a trace must reach the same collector instance, so put a trace-ID-aware load-balancing exporter in front of the sampling tier. A typical policy:

- keep 100% of errors, guardrail trips, runs over p95 cost or latency, and runs with negative user feedback;
- keep 100% of runs on a new prompt or model version during its rollout;
- keep 1-5% of everything else.

### Metrics before sampling

Emit unsampled metrics from spans with a span-metrics connector: RED metrics (rate, errors, duration), tokens, cost per agent and steps per run. Aggregates then stay accurate whatever the sampling rate. Keep high-cardinality fields (user ID, prompt text) out of metric labels.

## Making traces useful

- **Derived signals:** cost per successful task, the distribution of steps to completion, tool error rate by tool version, re-plan and loop rate (for example, the same tool with the same arguments more than 3 times), and agent-to-agent token amplification. Anthropic reports its multi-agent research system uses about 15x more tokens than a chat interaction, so a fan-out that multiplies tokens by an order of magnitude is normal and needs to be visible per agent.
- **Online evals attached to traces.** Write LLM-judge or heuristic scores and user feedback back as span attributes or linked records (the conventions now include `gen_ai.evaluation.*` attributes). You can then filter for "runs with judge score below 3 after prompt v42".
- **Diffing.** Compare the distribution of trace shapes before and after a deploy. Average tool calls per run jumping from 6 to 11 is a regression signal even if no request failed.
- **Version stamps on every span.** Prompt, model, tool and workflow versions turn "what changed?" into a query instead of an investigation. The same stamps make prompt rollbacks verifiable (see [Rolling Back a Bad Prompt Change in Production](/blog/prompt-rollback-production/)).

## How to measure it

- Mean time to root cause for agent incidents, for example from hours to under 15 minutes.
- Replayability: the percentage of production failures that can be converted into replayable eval cases.
- Trace completeness: the share of traces with no orphan spans or broken context. Target above 99%, and enforce it with a CI test that runs a sample workflow (including a queue hop and a parallel fan-out) and asserts the expected span tree.
- Overhead: tracing adds under 1-2% to latency, with a known cost per million spans.
- Sampling coverage: the share of error runs retained (should be 100%) and the share of total traffic stored.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| OTel + GenAI conventions to a general backend (Grafana Tempo, Honeycomb, Datadog) | Vendor-neutral; joins LLM spans with infrastructure spans; one pipeline | GenAI UX (prompt diffing, eval views) is thinner | Platform teams; agents embedded in larger systems |
| LLM-specific platforms (Langfuse, LangSmith, Arize Phoenix) | Prompt and completion views, eval and dataset loops built in | Another silo; may miss non-LLM spans; data residency | Fast-moving agent teams; most can still ingest OTel |
| Head sampling | Simple, cheap, predictable | Drops rare failures | Very high volume, uniform traffic |
| Tail sampling | Keeps errors and outliers | Collector state and memory; needs trace-aware load balancing | Default for agent workloads |
| Inline content in spans | Easy debugging | Cost, PII exposure, attribute size limits | Development and staging only |
| Content in a separate blob store | Cheap, ACL-able, separate retention | Extra hop; pointers can dangle | Production |

- Fidelity vs privacy and cost: capture structure for everything, content for a sampled or flagged subset, and full replay data for failures.
- Instrument the framework, not each agent. Auto-instrument the LLM client, the tool registry and the handoff primitive, so coverage does not depend on every developer remembering to add spans.

## Common mistakes

- **Broken context propagation** across async code, thread pools and queues. The result is orphan spans and fragmented traces, exactly where multi-agent bugs live.
- **Head-based sampling** that discards the 0.5% of failing runs you needed to see.
- **Logging raw prompts with PII or secrets** into a broadly accessible observability tool, which turns a debugging aid into a compliance incident.
- **Cardinality explosions** from putting user IDs, prompt hashes or model outputs into metric labels, which blows up metric storage and cost.
- **Streaming spans that end at the first token.** Latency and token counts come out wrong. A span must close at stream end and record TTFT as an attribute or event.
- **No version stamps.** Without them you cannot correlate a regression with a prompt, model or tool change, and tracing becomes a picture rather than a debugging tool.

## Further reading

- [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md): the standard attribute and span names for LLM operations, now maintained in their own repository.
- [OTel GenAI agent and framework spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md): conventions for `invoke_agent`, `invoke_workflow`, `create_agent` and `execute_tool`.
- [OpenTelemetry: Sampling](https://opentelemetry.io/docs/concepts/sampling/): head vs tail sampling tradeoffs.
- [OTel Collector tail sampling processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/tailsamplingprocessor) and [span metrics connector](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/connector/spanmetricsconnector): the components behind the sampling and unsampled-metrics design.
- [W3C Trace Context](https://www.w3.org/TR/trace-context/): the `traceparent` and `tracestate` propagation standard.
- [Dapper, a Large-Scale Distributed Systems Tracing Infrastructure (Sigelman et al., 2010)](https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/): Google's foundational tracing paper (sampling, low overhead, annotations).
- [How we built our multi-agent research system (Anthropic, 2025)](https://www.anthropic.com/engineering/multi-agent-research-system): production lessons on tracing, token amplification (about 15x chat) and debugging non-deterministic agents.
- [Building effective agents (Anthropic, 2024)](https://www.anthropic.com/engineering/building-effective-agents): agent patterns such as orchestrator-workers that shape the span topology.
