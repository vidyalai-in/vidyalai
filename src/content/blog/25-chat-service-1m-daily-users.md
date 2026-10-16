---
title: "Designing a ChatGPT-Style Chat Service for 1M Daily Users"
description: "Back-of-envelope sizing, architecture and overload strategy for an LLM chat product at 1M DAU, where GPUs set both the cost and the latency."
slug: "chat-service-1m-daily-users"
category: "production-engineering"
tags: ["architecture", "serving", "cost", "streaming", "caching", "routing"]
seriesOrder: 25
pubDate: 2026-10-16
sourceDoc: "15jgtoXCDflquvlaEjUw7LYBq2BmcoregHbUPWEtQCxw"
---

## TL;DR

At 1M daily active users (DAU) sending about 20 messages each, the traffic is 20M messages a day: about 230 messages per second on average, around 700-750 per second at peak, and roughly 9,000 concurrent streams. That is modest for the web tier and substantial for GPUs, so the design centers on the inference layer and its cost. The architecture is a stateless API and SSE streaming gateway, a conversation service on a sharded store, an orchestration layer that builds context (history truncation or summarization, retrieval, tools, safety), and a GPU fleet with continuous batching, prefix caching and conversation-affinity routing. The concerns that decide success are cost per DAU, p99 TTFT (time to first token) under load spikes, graceful degradation (model fallback, queueing, rate limits), and safety and abuse controls; those matter far more than the CRUD.

## Requirements to pin down first

- **Which model, hosted where?** A self-hosted open-weight model or a vendor API? One model, or a tiered router that sends simple queries to a small, fast model and hard ones to a frontier model? This decides whether you run a GPU fleet at all and how much of the cost you can control.
- **Which features?** Multimodal input, file uploads, web search, tools, memory, share links? Each one adds a subsystem with its own storage, safety and cost profile.
- **What are the SLOs and the economics?** For example: TTFT p95 under 1 s, streaming at 30 tok/s or more, 99.9% availability. What is the cost budget per DAU, and is there a free/paid split? The split drives rate limits, model tiers and the order in which you shed load.
- **Where are the users, and what are the rules?** Regions, data residency, retention, training-data opt-out, and abuse and CSAM (child sexual abuse material) reporting obligations. These constrain where data and GPUs can live.

## Back-of-envelope sizing

### Messages and streams

```text
Messages:   1M DAU × 20 messages/day          = 20M messages/day
Average:    20M ÷ 86,400 s                    ≈ 230 messages/s
Peak:       3x diurnal peak                   ≈ 700 messages/s; design for ~750/s
Burst:      2x headroom on the web tier       ≈ 1,500 messages/s admission capacity
Streams:    750/s × ~12 s per stream          ≈ 9,000 concurrent SSE connections
```

The 12 s stream duration is about 400 tokens at 40 tok/s plus 1-2 s of TTFT. Nine thousand mostly idle connections fit on a handful of gateway pods (see [Streaming LLM Responses over SSE](/blog/sse-streaming-disconnects/) for the stream-log write rate this implies).

### Tokens

```text
Output:  20M × 400 tokens       = 8B output tokens/day
         8B ÷ 86,400 s          ≈ 93k tok/s average, ~280-300k tok/s at peak
Input:   20M × ~3K tokens       = 60B input tokens/day (multi-turn history)
         at peak                ≈ 700/s × 3K = 2.1M input tok/s before caching
```

With prefix caching of the system prompt and prior turns, **70-90% of input KV can be reused**, but only if each request is routed back to the replica that holds its conversation's cache. That leaves roughly 210k-630k input tok/s of real prefill at peak.

### Storage

```text
Text:    20M messages × ~2 KB    ≈ 40 GB/day ≈ 15 TB/year before replication
```

Attachments are separate and live in object storage.

### GPUs and cost

Assume a 70B-class model in FP8 on 4x H100 replicas, each producing about 5k output tok/s at the TPOT (time per output token) SLO, and about 22k prefill tok/s. The derivation of those per-replica numbers is in [Serving Open-Weight LLMs at Scale](/blog/serving-open-llms-at-scale/).

```text
Decode:    280k tok/s ÷ 5k per replica              ≈ 56 replicas
Prefill:   210k-630k tok/s ÷ 22k per replica        ≈ 10-29 replicas of compute,
           largely absorbed by chunked prefill alongside decode
Headroom:  +30% for failures, deploys, forecast error
Total:     ≈ 80 replicas × 4 GPUs                   ≈ 320 H100s
Cost:      320 × $2.50/GPU-hr × 24 h                ≈ $19,200/day
Per DAU:   $19,200 ÷ 1M                             ≈ $0.02 per DAU per day (~$0.58/month)
Per token: $19,200 ÷ 8M thousand output tokens      ≈ $0.0024 per 1K output tokens
```

The $2.50/GPU-hour figure is an assumption; rental prices vary widely. On Blackwell, the same model fits on 2x B200 replicas with similar per-replica throughput, so about half as many GPUs at a higher hourly price.

Routing 60% of traffic to an 8B-class model cuts inference cost about 2x: the large model keeps 40% of the load, and the small model serves the rest at a small fraction of the per-token cost. Inference typically dominates total cost (80-90% is a reasonable planning figure), so model routing and cache hit rate are the biggest levers you have.

## The architecture

![Architecture diagram. Web and mobile clients reach an edge layer with CDN, WAF, global load balancing and rate limits. The edge routes to a stateless chat API and to an SSE gateway. The chat API reads and writes the conversation store with a hot cache, and calls the orchestrator, which handles context, tools and safety. The orchestrator also reads the conversation store and calls an inference gateway that routes by conversation affinity to a GPU fleet with a large-model pool and a small-model pool. The inference gateway writes tokens to a stream log, which the SSE gateway tails. The orchestrator also feeds an async pipeline for metering, titles and evals.](/diagrams/q25-chat-architecture.svg "Chat at 1M DAU: a thin, stateless web tier in front of an inference tier that carries the cost.")

The web tier is a few dozen stateless pods; the inference gateway and GPU fleet are where design effort pays off.

### 1. Edge

CDN and WAF (web application firewall), then a global load balancer, then the regional API. Authentication, per-user and per-tier **token-bucket rate limits** (messages per minute plus tokens per day), and bot and abuse detection all happen here, before any GPU time is spent.

### 2. Chat API (stateless)

`POST /conversations/{id}/messages` with an idempotency key, then `GET .../stream` for SSE backed by a resumable stream log. The API also handles conversation CRUD, titles, search and share links.

### 3. Conversation store

- Messages are partitioned by `conversation_id` and clustered by time, in a wide-column store (Cassandra or ScyllaDB, DynamoDB) or sharded Postgres (for example with Vitess).
- A user index lists conversations by `user_id`.
- Messages are append-only, with branching: an edit or regenerate creates a new message with a `parent_message_id`, so a conversation is a tree.
- A Redis hot cache holds recent conversations.
- Search uses asynchronous indexing into OpenSearch or a vector store.

### 4. Orchestrator

Loads history, then manages context: keep the last N turns verbatim plus a rolling summary once the conversation exceeds its budget (for example, a 32K-token budget against a 128K model limit, for cost). It also calls retrieval and memory, tools or web search, and a model router that classifies difficulty and tier. Input and output **safety classifiers** run in parallel with generation; output moderation runs over streamed chunks and can cut the stream.

### 5. Inference gateway

Picks the model pool and then the replica, using **conversation-affinity routing**: a consistent hash on `conversation_id` with bounded load, so each turn lands where the conversation's prefix cache lives without letting one hot replica overload. It also owns priority queues (paid over free, interactive over batch), request deadlines and cancellation propagation down to the engine.

### 6. Inference fleet

vLLM- or SGLang-class engines with continuous batching, chunked prefill and FP8 weights, in multiple regions. Autoscale on queue depth and KV-cache utilization, not CPU. Keep pre-warmed capacity, because loading weights onto a cold node takes minutes.

### 7. Async pipeline

Usage metering, title generation and summarization (batch, low priority, on off-peak GPUs), analytics, feedback, eval sampling, and training-data pipelines that respect opt-outs.

### 8. Observability and change management

One trace per message with prompt and model versions on every request (see [Tracing Every Step of a Multi-Agent Workflow](/blog/tracing-multi-agent-workflows/)), and rollouts and rollbacks through a prompt and model registry (see [Rolling Back a Bad Prompt Change in Production](/blog/prompt-rollback-production/)).

## Degradation and reliability

### The overload ladder

When demand exceeds capacity, step down in this order:

1. Queue with a deadline.
2. Shed the free tier first.
3. Route to a smaller model.
4. Cap max output tokens.
5. Return "at capacity" with a `Retry-After` header.

Never let queues grow without bound. Once they do, TTFT collapses for everyone, clients time out and retry, and the retries amplify the load.

### Regions and failure isolation

- **Multi-region active-active** for the API. GPU pools can spill over across regions, with a latency penalty. The conversation store is replicated, with a home region per user.
- **Bad replicas** (NaN outputs, a stuck NCCL collective) are detected by health probes plus output sanity checks, and drained automatically.

## How to measure it

- **User-facing:** TTFT at p50, p95 and p99; tokens per second; error and "stopped or truncated" rate; regenerate and thumbs-down rate; D1 and D7 retention.
- **Efficiency:** GPU-hours per 1K messages, prefix-cache hit rate, goodput (requests per second that meet the SLO), cost per DAU, and router accuracy (quality held steady while the small model's share of traffic rises).
- **Safety:** moderation precision and recall, abuse rate, and time to mitigate a jailbreak wave.
- **Capacity:** peak-to-average ratio, queue depth and shed rate per tier, and headroom remaining at the daily peak.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Vendor model API | No GPU operations; frontier quality; fast launch | Per-token cost at scale; rate limits; less control | Early stage, roughly under $1M/year of inference, or when model quality is the moat |
| Self-hosted open model | 3-10x cheaper at steady high volume; control over latency and data | GPU capacity planning, on-call, possible quality gap | Sustained volume, predictable load, data constraints |
| Single model | Simple, consistent behavior | Overpays on easy queries | Small scale, premium product |
| Tiered model router | 2-4x cost reduction | Misroutes hurt quality; harder to evaluate | 1M+ DAU with a mixed-difficulty workload |
| Wide-column store (Cassandra/ScyllaDB, DynamoDB) | Linear write scaling; simple partitioning | Weak ad-hoc queries; needs a separate search index | Huge append-only message volume |
| Sharded Postgres / Vitess | Transactions, flexible queries, familiar | Resharding operations | Moderate scale, richer relational features |

- **Context length vs cost.** Every extra 1K tokens of history is paid again on every turn, so total prefill over a conversation grows quadratically with its length unless the prefix is cached. Summarization and cache affinity are core design elements, not optimizations.
- **Safety checks in parallel vs in series.** Running them in parallel with generation saves about 100-300 ms of TTFT, but you may stream tokens you then have to retract.

## Common mistakes

- **Spending the design on CRUD and hand-waving inference.** The GPUs carry most of the cost and set the latency, so a design that skips them cannot be defended.
- **No back-of-envelope.** Without one you cannot defend capacity, cost per DAU or the number of GPUs.
- **Ignoring multi-turn context growth and prefix caching.** Random load balancing destroys the cache hit rate and can double prefill cost.
- **No overload strategy.** A launch or viral spike produces unbounded queues and timeouts, and retries amplify the load into an outage.
- **Forgetting abuse.** Scripted free-tier scraping, jailbreak waves and prompt injection through uploads need rate limits, classifiers and per-user token budgets, or they consume the free tier's GPU budget.
- **A weak data lifecycle.** Deletion requests must cascade to caches, search indexes, stream logs, backups and training sets, and training opt-outs must be respected, or you have a compliance problem.

## Thinking at scale

- **Make cost per DAU a first-class metric.** It ties model choice, routing share, cache hit rate and context budget to one number the product and finance teams both understand.
- **Plan the build-vs-buy migration.** Launching on a vendor API and moving the high-volume, easier traffic to a self-hosted tier later is a common path. Design the inference gateway as the seam from day one, so the switch is a routing change rather than a rewrite.
- **Own the data lifecycle centrally.** Deletion, retention and training opt-out touch every store in the diagram. One team should own the cascade and test it end to end.

## Further reading

- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180): the serving-engine foundation for the GPU tier.
- [SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2023)](https://arxiv.org/abs/2312.07104): RadixAttention prefix-cache reuse across multi-turn chat, the key cost lever.
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (Qin et al., 2024)](https://arxiv.org/abs/2407.00079): how Kimi serves chat at scale with KV-cache-centric scheduling and overload handling.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (Zhong et al., 2024)](https://arxiv.org/abs/2401.09670): goodput-oriented capacity planning under TTFT and TPOT SLOs.
- [The Tail at Scale (Dean and Barroso, 2013)](https://research.google/pubs/the-tail-at-scale/): tail-latency techniques for fan-out services.
- [How Discord Stores Trillions of Messages (Discord, 2023)](https://discord.com/blog/how-discord-stores-trillions-of-messages): message-store partitioning and hot-partition lessons that apply directly to chat history.
- [Prompt caching (Claude API docs, Anthropic)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): API-level prefix caching economics if you use a vendor model.
- [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events): the streaming transport for the chat UI.
