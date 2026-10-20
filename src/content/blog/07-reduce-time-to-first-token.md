---
title: "Reducing Time to First Token in a Chat App"
description: "Decompose TTFT into network, pre-model work, queueing and prefill, then cut each stage: parallelism, prefix caching, chunked prefill and headroom."
slug: "reduce-time-to-first-token"
category: "cost-latency-serving"
tags: ["latency", "serving", "caching", "gpu", "streaming"]
seriesOrder: 7
pubDate: 2026-10-16
sourceDoc: "1wx9vLMMIykYpYxwlDoHQz8GunZXg7raHlJEYPtY-TAI"
---

## TL;DR

TTFT (time to first token) is the sum of network time, queueing, prefill (the forward pass over the whole prompt), and everything that runs before the model is even called: auth, retrieval, guardrails. Start by decomposing p50 and p95 TTFT into those stages, then attack the largest one. Usually that means running pre-model work in parallel and streaming from the edge, using prefix (KV) caching so a long system prompt and conversation history are not re-prefilled every turn, keeping prefill from being starved by decode (chunked prefill or disaggregation), and tightening admission control so requests do not sit in a queue. A typical result is p95 TTFT dropping from 2-3 s to 300-600 ms without changing the model.

## Requirements to pin down first

- **Which percentile matters: p50, or p95/p99?** Tail TTFT is usually driven by queueing and long prompts, not by average prefill speed. Optimizing the median and the tail lead to different fixes.
- **Hosted API or your own serving stack?** On an API, your controls are prompt shape, caching, region, model choice and the pipeline around the call. Self-hosted, the scheduler and hardware are also in scope.
- **How long are typical prompts?** For a 70B-parameter model on a tensor-parallel group of H100 GPUs, a 500-token prompt prefills in tens of milliseconds, while 30k tokens of history plus retrieved context takes seconds (worked out below).
- **What must happen before the first token?** Moderation, retrieval, tool planning? And does the first visible token have to be the first token of the final answer, or can it be a status message?

## Step 1: Measure the waterfall

Trace every request with spans: client, edge, auth, retrieval, prompt build, queue wait, prefill, first token, network back. Report TTFT *as the user perceives it*, from client send to first rendered token, not only the server-side number.

![A top-to-bottom flow of one chat request in three groups: network (client send, then edge and TLS) at 100-200 ms; pre-model work run in series (auth, embedding and vector search, rerank, moderation) at 300-800 ms; and the inference server (queue wait at 0-1500 ms, prefill at 200-2000 ms, then the first token streams back to the client).](/diagrams/q07-ttft-waterfall.svg "Where TTFT goes in a typical unoptimized chat request at p95 (illustrative budgets).")

Every stage before the first token adds to TTFT in series, so the biggest box is where to start.

An illustrative before-and-after budget at p95:

| Stage | Unoptimized p95 | After fixes | Main fix |
|---|---|---|---|
| Network / TLS | 100-200 ms (cross-region) | 50-100 ms | Regional endpoint, warm connection |
| Pre-model work | 300-800 ms (embedding, vector search, rerank, moderation in sequence) | 50-150 ms | Parallelize, skip when not needed |
| Queue wait | 0-1,500 ms (bursty load, no headroom) | under 100 ms | Headroom, admission control, priorities |
| Prefill | 200-2,000 ms (scales with prompt length) | 50-250 ms | Prefix caching, shorter prompts |
| **Total** | **about 2-3 s** | **about 300-600 ms** | |

## Step 2: Remove pre-model latency

- Parallelize. Run moderation alongside retrieval, or alongside generation with output gating (hold back the stream until the check passes). Run retrieval in parallel with intent classification.
- Retrieve speculatively. Start embedding the query as the user types, or at the latest on submit. Cache embeddings for frequent queries.
- Skip retrieval when it is not needed. A cheap classifier or heuristic can detect chit-chat and follow-ups that do not need fresh documents.
- Use a fast reranker, such as a small cross-encoder at about 20 ms, or drop reranking on latency-critical paths.

## Step 3: Make prefill cheaper, or avoid it

### How long prefill really takes

Prefill is compute-bound. A useful estimate is about `2 × parameters × prompt_tokens` FLOPs for the dense layers, plus the attention term, which grows with the square of prompt length.

For a 70B model (80 layers, hidden size 8,192) and a 30k-token prompt:

- Dense layers: `2 × 70e9 × 30,000 ≈ 4.2e15` FLOPs.
- Causal attention: roughly `2 × 80 × 30,000² × 8,192 ≈ 1.2e15` FLOPs.
- Total: about 5.4 PFLOPs.

An H100 SXM delivers about 989 dense BF16 TFLOPS ([NVIDIA lists 1,979 with sparsity](https://www.nvidia.com/en-us/data-center/h100/)). At a realistic 50% utilization that is about 500 TFLOP/s per GPU. The model also has to fit: 70B parameters in BF16 is about 140 GB of weights, which does not fit on one 80 GB H100. So the realistic deployment is a tensor-parallel (TP) group:

| Setup | Effective compute | 30k-token prefill | 500-token prefill |
|---|---|---|---|
| 1x H100, BF16 | does not fit in memory | n/a | n/a |
| TP=4, BF16 | about 2,000 TFLOP/s | about 2.7 s | about 35 ms |
| TP=8, BF16 | about 4,000 TFLOP/s | about 1.4 s | about 20 ms |
| TP=4, FP8 | about 4,000 TFLOP/s | about 1.4 s | about 20 ms |

These are back-of-envelope figures; communication overhead and kernel efficiency move them by tens of percent. Even so, a long uncached prompt costs 1-3 seconds of prefill on a 4-8 GPU group, and avoiding that work is the biggest TTFT win available.

### Prefix and KV caching

Keep the system prompt and conversation history byte-stable so turn N reuses the KV cache (the attention keys and values computed during prefill) from turn N-1. Then only the new user message needs prefilling.

- On an API, this is prompt caching. Anthropic reported latency reductions of up to 85% for long prompts when it launched the feature; in its examples, a 100k-token cached prompt went from 11.5 s to 2.4 s TTFT ([Anthropic, prompt caching announcement](https://claude.com/blog/prompt-caching)). Its [current docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) note that the 5-minute and 1-hour caches behave the same for latency, and that pre-warming the cache removes the miss penalty on the first user interaction.
- Self-hosted, use [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) or [SGLang RadixAttention](https://arxiv.org/abs/2312.07104). Add **session-affinity routing** so each conversation returns to the replica that holds its KV cache.

### Shorter prompts

Compact old turns into summaries, retrieve fewer and better chunks, and load tool schemas lazily, so there are fewer tokens to prefill.

### Hardware and parallelism

Because prefill is compute-bound, more tensor parallelism and FP8 cut prefill time nearly linearly, as the table above shows. This costs GPUs, so use it after caching and prompt trimming, not instead of them.

## Step 4: Fix scheduling (self-hosted)

- **Chunked prefill** ([Sarathi-Serve](https://arxiv.org/abs/2403.02310)). Split long prefills into chunks that are mixed into decode batches. A 30k-token prompt then no longer stalls everyone else's token stream, and new requests get scheduled quickly.
- **Prefill/decode disaggregation** ([DistServe](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677)). Run prefill and decode on separate GPU pools and transfer the KV cache between them. This lets you tune TTFT and inter-token latency independently. DistServe reported serving up to 7.4x more requests within both latency SLOs. It pays off at large scale with long prompts.
- Admission control and headroom. Queueing delay grows sharply above about 70-80% utilization. Autoscale on queue depth and TTFT, not on GPU utilization. Use priority classes so interactive chat goes ahead of batch work.
- Warm replicas. Loading a model on a cold start takes tens of seconds to minutes and must never sit on the request path.

[Serving Open-Weight LLMs at Scale](/blog/serving-open-llms-at-scale/) covers the scheduler, KV-cache and fleet-sizing side of this in more depth.

## Step 5: Network and client

- Stream with SSE (server-sent events) or WebSocket over HTTP/2 or HTTP/3, with keep-alive connections and regional endpoints near users.
- Remove buffering proxies. Nginx `proxy_buffering` left on will silently hold the stream until the response is complete.
- Pre-warm the connection when the chat input gets focus, so the TLS handshake is done before the user presses Enter.
- Handle disconnects and resumption cleanly; [Streaming LLM Responses over SSE and Surviving Disconnects](/blog/sse-streaming-disconnects/) covers the wire format and proxy settings.

## Step 6: Perceived latency

- Show a skeleton or "thinking" state immediately.
- For reasoning models, stream a short status ("Searching your docs...") while hidden reasoning runs.
- For simple turns, route to a smaller and faster model. [Routing Requests Between a Cheap and a Strong Model](/blog/routing-cheap-and-strong-models/) covers how to do that without losing quality.

## Model-level options

Speculative decoding ([Medusa](https://arxiv.org/abs/2401.10774), [EAGLE](https://arxiv.org/abs/2401.15077)) mainly improves *inter-token* latency, not TTFT. It speeds up decode, while TTFT is dominated by prefill and queueing. The two are often confused. Smaller or distilled models and quantization improve both.

## How to measure it

- Latency distributions: p50/p95/p99 TTFT by prompt-length bucket and by region, plus inter-token latency, also called TPOT (time per output token).
- Goodput: the fraction of requests that meet *both* the TTFT SLO and the TPOT SLO. Throughput that misses either SLO does not count.
- Realistic load tests: use production-shaped arrival processes (bursty, with heavy-tailed prompt lengths), not uniform synthetic load. Uniform load hides the queueing tail.
- Client-side measurement: instrument first rendered token in the client and compare it with server-side TTFT to expose proxies, TLS and rendering costs.
- Business link: tie TTFT to an engagement metric (abandonment, messages per session) through an A/B test. That data justifies the GPU cost of headroom.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Prefix/KV caching + session affinity | Large TTFT cut on multi-turn chats; also saves compute | Affinity hurts load balance; KV memory pressure evicts other work | Multi-turn chat, long system prompts |
| Chunked prefill | Smooths tail TTFT and TPOT; simple to enable | Slightly lower peak throughput; chunk size needs tuning | Mixed prompt lengths on shared replicas |
| Prefill/decode disaggregation | Tune each SLO independently; best goodput at scale | KV transfer needs a fast interconnect; operational complexity | Large fleets, long prompts, strict SLOs |
| Overprovision headroom | Removes the queueing tail directly | Expensive idle GPUs | Latency-critical consumer chat |
| Smaller/faster model | Cuts every stage | Quality risk | Easy turns, via a router |
| Parallelize pre-model work | Cheap and often the biggest win | Wasted work when results are discarded | Always |

- The main decision is where to sit on the cost-latency curve. Cost per token scales roughly with `1 / utilization`, so running at 60% instead of 85% costs about 1.4x more per token (`85 / 60`). In return, queueing delay falls several-fold: in a simple M/M/1 queue model, mean wait scales with `ρ / (1 − ρ)`, which is 5.7 at 85% and 1.5 at 60%, and p99 wait drops about 3x. Bursty real traffic widens the gap. Make this call with product, using engagement data.
- Session affinity and load balancing conflict. Use consistent hashing with bounded load, and fall back to recomputing the prefix when the preferred replica is full.

## Common mistakes

- **Optimizing the mean instead of p95/p99.** The tail comes from queueing and long prompts, and the mean hides both, so users still see multi-second stalls.
- **Expecting speculative decoding to fix TTFT.** It speeds up decode. TTFT is prefill plus queueing, so the first token arrives no sooner.
- **Cache-busting prompts.** Timestamps, randomized few-shot order, or per-user content placed before shared content each force a full prefill on every turn.
- **Measuring server-side only.** Buffering proxies, client rendering and TLS handshakes can add hundreds of milliseconds that server metrics never see.
- **Autoscaling on GPU utilization.** It lags, and utilization is high even when TTFT is fine, so the fleet scales at the wrong times. Scale on queue depth and TTFT SLO burn rate.
- **Adding safety gates in series.** Each sequential guardrail adds 100-300 ms. Run them in parallel or gate the stream.

## Thinking at scale

- Treat headroom as a product decision. The extra GPUs that keep p99 TTFT low are a recurring cost. Owning that number jointly with product, backed by an engagement A/B, keeps it from being cut in the next cost review.
- Prompt stability is a cross-team contract. Prefix caching only works if every team that touches the prompt keeps the shared prefix byte-stable. Put a cached-token-ratio alert on each feature so a careless change is caught within hours.
- Plan disaggregation as a staged migration. Moving to separate prefill and decode pools changes capacity planning, interconnect requirements and failure modes. Pilot it on the long-prompt traffic class first.

## Further reading

- [Sarathi-Serve (Agrawal et al., 2024)](https://arxiv.org/abs/2403.02310): chunked prefill and stall-free batching for the throughput-latency tradeoff.
- [DistServe (Zhong et al., 2024)](https://arxiv.org/abs/2401.09670): disaggregating prefill and decode to optimize goodput under TTFT and TPOT SLOs.
- [Splitwise (Patel et al., 2023)](https://arxiv.org/abs/2311.18677): phase splitting of prefill and decode across machines.
- [vLLM / PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180): KV-cache memory management behind modern serving engines.
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/): how prefix KV reuse works in vLLM.
- [SGLang / RadixAttention (Zheng et al., 2023)](https://arxiv.org/abs/2312.07104): radix-tree KV cache reuse across requests.
- [Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): API-level prefix caching for lower latency and cost, including cache pre-warming.
- [Anthropic prompt caching announcement](https://claude.com/blog/prompt-caching): measured TTFT reductions for long cached prompts.
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/): memory and tensor-core throughput figures used in the prefill estimate.
- [Medusa (Cai et al., 2024)](https://arxiv.org/abs/2401.10774) and [EAGLE (Li et al., 2024)](https://arxiv.org/abs/2401.15077): decode-side acceleration, useful to contrast with TTFT work.
