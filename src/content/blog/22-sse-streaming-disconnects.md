---
title: "Streaming LLM Responses over SSE and Surviving Disconnects"
description: "Stream tokens with server-sent events, decouple generation from the connection, resume with Last-Event-ID, and cancel cleanly so GPUs are freed."
slug: "sse-streaming-disconnects"
category: "production-engineering"
tags: ["streaming", "latency", "architecture", "serving"]
seriesOrder: 22
pubDate: 2026-10-16
sourceDoc: "15jgtoXCDflquvlaEjUw7LYBq2BmcoregHbUPWEtQCxw"
---

## TL;DR

Server-sent events (SSE) are a simple fit for token streaming: one-way, plain HTTP, and compatible with most proxies. You send a `text/event-stream` response with `id:`, `event:` and `data:` fields, flush small batches of tokens every 20-50 ms, and send heartbeat comments so intermediaries do not time out the connection. The hard part is disconnects. Decouple generation from the HTTP connection: the model writes tokens to a short-lived per-message log keyed by message ID, and the SSE handler tails that log, so a dropped client can resume with `Last-Event-ID`. Whether generation continues or stops after a disconnect is an explicit product and cost policy, and when it stops, the cancellation must reach the inference engine so the KV cache and GPU batch slot are freed.

## Requirements to pin down first

- **Which clients consume the stream?** Browser, mobile app or API SDK? The browser `EventSource` API only supports GET and cannot set custom headers. If you need a POST body or an `Authorization` header, use `fetch()` with a `ReadableStream` parser instead, and implement reconnection yourself.
- **What happens on disconnect?** A consumer chat app usually keeps generating (the user switched tabs and will come back). An API usually aborts (the client is gone and every token costs money). The answer decides whether generation can be tied to the connection at all.
- **Is mid-answer resume required?** If a page refresh or network blip must pick up mid-answer, you need a replayable stream log. If "show the final answer when it is done" is enough, a polling fallback to the message API may do.
- **What sits in front of the servers?** List every CDN, load balancer and proxy layer, with its idle timeout and buffering behavior. These decide your heartbeat interval and which headers you must set.

## Wire format and server mechanics

### Headers

```http
HTTP/2 200
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

`X-Accel-Buffering: no` tells nginx not to buffer this response (nginx buffers proxied responses by default). Do not gzip the stream unless your compression layer flushes per event: any buffering kills the streaming feel.

### Frames

Each event is a few `field: value` lines followed by a blank line. Use the message ID plus a sequence number as the event `id`, so a reconnecting client can say exactly where it stopped:

```text
id: msg_8f2c:41
event: delta
data: {"t":"Hello"}

id: msg_8f2c:42
event: delta
data: {"t":" world"}

```

Use typed events: `delta`, `tool_call`, `usage`, `error` and `done`. A final `done` event (some APIs send a `[DONE]` sentinel instead) lets the client tell **completion** apart from **truncation**. Anthropic's Messages API is a good production example of this shape: `message_start`, then `content_block_start` / `content_block_delta` / `content_block_stop` per block, `message_delta`, and a final `message_stop`, plus `ping` and in-band `error` events.

### Coalescing

A 70B-class model at about 40 tok/s emits a token every 25 ms. Flush every 20-50 ms or every N tokens, whichever comes first. Writing every token individually at 10k concurrent streams costs a syscall and a TLS record per token for no visible benefit.

### Heartbeats

A line starting with a colon is an SSE comment, which clients ignore but proxies see as traffic:

```text
: ping

```

Send one every 15 s. The AWS Application Load Balancer's default idle timeout is 60 s, and so is nginx's default `proxy_read_timeout`. Reasoning ("thinking") models and slow tool calls can easily stay silent longer than that. Note that ALB does not count HTTP/2 PING frames as activity, so the heartbeat must be real response data like the comment above.

### Errors mid-stream

By the time a mid-stream error happens, the server has already sent HTTP 200, so errors must be in-band:

```text
event: error
data: {"code":"overloaded","retryable":true}

```

Clients must treat a stream that closes without `done` as an error, never as a complete answer.

### HTTP/2 or HTTP/3 end to end

Under HTTP/1.1, browsers allow only about 6 connections per origin, and each open SSE stream holds one. A user with several tabs open can exhaust the pool and block the app's other requests. HTTP/2 and HTTP/3 multiplex streams over one connection and remove the limit.

## Decoupling generation from the connection

![Sequence diagram. The client POSTs a message with an idempotency key to the chat API, which enqueues a generation and returns a message ID. The client opens a stream on an SSE gateway. The generation worker appends numbered chunks to a stream log, and the gateway tails the log and forwards frames 1 to 41. The network drops, but the worker keeps appending chunks 42 to 60. The client reconnects to any gateway replica with Last-Event-ID set to sequence 41; the gateway reads from sequence 42, replays 42 to 60 and continues tailing live. Finally the worker appends a done event and persists the final text, and the gateway sends the done event.](/diagrams/q22-sse-resume.svg "Generation writes to a stream log; any gateway can replay from Last-Event-ID after a disconnect.")

Because the log, not the connection, owns the tokens, a dropped connection costs a replay instead of the answer.

1. **Create.** `POST /messages` validates the request, writes a message row with status `generating` and an idempotency key, enqueues the generation and returns a `message_id`. (It can also upgrade straight to the stream to save a round trip.)
2. **Generate.** A generation worker calls the inference engine and appends chunks to a **per-message stream log**, keyed by `message_id`, with monotonically increasing sequence numbers. Redis Streams works well; an in-memory ring buffer works if gateways can reach the worker that holds it. Give entries a TTL of about 10 minutes.
3. **Tail.** The SSE gateway (`GET /messages/{id}/stream`) reads from the sequence after `Last-Event-ID` (or from 0) and tails the log. Gateways are stateless, so any replica can resume any stream.
4. **Complete.** The worker persists the final text to the durable message store, marks the message complete, appends a `done` event, and lets the log expire.
5. **Client.** On a network error, reconnect with exponential backoff plus jitter, sending `Last-Event-ID`. Deduplicate by sequence number and render idempotently. If the log has already expired, fetch the completed message from the REST API.

`EventSource` sends `Last-Event-ID` automatically on reconnect (it is part of the WHATWG spec). With a `fetch()`-based client, store the last `id` you processed and send it yourself.

## Disconnect detection and cancellation

### Detecting a dead client

Detect disconnects through the request context (`ctx.Done()` in Go, `await request.is_disconnected()` in Starlette) or through a failed write. TCP can take a long time to notice a dead peer that never sent a FIN, so the heartbeat writes double as liveness probes: a write that fails is your signal.

### A policy per surface

| Surface | On disconnect |
|---|---|
| Consumer chat | Keep generating, up to a cap, so the answer is there when the user returns |
| API | Abort after a short grace period (about 2-5 s) to allow a reconnect |
| Explicit user "Stop" | Abort immediately and persist the partial answer, marked `stopped` |

### Propagate the cancel all the way down

Gateway, then worker, then the engine's `abort(request_id)`, which frees the sequence's KV blocks and batch slot. A dropped cancel leaves a zombie sequence that keeps generating to `max_tokens`. On a 70B-class model in BF16 (about 320 KiB of KV per token, see [Serving Open-Weight LLMs at Scale](/blog/serving-open-llms-at-scale/)), a 32K-token context holds 32,768 × 327,680 bytes ≈ **10.7 GB** of KV. At scale, zombie generations can waste a meaningful share of decode capacity (5-15% is a plausible range if cancels are routinely dropped), so track a "tokens generated after client gone" metric and drive it toward zero.

### Billing and agents

- **Billing:** bill the tokens actually generated up to the cancel, and record them in the usage event.
- **Agents with tools:** never abort halfway through a side-effecting tool call. Define cancel points between steps.

## Capacity math

Start from concurrency, not arrival rate. By Little's law, concurrent streams = arrival rate × average stream duration. At 400 output tokens and 40 tok/s, a stream lasts about 10 s plus 1-2 s of TTFT, so about 12 s.

```text
Concurrent streams = 700 new streams/s × 12 s          ≈ 8,400 streams
Token rate         = 8,400 streams × 40 tok/s           ≈ 336,000 tok/s
Log appends        = per-token:     336,000 appends/s
                     50 ms flushes: 8,400 × 20/s       ≈ 168,000 appends/s
                     (a 20 ms flush interval is shorter than the 25 ms token
                      gap, so it degenerates to ~1 token per append: ~336k/s)
```

So the stream log takes roughly **170k-340k appends per second**, not a few thousand. Two techniques keep that manageable:

- **Pipeline per worker.** Each generation worker batches the appends for all its streams into one pipelined round trip per flush interval. With, say, 200 workers flushing every 25-50 ms, that is 4,000-8,000 round trips per second carrying the 170k-340k entries.
- **Shard by message ID.** Spread streams across several Redis shards (hash on `message_id`) so no single shard carries the whole append rate, and benchmark per-shard throughput with your entry size.

Memory is the other constraint. Assuming about 100 bytes per entry, 2 tokens per entry and a 10-minute TTL after completion, the log holds about 700 × 612 s × 200 entries ≈ 86M entries, around **9 GB**. Shortening the post-completion TTL to 60 s (the final text is already in the durable store) cuts that to about 1 GB.

The gateway itself is cheap. One async gateway pod can hold 10-50k mostly idle connections, so 8,400 streams fit on a handful of pods. The real limits are file descriptors, load-balancer connection limits, and the stream log.

## How to measure it

- **Client-measured TTFT** (time to first visible token) and p99 inter-chunk gap. Server-side timings miss proxy buffering.
- **Percentage of streams ending without `done`**, and the resume success rate after reconnects.
- **Abort-propagation latency** (client gone until KV freed) and the zombie token rate (tokens generated after the client left, under an abort policy).
- **Gateway connection count and heartbeat-timeout rate**, split by proxy path (CDN, corporate proxies, mobile networks).
- **Stream-log health:** append rate, p99 append latency per shard, and memory.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| SSE (fetch + stream parser) | Plain HTTP, proxy- and CDN-friendly, built-in resume semantics, easy to debug | One-way; HTTP/1.1 connection limits; proxies can buffer | Default for LLM token streaming |
| WebSockets | Bidirectional, low overhead per message | Sticky, stateful; load balancer and proxy quirks; you build your own resume | Realtime voice, collaborative editing, interactive tool approval mid-stream |
| Long-polling / chunked JSON | Works everywhere | Higher latency, more requests | Legacy clients, very restrictive networks |
| Couple generation to the connection | Simplest; auto-cancel saves GPU | Any blip loses the answer; no resume | Prototypes, internal tools |
| Decoupled generation + stream log | Resumable, deploy-safe, multi-device | Redis or log infrastructure; cancel must be enforced explicitly | Production consumer chat |

- **Keep vs cancel on disconnect** is a product, cost and safety decision. Make it per surface and measurable.
- **Flush cadence** trades smoothness and perceived speed against syscall and network overhead. 20-50 ms works for most products.

## Common mistakes

- **Proxy buffering and idle timeouts.** nginx, a CDN or a corporate proxy buffers the whole response, or kills it at 60 s, and users see nothing until the end. Fix with heartbeats, disabled buffering, and tests through the real edge.
- **Not propagating cancellation to the engine.** Zombie generations keep holding KV cache and batch slots, shrinking capacity for everyone.
- **Treating TCP close as completion.** Without a terminal `done` event, truncated answers get persisted as if they were complete.
- **Reconnect storms after a gateway deploy or outage.** Without jitter and backoff, every client reconnects at once and stampedes the stream log and the auth service.
- **Non-idempotent retries.** A client retries the POST and triggers two generations: double cost and duplicate messages. Idempotency keys prevent it.
- **Auth with `EventSource`.** It cannot set headers, so teams put tokens in query strings, where they leak into access logs. Use `fetch()` streaming with an `Authorization` header, or a short-lived single-use stream token.

## Thinking at scale

- **Deploys.** Long-lived streams make deploys slow and disruptive. Drain connections with a maximum stream duration, and send a `reconnect` event so clients resume on a new pod. That is safe precisely because state lives in the log, not in the pod.
- **One policy owner.** The keep-vs-cancel rule, the grace period and the generation cap belong to one team and one config, per surface, with the zombie-token metric on its dashboard. Otherwise each client team picks its own behavior and GPU waste becomes nobody's problem.
- **Multi-device for free.** Once streams are keyed by message ID in a shared log, a second device (phone and laptop) can tail the same answer. Plan auth for that from the start.

## Further reading

- [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events): event format, `id`, `retry`, reconnection.
- [WHATWG HTML Standard: Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html): the normative spec, including `Last-Event-ID` semantics.
- [MDN: AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController): client-side cancellation of `fetch()` streams.
- [Claude API: Streaming Messages (Anthropic)](https://platform.claude.com/docs/en/build-with-claude/streaming): a production typed-event SSE schema (`message_start`, `content_block_delta`, `message_stop`, `ping`, `error`).
- [OpenAI API: Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses): another widely used SSE contract, with typed events such as `response.output_text.delta` and a per-event sequence number.
- [nginx: ngx_http_proxy_module](https://nginx.org/en/docs/http/ngx_http_proxy_module.html): `proxy_buffering`, `proxy_read_timeout` (60 s default) and `X-Accel-Buffering`.
- [AWS: Application Load Balancer attributes](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/edit-load-balancer-attributes.html): the 60 s default idle timeout and why HTTP/2 PINGs do not reset it.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180): why freeing KV on abort matters for capacity.
