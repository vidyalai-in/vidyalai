---
title: "Designing a Semantic Cache with Hit-Rate Tracking"
description: "A layered semantic cache for LLM responses: scoped keys, tuned thresholds, borderline verification, invalidation, and false-hit-aware hit-rate metrics."
slug: "semantic-cache-hit-rate-tracking"
category: "cost-latency-serving"
tags: ["caching", "vector-search", "cost", "observability"]
seriesOrder: 8
pubDate: 2026-10-16
sourceDoc: "1wx9vLMMIykYpYxwlDoHQz8GunZXg7raHlJEYPtY-TAI"
---

## TL;DR

A semantic cache embeds each normalized request, looks for a nearest neighbor above a similarity threshold within the same scope (tenant, model, prompt version, context fingerprint), and returns the stored response instead of calling the LLM. The hard part is deciding when a cached answer is safe to reuse, not running the vector search. That takes a per-domain threshold tuned on labeled query pairs, a verification step for borderline matches, and strict invalidation and tenant isolation. Track exact-hit, semantic-hit and miss rates, and also the *false-hit rate* from sampled audits: a cache with a 30% hit rate and 5% wrong answers is a quality incident, not a cost saving.

## Requirements to pin down first

- **Is the traffic actually repetitive?** FAQ and support bots often see 20-40% near-duplicate queries. Open-ended multi-turn chat is often under 5%. Replay a week of logs offline before building anything; a low ceiling means the cache is not worth its risk.
- **Are responses context-dependent?** Personalized, time-sensitive, or RAG-over-changing-data responses are dangerous to share. They shrink what can be cached and push you toward caching intermediate results instead.
- **What does a wrong hit cost?** A wrong customer-facing medical or financial answer is very different from a slightly off internal search suggestion. That cost sets the tolerated false-hit rate, which in turn sets the threshold.
- **What is the latency budget for the lookup itself?** Aim for under 10-20 ms so misses are not noticeably penalized.

## The request path

![A request flows through normalization and scope-key construction, then an exact-match cache. On an exact miss it goes to a semantic ANN lookup filtered by scope. Similarity at or above tau-high returns the cached answer; between tau-low and tau-high a verifier decides; below tau-low, or on a verifier rejection, the LLM is called. The response then passes a write-admission check before being stored, and source changes or TTL expiry invalidate entries.](/diagrams/q08-semantic-cache-flow.svg "Layered semantic cache: exact layer, semantic layer with two thresholds, verifier for the borderline band, and gated writes.")

Every path that does not end in a confident or verified hit falls through to the LLM, and only admitted responses are written back.

### 1. Normalize and scope

Lowercase, strip whitespace, and canonicalize numbers and dates where it is safe to do so. Then build a **scope key**:

```text
scope = hash(tenant, model_id, system_prompt_version, tool_set, locale, user_context such as auth role)
```

Lookups never cross scopes. Putting the model and prompt version in the key also means a version bump automatically starts a fresh cache.

### 2. Exact layer first

Look up `hash(scope, normalized_prompt)` in a key-value store such as Redis. It costs under 1 ms, has no false hits, and often catches 5-15% of traffic (retries and popular queries).

### 3. Semantic layer

- **Embed only the semantic part:** the last user turn, plus a short summary of prior turns in a multi-turn conversation. Use a small, fast embedding model (about 5-15 ms on CPU or GPU, 384-1,024 dimensions).
- **Search** with an ANN (approximate nearest neighbor) index, such as [HNSW](https://arxiv.org/abs/1603.09320) in a vector database or [FAISS](https://arxiv.org/abs/1702.08734), filtered by scope key, with top-k of 3-5.
- **Decide with two thresholds:**
  - Similarity at or above `τ_high` (for example 0.95 cosine): accept the hit.
  - `τ_low ≤ similarity < τ_high`: **borderline**. Optionally verify with a cross-encoder or a small LLM judge ("Do these two questions have the same answer?"). This costs about 30-80 ms, still far cheaper than generation.
  - Below `τ_low`: a miss.

### 4. On a miss, decide whether to write

Call the LLM, then apply a **write-admission** policy. Skip writing responses that contain user-specific data, involve tool calls with side effects, are low-confidence or refusals, or answer long, tail-unique queries that will never recur. For admitted responses, store:

```text
{embedding, normalized_query, response, scope, model_version, created_at,
 source_doc_versions, ttl, hit_count}
```

### 5. Invalidation

- **TTL by content class:** hours for news, days or weeks for a stable FAQ.
- **Event-driven purge** when source documents change. Storing document IDs and versions with each entry makes this possible.
- **Version bumps:** a new model or prompt version is part of the scope key, so old entries simply stop matching.

### 6. Eviction

Use LRU or LFU (least recently or least frequently used) eviction with size caps. Hit distributions are Zipfian, so a few hundred thousand entries usually capture most of the value.

## Tuning the threshold

1. **Build a labeled pair set from logs:** about 2,000-5,000 query pairs scored "same answer" or "different answer" by humans or a strong judge model.
2. **Include hard negatives:** "cancel my order" vs. "cancel my subscription"; "flights to Paris in May" vs. "flights to Paris in March".
3. **Plot precision against hit rate as τ varies,** and pick τ to meet a target false-hit rate, for example at most 0.5% for customer-facing traffic.
4. **Set thresholds per route or intent,** not globally. Thresholds are domain-specific.

Recent work goes further: [vCache](https://arxiv.org/abs/2502.03771) learns a separate threshold for each cached entry online, with a user-specified bound on the error rate.

## How to measure it

Emit one event for every lookup:

```text
{scope, layer: exact | semantic | verified | miss, similarity, latency_ms, entry_age, entry_id}
```

Build these views from it:

- **Hit rate by layer, tenant and intent,** measured two ways: per request, and by **cost saved** (tokens avoided × price). Hits on short prompts save little, so the cost-weighted view is the honest one.
- **Similarity histogram of accepted hits.** Watch for mass drifting toward the threshold, an early sign of a distribution shift.
- **False-hit rate.** Sample about 1% of semantic hits, regenerate a fresh answer in shadow, and have a judge compare the two. Also compare user signals (regenerate clicks, thumbs-down) on hit responses against miss responses, and alert if hits get worse feedback.
- **Cache efficiency.** Entries written vs. entries ever hit (often 80% or more are never reused, which argues for stricter write admission), the hit-count distribution, and the eviction rate.
- **Latency.** Lookup p99, plus the end-to-end win on hits (often 1-5 s down to about 50 ms).
- **Offline replay harness.** Replay logged traffic against new thresholds or embedding models to predict hit rate and false hits before deploying.

### Expected numbers

These are typical ranges, not guarantees:

| Workload | Combined hit rate | False-hit rate |
|---|---|---|
| FAQ / support bot | 20-40% | under 1% with verification |
| General assistant | 3-10% | depends heavily on threshold |

For general chat, provider prompt caching usually saves more money than semantic caching. Prompt caching discounts the repeated prefix of every request with no correctness risk, while a semantic cache only helps the small fraction of whole requests that repeat. See [Cutting LLM Cost by 70% Without Losing Quality](/blog/cut-llm-cost-70-percent/) for the prompt caching numbers.

### A worked example: is the lookup worth it?

Suppose a 5% hit rate, LLM responses that take 2 s, cache hits served in 50 ms, and 30 ms of lookup added to every request.

- Mean latency saved per request: `0.05 × (2,000 − 50) ≈ 98 ms`.
- Mean latency added per request: about 30 ms.
- Net mean: about 68 ms faster.

The mean improves, but 95% of requests (the misses) each get 30 ms slower, so p50 and p95 latency get worse. Whether that trade is acceptable depends on your SLO and on how much the 5% of avoided LLM calls saves. At low hit rates, a cheaper lookup (exact layer only, or a smaller embedding model) is often the better design.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Exact-match cache only | Zero false hits; trivial | Low hit rate on free text | Always, as layer 1; templated or programmatic traffic |
| Semantic, fixed global threshold | Simple; good hit rate | Wrong-answer risk varies by domain | Prototypes, low-stakes domains |
| Semantic + verifier for borderline | High precision at decent recall | Extra 30-80 ms and cost on borderline lookups | Customer-facing, moderate stakes |
| Per-intent or learned thresholds | Best precision-recall frontier | Needs labeled data and maintenance | Mature, high-volume deployments |
| Provider prefix caching instead | No correctness risk; works for any repeated prefix | Saves input tokens only, not generation | Long shared prompts, unique questions |
| Cache retrieval results, not answers | Safer; the answer stays fresh and personalized | Smaller savings | Personalized or time-sensitive answers |

- **The real decision is the tolerated false-hit rate.** Set it from the cost of a wrong answer, then let it determine τ and therefore the hit rate. Do not start from a target hit rate.
- **Isolation beats hit rate.** Never share entries across tenants or permission levels, even if that halves the hit rate.

## Common mistakes

- **Treating embedding similarity as answer equivalence.** Negations, entities, dates and numbers ("2023 revenue" vs. "2024 revenue") embed nearly identically, so the cache confidently returns the wrong answer. Use hard-negative evals and explicit entity and number checks.
- **Cross-tenant or cross-user leakage.** A cached answer containing user A's data served to user B is a security incident, not a quality bug. Scope keys and write-admission rules are mandatory.
- **Ignoring multi-turn context.** "What about the second one?" means different things in different conversations. Embedding only the last turn produces false hits.
- **Serving stale answers** after document, price or policy changes, or after a model or prompt upgrade. Version everything in the key or the entry metadata.
- **Measuring only hit rate.** Without false-hit audits and a cost-weighted hit rate, you cannot tell whether the cache helps or hurts.
- **Ignoring lookup overhead on misses.** At low hit rates, adding latency to every miss can make the user-visible tail worse even when the mean improves (see the worked example).

## Further reading

- [GPTCache (Zilliz, GitHub)](https://github.com/zilliztech/GPTCache): open-source semantic cache with pluggable embedding, storage and similarity evaluation.
- [vCache: Verified Semantic Prompt Caching (Schroeder et al., 2025)](https://arxiv.org/abs/2502.03771): learned per-entry thresholds with user-defined error-rate guarantees.
- [MeanCache (Gill et al., 2024)](https://arxiv.org/abs/2403.02694): user-centric semantic caching, with analysis of false hits and context-dependent queries.
- [Sentence-BERT (Reimers and Gurevych, 2019)](https://arxiv.org/abs/1908.10084): the bi-encoder embedding approach behind most semantic caches.
- [HNSW (Malkov and Yashunin, 2016)](https://arxiv.org/abs/1603.09320): the graph-based ANN index used by most vector stores.
- [FAISS: Billion-scale similarity search with GPUs (Johnson et al., 2017)](https://arxiv.org/abs/1702.08734): the vector search library for self-managed ANN.
- [Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): the complementary, zero-risk prefix caching layer.
