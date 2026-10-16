---
title: "Cutting LLM Cost by 70% Without Losing Quality"
description: "How to stack prompt caching, model routing, batch APIs and token hygiene into a 70-80% cost cut, with an eval gate that proves quality held."
slug: "cut-llm-cost-70-percent"
category: "cost-latency-serving"
tags: ["cost", "caching", "routing", "evals"]
seriesOrder: 6
pubDate: 2026-10-16
sourceDoc: "1wx9vLMMIykYpYxwlDoHQz8GunZXg7raHlJEYPtY-TAI"
---

## TL;DR

A 70% cost cut almost never comes from one trick. It comes from stacking three or four multiplicative levers on a cost model you have actually measured: caching the prompt prefix, routing easy traffic to a smaller model, sending non-interactive work through a batch API, and trimming wasted tokens. Because the savings multiply, together they typically remove 70-85% of spend (for example `0.5 × 0.6 × 0.8 = 0.24` of the original bill). Before touching anything, build a frozen eval set and a quality guardrail, so "without hurting quality" is something you test rather than something you hope for. Then roll out one lever at a time behind a feature flag and measure each against that guardrail.

## Requirements to pin down first

- **What does the spend actually consist of?** Break it down by endpoint and feature, input vs. output tokens, cached vs. uncached, and interactive vs. offline. In many products 10-20% of call sites cause around 80% of spend. A common culprit is one agent loop that re-sends a 30k-token context on every step. The breakdown tells you which lever to pull first.
- **What does "quality" mean for this product?** Task success rate, human preference win-rate, a downstream business KPI (resolution rate, thumbs-down rate), or all three? Which slices must not regress at all, such as enterprise tenants or safety-critical flows? This defines the guardrail every lever has to pass.
- **Hosted API or self-hosted?** On an API, the levers are tokens, model choice, caching and batch. On your own GPUs you also control utilization, batching, quantization and hardware choice.
- **Which latency SLOs (service-level objectives) constrain you?** Batch processing and aggressive cascades both add latency. A strict interactive SLO rules some levers out for some traffic.

## Step 0: a cost model and a quality gate

You cannot cut what you have not measured, and you cannot claim "no quality loss" without a test.

**Instrument every call** with: feature, tenant, model, input tokens, output tokens, cached tokens, latency and an outcome signal. With that in place, the north-star metric becomes cost per *successful task*, not cost per call. Comparing per task also absorbs tokenizer differences between model generations. Anthropic notes that Claude 4.7 and later models use a tokenizer that produces roughly 30% more tokens for the same text, so per-token prices alone can mislead.

**Build a frozen eval set** of roughly 1,000-3,000 real production samples, stratified by feature and difficulty. Score it with task-specific checks plus a calibrated LLM judge. Require judge-human agreement of at least 0.8 on a spot-checked subset before trusting the judge ([Making LLM-as-a-Judge Trustworthy](/blog/trustworthy-llm-as-a-judge/) covers calibration in depth).

**Define the guardrail as a non-inferiority test.** For example: the quality drop must be under 1 point absolute at 95% confidence, both overall *and* on each critical slice. Every lever in this post ships only if it passes that test.

## The five levers

![A waterfall of monthly spend: a 1 million dollar baseline drops to 600 thousand after prompt caching, 270 thousand after routing 65% of traffic to a cheaper model, 243 thousand after moving 20% of the remainder to batch, and about 207 thousand after token trimming.](/diagrams/q06-cost-levers.svg "Illustrative cost waterfall: each lever multiplies the bill left by the previous one.")

Each lever applies to the bill the previous lever left behind, which is why four moderate cuts compound into a large one.

### Lever 1: Prompt and prefix caching (often 30-60% of input cost)

Restructure prompts as **static, then semi-static, then dynamic**: system prompt, tool schemas and few-shot examples first, then retrieved documents, then the user turn. Provider prompt caching bills reads of a cached prefix at a small fraction of the base input price.

Anthropic's current multipliers, per its [pricing page](https://platform.claude.com/docs/en/about-claude/pricing):

| Operation | Price relative to base input |
|---|---|
| Cache write, 5-minute TTL (time to live) | 1.25x |
| Cache write, 1-hour TTL | 2x |
| Cache read (hit or refresh) | 0.1x on most models; 0.05x on Claude Opus 5.5 and Sonnet 5.5; 0.025x on Claude Fable 5.1 and Mythos 5.1 |

The 1-hour TTL costs more to write but pays off when a prefix is reused at intervals longer than five minutes, such as batch jobs or low-traffic tenants. A 5-minute write breaks even after a single read (1.25 + 0.1 < 2 × 1); a 1-hour write breaks even after two reads (2 + 0.1 + 0.1 < 3 × 1).

OpenAI applies prompt caching automatically for prompts of at least 1,024 tokens. On its [prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching), GPT-5.6 and later models bill cache reads at 0.1x (0.05x for GPT-6.1 Sol) and cache writes at 1.25x, with a 30-minute retention window.

For chat and agents, where the conversation prefix is re-sent on every turn, cache hit rates of 70-90% of input tokens are realistic. How much of the *bill* that removes depends on input's share of spend, not its share of tokens. A worked example, using a 5x output-to-input price ratio:

- Tokens are 85% input and 15% output. Spend is `0.85 × 1 + 0.15 × 5 = 1.6` units, so input is 53% of spend.
- Caching 80% of input tokens at 0.1x cuts input spend to `0.85 × (0.2 + 0.8 × 0.1) = 0.24` units.
- The new total is about 0.99 units: a **38% cut** to the bill.

When input dominates spend more heavily (long RAG contexts with short answers), the same caching can approach a 50-65% cut. When outputs are long, it is smaller.

**Self-hosted equivalent:** vLLM automatic prefix caching or SGLang RadixAttention. There the saving is avoided prefill compute and higher throughput per GPU rather than a billing discount.

The cache-hit side of this is covered in more depth in [Designing a Semantic Cache with Hit-Rate Tracking](/blog/semantic-cache-hit-rate-tracking/), which also explains when provider prefix caching beats a semantic cache.

### Lever 2: Model routing and cascades (often 40-60% of the remaining spend)

Price gaps between model tiers are now very large. At October 2026 list prices, Claude Sonnet 5.5 costs $2 / $10 per million input / output tokens and Claude Haiku 5.5 costs $0.10 / $0.50 for prompts up to 100k tokens: a 20x gap between adjacent tiers. Between a provider's top and bottom tiers the gap reaches about 100x. If 60-70% of traffic is "easy" (classification, extraction, short FAQ), a learned router or a cascade keeps quality within noise:

- A **router** predicts, before generation, whether the cheap model is good enough.
- A **cascade** calls the cheap model first and escalates to the strong model on low confidence or a failed verifier.

[FrugalGPT](https://arxiv.org/abs/2305.05176) reported matching the best individual model's performance with up to 98% cost reduction on its benchmarks. [RouteLLM](https://arxiv.org/abs/2406.18665) reported cost reductions of more than 2x in some settings without degrading response quality. Routing gets its own deep dive in [Routing Requests Between a Cheap and a Strong Model](/blog/routing-cheap-and-strong-models/).

**Distillation** goes further. Fine-tune a small model on the highest-volume narrow tasks, using the large model's outputs as labels. A fine-tuned small model often *beats* a prompted frontier model on a narrow task: in [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301), a 770M-parameter T5 outperformed few-shot prompted 540B PaLM on a benchmark. Per-call cost is commonly 10-50x lower, depending on which models you compare.

### Lever 3: Batch and async (50% off eligible traffic)

Evals, nightly enrichment, embedding backfills and summarization pipelines belong on batch APIs. Both Anthropic's [Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing) and OpenAI's [Batch pricing](https://developers.openai.com/api/docs/pricing) charge 50% of the standard rate. Anthropic says most batches finish within an hour, with a 24-hour limit.

The batch discount stacks with prompt caching. Because batched requests run concurrently and in any order, cache hits inside a batch are best-effort; Anthropic reports typical batch cache hit rates of 30-98% and suggests the 1-hour TTL for shared context in batches.

### Lever 4: Token hygiene (10-30%)

- **Output tokens are the expensive ones.** At current list prices output costs about 4-8x as much as input: 5x on every current Claude model, and 5-8x across OpenAI's current GPT-5 and GPT-6 lineup (for example $0.05 vs. $0.40 for gpt-5-nano). Cap `max_tokens`, require structured or terse output, and stop asking for chain-of-thought where the task does not benefit. With reasoning models, set the thinking budget per task.
- **Retrieve less, retrieve better.** Moving from top-20 to top-5 chunks with a reranker often *improves* quality while cutting context by 60-75%.
- **Compact history and tools.** Summarize long conversation history. Remove tool schemas the current step cannot use. Use prompt compression ([LLMLingua](https://arxiv.org/abs/2310.05736) reports up to 20x compression with little loss on its benchmarks) only where you have measured it on your own tasks.
- **Deduplicate requests.** Retries, double-submits and the same prompt sent by many users can all be served by an exact-match cache.

### Lever 5 (self-hosted): serving efficiency

Continuous batching, FP8 or INT4 weight quantization (AWQ, GPTQ), speculative decoding, and right-sizing the GPU SKU all lower cost per token. Utilization matters most: cost per token scales roughly with `1 / utilization`, so moving from 30% to 70% utilization cuts it to about 43% of its former level (`30 / 70`), more than halving it. [Serving Open-Weight LLMs at Scale](/blog/serving-open-llms-at-scale/) works through the fleet math.

## A worked example

All numbers are illustrative. Each step applies to the bill left by the previous step.

| Step | Assumption | Change | Monthly bill |
|---|---|---|---|
| Baseline | | | $1,000k |
| Prompt caching | about 80% of input tokens cached (see Lever 1) | -40% | $600k |
| Routing | 65% of traffic to a 10x-cheaper tier, plus escalations | -55% | $270k |
| Batch | 20% of the remainder moved to 50%-off batch | -10% | $243k |
| Token trimming | shorter outputs and contexts | -15% | about $207k |

Checking the routing step: with perfect routing, cost falls to `0.35 + 0.65 × 0.1 = 0.415` of the previous bill, a 58.5% cut. Budgeting for escalations (about 7% of the routed traffic paying for both calls adds `0.65 × 0.07 ≈ 0.046`) brings it to about 0.46, so -55% is a fair planning figure.

The end result is about $207k, a **79% reduction**. Planning for slightly more than 70% leaves headroom if the escalation rate turns out higher than expected. The levers are not perfectly independent (routing changes which model's cache you hit, for example), so re-measure after each rollout instead of trusting the product of estimates.

## How to measure it

- **Cost metrics:** cost per successful task, the cached-token ratio (cached input tokens over all input tokens), the router escalation rate, and p50/p95 latency.
- **Offline quality:** the non-inferiority test on the frozen eval set, overall and per critical slice, before each lever ships.
- **Online quality:** an A/B test on the business KPI, followed by a 1-5% holdout that stays on the old stack for weeks to catch slow drift.
- **Per-slice dashboards.** Averages hide the long-tail tenant whose quality dropped 8 points.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Prompt/prefix caching | Near-zero quality risk; large input savings; also cuts TTFT (time to first token) | Requires prompt restructuring; TTL misses; write premium (1.25x or 2x) | Always, first. Long shared prefixes, chat, agents |
| Routing/cascade | Biggest lever when traffic is heterogeneous | Router errors cause silent quality loss; cascades add latency for hard queries | Mixed difficulty, high volume |
| Distill/fine-tune a small model | 10-50x cheaper per call; often higher quality on narrow tasks | Training and maintenance cost; goes stale when the task drifts | Stable, high-volume, narrow tasks |
| Batch API | 50% off with zero quality change | Up to 24 hours of latency | Offline and async work |
| Token trimming / compression | Cheap to do; often improves quality | Aggressive compression drops key facts | Bloated RAG or agent contexts |
| Self-host + quantize | Lowest marginal cost at scale | Ops burden; quantization regressions on long-tail tasks | Roughly $1-2M/yr or more of steady load on tasks open-weight models handle well |

- **Sequence by risk.** Ship the zero-quality-risk levers (caching, batch, dedup) first, then the quality-risk levers (routing, distillation, quantization), each behind its own flag.
- **Optimize cost per *successful* outcome.** A cheaper model that causes more retries or human escalations can cost more end to end.

## Common mistakes

- **Breaking cache prefixes by accident.** A timestamp, request ID or non-deterministically ordered tool list near the top of the prompt drops the hit rate to zero, and the bill quietly returns. Monitor the cached-token ratio as a first-class metric.
- **An unrepresentative eval set.** Validating routing on easy samples and then shipping to a heavy tail produces regressions the eval never saw. Stratify the set and test per slice.
- **Trusting a biased LLM judge.** Judges favor longer, more fluent outputs and their own model family, which can hide a real regression. Calibrate against human labels and check judge-human agreement on the slices being changed.
- **Ignoring output and reasoning tokens.** Teams optimize input while a reasoning model spends thousands of hidden thinking tokens per call at output prices. Track thinking tokens per task and set budgets.
- **Ignoring second-order costs.** Lower quality raises retries, support tickets and churn, which can erase the token savings. Count them.
- **No rollback or holdout.** Without a persistent holdout, slow quality drift after the launch A/B ends goes undetected.

## Thinking at scale

- **Make cost per successful task a shared metric.** The 10-20% of call sites that drive most spend usually belong to different teams. A per-feature dashboard of cost per successful task gives each owner a number to move and stops savings in one place being offset by growth in another.
- **Keep the holdout running.** A 1-5% slice on the old stack for weeks after launch is the only reliable way to see slow drift from provider model updates or traffic shifts.
- **Revisit self-hosting at a threshold.** Below roughly $1-2M a year of steady, open-weight-friendly load, the operational burden of self-hosting usually outweighs the per-token savings. Above it, re-run the comparison as prices and models change.
- **Re-validate on every model or price change.** Tier price gaps, cache multipliers and tokenizers all changed during 2025-2026. Each change can move the optimal routing threshold or caching layout.

## Further reading

- [FrugalGPT (Chen, Zaharia, Zou, 2023)](https://arxiv.org/abs/2305.05176): LLM cascades, prompt adaptation and approximation; matched GPT-4 performance with up to 98% cost reduction on its benchmarks.
- [RouteLLM (Ong et al., 2024)](https://arxiv.org/abs/2406.18665): learned routers trained on preference data; more than 2x cost reduction in some settings without quality loss.
- [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing): current per-model prices, cache write and read multipliers, and batch prices.
- [Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): cache pricing multipliers, 5-minute and 1-hour TTLs, minimum cacheable lengths, breakpoints.
- [Anthropic Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing): 50% batch discount that stacks with caching; most batches finish within an hour.
- [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching): automatic prefix caching, minimum length, retention and how to structure prompts for it.
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing): current input, cached-input and output prices, and batch prices.
- [LLMLingua (Jiang et al., 2023)](https://arxiv.org/abs/2310.05736): prompt compression with small-model perplexity filtering, up to 20x compression.
- [Distilling Step-by-Step (Hsieh et al., 2023)](https://arxiv.org/abs/2305.02301): a 770M distilled model outperforming few-shot 540B PaLM on a narrow benchmark.
- [vLLM / PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180): the serving-efficiency foundation for self-hosted cost cuts.
