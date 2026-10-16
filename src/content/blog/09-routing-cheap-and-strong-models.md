---
title: "Routing Requests Between a Cheap and a Strong Model"
description: "Design an LLM router that sends each request to the cheapest good-enough model: rules, learned routers, cascades, threshold tuning and drift monitoring."
slug: "routing-cheap-and-strong-models"
category: "cost-latency-serving"
tags: ["routing", "cost", "evals", "llm-as-judge"]
seriesOrder: 9
pubDate: 2026-10-16
sourceDoc: "1wx9vLMMIykYpYxwlDoHQz8GunZXg7raHlJEYPtY-TAI"
---

## TL;DR

Model routing is a cost-quality optimization: send each request to the cheapest model whose expected quality is within tolerance of the strong model's. In practice, a learned router predicts the probability that the cheap model's answer is as good as the strong model's, and compares it to a threshold tuned for a cost or quality target. Cascade-style escalation adds a second chance: when the cheap model's output fails a verifier, the request goes to the strong model. Train the router on your own traffic, labeled by comparing both models' outputs, and judge it by the quality-cost curve (for example, 95% of the strong model's quality at 30-40% of its cost), not by router accuracy. Production also needs hard override rules (tenant tier, safety-sensitive intents), calibration monitoring, and a shadow-evaluated holdout, because the traffic mix drifts.

## Requirements to pin down first

- **How large is the cost and latency gap between the models?** A 10x price gap and 3x latency gap leaves a lot of room to save. A 2x gap leaves very little after router and escalation overhead. Current gaps are large: at October 2026 list prices, Claude Sonnet 5.5 is 20x the per-token price of Claude Haiku 5.5 (for prompts up to 100k tokens), per [Anthropic's pricing page](https://platform.claude.com/docs/en/about-claude/pricing).
- **What is the quality metric, and the tolerance?** "Non-inferior within 1 point on task success" and "win-rate of at least 48% against strong-only" lead to different thresholds.
- **How heterogeneous is the traffic?** If 90% of queries are genuinely hard, routing cannot help much. Measure first: run both models on a sample and compute the fraction where the cheap model is as good. That fraction is the ceiling on savings.
- **Is there latency budget for cascades?** A cascade spends the cheap call's latency on every escalated request before the strong call even starts.
- **Same provider and model family?** That affects prompt compatibility, and prefix caching: switching models loses the cache.

## Router designs, from simplest to strongest

![A request enters a gateway that applies hard override rules, then a learned router scores it. Clearly easy requests go to the cheap model, clearly hard ones go to the strong model, and the uncertain middle band goes to the cheap model followed by a verifier. If the verifier passes, the cheap answer is returned; if it fails, the request escalates to the strong model.](/diagrams/q09-hybrid-router.svg "A hybrid router: confident decisions go straight to one model, and only the uncertain band pays for a cascade.")

The hybrid design spends verification and escalation cost only on the requests where the router is unsure.

### 1. Rules or intent-based routing

A route table by feature or intent, such as "classification goes to the cheap model; code generation goes to the strong model". It is transparent, has zero latency, and makes a good baseline. Rules often capture around half of the achievable savings on their own.

### 2. Learned pre-generation router

A small classifier runs on the prompt before any model is called. Options include a fine-tuned BERT-class encoder, or an embedding plus an MLP or kNN head. Add features such as prompt length, conversation depth, tool count and tenant tier. The output is `P(cheap model is good enough)`, and inference takes 2-10 ms.

[RouteLLM](https://arxiv.org/abs/2406.18665) trains routers on preference data (Chatbot Arena comparisons plus augmentation). Its matrix-factorization and BERT-based routers both generalized reasonably to new model pairs, and it reported cost reductions of more than 2x in some settings without degrading quality. [Hybrid LLM](https://arxiv.org/abs/2404.14618) uses a similar quality-gap predictor with a tunable threshold and reported up to 40% fewer calls to the large model with no drop in response quality.

### 3. Cascade (post-generation)

Call the cheap model first, score its output, and escalate if the score is low. Scoring options:

- Self-consistency: agreement across 2-3 samples.
- A verifier model.
- Token log-probability confidence.
- Programmatic validation: schema checks for JSON, tests for code.

[FrugalGPT](https://arxiv.org/abs/2305.05176) and [AutoMix](https://arxiv.org/abs/2310.12963) are cascade-style approaches; AutoMix uses the small model's self-verification to decide when to escalate. Cascades are more precise than pre-generation routers because they judge the actual output, but escalated requests pay for two calls and take longer.

### 4. Hybrid

The router sends clearly hard requests straight to the strong model, clearly easy ones to the cheap model, and only the uncertain middle band through the cascade. This gives the best cost-quality frontier, at the price of the most moving parts.

## Building the training data

- **Sample 20,000-50,000 production prompts,** stratified by feature and difficulty. Run both models on each.
- **Label with an LLM judge** in pairwise mode, scoring both orderings to cancel position bias, plus programmatic checks where they exist. Give about 1,000-2,000 examples human labels to calibrate the judge (see [Making LLM-as-a-Judge Trustworthy](/blog/trustworthy-llm-as-a-judge/)).
- **Define the target:** `y = 1` if the cheap answer is at least as good as the strong one (or within a margin).
- **Include hard negatives:** short prompts that are deceptively hard, such as "prove this" or "fix this bug". Length alone does not predict difficulty.
- **Retrain monthly or on drift.** Both models change underneath you when providers update them.

Distillation widens the routable fraction: fine-tuning the cheap model on your traffic ([Distilling Step-by-Step](https://arxiv.org/abs/2305.02301)) makes more requests "easy".

## Choosing the threshold

- **Sweep the threshold** and plot quality against the percentage of calls routed to the strong model. This is the deferral curve.
- **Pick an operating point,** for example "85% of the strong model's win-rate with 25% of calls going to the strong model".
- **Report RouteLLM's metrics:** APGR (average performance gap recovered, the area under the curve of how much of the cheap-to-strong quality gap you recover) and CPT (call-performance threshold, the share of strong-model calls needed to reach a given performance level).
- **Set per-slice thresholds** where slices differ. Enterprise tenants, for example, can get a more conservative threshold.

### A worked example of the cost math

Illustrative numbers: the cheap model costs 10% of the strong model per request. The router sends 70% of traffic to the cheap path, and 10% of that cheap traffic is escalated by the verifier.

- Strong-only requests: `0.30 × 1.0 = 0.30`
- Cheap calls: `0.70 × 0.1 = 0.07`
- Escalated strong calls: `0.70 × 0.10 × 1.0 = 0.07`
- Total: `0.44` of the strong-only cost, a **56% saving**.

The escalated 7% of traffic also has a latency of roughly cheap plus strong. Check that this tail fits the product's SLO.

## Serving architecture

- **The router runs in the gateway** with a budget of about 5 ms.
- **Hard overrides come first:** safety-sensitive intents, paid tier, an explicit user model choice, long-context requirements, and tool sets the cheap model handles poorly.
- **Keep conversations sticky.** Switching models mid-conversation breaks prefix caching and can cause style or persona whiplash. Route per conversation, or only ever escalate upward.
- **Provide a fallback path.** On cheap-model errors, timeouts or refusals, retry on the strong model.
- **Log everything:** the decision, router score, chosen model and outcome for every request.

Routing as a pattern is one of the basic workflows described in Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): classify an input and direct it to a specialized follow-up task.

## How to measure it

- **Offline:** the deferral curve on a held-out set, and the calibration of the router score (a reliability diagram and ECE, expected calibration error). A well-calibrated score makes threshold changes predictable.
- **Online:** an A/B test of routed vs. strong-only traffic, comparing the business KPI and regenerate and thumbs-down rates per arm.
- **Shadow slice:** keep a permanent 1-2% slice where both models run, to estimate the router's live false-easy rate (hard requests sent to the cheap model).
- **Drift alarms:** the fraction routed to the cheap model changing more than an agreed percentage week over week, a shift in the score distribution, or a rising escalation rate.
- **Expected outcome:** for typical assistant traffic, 50-80% routed to the cheap model, total cost down 40-70%, and quality within noise on aggregate metrics. These are typical ranges, not guarantees; verify the long tail separately.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Static rules by feature/intent | Transparent, zero latency, easy to debug | Coarse; misses difficulty within an intent | Starting point; clearly separated features |
| Learned pre-generation router | One model call per request; milliseconds of overhead; fine-grained | Needs labeled data; silent errors; drifts | High volume, heterogeneous difficulty |
| Cascade with verifier | High precision; uses actual output quality | Escalated requests pay about 2x cost and latency | Verifiable tasks (code, JSON, math), relaxed latency |
| Hybrid router + cascade band | Best cost-quality frontier | Most complex system | Mature, large-scale deployments |
| Always the strong model | Simplest, maximum quality | Highest cost | Low volume or very high stakes |

- **Errors are asymmetric.** Routing a hard query to the cheap model (a false-easy) hurts users. Routing an easy query to the strong model only wastes money. Tune for a false-easy budget, not for accuracy.
- **Find where the router's value comes from.** If a simple length or intent rule gets 80% of the gain, ship the rule and spend the effort elsewhere.

## Common mistakes

- **Optimizing router accuracy instead of the cost-quality curve.** A 90%-accurate router can still be worse than a rule if its errors land on high-value queries.
- **Label leakage and judge bias.** A judge that favors longer answers makes the strong model look better everywhere, which teaches the router never to route to the cheap model.
- **Switching models mid-conversation.** It breaks prompt caching (raising cost), changes tone, and the cheap model may not follow context the strong model established.
- **Assuming the router stays valid after model updates.** If a provider quietly improves the cheap model, you are now under-routing to it and overpaying. Re-label regularly.
- **Ignoring cascade latency.** p95 latency for escalated requests is cheap plus strong, which product may not accept for chat.
- **Ignoring gaming.** Users learn that "this is very important and complex" gets them the strong model. Watch for prompt features being exploited.

## Thinking at scale

- **Treat the router as a model with its own lifecycle.** It needs an owner, a retraining cadence, a labeled eval set and a rollback plan, just like any other production model. Both underlying LLMs change on the provider's schedule, not yours.
- **Keep the shadow slice permanently.** The 1-2% of traffic where both models run is the only ongoing ground truth for the false-easy rate. It is cheap insurance against silent quality loss.
- **Revisit the model pair as prices change.** Tier gaps moved substantially during 2025-2026. A new cheap model can widen the routable fraction overnight, and a price cut on the strong model can make routing not worth its complexity.

## Further reading

- [RouteLLM (Ong et al., 2024)](https://arxiv.org/abs/2406.18665): learned routers from preference data; deferral curves, APGR and CPT metrics.
- [FrugalGPT (Chen, Zaharia, Zou, 2023)](https://arxiv.org/abs/2305.05176): LLM cascades with a learned answer scorer.
- [AutoMix (Aggarwal, Madaan et al., 2023)](https://arxiv.org/abs/2310.12963): cascades driven by the small model's self-verification.
- [Hybrid LLM (Ding et al., ICLR 2024)](https://arxiv.org/abs/2404.14618): quality-aware routing between small and large models with a tunable quality threshold; up to 40% fewer large-model calls.
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): routing as a core workflow pattern and when to use it.
- [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing): current per-tier prices that set the size of the routing opportunity.
- [Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): why model stickiness matters for cache reuse.
- [Distilling Step-by-Step (Hsieh et al., 2023)](https://arxiv.org/abs/2305.02301): making the cheap model better on your traffic, which widens the routable fraction.
