---
title: "Adding a Reranker and Proving It Helped"
description: "How to add a cross-encoder reranker to a RAG pipeline, size it, and prove the gain with paired offline evals, end-to-end checks and a guarded A/B test."
slug: "reranker-evaluation"
category: "retrieval-rag"
tags: ["reranking", "retrieval", "evals", "latency", "cost"]
seriesOrder: 5
pubDate: 2026-10-16
sourceDoc: "1h3m8wtZykYFchMjt8t57Aqya0AhrdYy0tlem7E4EI3A"
video: { youtube: "cWgwDQHYzTI", duration: "11:57", url: "https://youtu.be/cWgwDQHYzTI" }
---

## TL;DR

Add a cross-encoder reranker that rescores the first stage's top 50-200 candidates and passes the top 5-20 to the LLM (large language model). Then judge it with a controlled comparison. Offline, on a labeled and stratified query set, the reranker should raise nDCG@10 and recall at the *context cutoff k* with statistical significance, and end-to-end answer correctness and faithfulness should rise at equal or smaller context size. Online, an A/B test should show a user-facing quality gain with p95 latency and cost per query as guardrails. The reranker "helped" only if the quality gain survives the latency budget and holds on every important slice.

## Requirements to pin down first

- **What is first-stage recall@N at the candidate depth?** A reranker can only reorder what it is given. If recall@100 is 0.70, the ceiling is 0.70, and the fix belongs upstream (see [BM25 vs Dense vs Hybrid Search](/blog/bm25-dense-hybrid-search/)).
- **How much latency budget is left?** A typical reranking slot is 50-250 ms at p95. That bounds model size and candidate depth.
- **Hosted or self-hosted?** Hosted APIs (Cohere Rerank, Voyage AI rerankers) are fast to adopt; self-hosted models (bge-reranker, monoT5, a fine-tuned cross-encoder) give control. Data-residency constraints may decide this for you.
- **How many passages does the LLM consume: 5 or 20?** That k is the metric cutoff to evaluate at, because rank 30 versus rank 40 never reaches the model.

## Designing the reranking stage

![Two-part diagram. Online path: a query goes to a first-stage hybrid retriever returning N equal to 100 candidates, then to a cross-encoder reranker, then the top k passages go to the LLM, which produces the answer. A dashed timeout edge from the reranker leads to a fallback that uses the first-stage order and still feeds the top k. Measurement loop: an offline paired evaluation of nDCG@10, recall@k and end-to-end quality feeds a pre-registered ship rule, which gates an online A/B test with p95 latency and cost guardrails; the A/B test monitors the reranker in production, and production logs feed new queries back into the offline eval set.](/diagrams/q05-rerank-measurement.svg "Retrieve-then-rerank online, with a timeout fallback; offline evals and a pre-registered rule gate the online A/B test.")

The reranker sits on the latency-critical path, so it needs both a measurement loop that justifies it and a fallback that bounds its worst case.

### 1. Choose the candidate depth N

Start at N = 100. Plot first-stage recall@N against N and pick the point where the curve flattens, often between 50 and 200. Reranking cost grows linearly with N, so depth past the knee buys latency and little else.

### 2. Choose the model

- Cross-encoders: BERT, MiniLM or XLM-R-class models, roughly 20-600M parameters, that read each (query, passage) pair jointly with full attention. This is the default.
- LLM rerankers, listwise or pointwise (RankGPT-style): strong zero-shot, but 10-100× more expensive. Use them to label data and distill a smaller model, or for low-QPS (queries per second), high-value queries.
- Late interaction (ColBERT): a cheaper middle ground that can serve as the reranker, especially if you already run it.

A worked latency estimate: reranking 100 passages of about 256 tokens is 25,600 tokens. A 100M-parameter cross-encoder needs about `2 × 100M` FLOPs per token, so about 5 TFLOPs per query. A data-center GPU sustaining 50-150 TFLOPs in fp16 finishes that in roughly 30-100 ms when batched. A 22M-parameter MiniLM is about 5× faster; a 560M-parameter model is about 5× slower. Benchmark on your hardware before setting N.

### 3. Format the input

Feed the title, the section header and the chunk text together. Headers carry a lot of signal. Check the model's maximum input length: many classic BERT- and MiniLM-based cross-encoders cap at 512 tokens, and anything longer is silently truncated, which can drop the evidence from long chunks. Some newer rerankers accept longer inputs; confirm the limit and the truncation behavior for the model you choose.

### 4. Set the output policy

Take the top k by score. Optionally drop candidates below a calibrated score threshold, so the LLM gets fewer, better passages. This can cut context tokens substantially (reductions of 30-50% are plausible), which can pay for the reranker by itself.

### 5. Fine-tune later

As a second phase, fine-tune on in-domain pairs from clicks, citations or LLM-labeled data, with hard negatives mined from the first stage. Domain fine-tuning commonly adds several nDCG points over an off-the-shelf model. It is also cheaper than fine-tuning the embedder, because changing the reranker requires no re-indexing.

Domain mismatch can make a reranker harmful: in the LegalBench-RAG benchmark, a general-purpose reranker performed worse than no reranker on legal text. [Chunking Legal Contracts for Retrieval](/blog/chunking-legal-contracts/) discusses that case.

## How to measure it

### Offline: the eval set

Use 500-2,000 real queries, stratified by type, with graded relevance judgments. Pool the judgments from the top 20 of *both* the baseline ranking and the reranked list. Otherwise, relevant passages that only the reranker surfaces are counted as misses.

### Offline: retrieval metrics at the cutoff the LLM sees

- nDCG@10 (normalized discounted cumulative gain), recall@5 and recall@10, and MRR (mean reciprocal rank).
- First-stage recall@N, reported alongside, to show the ceiling.

### Offline: statistics

Compare per query: same queries, two systems. Use a paired bootstrap or permutation test with 95% confidence intervals.

A worked example of detectability: if the per-query difference in nDCG@10 between the two systems has a standard deviation of about 0.25, then on 1,000 queries the standard error of the mean difference is `0.25 / sqrt(1000)` ≈ 0.008, so the 95% interval is about ±1.6 points. A gain of 2-3 points is detectable; a gain of 1 point is not.

Report the win/tie/loss distribution per query, not only the mean. A reranker that helps 30% of queries and hurts 10% needs a look at the losers.

### Offline: end to end

Run the full RAG pipeline with and without the reranker, holding the LLM, the prompt and k fixed. Measure answer correctness, faithfulness and citation precision with an LLM judge calibrated against human labels.

Also run a variant with the reranker and a smaller k (for example, 5 instead of 10). A common win is the same answer quality at half the tokens.

### Offline: slices and ablations

- Slice by query type, length, language, document type, and whether gold was already at rank 1 (where there is no headroom).
- Watch exact-match and ID queries in particular: cross-encoders sometimes demote lexically exact hits.
- Ablate to confirm the gain is not just from reranking deeper (did N change too?) or from an input-formatting change.

### Online

- Run an A/B test with a pre-registered primary metric, such as answer acceptance rate or the share of conversations resolved without escalation.
- Set guardrails: p95 and p99 latency, timeout rate, cost per query and abstention rate.
- Do a power analysis up front. Small quality changes in noisy implicit-feedback metrics need large samples, so compute the minimum detectable effect before launching.
- Consider interleaving for comparing ranking quality: it mixes both rankings into one result list per user and is far more sensitive per query than an A/B test on click metrics.
- Monitor the reranker's score distribution for drift, and fall back to the first-stage order if the reranker exceeds its time budget, so tail latency stays protected.

### Cost and latency accounting

Present the reranker's cost as net cost: reranker cost minus the generation tokens it saves.

A worked example for a self-hosted reranker: one GPU at roughly $1-2 per hour, fully utilized at 20-50 QPS on 100-candidate reranks, costs between `$1 / (50 × 3,600)` and `$2 / (20 × 3,600)` per query, which is about $0.006-0.03 per 1,000 queries.

On the savings side, if the reranker lets you drop k from 20 to 8 chunks of about 500 tokens, that is about 6,000 fewer input tokens per query. At an illustrative $1 per million input tokens, that saves about $6 per 1,000 queries, two to three orders of magnitude more than the reranker costs. Even at much cheaper generation prices, the reranker is usually net-negative cost when it lets you shrink k.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| No reranker, tune the first stage | Lowest latency, simplest | Lower precision at small k | Very tight latency, or first-stage nDCG already high |
| Small cross-encoder (~20-100M parameters) | 20-60 ms, big precision gain | Often a 512-token input limit, generic domain | Default production choice |
| Large cross-encoder or hosted API | Higher quality, multilingual | 100-300 ms, per-call cost, data egress | Quality-critical, moderate QPS |
| LLM listwise reranking | Best zero-shot, handles reasoning | 10-100× cost, seconds of latency | Offline labeling and distillation, low-QPS high-value queries |
| ColBERT as reranker | Cheap, strong | Index storage, infrastructure | Already running late interaction |

- Depth N trades against latency: reranking cost is linear in N, and gains usually flatten past 100-200 candidates.
- Offline nDCG does not always translate end to end. A strong LLM may already cope with a slightly noisy top 10, so measure both levels.

## Common mistakes

- **Reporting only the mean nDCG gain** with no paired significance test, slices or per-query win/loss analysis. Noise and slice regressions ship as "wins".
- **Ignoring first-stage recall@N.** A reranker cannot recover documents it never sees, so effort goes to the wrong stage.
- **Pooling bias.** Judging only the baseline's results understates the reranker's gains; pooling only the reranker's results overstates them.
- **Silent truncation.** Long chunks cut at the model's input limit drop the relevant span, and the reranker scores the wrong text.
- **Forgetting the latency tail and the failure mode.** Without a timeout fallback and known p99 behavior under load, the reranker becomes an outage source.
- **Declaring victory on retrieval metrics alone**, without an end-to-end answer-quality check or an accounting of the token savings that might make the reranker net cheaper.

## Thinking at scale

- Write the ship rule before the experiment. For example: ship if end-to-end correctness gains at least 2 points with a confidence interval excluding 0, no important slice regresses by more than 1 point, and p95 latency rises by no more than 150 ms. Pre-registering the rule prevents post-hoc rationalization.
- Treat the reranker as the cheapest place to inject domain knowledge. It can be fine-tuned and redeployed without re-indexing the corpus, unlike the embedder. In a system like the one in [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/), that makes it the natural first target for domain adaptation.

## Further reading

- [Passage Re-ranking with BERT (Nogueira & Cho, 2019)](https://arxiv.org/abs/1901.04085): the canonical cross-encoder reranker on MS MARCO.
- [Document Ranking with a Pretrained Sequence-to-Sequence Model (Nogueira et al., 2020)](https://arxiv.org/abs/2003.06713): monoT5 reranking.
- [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents (Sun et al., 2023)](https://arxiv.org/abs/2304.09542): RankGPT listwise LLM reranking and distillation into smaller models.
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (Khattab & Zaharia, 2020)](https://arxiv.org/abs/2004.12832): late interaction as an efficient reranking alternative.
- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models (Thakur et al., 2021)](https://arxiv.org/abs/2104.08663): cross-encoder reranking results across domains.
- [Contextual Retrieval in AI Systems (Anthropic, 2024)](https://www.anthropic.com/engineering/contextual-retrieval): measured incremental gain from adding reranking (failure reduction from 49% to 67%, reranking the top 150 to 20).
- [LegalBench-RAG (Pipitone & Alami, 2024)](https://arxiv.org/abs/2408.10343): a case where a general-purpose reranker hurt retrieval in a specialized domain.
- [Rerank overview (Cohere docs)](https://docs.cohere.com/docs/rerank-overview): a hosted reranker API with practical usage guidance.
