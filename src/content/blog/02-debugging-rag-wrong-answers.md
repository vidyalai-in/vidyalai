---
title: "Debugging a RAG System That Gives Wrong Answers"
description: "A stage-by-stage method for localizing RAG failures: build a replayable failure set, attribute each case to a stage, fix the biggest bucket, and gate on it."
slug: "debugging-rag-wrong-answers"
category: "retrieval-rag"
tags: ["rag", "evals", "llm-as-judge", "observability", "hallucination"]
seriesOrder: 2
pubDate: 2026-10-16
sourceDoc: "1h3m8wtZykYFchMjt8t57Aqya0AhrdYy0tlem7E4EI3A"
video: { youtube: "TLINef2z4F8", duration: "10:42", url: "https://youtu.be/TLINef2z4F8?si=Zq_OAVxCppKewd5P" }
---

## TL;DR

"Wrong answers" is a symptom, not a diagnosis, so the first job is to localize each failure to a pipeline stage. Collect a set of failing cases, trace each one through the pipeline, and bucket it: the answer is not in the corpus; it is in the corpus but was not retrieved; it was retrieved but ranked out of the context; it was in the context but the model ignored or misread it; or the question was ambiguous or the expected answer is wrong. The bucket counts tell you where to spend effort, and more often than not the biggest buckets are retrieval and parsing rather than the LLM (large language model). Fix the dominant bucket, add those cases to a regression set, and gate future changes on it.

## Requirements to pin down first

- **What does "wrong" mean?** Factually incorrect, incomplete, outdated, a hallucinated citation, or an unwanted refusal each point to different stages. Ask who reported it and what the base rate is: three anecdotes and 8% of traffic call for very different responses.
- **Did something change recently?** A new embedding model, chunker, index rebuild, prompt, LLM version or corpus update. Sudden regressions are almost always a deploy or a data change, and a diff of the change log is the fastest diagnostic you have.
- **Is the failure concentrated?** Look for clustering by query type, tenant, language, document type or time range. A concentrated failure usually has a single cause, such as one broken parser or one stale shard.

## Step 1: Build a replayable failure set

Pull 50-200 failing examples from logs: thumbs-down ratings, escalations, and a random sample of traffic graded by an LLM judge. The random sample matters because explicit complaints are biased toward certain query types.

For every request, the system must log enough to replay it exactly:

- the original and rewritten query,
- the retrieved chunk IDs with scores at every stage (each retriever, fusion, reranker),
- the final prompt as sent,
- the model and prompt versions,
- the output.

**If you cannot replay a request exactly, fixing observability is step zero.** Every later step depends on it.

## Step 2: Attribute each failure to a stage

This is the core move. For each case, have an SME (subject-matter expert), or a strong model followed by human spot-checks, find the gold evidence chunk(s). Then walk downstream and find the first stage where the gold evidence disappears.

![Decision flowchart for one failing case. Starting from the failing case with its gold evidence, the checks run in order: is gold in the raw corpus, intact after parsing, whole within one chunk, indexed and current with the correct ACL, in the top 200 retrieval candidates, and in the final context. A "no" at any check sends the case to the failure histogram under that stage's bucket. If gold reached the context, a final check asks whether the expected answer is valid: if not, it is a label bucket; if so, a generation bucket. The histogram then drives fixing the largest bucket and adding the cases to a regression suite.](/diagrams/q02-failure-attribution.svg "Walk each failing case downstream; the first stage where gold evidence disappears is the root cause.")

Each failing case lands in exactly one bucket, and the histogram of buckets, not the most memorable anecdote, decides what to fix first.

| Check | Question | Typical fix |
|---|---|---|
| Corpus coverage | Does the gold evidence exist in the raw source? | Content gap, or ingestion missed the source |
| Parsing | Is it intact in the parsed text? (tables flattened, OCR garbage, headers dropped) | Better parser, table handling |
| Chunking | Is the evidence split across chunks, or separated from the context that disambiguates it? | Structure-aware chunks, overlap, context headers |
| Index | Is the chunk in the index, current, with the right ACL and metadata? | Fix the pipeline, stale index, filter bugs |
| Retrieval | Is gold in the top 200 candidates? In the top k after fusion? | Hybrid search, query rewriting, embedder fine-tuning |
| Ranking | Is gold in the final context window? | Reranker, k, dedupe |
| Generation | Gold is in context, but the answer is wrong | Prompt, ordering, model choice, less noise |
| Eval / label | Is the "expected" answer actually wrong or ambiguous? | Fix the golden set |

Build a histogram of the buckets. An illustrative distribution, not a published benchmark, might look like this:

| Bucket | Share of failures |
|---|---|
| Corpus or parsing | 10-20% |
| Retrieval or ranking | 30-40% |
| Generation | 20-30% |
| Ambiguous question or bad label | 10-20% |

Your numbers will differ, which is the point of measuring. The last bucket matters: you cannot fix what was never broken, and leaving it in inflates the apparent error rate.

Budget the time honestly. Manual attribution takes roughly 2-5 minutes per case, so 200 cases is about 7-17 hours of SME time. That is why it pays to build the taxonomy manually once, then scale it with a calibrated LLM judge.

## Step 3: Targeted diagnostics per bucket

### Retrieval misses

Compare the production retriever against BM25 (keyword search) and against exact, brute-force kNN (k-nearest neighbors) search.

- **Exact kNN finds gold, ANN does not.** The problem is the approximate index: parameters such as `efSearch` (HNSW) or `nprobe` (IVF) are too low, or recall collapsed under a selective metadata filter.
- **Neither finds gold.** The problem is the representation. Common causes are vocabulary mismatch (acronyms, IDs and part numbers favor BM25), a query phrased very differently from the document (try HyDE, which embeds a hypothetical answer instead of the question, or query rewriting), or a domain far from the embedder's training data.

For when each retriever wins, see [BM25 vs Dense vs Hybrid Search](/blog/bm25-dense-hybrid-search/).

### Ranking misses

Gold is at rank 40 and the context takes k = 10. Add or tune a cross-encoder reranker, and check that the reranker sees each chunk *with its title and section header*. [Adding a Reranker and Proving It Helped](/blog/reranker-evaluation/) covers the measurement side.

### Generation failures

Run an ablation: give the model *only* the gold chunk.

- **It now answers correctly.** The problem is distraction from irrelevant context. Reduce k, rerank harder, and put the best evidence first.
- **It still fails.** The problem is reasoning or instruction following. Improve the prompt, require the model to quote the evidence before answering, use a stronger model, or decompose multi-hop questions.

### Stale or conflicting documents

The model picked an old version of a policy. Add date metadata, recency boosting and near-duplicate removal. Instruct the model to prefer the newest source and to flag conflicts it sees.

### Hallucinated citations

Validate that every cited ID was actually in the context. Then verify each claim against its cited span with an NLI (natural language inference) model or an LLM judge before returning the answer.

## Step 4: Fix, measure, prevent

- **Fix the largest bucket first, then re-run the whole golden set**, not just the failures. Changes such as a smaller k or a new chunker routinely regress other slices.
- **Report per-stage metrics**: retrieval recall@k, context precision, faithfulness, answer correctness and abstention rate.
- **Add every root-caused failure to a regression suite** with its stage label, and gate deploys on it.
- **Monitor online signals** that drift before users complain: retrieval score distributions, the empty-result rate, the abstention rate, and the share of answers with zero valid citations.

## How to measure it

Use paired comparisons on the same queries (old system vs new system) and report bootstrap confidence intervals.

Small eval sets are noisy. A worked example: on 300 queries with accuracy around 70%, the standard error of a single accuracy estimate is `sqrt(0.7 × 0.3 / 300)` ≈ 2.6 points. Pairing helps, but if 10% of queries flip between the two systems, the 95% interval on the difference is still about ±3.6 points. A 2-point change on 300 queries is therefore usually noise.

Calibrate any LLM judge against human labels before trusting it, with a target of at least 85% agreement, and re-check agreement after every prompt change. Optimizing against an uncalibrated judge is a Goodhart trap: the metric goes up while real quality does not.

Keep faithfulness and correctness as separate metrics. A faithful answer grounded in an outdated document is still wrong.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Stage-wise error attribution (manual, SME) | Highest-signal root cause | Slow, ~2-5 min per case | First pass, and any new failure class |
| LLM-judge attribution at scale | Covers thousands of cases cheaply | Judge errors, needs calibration | After the taxonomy is stable |
| Swap to a bigger LLM | Quick, fixes some reasoning failures | Does not fix retrieval; adds cost and latency | Only when the ablation shows gold-in-context failures |
| Tune retrieval (hybrid, rerank, fine-tune) | Fixes the most common bucket | Re-index cost, possible regressions | Recall@k on gold is low |
| Add a verification / abstention layer | Cuts hallucination | Added latency, more refusals | High-stakes domains |

- **Do not touch the prompt or the model until you know gold made it into the context.** Many teams invert this order.
- **Raising k trades precision and cost for recall.** Measure it rather than assuming it helps.

## Common mistakes

- **Starting with prompt tweaks or a model upgrade** before checking whether retrieval found the evidence. You spend effort on the stage that is often not the problem.
- **Debugging from three anecdotes** with no failure set, no traces and no base rate. Fixes target the memorable case, not the common one.
- **Re-testing only the failing cases.** Regressions elsewhere ship unnoticed; always re-run the full stratified suite.
- **Ignoring the "label is wrong / question is ambiguous" bucket.** It inflates the apparent error rate and sends work in the wrong direction.
- **Missing filtered-ANN recall collapse, or a silent version mismatch** between the query encoder and the document encoder. Both look like "the model got worse" while the index is actually broken.
- **Treating faithfulness and correctness as the same metric.** Answers faithfully grounded in stale documents pass the check and still mislead users.

## Thinking at scale

- **Build tooling, not heroics.** A trace viewer with stage-by-stage "gold hit" indicators lets anyone on the team triage a failure in minutes. Debugging RAG is a recurring workflow.
- **Guard the judge.** Re-calibrate LLM judges against humans on a schedule and after any prompt or model change, so the team does not optimize against a drifting metric.
- **Escalate systemic issues to their owners.** If 15% of failures are content gaps, that is a product and data-ownership conversation, not a modeling problem.
- **Design for attribution up front.** The system architecture in [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/) separates ingestion, retrieval and generation evals for exactly this reason.

## Further reading

- [Seven Failure Points When Engineering a Retrieval Augmented Generation System (Barnett et al., 2024)](https://arxiv.org/abs/2401.05856): a failure taxonomy that maps onto the stage checks in this post.
- [Ragas: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)](https://arxiv.org/abs/2309.15217): faithfulness, context precision and context recall metrics.
- [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems (Saad-Falcon et al., 2023)](https://arxiv.org/abs/2311.09476): LLM judges with statistical calibration against human labels.
- [Large Language Models Can Be Easily Distracted by Irrelevant Context (Shi et al., 2023)](https://arxiv.org/abs/2302.00093): why more context can produce worse answers.
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172): position effects when gold is in context but ignored.
- [Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022)](https://arxiv.org/abs/2212.10496): HyDE, a query-side fix for phrasing mismatch.
- [Contextual Retrieval in AI Systems (Anthropic, 2024)](https://www.anthropic.com/engineering/contextual-retrieval): fixes for chunks that lose their context, with measured retrieval-failure reductions.
