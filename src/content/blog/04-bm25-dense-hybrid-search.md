---
title: "BM25 vs Dense vs Hybrid Search: When to Use Each"
description: "How lexical, dense, learned-sparse and late-interaction retrieval fail differently, how to fuse them with RRF, and how to choose on your own traffic."
slug: "bm25-dense-hybrid-search"
category: "retrieval-rag"
tags: ["retrieval", "vector-search", "rag", "reranking", "evals"]
seriesOrder: 4
pubDate: 2026-01-07
sourceDoc: "1h3m8wtZykYFchMjt8t57Aqya0AhrdYy0tlem7E4EI3A"
video: { youtube: "Ayvmk_rm1fc", duration: "11:56", url: "https://youtu.be/Ayvmk_rm1fc?si=3oqueIgg-vuVa1x1" }
---

## TL;DR

BM25 wins on exact tokens: identifiers, rare terms, names and error codes. It needs no training and holds up well on unfamiliar domains. Dense retrieval wins on paraphrase and semantic matching, when queries and documents use different words, especially in-domain or after fine-tuning. Their errors are largely uncorrelated, so for almost any production system the default is hybrid: run both retrievers, fuse the lists with RRF (reciprocal rank fusion), and rerank. Still, the way to decide is to run all three on a labeled set drawn from your own traffic and choose by recall@k per query slice, because rankings flip across domains: the BEIR benchmark showed BM25 beating many dense models zero-shot.

## Requirements to pin down first

- **What do the queries look like?** Keyword-style (2-3 tokens on average, IDs, SKUs) or natural-language questions? The more queries look like identifiers, the more weight BM25 deserves.
- **How specialized is the vocabulary?** Biomedical, legal, code or internal acronyms all stress off-the-shelf embedders. Is there in-domain training data, such as click logs or labeled pairs, to fine-tune a dense model?
- **Is the corpus multilingual?** Cross-lingual retrieval is where dense models have a structural advantage over lexical matching.
- **What are the latency and cost constraints?** A second index is cheap in milliseconds but not free in operations.
- **Is there an existing search stack** (Elasticsearch, OpenSearch, Vespa)? If so, adding BM25 or vectors next to what exists is often nearly free.

## How each retriever works and where it breaks

### BM25 (sparse lexical)

BM25 scores a document by term frequency with saturation (controlled by `k1`, commonly 1.2), length normalization (`b`, commonly 0.75), and IDF (inverse document frequency), which rewards rare terms.

It gives exact match and rare-term precision, its scoring is transparent, it is cheap (an inverted index over 100M documents is routine), it needs zero training, and it is robust to domain shift. Its weaknesses are vocabulary mismatch ("car" vs "automobile"), no understanding of negation or intent, and weak results on long natural-language questions.

### Dense retrieval (bi-encoder)

A bi-encoder embeds the query and each document independently, then ranks by cosine or dot-product similarity over an ANN (approximate nearest neighbor) index.

Dense retrieval handles synonyms, paraphrase, cross-lingual matching and conceptual queries. Its weaknesses: exact rare tokens get blurred (part numbers, names, codes), quality degrades out of domain, failures are opaque, vectors cost compute and storage, and every model change means re-embedding the corpus. Off-the-shelf models can underperform BM25 on specialized corpora until fine-tuned.

The picture has shifted since BEIR (2021). Embedders trained on large, diverse weakly-supervised data, starting with E5 in 2022, beat BM25 on the BEIR average zero-shot, and that has become the norm for strong general-purpose embedders. BM25 still wins on individual datasets and on identifier-heavy queries, so the average alone does not settle the choice.

### Learned sparse retrieval (SPLADE and similar)

A model predicts term weights and expansion terms, and the result is served from an ordinary inverted index. It often approaches dense quality in-domain while keeping BM25-like interpretability and exact-match behavior.

### Late interaction (ColBERT)

ColBERT keeps one vector per token and matches query tokens against document tokens at search time. Quality and out-of-domain behavior are strong, but storing a vector per token costs an order of magnitude or more storage than one vector per passage when uncompressed. ColBERTv2's residual compression reduces the footprint 6-10×, and the PLAID engine brings search latency down to tens of milliseconds on a GPU.

## Hybrid retrieval done right

![Flowchart: a query goes to an optional query analyzer that can set per-retriever weights, then fans out in parallel to BM25 top 100 and dense ANN top 100. Both lists feed RRF fusion with k equal to 60. The fused top 100-200 candidates go to a cross-encoder reranker, and the reranked top k passages go to the LLM.](/diagrams/q04-hybrid-retrieval.svg "Hybrid retrieval: two retrievers in parallel, rank-based fusion, then a cross-encoder reranker before the LLM.")

The two retrievers run in parallel, so hybrid adds an index to operate but very little latency.

### 1. Retrieve in parallel

Run BM25 top 100 and dense top 100 concurrently. Combined latency is about the maximum of the two, typically 10-50 ms.

### 2. Fuse with RRF

Reciprocal rank fusion sums, for each document, the reciprocal of its rank in each list, offset by a constant:

```text
score(d) = Σ_i 1 / (k + rank_i(d)),   k ≈ 60, ranks start at 1
```

A worked example with k = 60: a document ranked 1st by BM25 and 5th by dense scores `1/61 + 1/65 ≈ 0.0318`. A document ranked 2nd by dense only scores `1/62 ≈ 0.0161`. Agreement between retrievers beats a single high rank.

RRF uses ranks, not raw scores, so it needs no score normalization, which makes it the right default. A weighted linear combination of normalized scores can beat RRF if you tune the weight α on labeled data, but it is brittle when score distributions drift after a model change.

### 3. Rerank

Rerank the fused top 100-200 with a cross-encoder, which scores each (query, passage) pair jointly. [Adding a Reranker and Proving It Helped](/blog/reranker-evaluation/) covers sizing and evaluation.

### 4. Optionally route or weight by query type

Queries that look like identifiers (regex matches for codes, quoted strings) get more BM25 weight. Long natural-language questions get more dense weight.

### Evidence that the gains stack

Anthropic's contextual retrieval experiments show the additive pattern. Measured as the top-20 retrieval failure rate, the baseline of 5.7% fell to 3.7% with contextual embeddings (35% fewer failures), to 2.9% when contextual BM25 was added (49%), and to 1.9% when reranking was added on top (67%). In that setup, reranking took the top 150 fused candidates down to 20.

## Choosing: a decision guide

### BM25 only

- Cold start with no labeled data.
- Heavily keyword or identifier queries: log search, code symbols, legal citations, e-commerce SKUs.
- Strict cost or latency limits, or a requirement for explainable ranking.
- Always as the baseline. If your dense system does not beat BM25 on your eval set, stop and investigate before shipping it.

### Dense only

- Short-text semantic matching: FAQ deduplication, support-ticket routing.
- Cross-lingual retrieval.
- Conversational queries with low lexical overlap, over corpora with little rare vocabulary.

Even here, prove on your eval set that the BM25 component adds nothing before dropping it.

### Hybrid

- The default for general enterprise RAG (retrieval-augmented generation), mixed query populations, and anywhere recall matters more than a few milliseconds.
- The added cost is running an inverted index next to the vector index, which is often free if you already run Elasticsearch, OpenSearch or Vespa.

For how hybrid fits into a full system, see [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/).

## How to measure it

- Build the eval set by sampling 500-2,000 queries from real traffic, stratified by type: navigational/ID, keyword, natural-language, multi-hop. Use pooled judgments: label the union of the top 20 from every system under test, so the eval is not biased toward the incumbent.
- For the first stage, report recall@100 and recall@50. The reranker can only reorder what the first stage gives it.
- For the final list, report nDCG@10 (normalized discounted cumulative gain) and MRR (mean reciprocal rank) after reranking.
- Always report metrics per slice. The hybrid gain usually concentrates in the slice where one retriever collapses, and the average hides it.
- Compare the overlap between the BM25 and dense top-50 sets. As a rule of thumb, if they overlap 80% or more, hybrid adds little; at 30-50% overlap, fusion typically adds several points of recall.
- For significance, use a paired bootstrap or permutation test over queries, and report confidence intervals.
- Online, track click-through on citations, answer acceptance and query reformulation rate.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| BM25 | Exact match, no training, cheap, interpretable, robust out of domain | Vocabulary mismatch, weak on natural-language questions | Keyword/ID queries, cold start, baseline |
| Dense bi-encoder | Semantic and paraphrase matching, multilingual | Misses rare tokens, domain shift, re-embedding cost | Conversational queries, cross-lingual, fine-tuned domain |
| Learned sparse (SPLADE) | Semantic expansion on an inverted index, exact match retained | Larger index than BM25, model inference at ingest | Dense-like quality on lexical infrastructure |
| ColBERT late interaction | Top quality, good out of domain | Storage and serving complexity | Quality-critical, moderate corpus size |
| Hybrid (RRF) + reranker | Best recall across slices, robust | Two indexes, more operations | Default for production RAG |

- RRF is robust with zero tuning. A tuned linear blend can win by a point or a few, but needs labels and re-tuning whenever a model changes.
- Judge a first-stage retriever on recall, not nDCG@10 alone. Its job is recall at the reranker's depth.

## Common mistakes

- **Claiming dense is strictly better than BM25.** BEIR showed BM25 is a hard zero-shot baseline, and BM25 still wins on identifier-heavy slices even against modern embedders. Dropping it loses those queries.
- **Adding raw BM25 scores to cosine similarities.** The scales are incompatible, so one retriever silently dominates. Use RRF or calibrated normalization.
- **Evaluating on a public benchmark instead of your own query distribution.** You optimize for someone else's traffic.
- **Pooling bias.** Labeling only the incumbent's results means the new retriever's finds count as "non-relevant", so better systems look worse.
- **Ignoring tokenization and analyzers for BM25** (stemming, code identifiers, CJK languages). Analyzer choice can matter as much as the algorithm.
- **Inconsistent ACLs and deletes across the two indexes.** A document deleted or restricted in one index leaks through the other.

## Thinking at scale

- The choice is not permanent. Instrument per-slice metrics so that traffic drift (for example, more ID lookups after a product launch) shows up as a measurable retriever mismatch rather than a vague quality complaint.
- Use each method to improve the other. One of the biggest levers is in-domain fine-tuning of the dense model with hard negatives mined from BM25.
- Hybrid means two indexes, which doubles the freshness pipelines and failure modes. Assign ownership for keeping deletes, ACLs and freshness consistent across both.

## Further reading

- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models (Thakur et al., 2021)](https://arxiv.org/abs/2104.08663): BM25 vs dense and other architectures out of domain.
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training (Wang et al., 2022)](https://arxiv.org/abs/2212.03533): E5, the first embedder reported to beat BM25 on BEIR zero-shot without labeled data.
- [Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906): the canonical dense bi-encoder vs BM25 result.
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking (Formal et al., 2021)](https://arxiv.org/abs/2107.05720): learned sparse retrieval.
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (Khattab & Zaharia, 2020)](https://arxiv.org/abs/2004.12832): late interaction.
- [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction (Santhanam et al., 2021)](https://arxiv.org/abs/2112.01488): residual compression that cuts late-interaction storage 6-10×.
- [PLAID: An Efficient Engine for Late Interaction Retrieval (Santhanam et al., 2022)](https://arxiv.org/abs/2205.09707): the serving engine that makes ColBERTv2 fast at scale.
- [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods (Cormack et al., 2009)](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf): the original RRF paper and the origin of k = 60.
- [Reciprocal rank fusion (Elasticsearch reference)](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion): RRF in a production search engine, with `rank_constant` defaulting to 60.
- [Contextual Retrieval in AI Systems (Anthropic, 2024)](https://www.anthropic.com/engineering/contextual-retrieval): measured additive gains from contextual embeddings, contextual BM25 and reranking.
