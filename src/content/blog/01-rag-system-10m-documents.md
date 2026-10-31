---
title: "Designing a RAG System for 10 Million Documents"
description: "How to size, build and measure retrieval-augmented generation over 10M documents: ingestion, hybrid retrieval, reranking, ACLs, cost and evals."
slug: "rag-system-10m-documents"
category: "retrieval-rag"
tags: ["rag", "vector-search", "retrieval", "evals", "architecture"]
seriesOrder: 1
pubDate: 2026-01-01
sourceDoc: "1h3m8wtZykYFchMjt8t57Aqya0AhrdYy0tlem7E4EI3A"
video: { youtube: "_ric0wnAKrE", duration: "11:06", url: "https://youtu.be/_ric0wnAKrE?si=EA90nmjjQ4MtQbOJ" }
---

## TL;DR

Treat a RAG (retrieval-augmented generation) system as a retrieval system first and a generation system second: most failures originate in retrieval, and retrieval is also the cheapest part to measure. The offline side parses, chunks (roughly 30-100M chunks for 10M documents), enriches and embeds documents into a sharded ANN (approximate nearest neighbor) index plus a BM25 keyword index. The online side does query understanding, hybrid retrieval of about 200 candidates, cross-encoder reranking down to 10-20 passages, and grounded generation with citations. Access control and freshness are enforced inside the index, never patched on afterwards. Before choosing any component, pin down the SLOs (service-level objectives: p95 latency, cost per query, freshness, recall@k) and build a labeled eval set, because every later decision is judged against those numbers.

## Requirements to pin down first

The answers to these questions change the architecture as well as the tuning.

- **What are the documents?** Average length, formats (PDF, HTML, scanned images), languages, and how much content is tables or code. 10M documents of about 5 pages each is roughly 25B tokens (at about 500 tokens per page). That is a very different system from 10M tweets.
- **What is the query mix?** Factoid lookup, multi-hop synthesis, or "find all X"? Exhaustive-recall queries need different machinery (filters, aggregation, agentic search) than top-k question answering.
- **What are the QPS and latency SLOs?** A workable starting point is 50 queries per second (QPS) at peak, p95 under 3 s end to end, and TTFT (time to first token) under 1.5 s. Those numbers set the budget for every online stage.
- **How fresh must results be?** An update rate of 1% per day is 100k documents per day (about 1.2 per second). Whether a new document must be searchable within minutes or within hours decides between streaming upserts and batch rebuilds.
- **What is the access-control model?** Per-user or per-tenant ACLs (access control lists) decide whether filtering happens before or during ANN search, or after it. Getting this wrong is the most common way RAG systems leak data.
- **What are the cost and hosting constraints?** VPC-only deployment, which model providers are allowed, and data residency rules can eliminate whole classes of managed services.

## The ingestion pipeline

![Two-part flowchart. Offline ingestion: 10M source documents go through layout-aware parsing, structure-aware chunking into 30-100M chunks, and enrichment with metadata, ACLs and context; enriched chunks feed both a versioned embedding step that builds a sharded, filter-aware ANN index and a BM25 sparse index. Online query path: a user query goes through rewrite and routing, hybrid retrieval with RRF fusion and ACL filtering over both indexes returning about 200 candidates, cross-encoder reranking to 10-20, context assembly, and grounded generation with citations.](/diagrams/q01-rag-pipeline.svg "Offline ingestion builds two indexes; the online path queries both, reranks, and generates with citations.")

Both indexes are built once offline and queried together online, so ACLs, metadata and freshness must be consistent across them.

### 1. Parse

Use layout-aware parsing: PDFs become structured blocks, scans go through OCR, and tables are kept as Markdown or HTML rather than flattened into text. Parsing quality sets a ceiling on everything downstream, and that ceiling is easy to overlook. Sample about 200 documents and read the parsed output before tuning anything else.

### 2. Chunk

Chunk along document structure, aiming for roughly 300-800 tokens per chunk, with headings carried as metadata. Store a parent pointer on every chunk so you can expand "small-to-big" at read time: retrieve on the small chunk, hand the model its larger parent section when needed.

Sizing check: 25B tokens over 10M documents is about 2,500 tokens per document. At 300-800 tokens per chunk that is about 3-8 chunks per document, or 30-80M chunks. Plan capacity for 100M to leave headroom for overlap, context headers and corpus growth. The rest of this post uses 100M as the sizing number.

For a domain-specific example of structure-aware chunking, see [Chunking Legal Contracts for Retrieval](/blog/chunking-legal-contracts/).

### 3. Enrich

Attach the metadata that retrieval and filtering will need: title, section path, document date, ACL principals, tenant, language and document type.

Optionally, prepend a short LLM-generated context sentence to each chunk before embedding and BM25 indexing. Anthropic calls this contextual retrieval. In its experiments, the top-20 retrieval failure rate fell from 5.7% to 3.7% with contextual embeddings alone (a 35% reduction), to 2.9% with contextual embeddings plus contextual BM25 (49%), and to 1.9% when reranking was added (67%). Anthropic put the one-time cost at $1.02 per million document tokens using Claude 3 Haiku with prompt caching, assuming 8k-token documents and 800-token chunks. At that rate, 25B tokens would cost about $25k once. Re-price it with whatever model you use today, but the order of magnitude makes it affordable at this scale.

### 4. Embed

Start with an off-the-shelf model shortlisted from the MTEB (Massive Text Embedding Benchmark) leaderboard, evaluated on your own queries, and fine-tune later on in-domain pairs.

Storage sizing for 100M chunks. An embedding model pools all of a chunk's tokens into one vector per chunk, so chunk length doesn't change these numbers. (Late-interaction models such as ColBERT keep a vector per token instead, which multiplies storage by the tokens per chunk.)

| Representation | Bytes per vector | Total for 100M |
|---|---|---|
| 1024 dims, fp32 | 4,096 | ~410 GB |
| 512 dims (Matryoshka truncation), int8 | 512 | ~51 GB |
| 1024-bit binary code | 128 | ~13 GB |
| 512-bit binary or 64-byte PQ code | 64 | ~6.4 GB |

Matryoshka embeddings are trained so that a prefix of the vector is itself a usable embedding, which makes truncation safe. Binary and PQ (product quantization) codes trade recall for size, so pair them with a rescoring pass over higher-precision vectors (int8 or fp16) for the top candidates. Those rescoring vectors usually live on SSD rather than in RAM.

Embedding throughput is a function of model size. For example, embedding 25B tokens in one day needs about 290k tokens per second (25B / 86,400 s). A 7B-parameter, LLM-based embedder might sustain on the order of 5-10k tokens per second per data-center GPU, which means roughly 30-60 GPUs for a day. A ~300M-parameter encoder is an order of magnitude cheaper per token and could do the same job on a handful of GPUs. Benchmark your chosen model on your hardware before committing to a schedule.

Budget for re-embedding the whole corpus every time the embedding model changes, and version every vector with its model ID.

### 5. Index

Two families cover most needs:

- **HNSW** (Hierarchical Navigable Small World graphs): in memory, with the best recall/latency tradeoff.
- **IVF-PQ** (inverted file with product quantization) or **DiskANN**: SSD-resident, much cheaper per vector.

Shard by hash or by tenant. Run a BM25 or SPLADE (learned sparse) inverted index next to the vector index, for example in OpenSearch, Elasticsearch or Vespa. Keep metadata filters in the same engine as the vectors so that filtering happens *during* ANN traversal, not after it.

### 6. Incremental updates

Use a CDC (change data capture) stream: re-parse, re-chunk and upsert changed documents. Tombstone deletions immediately, because deletes are a compliance issue as well as a relevance one. Run periodic compaction and rebuilds, since HNSW recall degrades under heavy deletes.

## The query path

### Latency budget for about 2.5 s p95

| Stage | Budget |
|---|---|
| Query rewrite / routing (small LLM or rules, optional) | 100-300 ms |
| Hybrid retrieval: ANN top 100 + BM25 top 100, fused with RRF, ACL-filtered | 30-80 ms |
| Cross-encoder rerank 200 → 20 (batched on GPU) | 80-200 ms |
| Context assembly (dedupe, parent expansion, ordering) | under 20 ms |
| Generation (~8-15k token context, streaming) | TTFT 400-800 ms, total 1-2 s |

The worst case of every pre-generation stage plus the worst TTFT is 300 + 80 + 200 + 20 + 800 = 1,400 ms, which fits the 1.5 s TTFT target with little slack. Per-stage p95s do not add linearly, so measure the end-to-end distribution rather than summing the table.

RRF (reciprocal rank fusion) merges ranked lists by rank position rather than raw score; [BM25 vs Dense vs Hybrid Search](/blog/bm25-dense-hybrid-search/) covers it in depth.

### Query understanding

Decompose multi-hop questions into sub-queries. Extract filters such as dates and entities. Route out-of-scope or chit-chat queries away from retrieval entirely.

### Reranking

A cross-encoder scores each (query, passage) pair jointly and is far more precise than the first-stage retrievers. It is also where "fewer, better chunks" comes from, which drives generation cost. [Adding a Reranker and Proving It Helped](/blog/reranker-evaluation/) covers how to size and evaluate it.

### Generation

Instruct the model to answer only from the provided passages, cite chunk IDs, and abstain when the evidence is missing. Put the strongest passages at the start and end of the context, because models attend less to the middle of long contexts (the "lost in the middle" effect). Use prompt caching for the static system prompt.

### Caching

Add a semantic cache for frequent queries and an embedding cache for query vectors. Hit rates depend heavily on the workload; enterprise assistants with repetitive questions can see something like 10-30%, but measure yours before counting on it.

## Cost back-of-envelope

At 50 QPS sustained around the clock, the system would serve 50 × 86,400 ≈ 4.3M queries per day. At about 12k input tokens per query, that is about 52B input tokens per day. Because 50 QPS is a peak figure, treat this as an upper bound; average traffic is usually well below peak.

For example, at an illustrative $1 per million input tokens, 52B tokens is about $52k per day before output tokens. Retrieval infrastructure for about 100M vectors costs on the order of a few thousand dollars a month either way. Generation therefore dominates retrieval by far more than 10×.

To lower cost, send fewer, better chunks (this is where the reranker pays for itself), route easy queries to smaller models, and cache prompts, queries and answers.

## How to measure it

### Retrieval

- Track recall@20 and nDCG@10 (normalized discounted cumulative gain) on a golden set of 500-2,000 queries labeled by SMEs (subject-matter experts) and stratified by query type.
- Targets: recall@50 ≥ 0.95 before the reranker and recall@10 ≥ 0.85 after it.
- Track ANN recall against exact kNN (brute-force nearest neighbors) separately, with a target of ≥ 0.95, so that index tuning does not hide inside model changes.

### Generation

- Faithfulness or groundedness: are the claims supported by the cited chunks?
- Answer correctness, citation precision, and abstention accuracy on unanswerable queries.
- Use an LLM judge calibrated against about 200 human labels, and report its agreement rate alongside every score.

### Online

- Thumbs-up/down and escalation rate, query reformulation rate, and click-through on citations.
- A/B tests with guardrail metrics for latency and cost per query.

### Regression gate

Every change to the chunker, embedder, index parameters or prompt runs the offline suite in CI. When something regresses, [Debugging a RAG System That Gives Wrong Answers](/blog/debugging-rag-wrong-answers/) describes how to localize it to a stage.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| HNSW in memory | Best recall/latency (under 10 ms), mature | RAM-heavy: full vectors plus graph links in memory; weak with heavy deletes | Latency-critical work with a budget for RAM |
| IVF-PQ / DiskANN on SSD | 5-20× cheaper memory, scales to billions | Lower recall without rescoring, more tuning | Cost-sensitive, very large corpora |
| Managed vector DB (Pinecone, etc.) | Fast to ship, operations handled | Cost at 100M vectors, less control over hybrid | Small team, time to market |
| Search engine with vectors (Vespa, OpenSearch, Elasticsearch) | Hybrid, filters and BM25 in one system | Heavier operations | Hybrid plus complex filters or ACLs |
| Long-context stuffing, no retrieval | Simple | Impossible at 25B tokens | Only small per-user corpora |

On HNSW memory: Faiss documents HNSW memory as `d * 4 + M * 2 * 4` bytes per vector. For 1024-dim fp32 vectors with M = 32 links, the graph adds only about 6% on top of the raw vectors; the cost is that all of it must sit in RAM. For compressed 512-byte vectors, the same links add about 50%.

- Pre-filter vs post-filter ACLs: post-filtering the top-k on a selective ACL can return zero results and leaks timing information. Prefer filtered ANN or per-tenant shards.
- Chunk size against reranker cost: smaller chunks give better precision but more candidates to rerank. Small-to-big expansion gets both.
- Fine-tuning the embedder: in-domain fine-tuning often adds several points of recall (gains of 5-15 points are plausible but vary widely by domain), and it locks you into a re-embedding cycle.

## Common mistakes

- **Jumping to "vector DB plus an LLM" with no SLOs, no eval set and no capacity math** (vectors × dims × bytes). The result is a system you cannot size, budget or evaluate.
- **Ignoring ACLs and deletion.** Retrieval-time permission leaks and stale deleted content are the incidents that get escalated.
- **Reporting end-to-end answer quality without separating retrieval recall from generation faithfulness.** You cannot tell which stage to fix.
- **Forgetting that ANN recall degrades** with filters, deletes and index drift. Without monitoring against exact kNN on a sample, recall erodes silently.
- **No plan for re-embedding 100M chunks** when the embedding model changes. The migration becomes a multi-week outage risk instead of a routine job.
- **Unbounded context** ("just send the top 50"). It raises cost and latency and also hurts quality through distraction.

## Thinking at scale

- Draw ownership boundaries by stage: give ingestion, retrieval and generation separate evals and separate on-call. A quality drop can then be traced to a stage in minutes rather than days.
- Plan for embedding model migration from day one. Dual-write to old and new indexes and run shadow traffic before cutting over.
- Treat exhaustive questions as a different product. Requests like "list all contracts with clause X" belong in structured extraction into a table, queried with SQL. Do not stretch top-k RAG to cover them.

## Further reading

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401): the original RAG formulation.
- [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (Malkov & Yashunin, 2016)](https://arxiv.org/abs/1603.09320): the default in-memory ANN graph index.
- [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node (Subramanya et al., 2019)](https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html): SSD-resident graph indexing for corpora that do not fit in RAM.
- [The Faiss library (Douze et al., 2024)](https://arxiv.org/abs/2401.08281): IVF, PQ and the memory/recall tradeoffs at scale.
- [Faiss: Guidelines to choose an index (Meta, wiki)](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index): per-vector memory formulas for HNSW, PQ and binary codes.
- [Contextual Retrieval in AI Systems (Anthropic, 2024)](https://www.anthropic.com/engineering/contextual-retrieval): chunk contextualization plus hybrid search plus reranking, with measured failure-rate reductions and cost.
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172): why context ordering and size matter.
- [Matryoshka Representation Learning (Kusupati et al., 2022)](https://arxiv.org/abs/2205.13147): truncatable embeddings for cheaper storage.
- [MTEB: Massive Text Embedding Benchmark (Muennighoff et al., 2022)](https://arxiv.org/abs/2210.07316): how to shortlist embedding models.
- [Seven Failure Points When Engineering a Retrieval Augmented Generation System (Barnett et al., 2024)](https://arxiv.org/abs/2401.05856): a practitioner's catalog of where RAG breaks.
