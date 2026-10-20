---
title: "Chunking Legal Contracts for Retrieval"
description: "Why fixed-size chunks fail on contracts, and how to chunk by clause with context headers, defined terms, cross-references and amendment lineage."
slug: "chunking-legal-contracts"
category: "retrieval-rag"
tags: ["chunking", "rag", "retrieval", "evals"]
seriesOrder: 3
pubDate: 2026-10-16
sourceDoc: "1h3m8wtZykYFchMjt8t57Aqya0AhrdYy0tlem7E4EI3A"
video: { youtube: "tl9HIgUdsnQ", duration: "11:36", url: "https://youtu.be/tl9HIgUdsnQ?si=Pu143r2_MqkymqpZ" }
---

## TL;DR

Contracts are highly structured and full of cross-references, so chunk them along their own hierarchy (article, section, clause) rather than by fixed token windows. Keep each clause whole when it fits (roughly 200-1,000 tokens) and split long clauses at sub-clause boundaries. Each chunk carries its heading path, the contract's identity (parties, type, effective date) and, at retrieval time, the definitions of the defined terms it uses. Retrieve on small clause-level chunks and expand to the parent section at generation time. Then validate the choice on a span-level clause-retrieval eval built from CUAD or LegalBench-RAG, because chunking is a hyperparameter you settle by measuring.

## Requirements to pin down first

- **What is the task?** Question answering over one contract, search across thousands ("which contracts have uncapped indemnity?"), clause comparison against a negotiation playbook, or extraction into fields? Cross-corpus "find all" tasks push toward structured extraction rather than chunk retrieval.
- **What is the source quality?** Native PDF or DOCX, scanned images, redlines, exhibits, schedules, amendments. Scans and redlines decide how much you can trust the structure parser.
- **How big are the documents?** Contracts range from 5 to 300+ pages. A master agreement plus amendments means the *governing* text may live in a later document, so the system has to know which version wins.
- **How precise must answers be?** Legal users need the exact clause text and a citation to the section number. Paraphrase-level retrieval is not enough, which rules out designs that only return "roughly relevant" passages.

## Why naive chunking fails on contracts

- Fixed 512-token windows split clauses mid-sentence. They separate "Notwithstanding the foregoing..." from the rule it modifies. Exceptions and carve-outs end up in a different chunk from the rule, which is how a system produces confidently wrong answers.
- Defined terms live elsewhere. "Confidential Information" is defined in §1.4 and used in §9. A §9 chunk on its own does not say what it covers.
- Cross-references point elsewhere. "Subject to Section 12.3" and "except as set forth in Schedule B" make a clause meaningless without its target.
- Boilerplate looks identical across contracts. A portfolio has thousands of near-identical governing-law clauses. Without contract-level metadata, retrieval cannot tell *whose* clause it found.

## A structure-aware chunking strategy

![Diagram of a parsed contract structure tree for an MSA between Acme and Globex, with Article 1 Definitions, Article 9 Liability and Article 12 Indemnity. Under Article 9 sits Section 9.2 Limitation of Liability and its sub-clause 9.2(b) Exceptions. Sub-clause 9.2(b) becomes a clause chunk with a context header and is retrieved by hybrid search. A dashed "subject to" cross-reference edge links 9.2(b) to Section 12.3. Article 1 feeds a defined-terms lookup. The LLM context is assembled from the retrieved clause, its parent section 9.2, the definitions of terms it uses, and the one-hop referenced Section 12.3.](/diagrams/q03-contract-chunking.svg "Retrieve on clause chunks; assemble context from the clause, its parent, its definitions and one hop of cross-references.")

The chunk that gets retrieved is small, but the context the model sees is assembled from the structure around it.

### 1. Parse to a structure tree

Detect the numbering hierarchy (Article 5 → 5.2 → 5.2(a)(ii)) from layout cues and regular expressions, with an LLM fallback for messy documents. Keep tables such as payment schedules and SLAs (service-level agreements) intact as Markdown. Treat exhibits and schedules as separate subtrees.

### 2. Make clause-level chunks

- One chunk per lowest heading that fits within about 800 tokens.
- Merge tiny siblings (for example, one-line sub-clauses) up to a minimum of about 200 tokens.
- Split oversized clauses at sub-clause boundaries with a small overlap of 1-2 sentences.

The chunk count follows from contract length. For example, a 40-page contract is roughly 20k tokens; with clause chunks averaging about 300 tokens, that is about 65 chunks. A 300-page master agreement with schedules (about 150k tokens) produces about 500. Expect tens to a few hundred chunks per contract.

### 3. Add a context header to every chunk

Prepend a header for embedding and BM25 indexing, and store it separately for display:

```text
[Contract: MSA, Acme Corp <-> Globex, effective 2023-04-01]
[Section 9.2 Limitation of Liability > (b) Exceptions]
```

Optionally add a 1-2 sentence LLM-generated summary that situates the clause in the contract, the technique Anthropic calls contextual retrieval. Because contract portfolios are full of near-identical boilerplate, legal text benefits from this more than most domains do.

The cost is modest, but check the assumptions. Anthropic reported $1.02 per million document tokens using Claude 3 Haiku with prompt caching (2024 prices), for 8k-token documents split into 800-token chunks. Each chunk's call re-reads the cached document, so the per-token cost grows with (document length / chunk size). For example, re-running that arithmetic at the same prices for 30k-token contracts and 300-token chunks gives roughly $4 per million tokens, so a portfolio of 10,000 such contracts (300M tokens) costs on the order of $1,200 to contextualize once. Re-price with the model you use.

### 4. Resolve defined terms at retrieval time

Extract the definitions section into a lookup table. At retrieval time, attach the definitions of the capitalized terms that appear in each retrieved chunk. Do not inline definitions into every chunk at index time: it bloats the chunk and dilutes its embedding.

### 5. Build a cross-reference graph

Store edges for "Section X refers to Section Y". At generation time, pull in one hop of referenced clauses, at least for "subject to" and "notwithstanding" links, since those change the meaning of the clause that cites them.

### 6. Retrieve small, expand to the parent

Retrieve on clauses, then expand to the parent section when the answer needs surrounding context. Also keep one contract-level summary chunk per contract for routing questions such as "which contracts are with Globex?".

### 7. Store metadata for filtering

Record contract type, parties, dates, governing law, version and amendment lineage, and a `superseded_by` link so retrieval prefers current terms.

### An alternative: late chunking

Late chunking runs a long-context embedding model over the whole document first, then pools token embeddings per chunk, so each chunk vector already "knows" its surroundings. It addresses the same problem as context headers without an LLM call per chunk. Include it in the head-to-head comparison below rather than assuming either approach wins.

## Pairing chunks with retrieval and generation

Legal queries mix semantics ("cap on liability") with exact strings ("Section 7.3", "LIBOR" in legacy contracts, party names). Use hybrid BM25 plus dense retrieval (see [BM25 vs Dense vs Hybrid Search](/blog/bm25-dense-hybrid-search/)) with a legal-domain or fine-tuned reranker.

In the LegalBench-RAG experiments, adding a general-purpose reranker (Cohere's `rerank-english-v3.0`) performed *worse* than no reranker on legal text, which the authors attribute to domain mismatch. Evaluate any off-the-shelf reranker on legal queries before adopting it; [Adding a Reranker and Proving It Helped](/blog/reranker-evaluation/) describes how.

Generation should quote the clause verbatim with its section number, so a lawyer can verify the answer against the source.

## How to measure it

### Build a span-level eval

Build 200-500 queries with gold clause spans. Two public resources help:

- **CUAD** (Contract Understanding Atticus Dataset): 510 commercial contracts with more than 13,000 expert labels across 41 clause categories.
- **LegalBench-RAG**: 6,858 query-answer pairs over a corpus of more than 79M characters, with ground truth given as character-index spans in the source documents. It draws on CUAD, ContractNLI, MAUD and PrivacyQA.

Add in-house questions that reflect your users.

### Metrics

- **Span-level recall@k**: did any retrieved chunk overlap the gold span?
- **Precision@k**: how much irrelevant text was fed to the model? LegalBench-RAG scores retrieval against gold spans rather than document IDs for this reason: a 2,000-token chunk that contains a 40-token answer is mostly noise.
- **Answer correctness** on an attorney-labeled subset.

### Compare chunkers head to head

Hold the rest of the pipeline fixed and compare:

1. fixed 512-token windows with 10% overlap,
2. recursive character splitting,
3. structure-aware clause chunking,
4. structure-aware chunking plus context headers,
5. late chunking.

Structure-aware chunking with context is expected to beat fixed windows on clause recall, but you need the numbers on *your* contracts. Public results are not conclusive: LegalBench-RAG compared only naive fixed-size (500-character) chunking against recursive splitting.

### Slice the results

Slice by contract length, scanned vs native documents, and query type (single clause, cross-reference, cross-contract). Averages hide the slice where a chunker collapses.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Fixed-size windows (512 tokens, overlap) | Trivial, robust to bad parsing | Splits clauses, loses carve-outs | Baseline only, or unparseable scans |
| Recursive or sentence splitting | Respects sentences | Ignores legal hierarchy | Mixed-quality corpora, as a fallback |
| Structure-aware clause chunks | Aligns with how lawyers cite and query | Needs a good parser; variable sizes | Default for contracts |
| Clause chunks + contextual headers or summaries | Disambiguates boilerplate across contracts | Ingest-time LLM cost (about $1-4 per million tokens with caching at 2024 small-model prices, depending on chunks per document) | Multi-contract corpora |
| Whole contract in long context | No chunking errors | Expensive per query; mid-context recall drops | Single-contract Q&A, under ~100 pages |
| Structured extraction to a table | Exact, exhaustive, auditable | Fixed schema, upfront labeling | Portfolio analytics, compliance |

- Chunk size trades precision against completeness. Clause-level chunks plus parent expansion get most of both.
- Overlap is a weak patch for structure. Prefer explicit links (definitions, cross-references) over a bigger overlap.

## Common mistakes

- **Choosing chunk size by intuition** ("512 feels right") instead of measuring span-level recall on a legal eval. The choice is never revisited and its cost is invisible.
- **Separating exceptions and carve-outs from the rule they modify.** Answers come out confidently wrong in the costliest direction.
- **Ignoring defined terms and cross-references.** The model answers from a clause whose meaning depends on text it never saw.
- **Leaving contract identity out of the chunk.** A near-identical clause from the wrong contract gets retrieved and cited.
- **Ignoring amendments and superseded versions.** The system cites terms that no longer apply.
- **Losing table structure during PDF parsing.** Fee schedules and SLA tables become unreadable number soup.

## Thinking at scale

- Portfolio questions need a different abstraction. For "all contracts with auto-renewal and under 30 days' notice", chunking is the wrong tool. Run clause extraction into a structured table at ingest and query it with SQL; RAG then handles only the long tail. The same split applies to any large corpus, as discussed in [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/).
- Model document lineage explicitly. Amendments override the base agreement. If the system cites superseded terms, that is a liability risk as well as a quality bug, so lineage deserves an owner and tests of its own.

## Further reading

- [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review (Hendrycks et al., 2021)](https://arxiv.org/abs/2103.06268): 510 contracts and 41 clause categories with expert span labels, a starting point for a retrieval eval.
- [LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain (Pipitone & Alami, 2024)](https://arxiv.org/abs/2408.10343): span-level legal retrieval benchmark, with chunking and reranker comparisons.
- [Contextual Retrieval in AI Systems (Anthropic, 2024)](https://www.anthropic.com/engineering/contextual-retrieval): prepending chunk-specific context before embedding and BM25 indexing, with cost figures.
- [Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models (Günther et al., 2024)](https://arxiv.org/abs/2409.04701): embed the whole document first, then pool per chunk, as an alternative to context headers.
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172): why stuffing whole contracts into context is not free.
- [Seven Failure Points When Engineering a Retrieval Augmented Generation System (Barnett et al., 2024)](https://arxiv.org/abs/2401.05856): includes chunking and consolidation failures.
