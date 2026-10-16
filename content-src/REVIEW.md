# Content review summary

Each of the 25 answers in the five "LLM System Design Answers" docs was reviewed for correctness, detail, sources and readability (1-5), then rewritten as a standalone article. Per-post detail, including every issue found and the evidence for each fix, is in `content-src/reviews/qNN.md`.

Scores are before → after.

| # | Article | Correctness | Detail | Sources | Readability | Issues fixed |
|---|---|---|---|---|---|---|
| 1 | [Designing a RAG System for 10 Million Documents](reviews/q01.md) | 3 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 12 |
| 2 | [Debugging a RAG System That Gives Wrong Answers](reviews/q02.md) | 4 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 7 |
| 3 | [Chunking Legal Contracts for Retrieval](reviews/q03.md) | 3 → 5 | 4 → 5 | 3 → 5 | 3 → 5 | 8 |
| 4 | [BM25 vs Dense vs Hybrid Search: When to Use Each](reviews/q04.md) | 4 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 7 |
| 5 | [Adding a Reranker and Proving It Helped](reviews/q05.md) | 4 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 10 |
| 6 | [Cutting LLM Cost by 70% Without Losing Quality](reviews/q06.md) | 3 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 9 |
| 7 | [Reducing Time to First Token in a Chat App](reviews/q07.md) | 3 → 5 | 4 → 5 | 3 → 5 | 3 → 5 | 7 |
| 8 | [Designing a Semantic Cache with Hit-Rate Tracking](reviews/q08.md) | 4 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 5 |
| 9 | [Routing Requests Between a Cheap and a Strong Model](reviews/q09.md) | 4 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 6 |
| 10 | [Building Tool-Using Agents That Work Reliably](reviews/q10.md) | 4 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 6 |
| 11 | [Detecting and Recovering from Agent Loops](reviews/q11.md) | 3 → 5 | 4 → 5 | 3 → 5 | 2 → 5 | 7 |
| 12 | [Designing Long-Term Memory for an LLM Assistant](reviews/q12.md) | 2 → 5 | 4 → 5 | 3 → 5 | 3 → 5 | 6 |
| 13 | [Guaranteeing Valid JSON Output from LLMs](reviews/q13.md) | 3 → 5 | 4 → 5 | 3 → 5 | 3 → 5 | 7 |
| 14 | [Detecting and Reducing LLM Hallucinations](reviews/q14.md) | 4 → 5 | 4 → 5 | 3 → 5 | 3 → 5 | 5 |
| 15 | [Building an Eval Suite for a Customer Support Bot](reviews/q15.md) | 4 → 5 | 3 → 5 | 3 → 5 | 3 → 5 | 6 |
| 16 | [Making LLM-as-a-Judge Trustworthy: Calibration, Bias and Drift](reviews/q16.md) | 3 → 5 | 4 → 5 | 4 → 5 | 2 → 5 | 7 |
| 17 | [Defending Tool-Using Agents Against Prompt Injection by Design](reviews/q17.md) | 4 → 5 | 4 → 5 | 4 → 5 | 2 → 5 | 8 |
| 18 | [Building a PII Redaction Gateway in Front of Your LLM](reviews/q18.md) | 4 → 5 | 4 → 5 | 4 → 5 | 2 → 5 | 6 |
| 19 | [Fine-Tuning vs RAG vs Prompting: An Eval-First Decision Procedure](reviews/q19.md) | 4 → 5 | 4 → 5 | 4 → 5 | 2 → 5 | 6 |
| 20 | [Fine-Tuning a Small Model with LoRA on 5,000 Examples](reviews/q20.md) | 2 → 5 | 4 → 5 | 4 → 5 | 2 → 5 | 7 |
| 21 | [Serving Open-Weight LLMs at Scale: Batching, KV Cache, Quantization](reviews/q21.md) | 3 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 7 |
| 22 | [Streaming LLM Responses over SSE and Surviving Disconnects](reviews/q22.md) | 3 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 8 |
| 23 | [Tracing Every Step of a Multi-Agent Workflow](reviews/q23.md) | 4 → 5 | 4 → 5 | 3 → 5 | 3 → 5 | 8 |
| 24 | [Rolling Back a Bad Prompt Change in Production](reviews/q24.md) | 4 → 5 | 3 → 5 | 4 → 5 | 3 → 5 | 5 |
| 25 | [Designing a ChatGPT-Style Chat Service for 1M Daily Users](reviews/q25.md) | 4 → 5 | 4 → 5 | 4 → 5 | 3 → 5 | 9 |
| | **Average** | **3.5 → 5.0** | **3.9 → 5.0** | **3.7 → 5.0** | **2.8 → 5.0** | **179** |

## Fixes that applied to every doc

- Tables rebuilt as real tables with a header row (the originals had an empty header row and literal `**bold**` markers), and stray escapes such as `\~`, `\+`, `\&gt;` removed.
- Interview voice removed: no "I'd", no "Pitfalls interviewers probe", no "What a principal would add", no invented first-hand anecdotes. Sections renamed to a single template (TL;DR, Requirements to pin down first, design sections, How to measure it, Tradeoffs, Common mistakes, Thinking at scale, Further reading).
- Links normalized to canonical URLs: `docs.anthropic.com` → `platform.claude.com/docs`, `anthropic.com/news|research/...` → `anthropic.com/engineering/...`, `platform.openai.com/docs` → `developers.openai.com/api/docs`, moved vLLM, Elastic and OpenTelemetry pages.
- Unsourced numbers either cited or explicitly labelled as illustrative planning figures.
- One diagram or more per article (26 in total), with full alt text.

## Most significant corrections

| # | Was | Now |
|---|---|---|
| 1 | 100M chunks for 10M docs; HNSW memory 1.5-2× vectors | 30-80M chunks (25B tokens ÷ 10M docs = 2,500 tokens/doc); HNSW graph overhead ≈ 6% at 1024-dim (Faiss formula) |
| 3 | Late Chunking cited as arXiv 2409.04669 (an unrelated paper) | arXiv 2409.04701 |
| 4 | BM25 beats dense models zero-shot (BEIR, 2021) | Still true on some datasets and ID-heavy queries, but modern embedders (e.g. E5) beat BM25 on BEIR overall |
| 6 | Output tokens cost 3-5× input; only 5-minute cache writes | 4-8× (5× on all current Claude models); 1-hour cache writes at 2×; worked example corrected to $207k / 79% |
| 7 | 70B prefill on one H100 | 70B BF16 needs ~140 GB, so TP=4 H100 (≈2.7 s for 30k tokens) or TP=8 / FP8 (≈1.4 s) |
| 12 | Generative Agents score = relevance × recency × importance | Weighted sum of min-max-normalized terms (Park et al. 2023, §4.1) |
| 13 | "Use structured outputs where available"; format limits hurt reasoning | Structured outputs and strict tool use are generally available; the reasoning penalty is disputed (Tam et al. vs .txt rebuttal) |
| 16 | 1,000 paired comparisons resolve a 3-point gap | ±3.1 points at n=1,000 is the bare minimum; ~2,200 decisive comparisons for 80% power |
| 20 | LoRA on 8B bf16 ≈ 16 GB including activations | 16 GB is weights alone; ≈46-48 GB without checkpointing, ≈23-25 GB with, ≈13-15 GB QLoRA |
| 21 | ≈$0.03 per 1K output tokens | ≈$0.001-0.003 per 1K (10-30× lower); examples refreshed to current MoE models and B200 |
| 22 | 700 streams/s × 40 tok/s ≈ 28k appends/s | Little's law: ≈8,400 concurrent streams, ≈336k tok/s, 170k-340k appends/s after coalescing |
| 25 | 250 msg/s (short answer) vs 230/s (detail) | 20M ÷ 86,400 ≈ 230/s throughout |

## Verification

- `node scripts/check-links.mjs`: all 168 external links resolve (openai.com returns 403 to bots; those pages were confirmed via search).
- `python3 scripts/check-arxiv.py`: all 96 arXiv IDs match the cited paper titles.
- Doc snapshots of the originals are in `content-src/original/`. (`q16-q20.md` was transcribed from the read output rather than saved byte-for-byte.)
