---
title: "Designing Long-Term Memory for an LLM Assistant"
description: "A tiered memory design with explicit write, read and forgetting policies, plus the benchmarks and production signals that show it works."
slug: "long-term-memory-assistant"
category: "agents-tool-use"
tags: ["memory", "agents", "retrieval", "privacy", "evals"]
seriesOrder: 12
pubDate: 2026-10-16
sourceDoc: "1WwZ8yG-cd8f7_IZixvOLmzYPvB9hYJ554S4N4h26QE0"
---

## TL;DR

Treat memory as a tiered system with explicit write and read policies, not as "put everything in a vector database". There are four tiers: a **working context** curated each turn within the window; **episodic memory**, timestamped summaries of past sessions; **semantic or profile memory**, distilled facts and preferences stored as structured records with provenance that can be overwritten; and **procedural memory**, learned instructions and skills. The hard problems are deciding what to write, resolving conflicts and staleness, retrieving the right few hundred tokens, and giving users control and deletion. Measure the system on long-horizon memory benchmarks plus production signals such as how often users have to repeat themselves.

## Requirements to pin down first

- **What horizon and volume?** Days or years? How many turns per user per day? This decides whether memory is about 1,000 facts or about 1 million events, and therefore whether a single table or a tiered, indexed store is needed.
- **Single user or shared memory?** Team or organization memory changes the ACL (access control list) and privacy model: every read must be filtered by who is asking.
- **What does "remember" mean for this product?** Preferences, ongoing projects, commitments ("remind me Friday"), or the full history? Each implies different stores and retrieval. Also decide the relative cost of a false memory versus a forgotten one, since that sets extraction thresholds.
- **What privacy and compliance rules apply?** GDPR right to erasure, data residency, sensitive categories such as health, and whether users can view and edit their memory. Erasure in particular must be designed in from the start.
- **What is the latency budget?** For example, memory retrieval must add less than 150 ms p95 to TTFT (time to first token). That rules out multi-hop retrieval on the hot path.

## The four memory tiers

1. **Working memory (the context window).** System prompt, pinned profile (about 300-800 tokens), a running summary of the current session, the last N turns verbatim, and retrieved memories. Even with 200k-1M token windows, keep the curated context around 10-30k tokens. Cost and latency grow with length, and recall degrades for information placed in the middle of long contexts (the "lost in the middle" effect).
2. **Episodic store.** An append-only log of raw turns in cold storage, which is the source of truth. On top of it, per-session summaries written at session end or every ~20 turns, each with timestamps, topics, entities and an embedding.
3. **Semantic / profile store.** Atomic facts as records:

   ```text
   {subject, predicate, value, confidence, source_turn_ids,
    created_at, last_confirmed_at, ttl, sensitivity}
   ```

   Example: `(user, dietary_restriction, vegetarian, 0.9, ...)`. Store it in Postgres with pgvector, or a key-value store plus a vector index. A small knowledge graph helps when entities and relations matter (people, projects).
4. **Procedural memory.** Instructions the user taught ("always use metric units", "format PRs like X"), stored as versioned rules and injected into the system prompt. Keep the active set small, under 1-2k tokens.

![Flowchart: a new conversation turn is appended to the raw turn log and also sent through the asynchronous write path, where a cheap model extracts proposed facts, a salience and sensitivity gate filters them, each proposed fact is compared with the top-k similar memories to choose ADD, UPDATE, DELETE or NOOP, and conflict resolution writes the result to the profile fact store or the procedural rules. The raw log is summarized into episode summaries. On the read path, the pinned profile and rules always enter the working context, hybrid retrieval with reranking pulls from profile facts and episode summaries, and memory tools let the model search, write and forget from inside the working context.](/diagrams/q12-memory-tiers.svg "Memory tiers with separate write and read paths.")

Writes and reads are separate pipelines: the write path decides what becomes a memory and how it changes existing ones, while the read path decides which few hundred tokens reach the context.

## The write path

This is where most memory systems succeed or fail.

### Extraction with explicit operations

After each turn, or in asynchronous batches, a cheap model proposes memory operations against the top-k most similar existing memories: `ADD`, `UPDATE`, `DELETE` or `NOOP`. This is the approach described in the Mem0 paper. Making updates explicit stops near-duplicate facts from piling up. Mem0 reports 91% lower p95 latency and more than 90% token-cost savings than a full-context baseline on its benchmark, which shows the payoff of storing distilled facts instead of replaying history.

### Salience gating

Only write what is likely to matter later: stated preferences, identity facts, commitments and ongoing goals. Don't write small talk. A useful target is roughly 0.1-0.5 new facts per turn. If the rate is much higher, the store fills with noise and retrieval quality drops.

### Conflict resolution

Two rules cover most cases:

- A newer explicit statement beats an older one.
- An explicit statement beats an inferred one.

Keep history with bitemporal fields (`valid_from` / `valid_to`). "I moved to Berlin" then retires "lives in London" by closing its validity interval, without deleting it from the audit log.

### Provenance

Every memory links to its source turn IDs. This lets the assistant cite where a memory came from, lets you audit it, and makes deletion work, because you can find every derived artifact of a turn.

### Sensitive data

Classify at write time. Health, finance and credentials are either not stored or stored only with explicit consent. Never store secrets.

## The read path

### Always-on context

The pinned profile and active procedural rules go into every request. They are small and almost always relevant.

### On-demand retrieval

Run hybrid BM25 plus dense retrieval over facts and episode summaries, filtered by user and ACL, then rerank. Budget about 1-3k tokens of retrieved memory.

For ranking, a well-known formulation comes from the Generative Agents paper (Park et al., 2023). It scores each memory as a **weighted sum** of three components, each min-max normalized to [0, 1]:

```text
score = α_recency · recency + α_importance · importance + α_relevance · relevance
```

In the paper all three α weights are set to 1. Recency is an exponential decay (factor 0.995 per sandbox game hour since the memory was last retrieved), importance is a 1-10 rating an LLM assigns at write time, and relevance is cosine similarity between the memory's embedding and the query's.

The sum matters. A product would let any near-zero component erase a memory. Consider a two-year-old but critical fact against yesterday's trivia, with normalized scores:

| Memory | Relevance | Recency | Importance | Sum | Product |
|---|---|---|---|---|---|
| "Allergic to peanuts" (2 years old) | 0.95 | 0.02 | 1.0 | **1.97** | 0.019 |
| "Had pasta yesterday" | 0.60 | 0.95 | 0.1 | 1.65 | **0.057** |

The sum ranks the allergy first; a product would rank yesterday's pasta first. In production, tune the α weights on your own eval set rather than keeping them all at 1.

### Agentic memory tools

Also expose `memory.search`, `memory.write` and `memory.forget` as tools, so the model can page memory in and out deliberately. This is the MemGPT pattern, which treats the context window like an operating system's main memory and external stores like disk. Anthropic's API ships a client-side memory tool built on this idea: the model issues file operations under a `/memories` directory and your application executes them against storage you control.

This works well for long tasks. Pair it with automatic retrieval anyway, because models under-call tools.

### Temporal queries

Questions like "what did we decide last month?" need time-indexed episode retrieval, not only semantic similarity. Parse the time expression into a date range and filter episodes by it before ranking.

## Consolidation and forgetting

- **Nightly "sleep" jobs.** Merge duplicates, re-summarize old episodes into coarser summaries (day → week → month), decay low-importance items that are never accessed, and lower confidence on stale facts until they are re-confirmed.
- **Hard delete that cascades.** A user deletion request must remove data from the raw log, summaries, facts, embeddings, caches and any fine-tuned artifacts. Provenance links make this possible. Design for it from day one; retrofitting it is far harder.

## How to measure it

### Offline benchmarks

Use LongMemEval- and LoCoMo-style evals. LongMemEval tests five abilities:

- **Information extraction**: recall of facts from a single session.
- **Multi-session reasoning**: combining facts across sessions.
- **Temporal reasoning**: questions anchored to dates and order.
- **Knowledge updates**: does the assistant use the *latest* fact?
- **Abstention**: does it say "I don't know" instead of inventing a memory?

Its authors report that commercial chat assistants and long-context LLMs show about a 30% accuracy drop when information must be remembered across sustained interactions, which is the gap a memory system has to close.

Also build an internal synthetic set: scripted multi-week personas with planted facts, later updates and deletions, and questions that probe each one.

### Component metrics

- **Extraction precision and recall** against human labels. Target precision above 90%, since false memories are worse than missed ones.
- **Retrieval recall@k**: whether the needed memory appears in the top k.
- **Memory-induced error rate**: answers that are wrong because of a bad or stale memory.

### Online metrics

- Rate of users re-stating information they gave before.
- Corrections such as "that's not right, I told you...".
- Edits and deletions in the memory panel.
- A/B tests on retention and satisfaction.
- p95 latency added by memory and storage cost per user.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Stuff full history in long context | Simplest; no retrieval errors | Cost and latency scale with history; lost-in-the-middle; window still finite | Short horizons, low volume, prototyping |
| Rolling summarization only | Cheap, bounded | Lossy; compounding drift; details vanish | Single long session (e.g. a coding task) |
| Vector store of raw chunks (RAG over history) | Easy; scales | Duplicates; no conflict resolution; stale facts surface | Search over past conversations as a feature |
| Structured fact store with ADD/UPDATE/DELETE | Precise, editable; supports updates and deletion | Extraction errors; schema design; more infrastructure | Personal assistants with preferences and a profile |
| Agentic memory tools (MemGPT-style) | Model controls paging; flexible | Under- or over-calling; harder to evaluate | Long autonomous tasks |

- **False memory vs. forgetting.** In most consumer products a confidently wrong memory hurts trust more than a forgotten one, so tune extraction for precision.
- **Implicit vs. explicit memory.** Inferring preferences is powerful and can feel creepy. Show users what is stored and let them edit it.
- **Personalization vs. sycophancy.** Memory can lock in user beliefs. Keep facts *about the user* separate from claims *about the world*.

## Common mistakes

- **No update or delete semantics.** "I'm no longer vegetarian" sits alongside the old fact, and the model picks one at random.
- **Erasure that doesn't cascade** to embeddings, summaries and backups. That is a GDPR failure, not just a bug.
- **Similarity-only retrieval.** It fails on temporal questions ("last month") and aggregate questions ("how many times did I...").
- **Memory as a prompt-injection vector.** A malicious document gets "remembered" as an instruction and replays in every future session. Sanitize writes and store instructions separately from facts.
- **Summaries of summaries drift.** Without the raw log as ground truth, errors compound and can't be corrected. Keep the raw log and allow re-derivation.
- **No abstention eval.** Without one, nothing stops the assistant from inventing plausible memories.

## Thinking at scale

- **Deletion is an architecture property.** Provenance from every derived artifact back to source turns is what makes a cascade delete possible across stores, caches and fine-tuned models.
- **Version the extraction pipeline.** A new extraction prompt or model changes what gets written. Keep the raw log so you can re-derive facts and summaries when it improves.
- **Separate ownership of write and read paths.** They fail differently (bad memories versus missed memories) and need their own metrics and on-call signals.
- **Plan for org memory.** Shared memory brings ACLs, conflicting statements from different people and stricter audit needs; retrofitting those onto a single-user design is costly.

Related: [Detecting and Recovering from Agent Loops](/blog/agent-loop-detection-recovery/) covers why a persistent tried-list must survive context compaction.

## Further reading

- [MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)](https://arxiv.org/abs/2310.08560): Virtual context management, where the model pages memory in and out across tiers via function calls.
- [Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)](https://arxiv.org/abs/2304.03442): The memory stream, retrieval as a weighted sum of recency, importance and relevance, and reflection-based consolidation.
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (Chhikara et al., 2025)](https://arxiv.org/abs/2504.19413): Extraction with ADD/UPDATE/DELETE/NOOP operations and graph memory, with latency and cost numbers.
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (Wu et al., 2024)](https://arxiv.org/abs/2410.10813): Five memory abilities including temporal reasoning, knowledge updates and abstention.
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172): Why stuffing the context is not a memory strategy.
- [Effective context engineering for AI agents (Anthropic, 2025)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): Compaction, structured note-taking and just-in-time retrieval for long-horizon agents.
- [Memory tool (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool): A production example of agentic, client-side memory tools.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401): The foundational retrieve-then-generate paper behind the read path.
