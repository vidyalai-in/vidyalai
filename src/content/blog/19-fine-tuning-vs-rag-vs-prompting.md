---
title: "Fine-Tuning vs RAG vs Prompting: An Eval-First Decision Procedure"
description: "Diagnose whether your gap is knowledge or behavior, then climb from prompting to RAG to fine-tuning only when the cheaper rung plateaus."
slug: "fine-tuning-vs-rag-vs-prompting"
category: "fine-tuning"
tags: ["fine-tuning", "rag", "evals", "cost", "architecture"]
seriesOrder: 19
pubDate: 2026-10-16
sourceDoc: "1peTYAI7ps58PRKJWERCzNUnsehYjGOwhNRHViG8g-bs"
---

## TL;DR

For most enterprise use cases, start with prompting plus RAG (retrieval-augmented generation), and use fine-tuning for behavior, never for knowledge. Prompting is the fastest loop and sets the baseline. RAG is the right tool when the problem is knowledge, because it handles freshness, access control, citations and deletion. Fine-tuning is the right tool when the problem is behavior: format, style, tool-use patterns, cost or latency through distillation into a smaller model, or a narrow skill that is hard to specify in a prompt. Fine-tuning is a poor way to inject new facts: it is sample-inefficient, it can increase hallucination, and it cannot respect per-user permissions. Decide by building an eval first, and climb the ladder only when the cheaper rung plateaus.

## Requirements to pin down first

**Is the gap knowledge or behavior?** Diagnose it from errors on the eval. If the model is wrong because it doesn't know a fact, that is a knowledge gap. If it knows the fact but formats the answer badly, calls tools wrongly or picks the wrong tone, that is a behavior gap. The two have different fixes.

**How fast does the knowledge change, and who is allowed to see it?** Documents that change daily and carry per-user ACLs (access control lists) effectively force RAG, because weights can neither update daily nor filter by user.

**What are the volume, latency and cost targets?** At high volume, prompt length dominates cost. As an illustrative example, 100 million calls a month with a 4,000-token few-shot prompt is 400 billion input tokens a month (`100M × 4,000`). At $1-3 per million input tokens, that is $400,000-$1.2 million a month for the prompt alone. Prompt caching softens this: cached prefix reads on the Claude API, for example, are billed at 10% or less of the base input price. Distilling the behavior into a fine-tuned small model (around 8B parameters) removes the long prompt entirely and can cut serving cost by 10-50x, depending on hardware and utilization.

**Do you control the model weights?** Closed APIs constrain which fine-tuning options exist. Fine-tuning also ties you to one model version, so you redo it on every upgrade.

**Is there labeled data?** Fine-tuning needs on the order of 500-10,000 high-quality examples. RAG needs a corpus and retrieval evals.

## The decision procedure

![Flowchart of the eval-first decision ladder. Build an eval of 200 to 500 graded cases first, then run a prompting baseline with a strong model and few-shot examples. If it meets the target, ship and re-check on each model upgrade. If not, run error analysis. Prompt-fixable errors go back to the prompting step. Knowledge gaps (missing, fresh or permissioned facts) lead to adding RAG, measuring retrieval recall first. Behavior gaps (format, style, tool use) lead to a LoRA fine-tune on 1 to 10 thousand examples. Cost or latency gaps lead to distilling into a small fine-tuned model. When both knowledge and behavior gaps remain, combine RAG with a fine-tuned reader trained RAFT-style. Every path returns to the target check on the same eval.](/diagrams/q19-decision-ladder.svg "The eval-first ladder: diagnose the gap, take the cheapest rung that fixes it, and re-check on the same eval.")

Every rung is judged on the same eval, and error analysis, not preference, decides which rung comes next.

### 1. Build the eval first

Collect 200-500 representative cases with graded outputs, using programmatic checks plus a calibrated LLM judge (see [Making LLM-as-a-Judge Trustworthy](/blog/trustworthy-llm-as-a-judge/)), and slice them by failure category. Without this eval, you have no evidence for any of the choices that follow.

Keep the sample-size math in mind: on 300 cases, a pass rate near 80% has a 95% confidence interval of about ±4.5 points (`1.96 × sqrt(0.8 × 0.2 / 300)`). Differences smaller than that between rungs are not evidence.

### 2. Set a prompting baseline

Use the strongest available model with a clear system prompt, a few well-chosen examples, structured outputs and task decomposition. This often gets 70-90% of the way to the target within days. Record the token cost and latency; they are the numbers later rungs have to beat.

### 3. Add RAG if the errors are knowledge errors

- Use hybrid retrieval (BM25 keyword search plus dense embeddings), chunks of about 300-800 tokens with contextual headers, a cross-encoder reranker over the top 50-100 candidates, and a final 5-10 chunks in context. Anthropic's Contextual Retrieval write-up shows how much these pieces add up to: contextual embeddings plus contextual BM25 cut the top-20 retrieval failure rate by 49%, and adding reranking cut it by 67% (from 5.7% to 1.9%).
- Measure retrieval separately from generation. Recall@k on a labeled query-to-document set should be 90% or more at k=10 before you blame the generator. Then measure faithfulness (groundedness): the share of generated claims supported by the retrieved context.
- Expect retrieval to be the bottleneck. The most common RAG failure is bad retrieval (chunking, missing metadata filters, vocabulary mismatch), not the LLM.

### 4. Fine-tune if the errors are behavior errors that persist under good prompting

Typical wins:

- strict output schemas the model keeps violating
- domain-specific reasoning patterns
- tool-call conventions
- brand voice
- distilling a frontier model's behavior into a small, cheap, fast model

Start with LoRA (low-rank adaptation) supervised fine-tuning on 1,000-10,000 examples; [Fine-Tuning a Small Model with LoRA on 5,000 Examples](/blog/lora-fine-tuning-5k-examples/) walks through the recipe. Add preference tuning with DPO (direct preference optimization) only when you have a clear good-versus-bad preference signal.

### 5. Combine

The mature end state is often RAG plus a model fine-tuned to use retrieved context well: cite sources, abstain when the context lacks the answer, and ignore distractor documents. RAFT (retrieval-augmented fine-tuning) trains exactly this behavior by mixing relevant and distractor documents into the training examples.

## Why not fine-tune for knowledge

On the empirical side, Ovadia et al. (2023) found that RAG consistently outperforms unsupervised fine-tuning for knowledge injection, for both facts the model had seen in pretraining and entirely new ones. Gekhman et al. (2024) found that models learn new facts through supervised fine-tuning slowly, and that as the examples carrying new knowledge are eventually learned, the model's tendency to hallucinate rises.

On the operational side, with knowledge in the weights:

- You can't revoke a fact, for example to honor a GDPR deletion request.
- You can't enforce per-user ACLs.
- You can't cite sources.
- Every knowledge update means a retrain-and-eval cycle.

LIMA (Zhou et al., 2023) supports the same framing from the other side: a small, high-quality fine-tuning set mostly shapes format and style, which is behavior, not knowledge.

## How to measure it

- Compare every rung on the same eval. Track quality, p50 and p95 latency, cost per 1,000 requests and engineering time-to-ship. Present the result as a Pareto frontier (the set of options where no other option is better on every axis) rather than as a single winner.
- Include maintenance cost. A fine-tuned model must be retrained and re-evaluated whenever the base model is upgraded, often every 3-6 months. A prompt-plus-RAG stack usually ports to a new model in days.
- Track regressions outside the target domain after fine-tuning. Check general capability, safety behavior and instruction following. Catastrophic forgetting (losing previously learned capabilities during fine-tuning) is a real cost.
- Measure RAG in two stages. Report retrieval recall@k and generation faithfulness separately, so each regression has an owner.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Prompting (plus few-shot, structured output) | Fastest iteration, no infrastructure, ports to new models | Token cost at scale, context limits, brittle for complex behavior | Always first; low-to-mid volume; rapidly changing requirements |
| RAG | Fresh knowledge, citations, ACLs, deletable, auditable | Retrieval quality ceiling, added latency (+100-300 ms), more moving parts | Knowledge-heavy tasks, private or changing corpora |
| Fine-tuning (LoRA or full) | Consistent behavior, shorter prompts, distills into a cheap fast model | Data and eval cost, retrain on base upgrades, forgetting, poor at new facts | Stable behavior gaps, high volume, cost- or latency-bound |
| RAG plus fine-tuned reader | Best quality on domain QA; the model learns to use context | Most complex to build and maintain | Mature product with proven volume and eval |

- A knowledge gap means RAG, and a behavior gap means fine-tuning. Prompting comes first in both cases.
- Volume and latency targets, more often than quality, are what make fine-tuning worth it, through distillation.
- Factor in the cost of being pinned to one model version.

## Common mistakes

- **Choosing an approach before building an eval or doing error analysis.** The choice is then a matter of taste, and nobody can tell whether it worked.
- **Fine-tuning to teach facts.** The result is more hallucination and answers that go stale with no way to update them short of retraining.
- **Blaming the LLM for RAG failures that are really retrieval failures.** Always measure recall@k separately.
- **Believing that long context removes the need for RAG.** Stuffing the whole corpus into the prompt ignores cost, latency, "lost in the middle" degradation (models use information in the middle of long contexts less reliably) and ACLs.
- **Ignoring the maintenance tax.** Every base model upgrade invalidates the fine-tune.
- **Skipping regression checks outside the target domain after fine-tuning.** The model can lose general capability or safety behavior without the target eval noticing.

## Thinking at scale

Prompting, RAG and fine-tuning are not mutually exclusive. The decision is where to spend the next engineer-month, and the answer comes from error analysis on the eval.

The base model keeps getting better, so re-run that analysis regularly. A fine-tune that beat the old frontier model by 5 points often loses to the next frontier model with a good prompt. Before committing to fine-tuning, make sure the pipeline (data, training config, eval, regression suite) is cheap and automated to re-run, so that each model upgrade is a scheduled job rather than a project.

Assign ownership along the same lines. The eval set is a long-lived asset owned by the product team. The retrieval stack and the fine-tuning pipeline are platform capabilities that several products share. Keeping them separate makes it cheap to swap the rung a product uses when the economics change.

## Further reading

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401): the original RAG paper.
- [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs (Ovadia et al., 2023)](https://arxiv.org/abs/2312.05934): RAG outperforms unsupervised fine-tuning for knowledge injection.
- [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? (Gekhman et al., 2024)](https://arxiv.org/abs/2405.05904): new facts learned through SFT increase hallucination.
- [RAFT: Adapting Language Model to Domain Specific RAG (Zhang et al., 2024)](https://arxiv.org/abs/2403.10131): fine-tuning a model to read retrieved context, including distractors.
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172): why stuffing context isn't a free substitute for retrieval.
- [Contextual Retrieval (Anthropic, 2024)](https://www.anthropic.com/engineering/contextual-retrieval): hybrid retrieval, contextual chunking and reranking, with measured failure-rate reductions.
- [Prompt caching (Claude API documentation)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): how cached prefixes change the cost of long static prompts.
- [LIMA: Less Is More for Alignment (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206): a small, high-quality SFT set mostly shapes format and style, which supports the "fine-tune for behavior" framing.
