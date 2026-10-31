---
title: "Detecting and Reducing LLM Hallucinations"
description: "Separate hallucination types, detect them with claim-level verification and uncertainty signals, and reduce them without making the model useless."
slug: "detecting-reducing-hallucinations"
category: "evaluation-reliability"
tags: ["hallucination", "evals", "rag", "llm-as-judge"]
seriesOrder: 14
pubDate: 2026-01-26
sourceDoc: "1WwZ8yG-cd8f7_IZixvOLmzYPvB9hYJ554S4N4h26QE0"
---

## TL;DR

Start by separating the types: **intrinsic** hallucinations contradict the provided source, **extrinsic or factual** hallucinations make unsupported claims about the world, and **tool or action** hallucinations invent API arguments, citations or IDs. Each needs different detectors and mitigations. To reduce them, ground answers in retrieval with required citations, give the model a way to abstain, keep it from answering beyond what it knows (calibration, and post-training that rewards "I don't know"), and verify after generation. To detect them, break outputs into atomic claims and check each one against sources with an NLI (natural language inference) model or an LLM judge, or use self-consistency and semantic entropy when there is no source. Measure with claim-level precision, abstention rate and calibration curves, and make the tradeoff explicit: fewer hallucinations usually means more refusals.

## Requirements to pin down first

- **Closed-book or grounded?** Open-domain chat, RAG over a corpus, summarization of a provided document, or agentic tool use? Grounded settings let you check claims against the source, which is much easier and more reliable than reference-free detection.
- **What does a hallucination cost here?** A wrong drug dose and a wrong movie-trivia answer need very different abstention thresholds and review paths.
- **What is the verification budget?** Can you afford 2-5x tokens for a verifier pass, or only a lightweight classifier? This decides between LLM-judge verification and cheaper NLI models or sampling.
- **Do you own the model?** If so, post-training levers are available. If not, every mitigation happens at inference time.

## Detecting hallucinations

![Flowchart: a user query goes through hybrid retrieval and reranking, then grounded generation that cites spans. If evidence is insufficient, the system abstains or escalates. Otherwise the answer is split into atomic claims, which are checked two ways in parallel: an NLI model or LLM judge compares each claim with evidence, and structural checks validate IDs, URLs and tool arguments against real systems. If all claims are supported, the answer is shown with citations; if not, unsupported claims are removed, flagged or revised before the answer is shown. Shown answers feed sampled audits and user flags.](/diagrams/q14-hallucination-defenses.svg "Layered defenses: reduce at generation time, detect per claim, and audit in production.")

No single layer is sufficient: grounding reduces hallucinations, claim-level checks catch what slips through, and production audits catch what the checks miss.

### 1. Claim decomposition

Split the response into atomic claims, as FActScore does. Scoring at the claim level beats a single response-level score because the usual failure is one wrong sentence in ten, which a response-level label either misses or over-penalizes.

### 2. Grounded verification (when a source exists)

For each claim, retrieve evidence (the provided context, or a search for open-domain questions) and classify the claim as *supported*, *contradicted* or *not enough info* using an NLI model or an LLM judge.

Citations need their own check: does the cited passage actually entail the sentence it is attached to? Measure **citation precision** (cited passages that support their sentence) and **citation recall** (sentences that needed a citation and got a supporting one) separately.

### 3. Reference-free signals (no source)

- **Self-consistency / SelfCheckGPT.** Sample N=5-10 responses. Claims that don't recur across samples are likely made up.
- **Semantic entropy.** Cluster sampled answers by meaning, using bidirectional entailment, and compute entropy over the clusters. High entropy indicates confabulation. This works well for short-form QA.
- **Token logprobs or verbalized confidence.** Cheap, but less reliable after RLHF (reinforcement learning from human feedback), which can make models overconfident.
- **Probes on hidden states.** If you own the weights, linear probes on activations can predict truthfulness. A good research signal; check robustness on your distribution before shipping it.

### 4. Structural checks for tool use

Validate IDs, URLs and citations against real systems. Resolve every URL and DOI, and check every function argument against the schema and the database. Tool hallucinations are the easiest to catch deterministically, so never leave them to a judge. Schema enforcement itself is covered in [Guaranteeing Valid JSON Output from LLMs](/blog/guaranteed-valid-json-output/).

## Reducing hallucinations

Ordered by usual return on investment.

### Retrieval grounding with quality controls

Use hybrid retrieval plus a reranker. Instruct the model to answer only from the context and to cite spans. Many RAG hallucinations are really retrieval misses: the right passage never reached the model, so it filled the gap. Measure retrieval recall@k first.

### Permission and incentive to abstain

State explicitly that "I don't know" or "the documents don't say" is a correct answer. In structured outputs, provide an `insufficient_evidence` status so the model has a valid way to decline.

### Generate, verify, revise

Chain-of-Verification (CoVe) has the model draft an answer, write verification questions, answer them independently of the draft, then revise. Alternatively, a separate verifier model removes or flags unsupported claims before the response is shown.

### Decoding

Lower the temperature for factual tasks. Constrain citations to the IDs of retrieved documents so the model cannot invent a reference.

### Product UX

Show inline citations users can click, confidence indicators where they are calibrated, and human review for high-stakes outputs.

### Post-training (if you own the model)

Reward calibrated abstention and penalize confident errors. Train on "unanswerable" examples, and run factuality RL with a claim-level reward from an automated fact-checker.

This matters because of how models are graded. Kalai et al. (2025), "Why Language Models Hallucinate", argue that models bluff partly because training and most benchmarks use binary 0/1 accuracy, where "I don't know" scores the same as a wrong answer. Under that rule, guessing always has higher expected score than abstaining. Their proposed fix is to state an explicit confidence target `t` in the instructions: a correct answer earns 1 point, "I don't know" earns 0, and a wrong answer costs `t/(1−t)` points.

A worked example of why that works: with confidence `p`, answering has expected score `p − (1−p)·t/(1−t)`, which is positive only when `p > t`. At `t = 0.5` (penalty 1), a 40%-confident guess has expected score `0.4 − 0.6 = −0.2`, so abstaining is better. At `t = 0.9` (penalty 9), the model should answer only when more than 90% confident. Under binary grading (`t = 0`) the same 40% guess scores `+0.4`, so bluffing wins.

## How to measure it

### Core metrics

- **Claim-level factual precision** (FActScore-style), paired with **supported claims per response** so the system can't win by saying less.
- **Abstention rate and selective accuracy.** Plot accuracy against coverage (the share of questions answered). A good system lifts the whole curve, not just one point on it.
- **Calibration.** ECE (expected calibration error) on verbalized or derived confidence.
- **Groundedness / faithfulness for RAG** (RAGAS-style), with retrieval recall measured separately so you know which component failed.

A worked example of the coverage tradeoff, with illustrative numbers: a system that answers everything at 80% accuracy is wrong on 20% of queries. If it abstains on the 15% of queries where it is least confident and reaches 92% accuracy on the remaining 85%, it is wrong on `0.85 × 0.08 = 6.8%` of queries, correct on `0.85 × 0.92 = 78.2%`, and declines 15%. Errors fall by about two-thirds at the cost of 1.8 points of correct answers and a 15% refusal rate. Whether that trade is right depends on the product surface.

### Detector quality

- Measure the AUROC (area under the ROC curve) of each detector against human-labeled hallucinations on in-domain data.
- Calibrate any LLM judge against 200-500 human labels and require agreement around Cohen's κ ≥ 0.7 before trusting it.

### Production

- User flags on answers.
- Sampled audits, for example 1% of traffic labeled weekly.
- Hallucination rate split by query category, because aggregate rates hide high-risk slices.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| RAG + mandatory citations | Big reduction; auditable | Only as good as retrieval; citations can be decorative | Any domain with a corpus |
| Post-hoc NLI/LLM-judge claim verification | Catches unsupported claims before display | Added latency and cost; judge errors | High-stakes, grounded outputs |
| Sampling consistency / semantic entropy | No reference needed | N× compute; weaker on long-form | Closed-book short answers, offline auditing |
| Abstention thresholds | Directly trades errors for refusals | Over-refusal hurts helpfulness | Domains where wrong > unhelpful |
| Post-training for calibration/factuality | Fixes at the source; no inference cost | Needs RL infrastructure and data; can over-refuse | You own the model |

- Choose a point on the precision/coverage curve for each product surface, together with product and legal. It is a business decision, not a single global number.
- Say less vs. say wrong: factual precision can be gamed by being vague. Always pair it with informativeness or claim count.

## Common mistakes

- **Treating "hallucination" as one thing.** Intrinsic, extrinsic and tool hallucinations have different fixes; a single metric hides which one is growing.
- **Assuming RAG removes hallucination.** Models still ignore, misread or over-generalize from the context, and retrieval misses get papered over with fluent guesses.
- **Trusting an LLM judge without measuring agreement with humans**, or using the same model to generate and to judge, which invites self-preference bias.
- **Optimizing hallucination rate alone.** The result is an assistant that refuses everything.
- **Citation checks that only confirm the citation exists**, not that it supports the claim. Decorative citations then pass.
- **Ignoring that accuracy-only grading rewards guessing.** If your evals score "I don't know" as wrong, they push the system toward bluffing.

## Thinking at scale

- Set abstention thresholds per surface, with owners. Product, legal and engineering should jointly own where each surface sits on the precision/coverage curve, and revisit it when the model changes.
- Fix your own scoreboard. Internal evals and dashboards should credit appropriate abstention, or every model and prompt change will drift toward confident guessing.
- Keep a human-labeled audit stream. Detectors and judges drift with model upgrades; a steady weekly sample is the only way to notice.
- Attribute failures to components. Track retrieval recall, faithfulness and claim precision separately so the retrieval team and the generation team see their own regressions.

For calibrating judges and building the eval harness these metrics run in, see [Building an Eval Suite for a Customer Support Bot](/blog/support-bot-eval-suite/).

## Further reading

- [Why Language Models Hallucinate (Kalai et al., 2025)](https://arxiv.org/abs/2509.04664): Argues binary accuracy grading rewards guessing over abstaining and proposes explicit confidence targets.
- [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models (Manakul et al., 2023)](https://arxiv.org/abs/2303.08896): Sampling-consistency detection without external references.
- [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation (Min et al., 2023)](https://arxiv.org/abs/2305.14251): Decompose into atomic facts and verify each against a knowledge source.
- [Detecting hallucinations in large language models using semantic entropy (Farquhar et al., Nature, 2024)](https://www.nature.com/articles/s41586-024-07421-0): Meaning-level uncertainty for detecting confabulations.
- [Chain-of-Verification Reduces Hallucination in Large Language Models (Dhuliawala et al., 2023)](https://arxiv.org/abs/2309.11495): Draft, verify with independent questions, then revise.
- [Language Models (Mostly) Know What They Know (Kadavath et al., 2022)](https://arxiv.org/abs/2207.05221): Calibration and P(True) self-evaluation as a basis for abstention.
- [Survey of Hallucination in Natural Language Generation (Ji et al., 2022)](https://arxiv.org/abs/2202.03629): The intrinsic vs. extrinsic taxonomy and a metric overview.
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods (Lin et al., 2021)](https://arxiv.org/abs/2109.07958): Benchmark for imitative falsehoods.
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)](https://arxiv.org/abs/2309.15217): Faithfulness and context metrics for grounded systems.
