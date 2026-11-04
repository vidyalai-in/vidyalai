---
title: "Benchmark Contamination: Detecting a Model That Has Seen the Test"
description: "We leaked ARC questions into a small model's training data on purpose, then tested n-gram checks, Min-K%, shuffled-order tests and black-box probes on it."
slug: "benchmark-contamination"
category: "evaluation-reliability"
tags: ["evals", "benchmarks", "contamination", "pretraining"]
series: "llm-evals"
seriesOrder: 5
pubDate: 2026-09-11
updatedDate: 2026-11-04
---

## TL;DR

Benchmark contamination means test questions, or text close to them, ended up in a model's training data, so the score measures recall instead of ability. We contaminated a model on purpose: SmolLM2-135M was trained further on 7.7 million tokens of web text with ARC-Challenge test questions mixed in at 1, 4 or 16 copies, plus a paraphrased set at 16 copies. Sixteen verbatim copies took its accuracy on those questions from 32.9% to 97.4%, and the paraphrased questions gained 49 points while a 13-gram overlap check found half or more of the text in only 1 of 234. Scores computed from the model's own token probabilities separated seen from unseen questions almost perfectly at 16 copies, barely at one copy, and "detected" contamination just as strongly in an 88M model that never saw the test. Black-box probes on Claude Haiku 5.5 found no verbatim recall, which proves nothing either way.

## Why it matters

Sixteen copies of 234 exam questions, 4.5% of a 12-minute training run, took a 135M-parameter model from 33% to 97% on those questions. On 234 questions it never saw, the same run moved it from 29% to 36%.

Meta's own analysis of Llama 2 found that its 70B model scored 80.0 on the HellaSwag items with no overlap with its training data and 92.2 on the items with 80% or more overlap ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)). When Scale AI wrote 1,205 new grade-school math problems in the style of GSM8K, some model families lost up to 8 points ([Zhang et al., 2024](https://arxiv.org/abs/2405.00332)).

[Part 2](/blog/how-benchmarks-are-built/) covered the defenses builders put in place before release: canary strings, private held-out sets, rolling questions. This part is about what happens after: given a model and a benchmark, how can you tell whether the model has seen it, and how far can you trust the answer?

## The intuition

Think of a student who got hold of the exam before sitting it. They might have the exact paper, a photocopy with the questions renumbered, a friend's rewording of each question, or a study guide someone wrote with the exam open. All four raise their mark, and only the first two are easy to catch by comparing their notes with the paper.

A teacher who suspects a leak has a few options. Search the student's notes for sentences from the exam. Watch how fluently they answer: a memorized answer comes out fast and smooth. Ask them to recite question 7 word for word. Or set a fresh exam on the same material and compare the marks.

Each has a counterpart for language models. The analogy breaks in one useful way: you can't read a student's mind, but you can read a model's probability for every word of a question, a far finer instrument than watching someone answer.

## Step 1: four kinds of contamination

Exact contamination is the test item itself in the training text: a GitHub copy of the benchmark file, a quiz site, a paper quoting examples. Near-duplicates are the same text with small changes: different whitespace, an HTML table, numbers written as words. Paraphrase contamination keeps the content and changes the words. Indirect contamination arrives through synthetic data: a model trained on outputs from a teacher model that had memorized the test, or on generated exercises seeded with benchmark questions.

All four inflate scores the same way. A benchmark is meant to sample a skill, and a model that has stored an item's answer gets it right without that skill. [Yang et al. (2023)](https://arxiv.org/abs/2311.04850) fine-tuned Llama-2 13B on rephrased MMLU test questions, and it reached 85.9% on MMLU "while being undetectable by n-gram overlap", against 54.8% before.

### The standard check: n-gram overlap

An n-gram is a run of n consecutive tokens (word pieces). The oldest detector searches the training data for n-grams from each test item. The GPT-3 paper flagged an item if any 13-gram matched; [From Scratch Part 9](/blog/evaluate-base-model/) ran that check on its own 88M model. Later reports changed the details:

| Report | Rule | Notes |
|---|---|---|
| GPT-3 (2020) | any 13-gram in common | flags common phrases too |
| GPT-4 (2023) | any of three random 50-character substrings, spaces and symbols removed | question only, not the answer |
| Llama 2 (2023) | token counts as contaminated if inside a shared n-gram over 10 tokens, 4 mismatches allowed; "dirty" at 80% or more | thresholds swept |
| Llama 3 (2024) | share of tokens inside an 8-gram seen in pretraining; threshold chosen per benchmark | some benchmarks not estimable |

We ran the 13-gram and 8-gram versions against the exact token stream our contaminated model trained on (Step 2 describes it). Every verbatim item was found except three, one per group, that are shorter than 13 tokens (a five-word question with a one-word answer, for example), so they contain no 13-gram to find. The paraphrased items were another story:

| Group (234 items each) | Any 13-gram found | Half its 13-grams found | Any 8-gram found | Half its 8-grams found |
|---|---|---|---|---|
| Verbatim, any number of copies | 233 | 233 | 234 | 234 |
| Paraphrase, 16 copies | 73 | 1 | 182 | 5 |
| Never in training | 5 | 3 | 23 | 3 |

A single-collision rule like GPT-3's would flag 31% of the paraphrased items, mostly through their unchanged answers. A majority rule like Llama 2's "dirty" bucket flags one. The five "never in training" hits are a second lesson: ARC-Challenge contains near-duplicates of itself. One unseen question differs from a seen one only in writing a number as a word instead of a digit. And every n-gram check needs the training data, which only the model's builder has.

## Step 2: contaminating a model on purpose

Testing a detector needs items you know were trained on and items you know weren't. No public model offers that, so we made one.

The setup:

- Model: SmolLM2-135M (Hugging Face, Apache 2.0), already pretrained on about 2 trillion tokens.
- Ordinary text: 7.3 million tokens of FineWeb-Edu (sample-10BT, ODC-By license).
- Benchmark: the 1,172 ARC-Challenge test questions (grade-school science, CC-BY-SA-4.0), split at random (seed 0) into five groups of 234: never in training, verbatim once, verbatim 4 times, verbatim 16 times, and paraphrased 16 times. Claude Haiku 5.5 (thinking off) rewrote every question in new words for $0.02 in one batch; the paraphrased group trained on these, with the answer text unchanged.
- Format: each group, in the dataset's own order, cut into "pages" of 9 items written as `Question: ...` / `Answer: ...`, the way a quiz page or a benchmark file would hold them. Each copy of a page is one document, dropped at a random point in the stream. Benchmark text was 4.5% of the 7.68M tokens.
- Training: one pass, 468 steps of 16 sequences of 1,024 tokens, AdamW at a peak learning rate of 2e-4 with cosine decay, BF16 autocast, seed 0. It took 741 seconds on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) with PyTorch 2.11 and transformers 5.21.1.

Scoring follows lm-evaluation-harness's ARC setup: after `Question: <question>\nAnswer:`, each option's text is scored by its log-probability per byte, as in the harness's normalized accuracy, and the highest wins. The same 1,170 items were scored before and after training, so every comparison is paired.

![Dot chart of ARC-Challenge accuracy before and after training for five groups of 234 items. Never in training: 29.1% to 35.5%, a gain over the never-seen drift of 0.0 points by definition. Verbatim, 1 copy: 30.8% to 42.3%, gain +5.1 points (95% interval -2.6 to +13.2). Verbatim, 4 copies: 27.4% to 62.4%, gain +28.6 (+20.5 to +36.8). Verbatim, 16 copies: 32.9% to 97.4%, gain +58.1 (+50.0 to +66.2). Paraphrase, 16 copies: 27.4% to 82.9%, gain +49.1 (+41.0 to +57.3).](/diagrams/e05-score-gain.svg "ARC-Challenge accuracy before and after contaminated training, SmolLM2-135M. Intervals are 95%.")

Even the never-seen group improved by 6.4 points (95% interval 1.2 to 11.6), because the run also taught the question-and-answer format and more science text. So the fair measure of contamination is the gain beyond that drift: a difference in differences, the seen group's change minus the never-seen group's change, with a bootstrap interval over items.

One copy added 5.1 points beyond drift, with an interval that includes zero. Four copies added 28.6. Sixteen copies took the group to 97.4%, meaning the model had stored the answers outright: 151 items went from wrong to right and none from right to wrong. The paraphrased group gained 49.1 points without ever seeing the questions' original wording, because what the model learned was which answer goes with which topic.

This matches what [Carlini et al. (2022)](https://arxiv.org/abs/2202.07646) measured across model families: memorization "significantly grows" with model capacity, with the number of times an example is duplicated, and with how much context the prompt gives.

## Step 3: detectors that read the model's probabilities

With the weights in hand, you can ask the model how surprised it is by each token of a test item. A membership inference attack, borrowed from privacy research, turns those probabilities into a score: higher means "looks like training data". Four reference-free scores are standard, plus one that needs a second model.

Take the first ARC-Challenge question, which was in the 16-copy group: "An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?", answer "Planetary days will become shorter." In SmolLM2's tokens that is 36 tokens, so 35 have a log-probability (the first has no context).

| Score | What it computes | Before training | After 16 copies |
|---|---|---|---|
| Loss | mean log-probability per token | -3.29 | -1.03 |
| zlib ratio | model's total surprise in bits divided by the text's zlib-compressed size in bits, negated | -0.144 | -0.045 |
| Min-K% | mean log-probability of the 20% least likely tokens (7 of 35) | -7.73 | -2.98 |
| Min-K%++ | the same, after standardizing each token's log-probability by the model's expected log-probability at that position | -1.65 | +0.02 |
| Reference ratio | mean log-probability minus a reference model's | 0 | +2.27 |

After training, the hardest token for the model was " An", the first word of the question, at -6.56. Before, the worst seven tokens averaged -7.73. A never-seen item about photosynthesis barely moved: its mean went from -2.62 to -2.30.

Loss (perplexity in another form) is the baseline. The zlib ratio, from [Carlini et al. (2021)](https://arxiv.org/abs/2012.07805), discounts text that is simply easy, like repeated phrases, by comparing with a compressor that knows nothing about the training data. [Min-K% Prob](https://arxiv.org/abs/2310.16789) (Shi et al., 2023) bets that unseen text contains a few very surprising tokens while memorized text doesn't, and looks only at the worst 20%. [Min-K%++](https://arxiv.org/abs/2404.02936) (Zhang et al., 2024) asks how a token's log-probability compares with what the model expected at that position: with $$\mu_t$$ the average log-probability under the model's own next-token distribution and $$\sigma_t$$ its standard deviation, each token gets the z-score below, and the lowest 20% are averaged.

$$
z_t = \frac{\log p(x_t \mid x_{<t}) - \mu_t}{\sigma_t}
$$

The reference ratio compares with a model that did not see the data. Here we had the perfect reference, the same model before contamination. An outside auditor never has that.

### How well they separate seen from unseen

The yardstick is AUROC (area under the receiver operating characteristic curve): the probability that a randomly chosen seen item scores higher than a randomly chosen unseen one. 0.5 is a coin flip and 1.0 is perfect separation. Each seen group's 234 items were compared with the 234 never-seen items, with 2,000 bootstrap resamples for the interval.

![Four panels of dot plots, one per contamination group, showing AUROC with 95% intervals for five detectors. Verbatim, 1 copy: loss 0.565, zlib 0.583, Min-K% 0.575, Min-K%++ 0.564, reference ratio 0.669. Verbatim, 4 copies: 0.769, 0.754, 0.776, 0.829, 0.916. Verbatim, 16 copies: 0.971, 0.968, 0.971, 0.984, 0.994. Paraphrase, 16 copies, scored on the original wording: 0.737, 0.705, 0.717, 0.729, 0.877.](/diagrams/e05-detectors.svg "How well each detector tells a group's items from never-seen items, scored by the contaminated SmolLM2-135M. Intervals are 95%.")

At 16 copies every detector works, and Min-K%++ at 0.984 is the best of the reference-free four. At one copy, all four reference-free scores sit between 0.56 and 0.58. With the threshold set so that only 5% of never-seen items are falsely flagged, Min-K%++ catches 8% of the one-copy items and 91% of the 16-copy items. The same detectors run on the model before training gave AUROCs between 0.47 and 0.52 for every group, which is the check that the random split did its job.

The paraphrased items, scored on their original wording, came out at 0.71 to 0.74, helped by their answers appearing verbatim. That is after 16 copies, the dose at which the verbatim group was detected almost perfectly.

### The shuffled-order test

[Oren et al. (2023)](https://arxiv.org/abs/2310.17623) found a detector with a guarantee. If a model never saw a benchmark, "all orderings of an exchangeable benchmark should be equally likely": nothing about item 4 makes it more likely to follow item 3. A model that read the benchmark file has learned its order. So compare the model's log-probability for a page of items in the dataset's order with the same items shuffled. Under no contamination the difference averages zero, and the shuffles themselves supply the null distribution, so you don't have to guess what unseen text should look like.

We ran it on each group's 26 pages of 9 items, with 100 shuffles per page and a one-sided t-test across pages, as in Oren et al.'s sharded version.

| Group | Before training: p | After training: p | Mean log-probability gain of the true order |
|---|---|---|---|
| Never in training | 0.88 | 0.45 | 0.4 nats |
| Verbatim, 1 copy | 0.31 | 0.0001 | 8.4 nats |
| Verbatim, 4 copies | 0.13 | 2e-14 | 33.6 nats |
| Verbatim, 16 copies | 0.16 | 5e-18 | 44.9 nats |
| Paraphrase, 16 copies | 0.29 | 0.0015 | 7.9 nats |

A nat is the natural-log unit of probability: 8.4 nats means the true order became about 4,400 times more likely than a typical shuffle. The test caught a single copy that the per-item scores could barely see. In Oren et al.'s own experiment, a 1.4B model pretrained on 20.2 billion tokens, "we cannot reliably detect duplication rates of 1", with the threshold around 4. Our contamination was a much larger share of a much shorter run, and it came last, with nothing to dilute it. The test also has a blind spot built in: it only fires if the benchmark appeared in its canonical order and in a format close to the one you test.

## Step 4: off-the-shelf models and black-box probes

With a public model you don't know which items it saw, so you need a comparison set you're sure it didn't see. A popular choice is freshly written text.

We scored the original ARC-Challenge questions against Haiku 5.5's paraphrases of the same questions, written in November 2026 and so impossible to have been trained on, with four models. One of them is [From Scratch Part 7](/blog/pretraining-run/)'s 88M model, trained only on 1.76 billion tokens of FineWeb-Edu. The 13-gram check from Step 1, run against all of those tokens, found any overlap for only 6 of the 1,172 questions.

| Model | Loss | zlib ratio | Min-K% | Min-K%++ |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.648 | 0.633 | 0.646 | 0.632 |
| SmolLM2-135M | 0.622 | 0.599 | 0.617 | 0.626 |
| Pythia-160M | 0.606 | 0.568 | 0.586 | 0.602 |
| From Scratch 88M | 0.643 | 0.606 | 0.618 | 0.627 |

Every model "prefers" the originals, with AUROCs of 0.57 to 0.65 and intervals about ±0.015 wide (paired bootstrap over 1,172 items). The 88M model, which almost certainly never read ARC-Challenge, scores as high as Qwen2.5-0.5B. What the detectors pick up is the difference between a teacher's wording and a language model's rewording. Length doesn't explain it: the paraphrases average 27.1 tokens and the originals 26.2.

This is the failure [Duan et al. (2024)](https://arxiv.org/abs/2402.07841) documented. On a benchmark where members and non-members come from the same distribution, Pythia's own training data, they found that attacks "barely outperform random guessing": for Pythia-dedup 12B on Wikipedia, loss 0.516 and Min-K% 0.517. On a version where non-members are newer Wikipedia pages, the same attacks reach 0.680 and 0.719. Duan et al. attribute the stronger results on earlier benchmarks, which pitted old text against new, to that kind of shift.

The shuffled-order test doesn't have this problem, because it compares a benchmark with itself. On Qwen2.5-0.5B (Apache 2.0) we ran it on the first 1,170 ARC-Challenge test questions in the dataset's order: 130 pages of 9, 20 shuffles each. The true order scored 1.2 nats higher on average, with p = 0.053, and 8 of the 130 pages were individually significant at 5%, about the 6.5 that chance alone would give. So there is no evidence of verbatim, in-order contamination, though the test can't rule out the rest: Qwen may have seen ARC in another format or order, and a 0.5B model memorizes less than the 7B models in Oren et al.'s tests. (A run on MMLU's test file hit our GPU time limit and is not reported.)

### Black-box probes on Claude Haiku 5.5

API models return text only, so you probe by asking. Guided completion ([Golchin and Surdeanu, 2023](https://arxiv.org/abs/2308.08493)) gives the model the first part of a test item, names the dataset, and asks for the rest; a near-exact completion suggests memorization. TS-Guessing ([Deng et al., 2023](https://arxiv.org/abs/2311.09783)) hides one wrong option of a multiple-choice item and asks the model to fill it in.

We ran both on Haiku 5.5 (thinking off, sent as `{"type": "disabled"}`), in one Message Batch of 2,400 requests that cost $0.05: 300 ARC-Challenge questions cut at a random point between 40% and 60% of their words, and 300 items each from ARC-Challenge and the 1,140-item MMLU subset used since [Part 3](/blog/eval-statistics/). Each probe had a control. For guided completion: the same prompt without the dataset's name, and the guided prompt on the start of each question's November 2026 paraphrase.

| Probe (300 items each) | Declined | Exact match among answers | Mean word overlap (ROUGE-L) |
|---|---|---|---|
| Golchin's wording ("exactly as it appears") | 145 (48%) | 0 of 155 | 0.248 |
| Dataset named, softer wording | 32 (11%) | 0 of 268 | 0.214 |
| Dataset not named | 3 (1%) | 1 of 297 | 0.181 |
| Dataset named, paraphrased question | 38 (13%) | 0 of 262 | 0.189 |
| TS-Guessing, MMLU, "exact text" | 33 (11%) | 45 of 267 (17%) | not scored |
| TS-Guessing, ARC, "exact text" | 56 (19%) | 32 of 244 (13%) | not scored |
| TS-Guessing, softer wording, MMLU and ARC | 576 of 600 (96%) | 3 of 24 | not scored |

Declines were found by matching phrases such as "I can't" and "I cannot" in the first 300 characters, spot-checked by hand. Up to a quarter of the TS-Guessing replies ran past the 60-token limit, mostly while explaining, so every reply was scored on its first line. ROUGE-L is the F1 of the longest common word subsequence between the completion and the true remainder.

Naming the dataset raised ROUGE-L by 0.029 (95% interval 0.014 to 0.043) on the same items, and the named-dataset prompt overlapped the real question more than it overlapped the paraphrase by 0.023 (0.003 to 0.042). None of 423 completions under a named dataset reproduced the rest of a question exactly.

All ten TS-Guessing exact matches with five or more words complete a pattern the other options make obvious: when three options cover three of the four combinations of two bird populations rising or falling, the hidden fourth writes itself. Many short matches are MMLU's "False, False" and "Wrong, Not wrong" grids.

Haiku also declined 48% of the prompts that asked for the continuation "exactly as it appears", usually saying it shouldn't reproduce test material verbatim, and 96% of the softer TS-Guessing prompts. A model trained to decline reciting benchmarks looks clean on these probes whatever it memorized, so a negative result says little. Haiku 5.5's training data runs to June 2026 ([Anthropic models overview](https://platform.claude.com/docs/en/models/overview), read November 4, 2026), years after both benchmarks were published.

![Flowchart from the question "Did the model see the test?" to four kinds of access, each with its detection method. Training data in hand: n-gram overlap. Weights in hand: membership scores (loss, zlib, Min-K%) and the shuffled-order test. Text-only API: guided completion and TS-Guessing. No access: a fresh copy of the test, such as GSM1k.](/diagrams/e05-detection-access.svg "Which detector you can use depends on what you can access.")

The less access you have, the weaker the evidence: only the training data gives a direct answer, and only a fresh test measures the damage rather than the exposure.

## In the real world

- The [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) (OpenAI, 2023) found about 0.6% of MMLU, 1% of GSM8K and 25% of HumanEval overlapping its training data by its substring rule, and concluded that "contamination overall has very little effect on the reported results". HellaSwag was not checked; it was scored on a private holdout.
- [Llama 2](https://arxiv.org/abs/2307.09288) (Touvron et al., 2023) concluded that "only HellaSwag and MMLU-Humanities appear to have been boosted due to contamination", more for 70B than 7B. On MMLU-Humanities the 70B model scored 62.2 on clean items and 85.8 on dirty ones.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Meta, 2024) estimated 85% of HellaSwag and 55% of PiQA as contaminated by 8-gram overlap, with estimated gains of 14.3 and 8.1 points for the 405B model. For MMLU and HumanEval, "8-gram overlap gives such high contamination scores that it is impossible to get a good performance gain estimate."
- [Oren et al. (2023)](https://arxiv.org/abs/2310.17623) ran the shuffled-order test on LLaMA2-7B, Mistral-7B, Pythia-1.4B and GPT-2 XL: "we do not find evidence of pervasive verbatim test set contamination", with ARC on Mistral (p = 0.001) right at the edge of significance after correcting for many tests. These are reported by the authors, not reproduced here.
- [GSM1k](https://arxiv.org/abs/2405.00332) (Zhang et al., 2024) found Phi, Mistral and some Llama models overfitting GSM8K while "Gemini, GPT, and Claude show little to no signs of overfitting", and a Spearman correlation of 0.36 between a model's likelihood of generating GSM8K items and its drop on GSM1k. Its authors chose not to release GSM1k at publication, to keep it clean.
- Golchin and Surdeanu report that their best method agrees with human experts 92% to 100% of the time, and that "GPT-4 is contaminated with AG News, WNLI, and XSum". Deng et al.'s abstract reports exact-match rates of 52% and 57% for ChatGPT and GPT-4 guessing hidden MMLU options.
- [Dekoninck et al. (2024)](https://arxiv.org/abs/2402.02823) showed contamination can be made invisible on purpose: fine-tuning on rephrased benchmark items inflated scores by up to 15% while "no method achieves more than 1.6% TPR@1%FPR".

## Common misconceptions

- **"An n-gram check shows a benchmark is clean."** It shows the exact text wasn't there. Our paraphrased items gained 49 points while a majority-overlap rule flagged 1 of 234, and the check needs training data that only the model's builder has.
- **"Min-K% can audit any open model."** Against a freshly written comparison set, it gave an 88M model that never saw ARC-Challenge an AUROC of 0.61 to 0.64. Without members and non-members from the same distribution, these scores measure style and date.
- **"A little contamination always shows up in the score."** One verbatim copy added 5.1 points with an interval from -2.6 to +13.2, and four copies added 28.6. The damage grows with exposure, and so does detectability.
- **"If a model had memorized a test, it would recite it when asked."** Haiku 5.5 declined up to 96% of such requests. Silence from a black-box probe is not evidence of a clean model.
- **"A shuffled-order test that finds nothing clears the model."** It only catches verbatim copies in the benchmark's own order and a similar format. Paraphrases, reformatted files and synthetic data slip past it.

## Key takeaways

- Contamination comes in four kinds: exact copies, near-duplicates, paraphrases and synthetic data. All of them inflate scores; only the first two are easy to find.
- In our controlled run, 16 verbatim copies took 234 questions from 33% to 97%, and 16 paraphrased copies added 49 points that a 13-gram check mostly missed.
- Probability-based detectors work when contamination is heavy and when you have a clean comparison set. On real models, the comparison set is the hard part, and a style shift alone gives an AUROC of about 0.6.
- The shuffled-order test brings its own null distribution, so a style shift can't fool it, but it only sees verbatim copies in the original order.
- Black-box probes on API models are weak: models decline to recite tests, and guessable options produce false matches. A fresh, private copy of the test, like GSM1k, measures the damage directly.

## What's next

Multiple-choice and number answers can be checked by a program. Most of what people ask language models can't: summaries, explanations, advice. Part 6 measures how well reference metrics like ROUGE and BERTScore, rubric graders and LLM judges agree with human ratings of open-ended answers.

## Further reading

- [Proving Test Set Contamination in Black Box Language Models (Oren et al., 2023)](https://arxiv.org/abs/2310.17623): the exchangeability test, its false-positive guarantee, and a pretraining experiment at known duplication counts.
- [Detecting Pretraining Data from Large Language Models (Shi et al., 2023)](https://arxiv.org/abs/2310.16789): Min-K% Prob and the WikiMIA benchmark.
- [Min-K%++: Improved Baseline for Detecting Pre-training Data from Large Language Models (Zhang et al., 2024)](https://arxiv.org/abs/2404.02936): standardizing each token's log-probability by the model's own distribution.
- [Do Membership Inference Attacks Work on Large Language Models? (Duan et al., 2024)](https://arxiv.org/abs/2402.07841): why these attacks look strong on temporally shifted data and near random on matched data.
- [Extracting Training Data from Large Language Models (Carlini et al., 2021)](https://arxiv.org/abs/2012.07805): the zlib and reference-model ratios.
- [Quantifying Memorization Across Neural Language Models (Carlini et al., 2022)](https://arxiv.org/abs/2202.07646): memorization grows with model size, duplication and context.
- [Time Travel in LLMs: Tracing Data Contamination in Large Language Models (Golchin and Surdeanu, 2023)](https://arxiv.org/abs/2308.08493): guided versus general instructions.
- [Investigating Data Contamination in Modern Benchmarks for Large Language Models (Deng et al., 2023)](https://arxiv.org/abs/2311.09783): TS-Guessing.
- [A Careful Examination of Large Language Model Performance on Grade School Arithmetic (Zhang et al., 2024)](https://arxiv.org/abs/2405.00332): GSM1k, a fresh copy of GSM8K.
- [Rethinking Benchmark and Contamination for Language Models with Rephrased Samples (Yang et al., 2023)](https://arxiv.org/abs/2311.04850): paraphrased test sets that n-gram checks can't see.
- [Evading Data Contamination Detection for Language Models is (too) Easy (Dekoninck et al., 2024)](https://arxiv.org/abs/2402.02823): contamination designed to evade every detector.
- The contamination appendices of [GPT-4](https://arxiv.org/abs/2303.08774) (Appendix C and D), [Llama 2](https://arxiv.org/abs/2307.09288) (A.6) and [Llama 3](https://arxiv.org/abs/2407.21783) (section 5.1.4).
