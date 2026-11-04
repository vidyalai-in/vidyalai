---
title: "Prompt and Harness Sensitivity: Same Model, Same Items, New Score"
description: "Format, few-shot examples, option order, scoring method and answer extraction all move MMLU scores. Measured on four open models, Haiku 5.5 and Sonnet 5.5."
slug: "eval-harness-sensitivity"
category: "evaluation-reliability"
tags: ["evals", "benchmarks", "statistics", "reproducibility"]
series: "llm-evals"
seriesOrder: 4
pubDate: 2026-09-09
updatedDate: 2026-11-04
---

## TL;DR

A benchmark score is produced by a model and a harness together: the program that writes the prompt, reads the reply and decides whether it is right. This part changes one harness choice at a time on the same 1,140 MMLU questions and measures how far the score moves, as a paired difference with a 95% confidence interval. On four open models, the prompt layout alone moved zero-shot accuracy by up to 11 points, and switching from letter to cloze scoring reversed which of two models was better, with the gap significant both ways. On Claude Haiku 5.5, rewording the prompt or reordering the options moved the score by at most 1.2 points, with every interval crossing zero, but asking for "only the letter" cost 9.3 points against letting it explain. Scoring the same Haiku and Sonnet 5.5 replies with five answer extractors put Sonnet anywhere from 53 points behind Haiku to 30 points ahead.

## Why it matters

In June 2023, the harness behind Hugging Face's Open LLM Leaderboard gave LLaMA-65B 48.8% on MMLU. Meta's [LLaMA paper](https://arxiv.org/abs/2302.13971) reported 63.4%. Nobody had changed the model or the questions.

[Fourrier et al.](https://huggingface.co/blog/open-llm-leaderboard-mmlu) tracked the gap to three programs that all claimed to run MMLU. The original code compared the probabilities of the letters A to D, Stanford's HELM (Holistic Evaluation of Language Models) had the model generate a letter, and EleutherAI's harness at the time scored each full answer. Five-shot, LLaMA-65B got 0.636, 0.637 and 0.488. Falcon-40B beat LLaMA-65B under one implementation and lost under the other two.

Part 3 treated a score as an estimate with an error bar from the choice of items. This part looks at a different kind of error: the choices the person running the benchmark makes, which no error bar over items covers. In our own runs, the same 285 free-form replies from Haiku 5.5 score 0.0%, 54.4% or 86.7%, depending only on the code that reads the answer.

## The intuition

Picture one student taking the same history exam three ways: a bubble sheet, an essay graded by a strict examiner who wants the letter in the first word, and an oral exam where the examiner listens to the whole explanation. The student knows exactly the same history each time. The three marks can still differ, because each format tests a little extra: neat bubbling, following instructions, or talking through a problem.

A benchmark harness is the format and the examiner. It picks the wording of the prompt, how many solved examples come first, the order of the options, whether the model writes or is only asked for probabilities, and how a written reply becomes a letter.

The analogy breaks in two places. A student doesn't get better at history when the options are listed as 1-4 instead of A-D, but a language model's choice can change, because the label is a token it has seen in different contexts with different frequencies. And the student can't be re-run: a model can, so every harness effect here can be measured exactly, as a paired difference on the same items.

## Step 1: three ways to score one question

An MMLU item is a question with four options, one of them keyed as correct. [MMLU](https://arxiv.org/abs/2009.03300) (Massive Multitask Language Understanding) has 14,042 test items in 57 subjects. Every experiment here uses the same fixed subset shared across this series: 20 items from each subject, 1,140 in all, drawn with seed 0 from `cais/mmlu` (MIT license).

![Flowchart. One MMLU item, a question with four options, can be scored three ways. If the prompt shows the options labeled A to D, the harness can compare the model's probabilities of the next token being A, B, C or D (letter log-prob), or let the model generate a reply that an extractor turns into a letter; generation is the only path for API models. If the prompt shows only the question, the harness scores each option's text as a continuation (cloze log-prob). All three paths end by picking the best option and comparing it with the answer key.](/diagrams/e04-scoring-paths.svg "Three scoring paths for the same multiple-choice item.")

The letter and cloze paths need log-probabilities, the logarithm of the probability the model assigns to each possible next token ([Math Part 4](/blog/probability-language-models/) covers them). Only open models give you those. The Claude API returns text and token counts, not log-probabilities, so Haiku and Sonnet can only be scored by generation followed by extraction.

Here is one item scored all three ways by Qwen2.5-0.5B, five-shot, with lm-eval's MMLU prompt. The question is from elementary mathematics: a coach rounded the number of runners to the nearest 10 and got 400; which could be the real number, 382, 397, 406 or 447? The key is B, 397.

| Method | A (382) | B (397) | C (406) | D (447) | Pick |
|---|---|---|---|---|---|
| Letter log-prob, ln p(" A") etc. | −1.928 | −1.101 | −1.096 | −1.693 | C (wrong) |
| Cloze, ln p of " 382" etc. (sum over tokens) | −9.38 | −6.22 | −6.24 | −11.27 | B (right) |
| Greedy generation after "Answer:" | not scored | not scored | not scored | not scored | " C" (wrong) |

The model is torn between 397 and 406 whichever way you ask, and the method decides which side of the coin it lands on. The letter scores of B and C differ by 0.005 nats, a natural-log unit. That's smaller than the rounding noise of bfloat16, the 16-bit number format the model runs in: computing the same prompt in one pass instead of reusing the cached examples gave B −1.088 and C −1.117, and the pick flipped to B.

A few-shot prompt puts k solved examples before the question; MMLU's convention is five, taken from its 285-item dev split (5 per subject). And the cloze sum favors short options, because every extra token adds a negative log-probability, so harnesses also report a length-normalized version that divides by the number of characters ([From Scratch Part 9](/blog/evaluate-base-model/) has the details: lm-eval calls it `acc_norm`).

## Step 2: four open models, 21 harness variants

Four open models ran every combination below on all 1,140 items: SmolLM2-135M, Qwen2.5-0.5B, Qwen2.5-0.5B-Instruct (all Apache 2.0) and Llama-3.2-1B (Llama 3.2 Community License), in bfloat16 on an RTX 4060 Laptop GPU with transformers 5.21.1. All four together took 31 minutes of GPU time.

- Letter scoring with five prompt formats (lm-eval's MMLU layout; "Question:" with (A) labels; "Choices:" with A: labels; labels 1-4; all lowercase with a-d), each zero-shot and five-shot.
- The default format with 1 or 5 examples from three draws: MMLU's dev split and two random draws from its validation split.
- Cloze scoring, zero- and five-shot, by summed and by per-character log-probability.
- Greedy generation of up to 12 tokens after "Answer:", stopped at a newline, read by the extractors of Step 4.

That is 21 scored variants per model. The code that reuses the examples' KV cache (the stored attention keys and values of the shared prompt) for every item in a subject is in `mcq.py`; in 32-bit floats it matches a plain forward pass to four decimals.

![Dot-and-band chart of accuracy on 1,140 MMLU items for four open models, three scoring methods each, with a dashed chance line at 25%. Bands span all prompt variants tried; dots are the five-shot prompt with 95% intervals. SmolLM2-135M: letter 23.0 to 26.7% (five-shot 25.2), cloze 30.4 to 32.2% (32.2), generate 21.5 to 25.5% (25.5). Qwen2.5-0.5B: letter 40.1 to 50.2% (48.3), cloze 33.4 to 36.6% (36.6), generate 49.2 to 49.7% (49.2). Qwen2.5-0.5B-Instruct: letter 36.0 to 49.3% (48.3), cloze 31.8 to 37.1% (37.1), generate 46.4 to 48.3% (48.3). Llama-3.2-1B: letter 29.0 to 42.4% (32.4), cloze 33.9 to 39.8% (39.8), generate 31.9 to 40.5% (31.9).](/diagrams/e04-open-models.svg "Four open models, three scoring methods, 21 prompt variants each.")

The scoring method changes which model is best. Under lm-eval's default (letter scoring, five-shot), Llama-3.2-1B trails Qwen2.5-0.5B by 16.0 points (95% CI −20.1 to −11.8). Under five-shot cloze scoring, per character, it leads by 3.2 points (+0.6 to +5.9). Both intervals exclude zero. Across the 21 variants, Qwen2.5-0.5B ranks first in 13, its Instruct sibling in 4 and Llama in 4, all of them cloze. A leaderboard that switched from one method to the other would reverse two models' order, with significance on both sides.

| Model | Zero-shot, 5 formats | Five-shot, 5 formats | 5-shot minus 0-shot | Five-shot, 3 draws |
|---|---|---|---|---|
| SmolLM2-135M | 23.1 to 26.7 | 23.0 to 25.5 | +2.1 (−1.8 to +6.0) | 24.7 to 25.8 |
| Qwen2.5-0.5B | 40.1 to 49.9 | 46.7 to 49.2 | −1.6 (−4.0 to +0.8) | 48.3 to 50.2 |
| Qwen2.5-0.5B-Instruct | 36.0 to 47.1 | 46.2 to 48.3 | +1.2 (−1.2 to +3.6) | 48.3 to 49.3 |
| Llama-3.2-1B | 31.2 to 42.4 | 29.0 to 32.4 | −8.5 (−12.0 to −5.0) | 32.0 to 34.4 |

Letter-scoring accuracy in percent. The difference column is for the default format and is paired: the same items scored both ways, so item difficulty cancels. Its 95% interval is clustered by subject, which allows for items in one subject succeeding or failing together.

Zero-shot, formatting alone moved the three larger models by 9.8 to 11.1 points between their best and worst layouts, each gap well outside its interval (Qwen2.5-0.5B: +9.8, 95% CI +6.2 to +13.4). Picking the best of five formats overstates the true spread a little, since the maximum of five noisy numbers is biased upward, but the run-to-run noise here is zero: these scores are deterministic. Five solved examples shrank the format spread to 2.1 to 3.3 points, probably because they show the model what kind of answer the layout expects.

The examples themselves mattered less than having some. Across three different five-example draws, no model moved by more than 2.4 points, and only one draw-to-draw gap reached the edge of its interval (Qwen2.5-0.5B, +1.8, 95% CI 0.0 to +3.7).

Llama-3.2-1B is the exception, and its letter choices show why. Zero-shot, it picked D for 17% of items; five-shot, for 7%, although D is the key for 26.8% of these items. After five examples it rarely says D at all, and its score fell 8.5 points. The small models' label preferences are extreme: SmolLM2-135M answered "1" on 91% of items when the options were numbered. Its letter scores sit at chance, 23 to 27%, while its cloze score of 32.2% is clearly above it. A model that can't yet map knowledge onto a label still shows the knowledge when you score the answer text, which is why [From Scratch Part 9](/blog/evaluate-base-model/) scored its 88M model that way.

Numeric precision adds noise that published numbers never show. Scoring the same five-shot prompts in one pass instead of through the cached examples changed the picked option on 26 to 75 of the 1,140 items, depending on the model, purely through bfloat16 rounding. The accuracy moved by at most 0.5 points, but individual answers did flip, like the rounding question above.

## Step 3: Haiku and Sonnet, through the API

The API experiments used the Message Batches API, one request per item and prompt, with the API's default sampling settings, so replies can differ from run to run. Thinking was switched off explicitly. That needed care: with the `thinking` field left out, Haiku 5.5 thought anyway and spent its whole 16-token pilot budget on hidden thinking, returning no text at all. Haiku takes `{"type": "disabled"}`; Sonnet 5.5 rejects that and asks for `{"type": "between_tools"}`, under which, per the API's error message, "the model does not think before responding".

The baseline prompt is lm-eval's MMLU layout (the subject line, the question, `A.` to `D.`, then `Answer:`) as the user message, with the system prompt "Reply with only the label of the correct option." and `max_tokens` 64. Every other condition changes one thing.

```python
import mcq   # evals/lab/mcq.py
item = {"subject": "elementary_mathematics", "answer": 1,
        "question": "A coach rounded the number of runners at a track meet to the nearest 10. "
                    "The rounded number of runners is 400. Which number could be the actual "
                    "number of runners at the track meet?",
        "choices": ["382", "397", "406", "447"]}
shared, own = mcq.prompt(mcq.FORMATS["num"], mcq.rotate(item, 1))
print(shared + own)
```

```text
Question: A coach rounded the number of runners at a track meet to the nearest 10. The rounded number of runners is 400. Which number could be the actual number of runners at the track meet?
1. 447
2. 382
3. 397
4. 406
Answer:
```

To know what "no effect" looks like, the baseline was sent twice. Because replies are sampled, the two runs disagreed on 4.5% of items, and the score moved by −0.7 points (95% CI −1.6 to +0.2). Anything inside that band is indistinguishable from re-running the same eval. [Agentic AI Part 11](/blog/agent-evals/) found the same kind of run-to-run spread in agent tasks.

![Dot-and-whisker chart of Haiku 5.5 accuracy changes against the baseline prompt (78.8%) on 1,140 MMLU items, each a paired difference with a 95% interval clustered by subject. Sending the same prompt again: −0.7 points (−1.6 to +0.2). Four other formats: "Question:" with (A) labels +0.8, "Choices:" with A: labels +0.4, labels 1-4 −1.2, lowercase with a-d labels +0.5, all intervals crossing zero. Rotating the options by one, two or three places: −0.6, +0.2 and +0.4, all crossing zero. Five solved examples first: −0.3, crossing zero. Removing the answer-only instruction so the model writes a free reply: +9.3 (+5.3 to +13.3), and with five examples +8.9 (+4.7 to +13.0).](/diagrams/e04-haiku-variants.svg "Haiku 5.5: one change at a time, against the answer-only baseline.")

Format didn't matter for Haiku. Four rewordings, three rotations of the options and five solved examples all moved the score by less than 1.3 points, and every interval crosses zero. That's the opposite of the open models in Step 2. It doesn't contradict [Sclar et al.](https://arxiv.org/abs/2310.11324), who found large spreads in bigger and instruction-tuned models too: they searched hundreds of formats on other tasks, while these four rewordings are tame by comparison.

What mattered was whether Haiku could think out loud. Without the instruction, its replies ran a median of 144 output tokens and reached 88.1% (95% CI 86.1 to 89.8). For the rounding question above, the answer-only prompt got "C" (406, wrong) in all five formats; the free reply worked through each option and ended "**Answer: B. 397**".

Two things make up the 9.3-point gap. First, Haiku ignored "only the label" on 59 of 1,140 items, started reasoning and was cut off at 64 tokens. A truncated reply has no answer, so it counts as wrong. Second, on the 1,076 items the answer-only prompt answered directly, the free reply was still 5.3 points better (95% CI +2.8 to +7.8).

Sonnet 5.5 on 285 of the items (5 per subject) shows how model-specific this is. Its free replies beat its answer-only replies by 7.4 points (95% CI +2.3 to +12.4), but on the 262 items it answered directly the two were identical, 93.1% each. All of Sonnet's gap came from the 22 replies the 64-token cap cut short; its free replies to those same 22 questions were right 21 times. Whether those items count as errors depends on a `max_tokens` setting that is easy to leave out of a methods section.

### Moving the right answer

[Zheng et al.](https://arxiv.org/abs/2309.03882) found that models "prefer to select specific option IDs as answers (like 'Option A')". Moving the correct answers of 0-shot MMLU to position D cost gpt-3.5-turbo 6.3 points (67.2 to 60.9), and llama-30b gained 15.2 when they were moved to A. They called the cause token bias: a prior on the label tokens themselves.

The test here rotates each item's options through all four places, so every item has its correct answer once at A, B, C and D. Haiku's accuracy was 79.5% with the answer at A and 76.5% with it at D, a paired gap of 3.0 points (95% CI +0.6 to +5.4). Among the 4,355 answered replies, it picked D 23.5% of the time when D was right exactly 25% of the time. The bias is real but small, and averaging over rotations removes it.

Consistency across orders separates the models more. Haiku chose the same option in all four orders on only 72.6% of the 285-item subset, against 85.3% for Sonnet. Counting an item as right only if it is right in all four orders, the idea behind [MMBench](https://arxiv.org/abs/2307.06281)'s CircularEval, Haiku drops from 78.2% averaged over rotations to 66.7%, and Sonnet from 86.9% to 81.4%. The gap between the two models grows from 8.7 points to 14.7 under the stricter rule.

## Step 4: reading the answer

A free reply has to be turned into a letter by an extractor: a rule that reads text and returns an option or nothing. `mcq.py` has five:

| Extractor | Rule | Example it accepts |
|---|---|---|
| `strict` | the whole reply is one label | `B` |
| `first` | the reply starts with a label | `B. Charles Lyell ...` |
| `answer_is` | the last "answer is X" or "Answer: X" | `The answer is **C**.` |
| `any` | the first standalone A-D anywhere | `... option (C) describes ...` |
| `flex` | strict, then first, then answer_is, then a bold or bracketed label, then a reply naming exactly one option's text | most of the above |

`answer_is` mirrors lm-eval's chain-of-thought MMLU task, whose "strict-match" filter looks for text after "The answer is", "answer is" or "The answer:". lm-eval's plain generative MMLU task keeps only the first line and compares it with the letter, which is close to `strict`.

![Grouped bar chart of accuracy on the same 285 free-form replies from each model, scored five ways, with 95% intervals. strict: Haiku 0.0%, Sonnet 0.0%, difference 0. first: Haiku 54.4%, Sonnet 1.4%, Sonnet minus Haiku −53.0 points (−59.9 to −46.1). answer is: Haiku 33.0%, Sonnet 62.5%, +29.5 (+16.6 to +42.3). any: Haiku 84.6%, Sonnet 81.1%, −3.5 (−9.3 to +2.3). flex: Haiku 86.7%, Sonnet 93.3%, +6.7 (+2.5 to +10.9).](/diagrams/e04-extractors.svg "The same replies, five extractors: who wins depends on the extractor.")

No request was re-sent. Sonnet 5.5 is 53 points behind Haiku under `first`, 30 points ahead under `answer_is`, tied under `any` and 6.7 points ahead under `flex`. The models write differently. 61% of Haiku's 1,140 free replies open with the answer ("B. Charles Lyell" followed by an explanation). Only 8 of Sonnet's 285 do; it opens with "The answer is" or reasons first and ends with "Answer: B". An extractor tuned to one style penalizes the other.

Each lenient rule has its own failure. `any` takes the first lone capital A-D it sees, so a reply that mentions another option before giving its answer is scored as that option. On Haiku's 1,140 free replies it disagreed with `flex` on 31 items, and on 27 of those `flex` had the keyed answer. Order matters too. One five-shot Haiku reply to the rounding question began "**Answer: D**" and ended "the correct answer is **B**". A first-match rule scores D, a last-match rule B.

Open models, stopped at the first newline, reply with a single letter almost every time, so `strict`, `first`, `any` and `flex` agree on them to within 0.2 points (`answer_is` finds nothing, since nobody writes "answer is"). Extraction becomes a measurement choice only once a model writes sentences.

Strictness also interacts with the prompt. With the answer-only instruction, Sonnet often wrote the label and the option text, like "A. Acculturation", which `strict` rejects: its answer-only accuracy was 80.0% strict and 86.0% flex. For Haiku the same two rules gave 78.2% and 78.8%.

## In the real world

The 2023 leaderboard episode is the cleanest public case. Here are four rows of the table in [Fourrier et al.](https://huggingface.co/blog/open-llm-leaderboard-mmlu), all five-shot MMLU:

| Model | Original (letter log-prob) | HELM (generate a letter) | Harness, early 2023 (full answer log-prob) |
|---|---|---|---|
| llama-65b | 0.636 | 0.637 | 0.488 |
| falcon-40b | 0.558 | 0.571 | 0.527 |
| llama-30b | 0.584 | 0.583 | 0.457 |
| gpt-neox-20b | 0.262 | 0.256 | 0.333 |

The full-answer method lowered the LLaMA models by more than 12 points and raised GPT-NeoX-20B by 7. The post says the implementations "even change the ranking order of the models on the leaderboard". The harness was later changed to match the original implementation, which is why lm-eval's default MMLU task today compares the letters A to D after the Hendrycks prompt.

[Biderman et al.](https://arxiv.org/abs/2405.14782), written by lm-eval's maintainers, measured the same split on ARC-Challenge, zero-shot. In cloze form Mistral-7B scored 50.1 and GPT-NeoX-20B 38.0; with options shown, MMLU-style, Mistral rose to 72.4 and GPT-NeoX fell to 26.6. Their conclusion: "the selection of prompt affects the model rankings". They also note that averaging MMLU over items or over the 57 subjects "affects results by several percentage points". Their checklist asks for exact prompts, model outputs and significance tests with every reported number.

[Sclar et al.](https://arxiv.org/abs/2310.11324) varied only formatting (separators, spacing, casing) on Super-NaturalInstructions tasks, not MMLU. They found "performance differences of up to 76 accuracy points" for LLaMA-2-13B in few-shot settings, and the sensitivity "remains even when increasing model size, the number of few-shot examples, or performing instruction tuning". Their tool, FormatSpread, reports the range of accuracy over a sample of plausible formats instead of one number. [Alzahrani et al.](https://arxiv.org/abs/2402.01781) applied the same idea to a leaderboard: shuffling options, swapping label symbols or switching to cloze scoring moved models "up to 8 positions".

Model cards rarely give enough detail to reproduce a number. Meta's [Llama 3.2 card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) reports 32.2 on five-shot MMLU for the 1B base model, labeled "macro_avg/acc_char" and computed with Meta's internal evaluation library: a macro average, and a metric name a reader has to decode. Qwen's [Qwen2.5 announcement](https://qwenlm.github.io/blog/qwen2.5-llm/) (September 2024) gives Qwen2.5-0.5B 47.5 on five-shot MMLU without saying how it was scored. Our letter scores happen to land near both cards, 32.4% for Llama-3.2-1B and 48.3% for Qwen2.5-0.5B, but on 8% of the test set and with a cloze score that tells a different story, the agreement says little.

## Common misconceptions

- **"MMLU accuracy is a property of the model."** It's a property of the model and the harness. The 2023 leaderboard gave LLaMA-65B 63.7% or 48.8% depending on the implementation, and our Qwen2.5-0.5B scores 36.6% with cloze scoring and 48.3% with letter scoring on the same items.
- **"Prompt sensitivity is a small-model problem that bigger models grow out of."** Haiku 5.5 barely noticed four rewordings and three option orders, but an "answer with only the letter" instruction cost it 9.3 points, and Sonnet 5.5's lead over Haiku ran from −53 to +30 points depending on the extractor. Bigger models move the sensitivity somewhere else.
- **"A lenient answer parser is the fair choice."** Lenient rules credit answers the model didn't give. The `any` rule scored the first stray capital letter and disagreed with a careful parser on 31 of Haiku's free replies, and was wrong on 27 of them; under it, Sonnet's 6.7-point lead under `flex` disappeared.
- **"Option-order bias means models like the letter A."** Zheng et al. found the preferred label varies by model: llama-30B over-picked A, gpt-3.5-turbo over-picked C. In our rotations Haiku 5.5 slightly under-picked D, and Llama-3.2-1B, five-shot, chose D for only 7% of items. Rotating the options and averaging measures the bias and removes it.
- **"Five-shot is the standard, so five-shot numbers are comparable."** Which five examples, in what order, from which split, and separated how? Changing the draw moved our open models by up to 2.4 points, and Qwen's 5-shot announcement doesn't say how it scored.

## Key takeaways

- A benchmark score is a joint measurement of a model and a harness. Change the prompt format, the examples, the option order, the scoring method or the answer parser, and the number can move more than the gap between two models.
- Measure every harness change as a paired difference on the same items, and compare it with the run-to-run noise of sending the identical prompt twice. For Haiku 5.5 that noise was 0.7 points.
- Small open models are fragile to formatting: Qwen2.5-0.5B's zero-shot score moved 9.8 points between two prompt layouts. Five solved examples shrank that spread to 2.5 points, and switching to cloze scoring reversed the order of Llama-3.2-1B and Qwen2.5-0.5B.
- For API models, the instruction and the extractor matter most. "Only the letter" cost Haiku 9.3 points, and five extractors applied to the same replies flipped which of Haiku and Sonnet looked better.
- Report the harness with the score: exact prompts, shots, scoring method, extraction rule, `max_tokens`, thinking setting, and the raw outputs. Biderman et al.'s checklist is a good template, and the same habit applies to a product eval suite like the one in [System Design: Building an Eval Suite](/blog/support-bot-eval-suite/).

## What's next

Even with a fixed, published harness, a score can be inflated by something no prompt choice controls: the test items, or text very close to them, may have been in the training data. Part 5 tests methods for detecting contamination and memorization on open models.

## Further reading

- [What's going on with the Open LLM Leaderboard? (Fourrier, Habib, Launay and Wolf, 2023)](https://huggingface.co/blog/open-llm-leaderboard-mmlu): three MMLU implementations, three scores for the same model, rankings changed.
- [Lessons from the Trenches on Reproducible Evaluation of Language Models (Biderman et al., 2024)](https://arxiv.org/abs/2405.14782): cloze vs MMLU-style formulations, and a checklist for reporting evaluations, from lm-evaluation-harness's maintainers.
- [Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design (Sclar et al., 2023)](https://arxiv.org/abs/2310.11324): up to 76 accuracy points from formatting alone, and FormatSpread for reporting a range.
- [Large Language Models Are Not Robust Multiple Choice Selectors (Zheng et al., 2023)](https://arxiv.org/abs/2309.03882): selection bias, token bias and PriDe's permutation-based debiasing.
- [When Benchmarks are Targets (Alzahrani et al., 2024)](https://arxiv.org/abs/2402.01781): small perturbations move leaderboard rankings by up to 8 positions.
- [Measuring Massive Multitask Language Understanding (Hendrycks et al., 2020)](https://arxiv.org/abs/2009.03300) and its [evaluation code](https://github.com/hendrycks/test): the original prompt and letter scoring.
- [lm-evaluation-harness MMLU tasks (EleutherAI)](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu): the default, cloze, generative and chain-of-thought variants side by side.
- [Multiple Choice Normalization in LM Evaluation (Gao, 2021)](https://blog.eleuther.ai/multiple-choice-normalization/): the ways to compare options of different lengths.
