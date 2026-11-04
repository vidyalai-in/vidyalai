---
title: "Reading a Model Card and Building Your Own Eval"
description: "Decode the conditions behind reported benchmark numbers, then build a 195-question eval of your own: items, power, harness, contamination and grader."
slug: "reading-model-cards"
category: "evaluation-reliability"
tags: ["evals", "benchmarks", "statistics", "reproducibility"]
series: "llm-evals"
seriesOrder: 11
pubDate: 2026-09-29
updatedDate: 2026-11-04
---

## TL;DR

A reported benchmark number is a measurement taken under conditions: trials, thinking effort, tools, subset, grader and what counted as a failure. This finale reads three real reports (the Claude Haiku 5.5 system card, the Llama 3 paper and the Qwen2.5 report) and turns what they state and leave out into a checklist, each question tied to the part of the series that explains it. Then it builds an eval from scratch with the series' tools in order: 195 questions about this series' own articles, validated three ways, sized with a power analysis, checked closed-book for contamination and scored by a hand-checked grader. With the article in context, Haiku 5.5 scored 98.5% with thinking off and 100% at low effort, a gap below the eval's 2.1-point minimum detectable effect; without the article it scored 19.4%. A closing table sums up the series.

## Why it matters

On one page of its system card, the report Anthropic published with the model, Claude Haiku 5.5 scores 72.4% on the OSWorld computer-use benchmark. A few lines down, the same runs on the same 82 tasks score 37.1%. Both numbers are correct: the first gives partial credit for each checkpoint of a task, the second counts a task only when every checkpoint passes.

Every headline number carries choices like that. The first ten parts of this series measured what each choice does: the questions chosen (Part 1), wrong answer keys (Part 2), the error bar (Part 3), the prompt and the answer parser (Part 4), leaked test items (Part 5), the grader (Parts 6 and 8), the voters (Part 7), the context length (Part 9) and what refusals do and don't show (Part 10). This part puts them to work twice: once as a reader of other people's numbers, and once as the builder of a new eval whose numbers you have to stand behind.

## The intuition

Think of a car's fuel-economy sticker. "5.2 litres per 100 km" is real, but it was measured on a test cycle: a set speed profile, a temperature, no roof box, no air conditioning. Your number on a mountain road in summer will differ, and a second maker's sticker means something only if both used the same cycle.

A model card, the document a developer publishes with a model to describe what it is for and how it was evaluated (Anthropic calls its version a system card), is a page of stickers. Each number comes with its own cycle: few-shot examples or none, thinking at maximum effort or off, five attempts averaged or one, a model grader or a test suite. Reading the card means reading the cycle, and building your own eval means writing down a cycle that someone else could rerun.

The analogy breaks in one place. Fuel-economy cycles are fixed by regulators, so every maker's sticker uses the same one. Benchmark conditions are chosen by whoever reports the number, often differently for their own model and for the competitors in the same table.

## Step 1: what a headline number hides

The numbers below come from three reports. The Claude Haiku 5.5 system card is dated October 7, 2026; the Llama 3 paper is arXiv version 3 (November 2024); the Qwen2.5 technical report is dated January 2025. All numbers are as reported by their authors, not reproduced here.

| Reported number | Where | Conditions the source states |
|---|---|---|
| SWE-Bench Pro 64.8% | Haiku 5.5 card, Table 8.1.A | Adaptive thinking at max effort, default sampling, "averaged over five trials", context up to 1M tokens |
| Terminal-Bench 4.0 39.2% | Same card, section 8.4 | 66 tasks, 10 trials each (660), Claude Code in `--bare` mode at max effort, no internet; ±1.9 points "with trials treated as independent"; 12 trials stopped by safeguards and counted as failures |
| OSWorld 2.1: 72.4% or 37.1% | Same card, section 8.9.3 | The offline subset, 82 of 108 tasks; partial score vs strict pass rate; five attempts per task |
| Humanity's Last Exam 45.9% or 57.4% | Same card, section 8.8.1 | Without and with search, fetch and code tools; 980k-token task budget; graded by Claude Opus 4.6; HLE discussion sites blocklisted |
| Global MMLU 87.8% | Same card, section 8.12.1 | Max effort, "a single trial"; the 0.1% of questions blocked or unanswered "left out of the average rather than scored as wrong" |
| GDPval-AA 1620 | Same card, section 8.10.2 | An Elo rating from blind pairwise comparisons, anchored at 1600, run by Artificial Analysis at max effort; 1277 at the default effort |
| MMLU 87.3 or 88.6 | Llama 3 paper, Table 2 | Llama 3 405B, 5-shot vs 0-shot with chain of thought; footnote symbols mark competitor numbers obtained another way |
| HumanEval 72.6 ± 6.8 | Llama 3 paper, Table 18 | Llama 3 8B, pass@1 on 164 problems, 95% interval |
| Gemma 2 9B HumanEval: 54.3 or 68.9 | Llama 3 paper, Table 2; Qwen2.5 report, Table 8 | The same model and benchmark in two reports; neither gives the other's harness |
| Qwen2.5-0.5B MMLU-redux: 45.1 or 24.1 | Qwen2.5 report, Tables 5 and 10 | Base model, 5-shot; instruct model, shots and format not stated; 24.1 is below the 25% that guessing gets |

In the table, "5-shot" means five solved examples placed in the prompt, and chain of thought means the model writes out its reasoning before the answer. Four patterns repeat.

The same run gives different numbers under different scoring rules. OSWorld's 72.4% and 37.1% are one set of attempts. Pass@1 here means the average success over five independent attempts, not "the best of five", which would be pass@5 ([Agentic AI Part 11](/blog/agent-evals/) covers the difference).

Effort and tools are part of the number. Thinking effort, the setting that controls how long the model reasons before answering, moved GDPval-AA from 1277 at the default to 1620 at max, "while using about a tenth of the output tokens" at the default. Tools moved Humanity's Last Exam by 11.5 points. Neither is the model's "real" score; they are two operating points.

Uncertainty is reported unevenly. The Terminal-Bench interval treats 660 trials as independent, though they are 10 tries at each of 66 tasks; [Part 3](/blog/eval-statistics/) showed that clustering on tasks can nearly double an interval. Global MMLU comes from a single trial. Llama 3 is unusual in printing a 95% interval next to each score in its code table: ±6.8 points on HumanEval's 164 problems is wider than many gaps people argue about.

Competitor columns are not always measured the same way. The Haiku card says competitor figures "are drawn from the respective developers' published system cards or benchmark leaderboards unless otherwise noted". The Llama 3 paper evaluated other models itself where it could and compared "the results with the reported numbers, selecting the best score". Gemma 2 9B appears with 54.3 on HumanEval in the Llama 3 paper and 68.9 in the Qwen2.5 report, a 14.6-point gap between two reports that both call it HumanEval. Within one report, Qwen2.5-0.5B goes from 45.1 on MMLU-redux as a base model to 24.1 as an instruct model, which almost certainly reflects a generation-style harness and answer extraction (the failure [Part 4](/blog/eval-harness-sensitivity/) measured) rather than a model that knows less than chance.

## Step 2: questions to ask of any reported number

The checklist below turns those patterns into questions. Each example comes from the three reports; the last column says where in this series the question is answered.

| Question | Example from the reports | Why it matters |
|---|---|---|
| Which benchmark version and subset? | OSWorld 2.1 offline subset, 82 of 108 tasks | Subsets and versions change the items ([Part 2](/blog/how-benchmarks-are-built/)) |
| How many items and trials, and what error bar? | Terminal-Bench: 66 tasks, 10 trials, ±1.9 with trials as independent | Items, not runs, set the precision; groups widen it ([Part 3](/blog/eval-statistics/)) |
| What prompt: shots, chain of thought, format? | Llama 3 405B MMLU: 87.3 at 5-shot, 88.6 at 0-shot with chain of thought | Harness choices move scores by more than model gaps ([Part 4](/blog/eval-harness-sensitivity/)) |
| How much thinking, which tools, what budget? | GDPval-AA 1277 at default effort, 1620 at max | Each is a different operating point |
| What counts as solved? | OSWorld partial score 72.4% vs strict 37.1% | pass@1, pass@k and partial credit answer different questions ([Part 8](/blog/code-math-benchmarks/)) |
| Who or what graded? | HLE graded by Claude Opus 4.6 | Model graders and test suites both err ([Part 6](/blog/grading-generative-outputs/), Part 8) |
| How were blocks, refusals and errors counted? | Global MMLU left 0.1% out; Terminal-Bench counted 12 stopped trials as failures | Excluding items raises scores; [Part 10](/blog/safety-evals/) covers refusals |
| What guards against contamination? | HLE blocklist; Llama 3's 8-gram overlap; Qwen2.5's longest-common-subsequence filter | Leaked items measure memory ([Part 5](/blog/benchmark-contamination/)) |
| Were competitors run the same way? | Gemma 2 9B HumanEval: 54.3 in one report, 68.9 in another | Copied numbers carry another lab's harness |
| Is the gap larger than the eval can detect? | ±6.8 points on HumanEval's 164 problems | Compare the gap with the minimum detectable effect (Part 3) |
| For ratings: whose votes, what anchor? | GDPval-AA anchored at 1600 | Elo-style ratings are relative ([Part 7](/blog/chatbot-arena-elo/)) |
| For long inputs: claimed or effective context? | "Context window sizes are evaluation-dependent" | [Part 9](/blog/long-context-evals/) measures where accuracy falls off |

No card answers every question for every number. Where an answer is missing, the number deserves less weight.

## Step 3: building an eval, from construct to items

Building an eval means deciding each of those conditions yourself. The worked example follows the series in order.

![Flowchart of six steps, top to bottom: 1, define the construct (Part 1); 2, write and validate items (Part 2); 3, size it with a power analysis (Part 3); 4, fix and publish the harness (Part 4); 5, guard against contamination (Part 5); 6, validate the grader (Parts 6 and 8); then run paired configurations and report confidence intervals. An arrow labeled disputed replies loops from step 6 back to step 2.](/diagrams/e11-build-eval.svg "Building an eval with the series' tools, in order. Disputed grader verdicts send items back for review.")

The loop at the end was used: hand-checking the grader sent one answer key back for a fix.

### The construct

A construct is the ability a score is supposed to stand for, and [Part 1](/blog/why-llm-evals-are-hard/) showed how easily a benchmark drifts from it. The one chosen here is narrow on purpose: given one technical article, can the model answer precise questions about it using only that article, and say so when the article doesn't contain the answer? It is the skill a documentation assistant needs, and it rules out general knowledge, summarizing and reasoning beyond the text.

The articles are Parts 1 to 8 of this series. That keeps licensing clean (it is this site's own text), and every answer can be checked against a sentence on the page. Each item has one of three kinds: a number (a count, a percentage, a cost), a short phrase of at most five words, or "not stated" for a question the article doesn't answer. The last kind tests abstention, the model's willingness to say it doesn't know.

### Writing and validating items

Claude Haiku 5.5 drafted the items, one request per article (adaptive thinking at medium effort, 8 requests, $0.034), with rules: self-contained questions, one number or a short phrase as the answer, no general knowledge, and a sentence copied verbatim from the article as evidence. It returned 195 candidates: 123 numbers, 52 phrases and 20 unanswerable questions (for Part 7 it wrote none of the last kind). One of them:

```json
{"id": "e03-01", "part": 3, "kind": "number",
 "question": "How many MMLU questions make up the fixed evaluation sample used throughout the article?",
 "answer": "1,140", "aliases": [],
 "evidence": "...1,140 of the 14,042 in the Hugging Face copy of its test split...",
 "canary": "VIDYALAI-EVALS-E11 canary GUID 5b0d6c2e-..."}
```

[Part 2](/blog/how-benchmarks-are-built/) showed that every benchmark ships with label errors, so validation came in three layers, from cheapest to most expensive.

1. Automatic checks. Is the evidence really in the article? Does it contain the answer? Is a number answer exactly one number? These flagged 12 of 195 items (6.2%). Reading them, 8 were fine (the evidence had an ellipsis or the answer sat in the next sentence) and 4 needed a shorter or cleaner answer key.
2. Two independent answer passes. Haiku 5.5 answered every item twice with the article in context and the key hidden (adaptive thinking, medium effort; 390 requests, $0.039). Both passes matched the key on 191 items. Of the 4 disputes, 2 were only the grader being strict about wording. The other 2 were real item faults: an "unanswerable" question that a table in the article does answer (AUROC 0.627), and a question that asked about a condition the article never states.
3. A hand check. Every flagged or disputed item was checked against the article, and so was a random 20% of the agreed items (38). All 38 were right, which still allows a residual error rate up to 9.2% (the upper end of the 95% Wilson interval, the confidence interval [Part 3](/blog/eval-statistics/) recommends for small counts).

In all, 6 of 195 items were fixed (3.1%) and none dropped. One flaw these checks can't catch is that the item writer and the model under test are the same model. Questions phrased the way Haiku would phrase them may be easier for Haiku, the same-maker effect [Part 6](/blog/grading-generative-outputs/) measured for judges. A stronger eval would have people, or a model from another family, write the items.

The validation passes also gave the first warning. They scored 100% and 99.0%. An eval that the validator aces may not be able to separate the configurations you care about.

## Step 4: sizing, harness and contamination

### Power before running

The comparison chosen in advance: Claude Haiku 5.5 with thinking off (`thinking: {"type": "disabled"}`) against adaptive thinking at low effort, both with the article in context. Power analysis from [Part 3](/blog/eval-statistics/) says how small a gap 195 items can see. Before any run, the input it needs, how often the two settings disagree on an item, can only be guessed. With one run per setting, the minimum detectable effect (MDE, the smallest true gap found 80% of the time at the 5% level) is 2.8 points if they disagree on 2% of items, 4.5 at 5% and 6.3 at 10%.

The validation passes suggested the low end, since almost every item was answered correctly. That set the plan: three runs per item per setting, to shrink the noise of sampling the same item again, and no claim smaller than the measured MDE.

### The harness, written down

[Part 4](/blog/eval-harness-sensitivity/) measured how much the prompt, the answer format and the parser move a score. The fix is to choose them once, before the results, and publish them:

```text
Model        claude-haiku-5-5 via the Message Batches API (asynchronous, half price); temperature and top_p not sent
Settings     off: thinking {"type": "disabled"}, max_tokens 400
             low: thinking {"type": "adaptive"}, effort "low", max_tokens 2000
System       fixed instructions, then the item's article (Markdown, frontmatter removed),
             cached for 1 hour; user turn: "Question: <question>"
Instructions "...Its last line must be "Answer: " followed by the answer alone: one number
             (with its unit or % sign if it has one) or a few words. If the article does
             not give the answer, the last line must be "Answer: not stated"."
Extraction   the text after the last "Answer:" in the reply
Grading      number: the first number equals the key (or a value the article also gives);
             phrase: the key or an alias appears as whole words; none: "not stated"
Runs         3 per item per setting; 195 items (123 number, 52 phrase, 20 not stated)
```

Every article is the shared prefix of about 25 items, so prompt caching, which stores a repeated prompt prefix and bills later reads of it at a tenth of the input price, paid for itself: 1,162 of 1,170 requests read the article from the cache (about 10,600 tokens each), and a 195-item pass cost $0.014 to $0.016 instead of about $0.11 uncached. No reply was cut off by `max_tokens` (0 of 1,170), and 1,166 put the `Answer:` line on a line of its own; the other 4 put it at the end of a sentence. Low effort cost no more than thinking off ($0.041 against $0.049 for 585 replies): the two wrote almost the same number of output tokens, and the gap is eight cache misses in the thinking-off batch.

### A contamination check

[Part 5](/blog/benchmark-contamination/) showed that leaked items inflate scores and that detection after the fact is weak. Two guards apply here. The item file carries a canary string, a unique marker that lets anyone filter it out of training data, and the items stay private: only the few quoted here are published.

The other guard is a closed-book run: each answerable question with the article's title but not its text. If the model had memorized the articles, it would score well. It got 34 of 175 right, 19.4% (14.2 to 25.9). Reading them, the hits were general knowledge the articles quote (GPQA Diamond's 198 questions, MBPP's 974 tasks), two-way comparisons a guess gets half the time, and round guesses such as "around 0.6" for a correlation of 0.60. One reply began: "I'm not aware of models named "claude-haiku-5-5" or "claude-sonnet-5-5" appearing in published MMLU-Redux results". Only 11 of 123 numbers were hit, against 23 of 52 phrases, so the numeric items are the ones that measure reading. Dropping all 34 guessable items changes nothing below: 98.1% with thinking off and 100% at low effort on the remaining 161.

## Step 5: the grader and the results

### Checking the grader

[Part 6](/blog/grading-generative-outputs/) and [Part 8](/blog/code-math-benchmarks/) found graders wrong in both directions, so the grader got its own check before any result was read. The sample: every open-book reply the grader marked wrong or that a looser rule scored differently (19), 60 random other open-book replies, and 27 closed-book replies, 106 in all, each compared by hand with the key and the article.

The first version agreed on 103 (97.2%). It missed "Answer: 9,219" because the model put it at the end of a sentence instead of on its own line. It rejected "82%" although the article's own summary rounds 82.3% to 82%. And it accepted "LMArena" for the key "Arena", because "arena" is a substring of it. The fixes: find the last `Answer:` anywhere, allow a second value only where the article itself gives one, and match phrases as whole words. The fixed grader agrees on all 106, and across all 1,345 replies it changes exactly those three verdicts. The looser rule, by comparison, would have failed eight correct open-book replies such as "claude-sonnet-5-5" for the key "Sonnet". Its core:

```python
import re

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")

def numbers(s):
    return [float(a.replace(",", "") + b) for a, b in NUM.findall(s)]

def final_answer(reply):
    found = re.findall(r"\banswer\W{0,3}:\s*(.+?)\s*$", reply, re.I | re.M)
    return found[-1].strip(" *`") if found else None

def grade(item, reply):
    ans = final_answer(reply)
    if ans is None:
        return 0
    abstained = "not stated" in ans.lower()
    if item["kind"] == "none" or abstained:
        return int(item["kind"] == "none" and abstained)
    if item["kind"] == "number":
        got = numbers(ans)
        ok = numbers(item["answer"])[:1] + item.get("alt_numbers", [])
        return int(bool(got) and got[0] in ok)
    keys = [item["answer"], *item["aliases"]]
    return int(any(re.search(rf"(?<!\w){re.escape(k.lower())}(?!\w)", ans.lower()) for k in keys))

item = {"kind": "number", "answer": "1,140", "aliases": []}
print(grade(item, "The sample has 20 questions per subject.\nAnswer: 1140 questions"))  # 1
```

It gives the same verdict as the full grader on all 1,345 logged replies.

### The results

![Two-panel chart. Top: accuracy with 95% intervals on the 195-question eval. Closed book with thinking off: 19.4% (14.2 to 25.9) on 175 answerable items. Open book with thinking off: 98.5% (97.0 to 99.9) over 3 runs. Open book with adaptive thinking at low effort: 100.0% (98.1 to 100) over 3 runs. Open book at medium effort, from the two validation runs on 193 unchanged items: 100.0% (98.0 to 100). Bottom: the paired difference, low effort minus thinking off, is +1.5 points, with an interval of +0.1 to +3.0 treating items as independent and -0.3 to +3.4 clustered by article; a shaded band marks the minimum detectable effect of ±2.1 points.](/diagrams/e11-own-eval.svg "Claude Haiku 5.5 on 195 questions about Parts 1 to 8. With the article in context, every setting is near the ceiling.")

With the article in context and thinking off, Haiku 5.5 scored 98.5% (97.0 to 99.9); the three runs gave 98.5%, 97.9% and 99.0%. At low effort it answered all 585 replies correctly, and at medium effort, in the validation passes, all 386 on the items that didn't change. It answered "not stated" on all 20 unanswerable items in every setting.

The paired difference is +1.5 points. Treating items as independent, its 95% interval is 0.1 to 3.0; clustered by article, as Part 3 recommends when items come in groups, it is -0.3 to 3.4. McNemar's test, which looks only at items where the two settings disagree, gives p = 0.25 on the first run: thinking fixed 3 items and broke none. With the measured variances, the MDE for this design is 2.1 points, so the observed gap is below what the eval can reliably detect; about 500 items would be needed to detect 1.5 points with 80% power.

All nine wrong answers came from five items, and four of them need more than finding one sentence: a value in a table, or two facts put together. On the difference-in-differences item, the model wrote: "The 49.1 figure is the raw gain from 27.4% to 82.9%; the article gives the difference-in-differences value only for the verbatim groups." The raw gain is 55.5 points; 49.1 is the difference in differences. The fifth item turned out to be ambiguous: asked "which HumanEval problem", the model named `file_name_check` in its reply but answered "not stated" because the article gives no problem number.

This eval is saturated for this model from the start. It separates open book from closed book by 79 points, but it can't rank two thinking settings that both sit at the ceiling, so it supports no claim that thinking helps here. The misses point to the next version: more items that need a table, two passages or several articles at once.

## The series in one table

Each part measured one way a score can mislead. Its headline result, in one line:

| Part | Topic | Headline measured result |
|---|---|---|
| [1](/blog/why-llm-evals-are-hard/) | Why measuring LLMs is hard | Haiku 5.5 and Sonnet 5.5 tied at 200 of 200 on ARC-Easy but sat 22.5 points apart on MMLU-Pro; with the question hidden, Haiku still scored 47.7% where chance is 25% |
| [2](/blog/how-benchmarks-are-built/) | How benchmarks are built | On 106 MMLU questions with a wrong official answer, Haiku matched the key 18.9% of the time; flawed items were 6.5% of the set and held 16.5% of its misses |
| [3](/blog/eval-statistics/) | The statistics of evals | Rerunning 1,140 questions moved the score by at most 1.0 point, while the choice of questions set ±2.3 (±4.2 clustered); pairing cut the items needed for a 3-point gap from 1,804 to 1,165 |
| [4](/blog/eval-harness-sensitivity/) | Prompt and harness sensitivity | Asking for "only the letter" cost Haiku 9.3 points; five answer extractors put Sonnet anywhere from 53 points behind Haiku to 30 ahead |
| [5](/blog/benchmark-contamination/) | Contamination and memorization | Sixteen leaked copies took a 135M model from 32.9% to 97.4% on those questions; paraphrased copies added 49 points that a 13-gram check mostly missed |
| [6](/blog/grading-generative-outputs/) | Grading open-ended outputs | A Haiku rubric judge tracked experts at Kendall tau 0.27 to 0.49, ROUGE-1 at 0.09 to 0.19; Haiku's pairwise verdicts matched experts 86.0% of the time, two experts 82.0% |
| [7](/blog/chatbot-arena-elo/) | Arenas and Elo rankings | Vote order alone moved online Elo ratings by up to ±34 points; Bradley-Terry intervals were ±7.7, and a 10-point gap needs about 13,700 votes per model |
| [8](/blog/code-math-benchmarks/) | Code and math benchmarks | HumanEval+ cut Haiku from 96.4% to 92.9%, but only 13 of 58 newly failed samples were real bugs; on MATH-500 the grader alone moved its score from 81.2% to 97.6% |
| [9](/blog/long-context-evals/) | Long-context evals | With thinking off, Haiku 5.5 found a literal needle 88% of the time at 62K tokens but answered the same needle asked with one inference 8% of the time from 8K on; adaptive thinking lifted variable tracking at 96K from 0% to 100% |
| [10](/blog/safety-evals/) | Safety and capability evals | On XSTest's safe prompts, read by hand, Qwen2.5-1.5B-Instruct refused 42.4% and Haiku 5.5 0.8%; substring graders called 67% to 89.5% of Haiku's replies to unsafe prompts "not refused" where a rubric judge found 5.8% |
| 11 | Reading a model card, building your own eval | 195 questions on Parts 1 to 8: Haiku 5.5 scored 98.5% with thinking off and 100% at low effort, a 1.5-point gap below the 2.1-point minimum detectable effect, and 19.4% without the article |

## In the real world

- Contamination guards now appear next to headline numbers. For Humanity's Last Exam with tools, the Haiku 5.5 card blocklisted sites known to discuss the benchmark, screened every correctly answered transcript for HLE-specific sources, and "re-graded confirmed cases of cheating as incorrect". The Qwen2.5 report removes a training sequence when its longest common subsequence with a test item is at least 13 tokens and at least 0.6 of the shorter one.
- Some guards report their own limits. Llama 3's 8-gram analysis estimated how much contamination raised each score, but for "MBPP, HumanEval, MMLU and MMLU-Pro", overlap "gives such high contamination scores that it is impossible to get a good performance gain estimate". Four of the most quoted benchmarks are the ones it couldn't assess.
- Charts carry conditions too. The FrontierCode figure in the Haiku card notes that "the result files carry no uncertainty intervals, so none are shown" and that "the y-axis begins at 10%, not at zero, so differences between points are visually magnified".
- The same habits apply inside a product team. [System Design: Building an Eval Suite](/blog/support-bot-eval-suite/) covers error analysis, layered test sets and release gates for a support bot; the eval built here is one layer of such a suite, with its harness and power analysis written down.

## Common misconceptions

- **"A higher number on a model card means a better model."** Only under the same conditions. The same Haiku 5.5 scores 1277 or 1620 on GDPval-AA depending on its effort setting, and 72.4% or 37.1% on OSWorld depending on the scoring rule.
- **"Pass@1 means the model got one try."** In the Haiku 5.5 card it is the success rate averaged over five independent attempts. It estimates how often one attempt succeeds; it says nothing about the best of several.
- **"Competitors in the same table were run the same way."** The card's competitor figures are "drawn from the respective developers' published system cards or benchmark leaderboards". Gemma 2 9B appears at 54.3 on HumanEval in one report and 68.9 in another.
- **"An interval that excludes zero settles the comparison."** The +1.5-point gain from low-effort thinking excluded zero when items were treated as independent and included it when clustered by article, and it sits below the eval's 2.1-point detection limit.
- **"Model-written items are clean once a model can answer them."** Two of the four disputes after validation were real item faults, and the grader check found a third key problem. The validator also aced the set, which said more about the set than about the model.

## Key takeaways

- A reported number is a measurement under conditions: subset, items and trials, prompt, effort, tools, grader, exclusions and who ran the competitors. Read the conditions before the number.
- When a card doesn't answer one of the checklist questions, give that number less weight, and never compare it with a number measured another way.
- Building an eval is this series in order: define the construct, write and validate items, size it with a power analysis, fix and publish the harness, check for contamination, and validate the grader before reading results.
- Each check found something. Validation fixed 6 of 195 items, the closed-book run showed which items measured reading, and hand-checking 106 replies caught three grader mistakes.
- An eval your best setting aces can't rank your settings. Here thinking off and thinking on scored 98.5% and 100%, a gap below what 195 items can detect, and the misses point to harder items as the next step.
- Publish the item format, the harness and the grader; keep the items themselves private, marked with a canary string.

## What's next

This is the last part of Measuring LLMs. Next on the site's reading path is the System Design series, which starts with [designing a RAG system for 10 million documents](/blog/rag-system-10m-documents/) and puts evals like this one inside real products. A series on images, speech and video models is also planned.

## Further reading

- [System Card: Claude Haiku 5.5 (Anthropic, 2026)](https://www.anthropic.com/claude-haiku-5-5-system-card): the capability tables decoded in Step 1, with trial counts, effort settings and contamination guards.
- [The Llama 3 Herd of Models (Grattafiori et al., 2024)](https://arxiv.org/abs/2407.21783): benchmark tables with 95% intervals, robustness checks on MMLU formats and an 8-gram contamination analysis.
- [Qwen2.5 Technical Report (Qwen Team, 2024)](https://arxiv.org/abs/2412.15115): base and instruct evaluation tables and the n-gram decontamination rule.
- [Model Cards for Model Reporting (Mitchell et al., 2018)](https://arxiv.org/abs/1810.03993): the proposal that released models come with "benchmarked evaluation in a variety of conditions".
- [Lessons from the Trenches on Reproducible Evaluation of Language Models (Biderman et al., 2024)](https://arxiv.org/abs/2405.14782): what to report so others can rerun an evaluation.
- [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations (Miller, 2024)](https://arxiv.org/abs/2411.00640): the paired standard errors and minimum detectable effect used in Step 4.
- [How Much Knowledge Can You Pack Into the Parameters of a Language Model? (Roberts, Raffel and Shazeer, 2020)](https://arxiv.org/abs/2002.08910): closed-book question answering, the check used here against guessable items.
- [System Design: Building an Eval Suite](/blog/support-bot-eval-suite/): the product-side workflow (error analysis, layered test sets, release gates) that this part's single eval would slot into.
