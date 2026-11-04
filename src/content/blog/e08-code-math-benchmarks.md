---
title: "Code and Math Benchmarks: What Unit Tests and Answer Checkers Miss"
description: "pass@k, weak tests and answer checking on HumanEval, HumanEval+ and MATH-500, measured with Claude Haiku 5.5, Sonnet 5.5 and two small Qwen models."
slug: "code-math-benchmarks"
category: "evaluation-reliability"
tags: ["evals", "benchmarks", "reasoning", "statistics"]
series: "llm-evals"
seriesOrder: 8
pubDate: 2026-09-21
updatedDate: 2026-11-04
---

## TL;DR

Code and math benchmarks look objective because a program does the grading: unit tests run the code, and a checker compares the final answer with a key. This part measures how much that program decides. On HumanEval, Claude Haiku 5.5 passed 96.4% of 1,640 samples; rerun against HumanEval+, about 100 times as many tests per problem, it passed 92.9%. Reading the 58 samples that changed showed only 13 were real bugs; 32 failed tests that encode quirks of the reference solution. Two small Qwen models lost 4 and 9 points. On 500 MATH problems, the same Haiku replies scored 81.2% by exact string match and 97.6% when every disputed reply was read, and the checker alone decided whether turning thinking up looked like a 7.8-point loss or a 1-point gain.

## Why it matters

Turn on Claude Haiku 5.5's adaptive thinking at high effort, give it 500 competition math problems, and grade the replies by exact string match. It scores 73.4%, against 81.2% with thinking off: a loss of 7.8 points (95% CI 4.2 to 11.4, p < 0.0001). Read the replies instead and the thinking run is right on 98.6% of problems, 1.0 point ahead and within the noise.

The model didn't change between those two numbers. The grader did, along with the small piece of code that pulls a final answer out of a page of reasoning. With thinking on, Haiku boxed its answer less often and ended with a check line more often, and a checker built for the other style marked those replies wrong.

Code and math are called "verifiable" domains: you don't need a human or an LLM judge to say whether an answer is right. [Agentic AI Part 11](/blog/agent-evals/) covered how to count repeated attempts with pass@k and pass^k. This part looks at the verifier itself: what a unit test or an answer checker accepts, what it misses, and how far a score moves when it is replaced by a stricter or a smarter one.

## The intuition

Picture a driving examiner with a checklist. If the checklist says "stopped at the stop sign" and "parked between the lines", a student who does both passes, even if they nearly hit a cyclist on a street the route never used. A longer route catches more bad drivers. It can also fail a good driver for something the examiner's own instructor does differently, like signaling earlier.

Unit tests are the route. HumanEval gives each problem about eight tests; a program that passes them can still crash on an empty list or a negative number nobody tried. Math answer checkers are the examiner's ear: `3/4`, `0.75` and `\dfrac{3}{4}` are the same answer, and a checker that only compares strings hears three different ones.

The analogy breaks in one useful place. A driving test can't be rerun on the same drive, but a benchmark can. Every reply here was graded several ways, so each grader's effect is a paired difference on identical outputs.

## Step 1: how code benchmarks grade

### HumanEval and MBPP

[HumanEval](https://arxiv.org/abs/2107.03374) (Chen et al., 2021) is "a set of 164 handwritten programming problems", each with "a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem". The model sees the signature and docstring and writes the body. The grader runs the hidden tests; the sample passes only if all of them pass. [MBPP](https://arxiv.org/abs/2108.07732) (Austin et al., 2021), Mostly Basic Programming Problems, has 974 short tasks written by crowdworkers, each with "three test cases that check for semantic correctness".

![Flowchart. A benchmark item, a problem plus its reference, leads to a model reply. For code, the grader pulls out the code block and runs hidden tests in a sandbox; for math, it pulls out the final answer and compares it with the reference answer. Both paths end in a verdict, pass or fail. Dotted arrows mark what each path misses: bugs the tests never exercise, and right answers written another way.](/diagrams/e08-verifier-pipeline.svg "How a code or math benchmark turns a reply into a verdict, and where each path can go wrong.")

Every number in this part comes from one of the two paths, and each experiment below changes only the grading step on one of them.

Running model-written code is risky, so the Codex team "developed a sandbox environment to safely run untrusted programs against unit tests". A sandbox is an environment that limits what a program can touch. The runner here, in the lab's new module `verify.py`, starts each program in its own Linux user and network namespace (`unshare --user --map-current-user --net`), so it keeps the same unprivileged user and has no network, inside a systemd scope capped at 512 MB of memory and 64 tasks. Each program gets a fresh temporary directory, a wall-clock timeout, CPU and file-size limits, and a guard that disables process, kill and file-deletion functions, adapted from the HumanEval harness. [Agentic AI Part 13](/blog/agent-sandboxing/) covers stronger isolation.

Before trusting a grader, check it on code with known answers. All 164 reference solutions passed their own tests, and 164 stub functions that return `None` all failed. The HumanEval+ copy on Hugging Face was less tidy: its check for problem 32, which asks for a root of a polynomial, crashes on every input, so its own reference solution fails. That one problem is graded on its original tests below.

### pass@k in one paragraph

With n samples per problem of which c pass, pass@k is the chance that at least one of k samples drawn from the n passes, `1 - C(n-c, k) / C(n, k)`, averaged over problems. Chen et al. sampled n = 200 per problem because the shortcut `1 - (1 - c/n)^k` "results in a consistent underestimate". [Agentic AI Part 11](/blog/agent-evals/) works the formula and its opposite, pass^k, through an example. One more point matters here: the problems are the sample, so the confidence interval should come from resampling problems, not samples.

## Step 2: HumanEval, then the same code against 100 times the tests

### The setup

[EvalPlus](https://arxiv.org/abs/2305.01210) (Liu et al., NeurIPS 2023) asked whether HumanEval's few tests let wrong code through. Its authors used ChatGPT to write "high-quality seed inputs", then "type-aware mutation" to grow them, keeping each new input's expected output from the reference solution. The result, HumanEval+, extends the tests "by 80×". In the Hugging Face copy used here (`evalplus/humanevalplus`, Apache 2.0), each problem has a median of 977 test inputs, from 12 to 1,100.

The experiment generated code once and graded it twice:

| Model | Setting | Samples per problem | Replies |
|---|---|---|---|
| `claude-haiku-5-5` | thinking disabled, default effort (medium), `max_tokens` 1,024 | 10 | 1,640 |
| `claude-sonnet-5-5` | `between_tools` (its lowest thinking setting), default effort (high), `max_tokens` 1,024 | 2 | 328 |
| Qwen2.5-1.5B-Instruct | greedy, bf16 on an RTX 4060 Laptop GPU | 1 | 164 |
| Qwen2.5-0.5B-Instruct | greedy, bf16 on the same GPU | 1 | 164 |

The prompt asked for "the complete function, including its signature and any imports it needs", in a single Python code block and nothing else, followed by the HumanEval prompt. Claude ran through the Message Batches API with its default sampling; the Qwen models (Apache 2.0) ran in transformers 5.21.1. Two of Haiku's 1,640 replies hit the token limit and were graded as written.

### The results

| Model | HumanEval pass@1 (95% CI) | HumanEval+ pass@1 | Drop (paired, 95% CI) | Relative drop |
|---|---|---|---|---|
| Haiku 5.5, 10 samples | 96.4% (93.6-98.7) | 92.9% (89.1-96.3) | 3.5 points (1.2-6.3) | 3.7% |
| Sonnet 5.5, 2 samples | 99.7% (99.1-100.0) | 94.8% (91.2-97.9) | 4.9 points (1.8-8.5) | 4.9% |
| Qwen2.5-1.5B-Instruct, greedy | 56.7% (49.1-64.1) | 48.2% (40.7-55.8) | 8.5 points (4.9-12.8) | 15.1% |
| Qwen2.5-0.5B-Instruct, greedy | 31.7% (25.1-39.2) | 27.4% (21.2-34.7) | 4.3 points (1.2-7.3) | 13.5% |

Intervals resample the 164 problems; the Qwen rates use Wilson intervals, a confidence interval for a proportion that behaves well near 0% and 100%. The drops are paired: the same code, graded twice, so problem difficulty cancels. No sample passed HumanEval+ and failed HumanEval, which is expected, since the extra tests only add constraints. For Qwen2.5-1.5B, 14 problems passed the original tests and failed the new ones, and none went the other way (McNemar's test, p = 0.0001).

![Grouped bar chart of pass@1 on HumanEval (blue) and HumanEval+ (orange) for four models, with 95% intervals and the paired drop on the right. Haiku 5.5: 96.4% to 92.9%, drop 3.5 points. Haiku pass@10: 98.8% to 95.1%, drop 3.7. Sonnet 5.5: 99.7% to 94.8%, drop 4.9. Qwen2.5-1.5B-Instruct: 56.7% to 48.2%, drop 8.5. Qwen2.5-0.5B-Instruct: 31.7% to 27.4%, drop 4.3.](/diagrams/e08-humaneval-plus.svg "The same code graded by HumanEval's tests and by HumanEval+'s. Drops are paired over 164 problems.")

The two Qwen models lost 13.5% to 15.1% of their score, in line with EvalPlus's report of pass@k "reduced by up-to 19.3-28.9%" across 26 models. Haiku and Sonnet lost 3.7% and 4.9%, so the weak tests flattered the weak models most.

The extra tests also changed a comparison. On the original tests Sonnet led Haiku by 3.3 points (95% CI 0.9 to 6.2), paired over problems. On HumanEval+ its lead shrank to 2.0 points (-1.5 to +5.4), no longer distinguishable from zero. Sonnet's 16 newly failed samples came from 8 problems; 3 of them were reference-quirk problems from the table below (é, `True` and the rounded float), which penalize careful code from either model.

pass@k inherits the same inflation. Haiku's pass@10, the chance that at least one of ten samples passes, fell from 98.8% to 95.1%, a drop of 3.7 points, the same as at pass@1. Drawing more samples can't help when every sample makes the same mistake, and on the problems the extra tests caught, Haiku usually did: on 6 of the 9, every sample that passed the original tests failed the new ones.

Ten samples also show how much a single run moves. Haiku solved 155 of the 164 problems in all 10 samples, 2 in none, and 7 sometimes. Treating each sample index as one benchmark run gives ten HumanEval scores from 95.7% to 97.0%. Only 20% of the score's variance came from sampling; the rest is which problems the benchmark contains, the same split [Part 3](/blog/eval-statistics/) measured on MMLU.

### Reading the 58 samples that changed

A stricter test suite is only better if what it catches is wrong. The 58 Haiku samples that passed HumanEval and failed HumanEval+ came from 9 problems. Rerunning each one input at a time found the first input it failed:

| What the new test caught | Problems | Samples | Example |
|---|---|---|---|
| a real bug | 3 | 13 | `multiply(-6, -9)` should multiply the unit digits 6 and 9; the code computed `-6 % 10`, which is 4 |
| an edge case the docstring doesn't mention | 1 | 3 | `fib(0)` expected 0; the code raised an error for n below 1 |
| a timeout on huge inputs | 1 | 10 | even digits between 2 and 987,654,322, by looping over every number |
| a choice made by the reference solution | 4 | 32 | `is_bored("I")` expected 0 boredoms, because the reference only counts "I" followed by a space |

The last row holds the most samples. The docstring for `file_name_check` asks for a name that starts with "a letter from the latin alphapet ('a'-'z' and 'A'-'Z')", and Haiku checked exactly that. HumanEval+ expects `'éxample.exe'` to be accepted, because the reference uses Python's `isalpha()`, which also accepts é. For `double_the_difference`, which should ignore numbers that are "negative or not integers", the reference counts `True` as the integer 1. For `closest_integer("-578000000007654321.12345")`, the expected answer is `-578000000007654272`, a floating-point rounding error; Haiku's exact `Decimal` arithmetic gave the true nearest integer and failed. EvalPlus generates expected outputs by running the reference, so every quirk of the reference becomes a test. Its authors also found "18 defects (11% of problems) even in the original ground-truth in HUMANEVAL" and fixed many of them.

Counting those 32 samples as passes puts Haiku's HumanEval+ score at 94.8%. So more than half of its 3.5-point drop came from tests that disagree with a reasonable reading of the problem. The Qwen failures were different: at least 10 of the 1.5B model's 14 were plain bugs, like counting 12 characters in "Hello, World!".

## Step 3: from functions to repositories and fresh problems

HumanEval is small, public since 2021 and nearly saturated for frontier models, and its problems have been copied across the web. Two newer designs attack different weaknesses.

[LiveCodeBench](https://arxiv.org/abs/2403.07974) (Jain et al., 2024) collects problems "from weekly contests on competition platforms and tag[s] them with a release date", from LeetCode, AtCoder and Codeforces. A model is scored only on problems released after its training cutoff. The dates exposed contamination, a model having seen benchmark items in training: the authors saw "a stark drop in the performance of DeepSeek, GPT-4-O, and Codestral on LeetCode problems released after Aug 2023, Oct 2023, and Jan 2024". It also tests self-repair, code execution and test output prediction.

[SWE-bench](https://arxiv.org/abs/2310.06770) moves from functions to repositories: real GitHub issues, graded by the tests the human fix added, plus the existing tests so the fix can't break anything. Its tests can be wrong in the same ways as HumanEval's. When OpenAI and the authors screened tasks for [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/), they flagged 61.1% for tests that "may unfairly mark valid solutions as incorrect", the same problem as the reference quirks above. [Agentic AI Part 8](/blog/coding-agents/) covers that screening, the later contamination audit and how coding agents solve such tasks; Anthropic's Claude Haiku 5.5 system card reports 64.8% on SWE-bench Pro for Haiku and 81.3% for Sonnet 5.5, averaged over five trials with adaptive thinking at max effort (reported, not reproduced here).

## Step 4: grading math answers

### GSM8K, MATH and MATH-500

Math benchmarks grade only the final answer. [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021) has 8.5K grade-school word problems that "take between 2 and 8 steps to solve", split into 7,473 training and 1,319 test problems, each with one number as its answer. Its paper is also where verifiers in the modern sense began: sample 100 solutions, rank them with a trained verifier, keep the best, which gave "approximately the same performance boost as a 30x model size increase".

[MATH](https://arxiv.org/abs/2103.03874) (Hendrycks et al., 2021) is harder: 12,500 competition problems in 7 subjects and 5 difficulty levels, each with "a final boxed answer", written as `\boxed{...}` in LaTeX. At release, large language models scored 3.0% to 6.9%. The paper says answers "are unique after normalization", and its code normalizes before comparing: it treats `\dfrac` as `\frac`, drops `\left` and `\right`, units, spaces and degree signs, and maps 0.5 to 1/2. MATH-500 is a 500-problem subset that [Lightman et al.](https://arxiv.org/abs/2305.20050) held out from training, chosen so that its "distribution of difficulty levels and subjects ... is representative of the MATH test set as a whole".

### Three graders, one set of replies

`verify.py` grades each extracted answer three ways:

- exact: the answer string equals the key after trimming spaces at the ends;
- normalized: both pass through the MATH repository's normalization first;
- symbolic: Hugging Face's [Math-Verify](https://github.com/huggingface/Math-Verify) (version 0.9.0, Apache 2.0) parses both into SymPy, a Python library for symbolic math, and tests equivalence, with sets, intervals and numeric tolerance.

```python
import verify   # evals/lab/verify.py

pairs = [("\\dfrac{3}{4}", "\\frac 34"), ("x = 9", "9"), ("10080", "10,\\!080"),
         ("2√5", "2 \\sqrt{5}"), ("3 < \\lambda \\le 4", "(3,4]")]
for pred, gold in pairs:
    verdicts = {name: grade(pred, gold) for name, grade in verify.GRADERS.items()}
    print(f"{pred:18} {gold:12} {verdicts}")
```

```text
\dfrac{3}{4}       \frac 34     {'exact': False, 'normalized': True, 'symbolic': True}
x = 9              9            {'exact': False, 'normalized': True, 'symbolic': True}
10080              10,\!080     {'exact': False, 'normalized': False, 'symbolic': True}
2√5                2 \sqrt{5}   {'exact': False, 'normalized': False, 'symbolic': False}
3 < \lambda \le 4  (3,4]        {'exact': False, 'normalized': False, 'symbolic': False}
```

All five predictions are correct. The last two are real Haiku answers that every grader rejected: one wrote a Unicode square root, the other an inequality where the key has an interval.

Before grading, an extractor pulls the answer out of the reply: the last `\boxed{}`; if there is none, a final bold line such as "**Answer: 13**"; then the last math expression of the final paragraph; then the last bold phrase. The extractor matters because Haiku often ignored the instruction `Put your final answer in \boxed{}`.

### The runs

Haiku 5.5 answered all 500 MATH-500 problems in three settings: thinking disabled (twice, `max_tokens` 4,096), adaptive thinking at effort low (twice) and adaptive thinking at effort high (once), both with `max_tokens` 8,192. No reply hit the limit. At effort low, Haiku thought on only 5 of 500 problems and scored the same as with thinking off, so the comparison below uses thinking off and effort high, where it thought on 376 problems and wrote a median of 460 output tokens against 212.

To find out which grader was right, every reply where the three graders disagreed (430) and every reply all three rejected (101) was read one by one, 531 replies with 250 distinct answer and key pairs.

![Grouped bar chart of MATH-500 accuracy for the same replies under four graders, with thinking off (blue) and adaptive thinking at effort high (orange), and the paired difference on the right. Exact: 81.2% and 73.4%, difference -7.8 points (-11.4 to -4.2). Normalized: 93.6% and 86.0%, -7.6 (-10.4 to -4.8). Symbolic (math-verify): 97.4% and 90.6%, -6.8 (-9.6 to -4.2). Read one by one: 97.6% and 98.6%, +1.0 (-0.4 to +2.6).](/diagrams/e08-math-graders.svg "Haiku 5.5 on MATH-500: one set of replies per setting, four ways to grade them.")

With thinking off, exact match scored 81.2% and normalization added 12.4 points (95% CI 9.6 to 15.4); most of those were `\dfrac` against `\frac`, a missing degree sign or "x = 9" against "9". Symbolic checking added another 3.8 (2.2 to 5.6): reordered sums like "11+11√5", sets in a different order, thousands separators, letter choices like `C` against `\text{(C)}`. Reading the replies added 0.2. Sending the same 500 requests a second time moved the symbolic score by -0.2 points (-1.6 to +1.2); the choice of grader moved it by 16.2. The gap grows with difficulty: on the 134 level-5 problems, exact match credited 98 and math-verify 124.

In the replies that were read, the symbolic grader never accepted a wrong answer: in all 430 disputed replies, it was the one that agreed with the reading. Its errors were all rejections of right answers: 49 of the 101 replies every grader rejected were correct. Assuming the 1,969 replies all three accepted are right, the graders matched the reading on 80.8% (exact), 93.5% (normalized) and 98.0% (symbolic) of all 2,500 replies.

At effort high the extractor became the problem. Haiku boxed its answer in 71.6% of replies against 93.2% with thinking off, and often ended with a check, like "As a consistency check, the sum of the roots is `-\frac{a}{2} = 128 > 0`", after already stating `a = -256`. The extractor took `128 > 0`. Of the 47 high-effort replies that every grader rejected, 40 were right. Math-Verify's own extractor, run on the whole reply, recovered more of them and scored 93.8%, still below the 98.6% from reading. Thinking changed the way Haiku wrote its answers, and the measured "loss" came from a grader built for the old format.

The whole experiment was cheap. On the lab's ledger, with the 50% batch discount, the 1,640 Haiku HumanEval samples cost $0.13, the 328 Sonnet samples $0.47, the 2,500 MATH-500 replies $0.27 (effort high cost twice as much per reply as thinking off) and the pilots $0.01: $0.88 in all. The two Qwen models took about five minutes of GPU time. Reading the disputed replies cost nothing on the ledger and moved the scores more than any model setting did.

## In the real world

- Hugging Face's [Math-Verify leaderboard post](https://huggingface.co/blog/math_verify_leaderboard) (February 2025) re-graded "all 3,751 models ever submitted to the Open LLM Leaderboard" on its MATH subset. "On average, models solved 61 more problems across the board, equating to a 4.66-point increase", and "DeepSeek models almost tripled their scores!" Some models moved "200 places or more", with nothing changed but the checker.
- [MathArena](https://arxiv.org/abs/2505.23281) (Balunović et al., 2025) scores models on competitions such as AIME (the American Invitational Mathematics Examination, 30 problems a year over two February exams, each answer an integer) right after they are held: "By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination." Compared with the fresh exams, most models sat "above this line on AIME with a margin of 10% − 20%, suggesting inflated performance on AIME 2024". Even fresh problems leak: a search found "8 problems from AIME 2025 and 1 problem from HMMT 2025 ... online in a similar form". Part 5 covers contamination detection.
- Final-answer grading can't see the proof. [Proof or Bluff](https://arxiv.org/abs/2503.21934) (Petrov et al., 2025) had expert judges grade full solutions to the six 2025 USA Math Olympiad problems within hours of their release: "only GEMINI-2.5-PRO achieves a non-trivial score of 25%, while all other models achieve less than 5%." Formal benchmarks make the proof checkable by a machine: [miniF2F](https://arxiv.org/abs/2109.00110) has 488 statements from the AIME, AMC and IMO, and [PutnamBench](https://arxiv.org/abs/2407.11214) 1,692 formalizations of 640 Putnam theorems in Lean 4 and Isabelle. A proof checker such as Lean never accepts a wrong proof, but it checks the formal statement, which a person had to translate correctly.
- Reported numbers carry their harness. Qwen's [Qwen2.5 announcement](https://qwenlm.github.io/blog/qwen2.5-llm/) (September 2024) gives Qwen2.5-0.5B-Instruct 35.4 and Qwen2.5-1.5B-Instruct 61.6 on HumanEval; with the prompt above, greedy, they scored 31.7% and 56.7% here. The Claude Haiku 5.5 system card (October 2026) reports no HumanEval, MATH or AIME numbers at all; it reports SWE-bench Pro and Humanity's Last Exam instead.

## Common misconceptions

- **"If the tests pass, the code is correct."** HumanEval averages 7.7 tests per problem. With about 100 times as many, Qwen2.5-1.5B-Instruct lost 8.5 points of its 56.7%, all on code that had passed.
- **"More tests are always a better grader."** Tests generated from a reference solution inherit its quirks. Of 58 Haiku samples that HumanEval+ newly failed, 32 were reasonable code that disagreed with the reference on é, `True` or floating-point rounding.
- **"Math has one right answer, so grading is trivial."** One answer, many spellings. Exact string match scored Haiku's replies 81.2%; the same replies read one by one were 97.6% right.
- **"A symbolic checker solves answer grading."** Math-Verify agreed with the reading on 98.0% of replies and never passed a wrong answer, but it rejected 49 right ones, mostly because the extractor in front of it picked the wrong part of the reply.
- **"A drop in a math score after a model change means it got worse."** Turning on thinking at effort high cost 7.8 points under exact match and gained 1.0 point when the replies were read. Check the extractor and grader on the new outputs before believing a change.

## Key takeaways

- Code and math benchmarks are graded by programs, and those programs make mistakes in both directions: tests that miss bugs, and checkers that reject right answers.
- Validate the grader first: reference solutions must pass, stubs must fail. The Hugging Face copy of HumanEval+ failed its own reference on one problem.
- Stronger tests deflate weak models the most: 13.5% to 15.1% of the score for two small Qwen models, 3.7% for Haiku and 4.9% for Sonnet, and more than half of Haiku's drop came from the reference's own quirks. Read what a new test catches before you trust it.
- For math, grade with a symbolic checker, log the extraction rule that fired, and read a sample of disagreements. On MATH-500 the grader moved Haiku's score by 16 points; resending the same requests moved it by 0.2.
- Any change to the model or prompt can change the reply format. Recheck extraction whenever you compare settings, or the grader will report the change as a gain or a loss.

## What's next

A unit test or a boxed answer is easy to check. Part 9 turns to long-context evals, where the question is whether a model used what was on page 300 of its input, and how simple needle-in-a-haystack tests compare with harder RULER-style tasks.

## Further reading

- [Evaluating Large Language Models Trained on Code (Chen et al., 2021)](https://arxiv.org/abs/2107.03374): HumanEval, the unbiased pass@k estimator and the execution sandbox.
- [Program Synthesis with Large Language Models (Austin et al., 2021)](https://arxiv.org/abs/2108.07732): MBPP, 974 crowd-written tasks with three tests each.
- [Is Your Code Generated by ChatGPT Really Correct? (Liu et al., 2023)](https://arxiv.org/abs/2305.01210): EvalPlus and HumanEval+, 80 times the tests, and defects in HumanEval's own solutions.
- [LiveCodeBench (Jain et al., 2024)](https://arxiv.org/abs/2403.07974): contest problems tagged by release date, and the contamination cliffs they revealed.
- [SWE-bench (Jimenez et al., 2023)](https://arxiv.org/abs/2310.06770) and [Introducing SWE-bench Verified (OpenAI, 2024)](https://openai.com/index/introducing-swe-bench-verified/): repository-level tasks and the screening of unfair tests.
- [Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168): GSM8K and best-of-100 sampling with a learned verifier.
- [Measuring Mathematical Problem Solving With the MATH Dataset (Hendrycks et al., 2021)](https://arxiv.org/abs/2103.03874) and its [code](https://github.com/hendrycks/math): 12,500 problems and the normalization rules used above.
- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050): process reward models and the MATH-500 subset.
- [Math-Verify (Hugging Face)](https://github.com/huggingface/Math-Verify) and [Fixing Open LLM Leaderboard with Math-Verify (Kydlíček et al., 2025)](https://huggingface.co/blog/math_verify_leaderboard): a symbolic answer checker and what it did to 3,751 leaderboard scores.
- [MathArena (Balunović et al., 2025)](https://arxiv.org/abs/2505.23281): grading on competitions as soon as they happen, and contamination on AIME 2024.
- [Proof or Bluff (Petrov et al., 2025)](https://arxiv.org/abs/2503.21934), [miniF2F (Zheng et al., 2021)](https://arxiv.org/abs/2109.00110) and [PutnamBench (Tsoukalas et al., 2024)](https://arxiv.org/abs/2407.11214): human-graded and machine-checked proofs.
