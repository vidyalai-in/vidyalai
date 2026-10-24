---
title: "Reflection and Self-Correction in Agents, Measured"
description: "Self-critique, test feedback and Reflexion on 40 coding tasks with Claude Haiku 5.5: what each one fixes, what it breaks, and what a fix costs."
slug: "agent-self-correction"
category: "agents-tool-use"
tags: ["agents", "evals", "tool-use", "cost"]
series: "agentic-ai"
seriesOrder: 4
pubDate: 2026-10-24
---

## TL;DR

Self-correction means letting a model revise its own answer, using feedback from itself (self-critique), from a checker such as unit tests (a verifier), or from a written lesson carried into the next try (Reflexion). Here all three run on 40 Python tasks with hidden unit tests and Claude Haiku 5.5. First attempts were right 97.5% of the time; one round of self-critique fixed 4 of the 5 wrong ones and broke none, but a second round broke two right ones. Self-critique repaired slips well (35 of 38 planted one-token bugs) and misreadings badly (7 of 18 tries on one misread rule), while test feedback fixed every error its tests caught and nothing else. You'll see the transcripts, the cost per fixed bug, and how to choose.

## Why it matters

Here is a model reviewing its own code, word for word. The function is supposed to parse durations like `"1h30m"`, and the specification says so in its one example. The reviewer walks through every rule, then writes:

> **Example:** `"1h30m"` gives `3600 + 1800 = 5400`.
> ...
> I found no real problems, so the function is unchanged.

The code it approved raises `ValueError` on `"1h30m"`. One test run would have shown that. When code with this bug went back with that one failing test, the model fixed it 18 times out of 18.

Asking a model to check its work is the cheapest second pass you can add to an agent, and methods such as Self-Refine and Reflexion are built on it. A 2023 study titled "Large Language Models Cannot Self-Correct Reasoning Yet" found that without outside feedback, self-correction made answers worse. Coding agents get outside feedback almost for free, because code can be run.

[Part 1](/blog/what-is-an-ai-agent/) built the agent loop, [Part 2](/blog/tool-calling-internals/) looked inside tool calls and [Part 3](/blog/react-planning/) covered reasoning before acting; this part covers checking after acting. The System Design article [Detecting and Recovering from Agent Loops](/blog/agent-loop-detection-recovery/) uses a Reflexion-style prompt to break loops; here the method itself is measured.

## The intuition

Think of proofreading your own essay. You read what you meant to write, not what's on the page, so typos you would spot in a stranger's essay slide past. You do catch some: a doubled word jumps out. Now hand the essay to a spell-checker. It catches every misspelling it knows and nothing else; it will never tell you that your argument skipped a step.

Self-critique is the proofreading: the same mind that made the mistake looks for it. A verifier, such as a test suite, is the spell-checker: blind to everything outside its list, and reliable on everything in it. Reflexion is writing yourself a note before the next draft: "last time I forgot the conclusion."

The analogy breaks in one useful place. A tired person proofreading misses errors at random. A model is more systematic: if it misread a rule while writing, it tends to misread it again while reviewing, because the same text produces the same reading. That makes its blind spots repeatable, which is what the experiments below found.

## Step 1: three ways to correct an answer

Every self-correction method has the same shape: produce an answer, get feedback on it, produce a revised answer. They differ in where the feedback comes from, and in the experiments below that decided most of the outcome.

### Self-critique: the model grades itself

[Self-Refine](https://arxiv.org/abs/2303.17651) (Madaan et al., 2023) has one model write an output, critique it and rewrite it, for up to 4 rounds, with no training and no tools. Across 7 tasks on GPT-3.5, ChatGPT and GPT-4 it reports gains of "∼20% absolute on average", largest where quality is a matter of taste: GPT-4's dialogue responses went from 25.4% to 74.6% preferred.

Math, which has a right answer, barely moved: GPT-4 went from 92.9% to 93.1%, and GPT-3.5 stayed at 64.1%. The model rarely spotted its own errors; "ChatGPT feedback for 94% instances is 'everything looks good'".

### Reflexion: write a lesson, then retry

[Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023) adds memory. An Evaluator scores each attempt; after a failure, a Self-Reflection step writes a few sentences of advice, and the next attempt starts with that advice in its prompt. The paper calls this "verbal reinforcement": the model learns through text in its context, with no change to its weights.

For code, the Evaluator is a set of up to 6 unit tests the model writes for itself. With GPT-4 that gave 91% pass@1 on HumanEval, a Python benchmark, against 80% for GPT-4 alone (pass@1 is the share of problems solved by the one answer submitted). Two details matter here. On MBPP, another Python benchmark, Reflexion scored 77.1% against GPT-4's 80.1%, because 16.3% of the programs that passed the model's own tests were wrong, and the loop stopped early on them. And in an ablation on the 50 hardest HumanEval problems translated to Rust, reflection without tests scored 52%, below the 60% baseline: without tests the agent "is unable to determine if the current implementation is correct" and keeps "performing harmful edits". Tests without the written reflection scored 60%; the full method 68%.

### The critique of the critique

[Huang et al.](https://arxiv.org/abs/2310.01798) (ICLR 2024) re-ran these methods and separated two settings. In intrinsic self-correction, the model decides by itself whether its answer is right. In the setting earlier papers had used, an oracle label (the true answer) decides when to stop. Correction then runs only on answers already known to be wrong, so a right answer can never be broken.

With oracle labels, GPT-3.5 on GSM8K (grade-school math) rose from 75.9% to 84.3%. Without them, it fell to 75.1% after one round and 74.7% after two, and GPT-4 fell from 95.5% to 91.5% and 89.0%. On CommonSenseQA, a multiple-choice test, GPT-3.5 dropped from 75.8% to 38.1%: told to look for problems, it switched away from right answers. Their summary: "The fundamental issue is that LLMs cannot properly judge the correctness of their reasoning." A [2024 survey](https://arxiv.org/abs/2406.01297) (Kamoi et al.) agrees that self-correction "works well in tasks that can use reliable external feedback". And [Tyen et al.](https://arxiv.org/abs/2311.08516) split the skill in two: models are poor at finding their reasoning mistakes but good at fixing them once told where they are.

Code is the easiest place to get reliable external feedback, because you can run it, so the experiment below uses code.

## Step 2: the experiment

### Forty functions with hidden tests

The task set is 40 Python functions, each specified by a docstring of about 126 words that states every rule and gives one example: a cron-expression matcher, a template renderer, an HTTP `Range` header parser, a tokenizer with nested comments, a digit-counting problem over numbers up to 10^18, and so on. Each task has 8 to 25 unit tests (526 in all). The first 3 to 5 per task (160 in all) are the public tests; the full set is the hidden suite used for grading. Reference solutions computed every expected output, and the trickier ones were checked against brute force on small inputs.

Finding tasks Haiku 5.5 gets wrong took three rounds. The first 40 were solved on the first try 38 and 39 times out of 40, at effort `low` and `medium`. Two rounds of harder tasks still gave 92 to 96%, with thinking on or off. Treat what follows as results for a strong model on clear specs, where most first attempts are right.

Model code ran in a throwaway directory under the scratch folder, in a fresh process with no network (a new network namespace via `unshare -rn`), a 1 GB memory limit, 10 seconds of CPU and a 15-second wall-clock timeout:

```python
p = subprocess.run(["unshare", "-rn", PY, "-I", "child.py"], cwd=d, capture_output=True,
                   text=True, timeout=timeout, preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
```

### Seven strategies from one first attempt

For every task and repeat, Haiku writes attempt 1 once, and every strategy starts from that same attempt. This pairs the comparisons: a difference between strategies can't come from a lucky first draft.

| Strategy | Feedback the model gets | When it runs | Rounds |
|---|---|---|---|
| Single attempt | none | | 1 call |
| Self-critique | its own review, prompted to check every rule | always | 2 |
| Test feedback | failing public tests, with expected and actual values | if public tests fail | up to 3 |
| Reflexion | the failing tests, then a lesson it writes before retrying | if public tests fail | up to 3 |
| Resample | none: a fresh, independent attempt | if public tests fail | up to 3 |
| Oracle feedback | failing hidden tests (an upper bound, not a real option) | if hidden tests fail | up to 3 |
| Test tool in the agent loop | a `run_public_tests` tool it may call | model decides | up to 6 steps |

![Flowchart of the correction strategies. Attempt 1 is written once. If the feedback comes from the model itself, it reviews its code against the spec and revises, and this always runs. If the feedback comes from a verifier, the public tests run; if all pass, the code is kept; if not, either the failures go back for a revision (test feedback) or the model writes a lesson and retries (Reflexion), and the tests run again.](/diagrams/a04-correction-loops.svg "The correction strategies compared. Self-critique always revises; verifier-based strategies revise only when a test fails.")

The critique prompt follows the default feedback prompt in Huang et al. ("Assume that this answer could be either correct or incorrect"). Here is the loop, from this part's experiment script, `reflect.py`:

```python
CRITIQUE = ("Review your code against the specification. Assume it could be either correct or "
            "incorrect. Check each rule and edge case carefully and report any real problems you "
            "find. Then give the final version of the complete function in a single ```python code "
            "block (unchanged if you found no real problems).")


def critique(llm, t, first, tag, rounds=2):
    """Intrinsic self-correction: no tests, no labels. Returns the code after each round."""
    messages = [{"role": "user", "content": ask(t)}, {"role": "assistant", "content": first}]
    out, tokens, cost = [], 0, 0.0
    for r in range(rounds):
        messages.append({"role": "user", "content": CRITIQUE})
        reply, tk, c = llm(messages, f"critique r{r + 1}", tag)
        messages.append({"role": "assistant", "content": reply})
        out.append(dict(code=extract_code(reply, t["id"]) or extract_code(messages[-3]["content"], t["id"]),
                        reply=reply, tokens=tk, cost=c))
    return out
```

Test feedback differs in two lines: it runs the tests first, and it stops as soon as they pass.

```python
def with_tests(llm, t, first, tag, rounds=3, oracle=False):
    """Revise with test feedback: run the tests, send failures back, repeat until they pass."""
    messages = [{"role": "user", "content": ask(t)}, {"role": "assistant", "content": first}]
    code, calls = extract_code(first, t["id"]), []
    for r in range(rounds):
        ok, fails = grade(t, code, public_only=not oracle)
        if ok:
            break
        messages.append({"role": "user", "content":
                         f"I ran your code against the test suite. {len(fails)} test(s) failed:\n"
                         + "\n".join(fails) + "\n\nFix the code and reply with the complete "
                         "corrected function in a single ```python code block."})
        reply, tk, c = llm(messages, f"{'oracle' if oracle else 'tests'} r{r + 1}", tag)
        messages.append({"role": "assistant", "content": reply})
        code = extract_code(reply, t["id"]) or code
        calls.append(dict(reply=reply, tokens=tk, cost=c, code=code))
    return code, calls
```

Reflexion first asks for a lesson ("explain what went wrong and write a lesson to follow on your next attempt. Do not write code"), then starts a fresh conversation containing the spec, the previous code, the failing tests and every lesson so far. The tool strategy is the [Part 1 harness](/blog/what-is-an-ai-agent/) with one tool.

Every call used `claude-haiku-5-5` with adaptive thinking at effort `medium` (its default) and `max_tokens` of 16,000, because early runs at 8,000 cut off some long answers. Each of the 40 tasks ran 5 times: 200 first attempts. Rates come with 95% Wilson intervals, a confidence interval for a proportion that behaves well near 0% and 100%. Because tasks repeat, read them as slightly too narrow.

## Step 3: results on the full task set

| Strategy | Pass rate (95% CI) | Right to wrong | Wrong to right | Tokens per task | Cost per task |
|---|---|---|---|---|---|
| Single attempt | 195/200, 97.5% (94.3-98.9) | | | 2,866 | $0.00128 |
| Self-critique, 1 round | 199/200, 99.5% (97.2-99.9) | 0 | 4 | 7,430 | $0.00296 |
| Self-critique, 2 rounds | 197/200, 98.5% (95.7-99.5) | 2 | 4 | 13,209 | $0.00463 |
| Test feedback | 198/200, 99.0% (96.4-99.7) | 0 | 3 | 2,926 | $0.00129 |
| Reflexion | 198/200, 99.0% (96.4-99.7) | 0 | 3 | 2,979 | $0.00131 |
| Resample | 198/200, 99.0% (96.4-99.7) | 0 | 3 | 2,917 | $0.00130 |
| Oracle feedback | 200/200, 100% (98.1-100) | 0 | 5 | 2,990 | $0.00132 |
| Test tool in the loop | 198/200, 99.0% (96.4-99.7) | | | 5,675 | $0.00155 |

Five first attempts were wrong. Three failed a public test, and every verifier strategy fixed those three. The other two passed the public tests and failed hidden ones (`human_size(1024**4 - 1)` printed `"1024.0 GiB"` instead of moving up to `"1.0 TiB"`), so no test-gated strategy ever looked at them. Self-critique fixed four of the five, including both that the public tests missed.

None of the differences is statistically solid. Comparing one round of self-critique with the single attempt, only 4 of the 200 runs changed outcome, all in one direction; an exact McNemar test (which looks only at the pairs that changed) gives p = 0.125. Treat the table as a description of 200 runs, not a ranking.

The strategies differ most in cost. Self-critique reviews every answer, so one round multiplied tokens by 2.6 and cost by 2.3: $0.084 per extra task solved ($0.337 of review for 4 fixes). Test feedback ran on 3 of 200 attempts and added $0.004 in total, $0.0013 per fix. The tool agent cost 21% more than a single attempt.

The second round of critique fixed nothing the first had missed and turned two passing answers into failing ones, the pattern Huang et al. report for GPT-4 on GSM8K.

## Step 4: when critique helps and when it hurts

Five wrong answers are too few to show which errors critique fixes, so two repair experiments start every strategy from a known-wrong attempt.

### Natural errors: slips versus misreadings

The first pool is every wrong first attempt Haiku made at the main settings across all runs: 12 attempts on 7 tasks, half of them the same mistake in `parse_duration`. Each went through every strategy 3 times (36 repairs per strategy).

| Strategy | Fixed (95% CI) | Public tests catch the bug (27) | Public tests miss it (9) |
|---|---|---|---|
| Self-critique, 1 round | 20/36, 55.6% (39.6-70.5) | 12 | 8 |
| Self-critique, 2 rounds | 24/36, 66.7% (50.3-79.8) | 17 | 7 |
| Test feedback | 27/36, 75.0% (58.9-86.2) | 27 | 0 |
| Reflexion | 27/36, 75.0% (58.9-86.2) | 27 | 0 |
| Resample | 26/36, 72.2% (56.0-84.2) | 26 | 0 |
| Oracle feedback | 36/36, 100% (90.4-100) | 27 | 9 |

In `parse_duration`, the code split the input on spaces, so `"1h30m"`, the spec's own example, was rejected. Critique fixed it 7 times out of 18 (38.9%, CI 20.3-61.4); in the other 11 it reviewed the code and passed it, sometimes asserting the example works, as in the opening quote. In `render_table`, the code sized a column by its cells and ignored the header, and critique fixed it 0 times out of 3. Test feedback fixed both every time, because the failure message showed the input, the expected output and the actual output.

Errors the model could see once it looked went the other way. In `tokenize`, the code rejected `1.foo` instead of reading it as the number 1 followed by `.foo`; critique fixed it 3 times out of 3, and the public tests never covered it. Overall, critique fixed 8 of the 9 repairs the public tests missed.

Classifying the replies shows where critique fails. In 17 of the 36 first-round reviews the model named a real problem, and it fixed the code every time. In the other 19 it called the code fine: 16 times the code stayed wrong, and 3 times it rewrote the function anyway and the rewrite passed. When Haiku found a problem, it was right; the failure is not seeing it, the split Tyen et al. describe.

Reflexion matched test feedback exactly, 27 for 27, at 1.7 times the cost per repair ($0.00169 against $0.00101), because the lesson is an extra call. Its lessons were accurate:

> The regex required a space before every component after the first, so `1h30m` failed because `30m` had no space in front of it. The spec says components *may* be separated by single spaces, which makes the space optional, not mandatory. ... I'll also check the grammar against the spec's own example, `"1h30m"`, before writing any code.

The test failure already carried that information, so the lesson added cost and no fixes.

### Planted bugs: slips are easy

The second pool isolates slips. A script took one correct solution Haiku wrote for each task and changed a single token: flipped a comparison (`<` to `<=`, `==` to `!=`), shifted an integer by one, or swapped `and` and `or`. It kept a mutant only if it still ran and failed between one test and 40% of the hidden tests. That gave 38 mutants, presented to Haiku as its own previous answer.

| Strategy | Fixed (95% CI) | Public tests catch it (21) | Public tests miss it (17) |
|---|---|---|---|
| Self-critique, 1 round | 35/38, 92.1% (79.2-97.3) | 18 | 17 |
| Test feedback | 21/38, 55.3% (39.7-69.9) | 21 | 0 |
| Oracle feedback | 38/38, 100% (90.8-100) | 21 | 17 |

Critique found 35 of 38 one-token bugs. A typical review, of a frame check changed from `> 10` to `> 9`:

> I found one real bug. The frame 1-9 check in the validation loop is `if rolls[i] + rolls[i + 1] > 9:`, which rejects every spare in frames 1-9 (a sum of exactly 10). ... The bug went unnoticed because the example `[10] * 12` contains only strikes.

The three misses (two flipped comparisons and one `and` swapped for `or`) sat in an expression parser, an HTTP range parser and a spreadsheet evaluator, where the wrong operator still reads plausibly. Planted bugs are easier for a reason: the model didn't put them there, so nothing in its reading of the spec defends them. The `parse_duration` bug came from the model's own reading and survived its review 11 times out of 18.

### How critique breaks working code

On the 195 right first attempts, the first round of critique claimed a real problem in 30 (15.4%, CI 11.0-21.1) and rewrote 39 functions (20.0%). None of those rewrites failed a hidden test. The second round claimed a problem in 21 and broke two, both on the same task:

> **One real problem:** My previous version started the loop at length 1 ... The specification says nothing about excluding the empty prefix. It says "shortest prefix," and the empty string is a prefix of every string. ... so `["dog"]` returns `[""]`.

The reviewer has a point: the spec never says the prefix must be non-empty, and the tests assume it. Under a literal reading, the change was defensible. It was also prompted by nothing but the instruction to look for problems, and it turned code that passed every test into code that failed one. In the natural pool the same task flip-flopped: round 1 fixed it 3 times out of 3, and round 2 undid 2 of those fixes. Claude Code's documentation warns: "A reviewer prompted to find gaps will usually report some, even when the work is sound."

### The break-even rule

A critique round pays off when the bugs it fixes outnumber the answers it breaks. Write `p` for the share of first attempts that are wrong, `f` for the share of those that critique fixes, and `b` for the share of right ones it breaks:

```text
gain from one round = p × f − (1 − p) × b
```

Here `p = 2.5%` and `f` between 56% (natural errors) and 80% (the main run) give 1.4 to 2.0 points, minus `97.5% × b`. Round 1 broke 0 of 195 (CI 0-1.9%) and probably came out ahead; round 2 broke 2 of 195 and lost a point back. The better first attempts get, the more the break rate decides the sign. A verifier gate sets `b` to zero: a strategy that runs only when a test fails never touches code that passes.

## Step 5: the verifier inside the agent loop

### When the model holds the test tool

In the tool strategy, Haiku got a `run_public_tests` tool and the instruction that it could check its code before answering. It called it once in 125 of 200 runs, twice in 22, three times in 1, and never in 52 (26.0%, CI 20.4-32.5). In 18 of the 148 runs that used it, the first draft failed a public test, and all 18 fixed it before answering. Both of the strategy's misses were runs that never called the tool.

That 12% draft failure rate (18 of 148, CI 7.8-18.4) is far above the 1.5% of standalone first attempts that failed a public test (3 of 200), and the first step of a tool run wrote fewer output tokens (median 1,506 against 1,910). That fits a model that tests drafts instead of checking them in its head, though it is a pattern, not a controlled comparison. Either way, a tool the model may skip is a weaker gate than one your code always runs, the [System Design article](/blog/reliable-tool-using-agents/)'s case for putting guardrails in code.

### Blind spots of the verifier

Every verifier strategy scored 0 on bugs the public tests didn't cover (0 of 9 natural, 0 of 17 planted), while oracle feedback, which saw the hidden tests, fixed everything. That's the Reflexion paper's MBPP problem in miniature: a test suite that passes wrong code ends the loop with a confident wrong answer. The fix is better tests, not more rounds.

Resampling, which gets no feedback at all, fixed 26 of the 27 caught natural bugs. For a model this strong a fresh draft is usually right, so the verifier's pass or fail signal did most of the work.

### A stronger reviewer

As a small comparison, `claude-sonnet-5-5` (effort `medium`) reviewed the same 12 wrong attempts for one round. It fixed 11 (91.7%, CI 64.6-98.5), including 5 of the 6 `parse_duration` attempts and the `render_table` one Haiku never fixed. Two of its 11 fixes came from rewriting code it had just called correct, as Haiku did. Each review cost $0.019, 12 times Haiku's $0.0016.

Yet writing `parse_duration` from scratch, Sonnet 5.5 made the same split-on-spaces mistake 4 times out of 4. It reviewed another model's code better than it wrote its own, which fits Claude Code's advice that "a fresh context improves code review since Claude won't be biased toward code it just wrote". With 12 and 4 runs, treat this as a hint.

All experiments in this article, including the three calibration rounds and two pilots, cost $2.87 on the ledger: $1.37 for the main run, $0.63 for calibration and pilots, $0.08 for extra first attempts to build the natural pool, $0.49 for the two repair experiments and $0.29 for Sonnet.

## In the real world

- Claude Code's [best practices](https://code.claude.com/docs/en/best-practices) lead with "Give Claude a way to verify its work": "Claude stops when the work looks done. Without a check it can run, 'looks done' is the only signal available." The suggested checks are tests, a build, a linter or a screenshot, enforced inside a prompt, by a `/goal` condition, by a Stop hook that blocks the turn until the check passes, or by a verification subagent.
- The same page warns that "chasing every finding leads to over-engineering" and suggests telling a reviewer to "flag only gaps that affect correctness or the stated requirements".
- Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) names the evaluator-optimizer workflow, "one LLM call generates a response while another provides evaluation and feedback in a loop", and says it fits "when we have clear evaluation criteria". The same post notes that "code solutions are verifiable through automated tests" and that agents need "ground truth" from the environment at each step.
- [CRITIC](https://arxiv.org/abs/2305.11738) (Gou et al., ICLR 2024) lets the model call tools such as a code interpreter while critiquing. On GSM8K with Text-Davinci-003 it raised a 70.1% baseline to 72.2%; without its tools it fell to 68.3%.

## Common misconceptions

- **"Asking the model to double-check always helps."** A second round here broke two passing answers, and on older models and reasoning tasks Huang et al. measured drops of up to 37.7 points. A critique that always runs puts every right answer at risk.
- **"If the model reviews its code and says it's fine, it's probably fine."** In the natural pool, Haiku said "no real problems" about wrong code 19 times out of 36, once while claiming the failing example returned the right value.
- **"Self-correction can't work without outside feedback."** For slips it can: critique fixed 35 of 38 planted one-token bugs with no tests. It struggles with errors that come from the model's own reading of the spec, which a second reading repeats.
- **"Passing the tests means the code is right."** Two first attempts passed every public test and failed hidden ones, and every test-gated strategy shipped them unchanged. A verifier is exactly as good as its tests.
- **"Reflexion's written lessons are what make it work."** Here it fixed the same 27 of 36 as plain test feedback, at 1.7 times the cost per repair. The test results carried the information.

## Key takeaways

- Where the feedback comes from decides the result. Tests fixed every bug they caught; self-critique fixed slips and missed misreadings.
- Self-critique revises whether or not a revision is needed. It rewrote 20% of correct answers in round one, and a second round broke two.
- Gate revisions on a verifier. Code that passes is never touched, and the extra cost here was $0.004 across 200 tasks, against $0.34 for one round of critique on every answer.
- Write tests for the cases that matter. Every test-based strategy scored zero on bugs the public tests missed.
- If a check is mandatory, run it in your code. The model skipped the test tool in a quarter of runs, and those skips were the only misses.

## What's next

Every strategy here grew the conversation: each critique round re-sent the spec, the code and the earlier reviews, and two rounds used 4.6 times the tokens of a single attempt. Part 5 is about context engineering: what fills a model's context window on a long task, how big tool results should be, and when to compact, edit or move history into memory files to keep a long run affordable.

## Further reading

- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366): Actor, Evaluator and Self-Reflection, self-written unit tests, and the ablation where reflection without tests hurt.
- [Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651): the generate, feedback, refine loop, with big gains on taste-based tasks and almost none on math.
- [Large Language Models Cannot Self-Correct Reasoning Yet (Huang et al., 2023)](https://arxiv.org/abs/2310.01798): intrinsic versus oracle self-correction, and how much the oracle label explains.
- [LLMs cannot find reasoning errors, but can correct them given the error location (Tyen et al., 2023)](https://arxiv.org/abs/2311.08516): finding mistakes and fixing them, measured separately.
- [When Can LLMs Actually Correct Their Own Mistakes? (Kamoi et al., 2024)](https://arxiv.org/abs/2406.01297): a critical survey and a checklist for fair self-correction experiments.
- [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing (Gou et al., 2023)](https://arxiv.org/abs/2305.11738): critique with tools, and the same method without them.
- [Building effective agents (Anthropic, 2024)](https://www.anthropic.com/engineering/building-effective-agents): the evaluator-optimizer workflow and why agents need ground truth from the environment.
- [Best practices for Claude Code (Claude Code docs)](https://code.claude.com/docs/en/best-practices): giving an agent a check it can run, and how to keep a reviewer from over-reporting.
- [Detecting and Recovering from Agent Loops](/blog/agent-loop-detection-recovery/): a Reflexion-style prompt used as a loop-recovery step in production.
