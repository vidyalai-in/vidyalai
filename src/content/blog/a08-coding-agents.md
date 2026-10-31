---
title: "Coding Agents: How They Fix Real Bugs, Measured"
description: "How coding agents solve SWE-bench-style issues with search, edit and test tools, measured on 28 bugs from real Python repos with Claude Haiku 5.5."
slug: "coding-agents"
category: "agents-tool-use"
tags: ["agents", "tool-use", "evals", "cost"]
series: "agentic-ai"
seriesOrder: 8
pubDate: 2026-07-07
---

## TL;DR

A coding agent fixes a bug the way a developer new to a codebase would: it searches for the code the issue talks about, reads it, edits a few lines and runs the tests. SWE-bench turned that job into a benchmark graded by hidden tests. This article builds a 28-task version of it from real bug fixes in 9 small Python libraries and runs a coding agent on Claude Haiku 5.5 with seven tool sets, 84 runs each. With search, string-replace edits and a test runner, it resolved 83 of 84 runs for $0.0018 each. Replacing string-replace edits with whole-file writes cut that to 66 of 84 and cost 7.5 times as much, mostly because the model overwrote files with placeholders. Giving it only the issue's title cut it to 72. Removing search, adding a repository map or taking away the test runner changed little on these small bugs, and Claude Sonnet 5.5 matched Haiku at 8.6 times the cost.

## Why it matters

Here is Claude Haiku 5.5 on a bug in marshmallow, a Python serialization library. At step 5 it had found the exact line at fault:

```text
713:        return only_field.data_key or self.field_name
```

Its edit tool in that run could only write whole files, and the file was 2,153 lines long. At step 7 it called the tool with the path of that file and the content `PLACEHOLDER`. The tool replied `Wrote src/marshmallow/fields.py: 1 lines (was 2154).` The model's next thought: "I just realized I overwrote the entire fields.py file with a placeholder, which is a serious problem". It searched for a backup, found none, and stopped with an honest apology.

With a string-replace edit tool, the same model fixed that bug in all 3 of its runs. When SWE-bench was published in 2023, the best model resolved 1.96% of its issues without an agent loop. Anthropic's system card for Claude Haiku 5.5 reports 64.8% on SWE-bench Pro, a harder successor, as an average over five trials.

[Part 3](/blog/react-planning/) covered how an agent reasons between actions and [Part 4](/blog/agent-self-correction/) showed that tests are the most reliable feedback an agent gets. This part puts both inside an agent that edits code.

## The intuition

Picture a contractor hired to fix one bug in a codebase they have never seen. They get a bug report and a laptop with the repository. They don't read all 6,000 lines. They search for a word from the report, open the file that turns up, read the function around the match, write a small test that shows the bug, change two lines, and run the tests before saying they're done.

A coding agent does the same steps through tools: a search tool, a file viewer, an edit tool and a test runner. How well it works depends on how those tools are shaped, the way the contractor's speed depends on a good editor and a fast test suite.

The analogy breaks in two places. The contractor remembers what they read; the agent has only what is in its context window, and pays for every token there on every step. And the contractor sees the whole screen; the agent sees exactly what a tool returns, so a tool that returns too little or too much changes what it can do.

## Step 1: what a SWE-bench task is

### Built from pull requests

[SWE-bench](https://arxiv.org/abs/2310.06770) (Jimenez et al., ICLR 2024) turned real GitHub history into a benchmark. The authors collected about 90,000 pull requests from 12 popular Python repositories and kept those that resolve an issue and change test files. They ran the tests before and after each fix and kept a task only if at least one test went from failing to passing. That left 2,294 tasks.

Each task gives the model "an issue text description and a complete codebase", and the model returns a patch. Two lists of tests grade it:

- `FAIL_TO_PASS`: tests that fail before the real fix and pass after it. They check that the bug is fixed.
- `PASS_TO_PASS`: tests that pass both before and after. In the paper's words, they "ensure that the revision does not break or violate any existing expected behavior."

A task counts as resolved only if the patch applies and every test in both lists passes. The fail-to-pass tests stay hidden from the model, as they would from a developer who has only the issue.

### Why it was hard in 2023

The codebases were large: 3,010 non-test files and 438,000 lines on average, while the real fixes touched 1.7 files and 32.8 lines. Models could not read a whole repository, so the paper picked files with BM25, a classic keyword-ranking method. The best model in the paper's first version, Claude 2, resolved 1.96%. Handed the files the real fix edited (the "oracle" setting), it reached 4.8%. In almost half the tasks BM25 found none of the right files, and the authors concluded that models "are simply ineffective at localizing problematic code."

### Verified, then contaminated

In August 2024 OpenAI and the SWE-bench authors released [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/). Ninety-three Python developers labeled 1,699 tasks, three annotators each. They flagged 38.3% for underspecified issues and 61.1% for tests that "may unfairly mark valid solutions as incorrect", and 500 tasks survived. GPT-4o scored 33.2% on Verified against 16% on the original set.

In February 2026 OpenAI [stopped reporting](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) Verified scores because the set "is increasingly contaminated": "all frontier models we tested were able to reproduce the original, human-written bug fix" or details of the problem statement for some tasks. Its audit found problems in the tests or the description of 59.4% of 138 tasks it examined. Anthropic's system cards for Claude Haiku 5.5 and Sonnet 5.5 report SWE-bench Pro instead, a set "drawn from actively-maintained repositories with large, multi-file diffs".

| System | Benchmark | Resolved | Source |
|---|---|---|---|
| Claude 2, BM25 retrieval, no agent loop | SWE-bench (2,294 tasks) | 1.96% | SWE-bench paper |
| GPT-4 Turbo in SWE-agent | SWE-bench Lite (300 tasks) | 18.0% | SWE-agent paper |
| GPT-4o in Agentless | SWE-bench Verified (500 tasks) | 33.2% | OpenAI, Aug 2024 |
| Claude 3.5 Sonnet (upgraded), bash and edit tools | SWE-bench Verified | 49% | Anthropic, Jan 2025 |
| Claude Haiku 5.5, mean of 5 trials | SWE-bench Pro | 64.8% | Haiku 5.5 system card |
| Claude Sonnet 5.5, mean of 5 trials | SWE-bench Pro | 81.3% | Sonnet 5.5 system card |

None of these rows was measured here, and they don't compare with each other: the task sets, years and scaffolds (the agent code and tools around the model) differ.

## Step 2: the tools are the interface

### The agent-computer interface

[SWE-agent](https://arxiv.org/abs/2405.15793) (Yang et al., NeurIPS 2024) named the design problem: an agent works through an agent-computer interface (ACI), "the commands the agent uses and the format of the feedback from the computer". Its interface had a search capped at 50 results, a file viewer showing 100 lines at a time, and an edit command whose linter rejects edits that introduce syntax errors.

Its ablations on SWE-bench Lite with GPT-4 Turbo are the closest published cousin of the experiment below:

| SWE-agent interface | Resolved on Lite |
|---|---|
| Default interface | 18.0% |
| Edit without the linter | 15.0% |
| No edit command (shell only) | 10.3% |
| No search commands | 15.7% |
| Iterative search (one result at a time) | 12.0% |
| File viewer shows 30 lines | 14.3% |
| File viewer shows the full file | 12.7% |

Viewing too little and too much both hurt. And a clumsy search tool was worse than none: iterative search cost 6 points, removing search 2.3.

### Search, view, edit, run

Anthropic's [SWE-bench Verified agent](https://www.anthropic.com/research/swe-bench-sonnet) for Claude 3.5 Sonnet had two tools, "a Bash Tool for executing bash commands, and an Edit Tool, for viewing and editing files and directories", and "many successful runs took hundreds of turns". The edit tool became the API's [text editor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool) (`text_editor_20250728`), built on string replacement: the model sends an exact snippet and its replacement, and the docs ask the application to "ensure that there is exactly one match for the old text or provide appropriate error messages."

[Agentless](https://arxiv.org/abs/2407.01489) (Xia et al., 2024) is the counterpoint: a fixed pipeline of localization, repair and patch validation "without letting the LLM decide future actions". With GPT-4o it resolved 32.00% of SWE-bench Lite at an average of $0.70 per issue, the best score among the open-source agents it was compared with.

![Flowchart of the coding agent. The issue text and a repository checkout go to a localize step (grep, glob, read_file), then a reproduce-and-edit step (create_file, str_replace), then run_tests in a sandbox with no network. If the tests fail, the agent goes back to localizing. If they pass, the agent stops and the grader restores the hidden tests and runs them. If every fail-to-pass and pass-to-pass test passes, the task is resolved; if any fails, it is not.](/diagrams/a08-coding-agent-loop.svg "The coding agent's loop and the grader. The agent never sees the fail-to-pass tests.")

The agent's tests and the grader's tests are different sets, so an agent can pass every check it runs and still fail the grader.

## Step 3: a 28-task benchmark from real repositories

### Mirroring real fixes

Real SWE-bench tasks need each repository's old dependencies, which is why the official harness gives every task its own Docker image. This experiment borrows a shortcut from [SWE-smith](https://arxiv.org/abs/2504.21798) (Yang et al., 2025), which calls it PR mirroring: take a fix that already landed and "undo its revisions in the current version of the repository". The bug returns in today's code, the tests the fix added become hidden tests, and everything runs in one Python 3.12 environment.

A script (`harvest.py`) walked the last 700 commits of 11 small, permissively licensed Python libraries, looking for fix commits that changed both tests and at most 40 lines of source. It reverse-applied the source part of each to the current `HEAD`, ran the commit's test files, and kept the tests that passed before the revert and failed after. That gave 69 candidates in 9 repositories; 28 were picked to spread across repositories and kinds of bug:

| Repository | License | `HEAD` commit | Source lines | Tasks |
|---|---|---|---|---|
| sqlparse | BSD-3-Clause | 60cdc64 | 4,363 | 5 |
| parse | MIT | 2825afb | 1,107 | 4 |
| marshmallow | MIT | b762ea2d | 5,050 | 4 |
| mistune | BSD-3-Clause | f7ec702 | 6,424 | 3 |
| humanize | MIT | a2006ea | 1,726 | 3 |
| tomlkit | MIT | eddd27e | 6,347 | 3 |
| cachetools | MIT | 6f23e2a | 1,713 | 2 |
| toolz | BSD-3-Clause | 451af60 | 3,799 | 2 |
| python-semver | BSD-3-Clause | 15a42d0 | 1,608 | 2 |

The real fixes changed 1 to 16 lines of one source file (median 4), far smaller than SWE-bench's average of 32.8. The 28 tasks have 88 fail-to-pass and 3,231 pass-to-pass tests. Nineteen fixes landed after June 2026, the training-data cutoff Anthropic's [models overview](https://platform.claude.com/docs/en/about-claude/models/overview) lists for Haiku 5.5 and Sonnet 5.5, so the model can't have seen those fixes, though it may have seen the buggy code.

### Issue, checkout and grader

Each task has a short issue written for this experiment in the style of a bug report: a title, a reproduction in a Python session and the expected behavior, never the file or the cause. One reads, in part:

```text
Lowercase month names crash date parsing

Matching is case-insensitive, so `{:tg}` happily matches a lowercase month name, but the
conversion then fails:

    >>> parse.parse('{:tg}', '1/Jul/2011')     # works
    >>> parse.parse('{:tg}', '1/jul/2011')
    KeyError: 'jul'
```

The agent's checkout is the repository with the fix undone and the fail-to-pass test functions deleted from the visible test files; agent instruction files (`AGENTS.md`, `CLAUDE.md`) were removed. After the run, the grader copies the original test files back, discarding any edits the agent made to them, and counts the task as resolved only if every fail-to-pass and pass-to-pass test passes. As a check, the untouched checkouts failed all 28 tasks and the checkouts with the real fix applied passed all 28.

Model-written code ran only in a throwaway copy under the scratch folder, in a new network namespace with no interfaces (`unshare -rn`), with a 2 GB memory limit and 120 seconds of CPU per test run, inside a 3 GB cap for the whole experiment.

### The agent and seven tool sets

The agent is the [Part 3](/blog/react-planning/) loop (`react.run_loop`) with tools from this part's `coding.py`. Every path is relative to the repository root, and a path that resolves outside it is an error. The edit tool follows the text editor tool's rule:

```python
        p = resolve(path)
        text = p.read_text()
        n = text.count(old_str)
        if n != 1:
            raise ValueError(f"old_str occurs {n} times in {path}; it must occur exactly once."
                             + (" Check whitespace and indentation, or read the file again." if n == 0 else
                                " Add surrounding lines to make it unique."))
        p.write_text(text.replace(old_str, new_str, 1))
        start = text[:text.index(old_str)].count("\n") + 1
        new_lines = p.read_text().splitlines()
        lo, hi = max(1, start - 3), min(len(new_lines), start + new_str.count("\n") + 3)
        return f"Edited {path}. Lines {lo}-{hi} now read:\n" + "\n".join(
            f"{i:5d}  {new_lines[i - 1]}" for i in range(lo, hi + 1))
```

The full kit is `grep` (up to 50 matching lines), `glob`, `read_file` (up to 400 numbered lines per call), `str_replace`, `create_file` (new files only, for a reproduction test) and `run_tests` (pytest with the model's arguments). Each other arm changes one thing:

| Arm | What changes from the full kit |
|---|---|
| Full | nothing |
| No test runner | no `run_tests`; the prompt says code can't be run |
| Whole-file rewrite | `write_file` (send the complete file) replaces `str_replace` and `create_file` |
| No search | `list_dir` replaces `grep` and `glob` |
| Repo map | the task message adds a map: every source file with its classes, methods and functions and their line numbers (about 2,000 tokens at most) |
| Repo map, no search | the map, and `list_dir` instead of `grep` and `glob` |
| Title only | the issue is cut to its first line, with no reproduction |

The map is a simple version of the one in the coding assistant [Aider](https://aider.chat/docs/repomap.html), whose map budget "defaults to 1k tokens".

Every run in the main experiment used `claude-haiku-5-5` with adaptive thinking at effort `medium` (its default), `max_tokens` of 32,000, prompt caching, and limits of 50 steps and $0.20 per run, which no run reached. Each of the 28 tasks ran 3 times in each arm: 84 runs per arm. Intervals are 95% Wilson intervals; because tasks repeat, read them as slightly too narrow. Paired comparisons use an exact McNemar test on the 84 matched task-and-repeat pairs, which counts only the pairs where one arm resolved the task and the other didn't.

## Step 4: seven tool sets, measured

| Arm | Resolved (95% CI) | Steps | Tokens per run | Cost per run | Cost per resolved task |
|---|---|---|---|---|---|
| Full | 83/84, 98.8% (93.6-99.8) | 6.8 | 31,505 | $0.00177 | $0.00179 |
| No test runner | 79/84, 94.0% (86.8-97.4) | 5.3 | 21,168 | $0.00148 | $0.00157 |
| Whole-file rewrite | 66/84, 78.6% (68.7-86.0) | 14.0 | 217,156 | $0.01335 | $0.01699 |
| No search | 83/84, 98.8% (93.6-99.8) | 7.4 | 57,916 | $0.00278 | $0.00282 |
| Repo map | 81/84, 96.4% (90.0-98.8) | 5.5 | 34,939 | $0.00187 | $0.00194 |
| Repo map, no search | 80/84, 95.2% (88.4-98.1) | 5.2 | 37,795 | $0.00204 | $0.00214 |
| Title only | 72/84, 85.7% (76.7-91.6) | 8.6 | 63,531 | $0.00280 | $0.00327 |

Tokens per run count everything the model read and wrote, cached or not. Two differences are far outside noise: whole-file rewrite lost 18 pairs the full kit won and won 1 (p = 0.0001), and title only lost 11 and won none (p = 0.001). The other four are within noise: no test runner lost 4 pairs and won none (p = 0.125), and the two map arms lost 3 and 4 and won 1 each.

On tasks this small the full kit is near the ceiling, so the other differences show up in cost. Removing search raised cost by 57% for the same result. The repo map moved the first source edit from step 4.0 to step 2.9 and cost 6% more, because the map rides along in every request. Without a test runner, runs were 16% cheaper and 5 points worse.

A typical full-kit run is short. On the `parse` month-name bug, Haiku grepped for `MONTH|month`, read lines 140 to 290 of `parse/__init__.py`, changed `MONTHS_MAP[m]` to `MONTHS_MAP[m.capitalize()]`, ran the tests and stopped: 5 model calls, $0.0011. In 28 of 84 full-kit runs it also wrote its own reproduction test.

### Did the model know these fixes?

If the model remembered the fixes, older ones would be easier. They weren't: with the full kit, Haiku resolved 26 of 27 runs on the 9 tasks fixed before July 2026 and 57 of 57 on the 19 fixed after its training-data cutoff; with only the title, 85.2% and 86.0%. The one task both models failed most often was fixed in 2018.

### A stronger model

The full kit and the title-only arm also ran on `claude-sonnet-5-5`, with adaptive thinking at effort `medium` (Sonnet's default is `high`) and a $2 limit per run, for 3 and 2 repeats. With the full kit it resolved 81 of 84 runs (96.4%, CI 90.0-98.8), and its only failures were the three runs of one task, described below, where it made the same mistake as Haiku. With only the title it resolved 51 of 56 (91.1%, CI 80.7-96.1); on the same 56 task-and-repeat pairs Haiku resolved 48, and Sonnet won 5 pairs and lost 2 (p = 0.45, within noise).

Sonnet worked differently. It took 4.6 steps instead of 6.8, read 14,218 tokens per run instead of 31,505, and ran the tests in only 39 of its 84 full-kit runs, against 82 for Haiku. A run cost $0.0152, 8.6 times Haiku's, for no gain on these tasks.

## Step 5: how the runs failed

### Whole-file writes destroy files

The whole-file arm failed 18 times, and in 17 of them the model had at some point replaced a source file with something less than half its length. In 25 of its 84 runs it wrote a one-line placeholder over a source file of more than 50 lines: `PLACEHOLDER` in 16 of them, `PLACEHOLDER_REPLACED_BELOW` in 6, as if the real content would follow in a later call. Sending a 2,153-line file back in full means generating about 20,000 tokens of exact code, so the model reached for shortcuts: 35 times it tried to write a script to `/tmp/patch.py` that would do a string replacement, rebuilding the missing tool, and the path check blocked every one.

Of the 30 runs that shrank a source file at some point, 13 still resolved the task, most by reading the file again and writing it back in full. Two did it by writing a test file that ran `git show HEAD:src/humanize/time.py` in a subprocess, through the only code-execution door they had:

> There's no backup copy in the tree, but the repo's `.git` history should still hold the original file. I'll write a small temporary test that uses `git show` to restore `src/humanize/time.py` from HEAD and reapplies the one-line change. Pytest runs Python, so the test can do the file I/O.

That worked because the harness had made the checkout a git repository to record the diff. A test runner executes whatever the model writes, which is why sandboxing, the subject of Part 13, matters. Across the arm, runs averaged 14,165 output tokens against 1,789 for the full kit, and 83 seconds against 24.

### Skipping the tests that would have caught it

The `cachetools` task asks for `RRCache.popitem()` to raise `KeyError` when the cache is empty, even with a custom choice function. Most fixes rewrote this block:

```python
        try:
            key = self.__choice(self.__keys)
        except IndexError:
            raise KeyError("%s is empty" % type(self).__name__) from None
```

and dropped the `from None`, which hides the original exception. An existing, visible test checks exactly that. With the full kit, Haiku ran the tests after editing and resolved 3 of 3 runs. With the repo map it went straight to the right method, edited it within 3 steps, never ran a test, and failed 6 of 6 runs; without a test runner it failed 3 of 3. That one task accounts for 9 of the 12 failures in those three arms. In the five arms that had both a test runner and string-replace edits, 10 of 420 runs never called the runner; 6 of those 10 failed, against 15 of the 410 runs that did.

With the map, the agent found the code sooner and, on this task, tested less. [Part 4](/blog/agent-self-correction/) found the same: a check the model may skip is weaker than one the harness always runs.

### Underspecified issues

Cutting the issue to its title cost 11 resolved pairs, and the failures were partial fixes. "fractional() gives degenerate output when the fraction rounds to a whole number" led every run to fix `2.9999999`, which printed `2 1/1`, and none to fix `0`, which printed `0/1`; the full issue listed both. On the `tomlkit` comma-first array bug, all 3 title-only runs fixed a different, plausible bug from the one the hidden tests check. This is the underspecification SWE-bench Verified filtered out of 38.3% of tasks.

### Fixing more than the issue asked

The only task the full kit failed, with either model, was `toolz.accumulate([])`, which raised `RuntimeError` instead of returning an empty list. Haiku failed it 8 times across the seven arms. In 7 of those 8 failures (the eighth was a destroyed file) and in all 5 of Sonnet's, the fix also changed `initial == no_default` to `initial is no_default`, a tidy-looking change that breaks a hidden test passing a pickled copy of the sentinel string, which is equal but not identical. The test suite's existing tests for other functions use the same pickled sentinel, so the convention was discoverable, but nothing in the issue pointed to it. By SWE-bench Verified's criteria the test might count as too strict; for the library's users, the change is a regression.

### Without search, the edit tool becomes a search tool

Without `grep`, runs took in 2.8 times as many characters of tool output and paid 57% more, yet resolved as often. They also improvised: in 31 of 84 runs Haiku called `str_replace` with identical old and new text, such as `"zzzz"` or `"class Pluck(Nested):"`, to learn from the error message whether and how often a string occurs in a file. Partly because of these probes, 26.1% of its `str_replace` calls failed, against 9.1% with the full kit. SWE-agent reported that 51.7% of its GPT-4 Turbo trajectories had at least one failed edit; here 6 of 84 full-kit runs did.

All experiments in this part cost $4.63 on the ledger: $2.19 for 588 Haiku runs, of which $1.12 was the whole-file arm, $2.40 for 140 Sonnet runs, and $0.04 for pilots.

## In the real world

- Claude Code, Anthropic's coding agent, is built from the same pieces. Per its [tools reference](https://code.claude.com/docs/en/tools-reference), its Edit tool "performs exact string replacement", and `old_string` "must appear exactly once". On macOS, Linux and WSL, "Claude Code leaves Glob and Grep out of the default tool set, and Claude searches with `find` and `grep` through the Bash tool instead."
- Anthropic's [SWE-bench post](https://www.anthropic.com/research/swe-bench-sonnet) reached the same edit design by testing: "We experimented with several different strategies for specifying edits to existing files and had the highest reliability with string replacement." When models mixed up relative paths, "we simply made the tool always require an absolute path."
- [SWE-smith](https://arxiv.org/abs/2504.21798) generated 50,137 task instances from 128 repositories, including 2,344 by PR mirroring, and used them to train SWE-agent-LM-32B to 40.2% on SWE-bench Verified.

## Common misconceptions

- **"Coding benchmarks measure the model."** They measure a model inside a scaffold. Here one model, on the same tasks, went from 98.8% to 78.6% when one tool changed from string replacement to whole-file writes.
- **"More context about the repository always helps."** The repo map got the agent to the right code a step sooner and resolved no more tasks; on one task it led the model to skip the test run that would have caught its regression.
- **"If the agent's tests pass, the issue is fixed."** The fix tests are hidden. The full kit's one failure ran the visible tests, saw them pass, and still broke the hidden test for `accumulate`.
- **"A model that knows the repository from training has an unfair edge."** On these tasks, fixes the model can't have seen were resolved as often as older ones. Contamination is real for public benchmarks, as OpenAI's audit shows, but it doesn't explain this result.
- **"Whole-file writes are simpler and therefore safer."** They were the most dangerous tool here: a quarter of runs overwrote a source file with a placeholder.

## Key takeaways

- A coding agent is a search, read, edit, test loop, and SWE-bench grades it with hidden fail-to-pass tests plus pass-to-pass tests that catch regressions.
- Edit-tool design mattered most. String replacement with a uniqueness check resolved 83 of 84 runs; whole-file writes resolved 66, cost 7.5 times as much and destroyed files.
- The issue matters as much as the tools. With only a title, the agent fixed the case it guessed and missed the rest.
- On small, well-described bugs, search tools, a repo map and even the test runner changed cost and speed more than success. SWE-bench's repositories are about 100 times larger, which is where SWE-agent measured search mattering.
- Make the checks the harness's job. Six of the 10 runs that skipped an available test runner failed, against 15 of the 410 that used it.

## What's next

Coding agents work through text: files, diffs and test output. [Part 9](/blog/computer-use-agents/) takes away the text interface. A computer-use agent sees the screen as screenshots and acts with clicks and keystrokes, and the next part measures what a screenshot costs in tokens and how such an agent fares on a small set of sandboxed browser tasks.

## Further reading

- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (Jimenez et al., 2023)](https://arxiv.org/abs/2310.06770): the benchmark, its fail-to-pass and pass-to-pass tests, and the retrieval baselines.
- [Introducing SWE-bench Verified (OpenAI, 2024)](https://openai.com/index/introducing-swe-bench-verified/): human screening of 1,699 tasks for underspecified issues and unfair tests.
- [Why SWE-bench Verified no longer measures frontier coding capabilities (OpenAI, 2026)](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/): the contamination audit and the switch to SWE-bench Pro.
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (Yang et al., 2024)](https://arxiv.org/abs/2405.15793): the ACI idea and the interface ablations.
- [Agentless: Demystifying LLM-based Software Engineering Agents (Xia et al., 2024)](https://arxiv.org/abs/2407.01489): a fixed localize, repair and validate pipeline instead of an agent loop.
- [SWE-smith: Scaling Data for Software Engineering Agents (Yang et al., 2025)](https://arxiv.org/abs/2504.21798): PR mirroring and other ways to make task instances.
- [Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet (Anthropic, 2025)](https://www.anthropic.com/research/swe-bench-sonnet): a two-tool scaffold and the tool-design lessons behind it.
- [Text editor tool (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool): the `str_replace` edit tool and its uniqueness rule.
- [Claude Haiku 5.5 system card (Anthropic, 2026)](https://www.anthropic.com/claude-haiku-5-5-system-card): SWE-bench Pro, Multilingual and Multimodal results.
- [Repository map (Aider docs)](https://aider.chat/docs/repomap.html): how a coding assistant summarizes a repository in about 1,000 tokens.
