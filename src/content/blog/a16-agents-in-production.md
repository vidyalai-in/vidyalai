---
title: "An Agent End to End: A Docs-Maintenance Agent, Measured"
description: "One docs-maintenance agent built from everything this series measured, run 612 times on real docs with seeded defects: what it finds, misses and costs."
slug: "agents-in-production"
category: "agents-tool-use"
tags: ["agents", "tool-use", "evals", "cost", "security"]
series: "agentic-ai"
seriesOrder: 16
pubDate: 2026-10-25
---

## TL;DR

This finale builds one agent that does a real job: it reviews a documentation page for broken cross-references, code examples that no longer run, and sentences that contradict the library's source code, then proposes a fix for each. It is built from the parts of this series: strict tools, adaptive thinking at a set effort level, a verifier, small tool results, checkpoints and budgets, a sandbox with a permission policy, and evaluation with repeats. On 18 pages of the python-semver docs with 24 seeded defects, Claude Haiku 5.5 found 98.3% of them across 90 runs, every recorded fix passed its check, and it cost $0.0014 per page. It also found four real defects already in the shipped docs, one of which a hand audit had missed. It caught only 60% of a harder set of facts that only the source code contradicts, and it tended to stop looking once the tools had handed it something to fix.

## Why it matters

In 2013, a tool called DocRef checked the documentation of five widely used API libraries and found "more than 1,000 new documentation errors," such as "broken code names and obsolete code samples." Documentation rots because nothing runs it. The code changes, the tests change with it, and the page that explains the code keeps describing last year's behavior.

That is a good job for an agent: tedious, in need of reading and judgment, and safe, because a human reviews the patch. It is also a good test of this series. Each part measured one piece of an agent: [the loop](/blog/what-is-an-ai-agent/), [tool calls](/blog/tool-calling-internals/), [thinking](/blog/react-planning/), [verifiers](/blog/agent-self-correction/), [context](/blog/context-engineering/), [reliability](/blog/agent-reliability/), [sandboxes](/blog/agent-sandboxing/) and [evaluation](/blog/agent-evals/). This part puts them in one harness and asks two questions: does the result work, and which pieces earned their place?

## The intuition

Think of a fact-checker at a magazine. Some checks are mechanical: does this phone number ring, does this quote appear in the transcript? A junior assistant with a script can do those. Other checks need reading: the article says the drug was approved in 2019, so someone has to find the approval letter and compare.

The docs agent works the same way. A link checker and a test runner do the mechanical checks, and what they report is always true. The model does the reading: it compares each sentence with the code that implements it, and it writes the fix.

The analogy breaks in two places. A fact-checker's source is usually right; here the source of truth is code, which can itself have bugs, so "the page disagrees with the code" means "fix the page" only by policy. And a fact-checker who finds three errors on a page does not usually stop reading. This agent sometimes did.

## Step 1: a docs folder with known defects

### The corpus

The input is a copy of the documentation of python-semver, a small library for semantic version numbers (version 3.1.0, commit `15a42d0`), made in a scratch folder. The repo itself is never edited. Its docs are written in reStructuredText (a plain-text markup format) and built with Sphinx, the standard Python documentation generator.

Three things on a page can go stale. A cross-reference is a link written as a role, such as ``:meth:`~semver.version.Version.bump_major` `` for a method or ``:ref:`sec.replace.parts` `` for a section label; it breaks when the target is renamed. A doctest example is a snippet in the style of an interactive Python session, a `>>>` line followed by the output it should print; Python's `doctest` module "executes those sessions to verify that they work exactly as shown." And a factual claim is a sentence such as "By default (False), the version string to parse has to follow the semver specification", which only the code can confirm.

The agent reviews the 18 pages of the "Usage" and "Migration" sections: 3,273 words, 108 cross-references and 156 doctest examples. The source of truth is the library itself, `src/semver/` (1,608 lines of Python) plus its changelog.

### Seeding defects

To measure recall (the share of real defects found) and precision (the share of reports that are real), you need to know where the defects are. So a script seeds 24 of them, each a one-line edit, eight of each kind:

| Kind | Example seed | What the page now says | What the code says |
|---|---|---|---|
| link | L3 | ``:ref:`increase-parts-of-version` `` | the label is `increase-parts-of-a-version` |
| link | L7 | ``<semver.versions.Version>`` | the module is `semver.version` |
| code | C6 | `bump_prerelease()` on `3.4.5-rc9` gives `'3.4.5-rc10'` | it gives `'3.4.5-rc9.0'` |
| code | C7 | `version.replace_parts(major=2, minor=2)` | the method is `replace` |
| fact | F1 | "By default (True)" for `optional_minor_and_patch` | the default is `False` |
| fact | F5 | `bump_when_empty=True` belongs to `bump_patch` | it is a parameter of `bump_prerelease` |

Half of the fact seeds are also contradicted by an example on the same page; the other half only by the source code. A second, harder set of six fact seeds goes into a separate copy: claims that only the source contradicts, with no example nearby. One says an incomplete dictionary's missing parts become `None` when compared (they become zero), another that a match expression without an operator means "at least this version" (it means "equal").

### Scoring

A report matches a seed if it is on the same page within three lines of it. A fix counts as correct if, applied to the page, the broken link resolves or the failing example passes with nothing new broken, or, for facts, if the edited lines no longer contain the wrong claim and do contain the right one (checked with a pattern per seed). Everything else gets read by hand and labeled as either a real defect that was already in the docs or a false positive.

Before running anything, a hand audit of the clean pages found three real defects in the shipped docs: two cross-references to `Version.isvalid`, a method that no longer exists, and a page saying `match()` "expects two arguments: a version string and a match expression" when it takes one. The agent later found a fourth.

## Step 2: the agent, built from the series

The agent reviews one page per run. Pages are independent, so there is no orchestrator; the runner runs three pages at a time. The lab module is `finale.py`, 364 lines on top of the [Part 1](/blog/what-is-an-ai-agent/) harness and [Part 12](/blog/agent-reliability/)'s durable loop.

![Flowchart of one review. A docs page goes into the agent loop, which runs Claude Haiku 5.5 at low effort. Every tool call passes a policy that allows or denies it. Read calls go to the page and the library source; check calls go to a sandbox that runs the link check, the examples and fix tests; both return results to the loop. Report calls record a finding, one per line, and the findings become a patch for human review.](/diagrams/a16-docs-agent.svg "One review: the model reads and decides, deterministic checks run in a sandbox, and findings go to a human as a patch.")

The model never writes to the docs: its only output is a list of findings that becomes a patch.

| Piece | From | Setting here |
|---|---|---|
| Strict tool schemas, descriptions that say when to use a tool | Part 2 | 7 tools, `strict: true` |
| Adaptive thinking (the model decides when to think) with an effort level (how much it thinks and writes) | Part 3 | effort `low` (and `medium` in Step 4) |
| A verifier | Part 4 | `check_fix` applies a fix to a scratch copy and re-runs the checks |
| Small tool results, one task per context window | Part 5 | 150-line reads, 40 search hits, one page per run, prompt caching (reusing the processed prompt prefix at a tenth of the price) |
| Exact string-replace edits | Part 8 | every fix is `old_text` to `new_text` |
| Repeats and intervals | Part 11 | 5 runs per page, 95% Wilson intervals |
| Checkpoints, budgets, idempotent tools (a repeated call changes nothing) | Part 12 | checkpoint per step, 30 steps and $0.05 per page, one finding per page, line and kind |
| Sandbox and permission policy | Part 13 | bubblewrap, no network, read-only corpus; allow or deny |

### Seven tools

`read_doc` and `read_source` return numbered line ranges; `search_source` greps the library. `check_links` resolves every cross-reference against the docs tree and the imported library and suggests close matches. `run_examples` runs the page's doctests. `check_fix` is the verifier, and `report_issue` records a finding. Its schema forces a fix:

```python
@tool("Report one defect on the page under review, with a proposed fix. kind is 'link', 'code' or "
      "'fact'. line is the line number where the defect is. old_text is copied exactly from the page "
      "and must occur once; new_text replaces it. Reporting the same page, line and kind again "
      "replaces the earlier report.",
      path={"type": "string"}, line={"type": "integer"},
      kind={"type": "string", "enum": ["link", "code", "fact"]},
      problem={"type": "string", "description": "One sentence: what is wrong and what the code says."},
      old_text={"type": "string"}, new_text={"type": "string"})
```

### Why the checks run in a sandbox

`check_fix` runs whatever doctest the model writes into its fix. That is model-written code, so it runs inside bubblewrap, a small Linux sandboxing program and the same one [Part 13](/blog/agent-sandboxing/) measured, with every namespace unshared, an empty environment and the corpus mounted read-only:

```python
cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
       "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "PYTHONPATH", "/corpus/src",
       "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
       "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
       "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
       "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
       "--ro-bind", str(corpus), "/corpus", "--ro-bind", str(CHECKERS), "/checkers"]
```

A hand-written fix that tried to escape confirmed it: inside, the home directory did not exist, writing to the docs raised "Read-only file system," and opening a socket raised "Network is unreachable." Each check took a median of 47 ms (links) and 55 ms (examples) in the sandbox against 38 ms and 42 ms without it, over 30 runs each, so isolation added 10 to 14 ms to a step whose model request, including the ledger's token count, took a median of 3.7 seconds.

The permission policy is short, because the agent can't change anything: it denies any path outside the corpus, any read of a file that isn't library source, and any `check_fix` or `report_issue` on a page other than the one under review.

## Step 3: what it finds, misses and costs

### Setup

Model `claude-haiku-5-5`, effort `low`, adaptive thinking, `max_tokens` 8,000. All 18 pages, 5 runs each: 90 runs. Every run made its API calls from this laptop over the public API, three pages at a time, on October 25, 2026; API latency varies with the time of day. Costs come from the lab ledger. Intervals are 95% Wilson confidence intervals; differences between configurations use a paired bootstrap over seeds, because the runs of one seed are not independent ([Part 11](/blog/agent-evals/) explains why).

### Results

| Measure | Result |
|---|---|
| Seeded defects found | 118 of 120 (98.3%, 94.1-99.5%) |
| by kind: link, code, fact | 40/40, 38/40, 40/40 |
| Seeds found in all 5 runs | 23 of 24 |
| Recorded fixes that pass their check | 118 of 118 |
| Real defects already in the docs, reported | `isvalid` links: 5 of 5 runs each; `match()` arguments and build-metadata rule: 0 of 5 each |
| False positives | 0 of 128 reports |
| Cost | $0.1295 total; $0.0014 per page; $0.0010 per real defect found |
| Time per page | median 23.7 s, p95 (95th percentile) 46.2 s |
| Steps per page | median 6, max 13 (budget 30); every run ended on its own |
| Context on the last step | median 5,263 tokens, max 14,293 |

None of the 128 reports was wrong. Each was a seed or one of the two broken `isvalid` links that shipped in the real docs, and every fix was applied and checked, not just read.

Most of the bill was output. Prompt caching served 89.7% of the 2.57 million input tokens the runs read, so input cost about $0.056 of the $0.13 and the 147,239 output tokens, mostly thinking, cost $0.074.

### The two misses were one tool problem

Both missed seeds were C2, an example whose expected output `-1` had been changed to `1`. The agent saw the failure every time; it could not record a fix. The page indents the example by 4 spaces, the model kept copying it with 6, and `report_issue` refused text that wasn't on the page. One run ended with:

```text
- **Not recorded (code, line 15):** `semver.compare("1.0.0", "2.0.0")` returns `-1`, but the page
  expects `1`. The fix is to change the expected output on line 15 from `1` to `-1`. Every attempt
  to match that two-line span failed with "occurs 0 times", so `report_issue` rejected it.
```

This was common: 49 of 170 `check_fix` calls (28.8%) failed on text that didn't match the page, every one of them on whitespace. The verifier caught them before they became bad reports, which is its job, but the agent then had to retry or give up. A second version of the tools accepts a multi-line `old_text` whose lines differ from the page only in leading spaces, and keeps the page's own indentation. Over another 90 runs it found C2 in 5 of 5, cut rejected `check_fix` calls to 22.8%, and found 118 of 120 seeds again: two fact seeds slipped instead.

### What it misses: facts only the code contradicts

The seeded facts were too easy: half sat next to an example that contradicted them. On the hard set, six claims only the source code contradicts, the same configuration found 18 of 30 (60.0%, 42.3-75.4%). It never caught H1, the claim that missing dictionary parts become `None`; every run read the comparison code and said the page had no defects; one concluded that "the dict case behaves as described." It caught the versionadded number (H6) and the dictionary keys rule (H2) in only 2 of 5 runs each.

The third real defect in the shipped docs, the page saying `match()` takes two arguments, shows the same weakness from another angle. On the seeded corpus that page also carries two seeds that `check_links` flags at once. The agent fixed those and reported the `match()` sentence in 0 of 5 runs. On the clean copy of the same page, with nothing for the tools to flag, it reported it in 2 of 3 runs. When the agent ran without the checker tools at all, it reported it in 4 of 5. The numbers are small, but they point one way: when the tools hand a low-effort agent something to fix, it fixes that and stops reading. Anthropic's effort docs warn about exactly this for Haiku 5.5: at `low`, "the model is more likely to skip a search, stop early, or skip a check."

### Real defects on the clean pages

On the clean pages (3 runs of each of the 18), the agent made 11 reports. Two were false positives. One argued that `VersionInfo` isn't really deprecated because no warning is raised, a reading of the code but not a defect in the page. The other was junk: a "finding" whose problem read "No defect found; placeholder not reported" with `old_text` equal to `new_text`, which a one-line harness check would reject.

The other nine were real, and one was new. The page on raising versions says that if "the last part of the dot-separated build metadata is not numeric, ``.0`` is appended." The agent pointed out that `_increment_string` increments the last run of digits anywhere in the string and appends `.0` only when there is none. It is right: `3.4.5+1abc` bumps to `3.4.5+2abc`, with no `.0`. The pre-run audit for this article had missed it.

## Step 4: what each piece was worth

The same pages ran under four more configurations and a no-model baseline: 612 agent runs in all, for $2.33. Seeded results are per seed per run; "real defects" counts how often each of the four defects already in the docs was reported, per run of its page, on the clean copy.

| Configuration | Seeded defects found | Hard facts found | Real defects, clean pages | False reports | Cost per seeded page |
|---|---|---|---|---|---|
| Checkers only, no model | 16 of 24 flagged, 12 fixed by rule | 0 of 6 | 2 of 4 (the links) | 0 | $0 |
| Haiku `low`, no checker tools | 104/120 (86.7%) | not run | not run | 6 | $0.0017 |
| Haiku `low` (Step 3) | 118/120 (98.3%) | 18/30 (60.0%) | 9/12 | 0 seeded, 2 clean | $0.0014 |
| Haiku `low`, lenient text matching | 118/120 (98.3%) | not run | not run | 0 | $0.0013 |
| Haiku `medium` | 72/72 (100%) | 24/30 (80.0%) | 11/12 | 3 seeded, 1 clean | $0.0018 |
| Sonnet 5.5 `low` | 72/72 (100%) | 26/30 (86.7%) | 6/8 | 0 | $0.0129 |

Haiku `low` ran 5 repeats on the seeded and hard pages and 3 on the clean ones; `medium` ran 3, 5 and 3; Sonnet 3, 5 and 2.

### The checkers are most of the job

With no model at all, the two scripts flagged all 16 link and code seeds and both broken `isvalid` links. A simple rule fixed 12 of the seeds: use the closest existing target for a link, and replace an example's expected output with what the code printed. The rule failed where judgment was needed. For the three examples that call a renamed method, it either pasted the `AttributeError` in as the new expected output or had nothing to offer; the agent changed the call back to the real method name, every time. And no script touches a single fact.

Taking the scripts away from the agent cost 11.7 points of recall (95% interval 3.3 to 21.7 points), all of it on links and examples: it found 77.5% of the broken links and 82.5% of the broken examples by reading. It missed `sec.convert.version` against `sec.convert.versions` in 5 of 5 runs, the kind of one-letter difference a human eye skips too. It also made six false reports, against none with the scripts, including one that pinned a correct link bug on the wrong line.

### Effort and model buy the hard facts

On the hard set, `medium` effort found 80.0% (62.7 to 90.5%) against `low`'s 60.0%, a gain of 20 points (6.7 to 36.7) for 1.5 times the cost per page, $0.0019 against $0.0012. Sonnet 5.5 at `low` found 86.7% (70.3 to 94.7%), 26.7 points above Haiku `low` (6.7 to 46.7) at 13 times the cost, $0.0164 per page. Haiku `medium` and Sonnet are within noise of each other on these 30 trials. Both still missed H1, the dictionary claim, in 4 of 5 runs.

Neither fixed the distraction from Step 3. On the seeded `match()` page, where the tools flag two other defects, the real argument-count error went unreported in all 16 runs across every configuration with the checkers: Haiku `low` and lenient, `medium`, and Sonnet. On the clean copy of the page, `medium` reported it in 3 of 3 runs and Sonnet in 2 of 2.

Sonnet also behaved differently with the tools. It called `check_fix` 6 times in 54 runs, where Haiku called it 170 times in 90, and reported fixes without verifying them, against the prompt's instructions. Its fixes were right anyway: 71 of 72 passed. A stronger model leaned less on the verifier, so the harness, not the prompt, has to decide whether an unverified fix is acceptable.

### The safety and reliability pieces never fired

No run crashed, hit its 30-step or cost budget, or reported the same finding twice, so the checkpoints, budgets and idempotency key were never used. The most expensive Haiku page cost $0.0049 and the longest took 14 steps. The policy denied 1 of 4,404 tool calls, a `read_source` on `docs/conf.py`. On a job this short and this read-only, that is the expected result: these pieces are insurance, and [Part 12](/blog/agent-reliability/) measured what happens without them when a run is long and its tools have side effects.

## In the real world

The mechanical half of this job has had tools for years. Sphinx's `doctest` builder runs marked code blocks as tests, and python-semver already runs its page examples under pytest, so a seeded code defect would fail its CI before release. Sphinx's `linkcheck` builder checks only external links; for internal cross-references there is `nitpicky` mode, in which "Sphinx will warn about all references where the target cannot be found." python-semver's configuration doesn't enable it, which is how two references to the removed `isvalid` method reached the published docs.

Facts are the part no build step checks. DocRef went after them in 2013 with natural-language parsing plus code analysis, and found more than 1,000 errors in five libraries. The agent here does the same comparison by reading, at about a tenth of a cent per page.

The design follows Anthropic's advice in "Building effective agents": "finding the simplest solution possible, and only increasing complexity when needed." Two of the three checks are plain scripts, the per-page runs are a fixed workflow, and the model only does what scripts can't. The same post recommends "extensive testing in sandboxed environments," which is what Steps 3 and 4 did.

## The series in one table

Each part of this series measured something. This is the one result from each that shaped the agent above.

| Part | The result that mattered | Where it shows up here |
|---|---|---|
| [1: the loop](/blog/what-is-an-ai-agent/) | Pasting a small docs folder into one prompt matched the agent (94% vs 95%) for less money | Links and examples are checked by plain scripts; the model only gets the reading |
| [2: tool calling](/blog/tool-calling-internals/) | Strict mode took schema-invalid calls from 5.8% to none; better descriptions took tool choice from 88% to 100% | Strict schemas, so no call failed validation; the failures were in text the model copied |
| [3: reasoning](/blog/react-planning/) | Thinking at the lowest effort raised accuracy from 77% to 88% for 44% more cost | Adaptive thinking at `low`; `medium` lifted hard facts from 60% to 80% for 1.5 times the cost |
| [4: self-correction](/blog/agent-self-correction/) | Test feedback fixed every error its tests caught, and nothing else | `check_fix` rejected 28.8% of attempted fixes before they were reported, and it can't check facts |
| [5: context](/blog/context-engineering/) | Small tool results cut cost per task 5.5 times; caching made the baseline 5 times cheaper | One page per run, line-range reads: median last-step context 5,263 tokens, 89.7% of input from cache |
| [6: MCP](/blog/model-context-protocol/) | Tool choice stayed right with 145 tools loaded; tool search added 1.8 s and its own failures | Seven tools, all loaded, no tool search |
| [7: multi-agent](/blog/multi-agent-systems/) | The team read 1.15 times the tokens and was cheaper only because each request stayed under Haiku's 100,000-token price line | No orchestrator: pages are independent, so three plain runs in parallel |
| [8: coding agents](/blog/coding-agents/) | String-replace edits resolved 83 of 84 runs; whole-file writes 66 at 7.5 times the cost | Every fix is an exact replacement; the agent never rewrites a page |
| [9: computer use](/blog/computer-use-agents/) | Page text alone did 95 of 100 tasks at 22% lower cost than screenshots | The agent reads reStructuredText source, never the rendered site |
| [10: research agents](/blog/research-agents/) | 23 of 60 quoted snippets didn't contain the fact they were attached to | Every finding names a line, and link and code fixes are re-run before they count |
| [11: evals](/blog/agent-evals/) | pass@1 was 85.0% but only 71.4% of tasks passed all 10 runs | 5 runs per page; the hard set's per-seed results ranged from 0 of 5 to 5 of 5 |
| [12: reliability](/blog/agent-reliability/) | Retries without idempotency keys raised double refunds from 63% to 93%; a key brought them to 0 of 30 | Checkpoints, budgets and an idempotent `report_issue`, never triggered in these runs |
| [13: sandboxing](/blog/agent-sandboxing/) | Bubblewrap added about 5 ms per command; a sandboxed policy interrupted 5.4% of calls, not 55.6% | Model-written doctests run in bubblewrap; no tool needs a human, so nothing interrupts |
| [14: training](/blog/training-agentic-models/) | Only 1 of Part 8's 28 tasks would teach a reinforcement learner anything | The easy seeds were at ceiling and separated nothing; the hard set did |
| [15: cost and latency](/blog/agent-cost-latency/) | Caching cut cost per completed task by 37% and the Batch API halved it, with no accuracy cost; turning thinking off was the only lever that cost accuracy | Cost is reported per page and per real defect found; caching served 89.7% of input, and thinking stays on |

## Common misconceptions

- **"A 98% score on seeded defects means the agent is 98% reliable."** It means the seeds were easy. Half the seeded facts sat next to an example that contradicted them. On six facts only the code contradicts, the same agent found 60%. A benchmark at its ceiling can't tell two configurations apart; you need a set it fails.
- **"The model should do the checking."** Without the link checker and the example runner, the agent found 77.5% of the broken links and 82.5% of the broken examples instead of 100% and 95%, and made six false reports instead of none. A script that resolves every reference doesn't skim; give it the mechanical checks and spend the model on reading.
- **"More tools means more coverage."** The tools changed where the agent looked. On the page where `check_links` flagged two seeds, the low-effort agent fixed them and reported the real `match()` error in 0 of 5 runs; without the checkers it reported it in 4 of 5.
- **"A run that reports nothing has checked everything."** On the clean copy of the docs, the agent missed the real build-metadata error in 2 of 3 runs of that page and said the page had no defects. A quiet run is a claim like any other: sample those runs and check them too.
- **"A stronger model makes the harness less important."** Sonnet 5.5 found more of the hard facts, but it also skipped the verifier: 6 `check_fix` calls in 54 runs, against the prompt's instructions. Its fixes happened to be right. The guarantees that hold whatever the model does, such as no write access, a sandbox and a check before a patch is applied, live in the harness.

## Key takeaways

- A useful agent is mostly harness. Here, scripts did the link and example checks, a sandbox ran the model's code, and the model did the one thing scripts can't: read a sentence against the code and write the fix.
- Built that way, Claude Haiku 5.5 found 98.3% of 120 seeded defects with no false reports, for about a tenth of a cent per page, and found a real error in the shipped docs that a human audit missed.
- Seeded defects that are too easy tell you nothing. On facts only the code contradicts, the same agent found 60%; `medium` effort raised that to 80% for 1.5 times the cost, close to Sonnet 5.5's 87% at 13 times the cost.
- Watch where the agent stops. When the tools handed it defects on a page, it fixed those and missed the real error in the prose, in all 16 runs with the checkers turned on.
- Measure the failures you can fix in the harness. Text matching, not reasoning, caused both misses in the main run, and a no-op report and an unverified fix are each one line of harness code away from being rejected.

## What's next

This is the last part of Agentic AI. If you want to know what happens inside the model that made these tool calls, start with [LLM Fundamentals](/blog/how-llms-work/); for serving it fast and cheaply, [LLM Inference](/blog/how-llm-inference-works/); for training one yourself, [Build an LLM From Scratch](/blog/build-llm-from-scratch/); and for building systems around it, [System Design: Building Tool-Using Agents That Work Reliably](/blog/reliable-tool-using-agents/). A later series, Measuring LLMs, will cover how evals and benchmarks are built, and how far to trust them.

## Further reading

- [Detecting API Documentation Errors (Zhong and Su, OOPSLA 2013)](http://www.cs.ucdavis.edu/~su/publications/oopsla13-docref.pdf): DocRef, which combined language and code analysis and found more than 1,000 errors in five libraries' docs.
- [doctest (Python docs)](https://docs.python.org/3/library/doctest.html): runs interactive examples in text to check they still work as shown.
- [sphinx.ext.doctest](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html), [Builders](https://www.sphinx-doc.org/en/master/usage/builders/index.html) and [nitpicky](https://www.sphinx-doc.org/en/master/usage/configuration.html): the build-time checks for code blocks, external links and internal references.
- [Building effective agents (Anthropic, 2024)](https://www.anthropic.com/engineering/building-effective-agents): workflows versus agents, and starting from the simplest design.
- [Effort (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/effort): effort levels on Claude Haiku 5.5 and Sonnet 5.5, and what `low` gives up.
- [Pricing (Claude API docs)](https://platform.claude.com/docs/en/about-claude/pricing): the per-token prices behind every cost in this article.
- [python-semver](https://github.com/python-semver/python-semver): the library whose docs this agent reviewed.
- [Building Tool-Using Agents That Work Reliably](/blog/reliable-tool-using-agents/): the System Design view of the same harness.
