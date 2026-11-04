---
title: "Long-Context Evals: Needles, RULER and What a Million Tokens Buys"
description: "Why needle tests saturate, how RULER and NoLiMa probe the real context size, and Claude Haiku 5.5 and two Qwen models measured from 500 to 505K tokens."
slug: "long-context-evals"
category: "evaluation-reliability"
tags: ["evals", "benchmarks", "long-context", "caching"]
series: "llm-evals"
seriesOrder: 9
pubDate: 2026-09-23
updatedDate: 2026-11-04
---

## TL;DR

A context window is the most text a model accepts; how much of that text it can use is a separate and usually smaller number. The classic needle-in-a-haystack test, which hides one sentence in filler and asks for it back, saturated in 2024 because its question repeats the needle's words; harder tests such as RULER, NoLiMa and HELMET find much shorter effective lengths. On Claude Haiku 5.5, which accepts 1M tokens, a literal needle was found 88% of the time at 62K tokens with thinking off, but the same needle asked with one inference was answered 8% of the time from 8K on, and five-step variable chains 0 to 17% of the time. Adaptive thinking lifted those to about 50% and 83 to 100%. Two Qwen models and the cost of every length were measured too.

## Why it matters

Hide six short sentences in 8,000 tokens of Jane Austen. Ask Claude Haiku 5.5 "Which character lives next to the Atomium?" and it names the right character 88% of the time. Ask "Which character has been to Brussels?", where the answer is the same sentence plus one fact every model knows (the Atomium is in Brussels), and it gets 1 of 24 right.

It does find the sentence. Many replies quote it: "The snippet does not say which character has been to Agra. It only mentions that 'Farid lives next to the Taj Mahal,' which is an irrelevant," cut off there by the token limit.

Llama 2 shipped in 2023 with a 4,096-token window; Claude Haiku 5.5 accepts a million. [Inference Part 11](/blog/long-context-inference/) covered what such windows cost to serve. This part is about what they're worth: how long-context benchmarks are built, why the famous one stopped telling models apart, and what a test must do to find where a model stops using its context.

## The intuition

Think of pressing Ctrl+F versus reading. If the question contains the word you need, you search: type "Atomium", jump to the match, read the name. Length barely matters, because the search skips the pages in between.

"Who has been to Brussels?" defeats the search box, because nothing on the page says Brussels. You have to read every sentence, notice that the Atomium is in Brussels, and connect the two, and that gets harder with every page. A literal needle test is a Ctrl+F test, and most long-context evaluation since 2024 has gone into building tests that searching can't pass.

The analogy breaks in one place. A model doesn't search and then read; attention does both at once, scoring every earlier token against the current one. Shared words make the right token score high. Without them the match has to come from meaning, a weaker signal that a long context drowns out more easily.

## Step 1: the needle test, and why it saturated

### Kamradt's 2023 test

In November 2023, Greg Kamradt [pressure-tested GPT-4-128K and Claude 2.1](https://github.com/gkamradt/LLMTest_NeedleInAHaystack): "Place a random fact or statement (the 'needle') in the middle of a long context window (the 'haystack')", ask the model to retrieve it, and "Iterate over various document depths (where the needle is placed) and context lengths". The haystack was Paul Graham's essays; the needle was "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day." The result was a heatmap of length against **depth**, how far into the context the needle sits (0% is the start, 100% the end).

Two weeks later, Anthropic [replied](https://www.anthropic.com/news/claude-2-1-prompting) that "Claude 2.1 is much more reluctant to answer when a sentence seems out of place in a longer context, and is more likely to claim it cannot answer based on the context given." Adding one sentence to the start of the response, "Here is the most relevant sentence in the context:", was "enough to raise Claude 2.1's score from 27% to 98% on the original evaluation." A 71-point swing from one sentence is the harness sensitivity of [Part 4](/blog/eval-harness-sensitivity/) at long-context scale.

### Why it stopped telling models apart

By February 2024, Google's [Gemini 1.5 report](https://arxiv.org/abs/2403.05530) claimed "near-perfect 'needle' recall (>99.7%) up to 1M tokens," and 99.2% at 10M. The test had saturated: in [Part 1](/blog/why-llm-evals-are-hard/)'s sense, top models all score near the ceiling, so it no longer ranks them.

The reason is lexical overlap. The NoLiMa authors measured it as ROUGE-1 precision, the share of the question's words that also appear in the needle: 0.905 for Kamradt's test. The question practically quotes the needle, so a model can find it the way a search engine would.

### Lost in the middle

[Liu et al. (2023)](https://arxiv.org/abs/2307.03172) moved the one document that answers a question among 20 Wikipedia passages (about 4,000 tokens). GPT-3.5-Turbo scored 75.8% with it first, 53.8% tenth and 63.2% last; with no documents at all, 56.1%. In the middle, the documents didn't help. This U-shape is called **lost in the middle**.

## Step 2: tests that can't be passed by searching

### RULER: more needles, chains and counts

[RULER](https://arxiv.org/abs/2404.06654) (Hsieh et al., NVIDIA, 2024) turned one test into 13, in four categories: "retrieval, multi-hop tracing, aggregation, and question answering." The retrieval tasks add distractors. In multi-key needles, several needles are inserted and "only one of them needs to be retrieved. The additional 'needles' are hard distractors." Multi-value and multi-query variants ask for several values at once.

**Variable tracking** is the multi-hop task: "a variable X1 is initialized with a value V, followed by a linear chain of variable name binding statements (e.g., X2 = X1, X3 = X2, ...), which are inserted at various positions of the input," and the model must return every name that holds V. The aggregation tasks count the most common words in a long list, a stand-in for summarization.

RULER generates 500 examples per length from 4K to 128K and defines a model's **effective context length** as the longest length at which it beats Llama 2 7B's score at 4K, 85.6%. Of 17 models that "all claim context sizes of 32K tokens or greater, only half of them can maintain satisfactory performance at the length of 32K." Inference Part 11 quoted the paper's table; the [GitHub leaderboard](https://github.com/NVIDIA/RULER), read on November 4, 2026 (last updated October 2025), includes:

| Model | Claimed context | Effective context on RULER |
|---|---|---|
| Qwen3-235B-A22B (reported by Qwen) | 128K | over 128K |
| GPT-4-1106-preview | 128K | 64K |
| Llama 3.1 70B | 128K | 64K |
| GLM4 9B | 1M | 64K |
| Mistral-Nemo | 128K | 16K |

### NoLiMa: needles with no words in common

[NoLiMa](https://arxiv.org/abs/2502.05167) (Modarressi et al., ICML 2025; the name is short for "no literal matching") removes the overlap. Its example needle is "Actually, Yuki lives next to the Semper Opera House." The question is "Which character has been to Dresden?" To answer, a model has to find the needle and know that the Semper Opera House is in Dresden: one hop of world knowledge. Its ROUGE-1 precision is 0.069, against 0.905 for Kamradt's needle.

The authors placed 58 question-needle pairs at 26 depths in book text and define effective length as the longest length still above 85% of the model's own short-context score. Of 13 models claiming at least 128K tokens, "at 32K, for instance, 11 models drop below 50% of their strong short-length baselines." GPT-4o fell "from an almost-perfect baseline of 99.3% to 69.7%," an effective length of 8K; Claude 3.5 Sonnet's was 4K. On Llama 3.3 70B at 32K, literal questions scored 98.5% and one-hop questions 56.2% (reported by the authors, not reproduced here).

## Step 3: building our own test

![Flowchart. Filler text from six Austen novels, six character needles such as "Omar lives next to the Atomium", and three variable chains such as "VAR A = 48213, VAR B = VAR A" combine into a haystack of a set length with needles at set depths. The haystack becomes a cached prefix, written once and read 14 times. Three kinds of question read it: literal (who lives next to the Atomium?), one-hop (who has been to Brussels?) and tracking (which variables hold 48213?). Every answer is graded and scored by length and depth.](/diagrams/e09-haystack-test.svg "One haystack, three kinds of question, one cached prefix.")

Each haystack carries all three tasks, so one cached copy of the text serves 15 questions.

### The haystack and the needles

The filler is six Jane Austen novels from Project Gutenberg (public domain in the US, and not redistributed here), 4.05 million characters or about 1.32 million Claude tokens. Each haystack gets six character needles in NoLiMa's form ("Actually, Omar lives next to the Atomium."), drawn from 20 names and 20 landmark-city pairs, and three variable chains in RULER's form, each a value and four hops, scattered at random depths. A script checked that no name, landmark or city occurs in the novels; Dresden does, once, so the Semper Opera House was swapped for Helsinki's Kiasma museum.

Each needle gets a literal question ("Which character lives next to the Atomium?") and a one-hop one ("Which character has been to Brussels?"); each chain gets one ("Find all variables that are assigned the value 48213..."). Five other needles of the same form act as distractors for every question, as in RULER's multi-key task. The prompt is NoLiMa's template, which asks for answers "based on either explicitly stated facts or strong, logical inferences."

| Setting | Value |
|---|---|
| Lengths (mean prompt tokens) | needles only (494), 3,442, 7,500, 15,758, 31,775, 62,447, 96,028, 199,453, 505,317 |
| Haystacks per length | 4 up to 96K (depths spread over 24 evenly spaced points), 1 at 200K and 500K |
| Items per length | 24 literal, 24 one-hop, 12 chains (200K: 6, 6, 3; 500K: 3, 6, 2) |
| Model | `claude-haiku-5-5`, thinking off (`{"type": "disabled"}`), default effort |
| Output limit | 48 tokens for names, 256 for chains |
| Controls | the needles alone, no filler; and "In which city is the Atomium?" for all 20 landmarks |

Haiku named all 20 cities correctly, so the one-hop fact is something the model knows.

### Caching makes it affordable

Haiku 5.5 [costs](https://platform.claude.com/docs/en/about-claude/pricing) $0.10 per million input tokens for prompts up to 100,000 tokens and $0.50 above, "even when part of its prompt is a cache hit." The [Batch API](https://platform.claude.com/docs/en/build-with-claude/batch-processing) halves both. Cache reads cost a tenth of the input price; a 1-hour cache write costs twice it.

Every question for a haystack shares the same long prefix, so the first request writes it to the cache and the other 14 read it. The batch docs warn that "cache hits are provided on a best-effort basis" in batches, with "hit rates ranging from 30 percent to 98 percent." Sending one primer request per haystack first, then the other 14 once the primer's batch had finished, gave a hit on all 336 follow-ups. The 360 requests at 3K to 96K tokens read 13.0 million prompt tokens for $0.15; without caching they'd have cost $0.65.

Above 100K tokens the requests ran one at a time, where a cache hit is reliable. In a batch, every request that missed the cache would pay for its own 500K-token write, and this project's spending guard has to assume that worst case. The first 200K request cost $0.125 (the cache write) and each later one $0.010. At 500K: $0.316, then $0.025 each, 6 to 30 seconds per request. Those two haystacks took $0.84 of the $1.09 spent on Haiku in this part.

### Validating the grader

A **lenient** grader, NoLiMa's rule for answers without chain of thought, accepts any reply that contains the right name. A **strict** grader also rejects replies that deny or hedge ("does not say", "no character", "cannot"). On 60 hand-checked replies (40 one-hop, 20 literal), the strict grader agreed 59 times and the lenient one 47 times: it credited 18 one-hop replies where the hand check found 8, because a denial that quotes the needle still contains the name. The numbers below are strict.

Replies that hit the token limit were graded on the text they contain: 51% of one-hop replies (the model explains its refusal at length), 7% of literal ones and 7% of chains. A cut-off denial is still a denial.

The prompt mattered before any of this. In a pilot at 16K tokens, three plainer wordings ("Which character's home is in Singapore?" among them) got 0 of 11 one-hop answers, every one a refusal such as "I can't find a character whose home is in Singapore in this text." NoLiMa's template, with its line about "strong, logical inferences", got 2 of 3 on the same haystack.

## Step 4: where Haiku 5.5 falls off

![Line chart of the share of questions Claude Haiku 5.5 answered correctly with thinking off, against prompt length on a log scale from needles only (494 tokens) to 505K tokens, with 95% intervals. Literal needles: 100% with needles only, then 88 to 92% from 3K to 62K, 75% at 96K, 6 of 6 at 200K and 2 of 3 at 500K. One-hop needles: 96% with needles only, 46% at 3K, then 4 to 8% from 8K to 96K and 0 of 6 at 200K and 500K. Variable tracking: 92% with needles only, then 0 to 17% at every length from 3K on. A dashed line at 100K marks where the long-prompt price starts.](/diagrams/e09-length-curves.svg "Claude Haiku 5.5, thinking off: three tasks on the same haystacks.")

Only the literal task stays above 85% of its needles-only score once filler is added; the other two drop by half or more at the first length tested.

### Three tasks, three effective lengths

With the needles alone, Haiku scored 100%, 96% and 92% on the three tasks. Adding filler changed them very differently:

| Prompt tokens | Literal | One-hop | Variable tracking |
|---|---|---|---|
| 494 (needles only) | 24/24, 100% (86 to 100) | 23/24, 96% (80 to 99) | 11/12, 92% (65 to 99) |
| 3,442 | 22/24, 92% (74 to 98) | 11/24, 46% (28 to 65) | 1/12, 8% (1 to 35) |
| 7,500 | 21/24, 88% (69 to 96) | 1/24, 4% (1 to 20) | 2/12, 17% (5 to 45) |
| 31,775 | 22/24, 92% (74 to 98) | 2/24, 8% (2 to 26) | 0/12, 0% (0 to 24) |
| 96,028 | 18/24, 75% (55 to 88) | 2/24, 8% (2 to 26) | 0/12, 0% (0 to 24) |
| 505,317 | 2/3 | 0/6 | 0/2 |

Brackets are 95% Wilson intervals ([Part 3](/blog/eval-statistics/)). Using NoLiMa's rule, the longest length still above 85% of the needles-only score, the literal task's effective length is about 62K tokens, the 64K target. One-hop and variable tracking have none: both fail at the shortest haystack tested, 3.4K tokens, out of a 1M window.

The one-hop gap is the cleanest number here, because each pair of questions shares a needle and a haystack. Over 144 needles from 3K to 96K, the literal question was right 86.8% of the time and the one-hop question 13.9%, a paired difference of 72.9 points (95% CI 65.0 to 80.8, with standard errors clustered by haystack). In 105 pairs only the literal question was right; in none was only the one-hop one.

### The failures have a shape

Most one-hop failures aren't misses. With filler, 84% of one-hop replies denied or hedged, 35% of the wrong ones named the right character anyway, and 27% of all of them called the needle out of place, with words such as "inserted", "unrelated", "irrelevant" or "stray". With the needles alone, the same sentences were the whole text and Haiku answered 23 of 24. Inside Austen's prose it recognized them as planted, the reaction Anthropic described for Claude 2.1 in 2023. A test meant to measure retrieval ends up measuring whether the model will vouch for a sentence it suspects was planted.

Variable tracking almost always failed the same way: 62 of the 72 wrong answers listed only the first variable, the one assigned the number directly. Literal errors were mostly distractor mix-ups: of 20 wrong literal answers, 11 named a different needle's character, 7 denied the needle existed and 2 named an Austen character. One needle that landed inside a line of dialogue was attributed to the speaker: "Miss Steele's statement about living next to the Christ the Redeemer statue".

### Position still matters

Pooling the literal questions from 3K to 96K, needles in the first or last fifth of the context were found 95% of the time (58 of 61) and needles in the middle three fifths 81% (67 of 83): a 14-point gap (95% CI 7 to 22, bootstrapping whole haystacks). That's the lost-in-the-middle shape, smaller than Liu et al.'s but there. One-hop scores were too low everywhere to show it.

### Thinking changes the picture

![Grouped bar chart of Claude Haiku 5.5 on the same requests with thinking off (blue) and adaptive thinking (orange), with 95% intervals. One-hop needles, 24 per length: 4% to 46% at 8K, 8% to 50% at 32K, 8% to 58% at 96K. Variable tracking, 12 chains per length: 17% to 100% at 8K, 0% to 83% at 32K, 0% to 100% at 96K.](/diagrams/e09-thinking.svg "Same haystacks, same questions: thinking off vs adaptive thinking.")

Every test above ran with thinking off, as most needle benchmarks do. Rerunning the one-hop and tracking questions at 8K, 32K and 96K with adaptive thinking (default effort, medium) changed them more than any length did:

| Task and length | Thinking off | Adaptive thinking | Difference (95% CI) | Mean output tokens with thinking |
|---|---|---|---|---|
| One-hop, 8K | 4% | 46% | +42 points (18 to 65) | 322 |
| One-hop, 96K | 8% | 58% | +50 points (30 to 70) | 358 |
| Tracking, 8K | 17% | 100% | +83 points (61 to 100) | 578 |
| Tracking, 96K | 0% | 100% | +100 points | 775 |

At 32K the gains were similar (one-hop 8% to 50%, tracking 0% to 83%). With a few hundred tokens to reason, Haiku followed every chain at 96K and accepted the one-hop inference about half the time; the rest were still denials ("No character is stated to have been to Yangon."). So "Haiku 5.5's effective context for multi-hop questions is under 4K" holds only with thinking off, and a long-context score needs its thinking setting next to it.

### An open model with a 32K window

The same haystacks, rebuilt to Qwen token counts, ran on [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (both Apache 2.0, "Full 32,768 tokens"), in bf16 on an RTX 4060 Laptop GPU (8 GB) with PyTorch 2.11 and transformers 5.21.1, greedy decoding. Each haystack was prefilled once and its KV cache, the stored keys and values of every earlier token, was copied for each question: prompt caching done by hand.

| Prompt tokens | 1.5B literal | 1.5B one-hop | 0.5B literal | 1.5B KV cache | 1.5B peak GPU memory | 1.5B prefill |
|---|---|---|---|---|---|---|
| 301 (needles only) | 24/24, 100% | 6/24, 25% | 16/24, 67% | 8 MiB | 2.9 GiB | 0.05 s |
| 3,695 | 17/24, 71% | 3/24, 13% | 12/24, 50% | 101 MiB | 3.2 GiB | 0.6 s |
| 16,202 | 21/24, 88% | 3/24, 13% | 14/24, 58% | 443 MiB | 4.3 GiB | 4.8 s |
| 31,631 | 14/24, 58% | 2/24, 8% | 10/24, 42% | 865 MiB | 5.7 GiB | 9.6 s |

Neither Qwen model tracked a single chain at any length, needles-only included (0 of 60 each), so that task is beyond them at any length. The 1.5B knew 19 of 20 landmark cities but answered only 6 of 24 one-hop questions without filler: its one-hop failure is mostly reasoning, not context. Its literal score at 31.6K, 58% (39 to 76), is 42 points under its needles-only score. The cache is 28 KiB per token (28 layers × 2 key-value heads × 128 numbers × keys and values × 2 bytes), yet the peak rose 2.8 GiB above the needles-only baseline at 31.6K: prefill activations cost more than the cache. Both models ran in 5 minutes of GPU time.

## Step 5: natural text, and what it costs

### Suites built from real documents

Synthetic needles are cheap and controllable, but nobody's job is finding planted sentences. [LongBench](https://arxiv.org/abs/2308.14508) (Bai et al., 2023) gathered "21 datasets across 6 task categories in both English and Chinese," from document question answering to summarization and code completion. [LongBench v2](https://arxiv.org/abs/2412.15204) (2024) went longer and harder: "503 challenging multiple-choice questions, with contexts ranging from 8k to 2M words." Human experts scored "only 53.7% accuracy under a 15-minute time constraint"; the best model answering directly, GPT-4o, scored 50.1%, and o1-preview, "which includes longer reasoning," 57.7%, the same kind of gain from reasoning that thinking gave Haiku above.

[HELMET](https://arxiv.org/abs/2410.02694) (Yen et al., ICLR 2025) asked whether synthetic scores predict real ones. It covers seven application categories up to 128K tokens, from long-document question answering to retrieval-augmented generation (RAG) and citations, and grades open-ended answers with a GPT-4o judge checked against people, as in [Part 6](/blog/grading-generative-outputs/). Across 59 models, its first finding is that "synthetic tasks like NIAH do not reliably predict downstream performance." At 128K, "none of the synthetic tasks achieves an average correlation higher than 0.8"; plain needles averaged about 0.68 Spearman correlation with the real tasks, all of RULER about 0.75.

Two more designs sit between synthetic and natural. [BABILong](https://arxiv.org/abs/2406.10149) (NeurIPS 2024) hides short reasoning tasks in long stretches of book text and reports that "popular LLMs effectively utilize only 10-20% of the context." [Michelangelo](https://arxiv.org/abs/2409.12640) (Google DeepMind, 2024) builds tasks that require a model to "chisel away" the irrelevant text, such as tracking a Python list through a long series of operations, and found that "all models experience significant fall off in performance before 32K."

### What a long-context eval costs

Cost grows with length, while the number of items needed for a tight interval stays the same. Liu et al. noted that running their experiments on GPT-4 "would cost upwards of $6000," and NoLiMa cut its needle placements "from the default 26 to 11" for its longest runs "due to cost limitations."

Here, on one of the cheapest models available and with a cache hit on every question but the first, a question cost about $0.00004 at 8K (in a batch), $0.010 at 200K and $0.025 at 500K (one at a time), plus $0.125 and $0.316 for each long haystack's cache write. The 500K points have 2 to 6 items each and intervals 39 to 73 points wide; four haystacks per length, as below 100K, would have cost about $2.90 more. That's why results above 128K often rest on a handful of items.

## In the real world

Reported long-context numbers come from tests like the ones above: needle heatmaps, RULER-style synthetic tasks, and natural-text suites such as LongBench v2. Before comparing two of them, check the conditions, which matter as much as the numbers: thinking or not, how many distractors, literal or inferred questions, and how many items per length.

For your own system, the test that counts is your documents at your lengths. Take real long inputs, pick facts whose answers you know, phrase questions the way users would (rarely in the document's own words), and run them at the thinking setting you'll ship, by depth as well as length. If you use the cache eviction or compression from Inference Part 11, run it with and without. Prompt caching makes this cheap when many questions share one document: here it cut the sub-100K grid's cost by three quarters.

## Common misconceptions

- **"A model with a 1M-token window can use 1M tokens."** Haiku 5.5 accepts 1M tokens, but with thinking off its effective length by NoLiMa's rule was under 3.4K on one-hop questions and about 62K on literal needles.
- **"A perfect needle-in-a-haystack heatmap means good long-context ability."** The classic question shares 90% of its words with the needle. HELMET found needle scores correlate about 0.68 with real tasks, and here the same needles went from 87% to 14% when the question needed one inference.
- **"Failures at long length mean the model couldn't find the fact."** Here 35% of failed one-hop replies named the right character. Claude 2.1 in 2023 and Haiku 5.5 here treated planted sentences as suspicious and declined to use them, so the grader and the prompt decide part of the score.
- **"Long-context scores are a property of the model."** Turning on adaptive thinking moved variable tracking at 96K from 0% to 100%. Thinking, prompt template, distractors and grader all belong next to the number.
- **"Bigger windows made retrieval obsolete."** Even literal lookups dropped to 75% at 96K with five distractors, against 92% at 3K, and a question at 500K cost over 600 times one at 8K.

## Key takeaways

- A context window is what a model accepts. Its effective context, the longest input it still handles well, depends on the task and is often far shorter.
- Literal needle tests saturated because the question repeats the needle. On Haiku 5.5 a literal needle held at about 88% to 62K tokens, while the same needle asked with one inference dropped to 8% from 8K on.
- RULER's chains and NoLiMa's non-literal needles expose the gap. Without thinking, Haiku listed only the first variable of a chain in 62 of 72 failures; with adaptive thinking it tracked every chain at 96K.
- Report long-context scores with their conditions: thinking setting, prompt template, distractors, grader, and items per length. A strict grader and a lenient one differed on 11 of 40 one-hop replies.
- Long-context evals get expensive fast. Cache the shared document, spend most of the budget below the price threshold, and treat a few items at 500K as a spot check, not a measurement.

## What's next

Everything in this series so far has measured what a model can do. Part 10 turns to what labs most want to know a model can't do: dangerous-capability evals, refusal and over-refusal benchmarks, red-teaming, and why "the model refused" isn't the same as "the model can't."

## Further reading

- [RULER: What's the Real Context Size of Your Long-Context Language Models? (Hsieh et al., 2024)](https://arxiv.org/abs/2404.06654) and its [leaderboard](https://github.com/NVIDIA/RULER): 13 synthetic tasks and the effective-length definition.
- [NoLiMa: Long-Context Evaluation Beyond Literal Matching (Modarressi et al., 2025)](https://arxiv.org/abs/2502.05167): needles that share no words with the question, and the prompt template used here.
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172): the U-shaped position effect.
- [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly (Yen et al., 2024)](https://arxiv.org/abs/2410.02694): seven real-task categories, and why synthetic scores don't predict them.
- [LongBench (Bai et al., 2023)](https://arxiv.org/abs/2308.14508) and [LongBench v2 (Bai et al., 2024)](https://arxiv.org/abs/2412.15204): bilingual natural-text suites, the second with a human expert baseline.
- [BABILong (Kuratov et al., 2024)](https://arxiv.org/abs/2406.10149) and [Michelangelo (Vodrahalli et al., 2024)](https://arxiv.org/abs/2409.12640): reasoning hidden in long text, and latent-structure queries.
- [Gemini 1.5 technical report (Gemini Team, 2024)](https://arxiv.org/abs/2403.05530): the near-perfect needle recall that showed the test had saturated.
- [LLMTest_NeedleInAHaystack (Kamradt, 2023)](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) and [Long context prompting for Claude 2.1 (Anthropic, 2023)](https://www.anthropic.com/news/claude-2-1-prompting): the original test, and the one-sentence prompt change that moved it from 27% to 98%.
- [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing) and [pricing](https://platform.claude.com/docs/en/about-claude/pricing) (Claude API docs): the cache, batch and long-prompt rules behind this part's costs.
- [Long-Context Inference](/blog/long-context-inference/) (Inference Part 11): what long prompts cost to serve, and how sparse attention and cache eviction trade away context.
