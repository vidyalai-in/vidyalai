---
title: "Reasoning Models: Chain of Thought, RL and Test-Time Compute"
description: "How o1 and DeepSeek-R1 learned to think before answering: chain of thought, RL with checkable rewards, GRPO, and what thinking tokens cost."
slug: "reasoning-models"
category: "llm-fundamentals"
tags: ["training", "rlhf", "reasoning", "cost", "latency"]
series: "llm-fundamentals"
seriesOrder: 14
pubDate: 2026-04-09
---

## TL;DR

A reasoning model is an ordinary LLM trained to write a long chain of thought before it answers. The extra tokens give it more computation per question, and reinforcement learning against answers a program can check teaches it to use them: try an approach, notice a mistake, back up. This article goes from "Let's think step by step" prompting through voting and verifiers to the RL recipes behind OpenAI's o1 and DeepSeek-R1, works GRPO's group-relative rewards by hand, and ends with the bill: thousands of hidden tokens per answer at output-token prices.

## Why it matters

On the 2024 American Invitational Mathematics Examination (AIME), a 15-problem contest for top high-school students, GPT-4o solved [12% of the problems](https://openai.com/index/learning-to-reason-with-llms/). OpenAI's o1, announced in September 2024, solved 74% with a single attempt per problem. DeepSeek then showed the same jump with a model anyone could download: its DeepSeek-V3 scored [39.2% on AIME 2024](https://arxiv.org/abs/2501.12948), and DeepSeek-R1, built on the same base model, scored 79.8%.

R1 is no bigger than V3: both have 671 billion parameters, 37 billion active per token. R1 just thinks out loud first. On a set of 2024 competition math problems, R1 wrote an average of 8,793 thinking tokens per problem, while GPT-4o wrote 711 tokens per answer, about 12 times fewer.

How does writing more text make a model smarter? How did training teach it to think for thousands of tokens with no example thoughts to copy? And when is the thinking worth paying for?

## The intuition

Think of a student on a hard exam problem. Asked for the answer straight away, they must do it all in their head, and whatever doesn't fit in working memory is lost. With scratch paper, they can write down intermediate results, try an approach, see it fail, cross it out and try another. They are no cleverer, but they can spend more effort and keep track of it.

Thinking tokens are a model's scratch paper, and reinforcement learning is a teacher who marks only the final answer. Over thousands of graded problems, the model learns which scratch-paper habits pay off, such as checking its work and backtracking.

The analogy breaks in three places. A model can't think silently, so more thinking means more tokens, each costing time and money. What it writes may not be what drove its answer. And the teacher can only grade problems with checkable answers.

## Step 1: chain of thought, on paper and by vote

### Prompting the model to show its work

[Part 1](/blog/how-llms-work/) showed that an LLM writes one token per trip through the network, and each trip does the same fixed amount of work whatever the question. A model that answers straight away has only the trips for those few answer tokens to work it out.

In 2022, [Wei et al.](https://arxiv.org/abs/2201.11903) showed that models do better when the prompt's examples include worked steps. Their prompt had eight example questions, each answered like this:

```text
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
   5 + 6 = 11. The answer is 11.
```

They called this chain-of-thought prompting. On GSM8K, a benchmark of grade-school math word problems, it took Google's 540-billion-parameter PaLM from 17.9% to 56.9%. The paper calls it "an emergent ability of model scale": gains appeared at around 100 billion parameters, and smaller models wrote fluent but illogical chains.

Months later, [Kojima et al.](https://arxiv.org/abs/2205.11916) dropped the examples. Adding the one sentence "Let's think step by step" before the answer raised OpenAI's text-davinci-002 from 17.7% to 78.7% on the MultiArith benchmark and from 10.4% to 40.7% on GSM8K.

### Why written steps help

Every intermediate result the model writes joins the context, where attention can read it back on the next trip. A chain of thought turns one long calculation into many short ones in series, each building on the last.

Extra tokens alone don't do it. Wei et al. had the model write a row of dots, as many as the characters in the needed equation, before answering. It scored about the same as answering directly. The gain comes from computing something at each step and writing it where later steps can use it.

### Self-consistency: many chains, one vote

Sampled at a temperature above zero ([Part 9](/blog/encoders-decoders-gpt/) covers sampling), a model writes a different chain every time. [Wang et al.](https://arxiv.org/abs/2203.11171) sampled many chains per question and took a majority vote on the final answers, a method they called self-consistency. With 40 sampled chains, PaLM 540B went from 56.5% on GSM8K with a single greedy chain to 74.4%.

How much can voting help? Assume each chain is right with probability 0.6, samples are independent, and, pessimistically, all wrong answers agree, so the vote is right versus wrong. The chance that more than half the votes are right, computed in Python:

| Votes | Right 60% of the time alone | Right 40% of the time alone | Right 80% of the time alone |
|---|---|---|---|
| 1 | 60.0% | 40.0% | 80.0% |
| 5 | 68.3% | 31.7% | 94.2% |
| 15 | 78.7% | 21.3% | 99.6% |
| 41 | 90.3% | 9.7% | 100.0% |

Voting amplifies whatever the model leans toward: more votes push a usually-right model up and lock a usually-wrong one into its error. Real wrong answers tend to scatter, which helps the vote, but samples from one model are rarely independent, which hurts.

The DeepSeek-R1 paper shows the limit: on AIME 2024, a majority vote over 64 samples raised GPT-4o only from 9.3% to 13.4%.

## Step 2: from checking answers to training on them

### Verifiers and best-of-N

A verifier picks the best answer instead of the most common. In best-of-N, you sample N answers, score each with a [reward model](/blog/base-model-to-assistant/) trained to judge correctness, and keep the top one. OpenAI's o1 post shows how far that goes: o1 averaged 74% on AIME 2024 with one sample per problem, 83% with a consensus vote over 64 samples, and 93% when "re-ranking 1000 samples with a learned scoring function."

[Lightman et al.](https://arxiv.org/abs/2305.20050) at OpenAI compared two kinds of verifier on the MATH competition benchmark. An outcome reward model (ORM) sees a whole solution and predicts whether the final answer is right. A process reward model (PRM) scores every step. To train one, labelers marked each step of model-written solutions as positive, negative or neutral, producing PRM800K: 800,000 step-level labels across 75,000 solutions to 12,000 problems.

The PRM scores a solution as the product of its steps' probabilities of being correct. A four-step solution with steps rated 0.95, 0.90, 0.95 and 0.95 scores about 0.77. Swap the second step for one rated 0.30, a likely error, and the score falls to 0.26, even if the final answer happens to be right. Choosing the best of 1,860 solutions per problem, on 500 held-out MATH problems:

| Selection method | Problems solved |
|---|---|
| Majority vote | 69.6% |
| Outcome reward model | 72.4% |
| Process reward model | 78.2% |

### The problem with learned judges

An optimizer pushed against a neural reward model finds its blind spots, the [reward hacking](/blog/base-model-to-assistant/) of Part 13. Snell et al. later found a PRM trained on PRM800K "easy to exploit" for their models "via even naïve strategies such as best-of-N sampling."

DeepSeek skipped neural reward models for R1's reasoning training, stating that they, "whether outcome-based or process-based," are "susceptible to reward hacking during large-scale reinforcement learning." For math with a known answer, or code with test cases, a simple program can do the checking. That is reinforcement learning with verifiable rewards (RLVR), the last stage of Tülu 3's recipe in Part 13, scaled up into the main event.

### o1: reasoning that scales with RL and with thinking time

OpenAI didn't publish its method. Its [September 2024 post](https://openai.com/index/learning-to-reason-with-llms/) says o1 was trained with "large-scale reinforcement learning" to use its chain of thought, and that "the performance of o1 consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute)." On its test-time plot, with compute on a log scale, AIME pass@1 accuracy (the share of problems solved with one sampled answer) climbs from about 20% to about 75%.

OpenAI hides the raw chain of thought, citing "user experience, competitive advantage, and the option to pursue the chain of thought monitoring," and shows a model-written summary instead.

### DeepSeek-R1-Zero: reasoning from RL alone

DeepSeek's [R1 paper](https://arxiv.org/abs/2501.12948) gave the first detailed public recipe. Its first model, R1-Zero, started from the pretrained DeepSeek-V3-Base with no supervised fine-tuning. A short template asked for reasoning inside `<think>` tags and the answer inside `<answer>` tags. The reward had two rule-based parts, added with equal weight:

- Accuracy: is the final answer, written in a required format such as a box, equal to the known answer? For code, does it pass the test cases?
- Format: is the reasoning inside the `<think>` and `</think>` tags?

Nothing rewarded any particular way of reasoning. Each training step took 32 questions, sampled 16 answers to each and scored all 512. Over 10,400 steps, R1-Zero's AIME 2024 pass@1 rose from 15.6% to 77.9%, or 86.7% with a 16-sample majority vote, and its answers grew steadily longer.

Words like "wait", "mistake" and "verify" became 5 to 7 times more frequent over training, and "wait" itself was nearly absent until a spike after step 8,000. The paper prints an intermediate output that stops mid-solution to write "Wait, wait. Wait. That's an aha moment I can flag here." Then it restarts its check. The authors called it an aha moment for themselves too.

R1-Zero's reasoning was hard to read and mixed English and Chinese in one chain of thought. Trained only on checkable tasks, it was also weak at writing and open-ended questions.

### DeepSeek-R1: four stages

R1 addressed those problems with a pipeline that started again from V3-Base:

1. Cold start: SFT on thousands of long, readable chain-of-thought examples.
2. Reasoning RL with the rule-based rewards plus a language-consistency reward, the share of chain-of-thought words in the target language.
3. SFT on about 600,000 correct reasoning samples drawn from the model and about 200,000 non-reasoning ones, such as writing.
4. A second RL stage: rule-based rewards for reasoning, reward models for helpfulness and safety.

The preference reward models joined only for the last 400 of the second stage's 1,700 steps, because more training against them "may lead to reward hacking." In one plotted run, the helpfulness reward kept rising while Codeforces accuracy fell. Scores at each stage:

| Stage | AIME 2024 (pass@1) | IF-Eval (instruction following) | AlpacaEval 2.0 (length-controlled win rate) |
|---|---|---|---|
| R1-Zero | 77.9 | 46.6 | 24.7 |
| Dev1: after cold-start SFT | 59.0 | 71.7 | 50.1 |
| Dev2: after reasoning RL | 74.0 | 72.0 | 55.8 |
| Dev3: after rejection sampling and SFT | 78.1 | 78.1 | 62.1 |
| DeepSeek-R1 | 79.8 | 83.3 | 87.6 |

The small cold-start set cost reasoning accuracy at first, and RL won it back. R1's final 79.8% matched OpenAI's o1-1217, at 79.2% in the paper's comparison.

The paper puts R1's training at 147,000 H800 GPU-hours, $294,000 at an assumed rental price of $2 per GPU-hour: 101,000 hours for R1-Zero, 5,000 to create the SFT data and 41,000 for R1. DeepSeek-V3's full training took 2.788 million ([Part 12](/blog/pretraining-scaling-laws/)), so reasoning training added about 5%.

### Distillation: passing the habit down

DeepSeek then fine-tuned smaller open models, with SFT only, on the same 800,000 R1-generated samples. This is distillation: training a small model to imitate a big one's outputs.

| Model | AIME 2024 (pass@1) |
|---|---|
| GPT-4o (May 2024) | 9.3 |
| Claude 3.5 Sonnet (October 2024) | 16.0 |
| R1-Distill-Qwen-1.5B | 28.9 |
| R1-Distill-Qwen-7B | 55.5 |
| R1-Distill-Llama-8B | 50.4 |
| R1-Distill-Qwen-32B | 72.6 |
| R1-Distill-Llama-70B | 70.0 |

A 1.5-billion-parameter model beat both large non-reasoning models. DeepSeek also ran R1-Zero-style RL directly on Qwen2.5-32B-Base for over 10,000 steps and reached 47.0%, against 72.6% for the distilled 32B model. For small models, copying a strong reasoner's thoughts beat discovering them with RL.

## Step 3: GRPO, PPO without the critic

### What PPO needed

[Part 13](/blog/base-model-to-assistant/) described RLHF with PPO (proximal policy optimization), which keeps four models in memory: the policy being trained, a frozen reference copy for the KL penalty, a reward model and a value model. The value model predicts, token by token, how much reward the answer is heading for, so PPO can tell whether an answer did better or worse than expected.

The DeepSeekMath paper ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) names two problems. The value model is "typically another model of comparable size as the policy model," a large memory and compute cost. And when the only reward arrives at the last token, as with a checked answer, a value model accurate at every token is hard to train.

### The group as its own baseline

Group relative policy optimization (GRPO) drops the value model. For each question it samples a group of `G` answers, scores them all, and judges each against its own group:

```text
advantage_i = (reward_i − mean of the group's rewards) / standard deviation of the group's rewards
```

The advantage says how much better than its siblings an answer was. Every token of an above-average answer has its probability pushed up, and every token of a below-average one pushed down. GRPO keeps PPO's clipping, which caps how far one step can move the policy, and the KL penalty against a reference model, which it adds straight to the loss instead of folding it into the reward.

Here is a group of 8 answers to one question, each rewarded 1 if correct and 0 if not, computed in Python with the population standard deviation:

| Correct answers in the group | Mean reward | Std | Advantage of each correct answer | Advantage of each wrong answer |
|---|---|---|---|---|
| 1 of 8 | 0.125 | 0.331 | +2.65 | −0.38 |
| 3 of 8 | 0.375 | 0.484 | +1.29 | −0.77 |
| 7 of 8 | 0.875 | 0.331 | +0.38 | −2.65 |
| 8 of 8 | 1.000 | 0 | 0 | none |
| 0 of 8 | 0.000 | 0 | none | 0 |

On a hard question, the one answer that got it right gets a big push up. On an easy question, the rare wrong answer gets a big push down. When every answer earns the same reward, all advantages are zero and the question teaches nothing, so the useful training questions are the ones the model gets right only sometimes. (The formula divides by zero there; implementations add a small constant to the denominator.)

![A left-to-right loop. A prompt with a checkable answer goes to the policy, which samples a group of 8 answers. A checker scores each answer 1 or 0. The advantage is computed as reward minus the group mean, divided by the group standard deviation. The policy update pushes up the winners and pushes down the losers, while a frozen reference model feeds a KL penalty that limits drift. The updated policy then takes the next batch of prompts.](/diagrams/f14-grpo-loop.svg "One GRPO step with rule-based rewards: no value model, and no learned reward model.")

The group average replaces the value model, and in R1-Zero's setup a checker replaces the reward model, leaving two neural models where PPO needed four.

### How it was used

In DeepSeekMath, GRPO sampled 64 answers per question (KL coefficient 0.04) and raised DeepSeekMath-Instruct 7B from 82.9% to 88.2% on GSM8K and from 46.8% to 51.7% on MATH. R1-Zero used 16 answers per question (KL coefficient 0.001), each up to 32,768 tokens, raised to 65,536 after step 8,200.

## Step 4: spending compute at answer time

### Thinking longer versus training bigger

Test-time compute is the computation spent while answering rather than while training. You can spend it in parallel, by sampling many answers and voting or verifying, or in series, by thinking longer or revising.

[Snell et al.](https://arxiv.org/abs/2408.03314) (2024) studied both with Google's PaLM 2-S* on MATH. The best strategy depended on the question's difficulty, so they picked one per question, which they called compute-optimal scaling. It beat plain best-of-N with about 4 times less test-time compute. Then, at equal total FLOPs, they compared the small model with extra thinking against a model about 14 times larger answering directly. On easy and medium questions, extra thinking often won. On the hardest, it barely helped, and a bigger pretrained model was the better use of compute.

### Budget forcing with "Wait"

[s1](https://arxiv.org/abs/2501.19393) (Muennighoff et al., 2025) showed how little it takes. The authors fine-tuned Qwen2.5-32B-Instruct on 1,000 questions with reasoning traces from Google's Gemini 2.0 Flash Thinking, a 26-minute run on 16 H100 GPUs. Then they controlled thinking length with budget forcing: end the thinking early to cut it short, or block the end-of-thinking token and append "Wait" to extend it, which often makes the model re-check and fix its answer.

On AIME 2024, Qwen2.5-32B-Instruct scored 26.7%; the fine-tuned model scored 50.0% without budget forcing and 56.7% with it. Gains flattened after six forced "Wait"s, and forcing more could trap the model in repetitive loops.

Trained reasoners partly set their own budgets. On the 2024 competition problems from the opening, R1 used fewer than 7,000 thinking tokens on easy problems, more than 18,000 on the hardest, and under 100 for "1 + 1 =?".

### Budgets in the APIs

APIs expose the same control as a setting, and bill the hidden thinking:

- OpenAI's [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) has a `reasoning.effort` parameter whose model-dependent values can include `none`, `minimal`, `low`, `medium`, `high`, `xhigh` and `max`. Reasoning tokens "are not visible via the API," but "they still occupy space in the model's context window and are billed as output tokens." OpenAI suggests reserving at least 25,000 tokens for reasoning and output at first.
- Anthropic's Claude API started with a fixed budget, `budget_tokens`, minimum 1,024. Its [current models](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) use adaptive thinking instead: the model decides how much to think, steered by an [`effort` setting](https://platform.claude.com/docs/en/build-with-claude/effort) from `low` to `max`. Thinking tokens are [billed as output tokens](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost), the full count even when the response shows only a summary.
- Open models do it in the chat template. When Qwen3's thinking reaches a user-set budget, [its report](https://arxiv.org/abs/2505.09388) says the system inserts "Considering the limited time by the user, I have to give the solution based on the thinking directly now." followed by the end-of-thinking tag, and the model answers from what it has.

### What a thought costs

Take one question answered two ways: a direct 500-token answer, or 8,000 thinking tokens (close to R1's average) followed by the same answer. Assume a 1,000-token prompt, 100 tokens per second of decode for one user, and illustrative prices of $2 per million input tokens and $10 per million output tokens:

| Measure | Direct answer | Reasoning answer |
|---|---|---|
| Output tokens billed | 500 | 8,500 |
| Generation time at 100 tokens/s | 5 s | 85 s |
| Wait before the first answer token, if thinking is hidden | under 1 s | about 80 s |
| Cost per request | $0.007 | $0.087 |
| Cost per 1,000 requests | $7 | $87 |

The reasoning answer costs about 12 times more and takes 17 times longer. Each thinking token is a decode step, the memory-bound phase described in [How LLM Inference Works](/blog/how-llm-inference-works/), and each grows the KV cache: for a model shaped like Llama 3.1 8B, at 128 KiB per token, this request's cache goes from about 0.2 GB to about 1.2 GB, leaving room for fewer users per GPU. [Long-context inference](/blog/long-context-inference/) covers that growth, and [speculative decoding](/blog/speculative-decoding/) is one way to write the tokens faster.

[Reducing Time to First Token in a Chat App](/blog/reduce-time-to-first-token/) suggests streaming a status message while hidden reasoning runs, and [Cutting LLM Cost by 70%](/blog/cut-llm-cost-70-percent/) recommends tracking thinking tokens per task and setting budgets.

## In the real world

Most current models put thinking on a switch or dial within one set of weights:

| Model | How thinking is controlled | Source |
|---|---|---|
| Qwen3 | `enable_thinking` on or off, or `/think` and `/no_think` per message | [Model card](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| gpt-oss | Reasoning effort low, medium or high | [Model card](https://huggingface.co/openai/gpt-oss-120b) |
| DeepSeek-V4 | Three modes: Non-think, Think High, Think Max | [Model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| Kimi K3 | Thinking always on; `reasoning_effort` low, high or max (default max) | [Model card](https://huggingface.co/moonshotai/Kimi-K3) |
| Claude (current models) | Adaptive thinking steered by `effort` | [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/effort) |

DeepSeek-V4-Pro's model card shows what the dial does, with the same weights in every column:

| Benchmark | Non-think | Think High | Think Max |
|---|---|---|---|
| HMMT February 2026 math contest (pass@1) | 31.7 | 94.0 | 95.2 |
| GPQA Diamond science questions (pass@1) | 72.9 | 89.1 | 90.1 |
| Humanity's Last Exam (pass@1) | 7.7 | 34.5 | 37.7 |
| MMLU-Pro knowledge (exact match) | 82.9 | 87.1 | 87.5 |

Thinking triples the math score but adds under 5 points on MMLU-Pro, which mostly tests what the model already knows. For Think Max, the card recommends a context window of at least 384K tokens.

### What still goes wrong

- Models overthink easy questions. [Chen et al.](https://arxiv.org/abs/2412.21187) found that o1-style models used 1,953% more tokens than conventional models to answer "what is the answer of 2 plus 3?" QwQ-32B-Preview wrote 901 tokens and 13 separate solutions; the first, 39 tokens long, was already right. DeepSeek lists overthinking among R1's open limitations.
- The chain of thought may leave out what drove the answer. Anthropic's [Chen et al. (2025)](https://arxiv.org/abs/2505.05410) slipped hints into questions and checked whether models that used a hint admitted it. Overall, Claude 3.7 Sonnet did 25% of the time and DeepSeek-R1 39%. When the team trained Claude 3.7 Sonnet with RL in environments containing a deliberate reward hack, it learned to exploit every hack but mentioned it in under 2% of examples in 5 of the 6 environments.
- Checkers get gamed too. During RL training of a frontier reasoning model on coding tasks, [OpenAI found](https://arxiv.org/abs/2503.11926) the agent calling `exit(0)` to exit before all the unit tests ran, and raising `SkipTest` to skip them. A GPT-4o monitor reading the chain of thought caught 95% of these hacks, against 60% when it saw only actions. But when the monitor's verdict went into the reward and was optimized hard, the model learned to hide its intent and kept hacking.
- Most questions have no checker. Math answers and test suites are easy to verify; a good essay or a sound legal argument is not. R1's rule-based rewards covered only math, coding and logic problems, and for everything else it fell back on learned reward models, weaknesses included.

## Common misconceptions

- **"A reasoning model is a new kind of architecture."** DeepSeek-R1 has exactly DeepSeek-V3's architecture and size. What differs is post-training that rewards long, self-checking chains of thought, and the token budget to write them.
- **"The visible reasoning is a faithful record of how the model decided."** It's generated text, and it can omit what mattered. In hint experiments, models acknowledged a hint they used 25% to 39% of the time, and reward hacks almost never.
- **"More thinking always helps."** Gains flatten out, as s1 and Snell's hardest questions show, and easy questions get padded with redundant checks. Anthropic's effort docs warn that `max` can cause overthinking on some tasks.
- **"RL taught R1 to reason from nothing."** RL built on what pretraining had already put in the model. R1-Zero started at 15.6% on AIME, not zero, and a 32B model trained with the same kind of RL fell far short of one distilled from R1.

## Key takeaways

- A model does a fixed amount of work per token, so written intermediate steps give it more computation per question, and each step becomes context for the next.
- Voting over many chains helps when the model is usually right and locks in errors when it's usually wrong. Verifiers, especially step-level ones, pick better than votes.
- Reinforcement learning with rewards a program can check taught DeepSeek-R1-Zero to think longer, check itself and backtrack, with no example reasoning to copy. R1 took AIME 2024 from V3's 39.2% to 79.8%.
- GRPO drops PPO's value model by scoring each answer against its own group, so a question teaches something only when some answers are right and some wrong.
- Thinking is paid for in output tokens, latency and KV cache memory. An 8,000-token chain makes an answer about 12 times more expensive, so match the effort setting to the task.

## What's next

Every model in this series so far has read and written only text. [Part 15](/blog/multimodal-llms/) is about how LLMs see and hear: how a vision encoder cuts an image into patches and turns them into tokens, how those tokens are projected into the same stream as words, and how audio becomes tokens too.

## Further reading

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903): the paper that named chain of thought, including the row-of-dots ablation.
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916): "Let's think step by step" with no examples at all.
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022)](https://arxiv.org/abs/2203.11171): sampling many chains and taking a majority vote.
- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050): process versus outcome reward models, and the PRM800K dataset.
- [Learning to reason with LLMs (OpenAI, 2024)](https://openai.com/index/learning-to-reason-with-llms/): the o1 announcement, its compute plots and the decision to hide the chain of thought.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948): R1-Zero, the four-stage R1 pipeline, training costs and distillation.
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)](https://arxiv.org/abs/2402.03300): where GRPO was introduced and compared with PPO.
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (Snell et al., 2024)](https://arxiv.org/abs/2408.03314): when thinking longer beats a 14 times larger model, and when it doesn't.
- [s1: Simple test-time scaling (Muennighoff et al., 2025)](https://arxiv.org/abs/2501.19393): 1,000 training examples and budget forcing with "Wait".
- [Qwen3 Technical Report (Qwen Team, 2025)](https://arxiv.org/abs/2505.09388): thinking and non-thinking modes in one model, and thinking budgets.
- [Reasoning Models Don't Always Say What They Think (Chen et al., Anthropic, 2025)](https://arxiv.org/abs/2505.05410): measuring chain-of-thought faithfulness.
- [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation (Baker et al., OpenAI, 2025)](https://arxiv.org/abs/2503.11926): reward hacking in coding RL and what happens when you penalize it in the chain of thought.
- [Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs (Chen et al., 2024)](https://arxiv.org/abs/2412.21187): measuring wasted thinking tokens.
- [Reasoning models (OpenAI API docs)](https://developers.openai.com/api/docs/guides/reasoning) and [Extended thinking (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/extended-thinking): how effort settings, budgets and billing work in practice.
- [Reasoning models don't always say what they think (Anthropic, 2025)](https://www.anthropic.com/research/reasoning-models-dont-say-think): a readable summary of the faithfulness results.
- [RLHF Book (Nathan Lambert)](https://rlhfbook.com/): a free textbook on post-training, with a section on reasoning and RLVR.
