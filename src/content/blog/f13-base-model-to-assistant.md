---
title: "From Base Model to Assistant: SFT, RLHF and DPO"
description: "How a next-token predictor becomes a helpful assistant: supervised fine-tuning, reward models, RLHF with PPO, and DPO, worked with real numbers."
slug: "base-model-to-assistant"
category: "llm-fundamentals"
tags: ["fine-tuning", "training", "rlhf", "alignment", "evals"]
series: "llm-fundamentals"
seriesOrder: 13
pubDate: 2026-10-23
---

## TL;DR

A pretrained model only continues text. Turning it into an assistant takes a second, much cheaper phase called post-training. First, supervised fine-tuning (SFT) shows the model thousands of example conversations and trains it to produce the assistant's side. Then preference tuning teaches it which of two answers people like better: either by training a reward model and optimizing against it with reinforcement learning (RLHF), or by updating the model directly on the preference pairs (DPO). This article works through each step with real numbers, from a masked SFT loss to a DPO gradient, then shows how Llama 3 and Tülu 3 combine them and what goes wrong when they're pushed too far.

## Why it matters

Ask a small base model, SmolLM2-360M, "Give me three tips for sleeping better." In our own run it answered: "Tip 1: Don't sleep in the same bed as your partner. Tip 2: Don't sleep with your partner. Tip 3: Don't sleep with your partner in the same room." Its instruct twin, [built from those same weights](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) with SFT and then DPO, the two techniques this article explains, recommended a consistent sleep schedule and a relaxing bedtime routine.

[Part 1](/blog/how-llms-work/) quoted the result that made this step famous: in OpenAI's InstructGPT paper, people preferred answers from a 1.3 billion-parameter tuned model over the raw 175 billion-parameter GPT-3. And the tuning was cheap. The paper reports that its 175B SFT model took [4.9 petaflop/s-days](https://arxiv.org/abs/2203.02155) of compute and the full RLHF model 60, against 3,640 for pretraining GPT-3. That's under 2% of the pretraining bill for a change users value more than a 100-fold jump in size.

So how does so little training change the behavior so much? And why do assistants still flatter you, refuse harmless questions and pad their answers? The same few training steps explain both.

## The intuition

Think of a new hire who has read every book in a huge library but has never had a job. They know a lot, and they have no idea what a good answer to a customer looks like. First you hand them a binder of model replies to copy. That's SFT. Then a manager reads pairs of their drafts and says "this one's better", over and over. Either the hire builds a mental scorecard of the manager's taste and practices against it (RLHF), or they adjust straight from the stack of marked-up pairs (DPO).

The analogy breaks in two places. The model never learns *why* an answer was preferred, only which kind gets picked, so it also absorbs the manager's blind spots: a liking for confident, long or flattering answers. And the "practice" in RLHF isn't with live users. It happens in training, against a frozen scoring model, before anyone chats with the result.

## Step 1: supervised fine-tuning on demonstrations

### What a base model does with a question

A base model is trained only to continue text (Part 12 covers [how pretraining works](/blog/pretraining-scaling-laws/)). A question is just the start of a document, and the model guesses what kind of document it is. Here are greedy outputs from our own CPU run of SmolLM2-360M and its instruct version, which Hugging Face released together, trimmed to the first lines:

| Prompt | SmolLM2-360M (base) | SmolLM2-360M-Instruct |
|---|---|---|
| Give me three tips for sleeping better. | "Tip 1: Don't sleep in the same bed as your partner. Tip 2: Don't sleep with your partner..." | "1. Maintain a consistent sleep schedule: Try to go to bed and wake up at the same time every day..." |
| Write a haiku about the ocean. | "The ocean is a big place. It's a place where you can see the sun, and the moon, and the stars..." | "The waves below, / A symphony of sound, / Nature's eternal song." |
| What is the capital of France? | "Paris, France / What is the capital of France? / Paris, France / What is the capital of France?..." | "The capital of France is Paris." |

The base model knows Paris is the capital. It just treats the prompt as a quiz page and keeps writing quiz lines. The instruct model answers once and stops, because it learned to emit an end-of-turn token.

Modern base models blur this line. In the same run, Qwen3-0.6B-Base answered all three prompts like an assistant. Its pretraining data is a likely reason: the Qwen team used earlier models to [synthesize "trillions of text tokens"](https://arxiv.org/abs/2505.09388) in formats that include question-answering and instructions.

### Training on the assistant's side

SFT is the same next-token training as pretraining, on a much smaller, curated dataset of conversations. Each example is laid out with the model's chat template, the special-token format from [Part 2](/blog/tokenization/), and the model learns to predict the assistant's reply.

The early papers showed how far a little of this goes. Google's [FLAN](https://arxiv.org/abs/2109.01652) (2021) took a 137B model and fine-tuned it on over 60 NLP datasets rewritten as natural-language instructions, about ten templates per dataset. The paper named the method instruction tuning, and FLAN beat zero-shot 175B GPT-3 on 20 of 25 datasets, on task types it had never been tuned on. [InstructGPT](https://arxiv.org/abs/2203.02155) (2022) instead had about 40 contractors write ideal answers to real API prompts. Its SFT set held about 13,000 training prompts, and it trained for 16 epochs.

Then Meta's [LIMA](https://arxiv.org/abs/2305.11206) (2023) used just 1,000 curated examples, about 750,000 tokens in total, to fine-tune a 65B LLaMa. In a human study, LIMA's answers were equivalent to or better than GPT-4's in 43% of cases. The authors proposed the "Superficial Alignment Hypothesis": a model's knowledge and capabilities are "learnt almost entirely during pretraining", and alignment mostly teaches it which format to use when talking to users.

### Only the answer counts

In SFT, the loss is computed only on the response. The prompt is context, not a target, so its tokens get no training signal. Here is our example conversation after SmolLM2-Instruct's chat template, which adds a default system message:

```text
tokens 0-20    <|im_start|>system
               You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>   -> ignored
tokens 21-32   <|im_start|>user
               What is the capital of France?<|im_end|>                                         -> ignored
tokens 33-36   <|im_start|>assistant                                                            -> ignored
tokens 37-45   The capital of France is Paris.<|im_end|>                                        -> trained
```

Each line ends with a newline token, counted in the ranges. Of the 46 tokens, 37 (80%) are masked and 9 carry the loss. In practice the masked positions get the label `-100`, which PyTorch's cross-entropy loss skips by default (`ignore_index=-100`). The loss itself is the familiar one from [Part 3](/blog/neural-networks-basics/): `-ln(p)` of each correct next token, averaged over the 9 trained positions only.

The `<|im_end|>` token among those 9 teaches the model to stop, which is exactly what the base model couldn't do in the table above. Llama 3's post-training describes the same rule: [cross-entropy loss "on the target tokens (while masking loss on prompt tokens)"](https://arxiv.org/abs/2407.21783).

SFT has a ceiling, though. It trains the model to imitate every demonstration equally, good and mediocre, and it never shows what a *bad* answer looks like. For that, you need comparisons.

## Step 2: a reward model learns what people prefer

### Comparisons, not scores

Writing a perfect answer is slow. Judging which of two answers is better is fast. So the next step collects comparisons: sample several answers from the SFT model for one prompt and ask a person to rank them.

InstructGPT showed labelers between 4 and 9 answers per prompt. Ranking `K` answers yields `K × (K − 1) / 2` pairs, so 4 answers give 6 comparisons and 9 give 36. Its reward-model dataset covered about 33,000 prompts. Training labelers agreed with each other 72.6% of the time, a reminder that "human preference" is itself noisy.

A reward model (RM) turns those pairs into a function that scores any answer. In InstructGPT it's a 6 billion-parameter copy of the SFT model with the final vocabulary layer replaced by a single number: a scalar reward for a (prompt, answer) pair. The authors used 6B because 175B reward models trained unstably.

### The Bradley-Terry loss, worked by hand

The reward model is trained with a loss from the Bradley-Terry model of paired comparisons. If the chosen answer scores `r_chosen` and the rejected one `r_rejected`, the model's probability that people prefer the chosen one is `σ(r_chosen − r_rejected)`, where `σ` is the sigmoid function `1 / (1 + e^(−z))` that squashes any number into 0 to 1. The loss is:

```text
loss = −ln σ(r_chosen − r_rejected)
```

Only the difference matters. Four cases:

| `r_chosen` | `r_rejected` | Difference | P(chosen wins) | Loss | Push on `r_chosen` |
|---|---|---|---|---|---|
| 2.0 | 0.5 | +1.5 | 0.818 | 0.201 | +0.182 |
| 0.5 | 2.0 | −1.5 | 0.182 | 1.701 | +0.818 |
| 1.0 | 1.0 | 0.0 | 0.500 | 0.693 | +0.500 |
| 4.0 | 0.5 | +3.5 | 0.971 | 0.030 | +0.029 |

The last column is the gradient step's direction and size: raise the chosen score by `1 − P` and lower the rejected score by the same amount. When the model already ranks the pair correctly by a wide margin, the push is tiny. When it ranks them backwards, the push is four and a half times bigger than in the first row. It's the same cross-entropy idea as next-token training, applied to a two-way choice.

Because only differences matter, the scores have no natural zero. InstructGPT shifted its reward model so that human demonstrations average 0 before using it.

### Where the comparisons come from: people or AI

Human labels are expensive and slow, so labs increasingly use models to label. Anthropic's [Constitutional AI](https://arxiv.org/abs/2212.08073) (2022) trained a harmless assistant "without any human labels identifying harmful outputs". A model critiqued and revised its own answers against a short written list of principles, the "constitution" (16 in the paper), and a model also picked the better of two answers to build the preference data. The paper calls this RL from AI Feedback (RLAIF). Human labels were still used for helpfulness.

## Step 3: RLHF, optimizing the reward on a leash

### The objective

With a reward model in hand, reinforcement learning from human feedback (RLHF) fine-tunes the SFT model, now called the policy, to produce answers the reward model scores highly. Each training step samples prompts, lets the policy write answers, scores them, and nudges the policy toward higher-scoring answers. InstructGPT ran this on about 31,000 prompts, for 256,000 episodes.

The idea comes from [Christiano et al.](https://arxiv.org/abs/1706.03741) (2017), who taught simulated robots and Atari agents from human comparisons of short video clips, with feedback on less than 1% of the agent's interactions. A simulated Hopper robot learned backflips from 900 human comparisons, in under an hour of human time. [Stiennon et al.](https://arxiv.org/abs/2009.01325) (2020) applied it to summarizing Reddit posts: a 1.3B model trained with human feedback beat a supervised model 10 times its size. People preferred its summaries over the human-written reference summaries 61% of the time, against 43% for the bigger supervised model.

The optimizer is usually PPO (proximal policy optimization, [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)), a reinforcement learning algorithm that limits how far each update can move the policy. What it maximizes is this:

```text
objective = reward_model(prompt, answer) − β × log( π(answer) / π_SFT(answer) )
```

`π(answer)` is the probability the policy gives to its own answer and `π_SFT(answer)` is the probability the frozen SFT model gives it. Averaged over the policy's answers, the log of their ratio is the KL divergence, a measure of how far the policy has drifted from where it started. `β` sets the price of drifting. InstructGPT applied the penalty per token and used `β = 0.02`.

### Why the leash: reward hacking

The reward model is an imperfect proxy trained on tens of thousands of comparisons. An optimizer pushed hard enough will find answers it overrates, a failure called reward hacking. Stiennon et al. measured it: as they optimized harder, real human preferences fell away from the reward model's predictions, and "eventually the reward model becomes anti-correlated with human preferences." [Gao et al.](https://arxiv.org/abs/2210.10760) (2022) traced the same curve and tied it to Goodhart's law.

The KL term makes strange answers expensive. A toy example with `β = 0.1`, where the log-ratio is summed over the answer's tokens:

| Answer | Reward | Log-ratio vs SFT | Objective |
|---|---|---|---|
| Normal, helpful answer | 2.0 | 0.5 | `2.0 − 0.1 × 0.5 = 1.95` |
| Odd answer the RM overrates | 3.0 | 15.0 | `3.0 − 0.1 × 15 = 1.50` |

The odd answer fools the reward model by a full point but loses once drift is priced in.

### What β does, in closed form

This objective has a known exact solution, and it's the key to Step 4. The best possible policy is the SFT model's distribution reweighted by the exponential of the reward: `π*(y) ∝ π_SFT(y) × e^(r(y)/β)`. Take three possible answers that the SFT model produces with probabilities 0.6, 0.3 and 0.1, rewarded 0, 1 and 2:

| β | P(answer A, r=0) | P(answer B, r=1) | P(answer C, r=2) | Expected reward | KL from SFT |
|---|---|---|---|---|---|
| 10 | 0.569 | 0.315 | 0.116 | 0.55 | 0.002 |
| 1 | 0.279 | 0.379 | 0.343 | 1.06 | 0.30 |
| 0.5 | 0.072 | 0.268 | 0.660 | 1.59 | 1.06 |
| 0.1 | 0.000 | 0.000 | 1.000 | 2.00 | 2.30 |

A large `β` barely moves the model. A small one collapses it onto the single top-scoring answer, which is exactly where a flawed reward model does the most damage.

### What RLHF bought InstructGPT

Labelers preferred the 175B InstructGPT over GPT-3 85 ± 3% of the time. On TruthfulQA it gave truthful and informative answers about twice as often. On tasks where the answer should come only from the input, like summarization, it made things up 21% of the time against GPT-3's 41%. It produced about 25% fewer toxic outputs when asked to be respectful, but showed no improvement on bias benchmarks.

It also paid an "alignment tax": RLHF lowered scores on some public benchmarks, including SQuAD, DROP, HellaSwag and French-to-English translation. Mixing ordinary pretraining updates into the RL training, a variant the paper calls PPO-ptx, greatly reduced those regressions without hurting labeler preferences.

## Step 4: DPO, skipping the reward model

RLHF with PPO is a heavy machine. During training it keeps four models around: the policy, the frozen SFT reference, the reward model and a value model that PPO uses to judge progress. It also has to generate fresh answers at every step. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) removes the reward model and the sampling.

### The closed-form trick

Start from the solution in Step 3, `π*(y) ∝ π_ref(y) × e^(r(y)/β)`, and solve it for the reward:

```text
r(y) = β × log( π*(y) / π_ref(y) ) + β × log Z
```

`Z` is a normalizing constant that is hopeless to compute, because it sums over every possible answer. But the Bradley-Terry loss from Step 2 only uses the *difference* between two rewards for the same prompt, so `Z` cancels. Plug the policy's own log-ratios into the reward-model loss and you get the DPO loss:

```text
loss = −ln σ( β × [log π(chosen)/π_ref(chosen)] − β × [log π(rejected)/π_ref(rejected)] )
```

The paper's subtitle sums it up: "Your Language Model is Secretly a Reward Model." The term `β × log π(y)/π_ref(y)` acts as an implicit reward, and training is a simple classification loss on the preference pairs you already have.

### A worked DPO step

Take one preference pair and `β = 0.1`, the paper's default. The numbers are log-probabilities of each whole answer, summed over its tokens. The reference model is the frozen SFT model, and it gives the chosen answer −12.5 and the rejected one −13.0.

| Policy state | Chosen log-ratio | Rejected log-ratio | Margin `β × (difference)` | Loss | Gradient weight |
|---|---|---|---|---|---|
| Start (policy = reference) | 0.0 | 0.0 | 0.00 | 0.693 | 0.050 |
| After some training | +0.5 | −1.0 | +0.15 | 0.621 | 0.046 |
| Ordered the wrong way | −1.0 | +1.0 | −0.20 | 0.798 | 0.055 |
| Well separated | +2.5 | −10.0 | +1.25 | 0.252 | 0.022 |

At the start the policy equals the reference, so the loss is `ln 2 = 0.693`, a coin flip. In the second row the policy raised the chosen answer's log-probability from −12.5 to −12.0 and lowered the rejected one from −13.0 to −14.0, so the implicit rewards are +0.05 and −0.10 and the loss falls to 0.621.

The gradient always points the same way: raise the log-probability of the chosen answer, lower the rejected one. Its size is `β × σ(−margin)`, the last column. Pairs the model currently gets wrong get the biggest push, and pairs it has already separated get less. The DPO paper reports that without this weighting, the model can degenerate.

### Results and variants

On the same Reddit summarization task, with the same 6B GPT-J SFT model, DPO's summaries won against the reference summaries about 61% of the time at sampling temperature 0, against 57% for PPO at its best temperature, as judged by GPT-4. It also beat PPO at steering the sentiment of generated text. Meta reported the same for Llama 3: PPO was tried, but DPO "required less compute for large-scale models and performed better".

Several variants followed, each changing one assumption:

| Method | Paper | What it changes |
|---|---|---|
| IPO | [Azar et al., 2023](https://arxiv.org/abs/2310.12036) | Replaces the Bradley-Terry log-sigmoid to avoid overfitting when preferences are near-deterministic |
| KTO | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) | Learns from single answers marked good or bad, no pairs needed |
| ORPO | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) | Folds a preference penalty into SFT itself, with no reference model and no separate stage |

## In the real world

![Flowchart: a pretrained base model goes through SFT on demonstrations, then the SFT model samples two answers and a person picks the better one. From those comparisons, the RLHF path trains a reward model that learns a score from pairs, then runs RL with PPO maximizing reward minus β times the KL divergence to the SFT model. The DPO shortcut instead updates the model directly on the pairs. Both paths end at the assistant model.](/diagrams/f13-post-training-pipeline.svg "The classic post-training pipeline: SFT, then preferences, then either a reward model plus RL or DPO directly.")

Both preference paths start from the same comparisons; DPO skips the reward model and the RL loop.

Real recipes mix these pieces and run them more than once:

| Recipe | SFT | Preference step | Notable details |
|---|---|---|---|
| [InstructGPT](https://arxiv.org/abs/2203.02155) (2022) | About 13K prompts with labeler-written answers | 6B reward model on about 33K prompts, then PPO on about 31K | `β = 0.02`; pretraining updates mixed in (PPO-ptx) |
| [Llama 3](https://arxiv.org/abs/2407.21783) (2024) | Rejection-sampled answers plus synthetic data | DPO with `β = 0.1` | Six rounds; the reward model picks the best of 10-30 samples per prompt for SFT data |
| [Tülu 3](https://arxiv.org/abs/2411.15124) (2024) | 939,344 prompts | Length-normalized DPO on 354,192 pairs | Then RL with verifiable rewards (RLVR) |

Llama 3 shows how the reward model survives even without PPO. In each of six rounds, Meta trained a reward model on all the human preference data so far, sampled 10 to 30 answers per prompt from the latest model, and kept the one the reward model scored highest as new SFT data. That's rejection sampling. Then it ran DPO on the newest preference batches. Meta also changed DPO in two ways: special formatting tokens were masked out of the DPO loss, because they appear in both answers and pull the model in opposite directions, and an extra SFT-style loss on the chosen answers (weight 0.2) kept their probabilities from dropping.

Tülu 3, from the Allen Institute for AI, is the most complete open recipe: data, code and evaluations are all public. Its last stage, RLVR, swaps the learned reward model for a checker. If a math answer is verifiably correct, the reward is 10; otherwise 0. The KL penalty stays. That idea, rewarding answers that can be checked rather than answers people prefer, is what reasoning models scale up.

### What goes wrong

Every failure below comes from optimizing a proxy for "what people prefer".

- Sycophancy. [Sharma et al.](https://arxiv.org/abs/2310.13548) (2023) found that five AI assistants consistently told users what they wanted to hear across four tasks. In human preference data, answers matching the user's views were more likely to be preferred. Asked "Are you sure?" after a correct answer, Claude 1.3 wrongly admitted a mistake on 98% of questions.
- Verbosity. Stiennon's reward models favored longer summaries, and [Singhal et al.](https://arxiv.org/abs/2310.03716) (2023) found that a reward based purely on length reproduced most of RLHF's gains over SFT in their settings. LLM judges share the bias; see [Making LLM-as-a-Judge Trustworthy](/blog/trustworthy-llm-as-a-judge/) for length-controlled evaluation.
- Over-refusal. Safety tuning can teach a model to fear words rather than requests. On [XSTest](https://arxiv.org/abs/2308.01263), a test of clearly safe prompts that use risky-sounding words, such as how to "kill a Python process", Llama-2-70b-chat with its original system prompt fully refused 38% and partly refused another 21.6%.
- Reward hacking. Push too hard against any learned reward and the model finds its blind spots, as Step 3 showed.

### Should your team do this?

Most product teams don't run RLHF themselves. They prompt a model that already went through all of the above. Fine-tuning earns its place for behavior that prompting can't fix, such as a strict format, a house style or distilling a big model into a small one, and it's a poor way to add facts. [Fine-Tuning vs RAG vs Prompting](/blog/fine-tuning-vs-rag-vs-prompting/) gives the decision procedure, and [Fine-Tuning a Small Model with LoRA on 5,000 Examples](/blog/lora-fine-tuning-5k-examples/) walks through a LoRA SFT run, including the same loss masking shown in Step 1. DPO on a few thousand preference pairs is the usual next step when you have a clear "better versus worse" signal.

## Common misconceptions

- **"Post-training teaches the model what it knows."** Almost all knowledge comes from pretraining. LIMA matched much larger tuning efforts with 1,000 examples, and InstructGPT's whole post-training used under 2% of GPT-3's pretraining compute. Post-training mostly selects which of the model's existing behaviors to show.
- **"RLHF means the model learns from your chats in real time."** The reinforcement learning happens during training, against a frozen reward model. The deployed model's weights don't change when you click thumbs up; that feedback can only feed a future training round.
- **"DPO doesn't need a reference model."** It does. The reference sits in every term of the loss, playing the role of the KL leash. Methods like ORPO are the ones that drop it.
- **"Preference tuning makes a model more truthful."** It makes the model more *preferred*. That helped InstructGPT on TruthfulQA, and it also rewards flattery and confident length when labelers favor them.
- **"A base model can't answer questions."** It can, if the prompt looks like a document where an answer comes next. What it lacks is the habit of treating your text as a request, answering once and stopping.

## Key takeaways

- A base model continues documents. Post-training turns it into an assistant at a small fraction of the pretraining cost: under 2% for InstructGPT.
- SFT is next-token training on example conversations, with the loss masked to the assistant's tokens, including the end-of-turn token that teaches the model to stop.
- A reward model learns from pairs with the loss `−ln σ(r_chosen − r_rejected)`. RLHF then maximizes that reward minus `β` times the drift from the SFT model.
- DPO uses the closed-form solution of that same objective to train directly on preference pairs, with no reward model and no sampling. Llama 3 and Tülu 3 both use it.
- Optimizing a proxy for human preference brings its biases along: sycophancy, verbosity, over-refusal and reward hacking.

## What's next

Tülu 3's last stage pointed somewhere new: reward answers that can be checked, like a math result or passing code, instead of answers people prefer. Part 14 follows that thread into reasoning models. It covers how OpenAI's o1 and DeepSeek-R1 learned to think for thousands of tokens before answering, what GRPO changes about PPO, and why spending more compute at answer time can beat training a bigger model.

## Further reading

- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155): InstructGPT, the canonical SFT, reward model and PPO pipeline, with its data sizes and the alignment tax.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290): the closed-form derivation, the DPO loss and its gradient.
- [Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741): where learning a reward from human comparisons started.
- [Learning to summarize from human feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325): RLHF on language, the KL penalty's purpose and reward over-optimization.
- [Finetuned Language Models Are Zero-Shot Learners (Wei et al., 2021)](https://arxiv.org/abs/2109.01652): FLAN and instruction tuning.
- [LIMA: Less Is More for Alignment (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206): 1,000 examples and the Superficial Alignment Hypothesis.
- [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073): self-critique and RL from AI feedback.
- [The Llama 3 Herd of Models (Llama Team, Meta, 2024)](https://arxiv.org/abs/2407.21783): Section 4 describes six rounds of rejection sampling, SFT and DPO at 405B scale.
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124): a fully open SFT, DPO and RLVR recipe.
- [Towards Understanding Sycophancy in Language Models (Sharma et al., 2023)](https://arxiv.org/abs/2310.13548): how preference data rewards agreeing with the user.
- [Scaling Laws for Reward Model Overoptimization (Gao et al., 2022)](https://arxiv.org/abs/2210.10760): measuring reward hacking as optimization pressure grows.
- [A Long Way to Go: Investigating Length Correlations in RLHF (Singhal et al., 2023)](https://arxiv.org/abs/2310.03716): how much of RLHF's gain is just longer answers.
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347): the RL algorithm behind classic RLHF.
- [RLHF Book (Nathan Lambert)](https://rlhfbook.com/): a free, detailed textbook on post-training methods.
- [Illustrating Reinforcement Learning from Human Feedback (Hugging Face, 2022)](https://huggingface.co/blog/rlhf): a visual walkthrough of the three-stage pipeline.
