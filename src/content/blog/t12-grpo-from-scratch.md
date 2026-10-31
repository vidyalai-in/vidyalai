---
title: "Build an LLM From Scratch: RL with Verifiable Rewards (GRPO)"
description: "GRPO on the 88M model: a checker for addition, 8 replies per problem, reward curves, pass@k before and after, and what RL couldn't teach it."
slug: "grpo-from-scratch"
category: "llm-training"
tags: ["training", "fine-tuning", "evals", "gpu", "rlhf"]
series: "llm-from-scratch"
seriesOrder: 12
pubDate: 2026-08-03
---

## TL;DR

This part trains Part 10's 88M chat model with reinforcement learning with verifiable rewards (RLVR) on addition, graded by a few lines of Python. GRPO (group relative policy optimization), written from scratch, scores 8 replies per problem against each other inside PPO's clipped ratio. The chat model scores zero, so GRPO has nothing to learn from until a 34-second warm-start on 4,000 examples, which also turns every prompt into a sum. Then 250 steps in 20 minutes take held-out greedy accuracy from 17.8% to 77.4% and pass@64 from 72% to 94%, finding a checker loophole ("042") on the way. Nothing carried over to 4-digit numbers. Mixing Part 10's conversations into the warm-start kept the chat model and its benchmarks, but the same GRPO run then reached 37.4%.

## Why it matters

[Part 10](/blog/sft-from-scratch/)'s model, asked "What is 17 + 25?" and sampled, answered 31. On 500 held-out sums like "Compute 44 + 641." it gets none right, and only 8 even if you take the last number in its reply as the answer. Supervised fine-tuning taught it what an answer sounds like, not how to add.

Twenty minutes of practice against a checker, after a half-minute warm-up, gets 387 of the same 500 right, greedy. Nobody wrote a new example in those twenty minutes: the model wrote 64,000 replies, a program marked each right or wrong, and GRPO pushed up the replies that beat their group.

That's the recipe behind DeepSeek-R1, scaled down to a laptop. [Math Part 16](/blog/alignment-math/) derived GRPO's update and [Fundamentals Part 14](/blog/reasoning-models/) told R1's story. This part runs it and tests a question the field argues about: does RL teach a model anything new, or only make it pick what it could already sample?

## The intuition

Think of a student with a practice sheet whose answers are printed at the back. They try each problem eight times, check every attempt against the key, and lean toward whatever the right attempts did. Nobody shows them a worked solution, and the key only says right or wrong.

The analogy breaks in two places. GRPO learns nothing from a problem where all eight attempts are wrong, or all right. And a student can work out why an attempt failed, while GRPO can only shift probability between attempts the model already makes; if none is ever right, there is nothing to shift toward.

## Step 1: a task and a checker

### Addition, checked by a program

The task is the sum of two whole numbers with 1 to 3 digits each, asked three ways ("What is 347 + 582?", "Add 347 and 582.", "Compute 347 + 582."), in Part 10's chat template. Part 2's tokenizer splits numbers into single digits. The fixed held-out sets are 500 problems like the training ones, plus 200 with exactly 4 digits per number and 200 with 5; training never sees a held-out pair.

The checker, from the new `grpo.py`:

```python
# grpo.py (excerpt)
FINAL = re.compile(r"The answer is (\d+)\.$")


def reward(reply, answer, kind="strict"):
    """1.0 if the reply is right, else 0.0. strict: the reply ends with "The answer is <answer>."
    contains: the answer appears anywhere in the reply as a number (a checker that can be gamed)."""
    if kind == "strict":
        m = FINAL.search(reply.strip())
        return float(m is not None and int(m.group(1)) == answer)
    return float(str(answer) in re.findall(r"\d+", reply))
```

The strict version wants one final answer, in one format, at the very end:

```python
from grpo import reward
for reply in ["48 + 75 = 123. The answer is 123.", "48 + 75 = 123.", "The answer is 123",
              "The answer is 123. Or maybe 124.", "The answer is 132."]:
    print(reward(reply, 123), reward(reply, 123, "contains"), repr(reply))
```

```text
1.0 1.0 '48 + 75 = 123. The answer is 123.'
0.0 1.0 '48 + 75 = 123.'
0.0 1.0 'The answer is 123'
0.0 1.0 'The answer is 123. Or maybe 124.'
0.0 0.0 'The answer is 132.'
```

A program decides, which is what makes the reward verifiable. The `contains` version exists to be gamed in Step 4.

### Starting pass rates: no signal

GRPO learns only from groups whose replies disagree. Before any RL:

| Model | Greedy, strict checker (500 problems) | Greedy, last number in the reply | Samples right at temperature 1 (pass@1) | pass@16 | Random groups of 8 with no right reply |
|---|---|---|---|---|---|
| Part 10's SFT model | 0.0% | 1.6% | 0 of 3,200 | 0.0% | 100% |
| After a 4,000-example warm-start | 17.8% | 17.8% | 11.4% | 47.0% | 64% |

Samples come from the first 200 problems; the last column is one minus pass@8.

The SFT model never ends with "The answer is N.", so the strict checker gives it zero, and its replies average 136 tokens of "we can use the formula for the sum of a set of numbers". With every reward zero, every advantage and so every gradient is zero, and twenty minutes of GRPO would change nothing.

There are three honest fixes: an easier curriculum (in a probe with a lenient checker, about 3% of samples on 1-digit sums were right), partial credit for the format, or a short SFT warm-start on the answer format. This part uses the warm-start, a tiny version of DeepSeek-R1's "thousands of cold-start data" fine-tuned in before RL. It also gives the model two ways to answer, so RL has a choice to make:

```python
# grpo.py (excerpt)
def direct(a, b):
    return f"{a} + {b} = {a + b}. The answer is {a + b}."


def worked(a, b):
    """Column addition, ones first, with the carries written out."""
    da, db, carry, steps = str(a)[::-1], str(b)[::-1], 0, []
    for i in range(max(len(da), len(db))):
        x, y = int(da[i]) if i < len(da) else 0, int(db[i]) if i < len(db) else 0
        s = x + y + carry
        terms = f"{x} + {y}" + (f" + {carry}" if carry else "")
        steps.append(f"{terms} = {s}, write {s % 10}" + (", carry 1" if s >= 10 else ""))
        carry = s // 10
    if carry:
        steps.append("Write the carry 1")
    return "Add the digits from right to left. " + ". ".join(steps) + f". The answer is {a + b}."
```

```python
from grpo import worked
print(worked(48, 75))
```

```text
Add the digits from right to left. 8 + 5 = 13, write 3, carry 1. 4 + 7 + 1 = 12, write 2, carry 1. Write the carry 1. The answer is 123.
```

`python grpo.py warmstart --out runs/warm4000 --n 4000 --lr 3e-4 --warmup 5 --fused` trains Part 10's weights on 4,000 problems, half answered directly and half worked, with Part 10's masked loss: 63 steps of 64 examples, 5 warm-up steps to 3 × 10⁻⁴ and cosine decay to zero, 34 seconds. A pilot with 1,000 examples learned the format but got 0.6% of samples right, and 96.5% of groups of 8 had no right answer at all. At 4,000, held-out greedy accuracy is 17.8% and 11.4% of samples are right. That is deliberately weak: a warm-start that solved the task would leave RL nothing to do.

## Step 2: GRPO in code

### Advantages from the group

[Math Part 16](/blog/alignment-math/) derived the update. For each problem, sample $$G$$ replies, score them, and give every token of reply $$i$$ the advantage

$$
A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}
$$

where $$r_i$$ is reply $$i$$'s reward. In code:

```python
# grpo.py (excerpt)
def group_advantages(r, std=True, eps=1e-4):
    """r: (prompts, G) rewards. GRPO: (r - group mean) / group std. Dr. GRPO: r - group mean."""
    adv = r - r.mean(1, keepdim=True)
    return adv / (r.std(1, keepdim=True, unbiased=False) + eps) if std else adv
```

```python
import torch
from grpo import group_advantages
r = torch.tensor([[1., 0, 0, 0, 0, 0, 0, 0],      # 1 of 8 replies right
                  [1., 1, 1, 0, 0, 0, 0, 0],      # 3 of 8
                  [1., 1, 1, 1, 1, 1, 1, 1]])     # all 8: no signal
print(group_advantages(r).round(decimals=2))
print(group_advantages(r, std=False).round(decimals=2))
```

```text
tensor([[ 2.6400, -0.3800, -0.3800, -0.3800, -0.3800, -0.3800, -0.3800, -0.3800],
        [ 1.2900,  1.2900,  1.2900, -0.7700, -0.7700, -0.7700, -0.7700, -0.7700],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
tensor([[ 0.8800, -0.1200, -0.1200, -0.1200, -0.1200, -0.1200, -0.1200, -0.1200],
        [ 0.6200,  0.6200,  0.6200, -0.3800, -0.3800, -0.3800, -0.3800, -0.3800],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
```

The lone right answer in a hard group gets +2.64 with the division by the standard deviation and +0.88 without it. The second tensor is Dr. GRPO's advantage, from [Liu et al. (2025)](https://arxiv.org/abs/2503.20783), "Understanding R1-Zero-Like Training: A Critical Perspective". They argue the division creates a "question-level difficulty bias": "Questions with lower standard deviations (e.g., those that are too easy or too hard, with the outcome rewards being almost all 1 or 0) are given higher weights during policy updates."

### The loss on every reply token

```python
# grpo.py (excerpt, docstring omitted)
def grpo_loss(logp, old, ref, adv, mask, eps=0.2, beta=0.04, agg="seq-mean", max_tokens=128):
    ratio = torch.exp(logp - old)
    a = adv[:, None]
    pg = -torch.min(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
    kl = torch.exp(ref - logp) - (ref - logp) - 1             # k3: never negative, averages to KL
    per_tok = (pg + beta * kl) * mask
    if agg == "seq-mean":
        loss = (per_tok.sum(1) / mask.sum(1).clamp(min=1)).mean()
    else:
        loss = per_tok.sum(1).mean() / max_tokens
    clipped = ((ratio - 1).abs() > eps) & mask.bool()
    return loss, kl.detach(), clipped
```

`logp`, `old` and `ref` are each reply token's log-probability under the policy being trained, the policy that sampled it, and the frozen starting weights; `mask` covers the reply and its closing `<|end|>`, as in Part 10. The ratio and clip are PPO's, and `kl` is Schulman's k3 estimator from Math Part 16. `seq-mean` averages each reply over its own length, as DeepSeekMath's objective is written. Dr. GRPO divides by a constant instead, `MAX_TOKENS` in their listing, because dividing by the length gives a "response-level length bias": for wrong answers, "longer responses are penalized less due to their larger |oi|", the reply's length.

One more knob: how many optimizer steps to take per batch of samples. With one step, `old` equals `logp`, the ratio is 1 and the clip never acts: DeepSeekMath's setting. With several minibatches per batch (`--updates`), as in DeepSeek-R1, later ones see a policy that has moved, and the clip matters.

![A top-to-bottom flowchart of one GRPO step. 32 addition problems go to the policy, a trained copy of the warm-start model, which samples 8 replies each: 256 replies. The replies go to a checker that asks whether the last sentence is 'The answer is N.' with the right N, giving a reward of 1 or 0, which becomes an advantage: reward minus the group mean, over the group standard deviation. The replies also go to a box of log-probabilities of reply tokens under the policy, the old policy and the reference, which is the starting weights, frozen. The advantage and the log-probabilities feed the loss, the clipped ratio times the advantage plus beta times the KL estimate, and an arrow labeled gradient goes back to the policy.](/diagrams/t12-grpo-step.svg "One GRPO step: sample, check, compare within the group, update.")

No value model or reward model is trained, so the GPU holds two copies of the 88M weights plus the policy's gradients and optimizer state: 6.1 GiB at peak, with generation.

### Published settings

| Recipe | Replies per prompt (G) | KL coefficient | Learning rate | Updates per batch of samples |
|---|---|---|---|---|
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) GRPO, 7B (2024) | 64 | 0.04 | 1 × 10⁻⁶ | 1 ("a single update following each exploration stage") |
| [DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948) (2025) | 16 | 0.001 | 3 × 10⁻⁶ | 16 minibatches per rollout of 8,192 replies |
| DeepSeek-R1, first RL stage | 16 | 0.001 | 3 × 10⁻⁶ | 16, with the clip ε raised to 10 |
| This part, 88M | 8 | 0.04 | 1 × 10⁻⁵ | 1 (4 in an ablation) |

R1-Zero sampled at temperature 1, up to 32,768 tokens and later 65,536, took 32 questions per step, and replaced its reference model with the policy every 400 steps. On the clip, the R1 paper warns: "A lower value can lead to the truncation of gradients for a significant number of tokens, thereby degrading the model's performance, while a higher value may cause instability during training."

## Step 3: the runs

### Pilots

Three 40-step pilots from the warm-start, β = 0.04, 32 problems and 8 replies per step, constant learning rate:

| Learning rate | Held-out greedy accuracy after 40 steps | Training reward, steps 30-39 | KL per reply token, steps 30-39 |
|---|---|---|---|
| 3 × 10⁻⁶ | 35.6% | 15.9% | 0.011 |
| 1 × 10⁻⁵ | 39.2% | 20.4% | 0.022 |
| 3 × 10⁻⁵ | 37.0% | 24.4% | 0.037 |

All three roughly doubled the warm-start's 17.8%, and on 500 problems the 95% interval is about ±4 points, so they tie. The fastest rate drifted furthest from the reference for the same gain. The main run uses 1 × 10⁻⁵, ten times DeepSeekMath's rate for a model 80 times smaller.

### The main run

```bash
python grpo.py train --init runs/warm4000/model_final.pt --out runs/grpo_main --lr 1e-5 --kl 0.04 \
    --steps 250 --eval-every 25 --show-every 25 --ckpt-every 25 --time-cap 1800 --fused
```

| Setting or result | Value |
|---|---|
| Model | 88,096,512 parameters; policy and reference both start from the warm-start weights |
| Batch | 32 new problems per step, G = 8 replies each: 256 replies, sampled at temperature 1, up to 128 tokens |
| Steps and samples | 250 steps, 64,000 replies, 3.52M generated tokens (55 per reply on average) |
| Optimizer | Part 5's AdamW, betas (0.9, 0.95), no weight decay, constant 1 × 10⁻⁵, clipping at 1.0 (never reached: largest gradient norm 0.76) |
| Loss | GRPO as published: std-normalized advantages, per-reply token mean, ε = 0.2, β = 0.04, one update per batch |
| Wall time | 20.2 minutes, including 11 held-out evaluations and 10 checkpoints; not resumed |
| Time split per step | generation 2.34 s (51%), old-policy and reference log-probs 0.80 s (17%), gradient step 1.43 s (31%) |
| Throughput | 56 replies per second over a training step (evaluations excluded); 6,015 generated tokens per second while sampling |
| Peak GPU memory | 6.1 GiB allocated |
| GPU | SM clock median 2,115 MHz (90% between 1,845 and 2,175), memory 8,000 MHz in 54% of samples and 7,000 in 45%, 48.1 W average, 78 °C median; `nvidia-smi` reported no power limit |
| Seed | 1337; each step's problems come from a seeded generator, so a resumed run sees the same ones |

Hardware and software are the series' usual: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), WSL2, driver 592.82, PyTorch 2.11.0 with CUDA 12.8, BF16 autocast over FP32 weights, eager mode. Sampling uses Part 11's key-value-cache sampler at 256 rows per batch (Part 11's 64-row limit came from 512-token prompts; these are 15). Generation takes half of every step even with 55-token replies, which is why RL systems are built around fast inference.

### Ablations

Four 100-step runs changed one thing each from the main run:

| Run | What changed | Held-out greedy accuracy at step 100 | Training reward, steps 90-99 | KL per reply token | Entropy per reply token | Tokens clipped |
|---|---|---|---|---|---|---|
| Main run | nothing | 53.0% | 39.2% | 0.036 | 0.050 | 0 |
| β = 0 | no KL penalty | 52.0% | 39.8% | 0.053 | 0.040 | 0 |
| Dr. GRPO | no std division, token sum over 128 | 55.0% | 33.3% | 0.024 | 0.061 | 0 |
| 4 updates per batch | 4 minibatches of 8 problems, one optimizer step each | 56.4% | 41.3% | 0.042 | 0.046 | 0.47% |
| contains checker | reward if the answer appears anywhere | 45.6% | 35.9% (its own checker) | 0.031 | 0.055 | 0 |

Each run took 8 to 9 minutes. A group-size ablation (G = 16 with 16 problems per step) was cut to stay inside the GPU budget.

![Two stacked line charts against GRPO step, 0 to 250. Top: training reward, the share of each step's 256 sampled replies that the run's checker accepts, 9-step average. The main run climbs from about 12% to about 40% at step 100, dips near 45% around step 190 and ends near 60%; four ablations that stop at step 100 track it closely, the 4-updates run slightly above, Dr. GRPO slightly below. Bottom: greedy accuracy on 500 held-out problems with the strict checker. All runs start at 17.8%. The main run reaches 34% at step 25, 53% at step 100, 66% at step 150, dips to 54% at step 200 and ends at 77.4%. At step 100 the ablations sit between 46% (contains checker) and 56% (4 updates per batch). A sixth line, the 250-step run from the mixed warm-start of Step 5, starts lower and climbs slower: training reward from about 5% to about 27%, held-out accuracy from 9% to 37.4%.](/diagrams/t12-reward-curves.svg "Training reward (top) and held-out greedy accuracy (bottom): the main run, four 100-step ablations, and the run from Step 5's mixed warm-start.")

The 95% interval on each accuracy is about ±4.4 points, so the first four rows tie: at this scale, none of the published knobs mattered much. Dropping the KL penalty let the policy drift 47% further for no gain. Dr. GRPO's gradients were 7 times smaller, which Adam's rescaling absorbs. Four optimizer steps per batch bought slightly more progress for the same generation cost, and only there did the clip act, on 0.47% of reply tokens.

The main run's curve is noisy too: greedy accuracy fell from 67.4% at step 175 to 54.4% at step 200, then reached 77.4% at step 250. The last evaluation may have landed on a high point, and the sampled pass@1 of 59.9% in Step 4 is the steadier number.

## Step 4: what the reward curves mean

### pass@k, before and after

pass@k is the share of problems where at least one of k samples is right. With n = 64 samples per problem and c of them right, the unbiased estimate from [Chen et al. (2021)](https://arxiv.org/abs/2107.03374) is 1 - C(n - c, k) / C(n, k):

```python
# grpo.py (excerpt)
def pass_at_k(n, c, k):
    """Chen et al.'s unbiased estimate of pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k)."""
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)
```

| Held-out set | Model | Greedy | pass@1 | pass@4 | pass@16 | pass@64 |
|---|---|---|---|---|---|---|
| 1 to 3 digits | Part 10's SFT model | 0.0% | 0.0% | 0.0% | 0.0% | not run |
| 1 to 3 digits | Warm start | 17.8% | 11.4% | 26.0% | 47.0% | 72.0% |
| 1 to 3 digits | After GRPO | 77.4% | 59.9% | 80.7% | 90.3% | 94.0% |
| 1 to 3 digits | Mixed warm start (Step 5) | 9.0% | 5.4% | 14.2% | 30.9% | 57.0% |
| 1 to 3 digits | Mixed warm start, after GRPO | 37.4% | 23.9% | 41.6% | 60.3% | 78.0% |
| 4 digits | all five | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 5 digits | all five | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Greedy uses 500 problems (200 for 4 and 5 digits); pass@k uses 64 samples on 200 problems (100 for 4 and 5 digits), 16 for the SFT model.

![Three side-by-side line charts of pass@k against k from 1 to 64 on a log scale. Left, 1 to 3 digits: the Part 10 SFT model is flat at 0% up to k = 16; the warm start rises from 11% at k = 1 to 26% at 4, 47% at 16 and 72% at 64; after GRPO the line starts at 60% and flattens out: 81% at 4, 90% at 16, 94% at 64. The mixed warm start of Step 5 runs from 5% to 57%, and after GRPO from 24% to 78%. Middle and right, 4 and 5 digits: all five models are at 0% for every k, their lines overlapping at zero.](/diagrams/t12-pass-at-k.svg "pass@k from 64 samples per problem at temperature 1, before and after GRPO, on three held-out sets.")

[Yue et al. (2025)](https://arxiv.org/abs/2504.13837) tested RLVR-trained models this way on math, code and visual reasoning and found that "while RLVR-trained models outperform their base models at smaller values of k (e.g., k=1), base models achieve higher pass@k score when k is large." Their reading is that RLVR "improves sampling efficiency towards correct paths" but "rarely elicit[s] fundamentally new reasoning patterns", so the base model bounds what RL can reach.

Here the curves never cross up to k = 64: GRPO is 22 points ahead at k = 64, and on 45 of the 200 problems the warm start got 0 of 64 samples right while the GRPO model got some, up to 61 of 64. The reverse happened on 1 problem. From the mixed warm start the pattern repeats: 49 problems gained, 7 lost, pass@64 from 57% to 78%. That doesn't contradict Yue et al. Their curves cross at hundreds of samples, and the warm start's is still climbing about 12 points per doubling of k; a problem solved 1% of the time is missed by all 64 samples about half the time. And their base models had read trillions of tokens, while this "base" is a 34-second warm start. On a model that weak, RL on its own correct replies partly finishes the warm start's job.

### Did replies get longer?

They grew from 42 to 60 tokens, sampled at temperature 1, but not the way R1-Zero's did. R1's paper says its model "naturally learns to solve reasoning tasks with more thinking time", writing ever longer chains. This model had two fixed styles to choose from, and RL chose one:

| Model | Sampled replies in worked style | Worked style: right | Direct style: right |
|---|---|---|---|
| Warm start | 59% | 13.4% | 8.5% |
| After GRPO | 98% | 60.5% | 26.8% (2% of replies) |

At the start, working through the columns was right more often than writing the sum straight away, so worked replies won their groups more often and the style took over within about 25 steps. Most of the rest probably came from the padding described below. Dr. GRPO's length claim can't be tested here, since length is fixed by style and column count: both runs' replies were 54 tokens at step 100.

![Four stacked line charts against GRPO step, 0 to 250, main run, 9-step averages. First: sampled reply length rises from 47 to about 55 tokens in the first 15 steps, then drifts up to about 57. Second: the share of worked-style replies rises from 74% to about 95% by step 25 and ends near 97%; the share of groups whose 8 replies all got the same reward falls from about 55% to about 42%. Third: KL from the reference per reply token rises from near 0 to 0.03 at step 60, 0.045 at step 100 and about 0.05 at the end. Fourth: entropy of the next-token distribution falls steadily from 0.11 to 0.03 nats per reply token.](/diagrams/t12-dynamics.svg "The main run step by step: reply length, worked-style share and zero-variance groups, KL from the reference, and entropy.")

### Longer numbers than in training

On 4-digit and 5-digit sums every model scored 0, with 6,400 samples each for the warm start and the GRPO model. The replies show why. On every one of the 200 4-digit problems, greedy, the GRPO model works exactly three columns and stops, often reading the wrong digits along the way:

```text
Compute 5243 + 5763.  (one of the 4-digit problems; the right answer is 11006)
Add the digits from right to left. 4 + 6 = 10, write 0, carry 1. 2 + 7 + 1 = 10, write 0, carry 1. 5 + 5 + 1 = 11, write 1, carry 1. Write the carry 1. The answer is 1100.
```

The warm start already did this on 178 of the 200. No training example had more than three columns, so no sampled reply ever had a fourth, and a reply GRPO never sees can't be rewarded. This is the bound Yue et al. describe in its plainest form: a 4-digit sum needs nothing new except one more repetition of a step the model already does well, and RL can't get there without a sample that takes it.

### Trying to game the checker

The `contains` checker pays for the right number anywhere in the reply, so a model could list several guesses and collect the reward. In 100 steps it didn't: its replies held 6.9 distinct numbers against the main run's 7.1, and its held-out reward stayed 2.6 to 4.0 points above its strict accuracy throughout. Listing guesses would need the model to sample that behavior first, and no warm-start example showed it.

The strict checker had a loophole instead. It reads the answer with `int()`, so "The answer is 042." counts as 42. The model found it:

```text
What is 17 + 25?  (GRPO model, greedy)
Add the digits from right to left. 7 + 5 = 12, write 2, carry 1. 1 + 2 + 1 = 4, write 4. 0 + 0 = 0, write 0. The answer is 042.
```

| Model | Greedy answers with a leading zero (500 problems) | Greedy accuracy | Same, if leading zeros failed |
|---|---|---|---|
| Warm start | 1.0% | 17.8% | 17.0% |
| GRPO main run, 250 steps | 29.8% | 77.4% | 49.6% |
| Ablations at 100 steps, strict checker | 4.2% to 11.4% | 52.0% to 56.4% | 42.6% to 52.0% |
| contains checker, 100 steps | 0.8% | 45.6% | 44.8% |
| Mixed warm start, then GRPO 250 steps | 0.0% | 37.4% | 37.4% |

The `contains` run is the control: its checker compares digit strings, "042" is not "42", and padding never took off there. The mixed run never found the loophole either, under the same strict checker, because its warm start padded only 0.6% of greedy replies and wrote no leading zero, so GRPO had nothing to reinforce. Under the strict checker, always working three columns is simpler than deciding how many a problem needs, and costs nothing, so it spread. A checker that compared strings would have failed 28 points of the main run's accuracy.

Is the working honest when the answer is right? A script re-checked every step of 1,600 sampled replies: of the GRPO model's 961 right answers, 600 had every column right, 347 had the padding above, and just 14 had a wrong step or no working. The checker graded only the answer, yet the working was almost always sound.

## Step 5: before and after

| Measure | Part 10's SFT model | Warm start | After GRPO | Mixed warm start | Mixed, after GRPO |
|---|---|---|---|---|---|
| Held-out sums, greedy (500) | 0.0% | 17.8% | 77.4% | 9.0% | 37.4% |
| Same, pass@1 / pass@64 (200 problems) | 0.0% / not run | 11.4% / 72.0% | 59.9% / 94.0% | 5.4% / 57.0% | 23.9% / 78.0% |
| 4- and 5-digit sums, best pass@k | 0% | 0% | 0% | 0% | 0% |
| Sampled reply length, tokens | 136 | 42 | 60 | 35 | 50 |
| Sampled replies that end | 29% | 100% | 100% | 100% | 100% |
| Part 10's 100 chat prompts, greedy replies that end | 33% | 99% | 100% | 31% | 24% |
| Same, median length of those that ended | 114 tokens | 55 tokens | 55 tokens | 107 tokens | 114 tokens |
| Same prompts, sampled replies that end | 39% | not kept | not kept | 47% | 36% |
| HellaSwag | 28.0 ± 0.9 | 27.5 ± 0.9 | 27.5 ± 0.9 | 28.0 ± 0.9 | 27.8 ± 0.9 |
| ARC-Easy acc | 44.1 ± 2.0 | 33.7 ± 1.9 | 34.3 ± 1.9 | 42.6 ± 2.0 | 43.6 ± 2.0 |
| PIQA acc | 58.6 ± 2.3 | 54.8 ± 2.3 | 54.7 ± 2.3 | 58.9 ± 2.2 | 59.0 ± 2.2 |

The benchmarks are [Part 9](/blog/evaluate-base-model/)'s, zero-shot, through `evals.py`, with HellaSwag scored by length-normalized accuracy; the SFT model's scores are Part 10's, and its sampled end rate comes from this part's run of Part 10's prompts. GRPO itself cost nothing: in both arms every benchmark is within its interval of its warm start's. The warm start cost a lot. ARC-Easy fell 10.4 points, five times its interval, and PIQA 3.8. Sixty-three steps at up to 3 × 10⁻⁴ on nothing but sums turned the chat model into an adder.

That shows in Part 10's fixed prompts and new arithmetic ones, greedy, quoted verbatim and trimmed:

| Prompt | Part 10's SFT model | After GRPO (sums-only warm start) |
|---|---|---|
| What is 17 + 25? | "To find 17 + 25, we can use the concept of a sequence of numbers. We start by finding the first number, 17, which is 17." (no end) | "Add the digits from right to left. 7 + 5 = 12, write 2, carry 1. 1 + 2 + 1 = 4, write 4. 0 + 0 = 0, write 0. The answer is 042." |
| What is 348 + 279? | "To find 348 + 279, we can use the concept of a ratio." (no end) | "Add the digits from right to left. 8 + 9 = 16, write 6, carry 1. 4 + 7 + 1 = 11, write 1, carry 1. 3 + 2 + 1 = 7, write 7. The answer is 716." |
| What is 25 - 17? | "To find 25 - 17, we can use the concept of a set of integers." (no end) | "Add the digits from right to left. 5 + 7 = 12, write 2, carry 1. 2 + 1 + 1 = 4, write 4. 0 + 0 = 0, write 0. The answer is 042." |
| What is the capital of France? | "The capital of France is Paris. It is a city known for its rich history, culture, and culture." (no end) | "Add the digits from right to left. 0 + 1 = 1, write 1. 0 + 0 = 0, write 0. 0 + 0 = 0, write 0. The answer is 001." |

348 + 279 is 627: the model has the procedure and still gets all three column sums wrong, the kind of slip behind the 22.6% of held-out problems it misses. It adds when asked to subtract, and answers a question about Paris by adding zeros. The chat statistics only look better than Part 10's because every reply is a 55-token sum.

### The fix: chat data in the warm-start

DeepSeek-R1 faced the same problem at scale and, for its third stage, collected "approximately 200k training samples that are unrelated to reasoning" to train on alongside about 600,000 reasoning samples. The small version: `python grpo.py warmstart --out runs/warm_mixed --n 4000 --chat 4000 --batch 128 --lr 3e-4 --warmup 5 --fused` adds the first 4,000 of Part 10's smol-smoltalk training conversations to the same 4,000 sums. Part 10 had trained on them already, so this is replay. The schedule is the same, 63 steps, now of 128 examples in micro-batches of at most 4,096 padded tokens, normalized by each step's trained tokens as in Part 10; the sums are only 7% of the 2.2 million trained tokens. It took 2.6 minutes. GRPO then ran as the main run did: 250 steps, 24.1 minutes, not resumed.

It worked for chat. Every benchmark is back within the SFT model's interval (ARC-Easy 43.6 against 44.1), and the replies to Part 10's prompts are Part 10's again, flaws included. After GRPO, greedy:

| Prompt | Mixed warm start, after GRPO |
|---|---|
| What is the capital of France? | "The capital of France is Paris. It is a city known for its historical landmarks, cultural institutions, and cultural institutions." (no end) |
| Give me three tips for sleeping better. | "Sleeping effectively is crucial for maintaining a healthy body and mind. One approach is to establish a consistent sleep schedule" (no end) |
| What is 17 + 25? | "Add the digits from right to left. 7 + 5 = 12, write 2, carry 1. 1 + 2 + 1 = 3, write 3. The answer is 32." |
| What is 25 - 17? | "Add the digits from right to left. 5 + 7 = 12, write 2, carry 1. 2 + 1 + 1 = 3, write 3. The answer is 32." |

It cost arithmetic. The mixed warm start got 9.0% greedy and 5.4% of samples right, half the sums-only warm start, and 79% of its groups of 8 had no right reply against 64%. With less signal per step, the same 250 GRPO steps took greedy accuracy to 37.4% and pass@64 from 57% to 78%, and the training reward was still rising at the end, so more steps would likely help. Greedy chat replies ended a little less often after GRPO, 24% against 31%, though on 100 prompts that gap is about one standard error. And it still adds when asked to subtract; nothing in RL on sums rewards subtraction.

## In the real world

- DeepSeekMath introduced GRPO and ran it on DeepSeekMath-Instruct 7B with a learned reward model, 64 samples per question and a KL coefficient of 0.04, taking GSM8K from 82.9% to 88.2% and MATH from 46.8% to 51.7%. Its own analysis anticipated Step 4's question: "RL enhances Maj@K's performance but not Pass@K", which the authors read as "boosting the correct response from TopK rather than the enhancement of fundamental capabilities" ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).
- DeepSeek-R1-Zero dropped the learned reward for rule-based accuracy and format checks and ran GRPO for 10,400 steps, taking AIME 2024 pass@1 from 15.6% to 77.9%. DeepSeek-R1 put that RL stage between a cold-start SFT and a second SFT, then a second RL stage of 1,700 steps at temperature 0.7, in which preference rewards joined only for the last 400 steps because more "may lead to reward hacking" ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)).
- The same paper reports what happened at smaller scale: "a 7B dense model and a 16B Mixture-of-Experts (MoE) model" "consistently failed to yield meaningful improvements" on AIME, and "as response lengths increased, these smaller models exhibited a tendency toward repetition and were unable to effectively leverage long chains of thought".
- Tülu 3 named RLVR. Its reward is α = 10 "if correct" and 0 otherwise, optimized with PPO; the final 8B run used β = 0.05 and took 65 hours on 8 GPUs. At β = 0.01 on prompts with format constraints, "RLVR overoptimizes against the verifiable reward function and does not output meaningful content" ([Lambert et al., 2024](https://arxiv.org/abs/2411.15124)). [OLMo 2](https://arxiv.org/abs/2501.00656) followed the recipe with PPO at 7B and 13B, and "For the 1B and 32B model, we performed RLVR with Group Relative Policy Optimization (GRPO)".
- Checkers now go well beyond arithmetic: unit tests for code, and sandboxed repositories for agents, where the model takes many tool-calling turns before anything is checked. [Agentic AI Part 14](/blog/training-agentic-models/) covers RL in those environments.

Compared with [Part 11](/blog/dpo-from-scratch/)'s DPO, two things changed. The data is online: every batch is sampled from the current policy, so the model learns from its own mistakes, where DPO's UltraFeedback pairs were text it would never write. And the signal comes from a checker, which can't be talked into liking longer replies but exists only for checkable questions.

## Common misconceptions

- **"GRPO can learn from any task with a checker."** It learns only from groups whose replies disagree. Part 10's model got 0 of 3,200 samples right, so every advantage was zero.
- **"A strict checker can't be gamed."** This one parsed the answer with `int()`, and after 250 steps 29.8% of greedy answers carried a leading zero, a format no training example used. Harmless here, but RL finds whatever the checker tolerates.
- **"RL only re-ranks answers the model could already sample."** That's Yue et al.'s finding for strong base models at large k. On this task, at up to 64 samples, pass@64 rose from 72% to 94%, and 45 problems the warm start never solved in 64 tries became solvable.
- **"Practice on a skill generalizes to harder versions of it."** Not past the training distribution: 4-digit sums stayed at 0%, because the model never wrote a fourth column and so was never rewarded for one.

## Key takeaways

- RLVR replaces the reward model with a program. GRPO samples a group of replies per prompt and scores each against the group's mean and standard deviation, so no value model is needed.
- A group teaches nothing when all its replies score the same. Part 10's model scored zero on addition, so it needed a warm start. One on sums alone cost 10 points of ARC-Easy and every chat reply; adding 4,000 of Part 10's conversations kept both, but halved the starting accuracy, and the same 250 GRPO steps then reached 37.4% instead of 77.4%.
- Twenty minutes and 64,000 sampled replies took held-out greedy accuracy from 17.8% to 77.4% and pass@64 from 72% to 94%, with no further cost on the benchmarks.
- RL chose between styles the model already had: worked replies went from 59% to 98%, and grew longer for that reason.
- It also found the checker's loophole, leading zeros, and never reached 4-digit sums, which needed a step it had never sampled.

## What's next

Part 13 swaps the checker for a teacher. A bigger model writes the training data, and the 88M model learns from it two ways: by copying the teacher's text (sequence distillation) and by matching its full next-token probabilities (logit distillation). Can a teacher's worked examples do what twenty minutes of RL couldn't, and carry the model past three digits?

## Further reading

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)](https://arxiv.org/abs/2402.03300): Section 4 introduces GRPO, its KL estimator and its settings; Section 5.2.2 compares Pass@K and Maj@K.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948): R1-Zero and R1's RL stages with every hyperparameter, and the small-model failures in Appendix G.
- [Understanding R1-Zero-Like Training: A Critical Perspective (Liu et al., 2025)](https://arxiv.org/abs/2503.20783): the length and difficulty biases in GRPO, and Dr. GRPO.
- [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (Yue et al., 2025)](https://arxiv.org/abs/2504.13837): pass@k at large k, before and after RLVR.
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124): Section 6, RLVR with PPO, its data and its overoptimization.
- [2 OLMo 2 Furious (OLMo team, 2024)](https://arxiv.org/abs/2501.00656): the Tülu 3 recipe applied again, with GRPO for the 1B and 32B models.
- [Evaluating Large Language Models Trained on Code (Chen et al., 2021)](https://arxiv.org/abs/2107.03374): the unbiased pass@k estimator.
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347): the clipped ratio GRPO keeps.
- [Approximating KL Divergence (John Schulman, 2020)](http://joschu.net/blog/kl-approx.html): the k3 estimator in GRPO's loss.
- [RLHF Book (Nathan Lambert)](https://rlhfbook.com/): a free book on reward models, policy gradients and RL with verifiable rewards.
