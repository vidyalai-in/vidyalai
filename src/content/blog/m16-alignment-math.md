---
title: "The Math of Alignment: Policy Gradients, PPO, DPO and GRPO"
description: "The equations behind RLHF and reasoning RL: the log-derivative trick, baselines, PPO's clip, Bradley-Terry, the KL-regularized optimum, DPO and GRPO."
slug: "alignment-math"
category: "llm-math"
tags: ["math", "training", "rlhf", "alignment", "fine-tuning"]
series: "llm-math"
seriesOrder: 16
pubDate: 2026-05-20
---

## TL;DR

Every method that turns a language model into an assistant or a reasoner maximizes the expected reward of its own answers. This article derives that quantity's gradient, the REINFORCE rule, which turns out to be cross-entropy on the model's own samples, weighted by reward. Then come a baseline, PPO's clipped ratio, Bradley-Terry preferences, the exact optimum of the KL-regularized RLHF objective, DPO derived from it, and GRPO's group baseline, each checked on a three-answer toy. A bonus proves that speculative decoding samples exactly from the big model, and a closing section looks back across all 16 parts.

## Why it matters

DeepSeek-R1-Zero went from 15.6% to 77.9% on the AIME 2024 math exam with no example solutions at all, only a program that checked final answers. The reward was a yes-or-no from a checker. You can't take the derivative of a checker, yet training needs a gradient for every one of the model's parameters.

Three lines of algebra get around it. The update R1-Zero used turns out to be next-token training, the same cross-entropy step from [Part 8](/blog/backpropagation-math/), applied to answers the model wrote itself and scaled by how much better than its siblings each one scored.

PPO, DPO and GRPO are short steps from there. PPO adds one clipping function, which lifted an average benchmark score in its paper from -0.39 to 0.82. DPO comes from solving the RLHF objective exactly. [Fundamentals Part 13](/blog/base-model-to-assistant/) and [Fundamentals Part 14](/blog/reasoning-models/) describe these methods in words; this part derives them.

## The intuition

Picture a cook whose diner only ever says "good" or "not good". The cook can't ask which pinch of salt made the difference, but can cook the praised dishes more often and the panned ones less. Repeat long enough and the menu drifts toward the diner's taste. Two refinements help: judge each verdict against the usual verdict (a baseline), and don't rewrite the whole menu after one dinner (PPO's clip and the KL leash).

The analogy breaks in two places. A model's menu is a probability distribution over trillions of answers, and every token of an answer shares one verdict, so the model never learns directly which sentence earned it. And in RLHF the diner is itself a learned model with blind spots, which the cook will find if nothing holds it back.

## Step 1: expected reward and the log-derivative trick

### A policy with three answers

In reinforcement learning, the thing being trained is called a policy, written $$\pi$$ (the Greek letter pi). For a language model, the policy is the model itself: given a prompt, it assigns a probability $$\pi(y)$$ to every possible answer $$y$$. A reward function $$r(y)$$ scores each answer.

Start with a toy policy that can give only three answers, A, B and C, with logits (raw scores) $$\mathbf{z} = [2, 1, 0]$$. Softmax from [Part 5](/blog/softmax-math/) turns them into probabilities $$\pi = [0.6652, 0.2447, 0.0900]$$. Only A is right, so the rewards are $$r = [1, 0, 0]$$.

The quantity to maximize is the expected reward, Part 4's probability-weighted average:

$$
J(\mathbf{z}) = \mathbb{E}_{y \sim \pi}\big[r(y)\big] = \sum_{y} \pi(y)\, r(y)
$$

Read $$\mathbb{E}_{y \sim \pi}$$ as "the average over answers $$y$$ drawn from the policy". Here $$J = 0.6652 \times 1 + 0.2447 \times 0 + 0.0900 \times 0 = 0.6652$$: the policy answers correctly 66.5% of the time.

### The gradient without differentiating the reward

The gradient of $$J$$ is the list of its slopes with respect to the logits ([Part 7](/blog/derivatives-gradients/)). The reward is fixed for each answer, so only the probabilities move:

$$
\nabla J = \sum_{y} r(y)\, \nabla \pi(y)
$$

The symbol $$\nabla$$ (nabla) means "the gradient with respect to the parameters". This sum runs over every possible answer, which is hopeless for a real model. The fix is the log-derivative trick. The derivative of $$\ln x$$ is $$1/x$$, so by the chain rule $$\nabla \ln \pi(y) = \nabla \pi(y) / \pi(y)$$, which rearranges to:

$$
\nabla \pi(y) = \pi(y)\, \nabla \ln \pi(y)
$$

Substitute it in, and the sum becomes an average under the policy again:

$$
\nabla J = \sum_{y} \pi(y)\, r(y)\, \nabla \ln \pi(y) = \mathbb{E}_{y \sim \pi}\big[r(y)\, \nabla \ln \pi(y)\big]
$$

That is the policy gradient, and an average can be estimated by sampling: draw answers, and multiply each one's reward by the gradient of its own log-probability. Ronald Williams named algorithms of this form REINFORCE in [1992](https://doi.org/10.1007/BF00992696), an acronym for "REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility." Nothing differentiates $$r$$, so the reward can be a checker, a human click or a reward model.

### It's cross-entropy on your own samples

For a softmax policy, the gradient of a log-probability has a short form. [Part 8](/blog/backpropagation-math/) showed that the gradient of the loss $$-\ln \pi(y)$$ with respect to the logits is $$\pi - \mathbf{e}_y$$, the probabilities minus the one-hot vector $$\mathbf{e}_y$$ that is 1 at answer $$y$$ and 0 elsewhere. Flip the sign:

$$
\nabla_{\mathbf{z}} \ln \pi(y) = \mathbf{e}_y - \pi
$$

So each REINFORCE sample contributes $$r(y)(\mathbf{e}_y - \pi)$$: the cross-entropy step you'd take if $$y$$ were the training label, scaled by its reward. Pretraining pushes up the token in the data; REINFORCE pushes up the answer the model wrote, as hard as the reward says.

```python
import numpy as np

z = np.array([2.0, 1.0, 0.0])            # logits for answers A, B, C
r = np.array([1.0, 0.0, 0.0])            # reward: only A is right
pi = np.exp(z) / np.exp(z).sum()
J = lambda z: (np.exp(z) / np.exp(z).sum()) @ r
print("pi        ", pi.round(4), "  J =", round(J(z), 4))

# exact gradient: sum over answers of pi(y) * r(y) * (onehot(y) - pi)
exact = sum(pi[y] * r[y] * (np.eye(3)[y] - pi) for y in range(3))
print("exact     ", exact.round(4))

h = 1e-5                                 # check by nudging each logit
print("nudging   ", np.array([(J(z + h * e) - J(z - h * e)) / (2 * h) for e in np.eye(3)]).round(4))

# REINFORCE: sample answers from the policy, average r * (onehot - pi)
rng = np.random.default_rng(0)
y = rng.choice(3, size=100_000, p=pi)
samples = r[y, None] * (np.eye(3)[y] - pi)
print("REINFORCE ", samples.mean(0).round(4))
```

```text
pi         [0.6652 0.2447 0.09  ]   J = 0.6652
exact      [ 0.2227 -0.1628 -0.0599]
nudging    [ 0.2227 -0.1628 -0.0599]
REINFORCE  [ 0.2226 -0.1628 -0.0599]
```

The sampled estimate matches the exact gradient to three decimals: raise A, lower B and C.

## Step 2: baselines, advantages and sequences

### Subtracting a baseline changes nothing on average

Under the policy, the gradient of the log-probability averages to zero:

$$
\mathbb{E}_{y \sim \pi}\big[\nabla \ln \pi(y)\big] = \sum_{y} \nabla \pi(y) = \nabla \sum_{y} \pi(y) = \nabla 1 = 0
$$

The first step is the log-derivative trick backward; the probabilities always add up to 1, and a constant has gradient 0. So for any number $$b$$ that doesn't depend on the sampled answer, $$\mathbb{E}[(r - b)\nabla \ln \pi] = \mathbb{E}[r \nabla \ln \pi]$$: subtracting a baseline leaves the expected gradient alone. The difference $$A = r - b$$ is the advantage, how much better than expected an answer did.

### But it changes the noise

Baselines matter because a reward model's scores have no natural zero (InstructGPT shifted its reward model so demonstrations average 0). Shift every reward by the same amount and the true gradient doesn't move, but the sampled estimate gets much noisier:

```python
import numpy as np

z = np.array([2.0, 1.0, 0.0])
pi = np.exp(z) / np.exp(z).sum()
rng = np.random.default_rng(0)
y = rng.choice(3, size=100_000, p=pi)
score = np.eye(3)[y] - pi                        # grad log pi(y), one row per sample

print("average grad log pi:", score.mean(0).round(3) + 0.0)
for r in [np.array([1.0, 0, 0]), np.array([3.0, 2, 2]), np.array([-1.0, -2, -2])]:
    for b in [0.0, pi @ r]:                      # no baseline, then the mean reward
        g = (r[y] - b)[:, None] * score          # advantage * grad log pi
        print(f"rewards {r}  b = {b:6.3f}  mean {g.mean(0).round(3)}  variance {g.var(0).sum():.3f}")
```

```text
average grad log pi: [ 0.     0.001 -0.001]
rewards [1. 0. 0.]  b =  0.000  mean [ 0.223 -0.163 -0.06 ]  variance 0.040
rewards [1. 0. 0.]  b =  0.665  mean [ 0.223 -0.164 -0.059]  variance 0.097
rewards [3. 2. 2.]  b =  0.000  mean [ 0.222 -0.16  -0.062]  variance 2.477
rewards [3. 2. 2.]  b =  2.665  mean [ 0.223 -0.164 -0.059]  variance 0.097
rewards [-1. -2. -2.]  b =  0.000  mean [ 0.223 -0.165 -0.057]  variance 1.517
rewards [-1. -2. -2.]  b = -1.335  mean [ 0.223 -0.164 -0.059]  variance 0.097
```

All six rows estimate the same gradient. Without a baseline, adding 2 to every reward makes the noise 62 times larger (2.477 against 0.040), because now every sampled answer gets pushed up, wrong ones included, and only the imbalance carries the signal. With the mean reward as baseline, the variance is 0.097 whatever the offset.

The mean isn't the best baseline: on rewards [1, 0, 0] no baseline beats it, and the exact variance-minimizing constant is 0.245, with variance 0.011. Williams noted in 1992, citing Dayan, that this optimum is "not the mean reinforcement" but "another quantity that is more difficult to estimate effectively." The mean is cheap and offset-proof.

### From one answer to a sequence of tokens

A real answer is a sequence of tokens $$y_1, \ldots, y_T$$. Part 4's chain rule writes its probability as a product of next-token probabilities, so its log-probability is a sum:

$$
\ln \pi(y) = \sum_{t=1}^{T} \ln \pi(y_t \mid y_{<t})
$$

Here $$y_{<t}$$ means all tokens before position $$t$$. Take the gradient of both sides, and the REINFORCE update for a whole answer is the sum of per-token updates, each scaled by the same advantage:

$$
A \, \nabla \ln \pi(y) = \sum_{t=1}^{T} A \, \nabla \ln \pi(y_t \mid y_{<t})
$$

With one reward at the end, every token of a good answer gets the same push, filler included. PPO in RLHF tackles this credit-assignment problem with a value model, a second network that predicts the expected reward from each partial answer so each token gets its own advantage. GRPO, in Step 5, uses one advantage per answer.

## Step 3: PPO, reusing samples with a clipped ratio

### The importance ratio

Generating answers is the expensive part of RL for LLMs, so you'd like several gradient steps per batch. After the first step, though, the samples come from an older policy $$\pi_{\text{old}}$$. Importance sampling corrects for that. For any function $$f$$:

$$
\mathbb{E}_{y \sim \pi_\theta}[f(y)] = \sum_y \pi_{\text{old}}(y) \frac{\pi_\theta(y)}{\pi_{\text{old}}(y)} f(y) = \mathbb{E}_{y \sim \pi_{\text{old}}}\big[\rho(y)\, f(y)\big]
$$

Here $$\theta$$ (theta) stands for the current parameters, and the importance ratio $$\rho = \pi_\theta / \pi_{\text{old}}$$ (rho) reweights each old sample. The PPO paper ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) maximizes the surrogate objective $$\mathbb{E}_{\pi_{\text{old}}}[\rho A]$$. At $$\theta = \theta_{\text{old}}$$ the ratio is 1 and its gradient is $$\nabla \pi_\theta / \pi_{\text{old}} = \nabla \ln \pi_\theta$$, so the first step is exactly REINFORCE.

### The clip

Maximized freely, the surrogate pushes $$\rho$$ as far as it can, out of the region where old samples say anything useful. PPO clips it, with $$\epsilon$$ (epsilon) a small number:

$$
L^{\text{CLIP}} = \mathbb{E}\Big[\min\big(\rho A,\ \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\, A\big)\Big]
$$

The clip function holds $$\rho$$ inside $$[1 - \epsilon, 1 + \epsilon]$$; the paper suggests "say, ε = 0.2". The minimum picks whichever version is worse for you. With $$\epsilon = 0.2$$ and advantages of +1 and -1, PyTorch's autograd gives:

| Ratio ρ | Objective, A = +1 | Slope, A = +1 | Objective, A = -1 | Slope, A = -1 |
|---|---|---|---|---|
| 0.5 | 0.50 | 1 | -0.80 | 0 |
| 0.7 | 0.70 | 1 | -0.80 | 0 |
| 0.9 | 0.90 | 1 | -0.90 | -1 |
| 1.0 | 1.00 | 1 | -1.00 | -1 |
| 1.1 | 1.10 | 1 | -1.10 | -1 |
| 1.3 | 1.20 | 0 | -1.30 | -1 |
| 1.5 | 1.20 | 0 | -1.50 | -1 |

A good answer (A = +1) gets pushed up until its probability is 1.2 times the old one; then the slope drops to 0. A bad answer gets pushed down until its ratio reaches 0.8. The cut-off applies only in the direction the update wants to go: if a step made a good answer less likely (ratio 0.5) or a bad one more likely (ratio 1.5), the slope stays and the mistake gets corrected. The paper calls the result "a lower bound (i.e., a pessimistic bound) on the unclipped objective."

On the paper's 7 simulated robot tasks, ε = 0.1, 0.2 and 0.3 averaged normalized scores of 0.76, 0.82 and 0.70, against -0.39 with no clipping. For an LLM, the ratio is computed per token.

## Step 4: preferences and the KL-regularized optimum

### Bradley-Terry is a two-answer softmax

RLHF's reward comes from comparisons. The Bradley-Terry model says that if two answers have scores $$r_w$$ (the winner) and $$r_l$$ (the loser), the chance people prefer the winner is a softmax over just the two:

$$
P(y_w \succ y_l) = \frac{e^{r_w}}{e^{r_w} + e^{r_l}} = \frac{1}{1 + e^{-(r_w - r_l)}} = \sigma(r_w - r_l)
$$

The middle step divides top and bottom by $$e^{r_w}$$; $$\succ$$ reads "is preferred to", and $$\sigma$$ is Part 7's sigmoid. [Christiano et al. (2017)](https://arxiv.org/abs/1706.03741) likened it to chess's Elo ratings, where a rating gap sets the odds of winning. InstructGPT ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)) trained its reward model with the loss $$-\ln \sigma(r_w - r_l)$$, whose slope in the gap $$d = r_w - r_l$$ is $$\sigma(d) - 1$$: predicted probability minus target, Part 8's $$p - y$$ once more.

### Solving the RLHF objective exactly

RLHF then maximizes the reward with a leash, a KL divergence from [Part 6](/blog/cross-entropy-perplexity/) that charges for drifting from a frozen reference policy $$\pi_{\text{ref}}$$, usually the supervised fine-tuned model:

$$
\max_{\pi}\ \mathbb{E}_{y \sim \pi}[r(y)] - \beta\, D_{\mathrm{KL}}(\pi \,\|\, \pi_{\text{ref}})
$$

The number $$\beta$$ (beta) sets the price of drift; Part 6 noted this is the reverse direction of KL. The problem has an exact answer, using nothing beyond Part 6. Write both terms as one average over $$y \sim \pi$$ and divide by $$-\beta$$, which turns the maximum into a minimum:

$$
\min_{\pi}\ \mathbb{E}_{y \sim \pi}\left[\ln \frac{\pi(y)}{\pi_{\text{ref}}(y)} - \frac{r(y)}{\beta}\right]
$$

Now define a new distribution by reweighting the reference with the exponential of the reward:

$$
\pi^*(y) = \frac{1}{Z}\, \pi_{\text{ref}}(y)\, e^{r(y)/\beta}, \qquad Z = \sum_y \pi_{\text{ref}}(y)\, e^{r(y)/\beta}
$$

The constant $$Z$$, called the partition function, makes the probabilities add up to 1. Take the log of the definition and rearrange: $$\ln \pi_{\text{ref}}(y) + r(y)/\beta = \ln \pi^*(y) + \ln Z$$. Substituting into the bracket gives

$$
\min_{\pi}\ \Big[ D_{\mathrm{KL}}(\pi \,\|\, \pi^*) - \ln Z \Big]
$$

The constant $$Z$$ doesn't depend on $$\pi$$, and Gibbs' inequality from Part 6 says a KL divergence is never negative and is zero only when the two distributions match. So the best policy is $$\pi = \pi^*$$, and the best achievable objective is $$\beta \ln Z$$. This is the derivation in Appendix A.1 of the [DPO paper](https://arxiv.org/abs/2305.18290).

Check it on [Fundamentals Part 13](/blog/base-model-to-assistant/)'s example: a reference of [0.6, 0.3, 0.1], rewards [0, 1, 2] and $$\beta = 0.5$$. The script computes the formula, then finds the optimum by gradient ascent:

```python
import torch
torch.set_default_dtype(torch.float64)

ref = torch.tensor([0.6, 0.3, 0.1])        # reference (SFT) policy over answers A, B, C
r = torch.tensor([0.0, 1.0, 2.0])          # rewards
beta = 0.5

# 1. The formula: pi* = ref * exp(r / beta) / Z
w = ref * torch.exp(r / beta)
Z = w.sum()
print("formula   ", (w / Z).numpy().round(4), " beta*log Z =", round((beta * torch.log(Z)).item(), 4))

# 2. RLHF: gradient ascent on  E[r] - beta * KL(pi || ref)
z = torch.zeros(3, requires_grad=True)     # start from uniform logits
opt = torch.optim.SGD([z], lr=0.5)
for step in range(5000):
    pi = torch.softmax(z, 0)
    objective = (pi * r).sum() - beta * (pi * torch.log(pi / ref)).sum()
    opt.zero_grad(); (-objective).backward(); opt.step()
print("RLHF      ", torch.softmax(z, 0).detach().numpy().round(4), " objective =", round(objective.item(), 4))

# 3. DPO: minimize the DPO loss on preferences drawn from Bradley-Terry with the same r
z = torch.zeros(3, requires_grad=True)
opt = torch.optim.SGD([z], lr=0.5)
pairs = [(0, 1), (0, 2), (1, 2)]
for step in range(5000):
    h = beta * (torch.log_softmax(z, 0) - torch.log(ref))      # implicit reward
    loss = 0
    for a, b in pairs:
        p_ab = torch.sigmoid(r[a] - r[b])                       # how often people prefer a over b
        loss = loss - p_ab * torch.nn.functional.logsigmoid(h[a] - h[b]) \
                    - (1 - p_ab) * torch.nn.functional.logsigmoid(h[b] - h[a])
    opt.zero_grad(); loss.backward(); opt.step()
print("DPO       ", torch.softmax(z, 0).detach().numpy().round(4), " implicit reward - h[A] =", (h - h[0]).detach().numpy().round(4))
```

```text
formula    [0.0725 0.2678 0.6597]  beta*log Z = 1.0567
RLHF       [0.0725 0.2678 0.6597]  objective = 1.0567
DPO        [0.0725 0.2678 0.6597]  implicit reward - h[A] = [0. 1. 2.]
```

Gradient ascent lands on the formula's policy and on the predicted best value, $$\beta \ln Z = 1.0567$$. Step 5 explains the DPO line.

### β is a temperature

Take logs of the optimum: $$\ln \pi^*(y) = \ln \pi_{\text{ref}}(y) + r(y)/\beta - \ln Z$$. The ideal RLHF policy is a softmax whose logits are the reference's log-probabilities plus the reward divided by $$\beta$$: [Part 4](/blog/probability-language-models/)'s temperature, with the reward as the scores. A large $$\beta$$ leaves the reference almost unchanged; as $$\beta \to 0$$, all probability piles onto the top-reward answer, the greedy limit.

## Step 5: DPO and GRPO

### DPO: the reward hides in the policy

The formula for $$\pi^*$$ can be run backward. Solve it for the reward:

$$
r(y) = \beta \ln \frac{\pi^*(y)}{\pi_{\text{ref}}(y)} + \beta \ln Z
$$

The constant $$Z$$ is a sum over every possible answer, impossible to compute for a language model. But Bradley-Terry only ever uses a difference of two rewards for the same prompt, and the $$\beta \ln Z$$ terms cancel in the subtraction. Replace $$\pi^*$$ with the policy being trained, $$\pi_\theta$$, put it into the reward model's loss, and you have the DPO loss of [Rafailov et al. (2023)](https://arxiv.org/abs/2305.18290):

$$
L_{\text{DPO}} = -\ln \sigma\big(h(y_w) - h(y_l)\big), \qquad h(y) = \beta \ln \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}
$$

The quantity $$h(y)$$ is the implicit reward. The DPO loss is the reward-model loss after a change of variables, so fitting it lands on the same $$\pi^*$$ as fitting a reward model and then solving RLHF.

The script's third line checks this, feeding DPO the preferences a Bradley-Terry crowd with rewards [0, 1, 2] would produce, in exact proportions over all three pairs. DPO sees only how often each answer won, yet it recovers the RLHF policy [0.0725, 0.2678, 0.6597], with implicit rewards, measured from A, of exactly [0, 1, 2]. That needs preferences that truly follow Bradley-Terry and data that covers the answers; with a finite set of pairs, DPO can move answers it never saw in odd directions, a problem variants like IPO address.

![A flowchart in two columns. Left: maximize the expected reward leads to REINFORCE, reward times the gradient of log pi, which leads to subtracting a baseline, A equals r minus b. That branches to PPO, reuse samples with a clipped ratio, and to GRPO, the group mean is the baseline; PPO also feeds GRPO. Right: Bradley-Terry, probability of winning equals sigma of the gap, leads to a reward model from pairs, then the RLHF objective, reward minus beta times KL, then the exact optimum, pi_ref times e to the r over beta, normalized, then DPO, solve for r and plug into Bradley-Terry, which also takes an arrow straight from Bradley-Terry. A dotted arrow labeled optimized with runs from the reward model to PPO.](/diagrams/m16-alignment-family.svg "Two roads from one objective: policy gradients on the left, the closed-form optimum on the right.")
### GRPO: the group is the baseline

GRPO, from the DeepSeekMath paper ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)), samples a group of $$G$$ answers to the same prompt, scores them all, and uses

$$
A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}
$$

for every token of answer $$i$$, where std is the standard deviation. In Step 2's terms, the group mean is a sampled estimate of the expected reward, $$J$$ for this prompt: a baseline with no value model to train.

Take a group of 6 answers with an R1-style reward, 1 point for a correct answer plus 1 for the right format (illustrative scores): [2, 1, 0, 2, 1, 1]. The mean is 1.167 and the population standard deviation 0.687, so the advantages are +1.213 for the two perfect answers, -0.243 for the three with one point and -1.698 for the one with none.

Step 2 required a baseline independent of the sampled answer, and the group mean includes the answer's own reward. But $$r_i - \text{mean} = \frac{G-1}{G}(r_i - \bar{r}_{-i})$$, where $$\bar{r}_{-i}$$ is the mean of the other $$G - 1$$ rewards. For the first answer, $$2 - 1.167 = 0.833$$ and $$\frac{5}{6}(2 - 1) = 0.833$$. So it's a legitimate leave-one-out baseline, shrunk by a constant. The division by the standard deviation is another matter: [Liu et al. (2025)](https://arxiv.org/abs/2503.20783) show it gives too-easy and too-hard questions more weight, a "question-level difficulty bias". Averaging each answer's token terms over its own length adds a "response-level length bias" that pushes long wrong answers down less per token. Their fix, Dr. GRPO, drops both divisions.

The full GRPO objective wraps this advantage in PPO's clip, per token, and subtracts $$\beta$$ times a KL estimate. For a sampled token, with $$x = \pi_{\text{ref}} / \pi_\theta$$ for that token, the paper uses

$$
k_3 = x - 1 - \ln x
$$

from a [2020 note](http://joschu.net/blog/kl-approx.html) by John Schulman. It's never negative, by the inequality $$\ln x \le x - 1$$ that proved Gibbs' inequality in Part 6. Its average under the policy is exactly $$D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$$: $$\mathbb{E}_{\pi_\theta}[x] = \sum \pi_{\text{ref}} = 1$$ cancels the $$-1$$, and $$-\ln x$$ averages to the KL. The plain estimator $$k_1 = -\ln x$$ has the same average but swings negative:

```python
import numpy as np

ref = np.array([0.6, 0.3, 0.1])        # reference policy
pi = np.array([0.55, 0.32, 0.13])      # current policy, a small step away
print(f"exact KL(pi || ref) = {(pi * np.log(pi / ref)).sum():.4f}")

rng = np.random.default_rng(0)
y = rng.choice(3, size=100_000, p=pi)  # answers sampled from the policy
x = ref[y] / pi[y]
for name, k in [("k1 = -log x", -np.log(x)), ("k3 = x - 1 - log x", x - 1 - np.log(x))]:
    print(f"{name:20} mean {k.mean():.4f}  std {k.std():.4f}  smallest {k.min():+.4f}")
```

```text
exact KL(pi || ref) = 0.0069
k1 = -log x          mean 0.0064  std 0.1195  smallest -0.0870
k3 = x - 1 - log x   mean 0.0068  std 0.0095  smallest +0.0020
```

Both average close to 0.0069, but $$k_3$$'s spread is 12 times smaller. Differentiated sample by sample as a loss, $$k_3$$'s expected gradient is $$\sum_y (\pi_\theta - \pi_{\text{ref}}) \nabla \ln \pi_\theta$$, the gradient of the forward KL $$D_{\mathrm{KL}}(\pi_{\text{ref}} \,\|\, \pi_\theta)$$. Near the reference the two directions nearly agree: on this toy, [-0.050, 0.020, 0.030] against the reverse KL's [-0.052, 0.018, 0.033].

## Bonus: speculative decoding is rejection sampling

### Plain rejection sampling

Rejection sampling draws from a target $$p$$ using proposals from an easier $$q$$. Pick a constant $$M$$ with $$p(x) \le M q(x)$$ everywhere, propose $$x$$ from $$q$$, and keep it with probability $$p(x) / (M q(x))$$. Proposing and keeping $$x$$ has probability $$q(x) \cdot p(x)/(M q(x)) = p(x)/M$$, proportional to $$p(x)$$, so kept samples follow $$p$$ exactly, but only a fraction $$1/M$$ is kept.

Step 4's optimum can be sampled this way with no training: propose from $$\pi_{\text{ref}}$$ and keep answer $$y$$ with probability $$e^{(r(y) - r_{\max})/\beta}$$:

```python
import numpy as np
rng = np.random.default_rng(0)

ref = np.array([0.6, 0.3, 0.1])        # propose answers from the reference policy
r = np.array([0.0, 1.0, 2.0])
beta = 0.5
accept = np.exp((r - r.max()) / beta)  # keep answer y with probability e^((r(y) - r_max)/beta)

y = rng.choice(3, size=1_000_000, p=ref)
kept = y[rng.random(y.size) < accept[y]]
print("acceptance rate", round(kept.size / y.size, 4))
print("kept answers   ", (np.bincount(kept) / kept.size).round(4))
```

```text
acceptance rate 0.1516
kept answers    [0.0719 0.2664 0.6617]
```

The kept answers follow $$\pi^* = [0.0725, 0.2678, 0.6597]$$ up to sampling noise, at an acceptance rate of $$Z / e^{r_{\max}/\beta} = 8.2765 / e^4 = 0.1516$$. A policy trained to $$\pi^*$$ skips the 85% of proposals this method throws away.

### Speculative sampling keeps its rejects

[Inference Part 9](/blog/speculative-decoding/) gave the rule (accept a drafted token $$x$$ with probability $$\min(1, p(x)/q(x))$$, else draw from $$\text{norm}(\max(0, p - q))$$) and the one-token calculation. Two steps in that calculation deserve spelling out, and the rule turns out to be the best possible.

The acceptance probability is $$\alpha = \sum_x q(x) \min(1, p(x)/q(x)) = \sum_x \min(p(x), q(x))$$. The leftover mass is

$$
\sum_x \max(0, p(x) - q(x)) = \sum_x \big(p(x) - \min(p(x), q(x))\big) = 1 - \alpha
$$

because $$\max(0, p - q) = p - \min(p, q)$$ for any two numbers. So the residual, normalized, is $$\max(0, p - q)/(1 - \alpha)$$, and a rejection (probability $$1 - \alpha$$) followed by a residual draw outputs $$x$$ with probability exactly $$\max(0, p(x) - q(x))$$. Add the accepted part, $$\min(p(x), q(x))$$, and you get $$p(x)$$. That is Appendix A.1 of [Leviathan et al. (2022)](https://arxiv.org/abs/2211.17192); [Chen et al. (2023)](https://arxiv.org/abs/2302.01318) prove the same as their Theorem 1.

Plain rejection sampling would need $$M = \max_x p(x)/q(x)$$. On Inference Part 9's example, $$p = [0.5, 0.2, 0.3]$$ and $$q = [0.3, 0.5, 0.2]$$, that's $$M = 5/3$$: 60% of proposals survive and the rest are thrown away. Speculative sampling accepts 70% and turns every rejection into a correct token, never retrying (Leviathan et al. compare the two in Appendix A.2).

No exact scheme can keep the draft more often: if the output $$X$$ follows $$p$$ and the draft $$Y$$ follows $$q$$, then $$P(X = Y = x)$$ can exceed neither $$p(x)$$ nor $$q(x)$$, so the draft survives with probability at most $$\sum_x \min(p(x), q(x)) = \alpha$$. That's one minus the total variation distance, half the sum of $$|p - q|$$, as Leviathan et al.'s Lemma 3.3 shows.

### The whole sequence

Inference Part 9 says the argument "repeats position by position", and the chain rule makes that precise. The first output token follows $$p(x_1)$$. Given it, however it was produced, the next is decided by the same rule on $$p(\cdot \mid x_1)$$ and $$q(\cdot \mid x_1)$$, so the pair follows $$p(x_1)\,p(x_2 \mid x_1)$$. A simulation with two drafts per step and a second token that depends on the first:

```python
import numpy as np
rng = np.random.default_rng(0)

# Target p and draft q over 3 tokens; the second token's distribution depends on the first.
p1 = np.array([0.5, 0.2, 0.3]);  q1 = np.array([0.3, 0.5, 0.2])
p2 = np.array([[0.1, 0.6, 0.3], [0.7, 0.2, 0.1], [0.3, 0.3, 0.4]])   # row = first token
q2 = np.array([[0.4, 0.4, 0.2], [0.2, 0.2, 0.6], [0.3, 0.6, 0.1]])
P = lambda out: p1 if not out else p2[out[0]]
Q = lambda out: q1 if not out else q2[out[0]]

def two_tokens():
    out = []
    while len(out) < 2:
        drafts = []
        for _ in range(2 - len(out)):                  # draft the tokens still missing
            drafts.append(rng.choice(3, p=Q(out + drafts)))
        for x in drafts:
            p, q = P(out), Q(out)
            if rng.random() < min(1, p[x] / q[x]):     # accept with prob min(1, p/q)
                out.append(x)
            else:                                      # reject: draw from norm(max(0, p - q))
                res = np.maximum(p - q, 0)
                out.append(rng.choice(3, p=res / res.sum()))
                break
    return out

counts = np.zeros((3, 3))
for _ in range(200_000):
    a, b = two_tokens()
    counts[a, b] += 1
target = p1[:, None] * p2                              # chain rule: p(x1) p(x2 | x1)
print("target:\n", target.round(3))
print("speculative:\n", (counts / 200_000).round(3))
print("largest gap:", np.abs(counts / 200_000 - target).max().round(4))
```

```text
target:
 [[0.05 0.3  0.15]
 [0.14 0.04 0.02]
 [0.09 0.09 0.12]]
speculative:
 [[0.051 0.299 0.151]
 [0.14  0.04  0.021]
 [0.09  0.089 0.12 ]]
largest gap: 0.0009
```

All nine pairs match the target's joint distribution to within 0.0009, about one standard error at 200,000 runs.

## In the real world

- InstructGPT framed RLHF as "a bandit environment which presents a random customer prompt and expects a response to the prompt": one answer per episode, Step 1's setting. It added "a per-token KL penalty from the SFT model" with $$\beta = 0.02$$ ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).
- The DPO paper used $$\beta = 0.1$$ by default and 0.5 for summarization, and its appendix implements the loss as `-F.logsigmoid(beta * (pi_logratios - ref_logratios))` ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).
- DeepSeekMath's GRPO run sampled 64 outputs per question with a KL coefficient of 0.04 and a learned reward model. The policy had "only a single update following each exploration stage," so every ratio sat at 1 and the clip never acted ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).
- DeepSeek-R1-Zero used rule-based rewards, accuracy plus format "with the same weight," 16 outputs per question and a KL coefficient of 0.001. Each rollout of 8,192 outputs was split into 16 mini-batches, so there the ratios did move. For R1's first RL stage the paper set $$\epsilon$$ to 10, since "a lower value can lead to the truncation of gradients for a significant number of tokens" ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)).
- Liu et al. found that all the popular open-source PPO implementations they examined divide the loss by response length, which "misaligns with the PPO objective" ([Liu et al., 2025](https://arxiv.org/abs/2503.20783)).

## Common misconceptions

- **"Reinforcement learning on LLMs needs a differentiable reward."** The log-derivative trick moves the gradient onto $$\ln \pi$$, so the reward can be a unit test, a regex or a person.
- **"Subtracting a baseline biases the gradient."** The average of $$\nabla \ln \pi$$ under the policy is zero, so a baseline that doesn't depend on the sampled answer only changes the noise: 25 times less on the toy after a reward shift of 2.
- **"PPO's clip keeps every probability within 20% of the old one."** It only removes the incentive to go further. Updates on other tokens can still push a ratio past 1.2.
- **"DPO optimizes something different from RLHF."** Under Bradley-Terry preferences both have the same optimum, which the toy recovered to four decimals. They differ in practice because DPO trains on a fixed set of pairs and never samples from its own policy.
- **"GRPO's KL term is the KL divergence."** It's a one-sample estimate that averages to the KL and can't go negative, and its gradient follows the forward direction.

## Key takeaways

- RL fine-tuning maximizes expected reward. Its gradient is reward times the gradient of the answer's log-probability, averaged: cross-entropy on the model's own samples, scaled by reward.
- A baseline changes nothing on average and can cut the noise many times over. GRPO's baseline is the mean reward of a group of answers to the same prompt.
- PPO reuses old samples through the ratio of new to old probability, clipped at 0.8 and 1.2, so the push stops once an answer has moved far enough.
- RLHF with a KL leash has an exact optimum: the reference's probabilities times e to the reward over beta, normalized. Solve it for the reward, put that into Bradley-Terry, and the normalizer cancels. That's DPO.
- Speculative decoding is rejection sampling that recycles its rejects, which is why its output follows the big model exactly.

## The series in review

Sixteen parts, six toolkits. Each row names the idea from those parts that came back most often.

| Theme | Parts | The idea to keep |
|---|---|---|
| Linear algebra | [Part 1](/blog/math-for-llms/), [Part 2](/blog/vectors-dot-products/), [Part 3](/blog/matrices-linear-maps/), [Part 13](/blog/low-rank-svd/) | A model is mostly matrix multiplication, which is many dot products; a matrix is a map, and the SVD shows how much of it matters |
| Probability and information | [Part 4](/blog/probability-language-models/), [Part 5](/blog/softmax-math/), [Part 6](/blog/cross-entropy-perplexity/) | Text is a chain of next-token probabilities; softmax makes them from scores; cross-entropy and KL divergence score them |
| Calculus and training | [Part 7](/blog/derivatives-gradients/), [Part 8](/blog/backpropagation-math/), [Part 9](/blog/optimizers-adam/), [Part 10](/blog/initialization-normalization/) | Gradients point uphill, backpropagation gets all of them for about twice the forward cost, AdamW takes the steps, and normalization and residuals keep signals stable |
| The transformer's own math | [Part 11](/blog/attention-math/), [Part 12](/blog/rope-math/) | Attention is a scaled, masked softmax over dot products; RoPE encodes position as rotation, so scores depend on relative distance |
| Numbers and counting | [Part 14](/blog/floating-point-math/), [Part 15](/blog/counting-flops-scaling-laws/) | Floating-point formats trade range for precision; parameters, FLOPs and memory can be counted, and scaling laws turn those counts into predictions |
| Alignment | Part 16 | Expected reward, its gradient, and the KL-regularized optimum behind PPO, DPO and GRPO |

One gradient showed up in nearly every part after Part 7: probability minus target, $$p - y$$. It trains the next-token predictor in Part 8, the reward model as $$\sigma(d) - 1$$ here, and, with the sign flipped and scaled by an advantage, the policy in RLHF and GRPO.

## What's next

That's the end of The Math Behind LLMs, and the full reading order is on the [series page](/series/llm-math/). To watch this math run on hardware, the [GPU series](/series/llm-gpu/) starts with [From torch.matmul to Silicon](/blog/gpu-from-python-to-silicon/) and follows one matrix multiply down to the chip. The [Inference series](/series/llm-inference/) begins with [How LLM Inference Works](/blog/how-llm-inference-works/): why a model doing billions of these operations per token mostly waits on memory.

## Further reading

- [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (Williams, 1992)](https://doi.org/10.1007/BF00992696): the REINFORCE family, its gradient theorem and the reinforcement baseline.
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347): the clipped surrogate objective, with Figure 1 plotting one clipped term for positive and negative advantages.
- [Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741): the Bradley-Terry reward model trained from comparisons.
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155): InstructGPT's reward-model loss and its PPO objective with a per-token KL penalty.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290): Appendix A derives the KL-regularized optimum and the DPO loss step by step.
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)](https://arxiv.org/abs/2402.03300): Section 4.1 introduces GRPO and its KL estimator.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948): GRPO with rule-based rewards at scale, with every hyperparameter listed.
- [Understanding R1-Zero-Like Training: A Critical Perspective (Liu et al., 2025)](https://arxiv.org/abs/2503.20783): the length and difficulty biases in GRPO, and Dr. GRPO.
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192): the acceptance rule, its correctness proof and the comparison with rejection sampling.
- [Accelerating Large Language Model Decoding with Speculative Sampling (Chen et al., 2023)](https://arxiv.org/abs/2302.01318): the same rule, found independently, with its proof as Theorem 1.
- [Approximating KL Divergence (John Schulman, 2020)](http://joschu.net/blog/kl-approx.html): the k1, k2 and k3 estimators and why k3 is unbiased and never negative.
- [Reinforcement Learning: An Introduction, chapter 13 (Sutton and Barto)](http://incompleteideas.net/book/the-book-2nd.html): the textbook treatment of policy gradients and baselines.
- [RLHF Book (Nathan Lambert)](https://rlhfbook.com/): a free, detailed book on reward models, PPO, DPO and RL with verifiable rewards for language models.
