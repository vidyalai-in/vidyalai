---
title: "Softmax and Log-Sum-Exp: From Scores to Probabilities, Safely"
description: "Why softmax uses the exponential, how one subtraction stops it overflowing, what log-sum-exp measures, and the online softmax inside FlashAttention."
slug: "softmax-math"
category: "llm-math"
tags: ["math", "attention", "sampling", "training", "gpu"]
series: "llm-math"
seriesOrder: 5
pubDate: 2026-04-24
---

## TL;DR

Softmax turns scores into probabilities: raise $$e$$ to each score and divide by the total. The exponential is the only smooth choice for which a fixed gap between two scores always means the same odds; a gap of 1 means 2.718 times as likely. Because only gaps matter, you can subtract the largest score first, which is how every framework keeps softmax from overflowing. The log of the total, log-sum-exp, acts as a smooth maximum and gives stable log-probabilities. And a running maximum plus a running total let you compute softmax block by block, the recurrence at the heart of FlashAttention.

## Why it matters

Take the 151,936 scores Qwen2.5-1.5B-Instruct produces for the word after "I opened the fridge and took out a". Feed them to softmax written exactly as the textbook formula, in 16-bit floats, and not one usable probability comes back. The 438 highest come out as `nan`, "not a number", and the other 151,498 as exactly 0.

The top score is only 18.44. But $$e^{18.44}$$ is about 102 million, and the largest number a 16-bit float can hold is 65,504. The 438 scores above 11.09 overflow to infinity, the total becomes infinity, and dividing by it wrecks everything.

The fix is a single subtraction, and it's exact. The same idea, pushed one step further, lets a GPU compute attention over a long context without ever holding a full row of attention scores in memory. [Part 4](/blog/probability-language-models/) used softmax as a black box. This part opens it.

## The intuition

Chess ratings work like softmax. The Elo system turns the gap between two players' ratings into expected results: every 400 points of advantage makes the stronger player's expected score ten times the weaker one's. Only the gap matters, so a 1,600 player facing a 1,200 player has the same prospects as a 2,400 facing a 2,000. And gaps compound: 800 points means 10 × 10 = 100 times.

Softmax is the same rule with a different scale. Every point of score advantage multiplies a token's odds by $$e \approx 2.718$$, and the contest runs among all 151,936 tokens at once.

The analogy breaks in two places. Elo ratings persist from game to game; a model's scores are computed fresh at every position and mean nothing on their own. And Elo compares two players at a time, while softmax shares one pot of probability among everyone, so a strong new candidate lowers every other token's probability, though not the ratios between them.

## Step 1: deriving softmax

### What a scores-to-probabilities map needs

A model's last layer produces one raw score per token, a logit, which can be any number ([Fundamentals Part 4](/blog/embeddings-and-logits/) shows where they come from). Turning logits into a probability distribution needs four things:

1. Every probability is positive. No token should be ruled out completely.
2. They add up to 1.
3. Order is kept: a higher score always means a higher probability.
4. Small changes in a score cause small, smooth changes in the probabilities, because training adjusts scores in tiny steps.

### Three recipes that fail

Try some obvious recipes on three scores, 1, -1 and 0.5:

| Recipe | Score 1 | Score -1 | Score 0.5 | What goes wrong |
|---|---|---|---|---|
| Divide by the total | 2.000 | -2.000 | 1.000 | negative, and bigger than 1 |
| Square, then divide | 0.444 | 0.444 | 0.111 | the lowest score ties the highest |
| Subtract the lowest, then divide | 0.571 | 0.000 | 0.429 | the lowest token always gets 0 |
| Raise e to each, then divide | 0.574 | 0.078 | 0.348 | nothing |

The last row is softmax: $$e^1 = 2.718$$, $$e^{-1} = 0.368$$ and $$e^{0.5} = 1.649$$, which total 4.735, and each one divided by 4.735 gives its row entry. Exponentials are always positive, they grow with the score, they're smooth, and dividing by the total makes them add up to 1.

### Why the exponential, and not something else

Plenty of positive, increasing, smooth functions would pass that table. One more requirement picks out the exponential: the odds between two tokens should depend only on the gap between their scores, as in Elo.

Write the recipe with an unknown positive function $$f$$: token $$i$$ gets probability $$f(z_i)$$ divided by the total of all the $$f$$ values, where $$z_i$$ is its score. The total cancels in a ratio, so the odds of token $$a$$ against token $$b$$ are $$f(z_a) / f(z_b)$$. Asking that this depend only on the gap means that for any score $$x$$ and gap $$c$$:

$$
\frac{f(x + c)}{f(x)} = g(c)
$$

for some function $$g$$ of the gap alone. Now step up by $$c$$ and then by $$d$$. The two ratios multiply, so

$$
g(c + d) = g(c)\, g(d)
$$

In words, adding gaps multiplies odds. The continuous functions that turn addition into multiplication are exactly the exponentials, $$g(c) = e^{\beta c}$$ for some constant $$\beta$$, and then $$f(x) = f(0)\, e^{\beta x}$$. The constant $$f(0)$$ cancels when you divide by the total, and $$\beta$$ only sets the scale, the knob Part 4 called temperature ($$\beta = 1/T$$). With $$\beta = 1$$:

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Here $$\mathbf{z}$$ is the list of $$n$$ scores, $$z_i$$ is the score of token $$i$$, and the sum in the denominator runs over all $$n$$ tokens. In words: raise $$e$$ to every score, then divide each result by their total.

The property we asked for now holds by construction:

$$
\frac{p_a}{p_b} = e^{z_a - z_b}
$$

where $$p_a$$ is the probability of token $$a$$. A gap of 0.693 means twice as likely, 1 means 2.718 times, and 2.303 means 10 times. Qwen's logits after the fridge prompt are 18.437 for " bottle" and 17.902 for " can", a gap of 0.535, and $$e^{0.535} = 1.71$$: the "1.71 times as likely" from Part 4, read straight off the scores.

A worked example to keep: scores [2, 1, 0]. The exponentials are 7.389, 2.718 and 1, which total 11.107, so softmax gives 0.665, 0.245 and 0.090.

With only two tokens, softmax becomes the logistic curve. Divide the top and bottom of $$e^{a} / (e^{a} + e^{b})$$ by $$e^{a}$$ and you get $$1 / (1 + e^{-(a - b)})$$, a function of the gap alone. Elo's formula is this curve, with base 10 and a scale of 400 points. Physicists wrote the same formula in 1868 as the Boltzmann distribution, with energies divided by a gas's temperature, which is where the sampling knob's name comes from. The machine learning name dates from 1989, when John Bridle called it the "normalised exponential (softmax)".

## Step 2: shifts, overflow and the max trick

### Only the gaps matter

Add the same number $$c$$ to every score. Since $$e^{z + c} = e^c e^z$$, every term picks up the same factor $$e^c$$, top and bottom:

$$
\frac{e^{z_i + c}}{\sum_j e^{z_j + c}} = \frac{e^{c}\, e^{z_i}}{e^{c} \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The factor cancels, so the probabilities don't move, and that one line is the proof. It's why GPT-2's logits can all sit near -100, as [Fundamentals Part 4](/blog/embeddings-and-logits/) found, and still give sensible probabilities. It also means $$n$$ scores carry only $$n - 1$$ numbers' worth of information: fix any one score and the gaps determine everything.

### Where the textbook formula breaks

Floating-point numbers have a largest value, and $$e^x$$ passes it quickly:

| Format | Bits | Largest number | e^x overflows when x is above |
|---|---|---|---|
| float32 (FP32) | 32 | 3.40 × 10^38 | 88.72 |
| bfloat16 (BF16) | 16 | 3.39 × 10^38 | 88.72 |
| float16 (FP16) | 16 | 65,504 | 11.09 |

The thresholds are the natural logs of the largest numbers, computed from NumPy's and PyTorch's `finfo`. At the other end, $$e^{-104}$$ rounds to exactly 0 in float32, the underflow Part 4 met.

That gives the textbook formula two ways to fail. If any score is large, its exponential becomes infinity, and infinity divided by infinity is `nan`. If every score is very negative, every exponential rounds to 0, and 0 divided by 0 is also `nan`. Here are both, on Qwen's real fridge logits. They come from Part 4's last snippet, saved with `np.save("fridge_logits.npy", z.numpy())`:

```python
import numpy as np

z = np.load("fridge_logits.npy")    # Qwen2.5-1.5B's 151,936 next-token logits (float32)

def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()

def softmax_stable(z):
    e = np.exp(z - z.max())         # the largest term becomes e^0 = 1
    return e / e.sum()

print(z.max())                                        # the top logit, " bottle"
print(softmax_naive(z).max())                         # float32: fine
print(softmax_naive(z + 100).max())                   # same distribution, shifted up
print(softmax_naive(z - 200).max())                   # same distribution, shifted down
print(softmax_naive(z.astype(np.float16)).max())      # 16-bit floats
print(softmax_stable(z + 100).max(), softmax_stable(z.astype(np.float16)).max())
```

```text
18.437366
0.15383023
nan
nan
nan
0.15382974 0.1538
```

In float32 the raw logits happen to be safe, since 18.44 is far below 88.72. Shift them up by 100, which mathematically changes nothing, and the top terms overflow. Shift them down by 200 and every term underflows. In float16, the unshifted logits already fail.

### The max trick

Shift invariance says you may subtract any number you like. Subtract the largest score, $$m = \max_j z_j$$:

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - m}}{\sum_{j} e^{z_j - m}}
$$

Now the largest exponent is exactly 0, so the largest term is $$e^0 = 1$$ and every other term lies between 0 and 1. Nothing can overflow. And because one term is exactly 1, the denominator lies between 1 and $$n$$, so it can never be 0. The last line of the output shows the stable version giving 0.1538 for every input. (The 0.15382974 differs in its last digits because adding 100 in float32 rounds each logit to a multiple of about 0.0000076.)

Goodfellow, Bengio and Courville's [Deep Learning](https://www.deeplearningbook.org/contents/numerical.html) puts it this way: subtracting the maximum "results in the largest argument to exp being 0, which rules out the possibility of overflow." Milakov and Gimelshein reported in 2018 that TensorFlow, PyTorch, MXNet, the Microsoft Cognitive Toolkit and Chainer all used this safe version. One problem is left: a tiny term can still underflow to 0, which is harmless for a probability but fatal for its log.

## Step 3: log-sum-exp, the smooth maximum

### Log-softmax

Take the log of softmax. The log of a ratio is a difference, and $$\ln e^{z_i} = z_i$$:

$$
\ln p_i = z_i - \ln \sum_{j} e^{z_j}
$$

The second term is the same for every token. It has a name, the log-sum-exp:

$$
\text{LSE}(\mathbf{z}) = \ln \sum_{j=1}^{n} e^{z_j}
$$

In words: exponentiate every score, add them up, take the log. So every log-probability is the token's logit minus one shared number. For the fridge prompt, LSE is 20.309, and " bottle" has logit 18.437, so its log-probability is $$18.437 - 20.309 = -1.872$$ and its probability $$e^{-1.872} = 0.1538$$. Part 4's `log_softmax` calls computed exactly this.

The max trick works here too. Pull $$e^m$$ out of the sum:

$$
\text{LSE}(\mathbf{z}) = m + \ln \sum_{j} e^{z_j - m}
$$

The sum is between 1 and $$n$$, so its log is safe. Log-softmax computed as $$z_i - m - \ln \sum_j e^{z_j - m}$$ never takes the log of a number that underflowed.

### Why it's called a smooth max

The sum $$\sum_j e^{z_j}$$ contains the term $$e^m$$, so it's at least $$e^m$$. It has $$n$$ terms, none bigger than $$e^m$$, so it's at most $$n e^m$$. Take logs of both bounds:

$$
m \le \text{LSE}(\mathbf{z}) \le m + \ln n
$$

Check it on [2, 1, 0]: the LSE is $$\ln 11.107 = 2.408$$, which sits between the max, 2, and $$2 + \ln 3 = 3.099$$. The upper bound is reached when all scores tie: the LSE of [2, 2, 2] is exactly 3.099.

So log-sum-exp is the maximum plus a bonus of at most $$\ln n$$. For Qwen's vocabulary, $$\ln 151{,}936 = 11.93$$, but the fridge logits give a bonus of only $$20.309 - 18.437 = 1.872$$. That bonus has a meaning: the top token's probability is $$e^{m - \text{LSE}}$$, so the bonus is exactly $$-\ln p_{\text{top}}$$. A bonus of 0 means the model is certain; a bonus of $$\ln n$$ means every token is equally likely.

### Temperature makes the max sharp

Divide the scores by a temperature $$T$$ before taking the log-sum-exp, and multiply by $$T$$ afterward. The same bounds become $$m \le T \cdot \text{LSE}(\mathbf{z}/T) \le m + T \ln n$$, so as $$T$$ shrinks toward 0, the smooth max squeezes onto the true max. Meanwhile the softmax itself heads for the argmax, the position of the max, written as a one-hot list (1 for the winner, 0 for everyone else). In ratio form, $$p_i / p_{\text{top}} = e^{(z_i - m)/T}$$: a gap of -1 at $$T = 0.1$$ becomes $$e^{-10} = 0.0000454$$.

```python
import numpy as np

def lse(z):
    m = z.max()
    return m + np.log(np.exp(z - m).sum())

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

z = np.array([2.0, 1.0, 0.0])
print(f"LSE = {lse(z):.4f}, bounds {z.max():.4f} to {z.max() + np.log(len(z)):.4f}")
print(f"LSE of [2, 2, 2] = {lse(np.array([2.0, 2.0, 2.0])):.4f}")

np.set_printoptions(precision=4)
for T in [10, 1, 0.5, 0.1]:
    print(f"T = {T:>4}: softmax {softmax(z / T)}, T * LSE(z/T) = {T * lse(z / T):.4f}")
```

```text
LSE = 2.4076, bounds 2.0000 to 3.0986
LSE of [2, 2, 2] = 3.0986
T =   10: softmax [0.3672 0.3322 0.3006], T * LSE(z/T) = 12.0194
T =    1: softmax [0.6652 0.2447 0.09  ], T * LSE(z/T) = 2.4076
T =  0.5: softmax [0.8668 0.1173 0.0159], T * LSE(z/T) = 2.0715
T =  0.1: softmax [9.9995e-01 4.5398e-05 2.0611e-09], T * LSE(z/T) = 2.0000
```

At $$T = 0.1$$ the smooth max is 2.0000 and softmax is one-hot to four decimals. At $$T = 10$$ the softmax is nearly flat and the smooth max overshoots to 12.02, inside its bound of $$2 + 10 \ln 3 = 12.99$$. Two tokens tied for the top score split the probability evenly in the limit. [Part 4](/blog/probability-language-models/) showed what these limits do to real sampling.

Log-sum-exp and softmax are also linked by a derivative. The slope of LSE with respect to score $$z_i$$ is $$e^{z_i} / \sum_j e^{z_j}$$, which is exactly $$p_i$$. Softmax is the gradient of the smooth max. The Deep Learning book notes that it "would perhaps be better to call the softmax function 'softargmax'", since it softens the argmax rather than the max.

### How sensitive softmax is

When score $$j$$ rises a little, probability $$i$$ changes at the rate

$$
\frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)
$$

where $$\delta_{ij}$$ is 1 when $$i = j$$ and 0 otherwise ([Part 7](/blog/derivatives-gradients/) builds these rates of change from scratch). For [2, 1, 0], raising the first score by 0.001 moved the probabilities by 0.000223, -0.000163 and -0.000060, matching $$0.665 \times 0.335$$, $$-0.245 \times 0.665$$ and $$-0.090 \times 0.665$$ per unit. A token's own rate, $$p(1 - p)$$, shrinks toward 0 as $$p$$ nears 1, which is why a nearly one-hot softmax stops learning ([Fundamentals Part 5](/blog/attention-explained/) shows this inside attention). [Part 11](/blog/attention-math/) covers the full grid of rates.

### Why frameworks fuse log-softmax with the loss

The training loss for one token is $$-\ln p$$ of the correct token ([Part 6](/blog/cross-entropy-perplexity/) names it). You could compute softmax, then take the log. Or you could compute $$z_{\text{correct}} - \text{LSE}(\mathbf{z})$$ in one step. Here's a token that trails the leader by 120 points but is the right answer:

```python
import torch
import torch.nn.functional as F

z = torch.tensor([0.0, -120.0], requires_grad=True)   # token 1 trails token 0 by 120
target = torch.tensor(1)                              # ...and token 1 is the right answer

two_step = -torch.log(torch.softmax(z, dim=0))[target]
fused = F.cross_entropy(z.unsqueeze(0), target.unsqueeze(0))
print(torch.softmax(z, dim=0))
print(two_step.item(), fused.item())

two_step.backward(); print(z.grad); z.grad = None
fused.backward();    print(z.grad)
```

```text
tensor([1., 0.], grad_fn=<SoftmaxBackward0>)
inf 120.0
tensor([nan, nan])
tensor([ 1., -1.])
```

The probability $$e^{-120}$$ is about $$10^{-52}$$, below float32's smallest number, so softmax rounds it to 0. The two-step loss is then $$-\ln 0 = \infty$$, and its gradient, the signal training needs most, is `nan`. The fused version never forms the tiny probability, so it returns the exact loss, 120, and a clean gradient. PyTorch's documentation for `log_softmax` says that while it is "mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and numerically unstable." Its `cross_entropy` takes raw logits for this reason, and is documented as "equivalent to applying LogSoftmax on an input, followed by NLLLoss."

The gradient [1, -1] is each probability minus the target's one-hot vector: [1 - 0, 0 - 1]. Bridle's 1989 paper already noted this "difference between the corresponding output and a one-from-N target"; [Part 8](/blog/backpropagation-math/) derives it.

## Step 4: online softmax, the core of FlashAttention

### Counting passes

The safe softmax from Step 2 reads its input three times: once to find the max, once to add up the exponentials, and once to divide. Milakov and Gimelshein's [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018) counts that as 4 memory accesses per element, 3 loads and a store. Their idea folds the first two passes into one. [Inference Part 7](/blog/flashattention/) showed it running on a worked example; here is the derivation, and why it can't drift.

### The recurrence, derived

Process the scores one at a time, keeping two numbers. After seeing $$k$$ scores, $$m_k$$ is the largest so far and $$l_k$$ is the sum of exponentials measured against that max:

$$
l_k = \sum_{j=1}^{k} e^{z_j - m_k}
$$

When score $$z_{k+1}$$ arrives, the max becomes $$m_{k+1} = \max(m_k, z_{k+1})$$. The old sum is measured against the old max, so convert it. Multiplying each of its terms by $$e^{m_k - m_{k+1}}$$ gives

$$
e^{z_j - m_k} \cdot e^{m_k - m_{k+1}} = e^{z_j - m_{k+1}}
$$

because exponents add. Rescale the old sum, then add the new term:

$$
l_{k+1} = l_k \, e^{m_k - m_{k+1}} + e^{z_{k+1} - m_{k+1}}
$$

That's the whole algorithm, and it's exact: the rescaling rewrites the same sum against a new reference point. Every exponent is zero or negative, so nothing overflows, and $$l$$ stays between 1 and $$k$$. The paper proves it by induction and notes that a float32 $$l$$ is safe for up to $$1.7 \times 10^{37}$$ elements.

Trace it on [1, 3, 2]:

| Score | Running max | Rescale old sum by | Running sum |
|---|---|---|---|
| 1 | 1 | nothing yet | e^0 = 1 |
| 3 | 3 | e^(1 - 3) = 0.1353 | 1 × 0.1353 + e^0 = 1.1353 |
| 2 | 3 | e^0 = 1 | 1.1353 + e^(-1) = 1.5032 |

The end state is $$m = 3$$, $$l = 1.5032$$, and $$m + \ln l = 3 + 0.4076 = 3.4076$$, the log-sum-exp. The pair $$(m, l)$$ is LSE stored in two safe pieces. These are the scores [2, 1, 0] reshuffled and shifted up by 1, and the LSE is the earlier 2.4076 plus 1. A second pass then writes each probability as $$e^{z_i - m} / l$$: 3 memory accesses per element instead of 4, which ran up to 1.3 times faster on an NVIDIA Tesla V100 in the paper's benchmarks.

A block of scores works the same way: take the block's max, rescale once, add the block's terms. Here it is over Qwen's 151,936 logits in blocks of 4,096:

```python
import numpy as np

def online_softmax_stats(z, block):
    m, l = -np.inf, 0.0                     # running max, running sum of e^(z - m)
    for start in range(0, len(z), block):
        zb = z[start:start + block]
        m_new = max(m, zb.max())
        l = l * np.exp(m - m_new) + np.exp(zb - m_new).sum()
        m = m_new
    return m, l

z = np.load("fridge_logits.npy").astype(np.float64)   # 151,936 logits
m, l = online_softmax_stats(z, block=4096)            # 38 blocks, one pass
p_online = np.exp(z - m) / l                          # second pass: the probabilities

p_full = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
print(m, l, m + np.log(l))                            # max, sum, log-sum-exp
print(np.abs(p_online - p_full).max())
```

```text
18.437366485595703 6.500672788630996 20.309272163084152
8.326672684688674e-17
```

The running sum ends at 6.5007, which is $$1 / 0.15383$$, one over the top token's probability, as it should be: the top term is $$e^0 = 1$$ and $$l$$ is the total. The largest difference from the full softmax is $$8 \times 10^{-17}$$, rounding in the last digit of a 64-bit float.

### Merging two partial results

The pass needn't run in order. Process two halves separately, ending with $$(m_a, l_a)$$ and $$(m_b, l_b)$$, and combine them with $$M = \max(m_a, m_b)$$:

$$
l = l_a \, e^{m_a - M} + l_b \, e^{m_b - M}
$$

Milakov and Gimelshein show this merge is associative and commutative: you can group and order the pieces any way you like. In log-sum-exp language, the LSE of the whole is the LSE of the two halves' LSEs. That's what lets [Flash-Decoding](/blog/flashattention/) split a long cache across many processors and merge their answers at the end.

### Adding the values: the attention recurrence

Attention only needs the weighted average of the value vectors (the vectors each token passes on). For one query with scores $$z_i$$ and values $$\mathbf{v}_i$$, the output is $$\mathbf{o} = \sum_i p_i \mathbf{v}_i$$. Carry a third running quantity, an un-normalized output, and rescale it by the same factor:

$$
\mathbf{acc}_{k+1} = \mathbf{acc}_k \, e^{m_k - m_{k+1}} + \sum_{i \in \text{block}} e^{z_i - m_{k+1}} \, \mathbf{v}_i
$$

At the end, $$\mathbf{o} = \mathbf{acc} / l$$. The scores are used once and thrown away; the state is two numbers and one vector per query. Here it is against standard attention for one query, 8,192 keys and a head size of 128, plus a two-way split merged by LSE, as if the halves ran on two SMs (streaming multiprocessors, a GPU's processing units):

```python
import numpy as np

rng = np.random.default_rng(0)
n, d = 8192, 128                                  # 8,192 keys, head size 128
q = rng.standard_normal(d)
K = rng.standard_normal((n, d))
V = rng.standard_normal((n, d))
s = K @ q / np.sqrt(d)                            # one query's 8,192 scores

# Standard: all scores at once
w = np.exp(s - s.max())
o_full = (w / w.sum()) @ V

# Online: blocks of 256 keys, never more than 256 scores held at once
def attend(s, V, block=256):
    m, l, acc = -np.inf, 0.0, np.zeros(V.shape[1])
    for i in range(0, len(s), block):
        sb, Vb = s[i:i + block], V[i:i + block]
        m_new = max(m, sb.max())
        scale = np.exp(m - m_new)                 # e^(m_old - m_new)
        e = np.exp(sb - m_new)
        l = l * scale + e.sum()
        acc = acc * scale + e @ Vb                # un-normalized output
        m = m_new
    return acc / l, m + np.log(l)                 # output and log-sum-exp

o_online, L = attend(s, V)
print(np.abs(o_online - o_full).max())

# Two halves computed separately (say, on two SMs), merged with their LSEs
o1, L1 = attend(s[:n // 2], V[:n // 2])
o2, L2 = attend(s[n // 2:], V[n // 2:])
L12 = np.logaddexp(L1, L2)                        # LSE of the two LSEs
o_merged = np.exp(L1 - L12) * o1 + np.exp(L2 - L12) * o2
print(L, L12, np.abs(o_merged - o_full).max())
```

```text
9.367506770274758e-17
9.487242073621518 9.487242073621518 1.2836953722228372e-16
```

Both match standard attention to about $$10^{-16}$$. In the merge, the weight $$e^{L_1 - L}$$ is the share of the total probability that falls in the first half, so the merged output is a weighted average of the two partial outputs.

![A left-to-right loop. Start with m equal to minus infinity, l equal to 0 and acc equal to 0. Read the next block of scores z and values v. Compute the new max M, the larger of m and the block's max. Rescale the old totals l and acc by e to the power m minus M. Add the block: l gains the sum of e to the z minus M, acc gains the sum of e to the z minus M times v, then set m to M. If there are more blocks, loop back to reading the next block; if not, the output is acc divided by l, and the log-sum-exp is m plus the natural log of l.](/diagrams/m05-online-softmax.svg "The online softmax loop for one attention query. M is the new running max.")

Only the max, the sum and the output vector outlive each block.

That's the FlashAttention recurrence. The [FlashAttention paper](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) writes it as a decomposition of the softmax of two concatenated blocks, $$m$$ and $$\ell$$ included, and its first version divided by the running sum after every block and wrote the result back to memory. [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023) made the two tweaks the code above already uses: keep an "un-scaled" output and divide by $$\ell$$ only once, when the loop ends, and save only the logsumexp $$L = m + \log(\ell)$$ for the backward pass, instead of both $$m$$ and $$\ell$$.

## In the real world

- PyTorch's GPU softmax kernel for rows of up to 2,048 elements ([`PersistentSoftmax.cuh`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/PersistentSoftmax.cuh)) is Step 2 and Step 3 line for line: it finds `max_value`, adds up `std::exp(elements - max_value)`, takes the log of that sum for log-softmax, and writes `elements - max_value - sum`. Its running totals use a wider accumulator type, float32 for 16-bit inputs.
- Mixed-precision training keeps softmax in float32. PyTorch's [autocast](https://docs.pytorch.org/docs/stable/amp.html) lists `softmax`, `log_softmax`, `cross_entropy` and `nll_loss` among the "CUDA Ops that can autocast to float32". Hugging Face's [Llama code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) does the same by hand in its plain attention path: `softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`, computing the weights in float32 and converting back.
- FlashAttention hands log-sum-exp back to you. The library's [interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py) can return `softmax_lse`, documented as "the logsumexp of each row of the matrix QK^T * scaling", one float32 number per query row and head even when the inputs are 16-bit. That's the $$L$$ from Step 4, saved for the backward pass and for merging splits.
- OpenAI's gpt-oss models add one learned score per attention head to the softmax. The [model card](https://arxiv.org/abs/2508.10925) says each head "has a learned bias in the denominator of the softmax", which lets attention "pay no attention to any tokens". Hugging Face's [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py) appends that score as an extra column, subtracts the row max ("prevents overflow in BF16/FP16", says a comment), takes the softmax and drops the column, so a row's weights can add up to less than 1. The model card compares the design to attention sinks, which [Fundamentals Part 5](/blog/attention-explained/) describes.
- PaLM kept the log-sum-exp itself near 0. Its training ([Chowdhery et al., 2022](https://arxiv.org/abs/2204.02311)) added "an auxiliary loss of z_loss = 10^-4 · log^2 Z to encourage the softmax normalizer log(Z) to be close to 0", which the authors found "increases the stability of training". Here $$\log Z$$ is the LSE over PaLM's 256k-token vocabulary. Shift invariance makes this free: moving every logit up or down changes no probability.

## Common misconceptions

- **"Subtracting the max is an approximation."** It's exact. Adding or subtracting any constant multiplies the top and bottom of every fraction by the same $$e^c$$, which cancels. The only differences are float rounding, as in the 0.15382974 versus 0.15383023 above.
- **"Bigger logits mean the model is more confident."** Only the gaps matter. Logits of [118, 117, 116] give exactly the same probabilities as [2, 1, 0]. Confidence shows in how far the top logit sits above the log-sum-exp, which is $$-\ln p_{\text{top}}$$: 1.87 for the fridge prompt.
- **"Log-softmax is just the log of softmax."** In exact math, yes. In float32, a token that trails by 120 gets probability 0 from softmax and a log of minus infinity, while log-softmax returns -120 exactly. That's why loss functions take raw logits.
- **"Softmax is a smooth version of max."** It's a smooth version of argmax: it returns a list of weights pointing at the winner. A smooth version of max is log-sum-exp, whose gradient is softmax.

## Key takeaways

- Softmax raises $$e$$ to every score and divides by the total. The exponential is the one smooth choice where a fixed gap in scores means a fixed ratio of probabilities: a gap of 1 is a factor of 2.718.
- Adding the same number to every score changes nothing, so you can subtract the max first. That stops $$e^x$$ from overflowing, which happens above 88.72 in float32 and above just 11.09 in float16.
- Log-sum-exp lies between the max score and the max plus $$\ln n$$, and tends to the max as temperature goes to 0. Every log-probability is a logit minus the LSE, and computing it that way, fused with the loss, never takes the log of 0.
- Online softmax keeps a running max and a running sum, rescaling the sum by $$e^{m_{\text{old}} - m_{\text{new}}}$$ when the max grows. Add a running weighted sum of values and you have FlashAttention's inner loop, exact up to rounding.

## What's next

The fused loss in Step 3, minus the log-probability of the right token, is the number every LLM is trained to push down. Part 6 gives it its proper names: surprise, entropy, cross-entropy and perplexity. It explains why a loss of 2 nats means the model is as unsure as if it were choosing among about 7 equally likely tokens, and why predicting text well is the same problem as compressing it.

## Further reading

- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867): the online softmax recurrence, its proof by induction, the parallel merge, and the softmax plus top-k fusion.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135): section 3.1 decomposes softmax over blocks and builds the tiled attention kernel on it.
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)](https://arxiv.org/abs/2307.08691): section 3.1.1 introduces the un-scaled output and the stored logsumexp.
- [Deep Learning, chapter 4.1: Overflow and Underflow (Goodfellow, Bengio & Courville, 2016)](https://www.deeplearningbook.org/contents/numerical.html): the max trick and why log-softmax needs its own stable implementation; [chapter 6.2.2.3](https://www.deeplearningbook.org/contents/mlp.html) covers softmax output units and the "softargmax" remark.
- [Training Stochastic Model Recognition Algorithms as Networks can Lead to Maximum Mutual Information Estimation of Parameters (Bridle, 1989)](https://proceedings.neurips.cc/paper_files/paper/1989/file/0336dcbab05b9d5ad24f4333c7658a0e-Paper.pdf): an early use of the name "softmax", and the output-minus-target gradient.
- [torch.nn.functional.log_softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) and [torch.nn.CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (PyTorch docs): why the loss takes raw logits.
- [PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311): section 5 describes the z-loss on the softmax normalizer.
- [Softmax function (Wikipedia)](https://en.wikipedia.org/wiki/Softmax_function): the Boltzmann connection and the history of the name.
- [From Online Softmax to FlashAttention (Zihao Ye, University of Washington CSE 599M notes)](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf): a short derivation from safe softmax to the one-pass attention recurrence.
