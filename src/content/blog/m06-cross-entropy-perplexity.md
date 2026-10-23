---
title: "Entropy, Cross-Entropy and Perplexity"
description: "What an LLM's loss number means: surprise, entropy, cross-entropy, perplexity, bits per byte and KL divergence, and why predicting text is compressing it."
slug: "cross-entropy-perplexity"
category: "llm-math"
tags: ["math", "training", "evals", "tokens"]
series: "llm-math"
seriesOrder: 6
pubDate: 2026-10-23
---

## TL;DR

Every LLM is trained to push one number down: the cross-entropy loss. This article builds it from one idea, surprise, which is minus the log of a probability. Average surprise is entropy. Average surprise under the wrong model is cross-entropy, always at least the entropy, and the gap is the KL divergence. Converting units turns the loss into perplexity, bits per token and bits per byte. The last step shows why a model with low loss is also a compressor: Qwen2.5-1.5B stores two paragraphs of The Wonderful Wizard of Oz in about 9% of their size, where gzip needs 51%.

## Why it matters

Qwen2.5-1.5B-Instruct, a small open model, can squeeze the opening 1,468 bytes of The Wonderful Wizard of Oz into about 129 bytes. On the same text, gzip needs 753 bytes and xz needs 852.

The number that sets those 129 bytes is the one every training run watches: the loss. [Part 4](/blog/probability-language-models/) ended with one, 2.63 per token for "The cat sat on the mat." Is 2.63 good? What units is it in? Why does a perfect model not reach 0? And why do papers report "perplexity" or "bits per byte" instead?

All of it comes from one formula that Claude Shannon wrote down in 1948, a few lines of algebra, and the log rules from Part 4.

## The intuition

Play twenty questions with a friend who has picked one of 8 equally likely cards. The smart strategy halves the possibilities with every yes-or-no question: 8, then 4, then 2, then 1. Three questions, every time. If one card were much more popular than the rest, you'd ask "is it that card?" first and often finish in one.

Information theory measures uncertainty this way, as the number of yes-or-no questions the best strategy needs on average. Each answer is one bit. A model that predicts well knows which "cards" are popular, so it needs fewer questions to pin down what actually came next.

The analogy breaks in one place. Questions come in whole numbers, but the average this article computes can be 0.469 questions. The fix, arithmetic coding in Step 5, answers many guesses at once so the fractions add up across a whole text.

## Step 1: surprise and entropy

### Surprise is minus a log

How surprised should you be when something with probability $$p$$ happens? Three properties pin the answer down:

- Something certain ($$p = 1$$) carries no surprise.
- Rarer events are more surprising.
- Surprises from independent events add. Two fair coins both landing heads has probability $$\tfrac12 \times \tfrac12 = \tfrac14$$, and should be exactly twice as surprising as one head.

Probabilities of independent events multiply, and surprises should add. The function that turns products into sums is the logarithm from Part 4, so surprise is a log. A minus sign makes it positive, since every probability is at most 1:

$$
s(x) = -\log_2 p(x)
$$

Here $$s(x)$$ is the surprise of outcome $$x$$, and $$\log_2$$ is the base-2 log: the power you raise 2 to in order to get the number. Base 2 measures surprise in bits. A fair coin landing heads is $$-\log_2 \tfrac12 = 1$$ bit. Two heads are $$-\log_2 \tfrac14 = 2$$ bits, one per coin. One card out of 8 is 3 bits, the three questions of the game.

| Probability | Surprise in bits | Reads as |
|---|---|---|
| 1 | 0 | certain, no news |
| 0.9 | 0.152 | expected |
| 0.5 | 1 | a coin flip |
| 0.125 | 3 | one card in 8 |
| 0.1 | 3.32 | a long shot |
| 1/151,936 | 17.21 | one Qwen token picked at random |

Shannon introduced the unit in [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) (1948): with base 2, "the resulting units may be called binary digits, or more briefly bits, a word suggested by J. W. Tukey."

### Entropy is average surprise

A single surprise depends on what happened. To describe a whole source of randomness, average the surprise over everything that could happen, each weighted by its probability. That's the expected value from Part 4, applied to surprise, and it's called entropy:

$$
H(p) = -\sum_{x} p(x) \log_2 p(x)
$$

Here $$H(p)$$ is the entropy of distribution $$p$$, and the sum runs over every possible outcome $$x$$. In words: how surprised you'll be, on average, by the next draw. Shannon wrote the same formula and noted that it matches "entropy as defined in certain formulations of statistical mechanics," which gave it its name.

Work a biased coin by hand: heads 0.9, tails 0.1.

$$
H = 0.9 \times 0.152 + 0.1 \times 3.32 = 0.137 + 0.332 = 0.469 \text{ bits}
$$

Heads barely surprises, and tails, though surprising, comes up only one time in ten.

```python
import numpy as np

def entropy(p):                          # average surprise, in bits
    p = np.array(p)
    return -np.sum(p * np.log2(p))

print(f"fair coin    {entropy([0.5, 0.5]):.3f} bits")
print(f"biased coin  {entropy([0.9, 0.1]):.3f} bits")
print(f"fair die     {entropy([1/6] * 6):.3f} bits")
print(f"fridge toy   {entropy([0.50, 0.30, 0.15, 0.05]):.3f} bits")
print(f"uniform over 151,936 tokens  {entropy(np.full(151936, 1/151936)):.2f} bits")
```

```text
fair coin    1.000 bits
biased coin  0.469 bits
fair die     2.585 bits
fridge toy   1.648 bits
uniform over 151,936 tokens  17.21 bits
```

The "fridge toy" is Part 4's four-token distribution ("bottle" 0.50, "can" 0.30, "bowl" 0.15, "jar" 0.05). The last line is a model that has learned nothing: if Qwen2.5 spread its bets evenly over all 151,936 tokens in its vocabulary, every token would cost $$\log_2 151{,}936 = 17.21$$ bits.

Entropy is never negative, since no surprise is. For a fixed number of outcomes it is largest when all are equally likely: $$n$$ equal outcomes give $$\log_2 n$$ bits, and any lopsidedness lowers it.

### Entropy of English

Shannon also measured English. In [Prediction and Entropy of Printed English](https://sites.socsci.uci.edu/~rfutrell/teaching/itl-davis/readings/shannon1951prediction.pdf) (1951), he started from a 26-letter alphabet: $$\log_2 26 = 4.70$$ bits per letter if letters were equally likely, and 4.14 bits using real letter frequencies. Counting patterns up to eight letters long brought it to about 2.3 bits. Then he had people guess the next letter of real text, a human language model, and concluded that with up to 100 letters of context the entropy falls "to something of the order of one bit per letter." Each pattern a predictor knows takes bits off the bill, and an LLM plays the same guessing game with billions of learned parameters.

## Step 2: cross-entropy, KL divergence and Gibbs' inequality

### Paying for the wrong model

Entropy assumes you know the true probabilities, and you never do. A model has its own guess, $$q$$, while the world draws from $$p$$. Every time outcome $$x$$ happens, you pay the surprise your model assigns, $$-\log_2 q(x)$$, but it happens as often as the world says, $$p(x)$$. The average bill is the cross-entropy:

$$
H(p, q) = -\sum_{x} p(x) \log_2 q(x)
$$

In words: the average surprise when reality is $$p$$ and your beliefs are $$q$$. When $$q = p$$, it's just the entropy.

Take the fridge distribution as reality, $$p = [0.50, 0.30, 0.15, 0.05]$$, and a model that knows nothing, $$q = [0.25, 0.25, 0.25, 0.25]$$. Every outcome costs $$-\log_2 0.25 = 2$$ bits, so $$H(p, q) = 2$$ bits, against $$H(p) = 1.648$$. A closer model, $$r = [0.40, 0.40, 0.10, 0.10]$$, pays 1.722.

### KL divergence is the extra cost

The gap between what you pay and the best possible bill has its own name, the Kullback-Leibler (KL) divergence, from [Kullback and Leibler (1951)](https://doi.org/10.1214/aoms/1177729694):

$$
D_{\mathrm{KL}}(p \,\|\, q) = H(p, q) - H(p) = \sum_x p(x) \log_2 \frac{p(x)}{q(x)}
$$

Read $$D_{\mathrm{KL}}(p \,\|\, q)$$ as "the KL divergence of $$q$$ from $$p$$". The second form comes from subtracting the two sums term by term and using $$\log a - \log b = \log(a/b)$$. In words: the extra bits per outcome you waste by believing $$q$$ when the truth is $$p$$.

```python
import numpy as np

def cross_entropy(p, q):                 # reality p, model q, in bits
    return -np.sum(np.array(p) * np.log2(q))

def kl(p, q):
    return cross_entropy(p, q) - cross_entropy(p, p)

p = [0.50, 0.30, 0.15, 0.05]             # reality
q = [0.25, 0.25, 0.25, 0.25]             # a model that knows nothing
r = [0.40, 0.40, 0.10, 0.10]             # a model that is close

print(f"H(p)   = {cross_entropy(p, p):.3f}")
print(f"H(p,q) = {cross_entropy(p, q):.3f}   KL(p||q) = {kl(p, q):.3f}")
print(f"H(p,r) = {cross_entropy(p, r):.3f}   KL(p||r) = {kl(p, r):.3f}")
print(f"KL(q||p) = {kl(q, p):.3f}")

rng = np.random.default_rng(0)           # try 100,000 random models
best = min(cross_entropy(p, rng.dirichlet(np.ones(4))) for _ in range(100_000))
print(f"lowest H(p, random model) = {best:.4f}")
```

```text
H(p)   = 1.648
H(p,q) = 2.000   KL(p||q) = 0.352
H(p,r) = 1.722   KL(p||r) = 0.074
KL(q||p) = 0.449
lowest H(p, random model) = 1.6482
```

Out of 100,000 random models, none beat 1.6482 bits, a hair above $$H(p) = 1.6477$$. And swapping the arguments changes the answer: $$D_{\mathrm{KL}}(p \,\|\, q) = 0.352$$ but $$D_{\mathrm{KL}}(q \,\|\, p) = 0.449$$. KL divergence measures how far apart two distributions are, but it isn't a distance in the everyday sense, because the distance from A to B differs from B to A.

### Why no model beats the truth

The random search points to Gibbs' inequality: $$H(p, q) \ge H(p)$$, with equality only when $$q = p$$. Equivalently, KL divergence is never negative. The proof needs one fact about the natural log (written $$\ln$$, base $$e \approx 2.718$$): its curve always lies below the straight line that touches it at $$x = 1$$.

$$
\ln x \le x - 1 \quad \text{for every } x > 0
$$

Check a couple: $$\ln 2 = 0.693 \le 1$$, and $$\ln 0.5 = -0.693 \le -0.5$$. The two sides are equal only at $$x = 1$$. Now apply it with $$x = q(x)/p(x)$$ for each outcome, summing over outcomes with $$p(x) > 0$$:

$$
\begin{aligned}
-D_{\mathrm{KL}} &= \sum_x p(x) \ln \frac{q(x)}{p(x)} \\
&\le \sum_x p(x) \left( \frac{q(x)}{p(x)} - 1 \right) \\
&= \sum_x q(x) - \sum_x p(x) \le 1 - 1 = 0
\end{aligned}
$$

The first line is the KL divergence with the fraction flipped, which flips the sign. The second uses the inequality on every term. The third cancels the $$p(x)$$ and uses the fact that the $$p(x)$$ add up to exactly 1 and the $$q(x)$$, summed over only some outcomes, to at most 1. So $$D_{\mathrm{KL}} \ge 0$$, and it's 0 only if every ratio $$q(x)/p(x)$$ equals 1.

It also tells you what training can and can't do. Cross-entropy splits into two parts:

$$
H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)
$$

The first part belongs to the data and no model can touch it. The second belongs to the model and is zero only when the model matches reality exactly. Lowering cross-entropy can only mean shrinking the KL part, pulling $$q$$ toward $$p$$.

### Forward and reverse KL

Which way round you use KL changes what a model learns. Take a reality with two likely options, $$p = [0.49, 0.01, 0.49, 0.01]$$, and two candidate models: one that spreads out evenly, $$[0.25, 0.25, 0.25, 0.25]$$, and one that bets on a single option, $$[0.97, 0.01, 0.01, 0.01]$$. Forward KL, $$D_{\mathrm{KL}}(p \,\|\, q)$$, prefers the spread-out model (0.859 bits against 2.268), because it charges heavily wherever reality happens and the model said "unlikely". Reverse KL, $$D_{\mathrm{KL}}(q \,\|\, p)$$, prefers the single bet (0.899 against 1.837), because it only charges where the model puts its own probability. Forward KL is called mass-covering, reverse KL mode-seeking. Training on data and [distillation](https://arxiv.org/abs/1503.02531) (training a small model on a big model's full output distribution) use the forward direction. The KL "leash" in RLHF, from [Fundamentals Part 13](/blog/base-model-to-assistant/), scores the policy's own answers against a reference, the reverse direction, which Part 16 builds on.

## Step 3: the training loss is cross-entropy

### One-hot reality

When an LLM trains on a sentence, what is "reality" at each position? Only one token actually came next, so the target is a one-hot distribution: probability 1 on the actual token, 0 on every other. Put that into the cross-entropy formula and every term but one is multiplied by 0:

$$
H(\text{one-hot}, q) = -1 \cdot \ln q(\text{actual token}) = -\ln q(\text{actual token})
$$

That's the loss [Fundamentals Part 3](/blog/neural-networks-basics/) introduced in words: take the probability the model gave to the right answer and compute minus its log. If the fridge model gives "can" 0.30 and the next token really is "can", the loss is $$-\ln 0.30 = 1.204$$.

Averaged over the $$N$$ tokens of a text, with $$w_i$$ the token at position $$i$$:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \ln q(w_i \mid w_{<i})
$$

This is exactly Part 4's sum of log-probabilities, with the sign flipped and divided by the number of tokens. Lowering the loss and raising the probability of the training text are the same goal. Over billions of tokens of real text, this average estimates the cross-entropy between the true distribution of language and the model, so Step 2's split applies: the loss is the entropy of language plus the model's KL divergence from it.

### Nats and bits

Training code uses the natural log, so the loss comes out in nats rather than bits. Changing the base of a log only rescales it, because $$\log_2 x = \ln x / \ln 2$$:

$$
\text{bits} = \frac{\text{nats}}{\ln 2} \approx 1.443 \times \text{nats}
$$

Part 4's 2.63 nats per token is $$2.63 / 0.693 = 3.79$$ bits per token. A model that spreads its bets evenly over Qwen's vocabulary scores $$\ln 151{,}936 = 11.93$$ nats, which is the 17.21 bits from before.

### Perplexity: effective number of choices

A loss of 2.63 nats is hard to picture, but undoing the log turns it into a count. Perplexity is $$e$$ raised to the loss (or 2 raised to it, if the loss is in bits):

$$
\mathrm{PPL} = e^{L}
$$

A fair die has entropy $$\log_2 6$$ bits, so its perplexity is exactly 6, its six faces. In general, a model with perplexity $$k$$ is as uncertain, on average, as a uniform pick among $$k$$ options. The biased coin has perplexity $$2^{0.469} = 1.38$$, a coin that behaves like 1.38 equally likely sides. Part 4's sentence has $$e^{2.63} \approx 13.9$$: at each token Qwen was about as unsure as a pick among 14 equally likely tokens. And the model that knows nothing has perplexity 151,936.

### Measuring a real model

Here is Qwen2.5-1.5B-Instruct on the first two paragraphs of Chapter 1 of [The Wonderful Wizard of Oz](https://www.gutenberg.org/ebooks/55) (L. Frank Baum, 1900, public domain): 284 words from "Dorothy lived in the midst of the great Kansas prairies" to "as dull and gray as everything else." The text is as Project Gutenberg prints it, with one curly apostrophe and one long dash, 3 bytes each in UTF-8. As in Part 4, a blank-page `<|endoftext|>` token goes first, and the Instruct model reads plain text with no chat template. `F.cross_entropy`, PyTorch's built-in loss, takes the raw scores (logits) and averages $$-\ln q$$ over the tokens; [Part 5](/blog/softmax-math/) covers the log-softmax inside it.

```python
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)

text = open("oz_passage.txt", encoding="utf-8").read()      # the two paragraphs above
start = tok.convert_tokens_to_ids("<|endoftext|>")           # blank-page token, as in Part 4
ids = torch.tensor([[start] + tok(text)["input_ids"]])
with torch.no_grad():
    logits = model(ids).logits[0, :-1]                       # one prediction per position
targets = ids[0, 1:]                                         # the token that actually came next

loss = F.cross_entropy(logits, targets)                      # mean of -ln q(correct token)
n_tokens, n_bytes = len(targets), len(text.encode("utf-8"))
total_bits = loss.item() * n_tokens / math.log(2)

print(f"tokens {n_tokens}, UTF-8 bytes {n_bytes}")
print(f"loss {loss.item():.3f} nats/token = {loss.item() / math.log(2):.3f} bits/token")
print(f"perplexity {math.exp(loss.item()):.2f}")
print(f"total {total_bits:.1f} bits, bits per byte {total_bits / n_bytes:.3f}")
```

```text
tokens 335, UTF-8 bytes 1468
loss 2.135 nats/token = 3.080 bits/token
perplexity 8.46
total 1031.9 bits, bits per byte 0.703
```

The run used PyTorch 2.11 and Transformers 5.20 on a CPU. Qwen was, on average, as unsure as a pick among 8.5 equally likely tokens, but the average hides a wide spread. The median token cost 1.35 nats, 30 tokens cost under 0.1, and 36 cost over 5:

| Context | Next token | Model's probability | Surprise (nats) |
|---|---|---|---|
| "ground, called a" | " cycl" (of cyclone) | 0.000021 | 10.76 |
| "winds arose," | " mighty" | 0.000022 | 10.71 |
| "small, for the" | " lumber" | 0.00005 | 9.90 |
| "table, three or" | " four" | 0.996 | 0.004 |
| "great Kansas prair" | "ies" | 0.998 | 0.002 |
| "rusty looking cookst" | "ove" | 0.9995 | 0.0005 |

Content words that the context doesn't force cost 10 nats each. Finishing a word that's already half-written costs almost nothing. The first 20 tokens averaged 3.15 nats, the rest 2.07, because the model knew more about the story as it read.

## Step 4: bits per byte and comparing models

### Why perplexity doesn't travel

The 1,031.9 bits are a property of the model and the text. Perplexity divides them by the number of tokens, and that number depends on the tokenizer. Spread the same total over different units and you get very different "perplexities":

| Unit | Count in the passage | Bits per unit | Perplexity per unit |
|---|---|---|---|
| Words | 284 | 3.633 | 12.41 |
| Qwen tokens | 335 | 3.080 | 8.46 |
| Characters | 1,464 | 0.705 | 1.63 |
| UTF-8 bytes | 1,468 | 0.703 | 1.63 |

Only the unit changed. A model whose tokenizer cuts text into longer pieces does more work per token and shows a higher per-token perplexity even if it predicts the text equally well. The Hugging Face guide to [perplexity](https://huggingface.co/docs/transformers/perplexity) warns that "the tokenization procedure has a direct impact on a model's perplexity which should always be taken into consideration when comparing different models."

### Bits per byte

The fix is to divide by something every model agrees on: the bytes of the text itself, in UTF-8, the standard encoding that stores English letters in one byte each and other characters in two to four. The [Pile paper](https://arxiv.org/abs/2101.00027) (Gao et al., 2020) made this its "preferred metric," chosen "due to its invariance to different tokenization schemes and the ambiguity of measuring characters in Unicode." Its formula, with $$L_T$$ the number of tokens, $$L_B$$ the number of bytes and $$\ell$$ the loss in nats per token:

$$
\mathrm{BPB} = \frac{L_T}{L_B} \cdot \frac{\ell}{\ln 2}
$$

In words: convert the loss to bits per token, then multiply by tokens per byte. For the Oz passage, $$\tfrac{335}{1468} \times \tfrac{2.135}{0.693} = 0.703$$. The paper's Table 2 measured GPT-2 and GPT-3 this way on the whole Pile: 1.2253 bits per byte for the smallest GPT-2 and 0.7177 for the largest GPT-3 (davinci), which scored 0.7115 on its Project Gutenberg books.

![A flowchart. The model's probability for each actual next token in 335 tokens of Oz becomes a surprise per token, minus ln q. Averaging gives the cross-entropy loss, 2.135 nats per token. One branch raises e to the loss to get perplexity 8.46, the effective choices per token. The other divides by ln 2 to get 3.080 bits per token, then multiplies by tokens and divides by bytes to get 0.703 bits per byte, which arithmetic coding turns into about 129 bytes for 1,468 bytes of text.](/diagrams/m06-one-number.svg "One loss, many names: perplexity, bits per token, bits per byte and compressed size are unit conversions of the same average surprise.")

## Step 5: language modeling is compression

### Arithmetic coding

Shannon proved the link in 1948. His Theorem 9 says a source with entropy $$H$$ bits per symbol can be encoded in about $$H$$ bits per symbol, and no code can do better on average. Arithmetic coding gets there in practice, driven directly by a model's probabilities.

Start with the interval from 0 to 1. Split it into slices, one per token, each as wide as the token's probability, like the CDF from Part 4. The first token picks its slice. Split that slice the same way for the second token, and so on. After the whole message the interval's width is the product of all the probabilities, the chain rule from Part 4. Writing down any number inside it identifies the message, and a binary fraction that lands inside an interval of width $$w$$ needs about $$-\log_2 w$$ bits.

```python
from fractions import Fraction as F
import math

tokens = ["bottle", "can", "bowl", "jar"]
q = [F(50, 100), F(30, 100), F(15, 100), F(5, 100)]   # the model's probabilities
cdf = [sum(q[:i]) for i in range(len(q))]              # where each token's slice starts

def encode(message):
    low, width = F(0), F(1)
    for t in message:                                  # zoom into the token's slice
        i = tokens.index(t)
        low, width = low + width * cdf[i], width * q[i]
    k = 0                         # fewest bits whose whole range [x, x + 2^-k) fits inside the interval
    while True:
        k += 1
        x = math.ceil(low * 2**k)
        if F(x + 1, 2**k) <= low + width:
            return format(x, f"0{k}b"), width

def decode(bits, n):
    x, out = F(int(bits, 2), 2**len(bits)), []
    low, width = F(0), F(1)
    for _ in range(n):
        for i in reversed(range(len(q))):              # find the slice x falls in
            if x >= low + width * cdf[i]:
                out.append(tokens[i])
                low, width = low + width * cdf[i], width * q[i]
                break
    return out

msg = ["bottle", "bottle", "can", "bottle", "bottle", "bottle"]
bits, width = encode(msg)
print("interval width =", float(width), " -log2(width) =", round(-math.log2(width), 2), "bits")
print("code:", bits, f"({len(bits)} bits)")
print("decoded:", decode(bits, len(msg)))
```

```text
interval width = 0.009375  -log2(width) = 6.74 bits
code: 0010000 (7 bits)
decoded: ['bottle', 'bottle', 'can', 'bottle', 'bottle', 'bottle']
```

The width is $$0.5^5 \times 0.3 = 0.009375$$, worth 6.74 bits of surprise, and the code uses 7. A fixed code with 2 bits per token (four tokens need two bits each) would use 12. The decoder runs the same model, rebuilds the same slices and reads the tokens back, stopping after a known length or an end token. Over a whole message, however long, the code is never more than about 2 bits above the total surprise.

[Delétang et al. (2023)](https://arxiv.org/abs/2309.10668), "Language Modeling Is Compression," state the consequence directly: "minimizing the log-loss is equivalent to minimizing the compression rate," which is "exactly the same objective used to train current foundation models." Each nat per token taken off the loss takes 1.44 bits per token off the compressed text.

### Qwen against gzip

So 0.703 bits per byte means Qwen plus an arithmetic coder could store the passage in about $$1{,}031.9 / 8 = 129$$ bytes, 8.8% of its size. The same passage through standard compressors at their strongest settings:

| Compressor | Passage (1,468 bytes) | Size ratio | Whole book (217,466 bytes) | Size ratio |
|---|---|---|---|---|
| gzip -9 | 753 bytes | 51.3% | 73,859 bytes | 34.0% |
| bzip2 -9 | 776 bytes | 52.9% | 55,955 bytes | 25.7% |
| xz -9e | 852 bytes | 58.0% | 64,420 bytes | 29.6% |
| zstd -19 | 752 bytes | 51.2% | 64,908 bytes | 29.8% |
| Qwen2.5-1.5B + arithmetic coding | about 129 bytes | 8.8% | not run | not run |

The tools are gzip 1.12, bzip2 1.0.8, xz 5.4.5 and zstd 1.5.5. They find repeats within the file, so a short passage gives them little to work with, and even the whole book only gets them to 26%. The model arrives already knowing English. Delétang et al. found the same with 2,048-byte chunks of Wikipedia (enwik9): gzip reached 48.1%, LZMA2 (the method inside xz) 50.0%, and Chinchilla 70B 8.3%. Their Table 1 also has Chinchilla 70B, trained mostly on text, compressing ImageNet image patches to 48.0% (PNG: 58.5%) and LibriSpeech audio to 21.0% (FLAC: 30.3%); the abstract quotes 43.4% and 16.4%.

The decoder needs the model, though, and Qwen2.5-1.5B is about 3 GB of weights in 16-bit precision. Counting Chinchilla 70B's parameters as part of the compressed file, the paper's "adjusted compression rate" on 1 GB of enwik9 is 14,008%: the file grows 140-fold. Predictive power pays off only when the model is shared, like a dictionary both sides already own.

## In the real world

- Scaling laws are predictions about cross-entropy. Chinchilla's fitted loss in [Fundamentals Part 12](/blog/pretraining-scaling-laws/), `L = 1.69 + 406.4 / N^0.34 + 410.7 / D^0.28`, has a constant term the [paper](https://arxiv.org/abs/2203.15556) says "should correspond to the entropy of natural text": the $$H(p)$$ of Step 2, which no amount of parameters ($$N$$) or data ($$D$$) removes. The other two terms are the KL part, shrinking as models and data grow. On DeepMind's data and tokenizer, 1.69 nats is 2.44 bits per token, a perplexity floor of 5.42. Part 15 derives the compute-optimal split from this formula.
- Training code computes this loss directly. `torch.nn.CrossEntropyLoss` expects "the unnormalized logits for each class" and by default averages over tokens, giving the $$L$$ of Step 3.
- Perplexity depends on how much context the model sees. Hugging Face's [perplexity guide](https://huggingface.co/docs/transformers/perplexity) scores GPT-2 large on WikiText-2: cutting the text into separate 1,024-token chunks gives 19.44, and a sliding window that keeps at least 512 tokens of context gives 16.44. Context alone cut perplexity by 15%, the same effect as in the first 20 Oz tokens.
- Distillation trains on soft targets. [Hinton, Vinyals and Dean (2015)](https://arxiv.org/abs/1503.02531) train a small model with "the cross entropy with the soft targets," the big model's full distribution. The teacher's entropy is fixed, so this minimizes forward KL from teacher to student.

## Common misconceptions

- **"Lower perplexity always means a better model."** Only on the same text, with the same tokenizer and the same context length. The same Qwen predictions give perplexity 12.41 per word and 1.63 per character. Memorized text also scores low: the opening words of A Tale of Two Cities ("It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness,", 29 tokens) gets a loss of 0.583 nats per token, perplexity 1.79, because the model has seen it countless times.
- **"A perfect model would have zero loss."** Zero loss means probability 1 on every next token, which only works if text has no randomness. Language does, so the best possible loss is its entropy, the 1.69-nat floor in Chinchilla's fit, not zero.
- **"KL divergence is the distance between two distributions."** It is zero only when they match and positive otherwise, but it isn't symmetric. On the fridge example it's 0.352 bits one way and 0.449 the other, and the direction decides whether a model spreads out or locks onto one answer.
- **"An LLM is a better zip file."** As a coder it needs almost 6 times fewer bytes than gzip on the Oz passage, but only if the 3 GB model is already at both ends. Counted honestly, Chinchilla 70B's adjusted compression rate on enwik9 is 14,008%.

## Key takeaways

- Surprise is minus the log of a probability: a coin flip is 1 bit, a 1-in-151,936 guess is 17.21 bits. Logs make the surprises of independent events add up.
- Entropy is average surprise, the unpredictability of a source. Cross-entropy is average surprise when you believe the wrong distribution, and it's always at least the entropy. The difference is the KL divergence, which is never negative and isn't symmetric.
- An LLM's training loss is cross-entropy against the one-hot next token: minus the log-probability of the right token, averaged. It equals the entropy of language plus the model's KL divergence from it, and training can only shrink the second part.
- Perplexity is e to the loss, the effective number of equally likely choices per token: 8.46 for Qwen2.5-1.5B on the Oz passage. It changes with the tokenizer, so bits per byte, 0.703 here, is the fair way to compare models.
- Predicting is compressing. With arithmetic coding, the loss sets the compressed size: about 129 bytes for 1,468 bytes of Oz, where gzip needs 753, as long as the model itself doesn't count.

## What's next

This article showed what the loss measures, but not how training lowers it. [Part 7](/blog/derivatives-gradients/) starts on that: slopes, partial derivatives, the gradient as the direction of steepest climb, and the chain rule of calculus, which tells every one of a model's billions of parameters which way to move to shave a little off the cross-entropy.

## Further reading

- [A Mathematical Theory of Communication (Shannon, 1948)](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf): defines entropy and the bit, and proves that $$H$$ bits per symbol is the limit for lossless coding (Theorem 9).
- [Prediction and Entropy of Printed English (Shannon, 1951)](https://sites.socsci.uci.edu/~rfutrell/teaching/itl-davis/readings/shannon1951prediction.pdf): people as next-letter predictors, and the estimate of about one bit per letter of English.
- [On Information and Sufficiency (Kullback and Leibler, 1951)](https://doi.org/10.1214/aoms/1177729694): the paper that introduced what is now called the KL divergence.
- [Language Modeling Is Compression (Delétang et al., 2023)](https://arxiv.org/abs/2309.10668): arithmetic coding with Chinchilla and Llama 2 on text, images and audio, including the adjusted rates that count model size.
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling (Gao et al., 2020)](https://arxiv.org/abs/2101.00027): the bits-per-byte formula and GPT-2/GPT-3 results in Table 2.
- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556): the loss formula whose constant term is the entropy of natural text.
- [Distilling the Knowledge in a Neural Network (Hinton, Vinyals and Dean, 2015)](https://arxiv.org/abs/1503.02531): training on soft targets with cross-entropy.
- [Perplexity of fixed-length models (Hugging Face docs)](https://huggingface.co/docs/transformers/perplexity): computing perplexity in practice, sliding windows and the tokenizer caveat.
- [Elements of Information Theory, chapter 2 (Cover and Thomas)](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X): the standard textbook treatment of entropy, relative entropy and Gibbs' inequality.
