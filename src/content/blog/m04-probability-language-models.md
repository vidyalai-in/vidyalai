---
title: "Probability and the Chain Rule of Language"
description: "How an LLM scores any text as a product of next-token probabilities, why it adds logs instead, and what sampling, temperature and top-p do to the numbers."
slug: "probability-language-models"
category: "llm-math"
tags: ["math", "sampling", "tokens", "transformers"]
series: "llm-math"
seriesOrder: 4
pubDate: 2026-10-23
---

## TL;DR

At every step, a language model gives each of its roughly 150,000 tokens a number between 0 and 1, and those numbers add up to 1. This article builds the probability you need to read them, starting from dice. Conditional probability leads to the chain rule: the probability of a whole text is the product of its next-token probabilities, so a model that predicts the next token can score any text. Those products get so small that computers round them to zero, which is why everyone adds log-probabilities instead. The last two steps show how a token is drawn from a distribution, and how temperature, top-k, top-p and min-p reshape the distribution first, each in one line of math.

## Why it matters

Qwen2.5-1.5B-Instruct, a small open model, gives the sentence "The cat sat on the mat." a probability of about 1 in 98 million. Shuffle the same words into "mat the on sat cat The." and the probability falls by a factor of about 47 billion billion.

Numbers this small are a problem for computers. Multiply the probabilities of 1,000 tokens at 0.1 each in standard 32-bit decimals and the answer is exactly 0, reached at token 46. Yet models score long texts all the time: in training, in evaluation, and every time they pick a word.

The math that makes it work is school level: fractions that add up to 1, one definition of conditional probability, and the rule that the log of a product is a sum of logs. With it, the sampling settings in every chatbot API stop being vague knobs. Each is a precise operation on a list of numbers.

## The intuition

Picture a choose-your-own-adventure book where every page ends with a list of possible next words and the odds of each. You roll for the first word, turn to that word's page, roll again, and so on. The chance of any particular story is the chance of every choice along its path, multiplied together. A story that takes likely turns gets a reasonable number; one that takes a strange turn on every page gets a tiny one.

A language model is that book. Writing means rolling dice on each page. Scoring a text someone else wrote means following its path and multiplying the odds you pass.

The analogy breaks in two places. A real book would need a page for every possible beginning of every possible text, far more pages than there are atoms in the universe; the model stores no pages and computes each list of odds on the spot from the text so far. And each list has about 152,000 entries, not three or four.

## Step 1: distributions and conditional probability

### Numbers that add up to 1

Roll a fair six-sided die. Each face has probability 1/6, about 0.167: six numbers, none negative, adding up to exactly 1. That's a probability distribution, a set of possible outcomes with a probability for each, where none is negative and they all add up to 1.

A next-token prediction is the same object with a bigger die. Here is a toy one, a rounded version of what a real model predicts after "I opened the fridge and took out a" (Step 5 has the real numbers):

| Next token | Probability |
|---|---|
| bottle | 0.50 |
| can | 0.30 |
| bowl | 0.15 |
| jar | 0.05 |
| Total | 1.00 |

In symbols, a distribution $$p$$ over a vocabulary $$V$$ (the set of all tokens the model knows) obeys two rules:

$$
p(x) \ge 0 \text{ for every token } x, \qquad \sum_{x \in V} p(x) = 1
$$

Here $$p(x)$$ is read "the probability of $$x$$", and the $$\sum$$ adds $$p(x)$$ up over every token in $$V$$. For Qwen2.5-1.5B, $$V$$ has 151,936 tokens, the `vocab_size` in its [config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json), so every prediction is 151,936 numbers that add up to 1.

### Joint probability: counting pairs

Sequences need two more ideas, and a tiny corpus (a collection of text) is enough to build both:

```text
the cat sat on the mat
the dog sat on the mat
the cat ate the fish
```

Mark the start of each sentence with `<s>` and the end with `</s>`, then list every pair of neighboring words: (`<s>`, the), (the, cat), (cat, sat), and so on. There are 20 pairs, and (the, cat) appears twice. Pick one pair at random, and the chance it's (the, cat) is

$$
p(\text{the, cat}) = \frac{2}{20} = 0.1
$$

That's a joint probability: the chance of two things together, here "the previous word is *the*" and "the next word is *cat*".

### Conditional probability: narrowing down

Now suppose you already know the previous word is "the". What's the chance the next word is "cat"? Six of the 20 pairs start with "the":

| Pair starting with "the" | Count | Share of the 6 |
|---|---|---|
| the, cat | 2 | 2/6 = 0.333 |
| the, mat | 2 | 2/6 = 0.333 |
| the, dog | 1 | 1/6 = 0.167 |
| the, fish | 1 | 1/6 = 0.167 |
| Total | 6 | 1.000 |

Knowing the first word shrinks the world from 20 pairs to 6, and "cat" has 2 of them, so the answer is one in three. That's a conditional probability, written $$p(\text{cat} \mid \text{the})$$ and read "the probability of *cat* given *the*".

The joint probabilities give the same answer. A random pair starts with "the" with probability 6/20 = 0.3, and $$0.1 / 0.3 = 1/3$$. That's the definition of conditional probability:

$$
p(B \mid A) = \frac{p(A, B)}{p(A)}
$$

Here $$A$$ and $$B$$ are any two events, and $$p(A, B)$$ is the chance both happen. In words: among the cases where $$A$$ happens, the share that also have $$B$$. The last column of the table adds up to 1, so a conditional probability is itself a distribution. A language model's output is exactly this: a distribution over the next token, given the text so far.

## Step 2: the chain rule of language

### From one pair to a whole sentence

Multiply both sides of the definition by $$p(A)$$ and it turns around:

$$
p(A, B) = p(A)\, p(B \mid A)
$$

The chance of both is the chance of the first, times the chance of the second given the first. Apply it again to three events, treating "$$A$$ and $$B$$" as one event:

$$
\begin{aligned}
p(A, B, C) &= p(A, B)\, p(C \mid A, B) \\
&= p(A)\, p(B \mid A)\, p(C \mid A, B)
\end{aligned}
$$

Keep going and you get the chain rule of probability. For a text made of tokens $$w_1, w_2, \ldots, w_n$$:

$$
p(w_1, \ldots, w_n) = \prod_{i=1}^{n} p(w_i \mid w_{<i})
$$

The $$\prod$$ (capital pi) is the multiplying version of $$\sum$$, and $$w_{<i}$$ means "all the tokens before position $$i$$". In words: the probability of a text is the probability of its first token, times the second given the first, times the third given the first two, and so on to the end.

This is no approximation. It follows from the definition, so it holds for any text and any distribution over texts. Bengio and colleagues opened [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (2003) with this product, noting that a language model "can be represented by the conditional probability of the next word given all the previous ones".

### The chain rule with counts

Count-based models, called n-gram models, cut the history short to keep their tables small. A bigram model conditions each word only on the one before it: $$p(w_i \mid w_{<i}) \approx p(w_i \mid w_{i-1})$$. That shortcut is the approximation; the product isn't. Here's "the cat sat on the mat" under the bigram model:

```text
p(the | <s>)  = 3/3 = 1
p(cat | the)  = 2/6 = 1/3
p(sat | cat)  = 1/2
p(on | sat)   = 2/2 = 1
p(the | on)   = 2/2 = 1
p(mat | the)  = 2/6 = 1/3
p(</s> | mat) = 2/2 = 1

product = 1 × 1/3 × 1/2 × 1 × 1 × 1/3 × 1 = 1/18 ≈ 0.0556
```

The same calculation in Python, for three sentences:

```python
from collections import Counter

corpus = ["the cat sat on the mat",
          "the dog sat on the mat",
          "the cat ate the fish"]

pairs, firsts = Counter(), Counter()
for line in corpus:
    words = ["<s>"] + line.split() + ["</s>"]    # start and end markers
    for a, b in zip(words, words[1:]):           # every (previous, next) pair
        pairs[(a, b)] += 1
        firsts[a] += 1

def p_next(b, a):                                # p(next = b | previous = a)
    return pairs[(a, b)] / firsts[a]

total = sum(pairs.values())
print(total, pairs[("the", "cat")] / total, firsts["the"] / total, p_next("cat", "the"))

for s in ["the cat sat on the mat", "the cat sat on the fish", "the dog ate the fish"]:
    words = ["<s>"] + s.split() + ["</s>"]
    prob = 1.0
    for a, b in zip(words, words[1:]):
        prob *= p_next(b, a)                     # chain rule, with a one-word memory
    print(f"{s:24} {prob:.4f}")
```

```text
20 0.1 0.3 0.3333333333333333
the cat sat on the mat   0.0556
the cat sat on the fish  0.0278
the dog ate the fish     0.0000
```

"the cat sat on the fish" never appears in the corpus but still gets a probability, because each of its pairs does. "the dog ate the fish" shows the weakness: "dog ate" was never seen, so one factor is 0, and one zero wipes out the whole product.

The `</s>` marker matters too. Without it, "the cat" would beat "the cat sat on the mat" just by stopping early. Making "end here" a token the model must predict means the probabilities of all complete texts, short and long, add up to 1. LLMs do the same with an end-of-text token.

### Why a model only needs the next token

Writing down a probability for every possible text is hopeless: with 151,936 tokens there are $$151{,}936^{10} \approx 6.6 \times 10^{51}$$ texts of just ten tokens. The chain rule says you don't have to. A model that gives one good next-token distribution for any prefix gives you everything:

- To score a text, run the model over it, read off each actual token's probability, and multiply.
- To generate, draw a token from the first distribution, append it, and draw from the next. Each draw samples one factor of the product, so the whole text is a sample from it.

That's why an LLM is trained to predict the next token and nothing else. [How a Large Language Model Works](/blog/how-llms-work/) describes the loop in words, and [Fundamentals Part 9](/blog/encoders-decoders-gpt/) explains the causal mask that lets a decoder compute every factor for a whole text in one pass. An LLM conditions on the full history $$w_{<i}$$, not just the last word, which is how it can give "the dog ate the fish" a sensible probability where the bigram model gave 0.

## Step 3: logs and log-probabilities

### Products of small numbers vanish

Every factor in the chain rule is at most 1, so the product shrinks with every token. Computers store decimals in floating-point formats that have a smallest positive value: about $$1.4 \times 10^{-45}$$ for 32-bit floats (float32). Anything smaller rounds to exactly 0, a failure called underflow.

```python
import numpy as np

p = np.full(1000, np.float32(0.1))        # 1,000 tokens, each with probability 0.1
print(np.prod(p))                          # multiply them all
print(np.argmax(np.cumprod(p) == 0) + 1)   # the token at which the running product hits 0
print(np.sum(np.log(p)))                   # add their logs instead
```

```text
0.0
46
-2302.5852
```

The true product is $$10^{-1000}$$, and float32 gives up at token 46. Switching to 64-bit floats only moves the wall to token 324, while the sum of logs on the third line has no wall at all.

### Logarithms from scratch

The natural logarithm of a positive number $$x$$, written $$\ln x$$, is the power you raise $$e \approx 2.718$$ to in order to get $$x$$. So $$\ln 1 = 0$$ (because $$e^0 = 1$$) and $$\ln 0.1 \approx -2.303$$ (because $$e^{-2.303} \approx 0.1$$). Every number between 0 and 1 has a negative log, and the smaller the number, the more negative the log. Machine learning code writes `log` and means $$\ln$$, and so does this series.

The useful property comes from the rule for powers, $$e^a \cdot e^b = e^{a+b}$$: multiplying two numbers adds their exponents. So

$$
\ln(a \cdot b) = \ln a + \ln b
$$

Check it on the toy distribution: $$\ln 0.5 + \ln 0.3 = -0.693 - 1.204 = -1.897$$, and $$\ln(0.5 \times 0.3) = \ln 0.15 = -1.897$$. Applied to the chain rule, the product becomes a sum:

$$
\ln p(w_1, \ldots, w_n) = \sum_{i=1}^{n} \ln p(w_i \mid w_{<i})
$$

Each term is a log-probability, or logprob. The 1,000 tokens above sum to -2,302.59, which says the product was $$e^{-2302.59} = 10^{-1000}$$ without ever storing it. And since $$\ln$$ always grows with its input, comparing log-probabilities picks the same winner as comparing probabilities.

### Scoring real sentences

Here are the opening numbers. The script loads Qwen2.5-1.5B-Instruct on a CPU and scores two sentences token by token. It puts the model's `<|endoftext|>` token first, as a blank page, so the first word gets a probability too. `log_softmax` turns the model's raw scores straight into log-probabilities; [Part 5](/blog/softmax-math/) covers how.

```python
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)

def score(text):
    start = tok.convert_tokens_to_ids("<|endoftext|>")   # a "blank page" token
    ids = torch.tensor([[start] + tok(text)["input_ids"]])
    with torch.no_grad():
        logits = model(ids).logits[0]                    # one row of 151,936 scores per position
    logp = torch.log_softmax(logits, dim=-1)             # scores -> log-probabilities
    total = 0.0
    for t in range(1, ids.shape[1]):
        lp = logp[t - 1, ids[0, t]].item()               # log p(token t | everything before it)
        total += lp
        print(f"{tok.decode(ids[0, t])!r:8} p = {math.exp(lp):<8.2g} log p = {lp:6.2f}")
    print(f"total log p = {total:.2f}\n")

score("The cat sat on the mat.")
score("mat the on sat cat The.")
```

```text
'The'    p = 0.0063   log p =  -5.07
' cat'   p = 0.00031  log p =  -8.09
' sat'   p = 0.055    log p =  -2.90
' on'    p = 0.8      log p =  -0.23
' the'   p = 0.91     log p =  -0.09
' mat'   p = 0.45     log p =  -0.80
'.'      p = 0.29     log p =  -1.22
total log p = -18.40

'mat'    p = 1.4e-05  log p = -11.15
' the'   p = 0.00036  log p =  -7.92
' on'    p = 0.00039  log p =  -7.85
' sat'   p = 0.00019  log p =  -8.57
' cat'   p = 1.2e-05  log p = -11.32
' The'   p = 0.00092  log p =  -6.99
'.'      p = 5e-05    log p =  -9.90
total log p = -63.69
```

"cat" as the second word of a text is a long shot (0.00031), since a text could be about anything. Once "The cat sat" is down, " on" is 80% likely, " the" 91% and " mat" 45%. The total, -18.40, means $$p = e^{-18.40} \approx 1.0 \times 10^{-8}$$, about 1 in 98 million.

The scrambled sentence never gets a single token above 0.001, and totals -63.69. A gap of 45.29 in log space is a ratio of $$e^{45.29} \approx 4.7 \times 10^{19}$$ in probability, the "47 billion billion" from the opening. Logs turn ratios into differences, which are far easier to read.

![A vertical chain. A start token, the blank page, leads to "The" with probability 0.0063, then to "The cat" with probability 0.00031 of cat given The, then to "The cat sat" with probability 0.055, then four more steps to "The cat sat on the mat." The seven log-probabilities are summed to -18.40, and the sentence probability is e to the power -18.40, about 1 in 98 million.](/diagrams/m04-chain-rule.svg "The chain rule in action: each step multiplies in one next-token probability, and in log space each step adds one log-probability.")

## Step 4: drawing a token from a distribution

### The inverse-CDF trick

Generating text means picking a token at random, each with a chance equal to its probability. Computers come with one kind of randomness built in: a uniform random number $$u$$ between 0 and 1, where every value is equally likely. To turn $$u$$ into a token, line the probabilities up end to end. The running totals are called the cumulative distribution function, or CDF:

| Token | Probability | Running total (CDF) | Range of u that picks it |
|---|---|---|---|
| bottle | 0.50 | 0.50 | 0 to 0.50 |
| can | 0.30 | 0.80 | 0.50 to 0.80 |
| bowl | 0.15 | 0.95 | 0.80 to 0.95 |
| jar | 0.05 | 1.00 | 0.95 to 1 |

Pick the first token whose running total is bigger than $$u$$. If $$u = 0.62$$, it falls between 0.50 and 0.80, so the pick is "can". Each token owns a slice of the line from 0 to 1 exactly as long as its probability, so $$u$$ lands in the 0.30-long slice 30% of the time. This is called inverse transform sampling.

```python
import numpy as np

tokens = ["bottle", "can", "bowl", "jar"]
p = np.array([0.50, 0.30, 0.15, 0.05])
cdf = np.cumsum(p)                       # running totals: 0.5, 0.8, 0.95, 1.0
print("cdf:", cdf)

def sample(u):                           # u is a uniform random number in [0, 1)
    return tokens[np.searchsorted(cdf, u, side="right")]

print(sample(0.62))                      # 0.5 <= 0.62 < 0.8, so "can"

rng = np.random.default_rng(0)
draws = [sample(u) for u in rng.random(100_000)]
for t in tokens:
    print(t, draws.count(t) / len(draws))
```

```text
cdf: [0.5  0.8  0.95 1.  ]
can
bottle 0.50098
can 0.30086
bowl 0.14798
jar 0.05018
```

Over 100,000 draws, each token's share lands within about 0.002 of its probability.

### Greedy decoding is not "most likely text"

The alternative to sampling is greedy decoding: always take the most likely token, written $$\arg\max_x p(x)$$, "the $$x$$ that makes $$p(x)$$ largest". It doesn't produce the most likely text, and the chain rule shows why. An illustrative two-step choice:

```text
Step 1:  "can" 0.6,  "bottle" 0.4
Step 2 after "can":     best next token has 0.3   ->  0.6 × 0.3 = 0.18
Step 2 after "bottle":  best next token has 0.9   ->  0.4 × 0.9 = 0.36
```

Greedy takes "can" and ends with probability 0.18, while the path through "bottle" has 0.36. A locally best choice can lead to a part of the tree where every continuation is weak. Finding the truly most likely text means searching a tree that grows exponentially; [Holtzman et al.](https://arxiv.org/abs/1904.09751) call finding "the optimum argmax sequence" from a Transformer "not tractable". Beam search, covered in [Fundamentals Part 9](/blog/encoders-decoders-gpt/), is the usual partial search.

### Expected value, briefly

One more idea comes back later in the series. The expected value of a number that depends on a random outcome is its probability-weighted average:

$$
\mathbb{E}[X] = \sum_{x} p(x)\, x
$$

Here $$X$$ is the random number and $$\mathbb{E}$$ stands for "expected". For a fair die, $$\mathbb{E}[X] = \tfrac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = 3.5$$. No roll gives 3.5, but the average of many rolls approaches it: 100,000 simulated rolls averaged 3.498.

Training uses this fact. An LLM's training loss is the expected value of $$-\ln p(\text{next token})$$ over real text, estimated by averaging over billions of tokens. For "The cat sat on the mat." the average is $$18.40 / 7 = 2.63$$ per token. [Part 6](/blog/cross-entropy-perplexity/) turns that number into cross-entropy and perplexity.

## Step 5: reshaping the distribution before the draw

Every sampling setting changes the distribution before $$u$$ is drawn. [Fundamentals Part 9](/blog/encoders-decoders-gpt/) shows what they do to text and the order libraries apply them in; here is the math, on a real distribution. After "I opened the fridge and took out a", Qwen2.5-1.5B-Instruct's top six are " bottle" (15.38%), " can" (9.01%), " bowl" (4.07%), " jar" (3.39%), " bag" (2.71%) and " box" (2.63%). The other 151,930 tokens share the remaining 62.8%.

### Temperature is a power

The model's raw output is one score per token, a logit, and softmax turns the scores into probabilities (Part 5 derives it). Temperature $$T$$ divides every logit by $$T$$ first. With $$z_x$$ the logit of token $$x$$:

$$
p_T(x) = \frac{e^{z_x / T}}{\sum_{y} e^{z_y / T}}
$$

That's how libraries implement it, and how Holtzman et al. write it. Since $$e^{z/T} = (e^{z})^{1/T}$$, and $$e^{z_x}$$ is the token's probability times a constant that cancels out, temperature raises every probability to the power $$1/T$$ and renormalizes (divides by the total so the numbers add up to 1 again):

$$
p_T(x) = \frac{p(x)^{1/T}}{\sum_{y} p(y)^{1/T}}
$$

It's easiest to read on the ratio of two tokens, where the total cancels:

$$
\frac{p_T(a)}{p_T(b)} = \left( \frac{p(a)}{p(b)} \right)^{1/T}
$$

At $$T = 1$$, " bottle" is 1.71 times as likely as " can". At $$T = 0.5$$ the power is 2, so it becomes $$1.71^2 = 2.92$$ times as likely: every gap gets squared, and the leader pulls away. At $$T = 2$$ the power is 1/2, a square root, and every gap shrinks. The extremes follow:

- As $$T \to 0$$, the power $$1/T$$ grows without limit, any ratio above 1 blows up, and the top token takes all the probability. Sampling becomes greedy decoding.
- As $$T \to \infty$$, the power goes to 0, and any positive number to the power 0 is 1. Every token gets $$1/151{,}936 \approx 0.00066\%$$, and the model's preferences are gone.

On the real distribution:

| Temperature | " bottle" | " can" | " bowl" | " jar" | Tokens to reach 90% |
|---|---|---|---|---|---|
| 0.01 | 100.00% | 0.00% | 0.00% | 0.00% | 1 |
| 0.5 | 57.89% | 19.84% | 4.05% | 2.81% | 8 |
| 0.7 | 35.15% | 16.36% | 5.26% | 4.05% | 30 |
| 1 | 15.38% | 9.01% | 4.07% | 3.39% | 193 |
| 1.5 | 3.81% | 2.67% | 1.57% | 1.39% | 5,574 |
| 2 | 1.04% | 0.79% | 0.53% | 0.49% | 31,498 |

The last column counts the top tokens needed to cover 90% of the probability. Going from $$T = 1$$ to $$T = 2$$ multiplies it by 163, because a high temperature lifts the whole tail of near-zero tokens at once.

### Truncation is conditioning

Top-k, top-p and min-p all pick a set $$S$$ of tokens to keep and renormalize over it. With $$p(S)$$ the total probability of the kept tokens:

$$
p(x \mid x \in S) =
\begin{cases}
p(x) / p(S) & \text{if } x \in S \\
0 & \text{otherwise}
\end{cases}
$$

That's the definition of conditional probability from Step 1, with $$B$$ = "the token is $$x$$" and $$A$$ = "the token is in $$S$$". Truncating and renormalizing is conditioning on "the token is one we kept", the same move as narrowing 20 pairs to the 6 that start with "the". The nucleus sampling paper writes this rescaling as its equation 3. The rules differ only in how they choose $$S$$:

- Top-k keeps the $$k$$ most likely tokens.
- Top-p, or nucleus sampling ([Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)), keeps "the smallest set" of top tokens whose probabilities add up to at least $$p$$.
- Min-p ([Nguyen et al., 2024](https://arxiv.org/abs/2407.01082)) keeps every token with $$p(x) \ge p_{\min} \cdot \max_y p(y)$$, at least a fixed fraction of the top token's probability.

On the fridge distribution at $$T = 1$$:

| Rule | Tokens kept | Kept mass p(S) | " bottle" after renormalizing |
|---|---|---|---|
| None | 151,936 | 1.0000 | 15.38% |
| Top-k, k = 20 | 20 | 0.5878 | 26.17% |
| Top-p, p = 0.9 | 193 | 0.9001 | 17.09% |
| Min-p, 0.1 (cut-off 1.54%) | 12 | 0.4967 | 30.97% |

Each survivor is divided by the kept mass: for top-k, $$0.1538 / 0.5878 = 0.2617$$. All survivors grow by the same factor, $$1/p(S)$$, so their ratios to each other don't change. Truncation removes options without reordering the rest. Temperature does the opposite: it keeps every option and changes the ratios.

Temperature and top-p together, continuing from the scoring script:

```python
start = tok.convert_tokens_to_ids("<|endoftext|>")
ids = torch.tensor([[start] + tok("I opened the fridge and took out a")["input_ids"]])
with torch.no_grad():
    z = model(ids).logits[0, -1]             # 151,936 scores for the next token

def top_p_filter(p, top_p):                  # keep the smallest set of top tokens with mass >= top_p
    sorted_p, order = p.sort(descending=True)
    n = int((sorted_p.cumsum(0) < top_p).sum()) + 1
    kept = torch.zeros_like(p)
    kept[order[:n]] = sorted_p[:n]
    return kept / kept.sum()                 # renormalize: divide by the kept mass

for T in [0.5, 1.0, 2.0]:
    p = torch.softmax(z / T, dim=-1)         # temperature, then scores -> probabilities
    q = top_p_filter(p, 0.9)
    print(f"T = {T}: top token {p.max():.3f}, top-p 0.9 keeps {(q > 0).sum():>6} tokens, "
          f"top token after {q.max():.3f}")
```

```text
T = 0.5: top token 0.579, top-p 0.9 keeps      8 tokens, top token after 0.638
T = 1.0: top token 0.154, top-p 0.9 keeps    193 tokens, top token after 0.171
T = 2.0: top token 0.010, top-p 0.9 keeps  31498 tokens, top token after 0.012
```

Because temperature runs first, it changes which tokens top-p keeps: 8 or 31,498 for the same $$p = 0.9$$. The top-k, top-p and min-p counts match Hugging Face Transformers' own filter classes.

## In the real world

- Models ship with sampling settings. Qwen2.5-1.5B-Instruct's [generation_config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/generation_config.json) sets `do_sample: true`, `temperature: 0.7`, `top_k: 20`, `top_p: 0.8` and `repetition_penalty: 1.1`. Hugging Face's [default is greedy search](https://huggingface.co/docs/transformers/generation_strategies), so this file is what turns sampling on. Applied in Hugging Face's order to the fridge prompt, these settings keep 7 of 151,936 tokens: " bottle" 50.9%, " can" 23.7%, " bowl" 7.6%, " jar" 5.9%, " bag" 4.3%, " box" 4.1% and " sandwich" 3.6%. The repetition penalty didn't change those numbers, since no candidate appears in the prompt.
- Temperature 0 would divide by zero, so libraries special-case it. Hugging Face Transformers requires "a strictly positive float" and points to `do_sample=False` for greedy decoding. vLLM [treats any temperature below 0.00001 as greedy](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py), taking the $$T \to 0$$ limit by hand.
- Fast servers skip the CDF. vLLM's [sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py) avoids `torch.multinomial` because it "causes CPU-GPU synchronization". It draws an exponentially distributed random number $$q_x$$ for every token and picks the largest $$p(x) / q_x$$, a known trick that lands on each token with probability exactly $$p(x)$$ using only a division and a maximum, which GPUs do in parallel. On the toy distribution, 200,000 draws this way came out at 50.2%, 29.8%, 15.1% and 5.0%.
- Log-probabilities double as a confidence signal. OpenAI's Whisper speech recognizer ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)) starts at temperature 0 and raises it by 0.2, up to 1.0, when "the average log probability over the generated tokens is lower than −1", among other checks.
- Products favor short outputs. Every factor is below 1, so longer texts almost always have lower total probability, and a search for the most likely translation drifts short. The [original Transformer's](/blog/attention-is-all-you-need/) beam search corrected for this by dividing each candidate's log-probability by a function of its length.

## Common misconceptions

- **"The model computes the probability of a whole answer at once."** It outputs one next-token distribution per position. A text's probability is assembled from them by the chain rule, one factor per token.
- **"Greedy decoding gives the most likely text."** It gives the most likely token at each step. In the two-step example, greedy found a path with probability 0.18 when one with 0.36 existed.
- **"Top-p = 0.9 throws away 10% of the vocabulary."** It throws away the least likely 10% of the probability, which can be almost the whole vocabulary. On the fridge prompt at temperature 1, it dropped 151,743 of 151,936 tokens.
- **"A low log-probability means the text is wrong."** It means the text surprises the model. "The cat sat on the hat." scored -22.83 against -18.40 for "mat", about 84 times less likely, and it's a fine sentence. The model has just read "on the mat" far more often.
- **"Temperature lets the model consider new ideas."** Raising every probability to the power $$1/T$$ keeps the ranking of tokens fixed. It only stretches or squeezes the gaps.

## Key takeaways

- A probability distribution is a list of non-negative numbers that add up to 1. Every LLM step produces one over the whole vocabulary: 151,936 numbers for Qwen2.5.
- Conditional probability is joint probability divided by the probability of what you know. Applied repeatedly, it gives the chain rule: a text's probability is the product of its next-token probabilities. So a next-token predictor can score and generate any text.
- Those products underflow fast, so models add log-probabilities instead. "The cat sat on the mat." scores -18.40; scrambled, it scores -63.69.
- Sampling turns a uniform random number into a token by lining the probabilities up end to end. Greedy decoding takes the top token, which doesn't guarantee the most likely text.
- Temperature raises every probability to the power 1/T: near 0 it's greedy, very large it's uniform. Top-k, top-p and min-p keep a set of tokens and renormalize, which is conditioning on "the token is in the set".

## What's next

This part treated softmax as a black box that turns scores into probabilities. Part 5 opens it up: why it uses the exponential, why adding the same number to every score changes nothing, how naive code overflows, and the one-line max trick that fixes it. Along the way comes log-sum-exp, the function behind every log-probability in this article, and online softmax, the idea that lets FlashAttention skip storing the full table of attention scores.

## Further reading

- [The Curious Case of Neural Text Degeneration (Holtzman et al., 2019)](https://arxiv.org/abs/1904.09751): defines nucleus (top-p) sampling, writes truncation as renormalization (equation 3) and temperature as a rescaled softmax (equation 4).
- [A Neural Probabilistic Language Model (Bengio et al., 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf): opens with the chain rule of language and the n-gram approximation, then replaces count tables with a neural network.
- [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs (Nguyen et al., 2024)](https://arxiv.org/abs/2407.01082): the min-p rule, which scales the cut-off by the top token's probability.
- [Hierarchical Neural Story Generation (Fan et al., 2018)](https://arxiv.org/abs/1805.04833): an early use of top-k sampling for long-form writing.
- [Qwen2.5-1.5B-Instruct generation_config.json (Hugging Face)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/generation_config.json): a real model's shipped sampling settings.
- [vLLM's sampler source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py): top-k, top-p and exponential-noise sampling as production GPU code.
- [Generation strategies (Hugging Face docs)](https://huggingface.co/docs/transformers/generation_strategies): greedy search, sampling and beam search in Transformers.
- [Mathematics for Machine Learning, chapter 6 (Deisenroth, Faisal & Ong, 2020)](https://mml-book.github.io/): a free textbook chapter on probability and distributions, including the sum and product rules used here.
