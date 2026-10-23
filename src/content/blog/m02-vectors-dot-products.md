---
title: "Vectors and Dot Products: Length, Angle and Meaning"
description: "Length, angle, cosine and projection built from Pythagoras, why random vectors in 4,096 dimensions are nearly perpendicular, and real embedding arithmetic."
slug: "vectors-dot-products"
category: "llm-math"
tags: ["math", "embeddings", "vector-search", "interpretability", "transformers"]
series: "llm-math"
seriesOrder: 2
pubDate: 2026-10-23
---

## TL;DR

A vector has a length and a direction, and the dot product measures both at once: it equals the two lengths multiplied together, times the cosine of the angle between the vectors. This article builds that fact from Pythagoras and the law of cosines, with numbers you can check by hand, and gets cosine similarity, unit vectors and projection out of it. Then it shows why two random vectors in high dimensions are almost always nearly perpendicular, with a cosine spread of about 1 over the square root of the dimension. It ends with real embedding arithmetic on Qwen2.5-1.5B, including what king minus man plus woman actually gives.

## Why it matters

Pick two directions at random in the 4,096-dimensional space that Llama 3.1 8B uses for its vectors. The angle between them will be between 88.2 and 91.8 degrees about 95% of the time. In the flat, two-dimensional world of a sheet of paper, the same experiment gives any angle at all.

That fact is why a model can pack far more concepts than it has dimensions, why attention divides its scores by a square root, and why a cosine of 0.3 between two word vectors can be a strong signal.

[Part 1](/blog/math-for-llms/) showed that more than 99.9% of an LLM's arithmetic is dot products. This part asks what a dot product measures, using Pythagoras and a little trigonometry.

## The intuition

Imagine walking across a field while a steady wind blows from the west. Walk straight east and all of your walk goes along the wind; walk north and none of it does; walk northeast and part of it does, depending on the angle. The dot product of your walk with the wind is that part, scaled by how far you walked and how strong the wind is: "how much one arrow points along another".

The analogy breaks in high dimensions. A field has only two independent directions, so a random walk often goes roughly with the wind. In 4,096 dimensions a random arrow almost never points along any particular direction. That surprise is Step 4.

## Step 1: length, from Pythagoras to d dimensions

### Two dimensions

A vector with two entries, like $$\mathbf{a} = [3, 4]$$, can be drawn as an arrow from the origin: 3 steps right, 4 steps up. Its length is the hypotenuse of a right triangle with sides 3 and 4, so Pythagoras gives it:

$$
\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

### Three dimensions and beyond

Add a third entry, $$[2, 3, 6]$$, and apply Pythagoras twice. The diagonal along the floor has length $$\sqrt{2^2 + 3^2} = \sqrt{13}$$, and the arrow is the hypotenuse of a second right triangle with that diagonal as its base and height 6:

$$
\sqrt{13 + 6^2} = \sqrt{4 + 9 + 36} = \sqrt{49} = 7
$$

Each new dimension adds one more right angle and one more square. So for a vector with $$d$$ entries, square every entry, add them, and take the square root:

$$
\lVert \mathbf{a} \rVert = \sqrt{a_1^2 + a_2^2 + \cdots + a_d^2} = \sqrt{\sum_{i=1}^{d} a_i^2}
$$

The double bars $$\lVert \mathbf{a} \rVert$$ are read "the norm of a", the mathematician's word for length. You can't picture a 4,096-dimensional arrow, but you can compute its length.

This links back to Part 1: the sum of squares is the vector dotted with itself, $$\mathbf{a} \cdot \mathbf{a} = 9 + 16 = 25$$ for [3, 4]. So length is the square root of a dot product.

### Unit vectors and normalization

A unit vector has length exactly 1. Dividing any vector by its own length makes one, a step called normalization:

$$
\hat{\mathbf{a}} = \frac{\mathbf{a}}{\lVert \mathbf{a} \rVert} = \frac{[3, 4]}{5} = [0.6,\ 0.8]
$$

The hat on $$\hat{\mathbf{a}}$$ marks a unit vector. Check it: $$0.6^2 + 0.8^2 = 0.36 + 0.64 = 1$$. Every vector is its length times its unit vector, so length and direction are separate pieces of information, and normalizing keeps only the direction.

In a real model, lengths vary. The 151,665 token vectors in Qwen2.5-1.5B's embedding table have an average length of 1.01, but they range from 0.42 to 1.56, a factor of almost 4. That spread is one reason search systems normalize.

## Step 2: the dot product and the angle

### By numbers first

Take $$\mathbf{a} = [3, 4]$$ and $$\mathbf{b} = [4, 3]$$. Both have length 5. Their dot product, from Part 1, is

```text
a · b = 3×4 + 4×3 = 24
```

What does 24 have to do with the angle between them? Draw both arrows from the origin and join their tips. That makes a triangle with sides $$\lVert \mathbf{a} \rVert = 5$$, $$\lVert \mathbf{b} \rVert = 5$$ and a third side running from the tip of b to the tip of a, which is the vector $$\mathbf{a} - \mathbf{b} = [-1, 1]$$, of length $$\sqrt{2}$$.

The law of cosines from school trigonometry relates the three sides of any triangle to the angle $$\theta$$ (the Greek letter theta) between two of them. It's Pythagoras with a correction for angles that aren't 90 degrees:

$$
c^2 = p^2 + q^2 - 2pq\cos\theta
$$

Here $$p$$ and $$q$$ are the two sides that meet at the angle, 5 and 5, and $$c$$ is the side opposite it, $$\sqrt{2}$$. Plug in:

$$
2 = 25 + 25 - 2 \times 5 \times 5 \times \cos\theta
$$

So $$50\cos\theta = 48$$, and $$\cos\theta = 0.96$$. The dot product divided by both lengths gives the same number: $$24 / (5 \times 5) = 0.96$$.

A protractor agrees. Arrow a sits 53.13 degrees above the horizontal and arrow b 36.87 degrees, so the angle between them is 16.26 degrees, whose cosine is 0.96.

### The general proof

The match holds for any two vectors. To see why, expand the squared length of $$\mathbf{a} - \mathbf{b}$$ entry by entry, using $$(x - y)^2 = x^2 - 2xy + y^2$$:

$$
\lVert \mathbf{a} - \mathbf{b} \rVert^2 = \sum_{i=1}^{d} (a_i - b_i)^2
$$

$$
= \sum_{i} a_i^2 \;-\; 2\sum_{i} a_i b_i \;+\; \sum_{i} b_i^2
$$

$$
= \lVert \mathbf{a} \rVert^2 - 2\,(\mathbf{a} \cdot \mathbf{b}) + \lVert \mathbf{b} \rVert^2
$$

The law of cosines says the same squared length equals $$\lVert \mathbf{a} \rVert^2 + \lVert \mathbf{b} \rVert^2 - 2 \lVert \mathbf{a} \rVert \lVert \mathbf{b} \rVert \cos\theta$$. Put the two side by side and everything cancels except the middle terms:

$$
\mathbf{a} \cdot \mathbf{b} = \lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert \cos\theta
$$

In words: the dot product is the length of a, times the length of b, times the cosine of the angle between them. The left side is arithmetic (multiply and add) and the right side is geometry (lengths and an angle), so the equation lets you move between the two.

It holds in any number of dimensions, because two arrows from the origin always lie in one flat plane, even inside a 4,096-dimensional space. In high dimensions this is in fact how "angle" is defined.

### What the sign tells you

Because lengths are never negative, the sign of the dot product is the sign of the cosine:

| Vectors | Dot product | Cosine | Angle |
|---|---|---|---|
| [3, 4] and [4, 3] | 24 | 0.96 | 16.3° |
| [3, 4] and [-4, 3] | -12 + 12 = 0 | 0 | 90° |
| [3, 4] and [-3, -4] | -9 - 16 = -25 | -1 | 180° |

Positive means the arrows lean the same way, zero means perpendicular (also called orthogonal), negative means they lean apart. [Fundamentals Part 4](/blog/embeddings-and-logits/) stated these rules; now you know where they come from.

![A top-down concept map. Pythagoras gives the length of a vector, which leads to the law of cosines. The law of cosines and the dot product (multiply, then add) both feed into the central fact that a dot b equals the length of a times the length of b times the cosine of the angle. From that fact, one branch goes to cosine similarity, dividing out both lengths, which leads to retrieval and analogies (nearest by cosine) and to random vectors in d dimensions having a cosine of about 0 plus or minus 1 over the square root of d, which in turn gives room for many nearly perpendicular directions. The other branch goes to projection, the shadow of a on b, which leads to reading a direction: logits, probes and offsets.](/diagrams/m02-dot-product-map.svg "How the ideas in this article connect: one equation, two ways to use it.")

Everything below is one equation, the dot product as lengths times a cosine, used in two ways: divide the lengths out to compare directions, or keep one length to measure a shadow.

## Step 3: cosine similarity and projection

### Cosine similarity: direction only

Rearrange that equation and you get the cosine directly:

$$
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert} = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}
$$

That's cosine similarity: the dot product with both lengths divided out, which is the same as the dot product of the two unit vectors. It runs from -1 to 1 and ignores length entirely. [Fundamentals Part 4](/blog/embeddings-and-logits/) used it to show that GPT-2's " Monday" and " Tuesday" point almost the same way.

Why ignore length? Because length mixes in things you may not want to compare. In Qwen's table, the shortest row belongs to the rare symbol "⭞" and the longest to "edException", a fragment of programming code; neither length says much about meaning. If you ranked Qwen's tokens by raw dot product with a query, the rows of length 1.56 would get a nearly fourfold advantage over the rows of length 0.42 for the same direction. Cosine removes that.

Search systems reason the same way. A text embedding model turns each document into a vector, and retrieval finds the documents whose vectors have the highest cosine with the query's, as [BM25 vs Dense vs Hybrid Search](/blog/bm25-dense-hybrid-search/) and [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/) describe. Most systems normalize every vector to length 1 once, at indexing time, so a plain dot product is the cosine.

Normalizing also makes distance and cosine agree. For unit vectors the expansion from Step 2 becomes

$$
\lVert \hat{\mathbf{a}} - \hat{\mathbf{b}} \rVert^2 = 1 - 2\cos\theta + 1 = 2 - 2\cos\theta
$$

For our pair with cosine 0.96, the squared distance is $$2 - 1.92 = 0.08$$. Ranking by nearest distance and by highest cosine gives the same list.

### Projection: the shadow of one vector on another

Now keep one length. Dot $$\mathbf{a}$$ with the unit vector $$\hat{\mathbf{b}}$$:

$$
\mathbf{a} \cdot \hat{\mathbf{b}} = \lVert \mathbf{a} \rVert \cos\theta
$$

That's the length of a's shadow on the line through b, if light shines straight down onto that line. It's called the scalar projection of a onto b, and it's the precise meaning of "how much a points along b".

By hand, with $$\mathbf{a} = [3, 4]$$ and $$\mathbf{b} = [4, 3]$$, so $$\hat{\mathbf{b}} = [0.8, 0.6]$$:

```text
shadow length       a · b̂ = 3×0.8 + 4×0.6           = 4.8
shadow as a vector  4.8 × [0.8, 0.6]                 = [3.84, 2.88]
what's left over    [3, 4] - [3.84, 2.88]            = [-0.84, 1.12]
check: leftover · b̂ = -0.84×0.8 + 1.12×0.6 = -0.672 + 0.672 = 0
```

The leftover is perpendicular to b, so a splits cleanly into a part along b and a part across it. Pythagoras checks the split: $$4.8^2 + 1.4^2 = 23.04 + 1.96 = 25$$, the squared length of a. The shadow is 4.8 of a's length 5, which is $$5 \times 0.96$$, length times cosine.

This turns the plain dot product into something you can read. For any b,

$$
\mathbf{a} \cdot \mathbf{b} = \lVert \mathbf{b} \rVert \times (\text{shadow of } \mathbf{a} \text{ on } \mathbf{b})
$$

A logit, the score a model gives each vocabulary token, is the final hidden vector dotted with that token's row of the output table: the row's length times the hidden vector's shadow on it. A linear probe from [Fundamentals Part 16](/blog/llm-interpretability/) reads a concept the same way, as the shadow of an activation on one learned direction. And subtracting a shadow, as in the "leftover" line, removes a direction from a vector.

### A real direction in Qwen2.5-1.5B

Here's projection on a real model. Qwen2.5-1.5B has a [config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) with `hidden_size` 1,536 and `vocab_size` 151,936; 151,665 of those rows are real tokens, and the remaining 271 are padding. We loaded its embedding table on a CPU, normalized every row to length 1, and built one direction: the unit vector along " woman" minus " man". Then we measured the shadow of other tokens on it:

| Token | Shadow on the woman-minus-man direction | Its partner | Partner's shadow |
|---|---|---|---|
| " queen" | +0.248 | " king" | -0.032 |
| " actress" | +0.349 | " actor" | +0.046 |
| " mother" | +0.192 | " father" | +0.009 |
| " aunt" | +0.209 | " uncle" | -0.013 |
| " table" | -0.027 | " Monday" | +0.019 |

Every female word casts a clearly positive shadow; its male partner and the neutral words sit near zero. Nobody labeled any of these words. Training put a usable gender direction in the table, and a single dot product reads it.

## Step 4: high dimensions, where random means perpendicular

### The experiment

The snippet below draws 10,000 pairs of random vectors (each entry from a standard bell curve, which gives directions spread evenly in every way) and measures the cosine of each pair:

```python
import numpy as np

rng = np.random.default_rng(0)
for d in [2, 128, 4096]:
    a = rng.standard_normal((10_000, d))    # 10,000 random vectors
    b = rng.standard_normal((10_000, d))    # and 10,000 more to pair with them
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    angle = np.degrees(np.arccos(cos))
    lo, hi = np.percentile(angle, [2.5, 97.5])
    print(f"d={d:<5} spread of cosine {cos.std():.4f}   1/sqrt(d) {d ** -0.5:.4f}   "
          f"95% of angles in {lo:5.1f} to {hi:5.1f} degrees")
```

```text
d=2     spread of cosine 0.7046   1/sqrt(d) 0.7071   95% of angles in   4.6 to 175.1 degrees
d=128   spread of cosine 0.0895   1/sqrt(d) 0.0884   95% of angles in  80.0 to 100.2 degrees
d=4096  spread of cosine 0.0158   1/sqrt(d) 0.0156   95% of angles in  88.2 to  91.8 degrees
```

The spread is the standard deviation, the typical distance from the average (here 0). In 2 dimensions, angles cover almost the whole range from 0 to 180 degrees. In 4,096 dimensions, 95% are within 1.8 degrees of a right angle, and the spread follows $$1/\sqrt{d}$$ almost exactly.

### Why: a variance argument

Write out the dot product of two random vectors whose entries are independent, each with average 0 and spread 1:

$$
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_d b_d
$$

Each product $$a_i b_i$$ is as likely to be positive as negative, so its average is 0. Its variance is the average of its square, $$a_i^2 b_i^2$$, and because the two entries are independent and each squared entry averages 1, that's $$1 \times 1 = 1$$. Variance is the square of the spread, and it has one handy property: for independent pieces, variances add. So the sum of $$d$$ terms has variance $$d$$ and spread $$\sqrt{d}$$.

The lengths behave too. The squared length $$\lVert \mathbf{a} \rVert^2 = a_1^2 + \cdots + a_d^2$$ is a sum of $$d$$ terms that each average 1, so it lands close to $$d$$ and the length close to $$\sqrt{d}$$. We checked: at $$d = 4096$$, the length divided by $$\sqrt{d}$$ averaged 1.0002 with a spread of only 0.011. Then the cosine is roughly

$$
\cos\theta \approx \frac{\mathbf{a} \cdot \mathbf{b}}{\sqrt{d} \times \sqrt{d}} = \frac{\mathbf{a} \cdot \mathbf{b}}{d}
$$

and dividing something of spread $$\sqrt{d}$$ by $$d$$ leaves spread $$\sqrt{d}/d = 1/\sqrt{d}$$. Positive and negative terms in the dot product mostly cancel, while the lengths grow steadily.

By hand for Llama's $$d = 4096$$: $$1/\sqrt{4096} = 1/64 \approx 0.0156$$. A cosine of 0.0156 is an angle of 89.1 degrees, and two spreads, 0.031, is 88.2 degrees. That's the 88.2 to 91.8 degree band in the output.

The same argument, almost word for word, is in the [original Transformer paper](https://arxiv.org/abs/1706.03762). Its footnote 4 assumes the entries of a query q and a key k are "independent random variables with mean 0 and variance 1. Then their dot product ... has mean 0 and variance $$d_k$$." That's why attention divides its scores by $$\sqrt{d_k}$$, the square root of the key length: it brings the spread back to about 1. Part 11 works through the consequences.

### How many nearly perpendicular directions fit?

In $$d$$ dimensions you can have only $$d$$ exactly perpendicular directions, like the axes. But if "nearly perpendicular" is good enough, the count explodes.

Here's a rough estimate. Measured in spreads, a cosine of $$\varepsilon$$ (the Greek letter epsilon, a small allowed tolerance) is $$\varepsilon\sqrt{d}$$ spreads from zero. The bell curve makes a value that many spreads out rare: the chance is at most about

$$
2e^{-d\varepsilon^2/2}
$$

Pick $$N$$ random directions. They form about $$N^2/2$$ pairs, so the expected number of pairs with a cosine bigger than $$\varepsilon$$ is about $$N^2 e^{-d\varepsilon^2/2}$$. While that stays below 1, a random pick usually has no bad pairs at all. Solving for $$N$$:

$$
N \approx e^{d\varepsilon^2/4}
$$

The dimension sits in the exponent. For $$d = 4096$$ and a tolerance of 0.1 (an angle of at least 84.3 degrees), that's $$e^{10.24}$$, about 28,000 directions. Loosen the tolerance to 0.2 (78.5 degrees) and it's $$e^{40.96}$$, about $$6 \times 10^{17}$$. That's what random picks achieve, not a limit; careful placement does at least as well.

To check, we drew 10,000 random unit directions and found the largest cosine among all 49,995,000 pairs. Turned around, the formula predicts a worst cosine of $$\sqrt{4\ln N / d}$$, where $$\ln$$ is the natural logarithm:

| Dimensions | Predicted worst cosine | Measured worst cosine | Measured angle |
|---|---|---|---|
| 128 | 0.536 | 0.468 | 62.1° |
| 768 | 0.219 | 0.202 | 78.4° |
| 1,536 | 0.155 | 0.150 | 81.4° |
| 4,096 | 0.095 | 0.090 | 84.8° |

The estimate runs slightly high and gets closer as the dimension grows. In 4,096 dimensions, no two of 10,000 random directions come closer than 84.8 degrees.

This is the room that superposition uses: [Fundamentals Part 16](/blog/llm-interpretability/) showed that models store more features than they have dimensions, each as its own direction. Anthropic's [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Elhage et al., 2022) puts it this way: "Although it's only possible to have n orthogonal vectors in an n-dimensional space, it's possible to have exp(n) many 'almost orthogonal'" vectors, pointing to the Johnson-Lindenstrauss lemma for the formal version. The catch, which Fundamentals Part 16 measured, is that small overlaps add up when many features are active at once, so the trick works only when features are rare.

## Step 5: embedding arithmetic on a real model

Back to Qwen2.5-1.5B's embedding table. This snippet loads it on a CPU, normalizes the rows, and finds nearest neighbors by cosine. It downloads the 3.1 GB model file once, reads only the embedding table (0.93 GB as 32-bit numbers), and ran in under 5 seconds on our machine with the file already downloaded:

```python
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

repo = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(repo)
with safe_open(hf_hub_download(repo, "model.safetensors"), "pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").float().numpy()
E = E[:len(tok)]                                  # drop 271 unused padding rows
U = E / np.linalg.norm(E, axis=1, keepdims=True)  # every row scaled to length 1
tid = lambda w: tok.encode(w)[0]                  # each word here is one token

def nearest(v, k=4, skip=()):
    sims = U @ (v / np.linalg.norm(v))            # cosine with all 151,665 tokens
    best = [i for i in np.argsort(-sims) if i not in skip][:k]
    return ", ".join(f"{tok.decode([i])!r} {sims[i]:.2f}" for i in best)

q = U[tid(" king")] - U[tid(" man")] + U[tid(" woman")]
print("next to ' Monday':  ", nearest(U[tid(" Monday")], skip={tid(" Monday")}))
print("king - man + woman: ", nearest(q))
print("  inputs excluded:  ", nearest(q, skip={tid(" king"), tid(" man"), tid(" woman")}))
```

```text
next to ' Monday':   'Monday' 0.86, ' Tuesday' 0.81, ' Wednesday' 0.77, ' Thursday' 0.76
king - man + woman:  ' king' 0.68, ' queen' 0.57, ' King' 0.54, ' KING' 0.52
  inputs excluded:   ' queen' 0.57, ' King' 0.54, ' KING' 0.52, 'King' 0.52
```

The `U @ v` line is one matrix multiply: 151,665 dot products of length 1,536, one per token, exactly the kind of work Part 1 counted.

### Neighbors

The nearest neighbors are what you'd hope. After " Monday" come "Monday" without the leading space, then " Tuesday", " Wednesday" and " Thursday". With the same code, the closest tokens to " seven" include " nine", " six", " eight" and " five", and the second closest to " Paris" is "巴黎", the Chinese word for Paris, at 0.75. Many neighbors are near-copies: " King", "King" and " KING" are separate tokens with separate rows, and they crowd the top of most lists.

### King minus man plus woman

The famous example comes from word2vec. In [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013), the authors wrote that vector("King") - vector("Man") + vector("Woman") "results in a vector that is closest to the vector representation of the word Queen". Their method computes the sum, then searches for the closest word "measured by cosine distance", and adds in parentheses: "we discard the input question words during this search".

Qwen's table shows why that parenthesis matters. With every token allowed, the closest one to king minus man plus woman is " king" itself, at 0.68, and " queen" is second, at 0.57. Other analogies behave the same way: " Paris" minus " France" plus " Germany" puts " Paris" first, " Germany" second and " Berlin" third. For " father" minus " man" plus " woman", even excluding the inputs isn't enough: "父亲" (Chinese for "father") and "father" without a space come before " mother", in third place.

Projection explains why. The step you add, " woman" minus " man", has length 1.05; the trip you'd need, from " king" to " queen", has length 0.95; and the cosine between the two offsets is only 0.30. So the step's shadow on the king-to-queen direction is $$1.05 \times 0.30 \approx 0.31$$, a third of the way to queen. The rest of the step, length about 1.0, goes sideways. You end up still closer to king.

Still, 0.30 is real signal. Offsets between random pairs of tokens had cosines with the woman-minus-man offset centered on 0 with a spread of 0.046, so 0.30 is more than six spreads out. Other gender pairs land nearby: 0.37 for actor to actress, 0.24 for uncle to aunt, 0.20 for father to mother. And " Berlin" minus " Germany" lines up with " Paris" minus " France" at 0.51. The directions exist; they're a modest part of each vector.

Others have measured how much the exclusion rule matters. Linzen's [Issues in evaluating semantic spaces using word analogies](https://aclanthology.org/W16-2503/) (2016) found that when the inputs weren't excluded, the nearest word to the analogy vector was the word the offset had been added to in 93% of cases. Nissim, van Noord and van der Goot's [Fair Is Better than Sensational](https://aclanthology.org/2020.cl-2.7/) (2020) found that allowing the input words dropped word2vec's accuracy on the Mikolov test set from 74% to 21% (0.71 to 0.21 when each category counts equally), mostly because "man is to king as woman is to king". That matters beyond benchmarks: analogy searches were used to show social bias in embeddings, and a rule that forbids the input words can force a biased-looking answer.

## In the real world

- Qwen2.5-1.5B's input table is also its output table: its config sets `"tie_word_embeddings": true`. So the rows measured in Steps 3 and 5 also score every next-token logit, and their lengths, 0.42 to 1.56, act as a built-in boost or handicap for each token.
- Embedding APIs normalize for you. OpenAI's [embeddings guide](https://developers.openai.com/api/docs/guides/embeddings) says its "embeddings are normalized to length 1", so "Cosine similarity can be computed slightly faster using just a dot product" and "Cosine similarity and Euclidean distance will result in the identical rankings", the $$2 - 2\cos\theta$$ identity from Step 3. Its default vectors have 1,536 numbers for text-embedding-3-small and 3,072 for text-embedding-3-large.
- Vector databases do the same trick. The FAISS library's [metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) notes that its inner-product search "is not by itself cosine similarity, unless the vectors are normalized", and recommends normalizing both sides first. A semantic cache, as in [Designing a Semantic Cache with Hit-Rate Tracking](/blog/semantic-cache-hit-rate-tracking/), applies a cosine threshold such as 0.95 to the same kind of vectors.
- Llama 3.1 8B's attention heads have $$d_k = 128$$, so Step 4's variance argument has its scores divided by $$\sqrt{128} \approx 11.3$$.
- Normalization layers control length inside the model. RMSNorm ([Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467)), used by Llama and Qwen, divides each vector by the root mean square of its entries, which sets its length to $$\sqrt{d}$$ before a learned per-dimension scale. Part 10 covers it.
- Real embedding tables aren't random. Two random tokens in Qwen's table have an average cosine of 0.109, not 0: the rows all lean in a shared direction. Subtracting the table's average row brings the average to 0.000, but the spread, 0.038, is still 1.5 times the $$1/\sqrt{1536} = 0.026$$ of random vectors. Step 4's numbers are a baseline, not a description of trained models.

## Common misconceptions

- **"A bigger dot product always means more similar."** The dot product mixes length and direction, so a long vector can score high with almost everything. Cosine similarity, or normalizing first, compares direction alone.
- **"A cosine of 0.3 is weak."** It depends on the dimension. In 2 dimensions, 0.3 is unremarkable. In Qwen's 1,536, random offsets had a spread of 0.046, so the 0.30 between the woman-minus-man and king-to-queen offsets is more than six spreads above chance.
- **"A d-dimensional model can hold only d independent concepts."** It has only d exactly perpendicular directions, but exponentially many nearly perpendicular ones. Superposition uses that room, at the cost of small overlaps that matter when many concepts are active at once.
- **"King minus man plus woman equals queen."** In Qwen2.5-1.5B, the nearest token to the result is " king", with " queen" second. The textbook answer appears only because the search discards the input words, a rule whose removal cut word2vec's analogy accuracy from 74% to 21%.

## Key takeaways

- A vector's length comes from Pythagoras: square the entries, add, take the square root. It works the same in 2 dimensions or 4,096, and it's the square root of the vector's dot product with itself.
- The dot product equals the two lengths times the cosine of the angle between them. The law of cosines proves it, and you can check it by hand: 24 divided by 5 times 5 is 0.96 for [3, 4] and [4, 3].
- Cosine similarity divides the lengths out and compares direction only. Projection keeps one length and measures a shadow, which is what "how much one vector points along another" means, and how logits and probes read a direction.
- Two random vectors in d dimensions have a cosine near 0 with a spread of about 1 over the square root of d, because positive and negative terms cancel. That leaves room for exponentially many nearly perpendicular directions, which superposition relies on.
- Embedding arithmetic is real but modest. In Qwen2.5-1.5B, king minus man plus woman lands nearest to king, and queen wins only when the input words are excluded.

## What's next

So far each vector has been on its own, or paired with one other. [Part 3](/blog/matrices-linear-maps/) looks at what a whole matrix does to a vector: it stretches, rotates and squashes space as a linear map. You'll see why stacking linear layers collapses into a single one, why that makes nonlinear activations essential, and how to count the cost of any matrix multiply.

## Further reading

- [Efficient Estimation of Word Representations in Vector Space (Mikolov, Chen, Corrado & Dean, 2013)](https://arxiv.org/abs/1301.3781): word2vec, the king-to-queen example, and the analogy search that discards the input words.
- [Issues in evaluating semantic spaces using word analogies (Linzen, 2016)](https://aclanthology.org/W16-2503/): shows that without excluding the inputs, the analogy vector's nearest word is usually one of them, and proposes baselines.
- [Fair Is Better than Sensational: Man Is to Doctor as Woman Is to Doctor (Nissim, van Noord & van der Goot, 2020)](https://aclanthology.org/2020.cl-2.7/): how the exclusion rule shaped analogy accuracy and bias findings.
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): footnote 4 gives the variance argument for scaling dot products by the square root of the key length.
- [Toy Models of Superposition (Elhage et al., 2022)](https://transformer-circuits.pub/2022/toy_model/index.html): why nearly orthogonal directions let a model store more features than dimensions.
- [Qwen2.5-1.5B-Instruct config.json (Hugging Face)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json): the model whose embedding table this article measured.
- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467): RMSNorm, the length-controlling normalization in Llama and Qwen.
- [MetricType and distances (FAISS wiki)](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances): inner product, cosine and normalization in a production vector search library.
- [Vector embeddings (OpenAI API docs)](https://developers.openai.com/api/docs/guides/embeddings): normalized embeddings, and why cosine and Euclidean distance rank the same.
- [Mathematics for Machine Learning (Deisenroth, Faisal & Ong, 2020)](https://mml-book.github.io/): chapter 3 covers norms, inner products, angles and projections in more depth.
- [Essence of linear algebra (3Blue1Brown)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab): the dot product and duality episode shows the projection picture in animation.
