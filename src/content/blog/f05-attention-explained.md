---
title: "Attention Explained: Queries, Keys and Values"
description: "How attention lets each token pull in context: queries, keys and values worked by hand, why we divide by √d, the causal mask, and the length-squared cost."
slug: "attention-explained"
category: "llm-fundamentals"
tags: ["attention", "transformers", "self-attention", "long-context"]
series: "llm-fundamentals"
seriesOrder: 5
pubDate: 2026-10-23
---

## TL;DR

Attention is how a Transformer lets each token's vector take in information from the other tokens around it. Every token produces three vectors from its own: a query (what it is looking for), a key (what it offers) and a value (what it passes on). Each query is compared with every key, the scores are divided by the square root of the vector size, softmax turns each row of scores into weights that add up to 1, and each token's output is the weighted average of the values. In a model that writes text, a mask hides every later token, so each position can only look backward. By the end you'll have computed attention by hand for a four-token sentence, and you'll see why its cost grows with the square of the context length.

## Why it matters

In "The river bank flooded", the word "bank" starts life as exactly the same vector it has in "I opened a bank account". [Part 1](/blog/how-llms-work/) made that point: the embedding table has one row per token, and it doesn't know which sentence it's in. Somewhere between that first lookup and the prediction at the end, "bank" has to find out it's next to a river.

Attention is the part of the model that does that. It's also the reason long prompts get expensive so fast. At a 128,000-token context, one attention head in one layer compares about 17 billion pairs of tokens. Llama 3.1 8B has 32 heads in each of its 32 layers. At that length, the arithmetic spent on attention comes to about twice the arithmetic spent on all 8 billion of its weights.

The idea is older than the Transformer. It came from a 2014 paper on machine translation, and the 2017 paper that introduced the Transformer took it, simplified it, and built the whole model around it. This article builds single-head attention from scratch, with every number worked out.

## The intuition

Picture a party where every guest wears a badge listing what they can talk about: "rivers and boats", "banking", "grammar". Each guest also has a question on their mind, and a story they'll tell anyone who listens.

Someone walks in wondering "which kind of bank am I?" They look around the room, compare their question with every badge, and listen hardest to the guests whose badges match. What they hear is a blend of everyone's story, with the closest matches loudest. The question is the query, the badge is the key and the story is the value.

The analogy breaks in three places. Every guest does this at the same moment, so everyone is asking and answering at once. Nobody chooses their badge, question or story: all three come from the guest's own vector through matrices the model learned in training. And in the models that write text, a guest can only hear people who arrived before them.

## Step 1: the problem attention solves

After the embedding lookup, each token's vector describes that token alone. To predict what comes after "The river bank", the model needs the vector at "bank" to carry something about rivers. Each token has to absorb the context it needs from the others.

Before 2017, the best translation systems used recurrent neural networks (RNNs), which read a sentence one token at a time and carry a running summary forward. An encoder RNN read the whole source sentence into a single fixed-length vector, and a decoder RNN wrote the translation from that vector alone.

In 2014, [Bahdanau, Cho and Bengio](https://arxiv.org/abs/1409.0473) argued that "the use of a fixed-length vector is a bottleneck." A network would have to "compress all the necessary information of a source sentence into a fixed-length vector," and earlier work had shown that translation quality "deteriorates rapidly" as sentences get longer. Their fix: each time the decoder writes a word, it searches the source sentence for the positions that matter most and takes a weighted mix of the encoder's vectors at those positions. The paper calls this "a mechanism of attention."

The Transformer, from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), kept the weighted mix and dropped the recurrence. Every token looks at every other token directly, in one step, and all positions are computed at the same time. When the tokens attend to other tokens in the same sequence, it's called self-attention, and it's what this article builds.

## Step 2: queries, keys and values

The paper describes an attention function as "mapping a query and a set of key-value pairs to an output," where the output is "a weighted sum of the values." The weight for each value comes from how well the query matches that value's key.

In self-attention, all three come from the same token vectors. The model holds three learned weight matrices, `W_Q`, `W_K` and `W_V`, and multiplies each token's vector by each one. That's three of the matrix multiplications from [Part 3](/blog/neural-networks-basics/), and the three results are the token's query, key and value. Training sets the numbers in these matrices; nobody designs them.

### A four-token example

Take the sentence "The river bank flooded". Real token vectors have thousands of numbers. Ours have 3, and to make the arithmetic readable we'll pretend each one means something: how watery the token is, whether it's a place noun, and whether it's a grammar word. (Real dimensions don't come with labels like that. [Part 4](/blog/embeddings-and-logits/) looks at what embedding vectors really encode.)

| Token | Water | Place noun | Grammar |
|---|---|---|---|
| The | 0 | 0 | 1 |
| river | 1 | 0 | 0 |
| bank | 0 | 1 | 0 |
| flooded | 1 | 0 | 1 |

Now three small hand-picked weight matrices, each turning 3 numbers into 2. The query and key size is called `d_k`, so here `d_k = 2`. A trained model would learn these numbers; these were chosen to make the story clear.

| Input feature | `W_Q` | `W_K` | `W_V` |
|---|---|---|---|
| Water | [1, 0] | [2, 0] | [1, 0] |
| Place noun | [2, 0] | [0, 1] | [0, 1] |
| Grammar | [0, 1] | [0, 1] | [0, 0] |

Each token's query is the sum of the `W_Q` rows for its features, weighted by the feature values. "bank" has only the place-noun feature, so its query is the place-noun row of `W_Q`: [2, 0]. "flooded" has water and grammar, so its query is [1, 0] + [0, 1] = [1, 1]. Doing that for all three matrices:

| Token | Query | Key | Value |
|---|---|---|---|
| The | [0, 1] | [0, 1] | [0, 0] |
| river | [1, 0] | [2, 0] | [1, 0] |
| bank | [2, 0] | [0, 1] | [0, 1] |
| flooded | [1, 1] | [2, 1] | [1, 0] |

Read the first number of each key as "I'm about water" and the first number of each query as "I'm looking for water". The place noun "bank" asks that question strongly, with a 2. "river" and "flooded" advertise it, also with a 2. The values say what each token will hand over if someone listens: "river" and "flooded" pass on a water signal [1, 0], "bank" passes on a place signal [0, 1], and "The" passes on nothing.

Each token needs three separate vectors because what it looks for and what it offers differ. "bank" needs to find water words, but it doesn't advertise itself as one. With one vector doing both jobs, every token would match best with tokens just like itself.

## Step 3: scores, softmax and a weighted sum

![A left-to-right flow. Token vectors, one row per token, are multiplied by three learned matrices to give queries, keys and values. Queries and keys combine into a score for every query and key pair. The scores are divided by the square root of d_k, future positions are set to minus infinity by the causal mask, and softmax turns each row into weights that sum to 1. The weights multiply the values to give each token's output.](/diagrams/f05-attention-flow.svg "Single-head scaled dot-product attention, start to finish.")

Every token's output comes from four moves: score, scale, softmax and mix.

### Score every query against every key

The score is the dot product: multiply the two vectors number by number and add up the results. A large dot product means the query and key point the same way. For "bank" against "river", that's `2 × 2 + 0 × 0 = 4`. Doing it for every pair gives a 4 by 4 grid of scores. Each row is one token asking; each column is one token being asked.

| Query from ↓ / key from → | The | river | bank | flooded |
|---|---|---|---|---|
| The | 1 | 0 | 1 | 1 |
| river | 0 | 2 | 0 | 2 |
| bank | 0 | 4 | 0 | 4 |
| flooded | 1 | 2 | 1 | 3 |

Stacking all the queries into a matrix `Q` and all the keys into `K` gets the whole grid in one matrix multiplication, `Q × Kᵀ` (`Kᵀ` is `K` turned on its side, so rows meet columns).

### Scale by the square root of d_k

Next, every score is divided by `√d_k`. Here that's `√2 ≈ 1.414`, so the "bank" row [0, 4, 0, 4] becomes [0, 2.83, 0, 2.83]. In this tiny example the division changes little. In a real model it matters a lot, and the next section shows why.

### Softmax each row

Softmax, from [Part 1](/blog/how-llms-work/), turns a row of scores into weights that are positive and add up to 1: raise `e` (about 2.718) to the power of each score and divide by the row's total. For "bank", `e^0 = 1` and `e^2.83 = 16.92`, the total is 35.84, and the weight on "river" is `16.92 / 35.84 = 0.472`.

| Weights for ↓ | The | river | bank | flooded |
|---|---|---|---|---|
| The | 0.286 | 0.141 | 0.286 | 0.286 |
| river | 0.098 | 0.402 | 0.098 | 0.402 |
| bank | 0.028 | 0.472 | 0.028 | 0.472 |
| flooded | 0.123 | 0.249 | 0.123 | 0.505 |

Each row sums to 1, up to rounding. "bank" puts 94% of its attention on the two water words and almost none on "The" or itself.

### Take the weighted sum of the values

Each token's output is its row of weights times the values. For "bank":

```text
0.028 × [0, 0]  (The)
+ 0.472 × [1, 0]  (river)
+ 0.028 × [0, 1]  (bank)
+ 0.472 × [1, 0]  (flooded)
= [0.944, 0.028]
```

The output for "bank" is almost pure water signal. In a full model this output is added back onto the token's own vector (Part 6 covers that wiring), so after this layer the vector at "bank" carries context from "river" and "flooded": a strong hint that it's a riverbank.

### The formula

Everything above fits in one line. The paper's equation (1) is:

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

Read it from the inside out. `Q Kᵀ` is the grid of every query dotted with every key. Dividing by `√d_k` keeps the scores in a sensible range. `softmax` runs along each row, turning that token's scores into weights. Multiplying by `V` gives each token a weighted average of the values. The paper calls this "Scaled Dot-Product Attention", and the whole calculation fits in a few lines of Python:

```python
import numpy as np

X  = np.array([[0,0,1], [1,0,0], [0,1,0], [1,0,1]])  # The, river, bank, flooded
WQ = np.array([[1,0], [2,0], [0,1]])
WK = np.array([[2,0], [0,1], [0,1]])
WV = np.array([[1,0], [0,1], [0,0]])

Q, K, V = X @ WQ, X @ WK, X @ WV
scores  = Q @ K.T / np.sqrt(K.shape[1])               # divide by sqrt(d_k)
weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
output  = weights @ V
print(output.round(3))   # bank's row: [0.944 0.028]
```

## Step 4: why divide by √d_k

The paper's reason is in one sentence: "We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."

Its footnote gives the arithmetic. If the numbers in a query and a key are independent random values with mean 0 and variance 1, their dot product is a sum of `d_k` such products, and it has mean 0 and variance `d_k`. Its typical size, the standard deviation, is `√d_k`. Dividing by `√d_k` brings it back to about 1, whatever the vector size.

A quick simulation, 200,000 random query and key pairs for each size:

| Vector size `d_k` | Typical size of q·k | `√d_k` | After dividing by `√d_k` |
|---|---|---|---|
| 2 | 1.42 | 1.41 | 1.00 |
| 64 (the 2017 Transformer) | 7.98 | 8.00 | 1.00 |
| 128 (Llama 3.1 8B) | 11.33 | 11.31 | 1.00 |

The Transformer used `d_k = 64`, so it divided every score by 8. Llama 3.1 8B splits its 4,096-number vectors into [32 heads](https://arxiv.org/abs/2407.21783) of 128, so it divides by about 11.3.

Why are big scores a problem? Softmax exaggerates gaps. Take one pattern of four scores and stretch it by 8 and by 11.3, which is what skipping the division does at those vector sizes:

| Scores | Weight on top key | Weight on second | Largest slope |
|---|---|---|---|
| 1, 0.5, 0, -0.5 (scaled) | 45.5% | 27.6% | 0.248 |
| 8, 4, 0, -4 (unscaled, `d_k = 64`) | 98.2% | 1.8% | 0.018 |
| 11.3, 5.7, 0, -5.7 (unscaled, `d_k = 128`) | 99.65% | 0.35% | 0.0035 |

Stretched scores make softmax act like a hard pick: one key takes nearly everything. The last column is the biggest slope of the weights with respect to the scores, the number [Part 3](/blog/neural-networks-basics/) showed training relies on to know which way to move. It shrinks about 70 times between the first and last rows. With slopes that small, the matrices producing queries and keys barely learn.

Random vectors show the same thing. With 128-number random queries and keys and 8 keys per query, over 20,000 trials, the unscaled softmax gave more than 99% of its weight to a single key in 51% of trials. With the division, that never happened, and the top key got 36% of the weight on average.

## Step 5: the causal mask

Our weights table has a problem for a model that writes text. "bank" put 47% of its attention on "flooded", a word that comes after it. When a model generates text, the next word doesn't exist yet. And in training, all positions are processed at once on text that's already written, so a token that could look ahead would just copy the answer it's supposed to predict.

Decoder models, the kind that generate text (GPT and Llama among them), fix this with a causal mask, and attention with that mask is called causal attention. The paper describes it: to stop information flowing from later positions, they mask out the scores for those positions by "setting to −∞" every value in the input of the softmax that corresponds to an illegal connection. Since `e` to the power of minus infinity is 0, those positions get exactly zero weight.

Our scaled scores, masked:

| Query from ↓ / key from → | The | river | bank | flooded |
|---|---|---|---|---|
| The | 0.71 | -∞ | -∞ | -∞ |
| river | 0 | 1.41 | -∞ | -∞ |
| bank | 0 | 2.83 | 0 | -∞ |
| flooded | 0.71 | 1.41 | 0.71 | 2.12 |

And the weights after softmax:

| Weights for ↓ | The | river | bank | flooded |
|---|---|---|---|---|
| The | 1.000 | 0 | 0 | 0 |
| river | 0.196 | 0.804 | 0 | 0 |
| bank | 0.053 | 0.894 | 0.053 | 0 |
| flooded | 0.123 | 0.249 | 0.123 | 0.505 |

The upper right triangle is all zeros. "The" can only look at itself. "bank" can no longer lean on "flooded", so "river" now gets 89.4% of its attention, and its output becomes `0.894 × [1, 0] + 0.053 × [0, 1] = [0.894, 0.053]`: still mostly water, now learned from the past alone. The last row is unchanged, because the last token could already see everything.

Encoder models such as BERT, which read a whole text at once to understand it rather than to continue it, skip the mask. Part 9 compares the two families.

### The mask is what makes the KV cache work

Look at the masked table row by row. Adding a fifth token adds a fifth row and a fifth column, but the column is masked out of every earlier row, so nothing the first four tokens computed changes. Their keys and values stay exactly as they were.

So a model writing text can compute each token's keys and values once and keep them. Each new token computes only its own query, key and value, scores its query against all the stored keys, and mixes the stored values. That stored set is the KV cache, and [The KV Cache](/blog/kv-cache/) in the LLM Inference & Optimization series sizes it for real models: 128 KiB per token for Llama 3.1 8B.

## In the real world

### Cost grows with the square of the length

Every query meets every key, so a sequence of `n` tokens makes an `n × n` grid of scores in every head of every layer. The paper's Table 1 lists self-attention's cost per layer as `O(n² · d)`: it grows with the square of the sequence length `n` times the vector size `d`. Double the context and the score grid gets 4 times bigger.

For one head in one layer, with each score stored in 16 bits (2 bytes):

| Context length | Scores per head, per layer | Memory to store that grid |
|---|---|---|
| 1,024 tokens | 1.05 million | 2.1 MB |
| 4,096 | 16.8 million | 34 MB |
| 32,768 | 1.07 billion | 2.1 GB |
| 131,072 (128K) | 17.2 billion | 34.4 GB |
| 1,048,576 (1M) | 1.1 trillion | 2.2 TB |

Going from 4K to 128K tokens is 32 times more text and 1,024 times more scores. Llama 3.1 8B, with a [128K-token context window](https://arxiv.org/abs/2407.21783), 32 layers and 32 query heads, would face 1,024 such grids, about 17.6 trillion scores, for one 128K-token prompt.

The arithmetic follows the same curve. Computing `Q Kᵀ` and multiplying by `V` costs about `4 × n² × 128` floating-point operations (FLOPs) per head per layer. For Llama 3.1 8B, with the mask skipping roughly half the pairs:

| Prompt length | Attention FLOPs | FLOPs for the weights (about 2 per parameter per token) | Attention as a share of weights |
|---|---|---|---|
| 4,096 tokens | 4.4 × 10¹² | 6.6 × 10¹³ | 7% |
| 32,768 | 2.8 × 10¹⁴ | 5.2 × 10¹⁴ | 54% |
| 131,072 | 4.5 × 10¹⁵ | 2.1 × 10¹⁵ | 215% |

At short lengths attention is a rounding error next to the weights. At 128K it's about twice their cost, and at 1M tokens about 17 times. That's why long-context models invest in cheaper forms of attention, a topic for the [LLM Inference & Optimization series](/series/llm-inference/).

No real system stores those 34 GB grids. [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) starts from the fact that attention's "time and memory complexity" are "quadratic in sequence length", then computes exact attention in tiles so that it never writes the full `N × N` matrix to the GPU's main memory. That removes the memory problem but not the `n²` arithmetic.

During generation, with the KV cache, each new token is one query row against `n` cached keys, so the per-token cost grows linearly. For Llama 3.1 8B that's about 2.1 billion FLOPs of attention per token at 4K tokens of context and 69 billion at 128K, against about 16 billion for the weights.

### What attention heads look at in trained models

Researchers can read the weights table of a trained model directly. [Clark et al. (2019)](https://arxiv.org/abs/1906.04341) did this for BERT's 144 heads:

- Some heads have simple positional jobs. Four heads put more than 50% of their attention on the previous token, and five on the next token.
- Some heads track grammar. One head finds a verb's direct object with 86.8% accuracy, against 40.0% for simply guessing the word two positions to the left, the best fixed-offset baseline.
- A lot of attention goes nowhere useful. In layers 6 to 10, over half of BERT's attention lands on the `[SEP]` separator token, which the authors argue the model uses "as a sort of no-op".

Decoder models show the same habit. [Xiao et al. (2023)](https://arxiv.org/abs/2309.17453) found that models such as Llama 2 give "strong attention scores towards initial tokens as a 'sink' even if they are not semantically important." Their explanation is softmax itself: weights must add up to 1, so a head that has nothing to look at still has to put its attention somewhere. Keeping just 4 initial tokens in the cache, alongside a window of recent ones, let these models handle streams of 4 million tokens.

Some heads implement small algorithms. [Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) describe "induction heads" that complete patterns like `[A][B] ... [A] → [B]`: for example, if "Ms. Okafor" appeared earlier in a prompt, then at the next "Ms." such a head looks back at the token that followed "Ms." last time, which helps predict "Okafor". The paper found these heads appear at "precisely the same point as a sudden sharp increase in in-context learning ability" during training, and argues they may drive much of a model's ability to learn from its prompt. Part 16 returns to this kind of interpretability work.

## Common misconceptions

- **"Queries, keys and values are three different inputs."** In self-attention they're three views of the same token vectors, made by three learned matrices. The model learns what to ask, what to advertise and what to hand over.
- **"High attention weight means the token is important."** Not always. BERT parks over half its attention in some layers on a separator token, and decoders dump attention on the first token as a sink. A weight shows where information was read from, not why the model made its prediction.
- **"Attention knows word order."** It doesn't, on its own. Shuffle our four tokens and each token gets exactly the same weights and outputs, just in a different order. Position has to be added separately, which is Part 7.
- **"The mask only matters during generation."** It matters most in training, where the whole text is processed at once and every position could otherwise peek at the token it's supposed to predict. During generation, the future tokens don't exist yet anyway.
- **"Doubling the context doubles the cost."** The weights' share doubles. Attention's score grid quadruples, which is why 128K-token prompts cost so much more than 4K ones.

## Key takeaways

- Attention lets each token's vector take in context from other tokens. It's how "bank" learns that it's next to a river.
- Each token makes a query, a key and a value by multiplying its vector by three learned matrices. Scores are query-key dot products.
- The formula is `softmax(Q Kᵀ / √d_k) V`: score every pair, divide by the square root of the vector size, softmax each row, then take a weighted average of the values.
- Dividing by `√d_k` stops big vectors from producing huge scores that turn softmax into a hard pick and starve training of slope.
- The causal mask sets future scores to minus infinity so each token sees only the past. That's what makes the KV cache possible.
- Every token meets every other, so cost grows with the square of the length: a 128K-token prompt makes 17 billion scores per head per layer.

## What's next

So far there's one head, asking one kind of question. Real models run dozens of heads side by side, each with its own queries, keys and values, so one can track grammar while another looks for the previous mention of a name. [Part 6](/blog/transformer-block/) puts those heads together with the MLP, the residual stream and layer normalization to build a full Transformer block.

## Further reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): section 3.2 defines scaled dot-product attention, the √d_k footnote and the decoder mask; Table 1 gives the `O(n² · d)` cost.
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau, Cho & Bengio, 2014)](https://arxiv.org/abs/1409.0473): where attention began, as a fix for the fixed-length bottleneck in RNN translation.
- [What Does BERT Look At? An Analysis of BERT's Attention (Clark et al., 2019)](https://arxiv.org/abs/1906.04341): what real attention heads do, from previous-token heads to grammar-tracking heads.
- [In-context Learning and Induction Heads (Olsson et al., 2022)](https://arxiv.org/abs/2209.11895): attention heads that copy patterns, and their link to learning from the prompt.
- [Efficient Streaming Language Models with Attention Sinks (Xiao et al., 2023)](https://arxiv.org/abs/2309.17453): why decoders pile attention onto the first tokens.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135): computing exact attention without storing the `n × n` grid.
- [Attention in transformers, step-by-step (3Blue1Brown, 2024)](https://www.3blue1brown.com/lessons/attention): an animated walk through queries, keys, values and masking.
- [Attention? Attention! (Lilian Weng, 2018)](https://lilianweng.github.io/posts/2018-06-24-attention/): the history from Bahdanau's attention to the Transformer, with the math for each variant.
- [The Illustrated Transformer (Jay Alammar, 2018)](https://jalammar.github.io/illustrated-transformer/): the classic visual guide to self-attention, step by step.
