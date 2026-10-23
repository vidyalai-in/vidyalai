---
title: "Multi-Head Attention and the Transformer Block"
description: "How a Transformer block works: many attention heads, the MLP, residual connections and layer norm, with shapes and a full GPT-2 parameter count."
slug: "transformer-block"
category: "llm-fundamentals"
tags: ["transformers", "attention", "self-attention", "architecture", "training"]
series: "llm-fundamentals"
seriesOrder: 6
pubDate: 2026-10-23
---

## TL;DR

A Transformer block is the unit that every large language model repeats dozens of times. It has two halves. Multi-head attention runs several small attention heads side by side, each looking for a different kind of context, and merges what they find. The MLP then processes each token's vector on its own and holds much of the model's stored knowledge. Each half adds its result to a running vector per token, the residual stream, instead of replacing it, and layer normalization keeps the numbers going into each half on a steady scale. By the end you'll be able to follow every matrix in a block, with its shape, and count GPT-2's 124 million parameters yourself.

## Why it matters

Take the smallest GPT-2, delete any one of its middle blocks, and it barely notices. On a 469-token passage from *Alice's Adventures in Wonderland*, its perplexity, roughly the number of tokens it's torn between at each step, goes from 22.2 to somewhere between 23.4 and 28.3. Delete the first block instead and perplexity jumps to 1,684.

A 12-layer network can lose a layer and keep working because of how each block is wired: it adds a small correction to a shared vector, so removing a block removes one correction. That wiring, plus a normalization step in the right spot, is also what lets models stack 96 or 126 blocks and still train.

[Part 5](/blog/attention-explained/) built one attention head by hand. This article builds everything around it. Along the way you'll find that attention holds only a third of each block's parameters; the other two-thirds sit in a plain two-layer network.

## The intuition

Picture a shared document for each word of a sentence, edited by a team in rounds. In each round, a group of researchers reads the documents. Each researcher has one narrow question ("which word is my subject?", "what came just before me?"), checks the other documents for the answer, and writes a short note. Then a librarian reads each document on its own, recognizes what it's about, and adds relevant facts from memory. Nobody erases anything, and the next round starts from the updated document.

The researchers are attention heads. The librarian is the MLP. The document is the residual stream, and the rounds are the stacked blocks. Layer normalization is a copy editor who resets the volume before anyone reads, so a document that has grown long and loud reads the same as a quiet one.

The analogy breaks in two places. Real notes are numbers added onto a vector, and many of them overlap in the same space. And nobody assigns the questions. Each head's question, and each fact the librarian knows, comes from weights learned in training.

## Step 1: many heads instead of one

Part 5's single head gives each token one weighted average of the values, and an average can only point one way at a time. If "bank" needs to know about a nearby water word *and* about which grammar word starts its phrase, one set of weights has to split its attention between them, and the two signals blur together.

The 2017 paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) runs several heads in parallel instead: "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."

### Two heads on Part 5's sentence

Go back to "The river bank flooded" from Part 5, with its three made-up features per token (water, place noun, grammar) and the causal mask that lets each token see only itself and earlier tokens. Head 1 is Part 5's head: place nouns look for water words. Add a second head with its own three small matrices, chosen so that place nouns look for grammar words. Head 2's query for "bank" is [2, 0], its key for "The" is [2, 0], and among the tokens "bank" can see, only "The" has a value to pass on: a grammar signal, [0, 1]. The arithmetic is the same as in Part 5, with the same division by `√2`.

| Head | What "bank" looks for | Weights on The, river, bank | Output for "bank" |
|---|---|---|---|
| Head 1 | water words | 0.053, 0.894, 0.053 | [0.894, 0.053] |
| Head 2 | grammar words | 0.894, 0.053, 0.053 | [0, 0.894] |

Each head puts 89.4% of its attention on a different word, so neither has to compromise.

### Concatenate, then project

The two outputs are glued end to end, which is called concatenation: [0.894, 0.053, 0, 0.894]. Then one more learned matrix, the output projection `W_O`, maps those 4 numbers back to the 3 the model works in. Here's a simple hand-picked `W_O` that routes each head's findings into a feature:

| Concatenated number | To water | To place noun | To grammar |
|---|---|---|---|
| Head 1, number 1 | 1 | 0 | 0 |
| Head 1, number 2 | 0 | 1 | 0 |
| Head 2, number 1 | 0 | 0 | 0 |
| Head 2, number 2 | 0 | 0 | 1 |

The result for "bank" is [0.894, 0.053, 0.894]: a strong water signal from head 1 and a strong grammar signal from head 2, side by side. The paper's equation is:

```text
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
```

Every head `i` has its own query, key and value matrices and runs ordinary attention; the `h` outputs are concatenated and multiplied by `W_O`. You can also cut `W_O` into one slice per head: the result is then the sum of each head's output times its own slice. [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html) prefer that view for understanding models, and describe the heads as "independently additive": each head computes on its own and adds its contribution. Concatenation is the same arithmetic, packed into one bigger, faster matrix multiplication.

### How real models size their heads

In the toy, heads are 2 numbers wide and the model is 3 wide. Real models pick the head size so the pieces fit exactly: `d_k = d_model / h`, where `d_model` is the size of each token's vector and `h` the number of heads. The 2017 Transformer used `d_model = 512` and 8 heads of 64.

| Model | `d_model` | Heads per layer | Size per head `d_k` | Layers | Heads in total |
|---|---|---|---|---|---|
| Transformer base (2017) | 512 | 8 | 64 | 6 + 6 | 144 |
| GPT-2 small (2019) | 768 | 12 | 64 | 12 | 144 |
| GPT-3 175B (2020) | 12,288 | 96 | 128 | 96 | 9,216 |
| Llama 3.1 8B (2024) | 4,096 | 32 | 128 | 32 | 1,024 |

Sources: [Transformer paper](https://arxiv.org/abs/1706.03762) section 3.2.2, GPT-2's [published config](https://huggingface.co/openai-community/gpt2), [GPT-3 paper](https://arxiv.org/abs/2005.14165) Table 2.1, [Llama 3 paper](https://arxiv.org/abs/2407.21783) Table 3. The Transformer's 6 + 6 layers are an encoder and a decoder, and each decoder layer has two attention layers; Part 8 walks through them.

Each head still reads the whole token vector, and only its output is smaller: head `i`'s query matrix `W_Q^i` turns 768 numbers into 64. Twelve of those side by side make one 768 by 768 matrix, the size of a single full-width head's, so more heads don't cost more weights or arithmetic. In the paper's words, "the total computational cost is similar to that of single-head attention with full dimensionality."

More heads aren't automatically better. In the paper's Table 3, with compute held fixed, one head of 512 scored 24.9 BLEU (a translation quality score) against 25.8 for 8 heads of 64. Going to 32 heads of 16 dropped it to 25.4. The paper's summary: "single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads."

### Counting attention's parameters

A layer of attention holds four `d_model × d_model` matrices: all heads' queries, keys and values, plus `W_O`. That's `4 × d²` weights, where `d` is `d_model`. For GPT-2 small, `4 × 768² = 2,359,296`, and the published weights add 3,072 biases (the small constants from [Part 3](/blog/neural-networks-basics/)) for 2,362,368. GPT-2 stores the query, key and value matrices for all 12 heads in a single 768 by 2,304 matrix and splits the result afterwards. For GPT-3, `4 × 12,288²` is 604 million parameters of attention per layer.

## Step 2: the MLP, a memory per token

Attention moves information between tokens. The second half of the block is a feed-forward network, usually called the MLP (multi-layer perceptron): two layers of the neurons from Part 3, applied to every position separately, with the same weights at every position.

The 2017 paper writes it as:

```text
FFN(x) = max(0, x W1 + b1) W2 + b2
```

Multiplying the token's vector `x` by `W1` and adding the bias `b1` makes it wider. `max(0, ...)` is ReLU, the activation function that zeroes out negatives. Then `W2` and `b2` bring it back to the original size. GPT models swap ReLU for the smoother GELU from Part 3.

The middle layer is 4 times wider than the token vector. The 2017 paper went from 512 to 2,048 and back. The GPT-3 paper says "we always have the feedforward layer four times the size of the bottleneck layer." In GPT-2 small, each token's 768 numbers become 3,072, pass through GELU, and come back as 768.

That makes the parameter count `d × 4d + 4d × d = 8 × d²`, twice attention's: for GPT-2 small, `8 × 768² = 4,718,592` weights, plus 3,840 biases.

### What the MLP stores

[Geva et al. (2020)](https://arxiv.org/abs/2012.14913) open with that imbalance: feed-forward layers "constitute two-thirds of a transformer model's parameters, yet their role in the network remains under-explored." Their answer is that the MLP behaves like a key-value memory, the same pattern as attention but with the keys and values baked into the weights.

Each of the 3,072 columns of `W1` acts as a key. When a token's vector matches a key, that middle neuron fires, and its row of `W2`, the value, gets added to the output. Geva et al. looked at which text made each key fire in a 16-layer language model and found patterns people could name: sentences ending in "substitutes", military text ending in "base" or "bases", time ranges such as "from 7:00 pm Friday until". Lower layers tended to catch surface patterns like these; upper layers caught meaning, and their values pushed up the probability of the words likely to come next.

So "France" plus "capital" in the stream can trip a key whose value pushes the vector toward "Paris". [Part 1](/blog/how-llms-work/) gave that as the reason much of a model's knowledge seems to live in its MLPs.

## Step 3: the residual stream

In Part 5, the attention output for "bank" was "added back onto the token's own vector". That addition is a residual connection, and it's what lets blocks be stacked.

The idea comes from image recognition. [He et al. (2015)](https://arxiv.org/abs/1512.03385) found that simply stacking more layers made networks worse, and not from overfitting: on the CIFAR-10 image dataset, a plain 56-layer network had *higher training error* than a 20-layer one. Their fix was to let each group of layers compute a correction `F(x)` and output `F(x) + x`, with the input passed around the layers on a "shortcut connection". If a layer has nothing useful to add, it only has to learn to output zeros, which they argued is easier than learning to copy its input exactly. With shortcuts they trained networks of up to 152 layers.

The Transformer wraps a shortcut around both halves of every block. In the toy, "bank" started as [0, 1, 0], a pure place noun. Multi-head attention produced [0.894, 0.053, 0.894]. The residual connection adds them:

```text
[0, 1, 0] + [0.894, 0.053, 0.894] = [0.894, 1.053, 0.894]
```

"bank" keeps its place-noun 1 and now also carries water and grammar context.

### One stream, many writers

[Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html) give this running vector its name. "The residual stream is simply the sum of the output of all the previous layers and the original embedding." Both the attention and the MLP layers "read" their input from it "(by performing a linear projection), and then 'write' their result to the residual stream by adding a linear projection back in." They describe it as "a communication channel, since it doesn't do any processing itself and all layers communicate through it."

In that view, a 12-block model is 12 attention layers and 12 MLPs, each reading from and adding to one shared vector per token, between the embedding from [Part 4](/blog/embeddings-and-logits/) and the unembedding at the end. That explains the opening experiment. We ran GPT-2 small on the opening of *Alice's Adventures in Wonderland* (469 tokens), then skipped one block at a time by passing its input straight through:

| Block removed | Perplexity |
|---|---|
| None | 22.2 |
| Block 1 | 1,684.2 |
| Any one of blocks 2 to 11 | 23.4 to 28.3 |
| Block 12 | 104.8 |

Losing a middle block's notes costs a few points. The first and last blocks matter far more; the last one writes its notes right before the vector becomes logits.

The stream also grows. We measured the length of the vector at the last token of "The capital of France is" in GPT-2 small. It starts at 4.9 after the embeddings, reaches 56.5 after block 1, 73.4 after block 6, 289.9 after block 11 and 429.6 after block 12: about 88 times longer than where it started, because every block adds to it. Layer normalization deals with that growth.

## Step 4: layer normalization, and where to put it

Layer normalization, from [Ba, Kiros and Hinton (2016)](https://arxiv.org/abs/1607.06450), rescales one token's vector so that its numbers have an average of 0 and a spread of 1. It computes those statistics from the numbers in a single vector, on "a single training case", so it doesn't depend on what else is in the batch.

Take `x = [2, 4, 6, 8]`:

1. The mean is `(2 + 4 + 6 + 8) / 4 = 5`.
2. Subtract it: [-3, -1, 1, 3].
3. The variance, the average squared distance from the mean, is `(9 + 1 + 1 + 9) / 4 = 5`. The standard deviation is `√5 ≈ 2.236`.
4. Divide: [-1.342, -0.447, 0.447, 1.342].

`[20, 40, 60, 80]` gives exactly the same four numbers: layer norm keeps the pattern and drops the overall size and offset. Then it multiplies each position by a learned gain and adds a learned bias, so the model can restore whatever scale it finds useful. In formula form:

```text
LayerNorm(x) = g × (x − mean(x)) / √(variance(x) + ε) + b
```

`g` and `b` are learned, one pair per position, and `ε` is a tiny constant (0.00001 in GPT-2) that prevents division by zero. In GPT-2 small, each layer norm holds 768 gains and 768 biases.

### Post-norm: the 2017 layout

The original Transformer put the layer norm after the addition: "the output of each sub-layer is LayerNorm(x + Sublayer(x))".

```text
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + MLP(x))
```

This layout is now called post-norm. The stream itself gets normalized after every addition, so no path runs untouched from the first block to the last. [Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) showed that in this layout, at the start of training, "the expected gradients of the parameters near the output layer are large", so a large learning rate makes training unstable. The usual remedy is a learning-rate warm-up: start with a tiny learning rate and raise it over thousands of steps. Xiong et al. found it essential: on a German-to-English translation task, a post-norm Transformer trained with the Adam optimizer reached only 8.45 BLEU without warm-up, against around 34 with it.

### Pre-norm: the GPT-2 layout

The GPT-2 paper moved the norm: "Layer normalization (Ba et al., 2016) was moved to the input of each sub-block, similar to a pre-activation residual network (He et al., 2016) and an additional layer normalization was added after the final self-attention block."

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Each half reads a normalized copy, and the stream itself is never normalized inside the stack. The skip path runs untouched from the embeddings to the final layer norm before the logits. Xiong et al. proved that in this pre-norm layout "the gradients are well-behaved at initialization," and showed that pre-norm models could drop the warm-up entirely and still reach comparable results with less training time.

GPT-2 added one more fix aimed at the growing stream: at initialization it scaled the weights of the layers that write to the stream by `1/√N`, where `N` is the number of those layers, to account for "the accumulation on the residual path with model depth". [GPT-3](https://arxiv.org/abs/2005.14165) kept both changes, and Part 10 covers how later models changed the norm itself.

Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py), a small reimplementation of GPT-2, writes the whole pre-norm block in two lines:

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

## Step 5: the whole block, and the stack

![A left-to-right flow through one pre-norm Transformer block. The residual stream comes in and splits. One path goes through layer norm 1 and multi-head attention with 12 heads and W_O, then reaches an add node, where the other path, the skip path, joins it unchanged. From that sum, one path goes through layer norm 2 and the MLP, which widens 768 numbers to 3,072 and back to 768, and reaches a second add node, where a second skip path carries the sum around the MLP. The result is the residual stream out, passed to the next block.](/diagrams/f06-transformer-block.svg "One GPT-2 block: each half reads a normalized copy of the stream and adds its result back.")

Only the two halves see normalized numbers; the skip paths carry the stream past them unchanged.

### Follow the shapes

Take the five tokens of "The capital of France is" (GPT-2's [tokenizer](/blog/tokenization/) gives one per word) through one GPT-2 small block. The stream is a grid with one row per token and 768 numbers per row:

| Step | What happens | Shape |
|---|---|---|
| Stream in | one vector per token | 5 × 768 |
| Layer norm 1 | each row rescaled on its own | 5 × 768 |
| Queries, keys, values | one multiply by a 768 × 2,304 matrix, cut in three | 3 of 5 × 768 |
| Split into heads | 768 = 12 heads × 64 | 12 heads of 5 × 64, for each of Q, K, V |
| Scores, mask, softmax | `Q Kᵀ / √64` per head, as in Part 5 | 12 grids of 5 × 5 |
| Weighted values | weights times values, per head | 12 of 5 × 64 |
| Concatenate | heads glued back together | 5 × 768 |
| Output projection `W_O` | multiply by 768 × 768 | 5 × 768 |
| Add | stream plus attention output | 5 × 768 |
| Layer norm 2 | each row rescaled again | 5 × 768 |
| MLP up, then GELU | multiply by 768 × 3,072 | 5 × 3,072 |
| MLP down | multiply by 3,072 × 768 | 5 × 768 |
| Add, stream out | stream plus MLP output | 5 × 768 |

The block's input and output have the same shape, so the output of block 1 is a valid input to block 2, and so on up the stack. Only the 12 grids of scores mix information between rows; everything else works on each token alone.

### Counting every parameter in GPT-2 small

We loaded GPT-2 small's published weights and counted the parameters in one block:

| Part of the block | Shape | Parameters |
|---|---|---|
| Layer norm 1 (gain, bias) | 768 + 768 | 1,536 |
| Queries, keys, values, all heads | 768 × 2,304, plus 2,304 biases | 1,771,776 |
| Output projection `W_O` | 768 × 768, plus 768 biases | 590,592 |
| Layer norm 2 (gain, bias) | 768 + 768 | 1,536 |
| MLP up | 768 × 3,072, plus 3,072 biases | 2,362,368 |
| MLP down | 3,072 × 768, plus 768 biases | 2,360,064 |
| **One block** | | **7,087,872** |

Attention is 33.3% of the block, the MLP 66.6%, and the two layer norms 0.04%. The shortcut `12 × d² = 12 × 768² = 7,077,888` is within 0.14% of the real count; the gap is the biases and norms.

Now the whole model:

| Part of the model | Parameters | Share |
|---|---|---|
| 12 blocks × 7,087,872 | 85,054,464 | 68.4% |
| Token embeddings, 50,257 × 768 | 38,597,376 | 31.0% |
| Position embeddings, 1,024 × 768 (Part 7) | 786,432 | 0.6% |
| Final layer norm | 1,536 | 0.001% |
| Output table (tied to the token embeddings, Part 4) | 0 extra | 0% |
| **Total** | **124,439,808** | |

That's the 124 million from Part 4. (The GPT-2 paper's table lists this model as 117M; counting the published weights gives 124.4 million.)

The same shortcut scales to GPT-3. With `d = 12,288`, one block holds about `12 × 12,288² = 1.81 billion` parameters: 604 million in attention and 1.21 billion in the MLP. Ninety-six blocks make 173.9 billion, and adding the 0.6-billion embedding table lands within half a percent of the published 175 billion, as Part 1 worked out. In a small model like GPT-2, the vocabulary table is almost a third of everything. In GPT-3, the blocks are 99.6%.

## In the real world

- Llama 3.1 8B keeps the block and changes the parts. The [Llama 3 paper](https://arxiv.org/abs/2407.21783) lists 32 layers, `d_model = 4,096`, 32 attention heads and an MLP that widens to 14,336. But its 32 query heads share 8 sets of keys and values, a trick called grouped-query attention that shrinks the [KV cache](/blog/kv-cache/), and its MLP uses a three-matrix design called SwiGLU. By our arithmetic from those numbers, each block holds 41.9 million parameters of attention and 176.2 million of MLP, so the MLP is 81% of the block, and the 32 blocks hold about 6.98 billion of the 8 billion. Part 10 explains each change.
- Heads specialize. [Part 5](/blog/attention-explained/) showed heads that track the previous token, find a verb's direct object, or copy patterns from earlier in the prompt.
- Many heads can go. [Voita et al. (2019)](https://arxiv.org/abs/1905.09418) pruned an English-Russian translation model and found that removing 38 of its 48 encoder heads cost only 0.15 BLEU. The specialized heads were the last to be pruned. [Michel et al. (2019)](https://arxiv.org/abs/1905.10650) found that "a large percentage of attention heads can be removed at test time without significantly impacting performance," and that some layers could be cut to a single head.
- Layers specialize too. Geva et al. found lower MLP layers matching surface patterns and upper ones matching meaning, and our GPT-2 experiment found the first and last blocks far harder to lose than the middle ten.

## Common misconceptions

- **"Each head sees only a slice of the token's vector."** Every head reads the whole vector. The slicing happens in what it computes: each head's matrices project the full 768 numbers down to its own 64.
- **"More heads means a bigger, slower model."** At a fixed `d_model`, the heads split the same `4 × d²` weights. Eight heads of 64 cost the same as one head of 512, and the 2017 paper found eight worked better.
- **"Attention is most of the model."** In a GPT-style block, the MLP holds twice as many parameters as attention: 66.6% of each GPT-2 block, and 81% in Llama 3.1 8B.
- **"Each block transforms the vector into a new one."** Each block adds to it. The residual stream is the sum of the embedding and every layer's output, which is why one middle block can be removed with only a small loss.
- **"Layer norm normalizes across the batch."** That's batch normalization. Layer norm uses only the numbers in one token's vector, so it computes the same thing in training and in use, with any batch size.

## Key takeaways

- Multi-head attention runs several small heads in parallel, each with its own queries, keys and values, then concatenates their outputs and mixes them with one more matrix, `W_O`. It costs the same as one big head.
- Attention moves information between tokens. The MLP, two layers that widen each vector 4 times and shrink it back, works on each token alone and acts like a learned memory.
- Attention holds about `4 × d²` parameters per block and the MLP about `8 × d²`. Twelve blocks of 7.09 million, plus a 38.6-million embedding table, make GPT-2's 124 million.
- Residual connections make every block add to a shared vector per token, the residual stream, instead of replacing it. That's why a GPT-2 block can be removed with only a small loss.
- Layer norm rescales each token's vector to mean 0 and spread 1. Putting it before each half, as GPT-2 did, keeps the stream's path clean and makes deep models train stably.

## What's next

Shuffle the words of a sentence and attention, as built so far, gives every token exactly the same output, just in a different order. Nothing in a block knows that "dog bites man" differs from "man bites dog". [Part 7](/blog/positional-encoding/) covers how models add word order back in, from the 2017 paper's sine waves to the rotary position embeddings (RoPE) used by Llama.

## Further reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): section 3.2.2 defines multi-head attention, section 3.3 the feed-forward network, and Table 3 tests different numbers of heads.
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): the GPT-2 paper, where layer norm moves to the input of each sub-block.
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385): where residual connections come from, and why deeper plain networks trained worse.
- [Layer Normalization (Ba, Kiros & Hinton, 2016)](https://arxiv.org/abs/1607.06450): normalizing each example on its own instead of across a batch.
- [On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745): why post-norm needs a warm-up and pre-norm doesn't.
- [Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2020)](https://arxiv.org/abs/2012.14913): what the MLP's two matrices store.
- [A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)](https://transformer-circuits.pub/2021/framework/index.html): the residual stream as a communication channel, and heads as independent, additive parts.
- [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165): GPT-3's sizes, head counts and 4x MLP in Table 2.1.
- [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned (Voita et al., 2019)](https://arxiv.org/abs/1905.09418) and [Are Sixteen Heads Really Better than One? (Michel et al., 2019)](https://arxiv.org/abs/1905.10650): how many heads a trained model really needs.
- [nanoGPT model.py (Andrej Karpathy)](https://github.com/karpathy/nanoGPT/blob/master/model.py): a complete GPT-2 block, with its layer norm, attention and MLP classes, in about 90 readable lines.
- [How might LLMs store facts (3Blue1Brown, 2024)](https://www.3blue1brown.com/lessons/mlp): an animated look inside the MLP.
- [The Illustrated Transformer (Jay Alammar, 2018)](https://jalammar.github.io/illustrated-transformer/): multi-head attention, residuals and layer norm, drawn step by step.
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/): the 2017 paper, line by line, with working code.
