---
title: "The Math You Need for LLMs, and the Shapes Inside One"
description: "Scalars, vectors, matrices and tensors, matrix multiplication as many dot products, and the shape of every tensor in one forward pass of Llama 3.1 8B."
slug: "math-for-llms"
category: "llm-fundamentals"
tags: ["math", "transformers", "architecture", "embeddings", "tokens"]
series: "llm-math"
seriesOrder: 1
pubDate: 2026-10-23
---

## TL;DR

A large language model is a long chain of multiplications and additions on grids of numbers. To follow that chain you need four kinds of object (scalars, vectors, matrices and tensors), one operation (the dot product, which matrix multiplication repeats many times), and the habit of tracking shapes, the sizes of every grid. This article builds all three from high-school algebra. Then it follows real data through Llama 3.1 8B, from token IDs to 128,256 scores, writing down the shape and the cost of every step. By the end you'll be able to read a line like `torch.einsum("btd,df->btf", x, W)`, predict the shape it produces, count its arithmetic, and see where the rest of this 16-part series is headed.

## Why it matters

To write a single token, Llama 3.1 8B performs about 15.5 billion arithmetic operations. More than 99.9% of them sit inside one kind of calculation: multiply two lists of numbers position by position and add up the results.

That calculation is the dot product, and a twelve-year-old can do it on paper. The hard part of following a model is bookkeeping: billions of numbers arranged in grids of different sizes, and knowing which grid is which.

So you need surprisingly little math to follow all of it: four ideas, which this 16-part series builds from scratch. This part starts with the first, linear algebra, and ends with a map of the rest. If you haven't seen how an LLM works at all, [How a Large Language Model Works](/blog/how-llms-work/) gives the big picture in words.

## The intuition

Think of a spreadsheet. One cell holds a single number. A column holds a list of numbers. A whole sheet is a grid with rows and columns. A workbook of identically laid-out sheets, one per month, is a stack of grids.

Those four things are, in order, a scalar, a vector, a matrix and a tensor with three axes. The operation an LLM repeats billions of times is a spreadsheet formula you've probably written: `SUMPRODUCT`, which multiplies two columns cell by cell and adds the results. Matrix multiplication is that formula, filled across a whole grid of cells.

The analogy breaks in three places. A spreadsheet's columns have names like "price" and "quantity"; the 4,096 numbers in a model's vector have no names, and a single one rarely means anything you could label. Spreadsheets stop at three axes in practice, while the attention code inside Llama juggles five. And nobody types the numbers in: every one of the model's 8 billion was learned, which is what Parts 7 to 9 of this series explain.

## Step 1: scalars, vectors, matrices and tensors

### Scalars and vectors

A scalar is a single number, like the temperature setting 0.7 that controls how random a chatbot's answers are.

A vector is an ordered list of numbers. Here is one with three entries:

$$
\mathbf{x} = [2,\ 3,\ 0]
$$

Bold lowercase letters like $$\mathbf{x}$$ are the usual way to write vectors. The entries are numbered, so $$x_1 = 2$$, $$x_2 = 3$$ and $$x_3 = 0$$; the small number below the letter, the subscript, says which entry you mean. Order matters: [3, 2, 0] is a different vector.

When Llama 3.1 8B reads a token, it turns it into a vector of 4,096 numbers. Mathematicians write that fact like this:

$$
\mathbf{x} \in \mathbb{R}^{4096}
$$

Read it as "x is a list of 4,096 real numbers". The symbol $$\in$$ means "is a member of", and $$\mathbb{R}$$ stands for the real numbers, the ordinary numbers with decimals, like 0.37 or -12.5. The superscript says how many of them are in the list. That one count is called the vector's dimension, and in a model it's usually written $$d$$ or `d_model`. GPT-2's smallest version uses 768; Llama 3.1 8B uses 4,096. [Fundamentals Part 4](/blog/embeddings-and-logits/) shows where these vectors come from.

### Matrices

A matrix is a grid of numbers with rows and columns. Here is one with 2 rows and 3 columns, a 2 × 3 matrix:

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
$$

Matrices get capital letters. The entry in row $$i$$ and column $$j$$ is written $$A_{ij}$$, so $$A_{23} = 6$$: row 2, column 3. The size is always rows first, then columns.

Almost every learned weight in Llama's blocks lives in a matrix. The first matrix of each block's MLP (multi-layer perceptron, the part of each block that processes every token on its own) turns 4,096 numbers into 14,336, and it's a grid of that size:

$$
W \in \mathbb{R}^{4096 \times 14336}
$$

That's "W is a grid of real numbers with 4,096 rows and 14,336 columns", 58,720,256 numbers in all. A model's parameters, the numbers learned during training, are mostly entries of matrices like this one.

### Tensors and shapes

A tensor is the general case: a block of numbers with any number of axes. A scalar has zero axes, a vector one, a matrix two. A batch of sentences inside a model has three: which sentence, which token in it, and which of the 4,096 numbers.

The list of axis sizes is the tensor's shape. Two sentences of five tokens each, each token a 4,096-number vector, make a tensor of shape (2, 5, 4096), holding `2 × 5 × 4,096 = 40,960` numbers. You multiply the sizes to get the count.

| Object | Axes | Example inside Llama 3.1 8B | Shape |
|---|---|---|---|
| Scalar | 0 | the scaling factor in attention, 1/√128 | () |
| Vector | 1 | one token's vector | (4096) |
| Matrix | 2 | the embedding table, one row per token | (128256, 4096) |
| 3-axis tensor | 3 | a batch of B sentences of T tokens | (B, T, 4096) |
| 4-axis tensor | 4 | attention scores, one T × T grid per head | (B, 32, T, T) |

## Step 2: the dot product and matrix multiplication

### The dot product

The dot product takes two vectors of the same length, multiplies them position by position, and adds up the results. Take [1, 2, 3] and [7, 9, 11]:

```text
[1, 2, 3] · [7, 9, 11] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58
```

In symbols, for two vectors with $$d$$ entries each:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{d} a_i \, b_i
$$

The big $$\sum$$ (the Greek capital sigma) means "add up". The little $$i = 1$$ below it and $$d$$ above it say: let $$i$$ run from 1 to $$d$$, compute $$a_i b_i$$ for each, and add the results. [Fundamentals Part 4](/blog/embeddings-and-logits/) shows that the result measures how much two vectors point the same way; Part 2 of this series shows why.

### Matrix multiplication, by hand

Now multiply the 2 × 3 matrix $$A$$ from above by a 3 × 2 matrix $$B$$:

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad
B = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix}
$$

Each entry of the answer is a dot product: a row of $$A$$ with a column of $$B$$. Row 1 of $$A$$ is [1, 2, 3] and column 1 of $$B$$ is [7, 9, 11], so the top-left entry is the 58 from before. All four:

```text
row 1 · col 1 = 1×7 + 2×9  + 3×11 =  58
row 1 · col 2 = 1×8 + 2×10 + 3×12 =  64
row 2 · col 1 = 4×7 + 5×9  + 6×11 = 139
row 2 · col 2 = 4×8 + 5×10 + 6×12 = 154
```

$$
AB = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}
$$

A matrix multiplication is many dot products, one per entry of the result, arranged in a grid. In general, entry $$(i, j)$$ of the product is row $$i$$ of $$A$$ dotted with column $$j$$ of $$B$$:

$$
(AB)_{ij} = \sum_{p=1}^{k} A_{ip} \, B_{pj}
$$

Here $$k$$ is the length of each row of $$A$$, which must equal the length of each column of $$B$$, and $$p$$ walks along both at once.

### The shape rule

A dot product needs two lists of the same length. So a row of $$A$$ (length = number of columns of $$A$$) must match a column of $$B$$ (length = number of rows of $$B$$). That gives the one rule worth memorizing:

$$
(m \times k) \text{ times } (k \times n) \;\to\; (m \times n)
$$

The inner sizes must match, and they disappear. The outer sizes survive. In our example, (2 × 3) times (3 × 2) gives 2 × 2. Try it the other way round, (3 × 2) times (2 × 3), and the inner sizes are both 2, so it works, but the result is a 3 × 3 matrix: order matters in matrix multiplication.

One more operation shows up everywhere: the transpose, which flips a matrix so rows become columns. It's written with a superscript $$\top$$. The transpose of our 2 × 3 matrix $$A$$ is a 3 × 2 matrix $$A^{\top}$$ whose first column is [1, 2, 3]. Code often transposes a matrix just to make the inner sizes line up.

### Counting the work

How much arithmetic does a matrix multiply take? The result has $$m \times n$$ entries. Each is a dot product of length $$k$$: $$k$$ multiplications, and $$k$$ additions into a running total that starts at zero. A multiplication or addition of decimal numbers is called a FLOP (floating-point operation), so:

$$
\text{FLOPs} \approx 2mkn
$$

For our small example, `2 × 2 × 3 × 2 = 24`. Strictly, it's 12 multiplications and 8 additions, because the first product in each sum has nothing to be added to. At the sizes inside an LLM, that difference vanishes, and everyone uses 2mkn. Kaplan and colleagues put it this way: "the factor of two comes from the multiply-accumulate operation used in matrix multiplication" ([Scaling Laws for Neural Language Models, 2020](https://arxiv.org/abs/2001.08361)).

Now a real one. Llama 3.1 8B's query projection, which we'll meet in Step 3, multiplies each token's 4,096-number vector by a 4,096 × 4,096 matrix:

```text
one token:               (1 × 4,096) times (4,096 × 4,096)
FLOPs = 2 × 1 × 4,096 × 4,096           =      33,554,432
8 prompts of 1,000 tokens: (8,000 × 4,096) times (4,096 × 4,096)
FLOPs = 2 × 8,000 × 4,096 × 4,096       = 268,435,456,000
```

Notice what changed: only $$m$$, the number of rows. Stacking 8,000 tokens into one matrix doesn't change the weights at all; it reuses them 8,000 times. That's why GPUs like big batches, a point the real-world section comes back to. Part 3 of this series looks at matrix multiplication from the other side, as a transformation of space.

## Step 3: the shapes inside one forward pass of Llama 3.1 8B

### The numbers that set every shape

Every shape in the model follows from a handful of numbers in its [config file](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json), which agree with Table 3 of the [Llama 3 paper](https://arxiv.org/abs/2407.21783) (the paper rounds the vocabulary to 128,000; the extra 256 rows are slots for special tokens):

| Config name | Value | What it sets |
|---|---|---|
| `hidden_size` | 4,096 | numbers per token vector, $$d$$ |
| `num_hidden_layers` | 32 | Transformer blocks, stacked |
| `num_attention_heads` | 32 | query heads per block |
| `num_key_value_heads` | 8 | key and value heads per block, shared by 4 query heads each |
| `head_dim` | 128 | numbers per head; 32 × 128 = 4,096 |
| `intermediate_size` | 14,336 | width inside the MLP |
| `vocab_size` | 128,256 | rows in the embedding table, and logits per token |

Two more letters describe the input, not the model: $$B$$, the batch size (how many sequences go through at once), and $$T$$, the number of tokens in each sequence.

### Following the data

Here is the trip from token IDs to scores, one row per step. The FLOPs column is for one token, which keeps the numbers readable. For the attention rows, it assumes the token can see 1,000 tokens of context, because those two steps compare the token against every earlier one. [Fundamentals Part 10](/blog/modern-llm-architecture/) explains what each piece does; here we only track sizes and cost.

| Step | Weight shape | Output shape | FLOPs per token |
|---|---|---|---|
| Token IDs | none | (B, T) | 0 |
| Embedding lookup | (128256, 4096) | (B, T, 4096) | 0 (a copy) |
| Query projection | (4096, 4096) | (B, T, 32, 128) | 33.6 million |
| Key projection | (4096, 1024) | (B, T, 8, 128) | 8.4 million |
| Value projection | (4096, 1024) | (B, T, 8, 128) | 8.4 million |
| Scores: query · key | none | (B, 32, T, T) | 8.2 million |
| Weighted sum of values | none | (B, T, 32, 128) | 8.2 million |
| Output projection | (4096, 4096) | (B, T, 4096) | 33.6 million |
| MLP gate and up | two of (4096, 14336) | (B, T, 14336) | 234.9 million |
| MLP down | (14336, 4096) | (B, T, 4096) | 117.4 million |
| **One block** | | (B, T, 4096) | **452.6 million** |
| **32 blocks** | | (B, T, 4096) | **14.48 billion** |
| Output table (logits) | (4096, 128256) | (B, T, 128256) | 1.05 billion |
| **Whole pass** | | | **15.53 billion** |

Every block takes in (B, T, 4096) and hands out (B, T, 4096), which is why 32 identical blocks can be stacked. The norms and the residual additions (adding each half's output back onto the stream) keep that shape too, and cost so little arithmetic that they don't get rows.

Splitting into heads is a reshape with no arithmetic: the query projection's 4,096 numbers per token get cut into 32 pieces of 128, so the shape goes from (4096) to (32, 128). Keys and values get only 8 heads, 1,024 numbers, because each one serves 4 query heads, a design called grouped-query attention.

The scores and the weighted sum of values are the only places where tokens meet. Each query head compares one token's 128 numbers with the keys of every visible token. That's a (T, 128) grid times a transposed (128, T) grid, giving a T × T grid of scores per head. For one token at position 1,000, that's `32 heads × 1,000 keys × 2 × 128 = 8,192,000` FLOPs, the 8.2 million in the table.

The three MLP matrices take 352 million of each block's 453 million FLOPs, and across the model they account for 75% of the arithmetic spent on weights. The output table adds 7%, and the attention projections the remaining 18%.

![A left-to-right flow of tensor shapes through Llama 3.1 8B. Token IDs of shape B by T go through an embedding lookup into a stream of shape B by T by 4,096. That enters a box labeled each of 32 blocks, where, top to bottom: queries of shape B by T by 32 by 128 and keys and values of shape B by T by 8 by 128; scores per head of shape B by 32 by T by T; values mixed and projected back to B by T by 4,096; MLP gate and up at B by T by 14,336; and MLP down back to B by T by 4,096. After the blocks, logits of shape B by T by 128,256.](/diagrams/m01-shape-flow.svg "The shapes of one forward pass of Llama 3.1 8B. B is the number of sequences, T the tokens in each.")

Inside a block, the scores tensor grows with T squared; every other tensor grows with B × T.

### Checking against the rule of thumb

A common shortcut says a forward pass costs about 2 FLOPs per parameter per token, one multiply and one add per weight. Llama 3.1 8B has 8,030,261,248 parameters, so the shortcut gives 16.06 billion.

The table says 15.53 billion. Two corrections explain the gap. The embedding table holds 525 million parameters but costs no arithmetic, because looking up a row is a copy, which removes 1.05 billion. Attention over 1,000 tokens of context adds 0.52 billion that no parameter accounts for. With those, the count is `16.06 - 1.05 + 0.52 = 15.53` billion, within 4% of the shortcut. The weight arithmetic, `2 × 7,504,658,432 = 15.01` billion, is exactly two FLOPs for every weight in the model's matrices other than the embedding table.

The attention term grows with context. At 8,192 tokens of context it's 4.3 billion FLOPs per token, 22% of the total. At 128,000 tokens it's 67 billion, and the per-token cost becomes 82 billion, five times the shortcut. "2 × parameters" is a rule for short contexts.

Everything outside matrix multiplication (norms, activation functions, softmax, the residual additions) is small. Counting generously, about 10 FLOPs per number they touch, it's around 0.33 million FLOPs per block at this context length, less than 0.1% of the total. That's where "more than 99.9% dot products" comes from.

### A whole batch

For a batch, multiply. Eight prompts of 1,000 tokens each are 8,000 tokens. The weight arithmetic is `8,000 × 15.01 billion ≈ 120 trillion` FLOPs. Attention adds about 2 trillion more, since token number $$t$$ looks back at $$t$$ tokens and the average is about 500. That's roughly 122 trillion FLOPs for the batch.

The shapes get big too. At B = 8 and T = 1,000, each layer's scores tensor (8, 32, 1000, 1000) holds 256 million numbers. The logits (8, 1000, 128256) hold 1.03 billion, about 2.05 GB at 2 bytes per number.

### Naming shapes with einsum

Code needs a compact way to say which axes get multiplied and summed. Einsum notation, short for Einstein summation, gives every axis a letter. Here's the MLP's first multiply in PyTorch:

```python
torch.einsum("btd,df->btf", x, W)
```

`btd` names the axes of `x`: batch, token, model dimension. `df` names the axes of `W`: model dimension in, MLP width out. `btf` names the axes of the result. The rule, in the words of the [PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html), is that the output "is then computed by summing the product of the elements of the operands along the dimensions whose subscripts are not part of the output." Here `d` appears in both inputs but not in the output, so it gets summed: that's a dot product of length 4,096 for every combination of `b`, `t` and `f`. The result has shape (B, T, 14336), and the call does the same thing as `x @ W`.

The plain matrix multiply from Step 2 is `"ij,jk->ik"`, the docs' own first example. Einsum is most useful for the attention scores, which have more axes than `@` handles comfortably.

The NumPy snippet below builds random tensors with Llama 3.1 8B's real sizes for 2 sequences of 5 tokens, runs one block, and prints the shapes. The weights are random stand-ins for trained ones, so the numbers mean nothing, but every shape is the real model's. It needs about 2.5 GB of memory, mostly for the embedding table, and took about 8 seconds on our test machine.

```python
import numpy as np

# Llama 3.1 8B sizes, from its config.json
d, n_heads, n_kv, hd, ffn, vocab = 4096, 32, 8, 128, 14336, 128256
B, T = 2, 5                                    # 2 sequences of 5 tokens each
rng = np.random.default_rng(0)

def rand(*shape):                              # random stand-ins for trained weights
    w = rng.standard_normal(shape, dtype=np.float32)
    w *= 0.02
    return w

ids = rng.integers(0, vocab, size=(B, T))      # token IDs, (B, T)
E = rand(vocab, d)                             # embedding table, (vocab, d)
x = E[ids]                                     # lookup, (B, T, d)

# Attention: project, then split the last axis into heads
q = np.einsum("btd,de->bte", x, rand(d, n_heads * hd))
k = np.einsum("btd,de->bte", x, rand(d, n_kv * hd))
v = np.einsum("btd,de->bte", x, rand(d, n_kv * hd))
g = n_heads // n_kv                            # 4 query heads share each key/value head
q = q.reshape(B, T, n_kv, g, hd)
k = k.reshape(B, T, n_kv, hd)
v = v.reshape(B, T, n_kv, hd)
scores = np.einsum("btkgh,bskh->bkgts", q, k) / hd ** 0.5    # a T x T grid per head
# (the real model applies the causal mask and softmax to the scores here)
att = np.einsum("bkgts,bskh->btkgh", scores, v).reshape(B, T, n_heads * hd)
x = x + att @ rand(n_heads * hd, d)            # output projection, added to the stream

# MLP: up to 14,336, gate, back down to 4,096
gate = x @ rand(d, ffn)
up = x @ rand(d, ffn)
x = x + (gate / (1 + np.exp(-gate)) * up) @ rand(ffn, d)

logits = x @ E.T   # reuses E to save memory; the real model has its own (d, vocab) table

for name, a in [("ids", ids), ("x", x), ("q", q), ("k", k),
                ("scores", scores), ("att", att), ("gate", gate), ("logits", logits)]:
    print(f"{name:7}{a.shape}")
```

```text
ids    (2, 5)
x      (2, 5, 4096)
q      (2, 5, 8, 4, 128)
k      (2, 5, 8, 128)
scores (2, 8, 4, 5, 5)
att    (2, 5, 4096)
gate   (2, 5, 14336)
logits (2, 5, 128256)
```

The scores line is the one to study. The query has axes `b t k g h`: batch, token, key-value group, which of the 4 query heads in that group, and the 128 numbers per head. The key has `b s k h`, with `s` for the token being looked at. The output `bkgts` drops `h`, so the 128 numbers per head get summed into dot products, and keeps both `t` and `s`, which makes a 5 × 5 grid for each of the 8 × 4 = 32 heads. Because `k` appears in both inputs and the output, each group of 4 query heads is matched only against its own shared key head: grouped-query attention in one string.

## Step 4: the four toolkits, and a map of the series

Everything in Step 3 used one toolkit, linear algebra. The other three answer different questions about the same model.

Linear algebra covers what the model computes, including the special matrices behind attention, rotary positions and low-rank compression. Probability and information cover what the outputs mean: how 128,256 logits become probabilities, and how the training loss measures surprise. That's where logarithms come in. Calculus and optimization cover how 8 billion numbers get learned, through gradients, backpropagation and optimizers. Numerics and counting cover the cost: decimals stored in 16, 8 or 4 bits, and parameters, FLOPs and memory counted well enough to predict a training run.

| Part | Title | Toolkit |
|---|---|---|
| 1 | The math you need for LLMs, and the shapes inside one | Linear algebra |
| 2 | Vectors and dot products | Linear algebra |
| 3 | Matrices as transformations | Linear algebra |
| 4 | Probability and the chain rule of language | Probability and information |
| 5 | Softmax and log-sum-exp | Probability and information |
| 6 | Entropy, cross-entropy and perplexity | Probability and information |
| 7 | Derivatives, gradients and the chain rule | Calculus and optimization |
| 8 | Backpropagation by hand | Calculus and optimization |
| 9 | Optimizers: from gradient descent to AdamW | Calculus and optimization |
| 10 | Keeping signals stable: initialization, normalization and residuals | Calculus and optimization |
| 11 | The math of attention | Linear algebra |
| 12 | Rotations, complex numbers and RoPE | Linear algebra |
| 13 | Low rank and the SVD | Linear algebra |
| 14 | Floating point and quantization math | Numerics and counting |
| 15 | Counting: parameters, FLOPs, memory and scaling laws | Numerics and counting |
| 16 | The math of alignment: policy gradients, PPO, DPO and GRPO | Probability and information |

Each part works a small example first, then the general formula, then checks it in Python, and links the Fundamentals or Inference article that covers the same idea in words.

## In the real world

- PyTorch stores weights the other way round. Its [`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) layer keeps "weights of the module of shape (out_features, in_features)" and computes "y = xA^T + b", where `A` is that weight matrix. So Llama's key projection, which turns 4,096 numbers into 1,024, is stored as a (1024, 4096) matrix and transposed on the fly to fit the shape rule. If you load a checkpoint and a shape looks backwards, this is usually why.
- Hugging Face's [Llama code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) uses the names in this article's table: `embed_tokens`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` and `lm_head`. It splits heads with a reshape to `(*input_shape, -1, self.head_dim)` and then puts the head axis before the token axis, (B, heads, T, 128); axis order varies between codebases, sizes don't. Its `repeat_kv` function copies the 8 key-value heads up to 32, going "from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)", where the einsum above shared them without copying.
- Servers don't build all the logits. At B = 8 and T = 1,000, the full logits tensor is 2.05 GB, but generating the next token only needs the last position's 128,256 scores. Hugging Face's Llama model has a `logits_to_keep` argument that slices the stream before the output table, so only the positions you need get scored.
- Batching is a shape change. Folding B sequences of T tokens into one (B·T, 4096) matrix makes one tall matrix multiply that reads the weights from memory once for all the rows. With a single token, each weight does one multiply-add per read. [How LLM Inference Works](/blog/how-llm-inference-works/) shows why that makes generation wait on memory rather than arithmetic, and how batching fixes it.
- Parameter counts come straight from shapes. Multiply out each weight shape, add the norms' 4,096 gains each, and you get 218,112,000 per block and 8,030,261,248 for the whole model. [Fundamentals Part 10](/blog/modern-llm-architecture/) does that count line by line, and [Fundamentals Part 6](/blog/transformer-block/) does the same for GPT-2.

## Common misconceptions

- **"You need advanced math to understand LLMs."** The forward pass is multiplication and addition, plus a few exponentials and square roots. Training adds derivatives, which Part 7 builds from the slope of a line. Research papers use heavier notation, but following how a model works takes high-school algebra and the four toolkits in this series.
- **"Tensors are something exotic."** In machine learning, a tensor is a block of numbers with a shape: a list is a 1-axis tensor and a grid is a 2-axis one. `torch.Tensor` and a NumPy array are both just numbers plus a shape.
- **"A model does something other than multiply and add."** More than 99.9% of Llama 3.1 8B's arithmetic is matrix multiplication. The rest (softmax, SiLU, RMSNorm) is a tiny share of the work, but it matters a lot: without those curved functions, a stack of matrix multiplies collapses into one matrix multiply, as Part 3 shows.
- **"A forward pass always costs 2 FLOPs per parameter."** That holds for short contexts. The embedding lookup is free, and attention adds work that grows with context: at 128,000 tokens, attention makes up 82% of Llama 3.1 8B's per-token arithmetic.

## Key takeaways

- A scalar is one number, a vector a list, a matrix a grid, and a tensor a block with any number of axes. A tensor's shape is the list of its axis sizes.
- A dot product multiplies two lists position by position and adds the results. A matrix multiply is a grid of dot products: (m × k) times (k × n) gives (m × n), for about 2mkn FLOPs.
- In Llama 3.1 8B, every block takes in a (B, T, 4096) stream and returns the same shape. Inside, the MLP widens to 14,336, and attention builds a T × T grid of scores for each of 32 heads.
- One token costs about 15.5 billion FLOPs at 1,000 tokens of context. That's close to 2 × 8.03 billion parameters, and more than 99.9% of it is dot products.
- Einsum strings name every axis with a letter, and letters missing from the output get summed. Once you can read `"btd,df->btf"`, you can follow the shapes in most model code.

## What's next

Everything in this article came down to dot products, so Part 2 takes a closer look at them. It shows what a vector's length and the angle between two vectors mean, why the dot product measures how much one vector points along another, and why two random vectors in 4,096 dimensions are almost always nearly perpendicular, which isn't true in the two or three dimensions you can draw.

## Further reading

- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): Table 3 lists the layers, widths, heads and vocabulary of the 8B, 70B and 405B models.
- [Llama 3.1 8B config.json (Hugging Face)](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json): every number that sets the shapes in this article, in one file.
- [Hugging Face Llama modeling code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py): the projections, reshapes and `repeat_kv`, to read alongside the shape table.
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361): section 2.1 and Table 1 count a Transformer's parameters and forward-pass FLOPs.
- [torch.einsum (PyTorch docs)](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) and [numpy.einsum (NumPy docs)](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html): the full einsum rules, with examples.
- [torch.nn.Linear (PyTorch docs)](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html): why stored weight shapes are (out, in).
- [Mathematics for Machine Learning (Deisenroth, Faisal & Ong, 2020)](https://mml-book.github.io/): a free book covering linear algebra, calculus and probability for machine learning; chapter 2 matches this article.
- [Essence of linear algebra (3Blue1Brown)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab): short animated videos that make vectors, matrices and dot products visual.
- [Let's build GPT: from scratch, in code, spelled out (Andrej Karpathy, 2023)](https://www.youtube.com/watch?v=kCc8FmEb1nY): a working Transformer built line by line, with the (B, T, C) shape of each tensor noted in the code as it goes.
