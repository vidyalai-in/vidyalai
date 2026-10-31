---
title: "The Math of Attention: Scaling, Masks, Jacobians and Low Rank"
description: "Attention derived by hand: why divide by √d, the mask as minus infinity, the softmax Jacobian, n² cost, GQA and MLA as factorizations, linear attention."
slug: "attention-math"
category: "llm-math"
tags: ["math", "attention", "transformers", "long-context", "architecture"]
series: "llm-math"
seriesOrder: 11
pubDate: 2026-05-08
---

## TL;DR

Attention is a soft dictionary lookup: every key matches a query a little, softmax turns the matches into weights, and the answer is a weighted average of the values. This part derives it with three tokens of four numbers each. The division by $$\sqrt{d}$$ falls out of Part 10's variance rules, and the causal mask adds minus infinity because $$e^{-\infty} = 0$$. The softmax's Jacobian is $$\text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^{\top}$$, which FlashAttention's backward pass uses without building it. The cost grows like $$n^2 d$$; multi-head, grouped-query and latent attention are low-rank factorizations; and linear attention and Mamba avoid the $$n^2$$ by reordering a matrix product, at the price of a fixed-size memory.

## Why it matters

Change the order of two matrix multiplications, and one attention head reading 128K tokens does 1,024 times less arithmetic: 8.6 billion operations instead of 8.8 trillion. Matrix multiplication is associative, so $$(AB)C = A(BC)$$, and [Part 3](/blog/matrices-linear-maps/) showed that the order you pick can change the cost enormously.

Standard attention can't take that shortcut, because one function sits between the two products: softmax. It's also why scores are divided by $$\sqrt{d}$$, why the mask uses minus infinity, and why gradients can stall. Linear attention and state-space models like Mamba drop it to escape the cost, and lose something in the trade.

[Fundamentals Part 5](/blog/attention-explained/) explained attention in words and worked it on a four-token sentence. This part derives each piece, using variance from [Part 10](/blog/initialization-normalization/), softmax from [Part 5](/blog/softmax-math/), Jacobians from [Part 7](/blog/derivatives-gradients/) and rank from Part 3.

## The intuition

A Python dictionary is a hard lookup: hand it a key, it finds the one stored key that matches exactly, and returns that key's value.

Attention makes the lookup soft. The query is scored against every stored key, softmax turns the scores into weights that add up to 1, and the answer is a blend of all the values, weighted by how well each key matched. When one key matches far better than the rest, it takes almost all the weight and the soft lookup becomes a hard one.

The analogy breaks in two places. A dictionary lookup fails when no key matches, while attention must hand out a full unit of weight even when no key fits, which is where the "attention sinks" in Fundamentals Part 5 come from. And every token is a query and a dictionary entry at the same time.

## Step 1: soft lookup, by hand

### From a hard lookup to weights

Write a hard lookup as a sum: the output is $$\sum_j w_j \mathbf{v}_j$$, where $$\mathbf{v}_j$$ is the value stored at position $$j$$ and $$w_j$$ is 1 for the matching key and 0 for the rest. To make it soft, let the weights grow with how well key $$\mathbf{k}_j$$ matches query $$\mathbf{q}$$. The dot product measures that match ([Part 2](/blog/vectors-dot-products/)), and softmax turns scores into positive weights that add up to 1 (Part 5). With a scale factor that Step 2 explains:

$$
w_j = \text{softmax}_j\!\left(\frac{\mathbf{q} \cdot \mathbf{k}_j}{\sqrt{d}}\right)
$$

Here $$d$$ is the number of entries in each query and key.

### Three tokens, four numbers each

Take three tokens with these queries, keys and values. A real model makes them by multiplying each token's vector by three learned matrices ([Fundamentals Part 5](/blog/attention-explained/)); these are picked for easy arithmetic.

| Token | Query | Key | Value |
|---|---|---|---|
| 1 | [1, 1, 0, 0] | [1, 0, 1, 0] | [1, 0, 1, 0] |
| 2 | [0, 0, 0, 2] | [0, 1, 0, 1] | [0, 2, 0, 0] |
| 3 | [2, 0, 2, 0] | [1, 1, 0, 0] | [0, 0, 0, 3] |

Token 3's query is [2, 0, 2, 0]. Its dot products with the three keys are $$2 + 2 = 4$$, $$0$$ and $$2$$. With $$d = 4$$, dividing by $$\sqrt{4} = 2$$ gives scores [2, 0, 1]. Their exponentials are 7.389, 1 and 2.718, which total 11.107, so the weights are 0.665, 0.090 and 0.245: the scores [2, 1, 0] from Part 5, in a different order.

The output is the weighted sum of the values:

```text
0.665 × [1, 0, 1, 0] + 0.090 × [0, 2, 0, 0] + 0.245 × [0, 0, 0, 3]
= [0.665, 0.180, 0.665, 0.734]
```

Token 3 mostly gets token 1's value, whose key lines up best with its query.

### The matrix form

Stack the queries as the rows of a matrix $$Q$$, and the keys and values as rows of $$K$$ and $$V$$, each $$n \times d$$ for $$n$$ tokens. Then $$QK^{\top}$$ is the $$n \times n$$ grid of every query dotted with every key, and the whole calculation is equation (1) of [Attention Is All You Need](https://arxiv.org/abs/1706.03762):

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right) V
$$

The softmax runs along each row. Here $$QK^{\top}$$ has rows [1, 1, 2], [0, 2, 0] and [4, 0, 2].

### The causal mask as minus infinity

A model that writes text mustn't let token 1 read tokens 2 and 3, which come after it. So it adds a mask matrix $$M$$ to the scores before softmax, with 0 where looking is allowed and $$-\infty$$ where it isn't:

$$
M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}
$$

Since $$e^{-\infty} = 0$$, a masked score adds nothing to the top or the bottom of the softmax fraction. The whole example in NumPy:

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)

Q = np.array([[1., 1, 0, 0], [0, 0, 0, 2], [2, 0, 2, 0]])
K = np.array([[1., 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
V = np.array([[1., 0, 1, 0], [0, 2, 0, 0], [0, 0, 0, 3]])
d = Q.shape[1]

S = Q @ K.T / np.sqrt(d)                       # scaled scores, 3 x 3
M = np.triu(np.full((3, 3), -np.inf), k=1)     # -inf above the diagonal

def softmax_rows(X):
    E = np.exp(X - X.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

P = softmax_rows(S + M)
print(P)
print(P @ V)
```

```text
[[1.    0.    0.   ]
 [0.269 0.731 0.   ]
 [0.665 0.09  0.245]]
[[1.    0.    1.    0.   ]
 [0.269 1.462 0.269 0.   ]
 [0.665 0.18  0.665 0.734]]
```

Token 1 sees only itself, token 2 splits 0.269 and 0.731 between tokens 1 and 2, and the upper triangle is exactly 0.

Setting the masked scores to 0 wouldn't work, because a score of 0 still gets weight $$e^0 = 1$$. Token 1's row would become [0.5, 0, 0], whose softmax is [0.452, 0.274, 0.274], with more than half its attention on tokens it shouldn't see.

A row with every entry masked would compute $$0/0$$, which comes out as NaN (not a number). That's one reason some code masks with a very large negative finite number instead: Hugging Face's [masking code](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py) uses `torch.finfo(dtype).min`, and [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) uses `-1e9`.

## Step 2: why divide by the square root of d

### The variance argument

Part 10 gave two rules about variance, the average squared distance from the mean. Rule one: multiplying by a constant $$a$$ multiplies variance by $$a^2$$. Rule two: for independent quantities, variances add.

Suppose the entries of a query and a key are independent random numbers with mean 0 and variance 1. One term $$q_i k_i$$ then has mean 0, and its variance is the average of $$q_i^2 k_i^2$$, which is $$1 \times 1 = 1$$. The score is a sum of $$d$$ such terms, so by rule two:

$$
\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d
$$

Its spread, the square root of the variance, is $$\sqrt{d}$$: about 11.3 for Llama 3.1 8B's 128-number heads. Dividing by $$\sqrt{d}$$ multiplies the variance by $$1/d$$ (rule one), back to 1 at any head size. This is footnote 4 of the 2017 paper, which assumes "the components of q and k are independent random variables with mean 0 and variance 1" and concludes that their dot product "has mean 0 and variance $$d_k$$."

### What unscaled scores do to softmax

The paper's worry was that large scores push "the softmax function into regions where it has extremely small gradients." To see how sharp softmax gets, give each of 500 random queries 1,024 random keys and measure the rows:

```python
import numpy as np

rng = np.random.default_rng(0)
n, trials = 1024, 500                        # 1,024 keys per query, 500 queries

def softmax_rows(X):
    E = np.exp(X - X.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

for d in [16, 64, 128, 256]:
    q = rng.standard_normal((trials, d))     # entries: mean 0, variance 1
    K = rng.standard_normal((n, d))
    S = q @ K.T                              # raw scores
    for name, X in [("raw", S), ("/sqrt(d)", S / np.sqrt(d))]:
        P = softmax_rows(X)
        H = -(P * np.log(P + 1e-300)).sum(axis=1)     # entropy of each row, in nats
        print(f"d={d:<4}{name:9} spread {X.std():5.2f}  top weight {P.max(axis=1).mean():.3f}"
              f"  entropy {H.mean():.2f}  rows with top > 0.99: {np.mean(P.max(axis=1) > 0.99):.0%}")
```

```text
d=16  raw       spread  3.99  top weight 0.420  entropy 2.45  rows with top > 0.99: 1%
d=16  /sqrt(d)  spread  1.00  top weight 0.017  entropy 6.43  rows with top > 0.99: 0%
d=64  raw       spread  8.04  top weight 0.702  entropy 0.93  rows with top > 0.99: 9%
d=64  /sqrt(d)  spread  1.00  top weight 0.016  entropy 6.43  rows with top > 0.99: 0%
d=128 raw       spread 11.38  top weight 0.817  entropy 0.53  rows with top > 0.99: 23%
d=128 /sqrt(d)  spread  1.01  top weight 0.017  entropy 6.43  rows with top > 0.99: 0%
d=256 raw       spread 16.02  top weight 0.861  entropy 0.37  rows with top > 0.99: 39%
d=256 /sqrt(d)  spread  1.00  top weight 0.017  entropy 6.43  rows with top > 0.99: 0%
```

Entropy ([Part 6](/blog/cross-entropy-perplexity/)) measures spread, and $$e$$ raised to it is the effective number of choices. With scaling, every head size gives 6.43, about 620 keys' worth of attention out of 1,024. Without it, at $$d = 128$$, the entropy is 0.53, about 1.7 keys, and in 23% of rows a single key takes more than 99% of the weight. At $$d = 256$$ that share reaches 39%.

The scaling removes the dependence on $$d$$. Trained queries and keys don't have unit-variance entries, and the model can learn larger or smaller ones, which acts like a learned temperature (Part 4). μP divides by $$d$$ instead, for reasons Part 10 covers.

## Step 3: the softmax Jacobian

### The full grid of rates

Part 5 stated how probability $$p_i$$ responds when score $$z_j$$ moves. To derive it, write $$p_i = e^{z_i} / Z$$, where $$Z = \sum_k e^{z_k}$$ is the total. Raising $$z_j$$ changes the top only if $$j = i$$, and it always changes the bottom, at rate $$e^{z_j}$$. The quotient rule from Part 7 gives:

$$
\frac{\partial p_i}{\partial z_j} = \frac{\delta_{ij}\, e^{z_i}}{Z} - \frac{e^{z_i}\, e^{z_j}}{Z^2} = p_i(\delta_{ij} - p_j)
$$

Here $$\delta_{ij}$$ is 1 when $$i = j$$ and 0 otherwise: a probability rises with its own score at rate $$p_i(1 - p_i)$$ and falls at rate $$p_i p_j$$ when another score rises. As a matrix:

$$
J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^{\top}
$$

where $$\text{diag}(\mathbf{p})$$ puts the probabilities on the diagonal of a square matrix and $$\mathbf{p}\mathbf{p}^{\top}$$ is the grid of all products $$p_i p_j$$.

For token 3's weights from Step 1, $$\mathbf{p} = [0.6652, 0.0900, 0.2447]$$:

$$
J = \begin{bmatrix} 0.2227 & -0.0599 & -0.1628 \\ -0.0599 & 0.0819 & -0.0220 \\ -0.1628 & -0.0220 & 0.1848 \end{bmatrix}
$$

The top-left entry is $$0.6652 \times 0.3348 = 0.2227$$, and nudging the first score by $$10^{-6}$$ reproduces the first column to four decimals. The matrix is symmetric, with a positive diagonal and negative entries elsewhere: raising one score takes probability from all the others. Every row adds up to 0, which is Part 5's shift invariance in derivative form, since raising all scores together changes nothing.

### Using it without building it

Backpropagation ([Part 8](/blog/backpropagation-math/)) never needs $$J$$ itself, only $$J$$ times an incoming gradient. If $$\bar{\mathbf{p}}$$ is the gradient of the loss with respect to the weights, the gradient with respect to the scores is

$$
\bar{\mathbf{z}} = J\bar{\mathbf{p}} = \mathbf{p} \circ \big(\bar{\mathbf{p}} - (\mathbf{p} \cdot \bar{\mathbf{p}})\big)
$$

where $$\circ$$ means multiply entry by entry. The $$\mathbf{p}\mathbf{p}^{\top}$$ part needs only the dot product $$\mathbf{p} \cdot \bar{\mathbf{p}}$$, so a row of $$n$$ scores costs about $$3n$$ operations instead of $$n^2$$.

Attention adds a second shortcut. The forward pass computes $$S = QK^{\top}/\sqrt{d}$$, $$P = \text{softmax}(S)$$ and $$O = PV$$. Given $$dO$$, the gradient of the loss with respect to the output, the backward pass in [FlashAttention's Appendix B.2](https://arxiv.org/abs/2205.14135) runs:

$$
dV = P^{\top} dO, \qquad dP = dO\, V^{\top}
$$

$$
dS_{ij} = P_{ij}\,(dP_{ij} - D_i), \qquad D_i = \sum_j P_{ij}\, dP_{ij}
$$

The second line is the formula above, applied to each row, and the trick is in $$D_i$$. Since $$dP_{ij} = dO_i \cdot \mathbf{v}_j$$, the sum is $$dO_i \cdot \sum_j P_{ij}\mathbf{v}_j$$, and that sum is the output row $$O_i$$. So $$D_i = dO_i \cdot O_i$$, a dot product of two vectors the backward pass already has. FlashAttention computes every $$D_i$$ first, then rebuilds $$P$$ block by block from the saved log-sum-exp, and never stores the $$n \times n$$ grid. Checked against PyTorch:

```python
import torch
torch.manual_seed(0)
n, d = 6, 4
Q, K, V = (torch.randn(n, d, dtype=torch.float64, requires_grad=True) for _ in range(3))
P = torch.softmax(Q @ K.T / d ** 0.5, dim=-1)
O = P @ V
dO = torch.randn(n, d, dtype=torch.float64)
O.backward(dO)                                 # PyTorch's own gradients

with torch.no_grad():
    dP = dO @ V.T
    D = (dO * O).sum(dim=1, keepdim=True)      # one number per row
    dS = P * (dP - D)
    dQ, dK, dV = dS @ K / d ** 0.5, dS.T @ Q / d ** 0.5, P.T @ dO
print([f"{(a - b).abs().max().item():.1e}" for a, b in [(dQ, Q.grad), (dK, K.grad), (dV, V.grad)]])
```

```text
['2.2e-16', '2.2e-16', '0.0e+00']
```

### When attention saturates

If one weight is 1 and the rest are 0, then $$\text{diag}(\mathbf{p})$$ and $$\mathbf{p}\mathbf{p}^{\top}$$ are the same matrix and $$J = 0$$, so no gradient reaches the scores or the query and key matrices behind them. Multiply Step 1's scores [2, 0, 1] by 8, about what skipping the scale does at $$d = 64$$: the top weight becomes 0.99966, and the top-left entry of $$J$$ drops from 0.2227 to 0.000335, 664 times smaller. Step 2 found 23% of unscaled rows in that state at $$d = 128$$.

Training can still grow scores until they saturate, so current models add the guards Part 10 described: QK-norm, which normalizes queries and keys, and soft-capping with $$\tanh$$.

## Step 4: what n tokens cost

For one head, $$QK^{\top}$$ multiplies an $$n \times d$$ matrix by a $$d \times n$$ one: $$2n^2 d$$ floating-point operations (FLOPs) by Part 1's rule. Multiplying the $$n \times n$$ weights by $$V$$ costs the same again, so a head costs $$4n^2 d$$ FLOPs and holds $$n^2$$ scores. That's the $$O(n^2 \cdot d)$$ in the 2017 paper's Table 1. With Llama 3.1 8B's shapes (32 layers, 32 query heads of $$d = 128$$, 8 key-value heads) with the per-head rows counting the full grid and the whole-model rows letting the causal mask skip about half the pairs:

| Quantity | 8,192 tokens | 131,072 tokens |
|---|---|---|
| Scores per head, per layer | 67.1 million | 17.2 billion |
| Memory for those scores at 2 bytes | 134 MB | 34.4 GB |
| Same, all 32 heads of one layer | 4.3 GB | 1,100 GB |
| Score and value FLOPs, one head, one layer, full grid | 3.4 × 10^10 | 8.8 × 10^12 |
| Attention scores and values, whole model | 1.76 × 10^13 | 4.50 × 10^15 |
| Q, K, V and output projections, whole model | 2.20 × 10^13 | 3.52 × 10^14 |
| Scores and values ÷ projections | 0.80 | 12.8 |

Sixteen times the tokens gives 256 times the scores, while the projections, which multiply each token by fixed weight matrices, grow 16 times. Per token, the projections cost 2 FLOPs per weight. Llama's four attention matrices hold $$4096 \times (4096 + 1024 + 1024 + 4096) = 41.9$$ million weights per layer, so 83.9 million FLOPs. A token at position $$t$$ costs $$4t \times 4096$$ in scores and values across all heads. They're equal at

$$
t = d_{\text{model}} + n_{\text{kv}} d_{\text{head}} = 4096 + 1024 = 5{,}120
$$

where $$n_{\text{kv}}$$ is the number of key-value heads. Past position 5,120, a token spends more on attention than on its attention weights; over a whole prompt, where the average token sits halfway, the crossover is 10,240 tokens. [Part 1](/blog/math-for-llms/) and [Fundamentals Part 5](/blog/attention-explained/) compare these terms with the rest of the model.

FlashAttention removes the memory problem by keeping the $$n \times n$$ grid in tiles on the chip (Part 5's online softmax; [Inference Part 7](/blog/flashattention/)), but not the $$n^2$$ arithmetic.

## Step 5: multi-head attention as low-rank pieces

### The QK circuit

Write head $$h$$'s query and key as $$\mathbf{q}_i = W_Q^h \mathbf{x}_i$$ and $$\mathbf{k}_j = W_K^h \mathbf{x}_j$$, where $$\mathbf{x}$$ is a token's vector of $$d_{\text{model}}$$ numbers and each $$W$$ is $$d_{\text{head}} \times d_{\text{model}}$$. The score is

$$
\mathbf{q}_i \cdot \mathbf{k}_j = \mathbf{x}_i^{\top} \big(W_Q^{h\top} W_K^h\big) \mathbf{x}_j = \mathbf{x}_i^{\top} W_{QK}^h\, \mathbf{x}_j
$$

The queries and keys have disappeared, leaving one $$d_{\text{model}} \times d_{\text{model}}$$ matrix, $$W_{QK}^h$$, that scores any pair of token vectors directly. For Llama 3.1 8B it's 4,096 by 4,096, 16.8 million entries built from two factors of 524,288 each. Part 3's rule that a product's rank is at most the smaller rank caps it at 128: only 128 of the 4,096 directions matter to this head's scores.

On the value side, the head reads $$\mathbf{v}_j = W_V^h \mathbf{x}_j$$ and its slice $$W_O^h$$ of the output projection writes back, so what reaches the residual stream is $$W_O^h W_V^h \mathbf{x}_j = W_{OV}^h \mathbf{x}_j$$, again 4,096 by 4,096 with rank at most 128. Multi-head attention is a sum of 32 such pieces, and since $$32 \times 128 = 4{,}096$$, together they can cover the whole space.

Anthropic's [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Elhage et al., 2021) calls these the QK and OV circuits: "W_O W_V and W_Q^T W_K can always be thought of as individual, low-rank matrices," and "keys, queries and value vectors are, in some sense, superficial."

### Checking a real model

Qwen2.5-1.5B has $$d_{\text{model}} = 1{,}536$$ and 12 query heads of 128, which share 2 key-value heads. Multiply out head 0's two circuits in the first layer:

```python
import glob, numpy as np
from safetensors import safe_open

path = glob.glob("hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors")[0]
with safe_open(path, "pt") as f:
    get = lambda n: f.get_tensor(f"model.layers.0.self_attn.{n}.weight").double().numpy()
    Wq, Wk, Wv, Wo = get("q_proj"), get("k_proj"), get("v_proj"), get("o_proj")

dh = 128                                         # head 0 uses key/value head 0
W_QK = Wq[:dh].T @ Wk[:dh]                       # 1536 x 1536
W_OV = Wo[:, :dh] @ Wv[:dh]                      # 1536 x 1536
for name, W in [("W_QK", W_QK), ("W_OV", W_OV)]:
    s = np.linalg.svd(W, compute_uv=False)
    print(f"{name} {W.shape}: rank {np.linalg.matrix_rank(W)}, "
          f"singular values #1 {s[0]:.3g}, #128 {s[127]:.3g}, #129 {s[128]:.3g}")
```

```text
W_QK (1536, 1536): rank 128, singular values #1 12, #128 0.208, #129 6.04e-15
W_OV (1536, 1536): rank 128, singular values #1 1.56, #128 0.0832, #129 1.06e-15
```

Singular values ([Part 13](/blog/low-rank-svd/)) measure how much a matrix stretches each independent direction. Both matrices have exactly 128 nonzero ones; the 129th is rounding error. The real model adds two details: Qwen's query and key projections have biases, and RoPE ([Part 12](/blog/rope-math/)) rotates queries and keys by position, so the score matrix depends on the distance between tokens, with rank at most 128 at every distance.

## Step 6: GQA and MLA as factorizations

### Grouped-query attention: shared right factors

In grouped-query attention (GQA), from [Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245), groups of query heads share one key head and one value head. In circuit terms, the four heads of a Llama group have score matrices $$W_Q^{h\top} W_K^{g}$$ with the same right factor $$W_K^g$$: each asks its own question of the same 128-number view of every token.

With one group it becomes multi-query attention (MQA), and with one group per head, ordinary multi-head attention. The cache holds $$2 \times n_{\text{kv}} \times d_{\text{head}}$$ numbers per token and layer: 2,048 for Llama 3.1 8B's 8 key-value heads, against 8,192 with 32, a factor of 4. Multi-query attention would cut it by 32, and Qwen2.5-1.5B's 2 key-value heads for 12 query heads cut it by 6. [Inference Part 2](/blog/kv-cache/) turns these factors into bytes and users per GPU.

### Multi-head latent attention: a low-rank key and value

DeepSeek's multi-head latent attention (MLA), from [DeepSeek-V2](https://arxiv.org/abs/2405.04434), factors keys and values through a small shared vector, the latent:

$$
\mathbf{c}_t = W_{DKV}\, \mathbf{h}_t, \qquad \mathbf{k}_t^h = W_{UK}^h\, \mathbf{c}_t, \qquad \mathbf{v}_t^h = W_{UV}^h\, \mathbf{c}_t
$$

Here $$\mathbf{h}_t$$ is token $$t$$'s vector, $$W_{DKV}$$ projects it down to the latent $$\mathbf{c}_t$$, and $$W_{UK}^h$$ and $$W_{UV}^h$$ rebuild head $$h$$'s key and value. DeepSeek-V2 has a 512-number latent and 128 heads of 128. The map from a token to all 16,384 key numbers is $$W_{UK} W_{DKV}$$, with an inner size of 512, so its rank is at most 512: MLA is a low-rank factorization, and the cache stores only the thin middle.

Associativity removes the need to rebuild keys. Head $$h$$'s score for the new token $$t$$ against cached token $$j$$ is

$$
\mathbf{q}_t^{h\top} \mathbf{k}_j^h = \mathbf{q}_t^{h\top} W_{UK}^h\, \mathbf{c}_j = \big(W_{UK}^{h\top}\, \mathbf{q}_t^h\big)^{\top} \mathbf{c}_j
$$

Moving the parentheses carries the query into the latent space. Since $$\mathbf{q}_t^h = W_Q^h \mathbf{h}_t$$, the product $$W_{UK}^{h\top} W_Q^h$$ can be multiplied out once, ahead of time. (DeepSeek-V2 compresses its queries too, so its $$W^{UQ}$$ plays the part of $$W_Q^h$$.) In the paper's words, "due to the associative law of matrix multiplication, we can absorb $$W^{UK}$$ into $$W^{UQ}$$, and $$W^{UV}$$ into $$W^{O}$$." A small check:

```python
import numpy as np
rng = np.random.default_rng(0)
d, dc, H, dh, n = 64, 16, 4, 8, 10       # model width, latent, heads, head size, cached tokens

W_DKV = rng.standard_normal((dc, d))      # down-projection: h -> latent c (cached)
W_UK  = rng.standard_normal((H, dh, dc))  # per-head up-projection: c -> key
W_Q   = rng.standard_normal((H, dh, d))   # per-head query projection

C = rng.standard_normal((n, d)) @ W_DKV.T # cache: 16 numbers per earlier token
h_now = rng.standard_normal(d)            # the token being generated
for i in range(H):
    q = W_Q[i] @ h_now
    naive = (C @ W_UK[i].T) @ q           # rebuild 8-number keys, then score
    absorbed = C @ (W_UK[i].T @ q)        # 16-number query against the latents
    print(f"head {i}: max difference {np.abs(naive - absorbed).max():.1e}")
```

```text
head 0: max difference 5.7e-13
head 1: max difference 4.5e-13
head 2: max difference 3.4e-13
head 3: max difference 2.3e-13
```

No key is built. The absorbed query has 512 numbers instead of 128, which costs multiply-adds, but decode is limited by memory reads ([Inference Part 1](/blog/how-llm-inference-works/)).

### Why RoPE gets in the way

Rotary position embedding (RoPE) multiplies each key by a rotation matrix $$R_j$$ that depends on its position. On the rebuilt key, the score becomes $$\mathbf{q}_t^{\top} R_t^{\top} R_j W_{UK} \mathbf{c}_j$$, with a position-dependent matrix between the query and $$W_{UK}$$, so no single matrix can be absorbed ahead of time. In DeepSeek-V2's words, "a RoPE matrix related to the currently generating token will lie between $$W^Q$$ and $$W^{UK}$$ and matrix multiplication does not obey a commutative law."

The fix "decouples" position: each head's query gets 64 extra numbers that carry RoPE, matched against one 64-number key shared by all heads and cached with the latent. That's $$512 + 64 = 576$$ cached numbers per token per layer, against $$2 \times 128 \times 128 = 32{,}768$$ for multi-head attention with the same heads, 57 times less, or what GQA would store with $$576/256 = 2.25$$ groups. [Part 12](/blog/rope-math/) derives why rotations behave this way.

## Step 7: around the n² wall

### Linear attention: reorder the product

Write one row of attention with a general similarity function $$\text{sim}$$:

$$
\mathbf{o}_i = \frac{\sum_j \text{sim}(\mathbf{q}_i, \mathbf{k}_j)\, \mathbf{v}_j}{\sum_j \text{sim}(\mathbf{q}_i, \mathbf{k}_j)}
$$

Softmax attention uses $$\text{sim} = e^{\mathbf{q} \cdot \mathbf{k} / \sqrt{d}}$$. [Katharopoulos et al. (2020)](https://arxiv.org/abs/2006.16236) used $$\phi(\mathbf{q}) \cdot \phi(\mathbf{k})$$ instead, where $$\phi$$ is applied to every entry and keeps similarities positive; their $$\phi(x) = \text{elu}(x) + 1$$ is $$x + 1$$ for positive $$x$$ and $$e^x$$ otherwise. Now $$\phi(\mathbf{q}_i)$$ comes out of both sums:

$$
\mathbf{o}_i = \frac{\phi(\mathbf{q}_i)^{\top} \sum_j \phi(\mathbf{k}_j)\, \mathbf{v}_j^{\top}}{\phi(\mathbf{q}_i)^{\top} \sum_j \phi(\mathbf{k}_j)}
$$

In matrix form, the numerator is $$\big(\phi(Q)\phi(K)^{\top}\big)V = \phi(Q)\big(\phi(K)^{\top}V\big)$$. The left side builds an $$n \times n$$ grid; the right side builds a $$d \times d$$ matrix first, for about $$4nd^2$$ FLOPs instead of $$4n^2d$$. For one head with $$d = 128$$ at 128K tokens, that's the opening's 8.6 billion against 8.8 trillion, a factor of $$n/d = 1{,}024$$.

![Two paths from the same Q, K and V, each n tokens by d numbers, to the same n by d output. Softmax attention multiplies Q by K-transpose first, giving an n by n grid of scores, adds the mask, takes the softmax of each row, and multiplies the weights by V, with cost that grows with n squared. Linear attention multiplies phi of K, transposed, by V first, giving a d by d state, then multiplies phi of Q by that state, with cost that grows with n.](/diagrams/m11-two-orders.svg "The same three matrices, multiplied in two orders. Softmax between the first two products forces the expensive order.")

Softmax attention must take the top path, because, as the paper notes, the exponential of a dot product doesn't split into a dot product of finite-size pieces.

### It's a recurrence

With a causal mask, row $$i$$ only sums over $$j \le i$$, so running totals work:

$$
S_i = S_{i-1} + \phi(\mathbf{k}_i)\, \mathbf{v}_i^{\top}, \qquad \mathbf{z}_i = \mathbf{z}_{i-1} + \phi(\mathbf{k}_i)
$$

$$
\mathbf{o}_i = \frac{\phi(\mathbf{q}_i)^{\top} S_i}{\phi(\mathbf{q}_i)^{\top} \mathbf{z}_i}
$$

The state is a $$d \times d$$ matrix $$S$$ and a $$d$$-vector $$\mathbf{z}$$, the same size for 10 tokens or 10 million. That's a recurrent neural network (RNN), hence the paper's title and its observation that "any transformer layer with causal masking can be written as a model that, given an input, modifies an internal state and then predicts an output." It reported generation "up to 4000x faster" on long sequences. A check:

```python
import numpy as np
rng = np.random.default_rng(0)
n, d = 512, 16
Q, K, V = (rng.standard_normal((n, d)) for _ in range(3))

def phi(x):                                  # elu(x) + 1, always positive
    return np.where(x > 0, x + 1, np.exp(x))

# Quadratic form: build the n x n grid, mask the future, normalize each row
A = np.tril(phi(Q) @ phi(K).T)
out_quadratic = (A @ V) / A.sum(axis=1, keepdims=True)

# Recurrent form: a d x d state and a d-vector, updated once per token
S, z = np.zeros((d, d)), np.zeros(d)
out_recurrent = np.zeros((n, d))
for i in range(n):
    S += np.outer(phi(K[i]), V[i])
    z += phi(K[i])
    out_recurrent[i] = phi(Q[i]) @ S / (phi(Q[i]) @ z)

print("max difference:", np.abs(out_quadratic - out_recurrent).max())
print("numbers kept: grid", n * n, " state", d * d + d)
```

```text
max difference: 4.440892098500626e-16
numbers kept: grid 262144  state 272
```

### What a fixed-size state forgets

Store $$n$$ random key-value pairs as $$S = \sum_j \mathbf{k}_j \mathbf{v}_j^{\top}$$ and look one up with $$\mathbf{k}_j^{\top} S$$. You get $$\mathbf{v}_j$$ back, plus every other value weighted by its key's overlap with $$\mathbf{k}_j$$: about $$1/\sqrt{d}$$ each (Part 2), but $$n - 1$$ of them. We stored and recalled random pairs with $$d = 64$$, and did the same with softmax attention:

| Pairs stored | Linear state (numbers) | Softmax cache (numbers) | Recall, linear | Recall, softmax |
|---|---|---|---|---|
| 16 | 4,096 | 2,048 | 0.91 | 1.00 |
| 64 | 4,096 | 8,192 | 0.69 | 1.00 |
| 256 | 4,096 | 32,768 | 0.45 | 1.00 |
| 1,024 | 4,096 | 131,072 | 0.24 | 1.00 |

Recall is the cosine between the value retrieved and the value stored; the linear column follows $$1/\sqrt{1 + n/d}$$. Softmax keeps every pair, so its memory grows with $$n$$ and its recall holds. [Jelassi et al. (2024)](https://arxiv.org/abs/2402.01032) proved that fixed-state models "are fundamentally limited by their fixed-size latent state" when copying, and found pretrained Transformers "dramatically outperform state space models at copying and retrieving information from context."

### State-space models and Mamba

A state-space model updates a hidden state $$\mathbf{h}$$ with a linear recurrence:

$$
\mathbf{h}_t = \bar{A}\, \mathbf{h}_{t-1} + \bar{B}\, x_t, \qquad y_t = C\, \mathbf{h}_t
$$

Here $$x_t$$ is the input at step $$t$$, $$y_t$$ the output, and $$\bar{A}$$, $$\bar{B}$$ and $$C$$ are matrices; $$\bar{A}$$ sets how much of the old state survives each step. Older designs kept them fixed. [Mamba](https://arxiv.org/abs/2312.00752) (Gu and Dao, 2023) computes them from the current token, with $$\bar{A} = \exp(\Delta A)$$ and an input-dependent step size $$\Delta$$, so the model can "selectively propagate or forget information along the sequence length dimension depending on the current token." The paper's Theorem 1 shows the simplest case is a gate, $$h_t = (1 - g_t)\, h_{t-1} + g_t\, x_t$$, with $$g_t$$ between 0 and 1 computed from the token: near 1 overwrites the memory, near 0 ignores the token.

Linear attention is this kind of recurrence with $$\bar{A} = 1$$: nothing is ever forgotten. Mamba's authors put the trade-off this way: "a fundamental problem of sequence modeling is compressing context into a smaller state," and attention "is both effective and inefficient because it explicitly does not compress context at all." Mamba reported 5 times the generation throughput of a Transformer of similar size, and the recall table shows the cost.

## In the real world

- PyTorch's [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) uses `scale_factor = 1 / math.sqrt(query.size(-1))` by default, and its reference code builds the causal mask with `masked_fill_(..., float("-inf"))`. Its `enable_gqa` option shares key-value heads by copying them with `repeat_interleave`.
- Gemma 2 27B doesn't divide by the square root of its head size. Its [config](https://huggingface.co/unsloth/gemma-2-27b/blob/main/config.json) has `head_dim` 128 but `query_pre_attn_scalar` 144 (4,608 divided by 32 heads), which Hugging Face's [Gemma 2 code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py) turns into `self.scaling = config.query_pre_attn_scalar**-0.5`, a division by 12 instead of 11.3. It also caps scores with `attn_logit_softcapping` 50.
- With Step 3's $$D_i$$, FlashAttention's backward pass "can also be computed with $$O(n)$$ extra memory," and DeepSeek-V2 reports that MLA's 576-number cache comes with performance "stronger than MHA."
- Production models mix attention with recurrences. AI21's [Jamba](https://arxiv.org/abs/2403.19887) uses "a : m = 1 : 7," one attention layer for every seven Mamba layers, with a 4 GB cache at 256K tokens ([Inference Part 11](/blog/long-context-inference/)). Qwen3-Next-80B-A3B's [model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) lists the layout "12 \* (3 \* (Gated DeltaNet -> MoE) -> 1 \* (Gated Attention -> MoE))": 36 of its 48 layers use [Gated DeltaNet](https://arxiv.org/abs/2412.06464), a gated linear-attention recurrence. Only 12 keep a cache, which by our arithmetic (2 key-value heads of 256) is $$12 \times 2 \times 2 \times 256 = 12{,}288$$ numbers per token, a quarter of what 48 attention layers would store.

## Common misconceptions

- **"Dividing by √d stops the numbers from overflowing."** Subtracting the max handles overflow (Part 5). The scaling controls sharpness: without it, at $$d = 128$$, 23% of random rows put over 99% of their weight on one key, where the Jacobian is nearly zero.
- **"The causal mask multiplies future scores by zero."** A score of 0 still gets weight $$e^0 = 1$$: zeroing token 1's future scores in Step 1 left 55% of its attention on tokens it can't see. Adding minus infinity drops those terms entirely.
- **"Backpropagating through attention means multiplying by each row's n × n Jacobian."** That would cost $$n^3$$ per head. The row gradient is $$\mathbf{p} \circ (\bar{\mathbf{p}} - \mathbf{p} \cdot \bar{\mathbf{p}})$$, and attention needs only one extra dot product per row, $$D_i = dO_i \cdot O_i$$.
- **"MLA saves compute."** It saves memory: 576 cached numbers per token per layer instead of 32,768. Its absorbed queries have 512 numbers instead of 128, so scoring each cached token takes more multiply-adds.
- **"Linear attention is a cheaper approximation of the same thing."** It uses a different similarity and a fixed-size memory. In Step 7's test, its recall fell to 0.24 with 1,024 stored pairs; softmax stayed at 1.00.

## Key takeaways

- Attention is a soft dictionary lookup: score the query against every key, divide by the square root of the head size, softmax, and average the values. The causal mask adds minus infinity, so future tokens get exactly zero weight.
- Query-key dot products have variance d. Dividing by the square root of d brings it back to 1 and keeps softmax from collapsing onto one key, where its Jacobian, diag p minus p p transpose, goes to zero.
- The backward pass never builds that Jacobian. Each row needs one extra number, the output gradient dotted with the output, which lets FlashAttention skip the n by n grid.
- Each head scores tokens with a model-width matrix of rank at most the head size. Grouped-query attention shares the key factor; latent attention factors keys and values through a small latent and uses associativity to skip rebuilding them.
- Softmax forces queries times keys first, which costs n squared. Linear attention and Mamba multiply keys by values first and keep a fixed-size state: fast, but forgetful, so production models mix the two.

## What's next

RoPE kept getting in the way here, putting a rotation between query and key and breaking MLA's absorb trick. [Part 12](/blog/rope-math/), Rotations, complex numbers and RoPE, builds rotation matrices from the plane up, shows why rotating both vectors makes the score depend only on the distance between tokens, and works out the math behind stretching a model's context window.

## Further reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): equation (1), footnote 4, the mask and Table 1's costs.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135): Appendix B.2 derives the backward pass and the $$D_i$$ shortcut.
- [A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)](https://transformer-circuits.pub/2021/framework/index.html): QK and OV circuits as low-rank matrices.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245): grouped-query attention.
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (DeepSeek-AI, 2024)](https://arxiv.org/abs/2405.04434): MLA, the absorb trick and decoupled RoPE (section 2.1, Appendix C).
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236): reassociation and the recurrent form.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752): selective state-space recurrences.
- [Repeat After Me: Transformers are Better than State Space Models at Copying (Jelassi et al., 2024)](https://arxiv.org/abs/2402.01032): what a fixed-size state can't copy.
- [Jamba: A Hybrid Transformer-Mamba Language Model (Lieber et al., 2024)](https://arxiv.org/abs/2403.19887) and [Gated Delta Networks: Improving Mamba2 with Delta Rule (Yang et al., 2024)](https://arxiv.org/abs/2412.06464): building blocks of hybrid models.
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/): the 2017 model in runnable PyTorch, mask included.
