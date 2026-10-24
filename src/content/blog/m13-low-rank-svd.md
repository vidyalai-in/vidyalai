---
title: "Low Rank and the SVD: How Much of a Matrix Really Matters"
description: "Rank-1 pieces, the SVD as rotate-stretch-rotate, Eckart-Young's best low-rank copy, real Qwen weight spectra, and why LoRA and MLA use thin matrices."
slug: "low-rank-svd"
category: "llm-math"
tags: ["math", "lora", "fine-tuning", "transformers", "architecture"]
series: "llm-math"
seriesOrder: 13
pubDate: 2026-10-24
---

## TL;DR

Every matrix is a sum of simple pieces, each one a column times a row. The singular value decomposition (SVD) finds those pieces and sorts them by size, and keeping the biggest few gives the best possible low-rank copy, a fact proved in 1936. This article builds the SVD from a 2 × 2 example you can do by hand, then takes apart real weight matrices from Qwen2.5-1.5B. Trained weights turn out not to be low rank: chopping them with the SVD wrecks the model, while 4-bit rounding barely hurts it. What is low rank is the change fine-tuning makes, which is why LoRA works, and DeepSeek's MLA builds low rank in on purpose.

## Why it matters

A LoRA adapter for Llama 3.1 8B's query projection trains 65,536 numbers in place of the matrix's 16,777,216, which is 0.39% of them. Its authors report quality on par with or better than changing every weight, on models from RoBERTa to GPT-3.

So you might guess that the weights themselves are mostly empty space, and that a model could be squeezed the same way. Try it on Qwen2.5-1.5B: replace every attention and MLP matrix in its 28 blocks with its best approximation at half the storage, and its perplexity (the effective number of choices per token) on a passage of *The Wonderful Wizard of Oz* goes from 8.46 to about 1.8 million. A model that guessed uniformly among its 151,936 tokens would do better.

The SVD explains both: the weights use most of their directions, while the changes made in fine-tuning use very few.

[Part 3](/blog/matrices-linear-maps/) defined rank as the number of independent directions that survive a matrix, and [Part 9](/blog/optimizers-adam/) used the SVD in one line for the Muon optimizer.

## The intuition

Picture a table of a shop chain's monthly sales: one row per store, one column per product. If every store sold products in the same proportions, just scaled by how big the store is, you could rebuild the whole table from two lists: a size for each store and a popularity for each product. Multiply them and you get every cell.

Real stores aren't that tidy. City stores sell more coffee, rural ones more tools, so you need a second pattern, then a third. The SVD finds these patterns in order of how much of the table they explain. If a few explain almost everything, a few short lists replace the whole grid.

The analogy breaks in two places. The SVD's patterns are forced to be perpendicular and can have negative entries, so they rarely match anything you could name, like "city versus rural". And a weight matrix transforms vectors, so what counts is how much its outputs change when you simplify it, which is a much harder test than how closely the numbers match.

## Step 1: rank-1 matrices are a column times a row

### A multiplication table

Here is a times table with 3 rows and 4 columns:

```text
      1   2   3   4
 1    1   2   3   4
 2    2   4   6   8
 3    3   6   9  12
```

Its 12 entries come from 7 numbers: the column [1, 2, 3] down the side and the row [1, 2, 3, 4] across the top. Entry $$(i, j)$$ is the side number times the top number. That operation, a column times a row, is called the outer product:

$$
(\mathbf{u}\mathbf{v}^{\top})_{ij} = u_i \, v_j
$$

Here $$\mathbf{u}$$ is a column of $$m$$ numbers, $$\mathbf{v}$$ is a column of $$n$$ numbers, and the transpose $$\mathbf{v}^{\top}$$ lays $$\mathbf{v}$$ on its side as a row. It's matrix multiplication from [Part 1](/blog/math-for-llms/) with an inner size of 1: (m × 1) times (1 × n) gives m × n.

Every column of the times table is a multiple of [1, 2, 3], so its rank is 1. Part 3's rank-1 matrix, with rows [1, 2] and [2, 4], is [1, 2] times [1, 2] laid on its side.

### Storage: m + n instead of mn

A rank-1 matrix stores $$m + n$$ numbers in place of $$m \times n$$. For the times table that's 7 against 12. For a 4,096 × 4,096 matrix, the size of Llama 3.1 8B's query projection, it's 8,192 against 16,777,216, a 2,048-fold saving.

One piece is rarely enough. A matrix of rank $$r$$ is a sum of $$r$$ rank-1 pieces, and stacking the columns side by side and the rows on top of each other writes that sum as a product of two thin matrices:

$$
W = \mathbf{u}_1\mathbf{v}_1^{\top} + \dots + \mathbf{u}_r\mathbf{v}_r^{\top} = B A
$$

Here $$B$$ is $$m \times r$$, with the $$\mathbf{u}$$'s as its columns, and $$A$$ is $$r \times n$$, with the $$\mathbf{v}$$'s as its rows. Storing them takes $$r(m + n)$$ numbers. Multiplying a vector by them in two steps, first by $$A$$ and then by $$B$$, takes $$2r(m+n)$$ FLOPs instead of $$2mn$$.

The saving disappears as $$r$$ grows. The thin pair costs the same as the full matrix when $$r(m+n) = mn$$, which gives a break-even rank:

$$
r_{\text{even}} = \frac{mn}{m+n}
$$

For a square matrix that is exactly half its size. Qwen2.5-1.5B's 1,536 × 1,536 query projection breaks even at rank 768; its 8,960 × 1,536 MLP matrices at rank 1,311. Keep that number in mind for Step 4.

## Step 2: the SVD, rotate, stretch, rotate

### A circle becomes an ellipse

Take this matrix:

$$
A = \begin{bmatrix} 3 & 0 \\ 4 & 5 \end{bmatrix}
$$

Feed it every point $$[\cos\theta, \sin\theta]$$ on the unit circle, where $$\theta$$ (theta) is the angle. The outputs form an ellipse, a stretched circle. The output is $$[3\cos\theta,\ 4\cos\theta + 5\sin\theta]$$, and its squared length, using $$\cos^2\theta + \sin^2\theta = 1$$ and $$2\sin\theta\cos\theta = \sin 2\theta$$, tidies up to:

$$
\lVert A\mathbf{x} \rVert^2 = 25 + 20 \sin 2\theta
$$

Since $$\sin 2\theta$$ swings between -1 and 1, the squared length swings between 5 and 45. It's largest at $$\theta = 45°$$ and smallest at $$\theta = 135°$$, a right angle apart:

| Input angle | Input | Output | Length of output |
|---|---|---|---|
| 0° | [1, 0] | [3, 4] | 5 |
| 45° | [0.707, 0.707] | [2.121, 6.364] | 6.708 = √45 |
| 90° | [0, 1] | [0, 5] | 5 |
| 135° | [-0.707, 0.707] | [-2.121, 0.707] | 2.236 = √5 |

The ellipse's longest radius, $$\sigma_1 = \sqrt{45} \approx 6.708$$, and its shortest, $$\sigma_2 = \sqrt{5} \approx 2.236$$, are the singular values of $$A$$ ($$\sigma$$ is the Greek letter sigma). The input directions that produce them, $$\mathbf{v}_1 = [1, 1]/\sqrt{2}$$ and $$\mathbf{v}_2 = [-1, 1]/\sqrt{2}$$, are the right singular vectors. Divide each output by its length and you get the left singular vectors, the directions of the ellipse's axes: $$\mathbf{u}_1 = [1, 3]/\sqrt{10}$$ and $$\mathbf{u}_2 = [-3, 1]/\sqrt{10}$$. Both pairs are perpendicular, which you can check with the dot product from [Part 2](/blog/vectors-dot-products/): $$1 \times (-3) + 3 \times 1 = 0$$.

### The decomposition

The matrix sends $$\mathbf{v}_1$$ to $$\sigma_1\mathbf{u}_1$$ and $$\mathbf{v}_2$$ to $$\sigma_2\mathbf{u}_2$$, and any input is a mix of the two, so the matrix is two rank-1 pieces:

$$
A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^{\top} + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^{\top}
$$

With the numbers in, $$\sigma_1\mathbf{u}_1\mathbf{v}_1^{\top} = 6.708 \times [0.316, 0.949]^{\top}[0.707, 0.707]$$, and the two pieces come out as clean grids:

$$
\begin{bmatrix} 1.5 & 1.5 \\ 4.5 & 4.5 \end{bmatrix} +
\begin{bmatrix} 1.5 & -1.5 \\ -0.5 & 0.5 \end{bmatrix} =
\begin{bmatrix} 3 & 0 \\ 4 & 5 \end{bmatrix}
$$

Written with matrices, putting the $$\mathbf{u}$$'s as the columns of $$U$$, the $$\mathbf{v}$$'s as the columns of $$V$$, and the singular values on the diagonal of $$\Sigma$$ (capital sigma), that's the singular value decomposition:

$$
A = U \Sigma V^{\top}
$$

Read it right to left. $$V^{\top}$$ turns the input so that $$\mathbf{v}_1$$ lines up with the first axis and $$\mathbf{v}_2$$ with the second. $$\Sigma$$ stretches each axis by its singular value. $$U$$ turns the result so that the axes point along $$\mathbf{u}_1$$ and $$\mathbf{u}_2$$. Here $$V^{\top}$$ is a rotation by -45° and $$U$$ a rotation by 71.6°.

![A top-to-bottom flow. The point x = (0.71, 0.71) on the unit circle goes through V transposed, a rotation by minus 45 degrees, and becomes (1, 0). Sigma stretches the axes by 6.71 and 2.24, giving (6.71, 0). U rotates by 71.6 degrees, giving Ax = (2.12, 6.36). A dotted branch from the stretch step reads: truncate, set 2.24 to 0, which gives the best rank-1 copy of A.](/diagrams/m13-svd-steps.svg "The SVD of A = [[3, 0], [4, 5]] applied to the input direction v1: rotate, stretch, rotate.")
### In general, and in NumPy

Every real $$m \times n$$ matrix has an SVD. $$U$$ and $$V$$ are orthogonal matrices: their columns have length 1 and are perpendicular to each other, so they move vectors without changing lengths or angles (a rotation, sometimes combined with a mirror flip). The singular values are never negative and come sorted, $$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$$, and there are $$\min(m, n)$$ of them. The number that aren't zero is the rank, so the times table has one nonzero singular value and Part 3's 768 × 768 product through a 64-wide middle has 64.

NumPy computes all of it in one call:

```python
import numpy as np
A = np.array([[3.0, 0.0],
              [4.0, 5.0]])
U, S, Vt = np.linalg.svd(A)
print("singular values:", S.round(4))
print("U:\n", U.round(4))
print("V transposed:\n", Vt.round(4))
print("U @ diag(S) @ Vt:\n", (U @ np.diag(S) @ Vt).round(4))
A1 = S[0] * np.outer(U[:, 0], Vt[0])          # the first rank-1 piece alone
print("rank-1 piece:\n", A1.round(4))
```

```text
singular values: [6.7082 2.2361]
U:
 [[-0.3162 -0.9487]
 [-0.9487  0.3162]]
V transposed:
 [[-0.7071 -0.7071]
 [-0.7071  0.7071]]
U @ diag(S) @ Vt:
 [[3. 0.]
 [4. 5.]]
rank-1 piece:
 [[1.5 1.5]
 [4.5 4.5]]
```

NumPy flipped the signs of $$\mathbf{u}_1$$ and $$\mathbf{v}_1$$, which changes nothing because the two minus signs cancel in $$\mathbf{u}_1\mathbf{v}_1^{\top}$$. Singular vectors are only defined up to sign.

### What the singular values tell you

The largest singular value, $$\sigma_1$$, is the most the matrix can stretch any input: no vector of length 1 comes out longer than 6.708. That's the matrix's spectral norm.

The squares of all of them add up to the sum of the squares of all the entries:

$$
\sigma_1^2 + \sigma_2^2 = 45 + 5 = 50 = 3^2 + 0^2 + 4^2 + 5^2
$$

The square root of that total, $$\sqrt{50}$$ here, is the Frobenius norm, written $$\lVert A \rVert_F$$: the length of the matrix if you unroll it into one long vector. So $$\sigma_i^2$$ is the share of the matrix's total "energy" that piece $$i$$ carries. In $$A$$, the first piece carries 45 of 50, or 90%.

## Step 3: the best low-rank copy

### Truncation and Eckart-Young

To get a rank-$$r$$ copy of a matrix, keep the first $$r$$ rank-1 pieces and drop the rest. This is the truncated SVD, written $$A_r$$. For $$A$$ with $$r = 1$$ it's the first grid above, [[1.5, 1.5], [4.5, 4.5]], and the error is the second grid. That error's Frobenius norm is $$\sqrt{1.5^2 + 1.5^2 + 0.5^2 + 0.5^2} = \sqrt{5}$$, exactly the dropped singular value.

Carl Eckart and Gale Young proved in 1936 that this is the best rank-$$r$$ copy there is: no matrix of rank $$r$$ comes closer in Frobenius norm, and Leon Mirsky extended the result in 1960 to every norm that, like these two, doesn't change when you rotate the matrix, the spectral norm included. The errors are:

$$
\lVert A - A_r \rVert_F = \sqrt{\sigma_{r+1}^2 + \sigma_{r+2}^2 + \dots}
$$

$$
\lVert A - A_r \rVert_2 = \sigma_{r+1}
$$

In words: the Frobenius error is the square root of the sum of the dropped singular values squared, and the spectral error is the largest dropped singular value. Dividing the first by $$\lVert A \rVert_F$$ gives the relative error, $$\sqrt{1 - \text{energy kept}}$$: keeping 90% of the energy leaves a relative error of $$\sqrt{0.1} = 32\%$$.

### Checking it on a random matrix

Here is the theorem tested on a random 60 × 40 matrix at rank 10, against 10,000 rival rank-10 matrices and 1,000 nudged versions of the SVD's answer:

```python
import numpy as np
rng = np.random.default_rng(0)

M = rng.standard_normal((60, 40))              # any matrix will do
U, S, Vt = np.linalg.svd(M, full_matrices=False)
r = 10
M_r = U[:, :r] @ np.diag(S[:r]) @ Vt[:r]       # keep the top 10 rank-1 pieces

print("rank of M_r:", np.linalg.matrix_rank(M_r))
print(f"error (Frobenius):  {np.linalg.norm(M - M_r):.4f}")
print(f"sqrt(sum of dropped sigma^2): {np.sqrt(np.sum(S[r:]**2)):.4f}")
print(f"error (spectral):   {np.linalg.norm(M - M_r, 2):.4f}   sigma_11 = {S[r]:.4f}")

# Try to beat it: 10,000 other rank-10 matrices, each the best fit
# that uses a random 10-dimensional set of output directions
best = np.inf
for _ in range(10_000):
    Q, _ = np.linalg.qr(rng.standard_normal((60, r)))   # random orthonormal directions
    best = min(best, np.linalg.norm(M - Q @ (Q.T @ M)))
print(f"best of 10,000 random rank-10 fits: {best:.4f}")
# Nudge the SVD answer: every small change makes it worse
worse = [np.linalg.norm(M - (U[:, :r] + 0.01 * rng.standard_normal((60, r))) @ np.diag(S[:r]) @ Vt[:r]) for _ in range(1000)]
print(f"1,000 nudged versions, smallest error: {min(worse):.4f}")
```

```text
rank of M_r: 10
error (Frobenius):  32.1483
sqrt(sum of dropped sigma^2): 32.1483
error (spectral):   9.4083   sigma_11 = 9.4083
best of 10,000 random rank-10 fits: 43.4723
1,000 nudged versions, smallest error: 32.2538
```

Both error formulas hold to four decimals. The best random rival misses by 43.5, and every nudged version does worse than 32.15, as the theorem predicts.

## Step 4: the SVD of a real weight matrix

### Spectra of Qwen2.5-1.5B

Qwen2.5-1.5B-Instruct has 28 blocks, a width of 1,536, an MLP width of 8,960, and 12 query heads sharing 2 key-value heads of 128 numbers ([config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)). The script below reads two matrices from block 15 (index 14) straight from the checkpoint and compares each with a random matrix of the same shape and spread:

```python
import glob, os
import numpy as np
from safetensors import safe_open

hub = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")) + "/hub"
path = glob.glob(hub + "/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors")[0]
f = safe_open(path, "pt")             # read single tensors, not the whole model
rng = np.random.default_rng(0)

def report(label, M):
    S = np.linalg.svd(M, compute_uv=False)               # singular values, largest first
    E = np.cumsum(S**2) / np.sum(S**2)                    # energy kept by the top r
    k = len(S)
    r90 = np.searchsorted(E, 0.90) + 1
    print(f"{label:9} sigma_1 {S[0]:6.2f}  median {np.median(S):5.2f}  "
          f"top 10%: {E[k // 10 - 1]:.0%}  top 25%: {E[k // 4 - 1]:.0%}  "
          f"90% needs r = {r90} of {k}")

for name in ["self_attn.q_proj", "mlp.up_proj"]:
    W = f.get_tensor(f"model.layers.14.{name}.weight").float().numpy()
    report(name.split(".")[1], W)
    report("random", rng.standard_normal(W.shape) * W.std())   # same shape, same spread
```

```text
q_proj    sigma_1   5.18  median  0.41  top 10%: 68%  top 25%: 89%  90% needs r = 398 of 1536
random    sigma_1   2.17  median  0.88  top 10%: 31%  top 25%: 62%  90% needs r = 783 of 1536
up_proj   sigma_1   5.31  median  2.43  top 10%: 22%  top 25%: 44%  90% needs r = 1158 of 1536
random    sigma_1   3.63  median  2.50  top 10%: 18%  top 25%: 39%  90% needs r = 1221 of 1536
```

The same measurement for every kind of matrix in that block, and for the embedding table:

| Matrix | Shape (out × in) | σ1 | Energy in top 10% | Energy in top 25% | Rank for 90% energy | Same, random matrix |
|---|---|---|---|---|---|---|
| q_proj | 1536 × 1536 | 5.18 | 68% | 89% | 398 (26%) | 783 (51%) |
| k_proj | 256 × 1536 | 2.92 | 46% | 75% | 112 (44%) | 204 (80%) |
| v_proj | 256 × 1536 | 1.57 | 20% | 44% | 187 (73%) | 204 (80%) |
| o_proj | 1536 × 1536 | 3.75 | 53% | 82% | 526 (34%) | 784 (51%) |
| gate_proj | 8960 × 1536 | 10.06 | 27% | 50% | 1,106 (72%) | 1,221 (79%) |
| up_proj | 8960 × 1536 | 5.31 | 22% | 44% | 1,158 (75%) | 1,221 (79%) |
| down_proj | 1536 × 8960 | 11.24 | 28% | 51% | 1,099 (72%) | 1,221 (79%) |
| embed_tokens | 151936 × 1536 | 135.43 | 28% | 45% | 1,232 (80%) | 1,355 (88%) |

The random column is the baseline for no structure at all. Random-matrix theory predicts where its singular values land: for an $$m \times n$$ matrix with $$m \ge n$$ and entries of spread $$s$$, between about $$s(\sqrt{m} - \sqrt{n})$$ and $$s(\sqrt{m} + \sqrt{n})$$, a result known as the Marchenko-Pastur law. For up_proj's shape that's 1.51 to 3.64; the random matrix's actual values ran from 1.52 to 3.62.

The trained matrices read in three groups. The query and output projections have real structure: a quarter of the directions carry 89% and 82% of the energy, against 62% for random. The MLP matrices and the value projection sit close to random: 90% of the energy needs 72% to 75% of all the directions. The embedding table has one giant direction, $$\sigma_1 = 135$$ against 23 for the next, pointing almost exactly (cosine 0.9995) along the average of all 151,936 token vectors, a common offset every token shares. Below it, the table is nearly as spread out as random.

So trained weights are not very low rank. Even the query projection, the most compressible, leaves a relative error of 32% at rank 398.

### Compressing one matrix

What does that do to the model? Here is perplexity, the "effective number of choices per token" from [Part 6](/blog/cross-entropy-perplexity/), on Part 6's 335-token Oz passage, with one matrix in block 15 replaced by truncated versions:

| Change | Perplexity |
|---|---|
| None | 8.46 |
| q_proj at rank 768 (break-even) | 8.47 |
| q_proj at rank 384 (50% storage) | 8.52 |
| q_proj at rank 192 (25% storage) | 8.60 |
| q_proj at rank 96 | 8.65 |
| up_proj at rank 768 (59% storage) | 8.59 |
| up_proj at rank 384 (29% storage) | 8.66 |

One matrix out of 196 barely matters, even with most of it gone. On the same passage's real inputs to that block, the error in the matrix's outputs tracked the weight error: at 25.8% storage, 39% for q_proj and 72% for up_proj.

### Compressing all of them

Now apply the same cut to all 196 attention and MLP matrices at once. To fit in 3 GB of memory, our script reads one block's weights at a time, takes their SVDs and runs the passage through every variant side by side; unmodified, it reproduces the 8.46 exactly. The last row rounds each weight to 4 bits instead, with one scale per group of 128 weights, the round-to-nearest method from [Inference Part 8](/blog/quantization/):

| All 196 matrices | Storage vs original | Perplexity |
|---|---|---|
| Unchanged | 100% | 8.46 |
| Keep 99% of each matrix's energy | 112.9% | 675 |
| Keep 90% of each matrix's energy | 84.6% | about 1.0 million |
| Best rank at 75% storage | 75% | about 324,000 |
| Best rank at 50% storage | 50% | about 1.8 million |
| Best rank at 25.8% storage | 25.8% | about 45 million |
| 4-bit rounding, groups of 128 | 25.8% | 9.99 |

Keeping 99% of the energy needs ranks above break-even, so more storage than the original, and still multiplies perplexity by 80. Every row from 90% down is worse than knowing nothing (perplexity 151,936): the model is confidently wrong.

4-bit rounding at the same 25.8% of the bits costs 1.5 points of perplexity. In block 15 its relative weight error was 12% to 13% per matrix, against 50% to 77% for the best low-rank copy at that storage. The 99%-energy row shows that size of error doesn't settle it: at most 10% error per matrix, less than rounding, and far more damage. Rounding spreads its error thinly over every direction, while truncation deletes whole directions, and some inputs need exactly those. This experiment doesn't isolate which ones, but Frobenius error alone clearly doesn't predict the damage.

## Step 5: why LoRA's update is low rank

### Gradients come in rank-1 pieces

If the weights aren't low rank, why does a low-rank change work? Start with the gradient. [Part 8](/blog/backpropagation-math/) derived the gradient of a matrix multiply: for one token, it's the error signal at the layer's output times the token's input, an outer product, so rank 1. A batch of $$b$$ tokens adds $$b$$ of them, so its rank is at most $$b$$:

```python
import numpy as np
rng = np.random.default_rng(0)
d_in, d_out = 1536, 1536
x = rng.standard_normal((64, d_in))          # 64 tokens going into a layer
delta = rng.standard_normal((64, d_out))     # gradient of the loss at the layer's output
grad_one = np.outer(delta[0], x[0])          # one token: dW = delta x^T (Part 8)
grad_all = delta.T @ x                       # all 64 tokens, summed
print("one token, rank:", np.linalg.matrix_rank(grad_one))
print("64 tokens, rank:", np.linalg.matrix_rank(grad_all), "of a possible", min(d_in, d_out))
```

```text
one token, rank: 1
64 tokens, rank: 64 of a possible 1536
```

Pretraining adds up trillions of these, and the weights end up using most directions, as Step 4 found. Fine-tuning on a few thousand examples of one task only nudges the model toward one behavior, and there's no reason that nudge should need all 4,096 directions.

### LoRA writes the change as two thin matrices

LoRA, from [Hu et al. (2021)](https://arxiv.org/abs/2106.09685), freezes the pretrained matrix $$W_0$$ and learns the change as a product, exactly the thin pair from Step 1:

$$
W = W_0 + \frac{\alpha}{r} B A
$$

Here $$B$$ is $$d \times r$$, $$A$$ is $$r \times k$$, and the rank $$r$$ is much smaller than either size. The paper uses "a random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training": the model starts out exactly as pretrained. The factor $$\alpha / r$$, where $$\alpha$$ (alpha) is a constant you set, keeps the size of the update roughly steady when you try a different $$r$$, so the learning rate doesn't need retuning.

For Llama 3.1 8B's 4,096 × 4,096 query projection:

| Rank r | Trainable numbers, r × (4,096 + 4,096) | Share of the matrix |
|---|---|---|
| 8 | 65,536 | 0.39% |
| 16 | 131,072 | 0.78% |
| 4,096 (full) | 16,777,216 | 100% |

After training, $$BA$$ is multiplied out once and added into $$W_0$$, so serving costs nothing extra ("no additional inference latency", in the paper's words). [Fine-Tuning a Small Model with LoRA on 5,000 Examples](/blog/lora-fine-tuning-5k-examples/) covers the practical side: which matrices to adapt, ranks of 16 to 32, and the memory budget.

### The evidence that the change is low rank

The LoRA paper builds on [Aghajanyan, Zettlemoyer and Gupta (2020)](https://arxiv.org/abs/2012.13255), who showed that fine-tuning has a low "intrinsic dimension": "by optimizing only 200 trainable parameters randomly projected back into the full space", RoBERTa reached 90% of full fine-tuning's performance on MRPC, a paraphrase-detection task. Hu et al. "hypothesize that the change in weights during model adaptation also has a low 'intrinsic rank'" and back it with three findings on GPT-3 175B:

- A rank as small as 1 was enough to adapt the query and value projections together on WikiSQL and MultiNLI (their Table 6).
- Adapters trained at rank 8 and rank 64 agree on their top singular direction and little else, and the authors suggest the other directions may hold mostly noise.
- The change "only amplifies directions that are not emphasized in W", and by a lot: an amplification factor of 21.5 at rank 4 in the 48th layer.

The paper also warns that for a language unlike the pretraining data, training every weight "could certainly outperform LoRA with a small r."

## Step 6: low rank by design

### Attention scores pass through 128 numbers

Multiply one attention head's query and key slices together and you get a 1,536 × 1,536 matrix that scores one token against another. Every score passes through 128 numbers, so by Part 3's rule for rank through a chain, its rank is at most 128; for block 15's first head, NumPy's `matrix_rank` returns exactly 128. [Part 11](/blog/attention-math/) used this to describe each head as a pair of low-rank circuits.

### MLA compresses keys and values through 512 numbers

DeepSeek's multi-head latent attention (MLA), introduced in [DeepSeek-V2](https://arxiv.org/abs/2405.04434), makes the factorization explicit. Each token's hidden vector $$\mathbf{h}$$, 5,120 numbers, is first squeezed into a latent of $$d_c = 512$$ numbers, and keys and values for all 128 heads of 128 numbers are rebuilt from it:

$$
\mathbf{c} = W^{DKV} \mathbf{h}, \quad \mathbf{k} = W^{UK} \mathbf{c}, \quad \mathbf{v} = W^{UV} \mathbf{c}
$$

Here $$W^{DKV}$$ is the 512 × 5,120 down-projection and $$W^{UK}$$, $$W^{UV}$$ are 16,384 × 512 up-projections, what the paper calls "low-rank joint compression". The combined key projection $$W^{UK}W^{DKV}$$ is a 16,384 × 5,120 matrix of rank at most 512. In weights, the three thin matrices hold 19,398,656 numbers where full key and value projections would hold 167,772,160, 11.6% as many.

The bigger saving is in the cache: only $$\mathbf{c}$$ is stored, plus a 64-number key for position. That's 576 numbers per token per layer against 32,768 for plain multi-head attention, 1.76%. The compressed keys carry no position rotation (the small key handles position), so the matrix products can be regrouped (Part 3): $$W^{UK}$$ can be folded into the query projection and $$W^{UV}$$ into the output projection, so full keys and values never need to be built. [The KV cache](/blog/kv-cache/) works through what that means in gigabytes and users per GPU.

### Muon and GaLore use the SVD during training

Muon, from Part 9, replaces each momentum matrix $$U\Sigma V^{\top}$$ by $$UV^{\top}$$: it sets every singular value to 1, so rarely used directions get as big a step as dominant ones. GaLore ([Zhao et al., 2024](https://arxiv.org/abs/2403.03507)) goes the other way and keeps only the top directions. Every 200 steps it takes the SVD of a weight's gradient, keeps the top $$r$$ left singular vectors, and runs Adam in that small space. Unlike LoRA it updates every weight; the paper reports "up to 65.5%" less memory for optimizer states and pre-training a 7B model on a 24 GB consumer GPU.

## In the real world

- PyTorch's [`torch.linalg.svd`](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.svd.html) defaults to `full_matrices=True`, which builds the full square $$U$$; pass `full_matrices=False` for the thin version. The work grows roughly like $$mn \cdot \min(m, n)$$. On our laptop CPU (AMD Ryzen 9 8945HS, 2 threads, PyTorch 2.11, a shared machine), an 8,960 × 1,536 matrix took 1.2 s and the same shape on its side, 1,536 × 8,960, took 4.8 s; transposing first is a cheap speedup.
- [`torch.svd_lowrank`](https://docs.pytorch.org/docs/2.14/generated/torch.svd_lowrank.html) estimates only the top singular values with a randomized method (Halko, Martinsson and Tropp, 2009); with `q=64` it took 0.08 s on the same matrix. Its docs still recommend `torch.linalg.svd` for dense matrices and keep the randomized one for very large sparse ones.
- Hugging Face PEFT's [`LoraConfig`](https://huggingface.co/docs/peft/package_reference/lora) defaults to `r=8`, `lora_alpha=8` and `lora_dropout=0.0`. It initializes A with Kaiming-uniform random values and B with zeros, so a fresh adapter "will be a no-op", and `use_rslora=True` swaps the $$\alpha/r$$ scaling for $$\alpha/\sqrt{r}$$.
- SVD compression of LLMs works better when it weighs directions by the activations that flow through them. [ASVD](https://arxiv.org/abs/2312.05821) reports compressing a network by 10% to 30% without training, and [SVD-LLM](https://arxiv.org/abs/2403.07378) whitens the inputs so that each dropped singular value maps directly to the loss it causes. Removing 10% to 30% is far short of the roughly 75% that 4-bit numbers remove, which is why [quantization](/blog/quantization/) is the default way to shrink a model.

## Common misconceptions

- **"LoRA works because LLM weights are low rank."** Qwen2.5-1.5B's MLP matrices need 72% to 75% of their directions to hold 90% of their energy, close to a random matrix. What's low rank is the change fine-tuning makes, which is what LoRA learns.
- **"Keeping half the singular values halves the storage."** Storing the thin pair costs $$r(m+n)$$ numbers, so a square matrix breaks even at half its size. Keeping 768 of q_proj's 1,536 singular values saves nothing.
- **"Singular values are the same as eigenvalues."** Only for symmetric matrices with no negative eigenvalues. The eigenvalues of $$A$$, the stretch factors along directions it doesn't turn, are 3 and 5; its singular values are 6.71 and 2.24.

## Key takeaways

- A rank-1 matrix is a column times a row, and it stores m plus n numbers instead of m times n. Any rank-r matrix is r such pieces, a product of two thin matrices.
- The SVD writes every matrix as rotate, stretch, rotate. The stretches are the singular values, sorted largest first; their squares add up to the matrix's energy.
- Eckart and Young proved that keeping the top r pieces gives the best rank-r copy. The error is the square root of the sum of the dropped singular values squared.
- Trained weights are not low rank. Cutting every Qwen matrix to half its storage sends perplexity from 8.46 to about 1.8 million, while 4-bit rounding gives 9.99.
- The changes are low rank. Gradients arrive as rank-1 pieces, LoRA learns a fine-tuning update with well under 1% of the numbers, and MLA routes keys and values through a 512-number latent by design.

## What's next

Step 4 ended with a contest that rounding won: 4-bit numbers kept Qwen almost intact at a quarter of the storage. [Part 14](/blog/floating-point-math/), Floating point and quantization math, explains why. It takes apart how computers store decimals (sign, exponent and mantissa), compares FP32, BF16, FP8 and FP4, and works out the error that scales, zero points and group sizes leave behind.

## Further reading

- [The Approximation of One Matrix by Another of Lower Rank (Eckart & Young, 1936)](https://doi.org/10.1007/BF02288367): the original proof that the truncated SVD is the best low-rank approximation.
- [Symmetric Gauge Functions and Unitarily Invariant Norms (Mirsky, 1960)](https://doi.org/10.1093/qmath/11.1.50): extends Eckart and Young's result to the spectral norm and every other rotation-invariant norm.
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685): the method, its initialization and scaling, and section 7's analysis of the update's rank.
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning (Aghajanyan et al., 2020)](https://arxiv.org/abs/2012.13255): fine-tuning in a few hundred random directions.
- [DeepSeek-V2 (DeepSeek-AI, 2024)](https://arxiv.org/abs/2405.04434): section 2.1 defines multi-head latent attention as low-rank joint compression of keys and values.
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection (Zhao et al., 2024)](https://arxiv.org/abs/2403.03507): SVD of gradients to shrink optimizer memory while training every weight.
- [ASVD (Yuan et al., 2023)](https://arxiv.org/abs/2312.05821) and [SVD-LLM (Wang et al., 2024)](https://arxiv.org/abs/2403.07378): activation-aware SVD compression of LLM weights.
- [Finding Structure with Randomness (Halko, Martinsson & Tropp, 2009)](https://arxiv.org/abs/0909.4061): the randomized SVD behind `torch.svd_lowrank`.
- [torch.linalg.svd (PyTorch docs)](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.svd.html) and [LoraConfig (Hugging Face PEFT docs)](https://huggingface.co/docs/peft/package_reference/lora): the APIs used in practice, with their defaults.
- [Mathematics for Machine Learning (Deisenroth, Faisal & Ong, 2020)](https://mml-book.github.io/): chapter 4, Matrix Decompositions, includes the SVD and low-rank matrix approximation.
- [Essence of linear algebra (3Blue1Brown)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab): animated linear maps, the picture behind rotate, stretch, rotate.
