---
title: "Matrices as Transformations, and Why Neural Networks Need a Bend"
description: "What a matrix does to space, why stacked linear layers collapse into one, how ReLU and SiLU fix that, and how batching changes a matmul's cost."
slug: "matrices-linear-maps"
category: "llm-fundamentals"
tags: ["math", "neural-networks", "transformers", "gpu"]
series: "llm-math"
seriesOrder: 3
pubDate: 2026-10-23
---

## TL;DR

A matrix moves every point in a space at once, and its columns say where it sends the basic directions. Multiplying two matrices chains two moves into one, so order matters and two linear layers in a row are really a single layer. Rank counts how many independent directions survive the move. Activation functions like ReLU and SiLU bend space so that depth stops collapsing, and the Transformer's MLP uses that to expand, bend and compress every token. Stacking tokens into a batch turns many thin multiplications into one fat one, which raises the arithmetic done per byte of weights read from memory.

## Why it matters

Take GPT-2's MLP, the small two-layer network inside every block, and delete its activation function. Its 4,718,592 weights now do exactly the same job as a single grid of 589,824 numbers, eight times fewer. Stack a hundred such layers and they still fold into one grid.

The fold is exact. It follows from what a matrix is, and it's why every neural network puts a small curved function between its matrix multiplications. Those curved functions cost less than 0.1% of a model's arithmetic, as [Part 1](/blog/math-for-llms/) counted, yet without them a stack of linear layers, however deep, can do nothing a single layer can't.

Part 1 treated matrix multiplication as bookkeeping: rows dotted with columns, shapes that must line up, and 2mkn FLOPs (floating-point operations). This part looks at the same operation as geometry: what a matrix does to the space of vectors, and what that says about depth, width and batching.

## The intuition

Draw a grid on a sheet of rubber, pin one corner to the table, and stretch, twist or slant the sheet. However you pull it, two things stay true: grid lines stay straight and evenly spaced, and the pinned corner doesn't move. Every such reshaping is a matrix, and the pinned corner is the zero vector.

Two reshapings in a row are still one reshaping of the same kind, which is why linear layers collapse. To get anything else you have to fold or crease the sheet, and that's what an activation function does.

The analogy breaks in two ways. A rubber sheet is two-dimensional, but a model's matrices map 4,096 dimensions to 14,336 and back, which nobody can picture. And a real sheet can't be squashed flat onto a line without tearing, while a matrix can do exactly that, losing a whole direction for good.

## Step 1: what a matrix does to the plane

### Four moves with numbers

Start in two dimensions, where you can draw. A point in the plane is a vector of two numbers, such as $$\mathbf{x} = [1, 2]$$: one step right, two steps up. In this part a vector is a column and the matrix acts on it from the left, $$A\mathbf{x}$$, each output entry being a row of $$A$$ dotted with $$\mathbf{x}$$.

Here are four 2 × 2 matrices and what each does to that point:

| Move | Matrix (rows) | Where [1, 0] lands | Where [0, 1] lands | Where [1, 2] lands |
|---|---|---|---|---|
| Stretch sideways | [2, 0], [0, 1] | [2, 0] | [0, 1] | [2, 2] |
| Rotate 90° | [0, -1], [1, 0] | [0, 1] | [-1, 0] | [-2, 1] |
| Shear | [1, 1], [0, 1] | [1, 0] | [1, 1] | [3, 2] |
| Project onto the x-axis | [1, 0], [0, 0] | [1, 0] | [0, 0] | [1, 0] |

The stretch doubles horizontal distances. The rotation turns the plane a quarter turn counterclockwise. The shear slides each point right by its height, turning squares into slanted parallelograms. The projection flattens the plane onto the horizontal axis, like a shadow cast straight down.

### The columns are where the basis vectors land

Compare the middle columns of the table with the matrices: [1, 0] always lands on the matrix's first column, and [0, 1] on its second. Every point is a mix of those two basic directions, called the basis vectors: [1, 2] is 1 step of [1, 0] plus 2 steps of [0, 1]. A matrix keeps that recipe and swaps in the new directions:

$$
A\mathbf{x} = x_1 \, \mathbf{a}_1 + x_2 \, \mathbf{a}_2
$$

Here $$\mathbf{a}_1$$ and $$\mathbf{a}_2$$ are the first and second columns of $$A$$, and $$x_1, x_2$$ are the entries of $$\mathbf{x}$$. In words: the output is a weighted sum of the matrix's columns, with the input's numbers as the weights. For the shear, `1 × [1, 0] + 2 × [1, 1] = [3, 2]`, the same answer the rows-times-vector rule gives. So you can read a matrix without multiplying anything: its columns say where each axis goes, and everything else follows.

### What "linear" means

A map that works this way is called a linear map. It obeys two rules: mapping a sum gives the sum of the mapped parts, and scaling the input scales the output by the same number:

$$
A(\mathbf{x} + \mathbf{y}) = A\mathbf{x} + A\mathbf{y}, \qquad A(c\,\mathbf{x}) = c\,A\mathbf{x}
$$

Here $$c$$ is any ordinary number. Check the first rule with the shear: [1, 2] + [3, -1] = [4, 1] maps to [5, 1], and separately [1, 2] maps to [3, 2] and [3, -1] to [2, -1], which also add up to [5, 1]. These two rules are the rubber sheet's straight, evenly spaced grid lines, written as algebra.

A neural network layer usually adds a bias too, $$W\mathbf{x} + \mathbf{b}$$, which shifts everything by a fixed vector $$\mathbf{b}$$. Strictly, that's an affine map, a linear map followed by a slide. It unpins the corner of the sheet but keeps the lines straight, so nothing below changes.

### The same thing at model scale

None of this needs two dimensions. The matrix behind a layer that turns 4,096 numbers into 14,336, like the up projection in Llama 3.1 8B's MLP, has 4,096 columns of 14,336 numbers each. Each column is where one of the 4,096 input directions lands, and every token's output is a weighted mix of those columns.

Code usually stores each token as a row and writes `x @ W`, which swaps the roles of rows and columns: the same arithmetic with the matrix flipped over its diagonal (transposed), as Part 1 noted for PyTorch's `nn.Linear`.

## Step 2: multiplying matrices means doing one move, then another

### Composition, worked by hand

Rotate the plane by 90° with $$R$$, then stretch it sideways with $$S$$. Follow the point [1, 0]. The rotation sends it to [0, 1], and the stretch leaves [0, 1] alone, because it only doubles horizontal distances. Final answer: [0, 1].

In the other order, the stretch sends [1, 0] to [2, 0], and the rotation turns that into [0, 2]. Doing $$R$$ then $$S$$ is written $$SR$$, with the first move on the right, nearest the vector, because $$S(R\mathbf{x})$$ applies $$R$$ first. Multiply the matrices with Part 1's row-times-column rule and you get one matrix for each order:

$$
SR = \begin{bmatrix} 0 & -2 \\ 1 & 0 \end{bmatrix}, \qquad
RS = \begin{bmatrix} 0 & -1 \\ 2 & 0 \end{bmatrix}
$$

The first column of $$SR$$ is [0, 1] and the first column of $$RS$$ is [0, 2], exactly where each order sent the point [1, 0]. So the matrix product is the combined move, and its columns are where the basis vectors end up after both steps:

$$
(BA)\mathbf{x} = B(A\mathbf{x})
$$

In words: multiplying by $$A$$ and then by $$B$$ is the same as multiplying once by the single matrix $$BA$$. Since $$SR \neq RS$$, matrix multiplication is not commutative, unlike ordinary multiplication, where 2 × 3 = 3 × 2. Part 1 showed this with shapes for non-square matrices; here two square matrices of the same size disagree.

Grouping doesn't matter: $$(CB)A = C(BA)$$, so a chain of matrices can be multiplied in whatever grouping is cheapest, as long as the left-to-right order stays put.

### Doing nothing, and undoing

The identity matrix $$I$$ has 1s on its diagonal and 0s elsewhere. Its columns are the basis vectors themselves, so it leaves every point where it is: $$I\mathbf{x} = \mathbf{x}$$.

The inverse of a matrix, written $$A^{-1}$$, undoes it: $$A^{-1}A = I$$. The sideways stretch [2, 0], [0, 1] is undone by a sideways halving, [0.5, 0], [0, 1]. The 90° rotation is undone by rotating back, [0, 1], [-1, 0]. The shear [1, 1], [0, 1] is undone by shearing the other way, [1, -1], [0, 1].

The projection has no inverse. It sent both [0, 1] and [0, 0] to the same place, [0, 0], so no matrix can tell them apart afterwards. The next step measures that kind of loss.

## Step 3: rank, the directions that survive

### A rank-1 example

Here is a matrix:

$$
A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
$$

Its first column is [1, 2] and its second is [2, 4], which is just twice the first. So every output, a weighted mix of the columns, is some multiple of [1, 2]. Feed it a few points:

| Input | Output |
|---|---|
| [1, 0] | [1, 2] |
| [0, 1] | [2, 4] |
| [3, -1] | [1, 2] |
| [2, -1] | [0, 0] |

The whole plane lands on the line through [1, 2]. Two different inputs, [1, 0] and [3, -1], give the same output, and [2, -1] is wiped out. The rank of a matrix is the number of independent directions its outputs can point in. This one has rank 1: two dimensions went in, one survived. The projection from Step 1 also has rank 1. The stretch, rotation and shear have rank 2, which is why they could be undone.

A matrix with $$m$$ rows and $$n$$ columns has rank at most the smaller of $$m$$ and $$n$$. When it reaches that limit it's called full rank. A square matrix can be inverted exactly when it's full rank.

### Rank through a chain

A chain of two maps can't keep more directions than its narrowest step:

$$
\operatorname{rank}(BA) \le \min\big(\operatorname{rank}(A),\ \operatorname{rank}(B)\big)
$$

In words, a combined move keeps at most as many directions as the weaker of its two parts. Squeeze 768 numbers through a layer 64 numbers wide and back out to 768, and whatever comes out lies in a 64-dimensional slice of the 768-dimensional space, however big the matrices around the bottleneck are.

This inequality comes back later in the series. Each attention head in Llama 3.1 8B compares tokens through 128-number queries and keys, so the matrix that scores one token against another, a 4,096 × 4,096 grid if you multiply it out, has rank at most 128. LoRA, a popular fine-tuning method, builds its update to a weight matrix from two thin matrices, so the update has small rank by design. Part 11 and Part 13 work through both.

## Step 4: why depth needs a bend

### Two linear layers are one layer

Put the last two steps together. A layer computes $$W_1\mathbf{x}$$, and the next one computes $$W_2$$ times that. By the composition rule:

$$
W_2(W_1\mathbf{x}) = (W_2W_1)\,\mathbf{x}
$$

The two layers are exactly one layer whose matrix is $$W_2W_1$$. Biases don't rescue it either: the second layer's output is $$W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$, which tidies up to one matrix $$W_2W_1$$ and one bias $$W_2\mathbf{b}_1 + \mathbf{b}_2$$. A hundred linear layers fold into one the same way, just as [Fundamentals Part 3](/blog/neural-networks-basics/) showed with single numbers: multiply by 3, then by 2, and you've multiplied by 6.

With GPT-2 small's MLP sizes, 768 numbers widened to 3,072 and brought back to 768, the fold is easy to check:

```python
import numpy as np
rng = np.random.default_rng(0)

# Two linear layers with GPT-2 small's MLP sizes: 768 -> 3,072 -> 768
W1 = rng.standard_normal((3072, 768))    # layer 1: 768 numbers in, 3,072 out
W2 = rng.standard_normal((768, 3072))    # layer 2: 3,072 in, 768 out
x = rng.standard_normal(768)             # one token's vector

two_layers = W2 @ (W1 @ x)               # run the layers one after the other
W = W2 @ W1                              # fold them into one matrix
print(W.shape, np.allclose(two_layers, W @ x))
print("rank:", np.linalg.matrix_rank(W))

# A narrow middle: 768 -> 64 -> 768
N1 = rng.standard_normal((64, 768))
N2 = rng.standard_normal((768, 64))
print("rank through a 64-wide middle:", np.linalg.matrix_rank(N2 @ N1))
```

```text
(768, 768) True
rank: 768
rank through a 64-wide middle: 64
```

The two layers hold `2 × 768 × 3,072 = 4,718,592` weights; the folded matrix holds `768 × 768 = 589,824`, eight times fewer, and gives the same outputs. The 3,072-wide middle adds nothing, because the result can't have rank above 768.

Folding also saves arithmetic. Running the two layers costs `2 × 2 × 768 × 3,072 ≈ 9.4 million` FLOPs per token, against `2 × 768 × 768 ≈ 1.2 million` for the folded matrix. Computing $$W_2W_1$$ once costs `2 × 768 × 3,072 × 768 ≈ 3.6 billion` FLOPs, which pays for itself after 439 tokens.

![Three panels stacked vertically. Top, two linear layers with no bend: x, 768 numbers, goes through W1 up to 3,072, then W2 down to 768, giving y. An arrow labeled same outputs, 8 times fewer weights leads to the middle panel, folded into one matrix: x goes through a single matrix W equals W2 times W1, 768 by 768, giving y. Bottom, with a bend between, no fold exists: x goes through W1 up to 3,072, then ReLU or GELU, then W2 down to 768, giving y.](/diagrams/m03-linear-collapse.svg "Without an activation function, two layers fold into one matrix. With one, they don't.")

What keeps a deep network from collapsing into one matrix is the small function between its layers.

### XOR: a problem no straight line can solve

The smallest problem a linear model can't solve shows what the bend buys. XOR (exclusive or) takes two bits and outputs 1 when exactly one of them is 1:

| Input x1 | Input x2 | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Suppose some linear map plus a bias, $$f(\mathbf{x}) = w_1x_1 + w_2x_2 + b$$, could separate these: high on the two 1-points and low on the two 0-points, with some threshold $$t$$ in between. Add up its values at the corners in pairs:

$$
\begin{aligned}
f(0,0) + f(1,1) &= 2b + w_1 + w_2 \\
f(0,1) + f(1,0) &= 2b + w_1 + w_2
\end{aligned}
$$

The two sums are always equal. But the first pair are both below $$t$$, so their sum is below $$2t$$, and the second pair are both above $$t$$, so their sum is above $$2t$$. Equal sums can't sit on opposite sides of $$2t$$, so no straight line separates XOR, whatever the weights.

The fix, worked in the [Deep Learning textbook](https://www.deeplearningbook.org/contents/mlp.html) by Goodfellow, Bengio and Courville, is one hidden layer of two ReLU units, where ReLU (rectified linear unit) keeps positive numbers and turns negative ones into 0. Both hidden units compute $$x_1 + x_2$$, the second shifted down by 1, and the output is the first unit minus twice the second:

```python
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # the four inputs
y = np.array([0, 1, 1, 0])                         # XOR: 1 when exactly one input is 1

# Best straight-line fit: y ≈ X @ w + b, by least squares
A = np.hstack([X, np.ones((4, 1))])
w_lin = np.linalg.lstsq(A, y, rcond=None)[0]
print("linear fit:", A @ w_lin)

# One hidden layer of 2 ReLU units, then a linear readout
W = np.array([[1, 1], [1, 1]])    # both hidden units compute x1 + x2
c = np.array([0, -1])             # ...the second one shifted down by 1
v = np.array([1, -2])             # readout: h1 - 2*h2

h = np.maximum(0, X @ W + c)      # ReLU bends the space
print("hidden points:", h.tolist())
print("output:", h @ v)

# Same network with the ReLU removed: collapses to one linear map
print("no ReLU:", (X @ W + c) @ v)
```

```text
linear fit: [0.5 0.5 0.5 0.5]
hidden points: [[0, 0], [1, 0], [1, 0], [2, 1]]
output: [0 1 1 0]
no ReLU: [2 1 1 0]
```

The best straight-line fit outputs 0.5 everywhere, as in the textbook. The ReLU network gets all four right, and the hidden points show why. Before the ReLU, the second unit's values are -1, 0, 0 and 1. The ReLU flattens the -1 to 0, which creases the plane: the point [0, 0] stays put, both [0, 1] and [1, 0] land on [1, 0], and [1, 1] moves to [2, 1]. In the textbook's words, "the two points that must have output 1 have been collapsed into a single point in feature space." In the new space a straight line does separate them, and the output layer is that line.

Remove the ReLU and the output becomes `2 - x1 - x2`, a tilted plane that gives [2, 1, 1, 0].

### ReLU, GELU and SiLU

A ReLU network is made of straight pieces, bending space along creases. Modern LLMs use smooth bends instead. GELU (Gaussian error linear unit), used by GPT-2 and covered in Fundamentals Part 3, and SiLU (sigmoid linear unit), used by Llama and Qwen, both look like a ReLU with the corner rounded off:

$$
\text{SiLU}(z) = z \cdot \sigma(z), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Here $$z$$ is the number going in, and $$\sigma$$ is the sigmoid, which squeezes any number into the range 0 to 1, so SiLU multiplies $$z$$ by a soft on-off switch. The name comes from [Elfwing, Uchibe and Doya (2017)](https://arxiv.org/abs/1702.03118); the same function is also called Swish. Some computed values:

| Input z | -3 | -1 | -0.5 | 0 | 0.5 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| ReLU | 0 | 0 | 0 | 0 | 0.5 | 1 | 3 |
| GELU | -0.004 | -0.159 | -0.154 | 0 | 0.346 | 0.841 | 2.996 |
| SiLU | -0.142 | -0.269 | -0.189 | 0 | 0.311 | 0.731 | 2.858 |

All three pass large positive numbers almost unchanged and push negative ones toward 0. SiLU never goes below -0.278, which it reaches at z ≈ -1.28. For this part's argument the exact shape matters little: any curve breaks the fold, and the network's matrices can no longer be multiplied into one.

## Step 5: the MLP expands, bends and compresses

Every Transformer block contains an MLP (multi-layer perceptron) that handles each token on its own, as [Fundamentals Part 6](/blog/transformer-block/) explains. In this part's terms it does three things.

First, it expands. Llama 3.1 8B's MLP maps each token's 4,096 numbers up to 14,336 ([config](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json): `hidden_size` 4096, `intermediate_size` 14336, `hidden_act` "silu"). As a linear map, that adds no information: the 14,336 outputs are mixes of only 4,096 inputs, so they span at most 4,096 directions. Second, it bends: each of the 14,336 numbers goes through its own curve. Now the width matters: each hidden number can switch on or off on its own, and 14,336 switches can carve up the input space far more finely than 4,096. Third, it compresses: a down projection maps the 14,336 bent numbers back to 4,096, a weighted mix of 14,336 learned columns.

Llama's MLP uses SwiGLU, worked through in [Fundamentals Part 10](/blog/modern-llm-architecture/): it computes two up projections, passes one through SiLU, and multiplies them number by number before the down projection. That product is a second bend. Without SiLU the MLP would still not be linear, because doubling the input would quadruple the product.

The counts, from the config:

```text
gate and up:  2 × 4,096 × 14,336     = 117,440,512 weights
down:         14,336 × 4,096          =  58,720,256 weights
whole MLP:    3 × 4,096 × 14,336     = 176,160,768 weights per block
```

That's 81% of each block's parameters, and the bends are why they aren't wasted. A plain two-matrix MLP of the same width with no bend would fold into a single 4,096 × 4,096 matrix: 117 million weights doing the work of 16.8 million, seven times fewer.

## Step 6: batching, FLOPs and bytes

### A batch is the same math, stacked

When a model processes many tokens at once, it stacks their vectors as the rows of a matrix $$X$$, one row per token, and computes $$XW$$ instead of $$\mathbf{x}W$$ for each token separately. Row $$i$$ of $$XW$$ is row $$i$$ of $$X$$ times $$W$$, so the answers are the same:

```python
import numpy as np
rng = np.random.default_rng(0)

W = rng.standard_normal((4096, 1024))   # a weight matrix: 4,096 numbers in, 1,024 out
X = rng.standard_normal((8, 4096))      # 8 tokens, one per row

batched = X @ W                                   # one matrix-matrix multiply
one_at_a_time = np.stack([x @ W for x in X])     # 8 vector-matrix multiplies
print(batched.shape, np.allclose(batched, one_at_a_time))

def flops_per_byte(m, k, n, bytes_per_number=2):
    flops = 2 * m * k * n                                    # the 2mkn count
    moved = bytes_per_number * (m * k + k * n + m * n)       # read X and W, write the result
    return flops / moved

for m in [1, 8, 64, 512, 4096]:                  # tokens through Llama's up projection
    print(f"{m:5} tokens: {flops_per_byte(m, 4096, 14336):7.1f} FLOPs per byte")
```

```text
(8, 1024) True
    1 tokens:     1.0 FLOPs per byte
    8 tokens:     8.0 FLOPs per byte
   64 tokens:    62.7 FLOPs per byte
  512 tokens:   441.1 FLOPs per byte
 4096 tokens:  1792.0 FLOPs per byte
```

### Counting bytes as well as FLOPs

A GPU must fetch the weights from memory before it can multiply by them. Take one token through Llama 3.1 8B's up projection, a (1 × 4,096) row times a (4,096 × 14,336) matrix. With $$m = 1$$, $$k = 4{,}096$$ and $$n = 14{,}336$$, the 2mkn count is 117,440,512 FLOPs. In BF16, a 16-bit number format, each weight takes 2 bytes, so the weights alone are 117,440,512 bytes: one FLOP per byte read.

Now stack $$m$$ tokens. The weights are read once and used for every row, so the FLOPs grow with $$m$$ while the bytes barely move. With 2 bytes per number:

$$
\text{FLOPs per byte} = \frac{2mkn}{2(mk + kn + mn)} = \frac{mkn}{mk + kn + mn}
$$

The top is the arithmetic; the bottom is the bytes: reading the input $$X$$ ($$mk$$ numbers), reading the weights $$W$$ ($$kn$$), and writing the output ($$mn$$). When $$m$$ is much smaller than $$k$$ and $$n$$, the $$kn$$ term dominates the bottom and the ratio is almost exactly $$m$$. For a weight matrix in BF16, FLOPs per byte is roughly the number of tokens in the batch: 8 tokens give 8.0, 64 give 62.7.

That ratio is called arithmetic intensity, covered in [Inference Part 1](/blog/how-llm-inference-works/). A GPU's ridge point is its peak FLOPs per second divided by its memory bandwidth in bytes per second. Below the ridge, the GPU waits on memory (memory-bound); above it, arithmetic is the limit (compute-bound). For an NVIDIA H100 SXM, the [spec sheet](https://www.nvidia.com/en-us/data-center/h100/) lists 3.35 TB/s and 1,979 BF16 teraFLOPS with sparsity, which is 989 without it, so the ridge is about 295 FLOPs per byte. With the formula above, Llama's up projection crosses that at 326 tokens; counting weights only, as Inference Part 1 does, it's about 295.

Generating text for one user is the $$m = 1$$ row: 1 FLOP per byte, about a three-hundredth of what the H100 could use, and batching users into one step is how servers climb toward the ridge. [GPU Part 1](/blog/gpu-from-python-to-silicon/) measured the same effect on a laptop GPU, whose ridge sits at about 114 FLOPs per byte; for this matrix the formula crosses it at 119 tokens.

## In the real world

- Real MLPs are far from linear. We fed the MLP in the 15th of Qwen2.5-1.5B-Instruct's 28 layers the real inputs of a 67-token passage from *Alice's Adventures in Wonderland* and tested the first linearity rule on 65 pairs of neighboring tokens. A linear map would give MLP(a + b) = MLP(a) + MLP(b) exactly; the two sides differed by 41% to 72% of the output's size, median 53%. The same test on the up and down projections alone, with the gate removed, agreed to within one part in a million, which is rounding.
- Most gates sit on the flat side of the bend. In the same run, averaged over the passage and all 28 layers, 78% of the 8,960 gate values in each MLP were negative (from 62% to 99.7% depending on the layer), so SiLU squashed them into the narrow range from -0.278 to 0. Qwen2.5-1.5B's [config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) lists 1,536 numbers per token, an MLP width of 8,960, 28 layers and `hidden_act` "silu".
- LoRA writes a fine-tuning update as the product of two thin matrices, so its rank is capped by their inner width, a small number the user picks. Its authors hypothesize that the change in weights during adaptation "has a low 'intrinsic rank'" and report cutting GPT-3's trainable parameters "by 10,000 times" ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)). Because the update is a product, it can be multiplied out once and added to the base weights before serving.
- Batching is how servers keep GPUs busy. Inference Part 1's table for Llama 3.1 8B on one H100 shows 1 user getting 209 tokens per second and 64 users sharing the same 4.79 ms step for 13,350 tokens per second in total, because each weight read now feeds 64 rows of the matmul.

## Common misconceptions

- **"Matrix multiplication works like ordinary multiplication, so order doesn't matter."** Rotating then stretching gave [0, 1] for the point [1, 0]; stretching then rotating gave [0, 2]. $$AB$$ and $$BA$$ are usually different matrices, and often only one of them exists.
- **"More layers always means more power."** Only with a bend between them. Without activation functions, any number of linear layers fold into one matrix, and a deep network can only draw straight lines.
- **"A wider hidden layer adds capacity on its own."** A linear map from 768 to 3,072 and back has rank at most 768. Width pays off only after the bend, where each hidden unit can switch on and off independently.
- **"Activation functions are a rounding error, since they cost under 0.1% of the FLOPs."** Cheap as they are, without them a stack of MLP layers of any depth folds into a single matrix.
- **"Batching changes what each user's request computes."** Row $$i$$ of $$XW$$ depends only on row $$i$$ of $$X$$. Batching changes how many FLOPs you get per byte of weights read; each token's math stays the same.

## Key takeaways

- A matrix moves every point of a space at once. Its columns are where the basic directions land, and every output is a weighted mix of those columns.
- Multiplying two matrices makes one matrix that does both moves. Order matters, and two linear layers are always exactly one linear layer.
- Rank counts the directions that survive. A chain can never keep more directions than its narrowest step.
- Activation functions like ReLU, GELU and SiLU bend space so the layers can't fold together. One hidden ReLU layer solves XOR, which no straight line can.
- Stacking tokens into a batch keeps the math the same, but in BF16 a weight matrix's FLOPs per byte rises to roughly the number of tokens, which is why GPUs serve many users at once.

## What's next

So far the model has only moved vectors around. At the end it turns one vector into 128,256 scores, one per token in its vocabulary, and those scores must become a choice of what to say next. [Part 4](/blog/probability-language-models/) builds the probability behind that: conditional probabilities, why the chance of a whole sentence is a product of next-token probabilities, and how temperature, top-k and top-p reshape the odds before the model picks a word.

## Further reading

- [Deep Learning, Chapter 6: Deep Feedforward Networks (Goodfellow, Bengio & Courville, 2016)](https://www.deeplearningbook.org/contents/mlp.html): section 6.1 solves XOR with the exact two-unit ReLU network used here, and shows why a linear model can't.
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685): rank in practice, as fine-tuning updates built from two thin matrices.
- [Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning (Elfwing, Uchibe & Doya, 2017)](https://arxiv.org/abs/1702.03118): where the name SiLU comes from.
- [Gaussian Error Linear Units (GELUs) (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415): the smooth activation function used by GPT models.
- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202): the SwiGLU MLP used by Llama, with its comparison against ReLU and GELU.
- [Llama 3.1 8B config.json (Hugging Face)](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json): the MLP's 4,096 and 14,336 and its SiLU activation.
- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/): the bandwidth and BF16 figures behind the ridge point.
- [Mathematics for Machine Learning (Deisenroth, Faisal & Ong, 2020)](https://mml-book.github.io/): chapter 2 covers linear maps, matrix products, inverses and rank properly.
- [Linear transformations and matrices](https://www.3blue1brown.com/lessons/linear-transformations), [Matrix multiplication as composition](https://www.3blue1brown.com/lessons/matrix-multiplication) and [Inverse matrices, column space, and null space](https://www.3blue1brown.com/lessons/inverse-matrices) (3Blue1Brown): the animated version of Steps 1 to 3.
- [Making Deep Learning Go Brrrr From First Principles (Horace He)](https://horace.io/brrr_intro.html): compute-bound versus memory-bound work, with intuition for why batching helps.
