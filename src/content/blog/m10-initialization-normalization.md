---
title: "Keeping Signals Stable: Initialization, Normalization and Residuals"
description: "Why signals explode or vanish in deep networks, and the math of the fixes: Xavier and He init, LayerNorm and RMSNorm, residuals, pre-norm and μP."
slug: "initialization-normalization"
category: "llm-math"
tags: ["math", "training", "normalization", "transformers", "architecture"]
series: "llm-math"
seriesOrder: 10
pubDate: 2026-10-24
---

## TL;DR

A deep network multiplies its input by one matrix after another, so any per-layer growth or shrinkage compounds. Two rules about variance say how much a layer scales its input, and from them come Xavier and He initialization, the starting weights that keep that factor at 1. Gradients travel back through the same matrices, so vanishing and exploding gradients are the same problem in reverse. During training, LayerNorm and RMSNorm keep sizes in range, and residual connections give gradients a path with a slope of exactly 1. Where the norm sits (pre-norm or post-norm) changes how gradients scale with depth, and μP does the same bookkeeping for width.

## Why it matters

Take a vector of 512 numbers and push it through 50 layers of random weights, each drawn with a spread of 1. Its entries grow to about $$10^{67}$$. Draw the weights with half the right spread instead, and after 50 layers the entries are about $$10^{-15}$$. Exactly one spread keeps the signal steady, and it's set by the width of the layer.

Qwen2.5-1.5B stacks 28 blocks with two branches each, and every training step sends gradients back through all of them. In a plain chain of 56 layers, a gain just 10% too high compounds to a factor of 208. [Part 7](/blog/derivatives-gradients/) showed that the chain rule multiplies Jacobians, one per layer; this part asks what keeps a product of many of them near 1.

## The intuition

Picture a chain of 50 audio amplifiers, each feeding the next. With a gain of 1.1 each, the sound at the end is 117 times louder than the input; with 0.9, it's 0.005 times as loud. A sound engineer sets each stage to unity gain, adds an automatic gain control that resets the level when it drifts, and runs a direct line around the effects so the clean signal always gets through. Those are initialization, normalization and residual connections.

The analogy breaks in two places. A matrix stretches some directions and shrinks others, so "gain" here means the average effect on a vector's size. And an amplifier's knob stays put, while a network's weights change on every training step.

## Step 1: how variance moves through a layer

### Two rules, by hand

The variance of a random quantity is the average of its squared distance from its mean; its square root is the spread, or standard deviation, which [Part 2](/blog/vectors-dot-products/) used for random cosines. Take a coin flip $$X$$ that gives -1 or +1 with equal chance. Its mean is 0 and both outcomes are 1 away from it, so its variance is 1.

Rule one: scaling multiplies variance by the square. Triple the flip to get -3 or +3, each 3 away from the mean, and the variance is 9:

$$
\text{Var}(aX) = a^2\,\text{Var}(X)
$$

Here $$a$$ is any fixed number.

Rule two: for independent quantities, variances add. Flip two coins and add them. The four equally likely sums are -2, 0, 0 and +2, with squared distances 4, 0, 0 and 4 from the mean of 0, so the variance is $$8/4 = 2$$: one plus one.

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)
$$

That holds when $$X$$ and $$Y$$ are independent (knowing one tells you nothing about the other), because the cross term $$2XY$$ in $$(X + Y)^2$$ then averages to zero.

### One layer

A linear layer computes each output as a dot product: $$y_i = \sum_j W_{ij}\, x_j$$, a sum of $$d$$ terms, where $$d$$ is the number of inputs (the fan-in). Suppose each weight is random with mean 0 and variance $$\sigma^2$$ ($$\sigma$$, sigma, is the weights' spread), independent of the inputs. One term $$W_{ij} x_j$$ has mean 0 and variance $$\sigma^2\,\mathbb{E}[x^2]$$, where $$\mathbb{E}[x^2]$$ is the average of the squared input. Adding $$d$$ independent terms adds their variances:

$$
\text{Var}(y) = d\,\sigma^2\,\mathbb{E}[x^2]
$$

In words: each layer multiplies the signal's mean square by $$d\sigma^2$$, the same calculation Part 2 did for two random vectors. To keep that gain at 1, choose

$$
\sigma = \frac{1}{\sqrt{d}}
$$

For a layer with 512 inputs that's $$1/\sqrt{512} \approx 0.044$$. With $$\sigma = 1$$ instead, the gain on the spread is $$\sqrt{512} \approx 22.6$$ per layer.

### Fifty layers

Here are 50 layers with no activation function, at three different spreads:

```python
import numpy as np

d, layers = 512, 50
rng = np.random.default_rng(0)
x0 = rng.standard_normal(d)                       # input with spread 1

def rms(v):                                       # typical size of an entry
    return np.sqrt(np.mean(v ** 2))

for name, sigma in [("sigma = 1", 1.0),
                    ("sigma = 1/sqrt(d)", d ** -0.5),
                    ("sigma = 0.5/sqrt(d)", 0.5 * d ** -0.5)]:
    x, sizes = x0.copy(), {}
    for layer in range(1, layers + 1):
        W = rng.standard_normal((d, d)) * sigma   # weights with spread sigma
        x = W @ x
        if layer in (1, 2, 10, 50):
            sizes[layer] = rms(x)
    print(f"{name:20}" + "  ".join(f"L{k}: {v:.3g}" for k, v in sizes.items()))
```

```text
sigma = 1           L1: 23.4  L2: 515  L10: 3.86e+13  L50: 6.39e+67
sigma = 1/sqrt(d)   L1: 0.972  L2: 0.943  L10: 0.864  L50: 0.804
sigma = 0.5/sqrt(d) L1: 0.487  L2: 0.232  L10: 0.00087  L50: 8.6e-16
```

The printed size is the RMS (root mean square): square every entry, average, take the square root. A gain of 22.6 per layer gives $$22.6^{50} \approx 5 \times 10^{67}$$. A gain of 0.5 gives $$0.5^{50} \approx 8.9 \times 10^{-16}$$. The middle row drifts to 0.804 because each layer's gain is 1 only on average, and the small wobbles compound too.

## Step 2: Xavier and He initialization

### Xavier: forward and backward at once

[Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) studied networks with tanh, whose slope is 1 near zero, so at the start of training Step 1's rule applies: fan-in times variance should be 1. They added a backward condition. Gradients flow back through the transposed matrix, where each input's gradient sums over the outputs it feeds (the fan-out), so the backward pass needs fan-out times variance to be 1. A layer wider on one side can't satisfy both, so they compromise:

$$
\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

Here $$n_{\text{in}}$$ and $$n_{\text{out}}$$ are the fan-in and fan-out. This is Xavier (or Glorot) initialization. Their paper draws uniform weights, and a uniform spread between $$-a$$ and $$a$$ has variance $$a^2/3$$, which gives limits of $$\pm\sqrt{6}/\sqrt{n_{\text{in}} + n_{\text{out}}}$$. The paper notes that the common heuristic of the time, uniform between $$\pm 1/\sqrt{n}$$, gives width times variance of only 1/3, so backpropagated gradients shrink from layer to layer.

### He: ReLU throws away half

ReLU (rectified linear unit) sets negative numbers to 0. If its input $$z$$ is symmetric around zero, half the values become 0 and the rest pass unchanged, so the output's mean square is half the input's variance:

$$
\mathbb{E}\bigl[\text{ReLU}(z)^2\bigr] = \tfrac{1}{2}\,\text{Var}(z)
$$

Feed that into Step 1's rule and each layer multiplies the mean square by $$\tfrac{1}{2} d\sigma^2$$. [He et al. (2015)](https://arxiv.org/abs/1502.01852) set that to 1, which gives weights with spread $$\sqrt{2/d}$$. This is He (or Kaiming) initialization: Xavier's rule with a factor of 2 to replace what ReLU discards. He et al. note that the rule needs $$\mathbb{E}[x^2]$$ rather than the variance, because ReLU outputs have a nonzero mean, and that this is where they part from Glorot and Bengio.

Repeat the 50-layer experiment with a ReLU after every layer:

```python
import torch
torch.manual_seed(0)
d, layers = 512, 50

def rms(t):
    return t.pow(2).mean().sqrt().item()

for name, var in [("1/d (no ReLU fix)", 1 / d), ("2/d (He)", 2 / d)]:
    x0 = torch.randn(d, requires_grad=True)
    x, sizes = x0, []
    for layer in range(layers):
        W = torch.randn(d, d) * var ** 0.5
        x = torch.relu(W @ x)
        sizes.append(rms(x))
    u = torch.randn(d)                 # gradient arriving at the top has spread 1
    (x * u).sum().backward()
    print(f"{name:18} forward L1: {sizes[0]:.3g}  L10: {sizes[9]:.3g}  L50: {sizes[49]:.3g}"
          f"   gradient at input: {rms(x0.grad):.3g}")
```

```text
1/d (no ReLU fix)  forward L1: 0.685  L10: 0.0375  L50: 3.56e-08   gradient at input: 3.23e-08
2/d (He)           forward L1: 1.02  L10: 0.902  L50: 0.596   gradient at input: 0.822
```

Leave the last column for Step 3. With variance $$1/d$$, each layer halves the mean square, so the size falls by $$1/\sqrt{2}$$ per layer, to $$2^{-25} \approx 3 \times 10^{-8}$$ after 50. With He's $$2/d$$, it holds near 1. He et al. saw the same thing in a real network: a 30-layer model initialized with Xavier "completely stalls", while their initialization trained.

## Step 3: gradients ride the same multiplication backward

### Jacobians multiply

Part 7 showed that a linear layer's Jacobian (its grid of partial derivatives) is its weight matrix, and an activation's Jacobian is diagonal, holding its slopes. Backpropagation multiplies the gradient by the transpose of each Jacobian in turn, so the gradient at layer 1 of a 50-layer stack is the top gradient times a product of 50 such matrices.

The variance rules apply unchanged. Each backward step multiplies the gradient's mean square by fan-out times $$\sigma^2$$ times the average squared slope of the activation. For ReLU the slope is 1 or 0 with equal chance, so the factor is $$\tfrac{1}{2} d\sigma^2$$ again. The last column of Step 2's output confirms it. A gradient with spread 1 enters at the top; with variance $$1/d$$ it reaches the input at $$3.23 \times 10^{-8}$$, almost exactly the forward signal's $$3.56 \times 10^{-8}$$. With He's $$2/d$$, it arrives at 0.822.

A gradient of $$3 \times 10^{-8}$$ means the first layer's weights barely move, however long you train. That is a vanishing gradient. Too much gain gives the mirror image, an exploding gradient, where one update throws the weights far off. Both are one fact: a product of many factors goes to zero or infinity unless the factors average 1.

### Saturating activations

The activation's slope matters too. Part 7 found that the sigmoid's slope is at most 0.25, so even at its steepest, 50 sigmoid layers multiply the gradient by $$0.25^{50} \approx 8 \times 10^{-31}$$ unless the weights make up the difference. Glorot and Bengio watched the top hidden layer of a sigmoid network saturate at 0 early in training, which slowed all learning. Modern networks use activations like GELU and SiLU, whose slope is near 1 for positive inputs.

Initialization only sets the gains at step zero. Once training changes the weights, nothing keeps the gains at 1, which is the job of the next two tools.

## Step 4: LayerNorm and RMSNorm

### The same vector, both ways

[Fundamentals Part 6](/blog/transformer-block/) and [Fundamentals Part 10](/blog/modern-llm-architecture/) showed what the two norms do. Here is the arithmetic on a fresh vector, $$\mathbf{x} = [4, -2, 6, 4]$$, with $$d = 4$$ numbers.

Layer normalization ([Ba, Kiros and Hinton, 2016](https://arxiv.org/abs/1607.06450)) first finds the mean, $$\mu = (4 - 2 + 6 + 4)/4 = 3$$ ($$\mu$$, mu). Subtracting it leaves $$[1, -5, 3, 1]$$. The variance is the average square of those, $$(1 + 25 + 9 + 1)/4 = 9$$, so the spread is 3. Divide:

$$
\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\text{Var}(\mathbf{x}) + \varepsilon}}
$$

That gives $$[0.333, -1.667, 1, 0.333]$$, which has mean 0 and variance 1. The $$\varepsilon$$ (epsilon) is a tiny constant, covered below. Then each position is multiplied by a learned gain $$g_i$$ and shifted by a learned bias $$b_i$$, which start at 1 and 0.

RMSNorm ([Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467)) skips the mean. It squares the raw entries, $$16 + 4 + 36 + 16 = 72$$, averages them to 18, and divides by the square root, $$\sqrt{18} \approx 4.243$$:

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \varepsilon}}
$$

That gives $$[0.943, -0.471, 1.414, 0.943]$$, whose mean square is 1, followed by a learned gain and no bias.

The mean square, 18, equals the squared mean, 9, plus the variance, 9, and that holds for any vector:

$$
\tfrac{1}{d}\textstyle\sum_i x_i^2 = \mu^2 + \text{Var}(\mathbf{x})
$$

So when a vector's mean is zero, the two norms agree exactly. RMSNorm's saving is in the work: LayerNorm needs the mean before it can compute the variance, two passes over the vector plus a subtraction per entry, while RMSNorm needs one pass. Fundamentals Part 10 gives the measured speedups.

### Scale invariance

Multiply the input by any positive number $$c$$ and the mean, the spread and the RMS all get multiplied by $$c$$, which cancels in the division. Up to the effect of $$\varepsilon$$:

$$
\text{LN}(c\,\mathbf{x}) = \text{LN}(\mathbf{x}), \qquad \text{RMSNorm}(c\,\mathbf{x}) = \text{RMSNorm}(\mathbf{x})
$$

So whatever size arrives, the output has size 1, however deep the stack. Put an RMSNorm after every layer of Step 1's 50-layer stack and the three spreads, which gave sizes from $$10^{-15}$$ to $$10^{67}$$, all finish at exactly 1.000, with gradients at the input of 0.573, 0.930 and 0.777.

The gradients stay balanced too. If the input to a norm is $$c$$ times bigger, the output doesn't change, so a small nudge to the input matters $$c$$ times less: the norm's Jacobian shrinks like one over the size of its input. [Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) prove this for LayerNorm (their Lemma 3). In the $$\sigma = 1$$ stack, each matrix multiplies the backward gradient by about 22.6 and the norm after it divides by about 22.6, so the two cancel.

### What ε is for

Here are both norms in NumPy, with $$\varepsilon = 10^{-5}$$, the value in GPT-2 and Llama 3.1 8B:

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def layer_norm(x, eps=1e-5):
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    return (x - mu) / np.sqrt(var + eps)

def rms_norm(x, eps=1e-5):
    return x / np.sqrt((x ** 2).mean() + eps)

x = np.array([4.0, -2.0, 6.0, 4.0])
print("mean", x.mean(), " variance", x.var(), " mean square", (x**2).mean())
for c in [1, 100, 0.01]:
    print(f"x * {c:<5} LayerNorm {layer_norm(c * x)}   RMSNorm {rms_norm(c * x)}")
print("x * 0.0001 RMSNorm", rms_norm(1e-4 * x), " no eps", (1e-4*x)/np.sqrt(((1e-4*x)**2).mean()))
print("constant [5,5,5,5] LayerNorm", layer_norm(np.full(4, 5.0)))
with np.errstate(invalid="ignore"):
    print("constant, eps=0       ", layer_norm(np.full(4, 5.0), eps=0.0))
```

```text
mean 3.0  variance 9.0  mean square 18.0
x * 1     LayerNorm [ 0.333 -1.667  1.     0.333]   RMSNorm [ 0.943 -0.471  1.414  0.943]
x * 100   LayerNorm [ 0.333 -1.667  1.     0.333]   RMSNorm [ 0.943 -0.471  1.414  0.943]
x * 0.01  LayerNorm [ 0.331 -1.657  0.994  0.331]   RMSNorm [ 0.94 -0.47  1.41  0.94]
x * 0.0001 RMSNorm [ 0.125 -0.063  0.188  0.125]  no eps [ 0.943 -0.471  1.414  0.943]
constant [5,5,5,5] LayerNorm [0. 0. 0. 0.]
constant, eps=0        [nan nan nan nan]
```

Scaling by 100 changes nothing. The last two lines show the first job of $$\varepsilon$$: a vector of equal entries has variance 0, so LayerNorm would compute 0 divided by 0 and return NaN (not a number), which then spreads through the network. With $$\varepsilon$$, the output is a harmless 0.

The other lines show a second job. At $$c = 0.01$$, adding $$10^{-5}$$ to a variance of 0.0009 changes the third digit. At $$c = 0.0001$$ the mean square is $$1.8 \times 10^{-7}$$, far below $$\varepsilon$$, and the output shrinks to 13% of its normal size. So $$\varepsilon$$ sets a floor: inputs smaller than about $$\sqrt{\varepsilon}$$ get shrunk rather than blown up.

## Step 5: residual connections, a path with slope 1

### I plus J

A residual connection adds a layer's input back to its output: $$\mathbf{y} = \mathbf{x} + f(\mathbf{x})$$, where $$f$$ is the branch, such as attention or an MLP. By the sum rule from Part 7, its Jacobian is

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = I + J_f
$$

where $$I$$ is the identity matrix (1s on the diagonal, 0s elsewhere), the Jacobian of "copy the input", and $$J_f$$ is the branch's Jacobian. Chain two residual layers and multiply out:

$$
(I + J_2)(I + J_1) = I + J_1 + J_2 + J_2 J_1
$$

The first term is $$I$$. However small the branches' Jacobians are, part of the gradient arrives at layer 1 exactly as it left the top, and the same holds for 50 layers: the product expands into terms for every subset of branches, including the empty one, $$I$$. [He et al. (2015)](https://arxiv.org/abs/1512.03385) introduced the idea for image networks, training residual nets 152 layers deep.

Here's a 50-layer tanh stack with weights at half Xavier's spread, with and without the residual connection:

```python
import torch
torch.manual_seed(0)
d, layers = 512, 50

def rms(t):
    return t.pow(2).mean().sqrt().item()

Ws = [torch.randn(d, d) * 0.5 / d ** 0.5 for _ in range(layers)]   # too-small init
u = torch.randn(d)
for name, residual in [("plain    x -> tanh(Wx)    ", False), ("residual x -> x + tanh(Wx)", True)]:
    x0 = torch.randn(d, requires_grad=True)
    x = x0
    for W in Ws:
        x = x + torch.tanh(W @ x) if residual else torch.tanh(W @ x)
    (x * u).sum().backward()
    print(f"{name}  output size {rms(x):.3g}   gradient at layer 1: {rms(x0.grad):.3g}")
```

```text
plain    x -> tanh(Wx)      output size 9.93e-16   gradient at layer 1: 5.95e-16
residual x -> x + tanh(Wx)  output size 4.94   gradient at layer 1: 8.46
```

Both stacks use the same 50 weight matrices. Without the shortcut, the gradient at layer 1 is $$6 \times 10^{-16}$$; with it, 8.46.

### The stream grows

The second line also shows a cost: the output grew from 1 to 4.94. Each branch adds to the stream, and by rule two the variances add. If every branch adds about the same variance $$c$$, then after $$l$$ layers

$$
\text{Var}(\mathbf{x}_l) \approx \text{Var}(\mathbf{x}_0) + l\,c
$$

so the stream's size grows like $$\sqrt{l}$$. In a pre-norm Transformer each branch reads a normalized copy of the stream, so its output is about the same size at every depth. We measured a toy pre-norm stack: 48 blocks of width 256, each with one GELU MLP branch, weights and embeddings drawn with GPT-2's spread of 0.02. The stream's size after block 1 was 0.111; after blocks 12, 24 and 48 it was 0.382, 0.536 and 0.769. Those are $$0.111\sqrt{12}$$, $$0.111\sqrt{24}$$ and $$0.111\sqrt{48}$$ to within 1.5%. [Fundamentals Part 6](/blog/transformer-block/) measured growth in trained GPT-2 small too: the stream at one token got about 88 times longer over 12 blocks.

GPT-2's fix follows from the formula. If each of $$N$$ branches adds variance $$c/N$$, the total added is $$c$$ at any depth. Scaling a branch's output weights by $$1/\sqrt{N}$$ does exactly that (rule one), and it's what the [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) does: "We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers." In our toy, with one branch per block and $$N = 48$$, the stream after block 48 measured 0.112 instead of 0.769.

## Step 6: pre-norm, post-norm, and growing wider

### Where the norm sits

![Two block layouts. Pre-norm, used by GPT-2 and Llama: the stream x splits; one path goes through a norm and then the branch f to an add node, and the skip path, labeled as passing gradients unchanged, goes straight to the add node; the output is x + f(Norm(x)). Post-norm, from 2017: the stream x goes through the branch f to an add node, joined by a skip path, and the sum then passes through a norm; the output is Norm(x + f(x)).](/diagrams/m10-pre-post-norm.svg "Pre-norm puts the norm on the branch, leaving the skip path a pure copy. Post-norm puts it on the main path, after the addition.")
In symbols, the two layouts are

$$
\text{post-norm: } \mathbf{x}_{l+1} = \text{Norm}\bigl(\mathbf{x}_l + f(\mathbf{x}_l)\bigr)
$$

$$
\text{pre-norm: } \mathbf{x}_{l+1} = \mathbf{x}_l + f\bigl(\text{Norm}(\mathbf{x}_l)\bigr)
$$

In post-norm, the Jacobian of each block is the norm's Jacobian times $$(I + J_f)$$, so every path from the top to the bottom crosses all $$L$$ norms, and there's no clean $$I$$ term. The stream is reset to size about 1 at every block, and it never grows. In pre-norm the block's Jacobian is $$I$$ plus a branch term, so Step 5's identity path survives, but the stream grows like $$\sqrt{l}$$.

Xiong et al. worked out the effect on gradients at the start of training. In post-norm, the gradient of the last layer's weights doesn't depend on the number of layers $$L$$; in pre-norm, it shrinks like $$1/\sqrt{L}$$ (their Theorem 1). The final norm sees a stream whose squared size grows linearly with depth (their Lemma 2), and a norm's Jacobian is one over its input's size. We checked with the toy stack, initialized with Xavier as in their setting, averaging three random seeds:

| Blocks L | Post-norm: last layer's gradient | Pre-norm: last layer's gradient |
|---|---|---|
| 6 | 1.390 | 1.003 |
| 12 | 1.385 | 0.782 |
| 24 | 1.388 | 0.612 |
| 48 | 1.433 | 0.451 |

Post-norm stays flat; pre-norm falls by 2.2 times from 6 to 48 blocks. That's less than the $$\sqrt{8} \approx 2.8$$ a pure $$1/\sqrt{L}$$ law predicts for 8 times the depth, since a toy this small doesn't match the theorem's assumptions exactly, but the direction is clear. Xiong et al. argue that post-norm's large gradients near the output are why it needs a learning-rate warm-up, a stretch of tiny learning rates at the start of training, and they trained pre-norm Transformers with no warm-up at all. Fundamentals Part 6 has their translation numbers.

Pre-norm pays for its stability with dilution: each new block adds to a stream that is already $$\sqrt{l}$$ times bigger, so deep blocks change it by a smaller fraction. [DeepNet](https://arxiv.org/abs/2203.00555) stays with post-norm but "up-scales the residual connection before performing layer normalization", multiplying the skip path by a constant greater than 1, and trained Transformers 1,000 layers deep. [Gemma 2](https://arxiv.org/abs/2408.00118) normalizes both the input and the output of each sublayer.

### μP: the same bookkeeping for width

Width causes a subtler problem, which Greg Yang and colleagues addressed with μP (the maximal update parametrization) in [Tensor Programs V (2022)](https://arxiv.org/abs/2203.03466). Initializing with variance $$1/d$$ keeps the forward pass steady at any width, because the random terms of a dot product mostly cancel and their sum grows like $$\sqrt{d}$$. A training update doesn't cancel. The Adam optimizer ([Part 9](/blog/optimizers-adam/)) moves each weight by roughly the learning rate, and the update is built from the very inputs it will multiply, so all $$d$$ terms of $$\Delta W \mathbf{x}$$ push the same way and the sum grows like $$d$$. We checked with an update of size 0.001 per weight, signed to line up with the input: its effect on the output was 0.214 at width 256, 0.811 at 1,024 and 3.264 at 4,096, while a random layer's output stayed near 1.0 at every width.

So in the standard setup, the best learning rate shrinks as the model gets wider, and a value tuned on a small model is wrong for a big one. μP rescales by width to fix that. For Adam, the paper's Table 3 makes the learning rate of hidden weight matrices proportional to $$1/\text{fan-in}$$, initializes the output layer with variance $$1/\text{fan-in}^2$$ instead of $$1/\text{fan-in}$$, and divides attention scores by $$d$$ instead of $$\sqrt{d}$$. The abstract reports the result: "by transferring from 40M parameters, we outperform published numbers of the 6.7B GPT-3 model", "with tuning cost only 7% of total pretraining cost."

## In the real world

- GPT-2's initialization, as reimplemented in [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py), draws every linear and embedding weight with spread 0.02, then reinitializes the output projections that write to the stream with `std=0.02/math.sqrt(2 * config.n_layer)`. The 2 counts two branches per block, attention and MLP, so $$N = 24$$ for GPT-2 small's 12 layers and the spread is 0.0041. The plain 0.02 is below Xavier's $$1/\sqrt{768} = 0.036$$, which the norms make up for.
- Qwen2.5-1.5B's [config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) lists `initializer_range: 0.02` and `rms_norm_eps: 1e-06`. Hugging Face's code uses that spread to initialize a fresh model; how Qwen initialized its own training isn't in the checkpoint. Reading a few trained tensors with safetensors: the first layer's query projection has spread 0.0486, and the MLP down projections in layers 0 and 27 have 0.0268 and 0.0253. All three are larger than the config's 0.02.
- Learned norm gains wander far from their starting value of 1. In the same checkpoint, the first block's input RMSNorm gains average 0.385, the last block's 1.108 and the final norm's 3.46, with a largest gain of 8.875.
- Hugging Face's `Qwen2RMSNorm` converts its input to 32-bit floats before squaring and averaging, even when the model runs in a 16-bit format. Part 14 covers why sums need the extra digits.
- Framework defaults aren't He. [PyTorch's nn.Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html) draws weights uniformly between $$\pm\sqrt{k}$$ with $$k = 1/\text{in\_features}$$, a variance of $$1/(3\,\text{in\_features})$$, the same 1/3 that Glorot and Bengio flagged. A plain ReLU stack of default layers shrinks by $$\sqrt{6}$$ per layer: we measured 0.000129 after 10 layers, as $$6^{-5}$$ predicts.
- Gemma 2 caps attention scores and final logits with `logits ← soft_cap ∗ tanh(logits/soft_cap)`, with a cap of 50 in attention and 30 at the output. Gemma 3 dropped soft-capping for QK-norm, which normalizes queries and keys and which Qwen3 also uses (Fundamentals Part 10).

## Common misconceptions

- **"Normalization makes initialization irrelevant."** It removes the exponential blow-up, but the stream between the norms still grows with depth: in the toy pre-norm stack, GPT-2's $$1/\sqrt{N}$$ scaling changed the final stream size from 0.769 to 0.112. And width ties initialization to the learning rate, which μP addresses.
- **"Residual connections make gradients bigger."** They add a term whose slope is exactly 1, so the gradient can't decay through a product of small factors. In the 50-layer tanh test it went from $$6 \times 10^{-16}$$ to 8.46. The branch terms still add on top, so the stream and the gradient can grow; residual stacks still need norms or scaled initialization.
- **"ε only prevents division by zero."** It also sets a floor. With $$\varepsilon = 10^{-5}$$, a vector whose mean square is $$1.8 \times 10^{-7}$$ comes out at 13% of the size a normalized vector should have, instead of being amplified.
- **"Xavier and He are two different theories."** Both come from one formula, $$\text{Var}(y) = d\,\sigma^2\,\mathbb{E}[x^2]$$. Xavier assumes an activation with slope 1 near zero and averages the forward and backward conditions; He accounts for ReLU zeroing half its inputs, which costs a factor of 2.

## Key takeaways

- Variances scale with the square of a multiplier and add over independent terms, so a layer multiplies the signal's mean square by fan-in times the weights' variance, and 50 layers raise that to the 50th power.
- Xavier initialization picks a variance of two over fan-in plus fan-out; He initialization picks two over fan-in for ReLU, to replace the half that ReLU zeroes. Both aim for a gain of 1.
- Gradients pass back through the same matrices, so they vanish or explode for the same reason the forward signal does.
- LayerNorm subtracts the mean and divides by the spread; RMSNorm divides by the root mean square only. Both make the output independent of the input's scale, and epsilon keeps tiny inputs from being blown up.
- A residual connection has Jacobian identity plus the branch's, so gradients always have a direct path. Pre-norm keeps that path clean but lets the stream grow like the square root of depth; post-norm caps the stream but needs warm-up. μP applies the same bookkeeping to width.

## What's next

Attention has its own size problem: the dot product of a query and a key grows with the head's width. Part 11, The math of attention, derives scaled dot-product attention, explains its division by the square root of d with this part's variance rules, and works through the softmax's Jacobian, the cost of long contexts, and grouped-query and latent attention as factorizations.

## Further reading

- [Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio, 2010)](https://proceedings.mlr.press/v9/glorot10a.html): the variance analysis behind Xavier initialization, with the forward and backward conditions and the compromise between them.
- [Delving Deep into Rectifiers (He, Zhang, Ren & Sun, 2015)](https://arxiv.org/abs/1502.01852): section 2.2 derives the factor of 2 for ReLU and shows a 30-layer network that Xavier initialization can't train.
- [Deep Residual Learning for Image Recognition (He, Zhang, Ren & Sun, 2015)](https://arxiv.org/abs/1512.03385): the residual connection, and networks 152 layers deep.
- [Layer Normalization (Ba, Kiros & Hinton, 2016)](https://arxiv.org/abs/1607.06450) and [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467): the two norms, their invariance properties and their costs.
- [On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745): the gradient analysis of post-norm and pre-norm, and training without warm-up.
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): section 2.3 gives GPT-2's pre-norm layout and the 1/√N residual scaling.
- [DeepNet: Scaling Transformers to 1,000 Layers (Wang et al., 2022)](https://arxiv.org/abs/2203.00555): a post-norm variant with an up-scaled skip path.
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (Yang et al., 2022)](https://arxiv.org/abs/2203.03466): μP, its Table 3 of width scalings, and hyperparameter transfer to GPT-3 6.7B.
- [Gemma 2: Improving Open Language Models at a Practical Size (Gemma Team, 2024)](https://arxiv.org/abs/2408.00118): logit soft-capping and RMSNorm on both sides of each sublayer.
- [nanoGPT model.py (Andrej Karpathy)](https://github.com/karpathy/nanoGPT/blob/master/model.py): GPT-2's initialization, including the scaled residual projections, in a few lines.
- [Let's build GPT: from scratch, in code, spelled out (Andrej Karpathy, 2023)](https://www.youtube.com/watch?v=kCc8FmEb1nY): a small GPT built line by line, including its residual connections and pre-norm layer norms.
