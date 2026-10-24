---
title: "Backpropagation by Hand: From the Chain Rule to Autograd"
description: "The chain rule as an algorithm: computational graphs, backward passes for matmul, ReLU, GELU and softmax, activation memory, and a 40-line autograd."
slug: "backpropagation-math"
category: "llm-math"
tags: ["math", "training", "backpropagation", "gradient-descent", "neural-networks"]
series: "llm-math"
seriesOrder: 8
pubDate: 2026-10-24
---

## TL;DR

Backpropagation is the chain rule from Part 7 run as an algorithm. Write the model as a graph of small operations, run it forward to get the loss, then walk backward applying one rule at every node: the gradient arriving from the loss side times the node's local slope, summed wherever a value was used twice. This part works the rule by hand, derives the backward pass of matrix multiplication, ReLU, GELU and softmax with cross-entropy (where the gradient is p minus y), and shows why the backward pass costs about twice the forward and needs so much memory. It ends with a working autograd engine in under 40 lines of Python, checked against PyTorch.

## Why it matters

Training GPT-3 the textbook way would need about 275 GB of memory for a single 2,048-token sequence, before storing one weight. The backward pass needs numbers the forward pass made, so the forward pass has to keep them.

The compute bill is easier to remember: a backward pass costs about twice a forward pass. That's where the 6 in the training-compute rule 6ND comes from. Both facts follow from one small rule, applied to every operation in the model.

[Part 7](/blog/derivatives-gradients/) built the chain rule and Jacobians, and [Fundamentals Part 3](/blog/neural-networks-basics/) walked a two-weight network backward in words. This part turns that walk into an algorithm and works out the backward pass of each operation in an LLM's training step.

## The intuition

Picture a factory whose product ships with a costly defect. Each station knows only its own job: how its output moves when its inputs move. Blame flows backward from the last inspector. Each station takes the number it receives ("this much cost per unit of your output"), multiplies it by its own local rate, and passes the result to the stations that fed it. A station that supplied two others adds up what both send back. Nobody needs to understand the whole factory, and one walk from end to start gives every station its number. That walk is backpropagation.

The analogy breaks in two places. Factory blame gets split into shares that add up to the total, while an addition passes the full gradient to both of its inputs. And a factory can forget what it built yesterday; a network can't, because most local rates depend on values from the forward pass.

## Step 1: a graph, a forward pass and one rule

### The graph

Take Part 7's one-point model: a line $$wx + b$$ fitted to the point $$x = 3$$, $$y = 7$$, with loss $$L = (wx + b - y)^2$$ and starting weights $$w = 1$$, $$b = 1$$. Break it into operations that each do one thing:

```text
u = w × x      = 1 × 3    =  3
v = u + b      = 3 + 1    =  4
e = v - y      = 4 - 7    = -3
L = e²         = (-3)²    =  9
```

That is a computational graph. Each line is a node that holds one value and remembers which values it was computed from. Running the lines top to bottom is the forward pass.

### The backward pass, node by node

Write $$\bar{u}$$ ("u bar") for $$\partial L / \partial u$$, how much the loss moves per unit change in $$u$$. The autodiff literature calls it the adjoint of $$u$$. Start at the end, where $$\bar{L} = 1$$ because the loss changes one-for-one with itself, and walk up the list. At each node, take its gradient, called the upstream gradient because it arrives from the loss side, and multiply by the local slope to each input:

$$
\bar{u} = \bar{v} \cdot \frac{\partial v}{\partial u}
$$

In words: an input's gradient is its output's gradient times the local slope that links them. The local slope depends only on the node itself: 1 for an addition, the other factor for a multiplication, $$2e$$ for a square.

| Node | Local slopes | Upstream gradient | Gradients it sends back |
|---|---|---|---|
| L = e² | dL/de = 2e = -6 | 1 | dL/de = -6 |
| e = v - y | 1 for v, -1 for y | -6 | dL/dv = -6, dL/dy = 6 |
| v = u + b | 1 for u, 1 for b | -6 | dL/du = -6, dL/db = -6 |
| u = w × x | x = 3 for w, w = 1 for x | -6 | dL/dw = -18, dL/dx = -6 |

Part 7 found $$\partial L / \partial w = -18$$ and $$\partial L / \partial b = -6$$ from a formula for the whole loss; the walk gets them without one. It also produces a gradient for $$x$$. In a deep network one layer's input is the output of the layer below, which needs exactly that gradient to continue the walk.

![A top-to-bottom computational graph for L = (w x + b - y) squared. Inputs w = 1 (gradient -18) and x = 3 (gradient -6) feed u = w × x, value 3, gradient -6, along edges with local slopes x = 3 and w = 1. u and b = 1 (gradient -6) feed v = u + b, value 4, gradient -6, with local slopes 1 and 1. v and y = 7 (gradient +6) feed e = v - y, value -3, gradient -6, with local slopes 1 and -1. e feeds L = e squared, value 9, gradient 1, with local slope 2e = -6.](/diagrams/m08-backward-graph.svg "Forward values flow down; gradients flow back up. Each input's gradient is the gradient below it times the local slope on the edge.")

Every gradient in the graph is a product of local slopes along the path to the loss, which the backward pass builds one multiplication at a time.

### When a value is used twice

The rule needs one addition. If a value feeds two nodes, a nudge to it reaches the loss along both paths, and the two effects add, by the sum rule from Part 7:

$$
\bar{u} = \sum_{\text{nodes } v \text{ that use } u} \bar{v} \cdot \frac{\partial v}{\partial u}
$$

Take $$q = ab + a$$ at $$a = 2$$, $$b = 5$$, so $$q = 12$$. The value $$a$$ feeds the multiplication, with local slope $$b = 5$$, and the addition, with local slope 1. Starting from $$\bar{q} = 1$$, the total is $$\bar{a} = 5 + 1 = 6$$, and a central difference gives 6.000000001. Code that overwrote instead of adding would report 5 or 1.

LLMs are full of this fan-out. Every residual connection uses its input twice, once in the branch and once in the skip, and every weight matrix is used by every token in the batch.

### The algorithm

A node's gradient is complete only after every node that uses it has sent its share. Walking a topological sort (an ordering in which every node comes after its inputs) in reverse guarantees that. The algorithm:

1. Forward: compute every node in topological order, saving whatever its local slopes will need.
2. Set the loss's gradient to 1 and every other gradient to 0.
3. Backward: visit nodes in reverse order, and for each one add (its gradient × local slope) to each input's gradient.

This is reverse-mode automatic differentiation. Step 6 writes it in Python.

## Step 2: why run it backward

The chain rule can also run forward. Forward mode picks one input, say $$w$$, and carries "how much does this node change per unit of $$w$$" alongside the values: $$du/dw = x = 3$$, then $$dv/dw = 3$$, $$de/dw = 3$$, and $$dL/dw = 2e \times 3 = -18$$. Same answer, but only one gradient. Getting the one for $$b$$ takes a second pass that starts from $$b$$.

Each forward-mode pass gives the slopes with respect to one input; each reverse-mode pass gives the slopes of one output with respect to every input. A training loss is one number that depends on billions of weights:

| Model | Weights | Forward-mode passes | Reverse-mode passes |
|---|---|---|---|
| The line above | 2 | 2 | 1 |
| Llama 3.1 8B | 8,030,261,248 | 8,030,261,248 | 1 |

For a function from $$n$$ inputs to one output, forward mode "requires n evaluations to compute the gradient", as Baydin and colleagues' [autodiff survey](https://arxiv.org/abs/1502.05767) puts it, while reverse mode needs one pass. Forward mode only wins with few inputs and many outputs.

In Part 7's terms, the backward pass multiplies the loss's gradient into one Jacobian at a time from the loss end. Steps 3 and 4 work out that product for each kind of node without building any Jacobian.

## Step 3: the backward pass of a matrix multiply

### A 2 × 2 example by hand

A linear layer computes $$Y = XW$$, where each row of $$X$$ is one token's vector and $$W$$ holds the weights. Suppose the layers above have already sent back $$\bar{Y}$$, one gradient for each entry of $$Y$$:

$$
X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
W = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}, \quad
\bar{Y} = \begin{bmatrix} 1 & 2 \\ -1 & 0 \end{bmatrix}
$$

The product is $$Y = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$. Where does the weight $$W_{11} = 5$$ appear? In the first column of $$Y$$, once per row:

```text
Y11 = X11·W11 + X12·W21 = 1·5 + 2·7 = 19
Y21 = X21·W11 + X22·W21 = 3·5 + 4·7 = 43
```

It's used twice, so this is fan-out. Its local slopes are $$X_{11} = 1$$ and $$X_{21} = 3$$, and the two contributions add:

```text
dL/dW11 = Ȳ11·X11 + Ȳ21·X21 = 1·1 + (-1)·3 = -2
```

Now the input $$X_{11} = 1$$. It appears in the first row of $$Y$$, once per column, with local slopes $$W_{11} = 5$$ and $$W_{12} = 6$$:

```text
dL/dX11 = Ȳ11·W11 + Ȳ12·W12 = 1·5 + 2·6 = 17
```

### The general rule

Each of those sums pairs a row of one matrix with a column of another, which is what a matrix product does. Doing all eight entries gives two formulas:

$$
\bar{X} = \bar{Y}\, W^{\top}
$$

$$
\bar{W} = X^{\top}\, \bar{Y}
$$

In words: the gradient for the inputs is the upstream gradient times the transposed weights, and the gradient for the weights is the transposed inputs times the upstream gradient. The transpose, from [Part 1](/blog/math-for-llms/), flips a matrix so rows become columns.

The shapes tell you where the transposes go. Say $$X$$ is $$m \times k$$ and $$W$$ is $$k \times n$$, so $$Y$$ and $$\bar{Y}$$ are $$m \times n$$. A gradient has the shape of the thing it belongs to, so $$\bar{X}$$ is $$m \times k$$ and $$\bar{W}$$ is $$k \times n$$. The only way to make $$m \times k$$ from $$\bar{Y}$$ and $$W$$ is $$(m \times n)$$ times $$(n \times k)$$, which is $$\bar{Y} W^{\top}$$. The only way to make $$k \times n$$ from $$X$$ and $$\bar{Y}$$ is $$(k \times m)$$ times $$(m \times n)$$, which is $$X^{\top} \bar{Y}$$.

For one token this is Part 7's result: $$\mathbf{y} = W\mathbf{x}$$ has Jacobian $$W$$, and its backward pass is $$\bar{\mathbf{x}} = W^{\top}\bar{\mathbf{y}}$$, the gradient times the Jacobian written as a column. PyTorch agrees with the hand results:

```python
import torch

X = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
W = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)
G = torch.tensor([[1., 2.], [-1., 0.]])     # dL/dY, sent back from above

Y = X @ W
(G * Y).sum().backward()                    # a loss whose dL/dY is exactly G

print("Y          ", Y.tolist())
print("autograd dX", X.grad.tolist(), " G W^T", (G @ W.T).tolist())
print("autograd dW", W.grad.tolist(), " X^T G", (X.T @ G).tolist())
```

```text
Y           [[19.0, 22.0], [43.0, 50.0]]
autograd dX [[17.0, 23.0], [-5.0, -7.0]]  G W^T [[17.0, 23.0], [-5.0, -7.0]]
autograd dW [[-2.0, 2.0], [-2.0, 4.0]]  X^T G [[-2.0, 2.0], [-2.0, 4.0]]
```

The same check on random 64 × 32 and 32 × 48 matrices in float64 also matches exactly. A bias $$\mathbf{b}$$ added to every row is fan-out once more: its gradient is the sum of the rows of $$\bar{Y}$$.

### Why backward costs twice the forward

Count with Part 1's rule that an $$(m \times k)$$ times $$(k \times n)$$ product costs $$2mkn$$ FLOPs (floating-point operations). The forward pass is one such product. The backward pass is two: $$\bar{Y}W^{\top}$$ is $$(m \times n)$$ times $$(n \times k)$$, and $$X^{\top}\bar{Y}$$ is $$(k \times m)$$ times $$(m \times n)$$, each $$2mkn$$ again:

$$
\underbrace{2mkn}_{\text{forward}} + \underbrace{2mkn + 2mkn}_{\text{backward}} = 6mkn
$$

The backward pass is twice the forward, and a training step is three times it. Each row of $$X$$ is a token, so per token a matrix with $$N = kn$$ weights costs $$2N$$ FLOPs forward and $$4N$$ backward. For Llama 3.1 8B's 4,096 × 14,336 MLP matrix, that's 117.4 million FLOPs per token going forward and 234.9 million coming back.

Add that up over every weight $$N$$ in the model and every training token $$D$$ and you get $$6ND$$, the rule [Fundamentals Part 12](/blog/pretraining-scaling-laws/) used to check Llama 3.1 405B's compute budget. [Part 15](/blog/counting-flops-scaling-laws/) derives it properly, with what this count leaves out (attention's token-to-token products, the free embedding lookup). The [scaling-laws paper](https://arxiv.org/abs/2001.08361) puts the backward pass at "approximately twice the compute as the forwards pass". Activations, softmax and norms add only a few FLOPs per number in either direction.

## Step 4: activations and the output layer

### ReLU: a mask

ReLU keeps positive numbers and zeroes the rest, so its local slope is 1 where the input was positive and 0 elsewhere (0 at the kink, by convention, as Part 7 noted). With $$\mathbf{h} = \text{ReLU}(\mathbf{z})$$:

$$
\bar{z}_i = \bar{h}_i \cdot [z_i > 0]
$$

The bracket is 1 when the condition holds and 0 when it doesn't, so ReLU's backward pass is a mask. For $$\mathbf{z} = [2, -1, 0.5]$$ and upstream gradient $$[0.3, 0.7, -0.2]$$, the result is $$[0.3, 0, -0.2]$$. The forward pass only has to remember one bit per number, which side of zero it fell on.

A hidden unit whose input is negative for every training example gets a gradient of exactly 0 at every step, so it never comes back. Step 6's toy network runs into this.

### GELU, exact and approximate

GELU (Gaussian error linear unit), the activation in GPT-2, is defined in [Hendrycks and Gimpel's paper](https://arxiv.org/abs/1606.08415) as:

$$
\text{GELU}(z) = z\,\Phi(z)
$$

Here $$\Phi(z)$$ is the share of a standard bell curve (mean 0, standard deviation 1) lying left of $$z$$: 0.5 at $$z = 0$$, close to 1 for large $$z$$. The product rule from Part 7 gives the slope:

$$
\text{GELU}'(z) = \Phi(z) + z\,\varphi(z)
$$

where $$\varphi(z) = e^{-z^2/2} / \sqrt{2\pi}$$ is the height of the bell curve at $$z$$, which is also the slope of $$\Phi$$. At $$z = 1$$, $$\Phi(1) = 0.8413$$ and $$\varphi(1) = 0.2420$$, so the slope is 1.0833.

Computing $$\Phi$$ exactly needs the error function erf, $$\Phi(z) = \tfrac{1}{2}\bigl(1 + \text{erf}(z / \sqrt{2})\bigr)$$. The paper also gives a cheaper approximation, and used it "in every experiment in this paper":

$$
\text{GELU}(z) \approx \tfrac{1}{2}\, z\, \bigl(1 + \tanh(s)\bigr)
$$

$$
s = \sqrt{2/\pi}\,\bigl(z + 0.044715\, z^3\bigr)
$$

Here tanh is the hyperbolic tangent, an S-shaped curve from -1 to 1. Central differences from Part 7 check the slope formula and measure how far the approximation drifts:

```python
import math
import numpy as np

Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))      # bell curve's area left of z
phi = lambda z: math.exp(-z * z / 2) / math.sqrt(2 * math.pi)  # bell curve's height at z
gelu = lambda z: z * Phi(z)                                  # exact GELU
gelu_slope = lambda z: Phi(z) + z * phi(z)                   # product rule

c = math.sqrt(2 / math.pi)
def gelu_tanh(z):                                            # the paper's tanh approximation
    return 0.5 * z * (1 + math.tanh(c * (z + 0.044715 * z**3)))

def central(f, z, h=1e-5):
    return (f(z + h) - f(z - h)) / (2 * h)

for z in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    print(f"z={z:4}  slope {gelu_slope(z):.6f}  central {central(gelu, z):.6f}"
          f"  tanh-version {central(gelu_tanh, z):.6f}")

zs = np.linspace(-6, 6, 120001)
print("max gap in value:", max(abs(gelu(z) - gelu_tanh(z)) for z in zs))
print("max gap in slope:", max(abs(gelu_slope(z) - central(gelu_tanh, z)) for z in zs))
```

```text
z=-2.0  slope -0.085232  central -0.085232  tanh-version -0.086099
z=-1.0  slope -0.083315  central -0.083315  tanh-version -0.082964
z= 0.0  slope 0.500000  central 0.500000  tanh-version 0.500000
z= 1.0  slope 1.083315  central 1.083315  tanh-version 1.082964
z= 2.0  slope 1.085232  central 1.085232  tanh-version 1.086099
max gap in value: 0.000473235519323335
max gap in slope: 0.0008684518396471219
```

The slope formula matches the nudges to six decimals. Between -6 and 6, the tanh version strays at most 0.00047 from the exact GELU, and its slope at most 0.00087. Small, but it's a different function, so PyTorch's `nn.GELU` offers both (`approximate='none'`, the default, or `'tanh'`), and GPT-2's [Hugging Face config](https://huggingface.co/openai-community/gpt2/blob/main/config.json) asks for the tanh version under the name `gelu_new`.

SiLU, the activation in Llama, works the same way with the slope Part 7 derived. Both need the full input $$z$$ saved for the backward pass, where ReLU needed one bit.

### Softmax with cross-entropy: p minus y

At the output, the model turns its logits $$\mathbf{z}$$ (one score per token in the vocabulary) into probabilities $$\mathbf{p}$$ with softmax, and the loss is $$L = -\ln p_c$$, where $$c$$ is the correct token. [Part 5](/blog/softmax-math/) showed that a log-probability is the logit minus the log-sum-exp, so:

$$
L = -z_c + \ln \sum_j e^{z_j}
$$

Part 5 also showed that the log-sum-exp's slope with respect to a logit $$z_i$$ is $$p_i$$. The first term contributes -1 when $$i$$ is the correct token and 0 otherwise. Writing $$\mathbf{y}$$ for the one-hot vector, 1 at the correct token and 0 elsewhere:

$$
\frac{\partial L}{\partial z_i} = p_i - y_i
$$

In words: each logit's gradient is the probability the model gave that token minus the probability it should have given. Part 7's two-token $$p - 1$$ is a special case.

Take three tokens with logits $$[2, 1, 0]$$, where the middle one is correct. Softmax gives $$\mathbf{p} = [0.6652, 0.2447, 0.0900]$$, so the gradient is $$[0.6652, -0.7553, 0.0900]$$. Gradient descent raises the correct logit and lowers the others in proportion to the probability they wrongly took. The entries sum to 0 because $$\mathbf{p}$$ and $$\mathbf{y}$$ each sum to 1.

```python
import numpy as np
import torch
import torch.nn.functional as F

z = np.array([2.0, 1.0, 0.0])            # logits for three tokens
target = 1                                # the correct next token is token 1

def loss(z):                              # cross-entropy = -log softmax(z)[target]
    return -(z[target] - np.log(np.exp(z).sum()))

p = np.exp(z) / np.exp(z).sum()
y = np.eye(3)[target]                     # one-hot: [0, 1, 0]
print("p      ", p.round(4))
print("p - y  ", (p - y).round(4))

h = 1e-5                                  # central differences, one logit at a time
num = [(loss(z + h * np.eye(3)[i]) - loss(z - h * np.eye(3)[i])) / (2 * h) for i in range(3)]
print("nudging", np.round(num, 4))

zt = torch.tensor(z, requires_grad=True)
F.cross_entropy(zt.unsqueeze(0), torch.tensor([target])).backward()
print("PyTorch", zt.grad.numpy().round(4))
```

```text
p       [0.6652 0.2447 0.09  ]
p - y   [ 0.6652 -0.7553  0.09  ]
nudging [ 0.6652 -0.7553  0.09  ]
PyTorch [ 0.6652 -0.7553  0.09  ]
```

Differentiating softmax and the log one after the other would multiply $$-1/p_c$$ into softmax's Jacobian, an $$n \times n$$ grid of rates. For Llama 3.1 8B's 128,256-token vocabulary, that grid would have 16.4 billion entries per token, and $$-1/p_c$$ blows up when $$p_c$$ rounds to 0 (Part 5's `nan`). The combined gradient is 128,256 subtractions with nothing to divide. That's why PyTorch's `cross_entropy` takes raw logits and runs log-softmax and the loss as one operation, whose gradient comes out as $$\mathbf{p} - \mathbf{y}$$.

## Step 5: what the backward pass has to remember

### The forward pass keeps its inputs

Every backward rule so far needs something from the forward pass:

| Operation | What the backward pass needs | Why |
|---|---|---|
| Y = XW | X | the weight gradient is Xᵀ times the upstream gradient |
| ReLU | which inputs were positive | the mask |
| GELU, SiLU | the input z | the slope depends on z |
| Softmax with cross-entropy | the probabilities p | the gradient is p - y |

The weights sit in memory anyway. The rest is activation memory: tensors the forward pass creates and the backward pass consumes, which live only between the two passes and grow with the number of tokens.

PyTorch's `saved_tensors_hooks` call a function on every tensor autograd saves, so you can count them. Here is a GPT-style MLP: width $$h$$ up to $$4h$$, a GELU, and back down:

```python
import torch
from torch.utils.checkpoint import checkpoint

s, h = 512, 1024                          # 512 tokens, width 1,024
mlp = torch.nn.Sequential(torch.nn.Linear(h, 4 * h), torch.nn.GELU(),
                          torch.nn.Linear(4 * h, h))
weights = {p.data_ptr() for p in mlp.parameters()}
x = torch.randn(s, h, requires_grad=True)

def saved_numbers(run):                   # count numbers autograd keeps for backward
    seen = {}
    def pack(t):
        if t.data_ptr() not in weights:   # skip the weights; count activations only
            seen[t.data_ptr()] = t.numel()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        run()
    return sum(seen.values())

print(saved_numbers(lambda: mlp(x)) / (s * h))
print(saved_numbers(lambda: checkpoint(mlp, x, use_reentrant=False)) / (s * h))
```

```text
9.0
1.0
```

The MLP keeps 9 numbers per token for each unit of width: the input to the first matrix (1), GELU's input (4) and GELU's output, which is the second matrix's input (4). The checkpointed version keeps only the input; more on that below.

### A whole layer of GPT-3

[Korthikanti and colleagues](https://arxiv.org/abs/2205.05198) did this count for a full GPT-style layer, assuming 16-bit numbers (2 bytes each) and 1 byte per entry for dropout masks (dropout randomly zeroes numbers during training). Their result, in bytes per layer:

$$
s\,b\,h \left(34 + \frac{5as}{h}\right)
$$

Here $$s$$ is the tokens per sequence, $$b$$ the sequences in the batch, $$h$$ the hidden width and $$a$$ the number of attention heads. The 34 covers matrix-multiply inputs, GELU, layer norms and dropout; the MLP's share is $$19sbh$$, the 9 numbers just measured at 2 bytes each plus a 1-byte dropout mask. The $$5as/h$$ term is attention's $$s \times s$$ grid of scores for every head.

For GPT-3 175B, the paper uses $$s = 2048$$, $$h = 12288$$ and $$a = 96$$, so $$5as/h = 80$$. One sequence then needs $$2048 \times 12288 \times 114$$ bytes, 2.87 GB per layer, and 275 GB across all 96 layers. The paper's A100 GPUs have 80 GB each, and the weights alone take 350 GB in 16 bits.

### Trading compute for memory

Activation checkpointing, also called activation recomputation, keeps only each layer's input and reruns the layer's forward during the backward pass to rebuild the rest. PyTorch's [`torch.utils.checkpoint`](https://docs.pytorch.org/docs/2.14/checkpoint.html) calls it "a technique that trades compute for memory". For GPT-3 that's $$2sbh$$ bytes per layer, 50 MB, or 4.8 GB for all 96 layers instead of 275 GB. The cost is one extra forward pass, about four forward passes per step instead of three; the paper observed 30 to 40% extra computation time. [Chen and colleagues (2016)](https://arxiv.org/abs/1604.06174) showed that checkpointing only some layers needs memory that grows like the square root of the layer count.

Korthikanti and colleagues take a middle path. Attention's score grid is 70% of GPT-3's activations but cheap to recompute, so they recompute only that part: 70% less activation memory for 2.7% more FLOPs. [FlashAttention](/blog/flashattention/) goes further and never stores the grid at all, rebuilding it tile by tile in the backward pass.

## Step 6: a tiny autograd in Python

Everything above fits in one small class. It follows [micrograd](https://github.com/karpathy/micrograd), Andrej Karpathy's "tiny Autograd engine", which runs backpropagation "over a dynamically built DAG" (directed acyclic graph). Each `Value` holds one number, the values it came from, and its local slopes to them, computed at the moment it's created:

```python
import math

class Value:
    """One number, plus what's needed to send gradients back through it."""
    def __init__(self, data, parents=(), local_grads=()):
        self.data = data                  # the number, computed going forward
        self.grad = 0.0                   # dL/d(this number), filled in going backward
        self.parents = parents            # the Values it was computed from
        self.local_grads = local_grads    # d(this)/d(each parent), saved going forward

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    __radd__, __rmul__ = __add__, __mul__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)

    def relu(self):
        return Value(max(self.data, 0.0), (self,), (float(self.data > 0),))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def backward(self):
        order, seen = [], set()
        def visit(v):                     # topological sort: inputs before outputs
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0                   # dL/dL = 1
        for v in reversed(order):         # outputs before inputs
            for p, local in zip(v.parents, v.local_grads):
                p.grad += local * v.grad  # local slope x upstream; += adds up fan-out
```

Each operation stores the "local slopes" column of Step 1's table: 1 and 1 for addition, the other factor for multiplication, the mask bit for ReLU, $$e^x$$ for exp (its own slope) and $$1/x$$ for log. The `backward` method is Step 1's algorithm: a depth-first search lists every node after its inputs, then the one rule runs in reverse order, with `+=` handling fan-out.

### A network to train

The test problem is XOR, which [Part 3](/blog/matrices-linear-maps/) showed no straight line can solve: output class 1 when exactly one of two inputs is 1. The network has 2 inputs, 4 ReLU hidden units and 2 logits, 22 weights in all, and the loss is the cross-entropy from Step 4, written with the log-sum-exp:

```python
import random
from tiny_autograd import Value           # the class above, saved as tiny_autograd.py

random.seed(1)
X = [(0, 0), (0, 1), (1, 0), (1, 1)]
Y = [0, 1, 1, 0]                          # XOR: class 1 when exactly one input is 1
shapes = [(4, 2), (4, 1), (2, 4), (2, 1)] # W1, b1, W2, b2: 2 inputs, 4 hidden, 2 logits
params = [Value(random.uniform(-1, 1)) for r, c in shapes for _ in range(r * c)]

def forward(ps):                          # works on Values and on torch tensors alike
    W1, b1, W2, b2 = ps[0:8], ps[8:12], ps[12:20], ps[20:22]
    total = 0
    for (x0, x1), y in zip(X, Y):
        hid = [(W1[2*i] * x0 + W1[2*i + 1] * x1 + b1[i]).relu() for i in range(4)]
        z = [sum((W2[4*j + i] * hid[i] for i in range(4)), b2[j]) for j in range(2)]
        total = total + ((z[0].exp() + z[1].exp()).log() - z[y])  # cross-entropy
    return total * 0.25                   # mean over the 4 examples
```

### Check it before trusting it

Before training, compare all 22 gradients with PyTorch's autograd, running the same `forward` on PyTorch tensors, and with Part 7's central differences:

```python
import torch

loss = forward(params)
loss.backward()                           # our engine
ours = torch.tensor([p.grad for p in params], dtype=torch.float64)

ts = [torch.tensor(p.data, dtype=torch.float64, requires_grad=True) for p in params]
forward(ts).backward()                    # PyTorch autograd, same function
theirs = torch.stack([t.grad for t in ts])

h, nudged = 1e-6, []                      # central differences, one weight at a time
for p in params:
    p.data += h;     up = forward(params).data
    p.data -= 2 * h; down = forward(params).data
    p.data += h
    nudged.append((up - down) / (2 * h))

print(f"loss {loss.data:.6f}, {len(params)} gradients")
print("largest gradient:       ", ours.abs().max().item())
print("max gap vs PyTorch:     ", (ours - theirs).abs().max().item())
print("max gap vs nudging:     ", (ours - torch.tensor(nudged)).abs().max().item())
```

```text
loss 0.838159, 22 gradients
largest gradient:        0.2463455518578762
max gap vs PyTorch:      0.0
max gap vs nudging:      6.0201815199611985e-09
```

The engine agrees with PyTorch exactly, since both do the same float64 arithmetic, and with nudging to about $$6 \times 10^{-9}$$, the kind of leftover rounding error Part 7 showed central differences carry.

### Training

Gradient descent from Part 7, with the gradients reset to zero before each backward pass, because `+=` would otherwise pile up the previous step's:

```python
for step in range(301):
    loss = forward(params)
    for p in params:
        p.grad = 0.0                      # forget the last step's gradients
    loss.backward()
    for p in params:
        p.data -= 0.5 * p.grad            # gradient descent, learning rate 0.5
    if step % 100 == 0:
        print(f"step {step:3}  loss {loss.data:.4f}")
```

```text
step   0  loss 0.8382
step 100  loss 0.1102
step 200  loss 0.0261
step 300  loss 0.0132
```

A loss of 0.0132 means the model gives the right class $$e^{-0.0132} = 98.7\%$$ probability on average (a geometric average, as in [Part 6](/blog/cross-entropy-perplexity/)'s perplexity). Of random seeds 0 to 9, three stall, at losses of 0.348, 0.478 and 0.693. With seed 5, one hidden unit's input is negative for all four examples, so the ReLU mask zeroes its gradient forever, and the other three are positive for all four, which makes the network a straight-line function. It stays at $$\ln 2 = 0.693$$, a coin flip.

## In the real world

- PyTorch's [autograd mechanics notes](https://docs.pytorch.org/docs/2.14/notes/autograd.html) describe Step 6 at scale: "Autograd is a reverse automatic differentiation system" that "records a graph recording all of the operations that created the data as you execute operations". The graph "is recreated from scratch at every iteration", so Python `if` statements and loops inside a model just work, and `x.pow(2)` saves `x` for the backward pass, like `local_grads` above.
- `loss.backward()` runs Step 1's algorithm and then frees the graph. Per the [docs](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.backward.html), with `retain_graph` false "the graph used to compute the grads will be freed", so a second `backward()` on the same graph fails unless the first set `retain_graph=True`. It also "accumulates gradients in the leaves", which is why training loops call `optimizer.zero_grad()`, like `p.grad = 0.0` above.
- Recomputing less pays off at scale. With selective recomputation plus sequence parallelism (splitting some activations across GPUs), Korthikanti and colleagues trained a 530-billion-parameter GPT-3-style model on 2,240 A100 GPUs at 54.2% model FLOPs utilization, against 42.1% with full recomputation, 29% faster.
- A real LLM's backward pass also runs through attention, where softmax's full Jacobian finally matters ([Part 11](/blog/attention-math/)), and through normalization layers and residual connections, whose fan-out adds an extra path for the gradient to every layer ([Part 10](/blog/initialization-normalization/)).

## Common misconceptions

- **"Backpropagation multiplies big Jacobian matrices."** It never builds them. Each node turns an upstream gradient into its inputs' gradients directly: a matmul uses two matmuls of the same size, ReLU uses a mask, and softmax with cross-entropy uses p - y. The full softmax Jacobian for Llama 3.1 8B's vocabulary would have 16.4 billion entries per token.
- **"The backward pass costs about the same as the forward pass."** For matrix multiplies, nearly all the work, it costs twice as much: one product for the inputs' gradient and one for the weights'. A training step costs 6 FLOPs per weight per token, against 2 for inference.
- **"Training memory is the weights plus the optimizer's state."** Activations are a third bill, and it grows with every token in the batch. Without recomputation, GPT-3 would need 275 GB of them for one 2,048-token sequence.
- **"If a value is used twice, the second use overwrites its gradient."** The gradients from every use add. In $$q = ab + a$$ the gradient for $$a$$ is $$b + 1$$, and the `+=` in Step 6's `backward` is what makes that happen.

## Key takeaways

- Backpropagation applies one rule at every node of a computational graph: an input's gradient is the output's gradient times the local slope, summed over every use of the input.
- Reverse mode gets the gradient of one loss with respect to every weight in a single backward pass. Forward mode would need one pass per weight, 8 billion of them for Llama 3.1 8B.
- For Y equals X times W, the input gradient is the upstream gradient times W transposed, and the weight gradient is X transposed times the upstream gradient. Those two matmuls make the backward pass twice the forward, the 6 in 6ND.
- ReLU's backward pass is a mask, GELU's slope is Phi of z plus z times phi of z, and softmax with cross-entropy collapses to p minus y, which is why frameworks compute them together.
- The backward pass needs values saved by the forward pass, about 275 GB for one GPT-3 sequence. Checkpointing trades an extra forward pass for most of that memory.

## What's next

Backpropagation hands you a gradient, one number per weight, and stops. [Part 9](/blog/optimizers-adam/), Optimizers: from gradient descent to AdamW, decides what to do with it: why plain gradient descent struggles, how momentum and Adam's running averages smooth the noisy steps of mini-batches, why AdamW handles weight decay separately, and the warm-up, cosine schedules and gradient clipping that keep a months-long training run stable.

## Further reading

- [Learning representations by back-propagating errors (Rumelhart, Hinton & Williams, 1986)](https://www.nature.com/articles/323533a0): the four-page Nature paper that made backpropagation the standard way to train networks with hidden layers.
- [Automatic differentiation in machine learning: a survey (Baydin, Pearlmutter, Radul & Siskind, 2015)](https://arxiv.org/abs/1502.05767): forward and reverse mode side by side, with the adjoint notation used here.
- [Reducing Activation Recomputation in Large Transformer Models (Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198): section 4 derives the per-layer activation memory formula; section 5 introduces selective recomputation.
- [Training Deep Nets with Sublinear Memory Cost (Chen, Xu, Zhang & Guestrin, 2016)](https://arxiv.org/abs/1604.06174): activation checkpointing and the square-root memory schedule.
- [Gaussian Error Linear Units (GELUs) (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415): the GELU definition and its tanh approximation.
- [Autograd mechanics (PyTorch docs)](https://docs.pytorch.org/docs/2.14/notes/autograd.html) and [torch.utils.checkpoint (PyTorch docs)](https://docs.pytorch.org/docs/2.14/checkpoint.html): how PyTorch records the graph, saves tensors and recomputes them.
- [micrograd (Andrej Karpathy)](https://github.com/karpathy/micrograd): the tiny autograd engine that Step 6 follows, with a small neural network library on top.
- [The spelled-out intro to neural networks and backpropagation: building micrograd (Andrej Karpathy, 2022)](https://www.youtube.com/watch?v=VMj-3S1tku0): micrograd built live, one operation at a time.
- [Calculus on Computational Graphs: Backpropagation (Christopher Olah, 2015)](https://colah.github.io/posts/2015-08-Backprop/): pictures of forward-mode and reverse-mode differentiation on small graphs.
