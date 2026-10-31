---
title: "Derivatives, Gradients and the Chain Rule"
description: "The derivative built from the slope of a tiny secant, gradients as the uphill direction, the chain rule and Jacobians, checked with NumPy and PyTorch."
slug: "derivatives-gradients"
category: "llm-math"
tags: ["math", "training", "gradient-descent", "backpropagation"]
series: "llm-math"
seriesOrder: 7
pubDate: 2026-04-28
---

## TL;DR

A derivative is a slope: how far a function's output moves per unit of nudge to its input. This article builds it from the slope of a tiny secant line and applies the resulting rules to exp, ln, ReLU, sigmoid and SiLU. With many inputs the slopes form a vector, the gradient, and the dot product from Part 2 shows why it points straight uphill. The chain rule multiplies rates along a chain of functions; for vectors the rates are Jacobian matrices. Last come slopes measured by nudging, why that's hopeless for 8 billion weights, and a hand derivation checked against PyTorch.

## Why it matters

Llama 3.1 8B has 8,030,261,248 weights, and every training step needs a slope for each one: 8,030,261,248 numbers that say which way each weight should move. Measuring them by nudging each weight and rerunning the model would take 8 billion forward passes for a single step.

Calculus gets all of them for the price of about three forward passes, roughly 2.7 billion times less work, and that is what makes training a large network possible.

[Fundamentals Part 3](/blog/neural-networks-basics/) explained gradient descent and backpropagation in words, with a one-weight example. This part builds the math underneath: what a derivative is, why the gradient points uphill, why rates multiply along a chain, and when a nudge can be trusted.

## The intuition

A car's speedometer shows a derivative. You could estimate speed the slow way: read the odometer, wait one second, read it again, and divide the change by one second. Shorter waits give better estimates of the speed right now, and the derivative is what the estimate settles on as the wait shrinks toward zero.

Training asks the same question of the loss, the number that scores how wrong a model is (in an LLM, the loss for one token is $$-\ln p$$ of the correct token, from [Part 4](/blog/probability-language-models/)). Turn one weight up a hair: how fast does the loss change?

The analogy breaks because a car's distance depends only on time, while a model's loss depends on billions of weights, each with its own "speed", and a training step changes all of them at once. Handling that takes the gradient.

## Step 1: the derivative is the slope of a tiny secant

### Slope with numbers

The slope of a straight line is rise over run: how much it goes up for each step to the right. A curve like $$f(x) = x^2$$ gets steeper as $$x$$ grows, so it has no single slope. You can still ask for its slope near one point, say $$x = 3$$, where $$f(3) = 9$$.

Take a second point a little to the right, at $$3 + h$$, where $$h$$ is a small step. The straight line through both points is called a secant, and its slope is rise over run:

$$
\frac{f(3 + h) - f(3)}{h}
$$

With $$h = 0.1$$: $$f(3.1) = 9.61$$, the rise is 0.61, and the slope is $$0.61 / 0.1 = 6.1$$. Shrink the step:

| Step h | f(3 + h) | Rise | Slope of the secant |
|---|---|---|---|
| 0.1 | 9.61 | 0.61 | 6.1 |
| 0.01 | 9.0601 | 0.0601 | 6.01 |
| 0.001 | 9.006001 | 0.006001 | 6.001 |

The slopes close in on 6. Algebra shows why. Expand $$(3 + h)^2 = 9 + 6h + h^2$$, subtract 9 and divide by $$h$$: the slope of the secant is exactly $$6 + h$$. As $$h$$ shrinks to zero, it becomes 6.

### From secant to derivative

That limit is the derivative of $$f$$ at 3. In general:

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

Read $$f'(x)$$ as "f prime of x", the slope of $$f$$ at the point $$x$$. The symbol $$\lim_{h \to 0}$$ means "the value this settles on as $$h$$ shrinks toward zero". Another way to write the derivative is $$\frac{df}{dx}$$, "change in f per change in x".

The derivative is also a local forecast. Zoom in close enough and a smooth curve looks like its tangent line, the straight line that touches it there with the same slope. So for a small step $$h$$:

$$
f(x + h) \approx f(x) + f'(x)\, h
$$

From $$x = 3$$, a step of 0.01 should raise $$x^2$$ by about $$6 \times 0.01 = 0.06$$; the true rise is 0.0601. Training relies on this forecast to predict what a small change to a weight does to the loss.

### The rules you need

A few rules cover nearly everything an LLM uses, each from the same expand-and-cancel move as the $$x^2$$ example.

- Power rule: the slope of $$x^n$$ is $$n x^{n-1}$$. Expanding $$(x + h)^n$$ gives $$x^n + n x^{n-1} h$$ plus terms with $$h^2$$ or higher, which vanish after dividing by $$h$$ and shrinking it. For $$x^2$$ that's $$2x$$, which is 6 at $$x = 3$$.
- Constant multiple: the slope of $$5f(x)$$ is $$5f'(x)$$.
- Sum rule: the slope of $$f(x) + g(x)$$ is $$f'(x) + g'(x)$$.
- Product rule: the slope of $$f(x)\,g(x)$$ is $$f'(x)\,g(x) + f(x)\,g'(x)$$: each factor's change times the other factor's size.
- Exponential: the slope of $$e^x$$ is $$e^x$$ itself, where $$e \approx 2.718$$. The secant slope is $$e^x (e^h - 1)/h$$, and $$(e^h - 1)/h$$ heads to 1: it's 1.0517 at $$h = 0.1$$ and 1.0005 at $$h = 0.001$$.
- Natural log: the slope of $$\ln x$$ is $$1/x$$, so the curve flattens as $$x$$ grows.

A secant slope with a tiny $$h$$ checks every rule:

```python
import numpy as np

def slope(f, x, h=1e-6):               # secant slope with a tiny h
    return (f(x + h) - f(x)) / h

checks = [  # (name, function, x, what the rule says)
    ("power:    x^3 at 2",        lambda x: x**3,          2.0, 3 * 2**2),
    ("constant: 5x^2 at 3",       lambda x: 5 * x**2,      3.0, 5 * 2 * 3),
    ("sum:      x^2 + x^3 at 2",  lambda x: x**2 + x**3,   2.0, 2 * 2 + 3 * 2**2),
    ("product:  x^2 * x^3 at 2",  lambda x: x**2 * x**3,   2.0, (2 * 2) * 2**3 + 2**2 * (3 * 2**2)),
    ("exp:      e^x at 1",        np.exp,                  1.0, np.exp(1.0)),
    ("ln:       ln x at 4",       np.log,                  4.0, 1 / 4),
]
for name, f, x, rule in checks:
    print(f"{name:26} secant {slope(f, x):9.5f}   rule {rule:9.5f}")
```

```text
power:    x^3 at 2         secant  12.00001   rule  12.00000
constant: 5x^2 at 3        secant  30.00001   rule  30.00000
sum:      x^2 + x^3 at 2   secant  16.00001   rule  16.00000
product:  x^2 * x^3 at 2   secant  80.00008   rule  80.00000
exp:      e^x at 1         secant   2.71828   rule   2.71828
ln:       ln x at 4        secant   0.25000   rule   0.25000
```

The product line is a fair test: $$x^2 \cdot x^3$$ is $$x^5$$, whose power-rule slope at 2 is $$5 \times 2^4 = 80$$. The leftovers in the secant column, like the 0.00008, are $$h$$ terms that haven't quite vanished; Step 5 comes back to them.

## Step 2: derivatives of the functions inside an LLM

### Exp and ln at the output

An LLM turns its final scores into probabilities with softmax, which is built from $$e^x$$ ([Part 5](/blog/softmax-math/) takes it apart), and training scores the correct token with $$-\ln p$$. By the log rule, the slope of that loss with respect to $$p$$ is $$-1/p$$. At $$p = 0.5$$ it's -2; at $$p = 0.01$$ it's -100. A model that gave the right token almost no probability gets a steep slope and a strong push.

### ReLU and its kink

ReLU (rectified linear unit) keeps positive inputs and turns negative ones into 0, so its slope is 1 on the right and 0 on the left. At exactly 0 the graph has a corner, a kink. The secant from the right has slope $$(\text{ReLU}(h) - 0)/h = 1$$ and the one from the left has slope 0, whatever the size of $$h$$. The two never agree, so strictly ReLU has no derivative at 0.

Software has to return something, and PyTorch 2.11 returns 0 (we checked `torch.relu` at 0.0). Its [autograd notes](https://docs.pytorch.org/docs/2.14/notes/autograd.html) name "`relu` or `sqrt` at `0`" as points where a function isn't differentiable and list the rules it follows there. An input landing on exactly 0.0 is rare, so the choice barely matters.

### Sigmoid: a slope made of itself

The sigmoid squeezes any number into the range 0 to 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Here $$z$$ is the input and $$\sigma$$ (the Greek letter sigma) names the function. Working out its slope takes the chain rule from Step 4, but the result is short, and a secant confirms it:

$$
\sigma'(z) = \sigma(z)\,\bigl(1 - \sigma(z)\bigr)
$$

In words: the sigmoid's slope is its output times one minus its output. At $$z = 0$$ the output is 0.5 and the slope is $$0.5 \times 0.5 = 0.25$$, its steepest. At $$z = 3$$ the output is 0.9526 and the slope only 0.0452: far from zero the curve is almost flat.

### SiLU: the product rule at work

SiLU (sigmoid linear unit), the activation in Llama's MLP that [Part 3](/blog/matrices-linear-maps/) introduced, is $$z$$ times $$\sigma(z)$$. The product rule gives its slope:

$$
\text{SiLU}'(z) = \sigma(z) + z\,\sigma(z)\bigl(1 - \sigma(z)\bigr)
$$

The first term is the slope of $$z$$ (which is 1) times $$\sigma(z)$$; the second is $$z$$ times the slope of $$\sigma$$. The slopes side by side, each checked against a secant in Python:

| Input z | -3 | -1 | 0 | 1 | 3 |
|---|---|---|---|---|---|
| ReLU slope | 0 | 0 | 0 (by convention) | 1 | 1 |
| Sigmoid slope | 0.0452 | 0.1966 | 0.25 | 0.1966 | 0.0452 |
| SiLU slope | -0.0881 | 0.0723 | 0.5 | 0.9277 | 1.0881 |

SiLU's slope is negative below z ≈ -1.28, the bottom of its dip, and peaks at about 1.0998 near z = 2.40. Unlike ReLU's, it is zero only at that one point.

## Step 3: partial derivatives and the gradient

### One data point, two knobs

Take the smallest model with two parameters: a line $$wx + b$$, with a weight $$w$$ and a bias $$b$$, fitted to one data point, $$x = 3$$ with the right answer $$y = 7$$. The squared-error loss is:

$$
L(w, b) = (wx + b - y)^2
$$

Start at $$w = 1$$, $$b = 1$$. The prediction is 4, the miss is $$4 - 7 = -3$$, and the loss is 9.

A partial derivative is the slope with respect to one input while the others stay still, written with a curly d: $$\partial L / \partial w$$ is "the partial derivative of L with respect to w". Nudge only $$w$$ by 0.001 and the loss drops to 8.982009, a secant slope of -17.991. Nudging only $$b$$ gives -5.999.

For exact values, write the miss as $$e = wx + b - y$$, so $$L = e^2$$. The miss changes by $$x$$ per unit of $$w$$ and by 1 per unit of $$b$$, so:

$$
\frac{\partial L}{\partial w} = 2e\,x = 2(-3)(3) = -18
$$

$$
\frac{\partial L}{\partial b} = 2e = 2(-3) = -6
$$

Stack the partial derivatives into a vector and you have the gradient, written $$\nabla L$$ ("nabla L" or "grad L"):

$$
\nabla L = \left[\frac{\partial L}{\partial w},\ \frac{\partial L}{\partial b}\right] = [-18,\ -6]
$$

A model with 8 billion weights has a gradient with 8 billion entries, one partial derivative per weight.

### Why the gradient points uphill

Now move both at once by a small step $$\mathbf{d} = [d_w, d_b]$$. Each contributes its slope times its change, and the contributions add:

$$
\Delta L \approx \frac{\partial L}{\partial w}\, d_w + \frac{\partial L}{\partial b}\, d_b = \nabla L \cdot \mathbf{d}
$$

The $$\Delta L$$ (delta L) is the change in the loss. The right-hand side is a dot product, so the result from [Part 2](/blog/vectors-dot-products/) applies: $$\nabla L \cdot \mathbf{d} = \lVert \nabla L \rVert \, \lVert \mathbf{d} \rVert \cos\theta$$, where $$\theta$$ is the angle between the step and the gradient.

For steps of a fixed length, only the cosine varies. It's 1 when the step points along the gradient, the steepest climb, at a rate equal to the gradient's length. It's -1 when the step points straight against it, the steepest descent. At 90 degrees it's 0 and, to first order, the loss doesn't change.

Here are steps of length 0.01 from $$(w, b) = (1, 1)$$, where the gradient's length is $$\sqrt{18^2 + 6^2} = 18.97$$:

| Step direction | Cosine with gradient | Predicted change | Actual change |
|---|---|---|---|
| Along the gradient | 1.000 | +0.1897 | +0.1907 |
| w only, +1 | -0.949 | -0.1800 | -0.1791 |
| b only, +1 | -0.316 | -0.0600 | -0.0599 |
| Perpendicular | 0.000 | 0.0000 | 0.0000 |
| Against the gradient | -1.000 | -0.1897 | -0.1887 |

Of all small steps of a given length, the one against the gradient lowers the loss most, and gradient descent takes that step.

### Walking downhill

With a learning rate $$\eta$$ (the Greek letter eta), a small positive number you choose, each step updates the parameters $$\boldsymbol{\theta}$$, here $$[w, b]$$:

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\, \nabla L(\boldsymbol{\theta})
$$

One data point has a whole line of perfect answers (any $$w$$ and $$b$$ with $$3w + b = 7$$), so this run averages the loss over three points on $$y = 2x + 1$$, whose one best answer is $$w = 2$$, $$b = 1$$.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])          # three data points on the line y = 2x + 1
y = np.array([3.0, 5.0, 7.0])

def loss_and_grad(w, b):
    err = w * x + b - y                     # how far off each prediction is
    loss = np.mean(err ** 2)                # mean squared error
    grad = np.array([np.mean(2 * err * x),  # dL/dw
                     np.mean(2 * err)])     # dL/db
    return loss, grad

for lr in [0.05, 0.15, 0.19]:
    p = np.array([1.0, 1.0])                # start at w = 1, b = 1
    for step in range(1, 1001):
        _, g = loss_and_grad(*p)
        p = p - lr * g                      # step against the gradient
        if step in (1, 2, 3, 10, 100, 1000):
            loss, _ = loss_and_grad(*p)
            print(f"lr={lr} step {step:4}: w={p[0]:<9.4g} b={p[1]:<9.4g} loss={loss:.3g}")
```

```text
lr=0.05 step    1: w=1.467     b=1.2       loss=0.941
lr=0.05 step    2: w=1.676     b=1.287     loss=0.201
lr=0.05 step    3: w=1.77      b=1.323     loss=0.0544
lr=0.05 step   10: w=1.856     b=1.326     loss=0.0153
lr=0.05 step  100: w=1.952     b=1.11      loss=0.00174
lr=0.05 step 1000: w=2         b=1         loss=6.11e-13
lr=0.15 step    1: w=2.4       b=1.6       loss=2.07
lr=0.15 step    2: w=1.48      b=1.18      loss=0.92
lr=0.15 step    3: w=2.1       b=1.438     loss=0.414
lr=0.15 step   10: w=1.874     b=1.249     loss=0.0106
lr=0.15 step  100: w=1.996     b=1.009     loss=1.26e-05
lr=0.15 step 1000: w=2         b=1         loss=1.31e-30
```

The starting loss is 4.67. At a learning rate of 0.05 it drops fast for three steps, then crawls along a long, shallow valley where the gradient is small. At 0.15 the first step overshoots (w jumps from 1 to 2.4, past the answer) and the next ones zigzag back, yet it gets there much sooner.

At 0.19 it falls apart:

```text
lr=0.19 step    1: w=2.773     b=1.76      loss=5.72
lr=0.19 step    2: w=0.8244    b=0.8835    loss=7.01
lr=0.19 step    3: w=2.998     b=1.821     loss=8.6
lr=0.19 step   10: w=-0.4309   b=0.2063    loss=35.9
lr=0.19 step  100: w=-2.311e+04 b=-1.016e+04 loss=3.53e+09
lr=0.19 step 1000: w=-2.134e+44 b=-9.386e+43 loss=3.01e+89
```

Each step overshoots by more than the last. Trying values puts the edge for this loss at about 0.1803: below it the zigzags shrink, above it they grow. The forecast $$\Delta L \approx \nabla L \cdot \mathbf{d}$$ holds only for small steps, and how small depends on how sharply the bowl curves, which [Part 9](/blog/optimizers-adam/) makes precise.

## Step 4: the chain rule

### Rates multiply

Picture three gears in a row. If the middle gear turns 3 times per turn of the first, and the last turns 2 times per turn of the middle one, the last turns $$3 \times 2 = 6$$ times per turn of the first.

Functions chain the same way. Take $$y = (3x + 1)^2$$, an inner step $$u = 3x + 1$$ followed by an outer step $$y = u^2$$. At $$x = 1$$, $$u = 4$$. The inner rate is $$du/dx = 3$$; the outer rate is $$dy/du = 2u = 8$$. So:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 8 \times 3 = 24
$$

Expanding gives $$9x^2 + 6x + 1$$, whose slope $$18x + 6$$ is 24 at $$x = 1$$, and a secant with $$h = 10^{-6}$$ gives 24.000009. The chain rule in words: the rate of a chain is the product of its links' rates, each measured where that link is working.

### The loss of a two-token model

A model with only two tokens in its vocabulary turns a score $$z$$ into the probability of the correct token with a sigmoid, which is what softmax reduces to with two choices (Part 5). Its loss $$L = -\ln \sigma(z)$$ has two links: $$p = \sigma(z)$$, then $$L = -\ln p$$.

At $$z = 1$$: $$p = 0.7311$$ and $$L = 0.3133$$. The outer rate is $$-1/p = -1.3679$$. The inner rate is $$p(1 - p) = 0.1966$$. Their product is -0.2689. Symbolically the $$p$$ cancels:

$$
\frac{dL}{dz} = -\frac{1}{p} \cdot p\,(1 - p) = p - 1
$$

The slope is just $$0.7311 - 1 = -0.2689$$, and a secant agrees. [Fundamentals Part 3](/blog/neural-networks-basics/) stated this "probability minus one" for an LLM's output; [Part 8](/blog/backpropagation-math/) derives the full softmax version.

### A two-layer network by hand

Now a network with one input $$x = 1.5$$, two weights, a SiLU in the middle and the two-token loss at the end:

```text
z1 = w1 × x        = 0.8 × 1.5          =  1.2
h  = SiLU(z1)      = 1.2 × σ(1.2)       =  0.9222
z2 = w2 × h        = -0.5 × 0.9222      = -0.4611
p  = σ(z2)                              =  0.3867
L  = -ln p                              =  0.9501
```

The model gives the right answer a 38.67% chance. Walk back from the loss, multiplying one local rate at a time:

- Loss per unit of $$z_2$$: $$p - 1 = -0.6133$$ (the two-token result above).
- Weight $$w_2$$: $$z_2$$ changes by $$h$$ per unit of $$w_2$$, so $$\partial L / \partial w_2 = -0.6133 \times 0.9222 = -0.5656$$.
- Hidden value $$h$$: $$z_2$$ changes by $$w_2$$ per unit of $$h$$, so the rate is $$-0.6133 \times (-0.5) = 0.3066$$.
- Through the SiLU: its slope at 1.2 is 0.9820, so the rate for $$z_1$$ is $$0.3066 \times 0.9820 = 0.3011$$.
- Weight $$w_1$$: $$z_1$$ changes by $$x = 1.5$$ per unit of $$w_1$$, so $$\partial L / \partial w_1 = 0.3011 \times 1.5 = 0.4517$$.

![A top-to-bottom chain for a two-weight network. w1 = 0.8, with input x = 1.5, leads to z1 = w1 × x = 1.2, with local rate x = 1.5. That leads to h = SiLU(z1) = 0.9222, with rate SiLU'(1.2) = 0.982. h and a separate box w2 = -0.5 both feed z2 = w2 × h = -0.4611, with rates w2 = -0.5 and h = 0.9222. Then p = sigmoid(z2) = 0.3867 with rate p(1 - p) = 0.2372, then loss = -ln p = 0.9501 with rate -1/p = -2.586. A final box says dL/dw1 is the product of the five rates, 0.4517.](/diagrams/m07-chain-rule-rates.svg "The chain rule on a tiny network: each arrow carries a local rate, and a weight's slope is the product of the rates on its path to the loss.")

The gradient for $$w_2$$ is negative, so gradient descent raises $$w_2$$. The gradient for $$w_1$$ is positive, so it lowers $$w_1$$: while $$w_2$$ is negative, a bigger hidden value hurts. Nudging agrees to four decimals (0.4517 and -0.5656). One step with a learning rate of 0.5 moves the weights to 0.574 and -0.217, raising $$p$$ to 0.467 and cutting the loss from 0.950 to 0.761.

### Jacobians: the chain rule for vectors

Real layers map vectors to vectors, and every output can respond to every input. The grid of all those partial derivatives is the Jacobian matrix $$J$$, with one row per output and one column per input. Entry $$J_{ij}$$ is $$\partial y_i / \partial x_j$$, how much output $$i$$ moves per unit of input $$j$$.

For example, take $$\mathbf{f}(a, b) = [ab,\ a + b^2]$$ at $$(a, b) = (2, 3)$$. The first output changes at rate $$b = 3$$ with $$a$$ and $$a = 2$$ with $$b$$; the second at rate 1 with $$a$$ and $$2b = 6$$ with $$b$$:

$$
J = \begin{bmatrix} 3 & 2 \\ 1 & 6 \end{bmatrix}
$$

It plays the slope's role in the local forecast: a small input change $$\mathbf{d}$$ changes the outputs by about $$J\mathbf{d}$$.

For a linear layer $$\mathbf{y} = W\mathbf{x}$$, output $$i$$ is $$\sum_j W_{ij} x_j$$, whose rate with respect to $$x_j$$ is $$W_{ij}$$. So a linear layer's Jacobian is its weight matrix, which fits [Part 3](/blog/matrices-linear-maps/): a linear map is its own straight-line forecast everywhere. An activation applied number by number, like SiLU, has a diagonal Jacobian, because each output depends only on its own input.

Chain two layers and the Jacobians multiply, in the order the matrices would compose. For $$\mathbf{y} = W_2\, \text{SiLU}(W_1\mathbf{x})$$:

$$
J = W_2 \cdot \text{diag}\bigl(\text{SiLU}'(\mathbf{z})\bigr) \cdot W_1
$$

Here $$\mathbf{z} = W_1\mathbf{x}$$ and $$\text{diag}(\cdot)$$ puts numbers on the diagonal of a square matrix. PyTorch can build the Jacobian for comparison:

```python
import torch
from torch.autograd.functional import jacobian
torch.set_printoptions(precision=4, sci_mode=False)

W1 = torch.tensor([[1.0, -1.0, 0.5],
                   [0.0,  2.0, 1.0]])         # layer 1: 3 numbers in, 2 out
W2 = torch.tensor([[2.0, -1.0],
                   [1.0,  1.0]])              # layer 2: 2 in, 2 out
x = torch.tensor([1.0, 1.0, 2.0])

layer1 = lambda v: W1 @ v
print(torch.equal(jacobian(layer1, x), W1))  # a linear layer's Jacobian is W itself

net = lambda v: W2 @ torch.nn.functional.silu(W1 @ v)
z = W1 @ x
s = torch.sigmoid(z)
silu_slope = s + z * s * (1 - s)              # SiLU'(z), one per hidden number
J_by_hand = W2 @ torch.diag(silu_slope) @ W1  # multiply the three Jacobians
print(silu_slope)
print(J_by_hand)
print(jacobian(net, x))
```

```text
True
tensor([0.9277, 1.0527])
tensor([[ 1.8553, -3.9607, -0.1250],
        [ 0.9277,  1.1777,  1.5165]])
tensor([[ 1.8553, -3.9607, -0.1250],
        [ 0.9277,  1.1777,  1.5165]])
```

When the last link is a loss, a single number, its Jacobian is one row: the gradient. Multiplying that row into the Jacobians from the loss end backward keeps every intermediate result a row vector. That order of multiplication is backpropagation, the subject of [Part 8](/blog/backpropagation-math/). The full Jacobian of Llama 3.1 8B's final 4,096-number vector with respect to all its weights would have 32.9 trillion entries, about 132 TB in 32-bit floats; backpropagation never builds it.

## Step 5: numerical versus analytic derivatives

### Forward and central differences

A slope derived by rule is an analytic derivative; a secant gives a numerical one, with an error you can predict. For $$x^3$$ at 2, the forward difference expands to:

$$
\frac{(2 + h)^3 - 8}{h} = 12 + 6h + h^2
$$

The error, $$6h + h^2$$, shrinks in step with $$h$$: ten times smaller $$h$$, ten times smaller error, written $$O(h)$$, "of order h". The central difference steps both ways from the point:

$$
\frac{f(x + h) - f(x - h)}{2h}
$$

For $$x^3$$ at 2 it gives $$12 + h^2$$, because the $$6h$$ terms from the two sides cancel. Its error is $$O(h^2)$$: ten times smaller $$h$$, a hundred times smaller error, for two function evaluations instead of one.

### Float precision sets a floor

By the algebra, smaller $$h$$ is always better, but floats keep only so many digits. A 64-bit float keeps about 16 significant digits (its machine epsilon, the gap between 1 and the next number it can store, is $$2.2 \times 10^{-16}$$); a 32-bit float keeps about 7 ($$1.2 \times 10^{-7}$$). Subtracting $$f(x)$$ from $$f(x + h)$$ cancels the matching leading digits, and dividing by a tiny $$h$$ magnifies the rounding error that's left.

Here are the errors for the two-token loss $$-\ln \sigma(z)$$ at $$z = 1$$, whose exact slope is $$\sigma(1) - 1 = -0.268941\ldots$$:

```python
import numpy as np

def loss(z):                                  # -ln sigmoid(z), the two-token loss
    return -np.log(1 / (1 + np.exp(-z)))

exact = 1 / (1 + np.exp(-1.0)) - 1            # sigma(1) - 1 = -0.268941...
print(f"{'h':>7} {'forward f64':>12} {'central f64':>12} {'central f32':>12}")
for k in range(1, 13):
    h = 10.0 ** -k
    fwd = (loss(1 + h) - loss(1)) / h
    cen = (loss(1 + h) - loss(1 - h)) / (2 * h)
    z32, h32 = np.float32(1), np.float32(h)
    cen32 = (loss(z32 + h32) - loss(z32 - h32)) / (2 * h32)
    print(f"{h:7.0e} {abs(fwd - exact):12.1e} {abs(cen - exact):12.1e} {abs(cen32 - exact):12.1e}")
```

```text
      h  forward f64  central f64  central f32
  1e-01      9.7e-03      1.5e-04      1.5e-04
  1e-02      9.8e-04      1.5e-06      2.2e-06
  1e-03      9.8e-05      1.5e-08      2.0e-05
  1e-04      9.8e-06      1.5e-10      1.2e-04
  1e-05      9.8e-07      8.7e-13      5.2e-03
  1e-06      9.8e-08      9.8e-11      4.5e-02
  1e-07      1.1e-08      5.7e-10      3.3e-01
  1e-08      7.8e-09      5.4e-10      2.7e-01
  1e-09      3.3e-09      3.3e-09      2.7e-01
  1e-10      4.4e-07      4.4e-07      2.7e-01
  1e-11      1.0e-06      4.6e-06      2.7e-01
  1e-12      1.0e-05      1.0e-05      2.7e-01
```

The forward difference's error drops one digit per row, the $$O(h)$$ pattern, until about $$h = 10^{-7}$$. The central difference drops two per row, the $$O(h^2)$$ pattern, down to $$8.7 \times 10^{-13}$$ at $$h = 10^{-5}$$; after that, rounding takes over and the error climbs. In 32-bit floats the best is about $$2 \times 10^{-6}$$, at $$h = 0.01$$. By $$h = 10^{-8}$$ the two nudged inputs round to the same float and the "slope" comes out as 0, wrong by 100%.

Each column falls and then rises again, with a best $$h$$ that depends on the float format. LLMs usually train in BF16, a 16-bit format whose machine epsilon is 0.0078. Run the same central difference in BF16 and the best result is off by 0.5% (at h = 0.2); at h = 0.001 it returns 0.

### Why nobody trains this way

A numerical gradient needs one extra forward pass per parameter, two with central differences: for Llama 3.1 8B, 8.03 billion or 16.06 billion forward passes per training step.

Backpropagation needs one forward and one backward pass, and the [scaling-laws paper](https://arxiv.org/abs/2001.08361) puts the backward pass at "approximately twice the compute as the forwards pass". That's about three forward passes against 8.03 billion, a ratio of roughly 2.7 billion. Numerical gradients are left with one job: checking analytic ones.

### Autograd does the hand derivation

PyTorch's autograd records every operation as it runs and, per its [documentation](https://docs.pytorch.org/docs/2.14/notes/autograd.html), traces that graph "from roots to leaves" to "automatically compute the gradients using the chain rule". Here it is on the network from Step 4, followed by `torch.autograd.gradcheck`, which compares autograd's answer with finite differences:

```python
import torch
import torch.nn.functional as F

x = torch.tensor(1.5)
w1 = torch.tensor(0.8, requires_grad=True)    # ask PyTorch to track these two
w2 = torch.tensor(-0.5, requires_grad=True)

h = F.silu(w1 * x)                            # hidden value
p = torch.sigmoid(w2 * h)                     # probability of the right answer
loss = -torch.log(p)
loss.backward()                               # chain rule, applied automatically
print(f"loss={loss.item():.4f} dL/dw1={w1.grad.item():.4f} dL/dw2={w2.grad.item():.4f}")

# gradcheck compares backward() with central differences, in float64
def net(w):
    return -torch.log(torch.sigmoid(w[1] * F.silu(w[0] * 1.5)))
w = torch.tensor([0.8, -0.5], dtype=torch.float64, requires_grad=True)
print(torch.autograd.gradcheck(net, (w,)))
```

```text
loss=0.9501 dL/dw1=0.4517 dL/dw2=-0.5656
True
```

Autograd matches the hand-derived 0.4517 and -0.5656. The check runs in float64 because, as the error table showed, a float32 numerical slope is good to only a few digits.

## In the real world

- A gradient has the same shape as the weights. Llama 3.1 8B's has 8,030,261,248 entries ([Part 1](/blog/math-for-llms/) counts them from the config), and the gradient of a 4,096 × 14,336 MLP matrix is itself a 4,096 × 14,336 matrix. In a 16-bit format it takes 16.06 GB, as much as the weights.
- Learning rates shrink as models grow. The [Llama 3 paper](https://arxiv.org/abs/2407.21783) lists peak learning rates of $$3 \times 10^{-4}$$ for the 8B model, $$1.5 \times 10^{-4}$$ for 70B and $$8 \times 10^{-5}$$ for 405B (Table 3). The 405B run used AdamW with "a linear warm up of 8,000 steps" and a cosine schedule decaying to $$8 \times 10^{-7}$$ over 1,200,000 steps. AdamW rescales each slope before stepping, so these rates aren't directly comparable with Step 3's; [Part 9](/blog/optimizers-adam/) explains.
- PyTorch's [gradcheck](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.gradcheck.gradcheck.html) does exactly what Step 5 described: it will "check gradients computed via small finite differences against analytical gradients". Its defaults are `eps=1e-06`, `atol=1e-05` and `rtol=0.001`, which "are designed for input of double precision"; the check "will likely fail if input is of less precision".

## Common misconceptions

- **"The gradient points toward the minimum."** It points in the direction of steepest climb from where you stand, and its opposite is steepest descent from there, which is usually not the way to the bottom. In the three-point example the best bias is 1, yet at a learning rate of 0.05 the first three steps pushed b from 1 up to 1.32, and after 100 steps it was still at 1.11.
- **"A smaller h always gives a more accurate numerical derivative."** Only until rounding takes over. In float64 the central difference is best near h = 0.00001, in float32 near h = 0.01, and at h = 0.00000001 float32 returns exactly 0.
- **"Backpropagation is a different kind of math from the chain rule."** It is the chain rule, multiplied out from the loss backward so every intermediate result is a vector instead of a Jacobian matrix.
- **"A bigger learning rate always trains faster."** Only up to a point. In the three-point example 0.15 beat 0.05, but 0.19 sent the loss from 4.67 to $$3 \times 10^{89}$$ in 1,000 steps.

## Key takeaways

- A derivative is the slope a secant settles on as its step shrinks to zero. For x squared at 3, the secant slope is exactly 6 plus h, so the derivative is 6.
- A few rules cover an LLM: power, sum, constant, product, e to the x and ln. Sigmoid's slope is sigma times one minus sigma, and minus ln of a sigmoid has slope p minus 1.
- The gradient holds one partial derivative per weight. A small step changes the loss by the gradient dotted with the step, so the gradient is the steepest way up and its opposite the steepest way down.
- The chain rule multiplies rates along a chain. For vectors the rates are Jacobian matrices, a linear layer's Jacobian is its weight matrix, and multiplying from the loss end backward is backpropagation.
- Numerical gradients are for checking. Central differences have error of order h squared but hit a rounding floor, and for Llama 3.1 8B they would cost 8 billion forward passes against backpropagation's three.

## What's next

This part worked out a few slopes by hand. [Part 8](/blog/backpropagation-math/), Backpropagation by hand, turns the chain rule into an algorithm: a computational graph, a backward pass for matrix multiplication, ReLU, GELU and softmax with cross-entropy (where p minus 1 comes back), why the backward pass costs about twice the forward, and a tiny autograd written in plain Python.

## Further reading

- [Mathematics for Machine Learning, chapter 5 (Deisenroth, Faisal & Ong, 2020)](https://mml-book.github.io/): a free textbook chapter on vector calculus, from derivatives to Jacobians and backpropagation.
- [Automatic differentiation in machine learning: a survey (Baydin, Pearlmutter, Radul & Siskind, 2015)](https://arxiv.org/abs/1502.05767): how autograd differs from numerical and symbolic differentiation, and why reverse mode suits a scalar loss.
- [torch.autograd.gradcheck (PyTorch docs)](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.gradcheck.gradcheck.html): finite-difference gradient checking, with its defaults and the float64 warning.
- [Autograd mechanics (PyTorch docs)](https://docs.pytorch.org/docs/2.14/notes/autograd.html): how PyTorch records the graph, applies the chain rule, and what it returns at points like ReLU's kink.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): Table 3 and section 3.4.1 give the learning rates and schedule used for the 8B, 70B and 405B models.
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361): section 2.1 counts the forward and backward compute behind the 6N rule.
- [Essence of calculus (3Blue1Brown)](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr): animated lessons on derivatives as rates; [Visualizing the chain rule and product rule](https://www.3blue1brown.com/lessons/chain-rule-and-product-rule) matches Steps 1 and 4.
- [The spelled-out intro to neural networks and backpropagation: building micrograd (Andrej Karpathy, 2022)](https://www.youtube.com/watch?v=VMj-3S1tku0): derivatives by nudging, then the chain rule, then a working autograd, all in plain Python.
