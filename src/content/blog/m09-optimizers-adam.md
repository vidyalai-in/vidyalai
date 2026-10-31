---
title: "Optimizers: From Gradient Descent to AdamW"
description: "Why gradient descent zig-zags, and how momentum, Adam's bias correction, AdamW's weight decay, warm-up, cosine decay, clipping and Muon fix it."
slug: "optimizers-adam"
category: "llm-math"
tags: ["math", "training", "gradient-descent", "optimizers"]
series: "llm-math"
seriesOrder: 9
pubDate: 2026-05-04
---

## TL;DR

An optimizer turns a gradient into a step. Plain gradient descent uses one learning rate for every weight, capped by the steepest direction of the loss, so it zig-zags across narrow valleys and crawls along them. This article works through the fixes: mini-batches, momentum, RMSProp and Adam with its bias correction, AdamW's decoupled weight decay, then the warm-up, cosine decay and gradient clipping that LLM runs wrap around AdamW, and Muon, a newer optimizer that treats each weight matrix as a whole. Every idea is checked in NumPy on toy problems you can run.

## Why it matters

Training Llama 3.1 8B with AdamW takes about 128 GB of memory for weights, gradients and optimizer state alone, eight times the 16 GB the same weights need to answer a prompt. Most of the difference comes from two numbers Adam keeps for every one of the 8 billion weights.

In exchange, Adam is hard to break. On a bowl that is 50 times steeper in one direction than the other, plain gradient descent needs 223 steps, and a learning rate 5% higher makes it blow up. Adam converges at any learning rate from 0.1 to 5, and doesn't care if the loss is multiplied by a thousand.

[Part 7](/blog/derivatives-gradients/) introduced gradient descent and showed it diverging. [Fundamentals Part 3](/blog/neural-networks-basics/) described Adam in words, and [Fundamentals Part 12](/blog/pretraining-scaling-laws/) showed the recipe labs use. This part derives why each piece of that recipe is there.

## The intuition

Picture a ball rolling down a long, narrow canyon. Gradient descent is a ball with no mass: at every instant it moves in whatever direction is steepest right where it is. In a narrow canyon that direction points mostly at the opposite wall, so it bounces from wall to wall and drifts slowly down the floor.

Give the ball mass and it keeps some of its speed. The wall-to-wall pushes alternate and cancel, while the gentle slope along the floor always points the same way and builds up speed. That's momentum. Adam is more like a hiker who takes short steps where the ground is steep and long ones where it is flat.

The analogy breaks in two places: a model's loss has billions of directions, not two, and the optimizer never sees the canyon, only a noisy estimate of the slope from a small sample of data.

## Step 1: the speed limit of gradient descent

### One direction

Start with the simplest bowl, $$f(x) = \tfrac{1}{2} a x^2$$, where $$a$$ is a positive number that sets how steep it is. Its slope is $$a x$$, so one step of gradient descent with learning rate $$\eta$$ (eta) gives:

$$
x \leftarrow x - \eta\, a x = (1 - \eta a)\, x
$$

Every step multiplies $$x$$ by the same number, $$1 - \eta a$$. Between -1 and 1, $$x$$ shrinks toward the bottom at 0. Below -1, each step overshoots by more than the last.

With $$a = 50$$, a learning rate of 0.039 gives a multiplier of $$1 - 1.95 = -0.95$$: $$x$$ flips sign every step but shrinks by 5%. A learning rate of 0.041 gives -1.05, and $$x$$ grows by 5% per step forever. The edge is where the multiplier hits -1:

$$
\eta < \frac{2}{a}
$$

Here $$a$$ is the curvature, the slope of the slope (the second derivative). Sharply curved bowls force small learning rates.

This explains a number from Part 7, whose three-point line fit diverged above a learning rate of about 0.1803, found by trial. That loss curves most sharply in one direction, with curvature 11.09, and $$2 / 11.09 = 0.1803$$. (The curvatures of a loss in its different directions are the eigenvalues of its Hessian, the matrix of all its second derivatives.)

### Two directions: the zig-zag

Now a valley that curves 50 times more sharply across than along:

$$
f(x, y) = \tfrac{1}{2}\left(x^2 + 50\, y^2\right)
$$

Each direction has its own multiplier: $$1 - \eta$$ for $$x$$ and $$1 - 50\eta$$ for $$y$$. One learning rate has to serve both, and the steep $$y$$ direction caps it below $$2/50 = 0.04$$. Starting from (10, 1) with $$\eta = 0.039$$:

| Step | x | y | Loss |
|---|---|---|---|
| 0 | 10 | 1 | 75.0 |
| 1 | 9.61 | -0.95 | 68.7 |
| 2 | 9.235 | 0.9025 | 63.0 |
| 3 | 8.875 | -0.8574 | 57.8 |
| 4 | 8.529 | 0.8145 | 53.0 |

The $$y$$ coordinate bounces across the valley, losing only 5% per step. The $$x$$ coordinate creeps along it at 3.9% per step, because a rate that keeps $$y$$ stable is tiny for the gentle direction. Getting the loss below $$10^{-6}$$ takes 223 steps.

The ratio of the largest curvature to the smallest, here 50, is the condition number. For plain gradient descent, the number of steps grows roughly in proportion to it. Part 7's line fit has a condition number of 46, which is why it crawled after its first few steps.

## Step 2: mini-batches, noisy but cheap

The true gradient of an LLM's loss is an average over trillions of training tokens. In practice every optimizer estimates it from a mini-batch, a random sample, so each step's gradient is noisy. Plain gradient descent on mini-batches is SGD (stochastic gradient descent).

Here is a toy version: 100,000 points scattered around the line $$y = 2x + 1$$, and a model $$wx + b$$ starting at $$w = b = 0$$. The script compares the full-data gradient for $$w$$ with 5,000 mini-batch estimates at each batch size:

```python
import numpy as np
rng = np.random.default_rng(0)
N = 100_000                                   # training examples
x = rng.uniform(-1, 1, N)
y = 2 * x + 1 + rng.normal(0, 0.5, N)         # true line y = 2x + 1, plus noise

def grad_w(w, b, idx):                        # slope of the mean squared error w.r.t. w
    err = w * x[idx] + b - y[idx]
    return np.mean(2 * err * x[idx])

w, b = 0.0, 0.0
full = grad_w(w, b, np.arange(N))
print(f"full-batch gradient: {full:.4f}")
for B in [1, 10, 100, 1000]:
    est = np.array([grad_w(w, b, rng.integers(0, N, B)) for _ in range(5000)])
    print(f"B={B:5}  mean={est.mean():7.4f}  std={est.std():.4f}  variance*B={est.var() * B:.3f}")
```

```text
full-batch gradient: -1.3306
B=    1  mean=-1.3391  std=1.7485  variance*B=3.057
B=   10  mean=-1.3422  std=0.5512  variance*B=3.038
B=  100  mean=-1.3338  std=0.1713  variance*B=2.936
B= 1000  mean=-1.3307  std=0.0557  variance*B=3.100
```

At every batch size, the mean of the estimates matches the full gradient: the mini-batch gradient is unbiased, right on average. The variance (the average squared distance from that mean) times $$B$$ stays near 3, so:

$$
\text{Var}\big[\,\hat{g}_B\,\big] = \frac{\sigma^2}{B}
$$

Here $$\hat{g}_B$$ is the gradient estimated from $$B$$ examples and $$\sigma^2$$ is the variance of a single example's gradient, about 3 in this toy. The typical error, the square root of the variance, shrinks like $$1/\sqrt{B}$$.

That square root is why the noise is worth accepting. A batch of 100 costs a thousandth of the full 100,000, but its error is only $$\sqrt{1000} \approx 32$$ times larger, what the [Deep Learning book](https://www.deeplearningbook.org/contents/optimization.html) calls "less than linear returns to using more examples". [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) made it precise with the gradient noise scale: up to that batch size, bigger batches cut the number of steps almost proportionally; past it, they mostly waste compute. The noise scale grows as training goes on, one reason LLM batches grow mid-run.

Noise has a cost too. With a fixed learning rate, SGD never settles; it jitters around the bottom. Running the toy for 20,000 steps with batches of 100 and averaging over the last 10,000:

| Learning rate | Average excess loss above the best possible |
|---|---|
| 0.3 | 0.0014 |
| 0.1 | 0.00037 |
| 0.03 | 0.00010 |

Cutting the learning rate by about 3 cuts the jitter by about 3.7. That is why training ends with a decaying learning rate (Step 5).

## Step 3: momentum

### A moving average of gradients

Momentum steps along a running average of recent gradients, $$m$$, instead of the latest gradient:

$$
m_t = \beta\, m_{t-1} + (1 - \beta)\, g_t
$$

Here $$g_t$$ is the gradient at step $$t$$ and $$\beta$$ (beta) is a number just below 1, usually 0.9. The parameters then move by $$-\eta\, m_t$$.

Unroll it and each past gradient's weight shrinks by a factor of $$\beta$$ per step: with $$\beta = 0.9$$, the newest counts 0.1, the one before 0.09, then 0.081. This is an exponential moving average (EMA), and its effective window is about $$1/(1 - \beta)$$ steps. With $$\beta = 0.9$$ the last 10 gradients carry 65% of the weight; with $$\beta = 0.999$$ the last 1,000 carry 63%.

### Why the zig-zag cancels

In the valley, the $$y$$ gradient flips sign every step. Average +1, -1, +1, -1 with $$\beta = 0.9$$ and it settles at a size of $$(1 - \beta)/(1 + \beta) = 0.053$$, damped about 19-fold. The $$x$$ gradient keeps its sign, so it passes through at full strength.

With the bounce damped, the learning rate can grow. For this form with $$\beta = 0.9$$, the stability edge rises from 0.04 to $$2(1 + \beta) / \big((1 - \beta) \cdot 50\big) = 0.76$$, and the best rate in a sweep is 0.45. The valley then takes 97 steps instead of 223. Retuning to $$\beta = 0.6$$ gets it to 34, so the usual 0.9 is a safe default rather than the best value for every problem.

PyTorch's [`SGD`](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html) drops the $$(1 - \beta)$$ factor by default, which multiplies the effective learning rate by $$1/(1 - \beta)$$, ten times for 0.9: worth knowing when copying learning rates between papers.

## Step 4: Adam, a step size for every weight

### RMSProp: divide by the typical gradient size

Momentum still uses one learning rate for every weight. At the valley's start, the $$y$$ gradient is 50 and the $$x$$ gradient 10, close to the opposite of what each direction needs. RMSProp, from Geoffrey Hinton's [2012 Coursera lecture slides](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) (which credit Tijmen Tieleman), keeps a moving average of each weight's squared gradient:

$$
v_t = \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2
$$

It divides each weight's gradient by $$\sqrt{v_t}$$, the root mean square (RMS) of its recent gradients. A weight whose gradients run around 50 gets divided by about 50; one whose gradients run around 0.01 gets divided by about 0.01. Every weight steps by roughly the learning rate.

That also makes the step blind to the loss's scale. Multiply the loss by $$c$$ and every gradient grows by $$c$$, $$\sqrt{v}$$ grows by $$c$$, and the two cancel.

### Adam = momentum + RMSProp + bias correction

[Adam](https://arxiv.org/abs/1412.6980) (Kingma and Ba, 2014) keeps both averages, $$m_t$$ with $$\beta_1$$ and $$v_t$$ with $$\beta_2$$. Both start at 0, which causes trouble in the first steps. Feed it the same gradient, 4, every step:

```python
import numpy as np
b1, b2 = 0.9, 0.999
g = 4.0                                   # the same gradient every step
m = v = 0.0
print(" t   m_t     v_t      m_hat  v_hat   step/lr raw  step/lr corrected")
for t in range(1, 1001):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2
    m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
    if t in (1, 2, 3, 10, 100, 1000):
        print(f"{t:4} {m:6.3f} {v:8.4f} {m_hat:7.3f} {v_hat:6.2f} {m / np.sqrt(v):10.3f} {m_hat / np.sqrt(v_hat):12.3f}")
```

```text
 t   m_t     v_t      m_hat  v_hat   step/lr raw  step/lr corrected
   1  0.400   0.0160   4.000  16.00      3.162        1.000
   2  0.760   0.0320   4.000  16.00      4.250        1.000
   3  1.084   0.0480   4.000  16.00      4.950        1.000
  10  2.605   0.1593   4.000  16.00      6.528        1.000
 100  4.000   1.5233   4.000  16.00      3.241        1.000
1000  4.000  10.1169   4.000  16.00      1.258        1.000
```

After one step, $$m_1 = 0.1 \times 4 = 0.4$$, a tenth of the true gradient, and $$v_1 = 0.001 \times 16 = 0.016$$, a thousandth of the true squared gradient. Both are dragged toward their starting 0, and they recover at different speeds, so the raw step swings to 3.2 times the learning rate at step 1 and 6.5 times at step 10.

The fix comes from adding up the weights. With a constant gradient $$g$$:

$$
m_t = (1 - \beta)\left(1 + \beta + \dots + \beta^{t-1}\right) g = (1 - \beta^t)\, g
$$

The bracket is a geometric series that sums to $$(1 - \beta^t)/(1 - \beta)$$. At $$t = 2$$: $$(1 - 0.81) \times 4 = 0.76$$, as in the table. The average is short by exactly a factor of $$1 - \beta^t$$, so Adam divides it out:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \alpha\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Here $$\theta$$ (theta) stands for the parameters, $$\alpha$$ (alpha) is the learning rate, and $$\epsilon$$ (epsilon) is a tiny number that prevents division by zero, all applied to each weight separately. The paper's suggested defaults are $$\alpha = 0.001$$, $$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$ and $$\epsilon = 10^{-8}$$.

As $$t$$ grows, $$\beta^t$$ heads to 0 and the correction fades. At step 1, $$\hat{m}_1 = g_1$$ and $$\hat{v}_1 = g_1^2$$, so every weight moves by exactly $$\alpha$$ against its gradient's sign, whatever the gradient's size. The paper describes the step as "approximately bounded by the stepsize setting α", a trust region: Adam's learning rate caps how far a weight moves, where gradient descent's multiplies the gradient.

### Three optimizers on one valley

Here are gradient descent, momentum and Adam on the valley from Step 1, counting steps until the loss is below $$10^{-6}$$:

```python
import numpy as np

def steps_to_converge(opt, lr, scale=1.0, max_steps=10_000):
    curv = scale * np.array([1.0, 50.0])     # the valley f = 0.5 * (x^2 + 50 y^2), times scale
    p = np.array([10.0, 1.0])                 # start far along the flat direction
    m = np.zeros(2); v = np.zeros(2)
    for t in range(1, max_steps + 1):
        g = curv * p                          # gradient
        if opt == "gd":
            p = p - lr * g
        elif opt == "momentum":
            m = 0.9 * m + 0.1 * g             # velocity: moving average of gradients
            p = p - lr * m
        else:                                 # Adam
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            m_hat, v_hat = m / (1 - 0.9**t), v / (1 - 0.999**t)
            p = p - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        loss = 0.5 * np.sum(curv * p**2) / scale
        if not loss < 1e10:
            return "diverged"
        if loss < 1e-6:
            return t
    return f">{max_steps}"

for opt, lrs in [("gd", [0.01, 0.039, 0.041]),
                 ("momentum", [0.1, 0.45, 0.8]),
                 ("adam", [0.1, 1.0, 5.0])]:
    print(f"{opt:9}", "  ".join(f"lr={lr}: {steps_to_converge(opt, lr)}" for lr in lrs))

print("loss scaled by 0.001 and 1000, middle learning rate:")
for opt, lr in [("gd", 0.039), ("momentum", 0.45), ("adam", 1.0)]:
    print(f"{opt:9}", [steps_to_converge(opt, lr, s) for s in (0.001, 1000)])
```

```text
gd        lr=0.01: 882  lr=0.039: 223  lr=0.041: diverged
momentum  lr=0.1: 162  lr=0.45: 97  lr=0.8: diverged
adam      lr=0.1: 305  lr=1.0: 151  lr=5.0: 131
loss scaled by 0.001 and 1000, middle learning rate:
gd        ['>10000', 'diverged']
momentum  ['>10000', 'diverged']
adam      [151, 151]
```

Tuned momentum beats Adam here, 97 steps to 151. Adam's advantage is that it is hard to break: its learning rate can range over a factor of 50, and scaling the whole loss by 1,000 or 0.001 changes nothing, while the other two crawl on the shrunken loss and explode on the enlarged one. An LLM's gradients can differ in size by orders of magnitude between weight matrices and drift over months of training, and a large run can't afford to retune its learning rate every time that happens.

## Step 5: AdamW, schedules and clipping

![A top-to-bottom flow of one AdamW training step. The mini-batch gradient g is clipped by its global norm to a maximum length of 1.0. It then feeds two moving averages, m, the average of g with beta1 = 0.9, and v, the average of g squared with beta2 = 0.95. Both go through bias correction, dividing by 1 minus beta to the power t. Their ratio, m over the square root of v, gives a step direction for each weight. That is multiplied by the scheduled learning rate lr(t). Finally the weights take the step and each is shrunk by lr(t) times lambda times w.](/diagrams/m09-adamw-step.svg "One AdamW step as LLM training runs it.")

Clipping is the only stage that looks at all the gradients together; every stage after it works on each weight separately.

### Decoupled weight decay

Weight decay pulls every weight a little toward zero each step, so weights the loss doesn't need fade away. The classic way to get it is L2 regularization: add $$\tfrac{\lambda}{2}\lVert\theta\rVert^2$$ to the loss, where $$\lambda$$ (lambda) sets the strength and $$\lVert\theta\rVert^2$$ is the sum of every weight squared. Its gradient is $$\lambda\theta$$, so with plain SGD:

$$
\theta \leftarrow \theta - \eta\,(g + \lambda\theta) = (1 - \eta\lambda)\,\theta - \eta\, g
$$

Every weight shrinks by the same fraction $$\eta\lambda$$ before the usual update. For SGD, L2 regularization and weight decay are the same thing.

Under Adam they aren't, because the extra $$\lambda\theta$$ goes into the gradient and gets divided by $$\sqrt{\hat{v}}$$ like everything else. A weight whose gradients are typically 100 has its decay divided by about 100; one whose gradients are typically 0.01 has it multiplied by about 100. A simulation shows it: single weights starting at 1.0, zero-mean noisy gradients of three sizes, learning rate 0.001, $$\lambda = 0.1$$, 5,000 steps, averaged over 10,000 copies:

| Typical gradient size | Weight after 5,000 steps, L2 inside Adam | Weight after 5,000 steps, decoupled |
|---|---|---|
| 0.01 | 0.000 | 0.606 |
| 1 | 0.608 | 0.607 |
| 100 | 0.995 | 0.606 |

L2 inside Adam wipes out the quiet weight and barely touches the busy one; the two methods agree only when the gradient size happens to be 1. Decoupled decay shrinks all three by $$(1 - 0.001 \times 0.1)^{5000} = 0.607$$.

[Loshchilov and Hutter](https://arxiv.org/abs/1711.05101) named the problem: with L2, "weights that tend to have large gradients in f do not get regularized as much as they would with decoupled weight decay". Their fix, AdamW, applies the decay to the weights directly. In the form PyTorch implements:

$$
\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\, \theta_{t-1} \right)
$$

The $$\lambda\theta$$ term now skips the division by $$\sqrt{\hat{v}}$$. PyTorch's [`AdamW`](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) defaults to `weight_decay=0.01`, and its decay "does not accumulate in the momentum nor variance"; plain [`Adam`](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html) with `weight_decay` does the L2 version. GPT-3 and Llama 2 used $$\lambda = 0.1$$, and Llama 3's scaling-law runs did the same: "The weight decay at each step is set to 0.1 times the learning rate at that step."

### Warm-up, then cosine decay

LLM runs change the learning rate every step. The usual shape is a linear warm-up from 0 to a peak over $$W$$ steps:

$$
\eta(t) = \eta_{\text{peak}} \cdot \frac{t}{W}
$$

followed by a cosine decay to a final value:

$$
\eta(t) = \eta_{\text{final}} + \tfrac{1}{2}\left(\eta_{\text{peak}} - \eta_{\text{final}}\right)\left(1 + \cos \pi p\right)
$$

Here $$p$$ is the fraction of the decay phase completed. As $$p$$ goes from 0 to 1, $$\cos \pi p$$ goes from 1 to -1, the bracket from 2 to 0, and the rate slides from peak to final: slowly at first, fastest in the middle, slowly at the end.

The [Llama 3 paper](https://arxiv.org/abs/2407.21783) trained the 405B model with a peak of $$8 \times 10^{-5}$$, "a linear warm up of 8,000 steps", and a cosine "decaying to $$8 \times 10^{-7}$$ over 1,200,000 steps". Taking the cosine to start where warm-up ends (the paper doesn't say exactly), the formulas give:

| Step | Phase | Learning rate |
|---|---|---|
| 1,000 | warm-up | 1.0 × 10⁻⁵ |
| 8,000 | peak | 8.0 × 10⁻⁵ |
| 308,000 | cosine, 25% | 6.8 × 10⁻⁵ |
| 608,000 | cosine, 50% | 4.0 × 10⁻⁵ |
| 908,000 | cosine, 75% | 1.2 × 10⁻⁵ |
| 1,208,000 | end | 8.0 × 10⁻⁷ |

[Fundamentals Part 12](/blog/pretraining-scaling-laws/) has the schedule as code. The decay follows from Step 2: a smaller rate means a smaller jitter around the bottom.

The warm-up exists because Adam's first steps are its least informed: after a handful of steps, $$\hat{v}$$ averages only a handful of squared gradients. Simulating 100,000 weights whose gradients have a true value of 0.1 plus noise of size 1:

| Step | Average step size, as a multiple of the learning rate | Weights moving the right way |
|---|---|---|
| 1 | 1.000 | 54% |
| 2 | 0.640 | 56% |
| 10 | 0.279 | 62% |
| 100 | 0.200 | 67% |
| 1,000 | 0.200 | 67% |

At step 1 every weight takes a full-size step, five times what it will settle at, and barely half go the right way. Warm-up keeps the learning rate small while the estimates are poor. [Liu et al. (2019)](https://arxiv.org/abs/1908.03265) traced warm-up to exactly this variance; on one translation task, "removing the warmup stage increases the training loss from 3 to around 10". Older post-norm Transformers needed warm-up for a related reason ([Fundamentals Part 6](/blog/transformer-block/) and [Fundamentals Part 8](/blog/attention-is-all-you-need/)).

### Gradient clipping by global norm

Now and then a batch produces a huge gradient. Clipping caps its length. Treat every gradient in the model as one long vector $$\mathbf{g}$$, take its length $$\lVert\mathbf{g}\rVert$$ (the square root of the sum of all entries squared, from [Part 2](/blog/vectors-dot-products/)), and if it exceeds a limit $$c$$, shrink it:

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{c}{\lVert\mathbf{g}\rVert}
$$

Suppose a model has two weight tensors with gradients [3, 4] and [12]. The global norm is $$\sqrt{9 + 16 + 144} = 13$$. With $$c = 1$$, everything is multiplied by 1/13, giving [0.231, 0.308] and [0.923]: same direction, length 1. Clipping each entry to the range -1 to 1 instead would give [1, 1] and [1], a different direction that forgets which weight mattered most.

PyTorch's [`clip_grad_norm_`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html) computes the norm "as if the norms of the individual gradients were concatenated into a single vector". The usual limit is 1.0: [GPT-3](https://arxiv.org/abs/2005.14165) clipped "the global norm of the gradient at 1.0" and [Llama 2](https://arxiv.org/abs/2307.09288) used "gradient clipping of 1.0". Adam already normalizes step sizes, but a spike still gets squared into $$v$$, where it shrinks that weight's steps for many steps afterward; clipping stops one bad batch from doing that.

## Step 6: beyond Adam, orthogonalized updates with Muon

Adam treats a 4,096 × 14,336 weight matrix as 58.7 million unrelated numbers. Muon ("MomentUm Orthogonalized by Newton-Schulz"), proposed by [Keller Jordan in December 2024](https://kellerjordan.github.io/posts/muon/), treats it as a matrix. Its starting observation is that momentum updates for weight matrices "typically have very high condition number": they are "almost low-rank matrices, with the updates for all neurons being dominated by just a few directions".

The singular value decomposition, [Part 13](/blog/low-rank-svd/)'s subject, writes any matrix as $$M = U \Sigma V^{\top}$$: a rotation, a stretch along a few special directions by amounts called singular values, and another rotation. Muon keeps the rotations and sets every stretch to 1, replacing the momentum matrix $$M$$ by $$UV^{\top}$$:

$$
M_t = \mu M_{t-1} + G_t, \qquad W_t = W_{t-1} - \eta\, \text{NS}(M_t)
$$

Here $$G_t$$ is the gradient of the weight matrix $$W$$, $$\mu$$ (mu) is the momentum factor, and NS is a Newton-Schulz iteration that approximates $$UV^{\top}$$ with matrix multiplications alone, which GPUs do fast. Rarely used directions get as much of the update as dominant ones.

```python
import numpy as np
rng = np.random.default_rng(0)

def newton_schulz(M, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315       # coefficients from Keller Jordan's Muon
    X = M / np.linalg.norm(M)               # divide by the Frobenius norm: singular values <= 1
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X

# a momentum matrix dominated by one direction, as gradients often are
u, v = rng.standard_normal((4, 1)), rng.standard_normal((1, 6))
M = 10 * u @ v + rng.standard_normal((4, 6))
print("singular values of M:          ", np.linalg.svd(M, compute_uv=False).round(2))
print("after 5 Newton-Schulz steps:   ", np.linalg.svd(newton_schulz(M), compute_uv=False).round(2))
U, S, Vt = np.linalg.svd(M, full_matrices=False)
print("exact U V^T (all ones):        ", np.linalg.svd(U @ Vt, compute_uv=False).round(2))
O = newton_schulz(M)
cos = np.sum(O * (U @ Vt)) / (np.linalg.norm(O) * np.linalg.norm(U @ Vt))
print(f"cosine between the two updates: {cos:.3f}")
```

```text
singular values of M:           [16.56  2.84  1.79  0.43]
after 5 Newton-Schulz steps:    [1.04 0.97 0.75 0.72]
exact U V^T (all ones):         [1. 1. 1. 1.]
cosine between the two updates: 0.987
```

A spread of 16.56 to 0.43, a factor of 38, becomes 0.72 to 1.04 after five iterations. The post reports that singular values as far as about 0.3 from 1 don't hurt training, and puts Muon's extra FLOPs below 1% for typical language-model training. Muon is only for the 2D matrices inside the network; the post says "AdamW should be used for the embedding and final classifier head layers", and it is also used for norms and other vectors.

Moonshot AI's report [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) added weight decay and rescaled each update by $$0.2\sqrt{\max(A, B)}$$ for an A × B matrix, so Muon could reuse AdamW's learning rate and weight decay. Their scaling experiments found Muon "∼2× more computational efficient than Adam with compute optimal training". The same team's [Kimi K2](https://arxiv.org/abs/2507.20534), 1 trillion parameters with 32 billion active, trained on 15.5 trillion tokens with MuonClip, a Muon variant, "with zero loss spike".

## In the real world

- The [ZeRO paper](https://arxiv.org/abs/1910.02054) counts mixed-precision Adam at 16 bytes per parameter: 2 for 16-bit weights, 2 for 16-bit gradients, and 12 for "an fp32 copy of the parameters, momentum and variance". For Llama 3.1 8B's 8,030,261,248 parameters that's 16.06 + 16.06 + 32.12 + 32.12 + 32.12 = 128.5 GB, more than one 80 GB GPU holds before a single activation. [Fundamentals Part 12](/blog/pretraining-scaling-laws/) shows how ZeRO splits it across GPUs; [Part 14](/blog/floating-point-math/) explains the fp32 master copy.
- Cheaper optimizers attack those 12 bytes. [Adafactor](https://arxiv.org/abs/1804.04235) keeps "only the per-row and per-column sums" of the squared-gradient averages: for a 4,096 × 14,336 matrix, 18,432 numbers instead of 58,720,256. [Lion](https://arxiv.org/abs/2302.06675), found by automated program search, "only keeps track of the momentum" and steps by its sign. Muon, too, keeps one momentum buffer per matrix.
- Real recipes are AdamW with a shorter $$\beta_2$$. [GPT-3](https://arxiv.org/abs/2005.14165) used $$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$, $$\epsilon = 10^{-8}$$, weight decay 0.1, warm-up over the first 375 million tokens and cosine decay to 10% of the peak. [Llama 2](https://arxiv.org/abs/2307.09288) used $$\beta_2 = 0.95$$, $$\epsilon = 10^{-5}$$, 2,000 warm-up steps, decay to 10%, weight decay 0.1 and clipping at 1.0. A $$\beta_2$$ of 0.95 averages over about 20 steps instead of 1,000, so $$v$$ follows changes in gradient size much sooner.
- Learning rates aren't comparable across optimizers. [Part 7](/blog/derivatives-gradients/) listed Llama 3's peaks, $$3 \times 10^{-4}$$ for 8B down to $$8 \times 10^{-5}$$ for 405B; under Adam those are roughly the most any weight moves per step. LoRA fine-tuning typically uses 1e-4 to 2e-4 with AdamW, about 10 times the rate for full fine-tuning ([Fine-Tuning a Small Model with LoRA on 5,000 Examples](/blog/lora-fine-tuning-5k-examples/)).

## Common misconceptions

- **"Adam is always faster than SGD."** On the valley, tuned momentum converged in 97 steps and Adam in 151. Adam's advantage is that one learning rate works across a factor of 50 and any scaling of the loss, which makes it the safe default for LLMs.
- **"Weight decay is just L2 regularization."** Only for plain SGD. Inside Adam, the L2 gradient is divided by each weight's gradient size: in the simulation, a weight with gradients of size 100 kept 99.5% of its value while one with size 0.01 went to zero.
- **"Gradient clipping changes the direction of the update."** Clipping by global norm multiplies every gradient by the same number, so the direction stays the same; clipping each entry separately would change it.

## Key takeaways

- Gradient descent multiplies each direction by one minus the learning rate times its curvature, so the rate must stay below two over the steepest curvature. In a valley 50 times steeper one way, that leaves the gentle direction crawling: 223 steps.
- Mini-batch gradients are right on average, and their noise shrinks like one over the square root of the batch size. The leftover jitter scales with the learning rate, which is why the rate decays at the end.
- Momentum averages gradients over about one over one minus beta steps, ten for beta 0.9. Alternating gradients cancel and steady ones add up, cutting the valley from 223 steps to 97.
- Adam divides a moving average of each gradient by the root of a moving average of its square, after correcting both for starting at zero. Each step comes out near the learning rate, whatever the gradient's scale.
- AdamW decays weights directly instead of through the gradient. LLM runs add warm-up, cosine decay and clipping at a global norm of 1.0; Muon goes further by evening out the singular values of each matrix's update.

## What's next

Every optimizer here assumes the gradients reaching it are usable. In a network a hundred layers deep, they can shrink to nothing or blow up on the way down. [Part 10](/blog/initialization-normalization/), Keeping signals stable, shows how initialization, normalization and residual connections keep signals and gradients the right size from the first layer to the last.

## Further reading

- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980): the algorithm, its defaults, the bias-correction derivation (section 3) and the step-size bound (section 2.1).
- [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101): why L2 regularization and weight decay differ under Adam, with AdamW in Algorithm 2.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054): section 3.1 derives the 16 bytes per parameter of mixed-precision Adam.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): section 3.4.1 gives the 405B model's learning rate, warm-up, cosine schedule and batch sizes.
- [Muon is Scalable for LLM Training (Liu et al., 2025)](https://arxiv.org/abs/2502.16982): Muon's update rule, Newton-Schulz iteration, weight decay and the scaling comparison with AdamW.
- [An Empirical Model of Large-Batch Training (McCandlish et al., 2018)](https://arxiv.org/abs/1812.06162): the gradient noise scale and how it predicts the largest useful batch size.
- [On the Variance of the Adaptive Learning Rate and Beyond (Liu et al., 2019)](https://arxiv.org/abs/1908.03265): why Adam needs warm-up, traced to the variance of its early estimates.
- [Deep Learning, chapter 8 (Goodfellow, Bengio & Courville, 2016)](https://www.deeplearningbook.org/contents/optimization.html): SGD, momentum, RMSProp and Adam, with figure 8.5 showing momentum in a narrow valley.
- [Neural Networks for Machine Learning, lecture 6 (Hinton, Srivastava & Swersky, 2012)](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf): the slides that introduced RMSProp, alongside momentum.
- [Muon: An optimizer for hidden layers in neural networks (Keller Jordan, 2024)](https://kellerjordan.github.io/posts/muon/): the original post, with the Newton-Schulz code and speedrun results.
- [Why Momentum Really Works (Gabriel Goh, Distill, 2017)](https://distill.pub/2017/momentum/): interactive plots of momentum on ill-conditioned bowls, the picture behind Step 3.
