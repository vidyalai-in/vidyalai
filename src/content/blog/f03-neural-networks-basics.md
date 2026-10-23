---
title: "Neural Networks: Weights, Gradient Descent and Backpropagation"
description: "How a neural network learns: neurons as weighted sums, loss, gradient descent worked by hand, and backpropagation explained without calculus."
slug: "neural-networks-basics"
category: "llm-fundamentals"
tags: ["neural-networks", "training", "gradient-descent", "backpropagation", "gpu"]
series: "llm-fundamentals"
seriesOrder: 3
pubDate: 2026-10-23
---

## TL;DR

A neural network is a stack of simple arithmetic. Each "neuron" multiplies its inputs by weights, adds them up, and bends the total with an activation function, and a whole layer of neurons is one matrix multiplication, which is why GPUs run these models. To learn, the network scores its own error with a single number called the loss, works out which way each weight should move to shrink it, and moves every weight a little. Backpropagation finds that direction for every weight in one backward sweep. We'll work each piece by hand, then scale up to an LLM, which learns the same way.

## Why it matters

Every one of GPT-3's 175 billion parameters started as a random number. Training turned that noise into a model that writes code and explains history, and the procedure that did it fits in one paragraph: guess, measure how wrong the guess was, work out which way each number should move, move it a little, repeat.

[Part 1](/blog/how-llms-work/) said training "nudges every parameter" using an algorithm called backpropagation, and stopped there. The core idea was laid out in a four-page 1986 Nature paper by [Rumelhart, Hinton and Williams](https://www.nature.com/articles/323533a0), and the same idea trained Llama 3.1 405B with [3.8 × 10²⁵ arithmetic operations](https://arxiv.org/abs/2407.21783).

Knowing the loop also explains why training needs far more memory than running a model, and how a model can ace its training data and still fail on new text.

## The intuition

Picture a hiker in thick fog, trying to reach the bottom of a valley they can't see. All they can do is feel which way the ground slopes under their boots, take a step downhill, and feel again.

Training a neural network is that hike. The hiker's position is the current set of weights, and the height of the ground is the loss, a number that says how wrong the network's answers are. The slope under the boots is the gradient. The step length is the learning rate: too short and the hike takes forever, too long and the hiker jumps clean over the valley onto the opposite slope.

The analogy breaks in two places. The hiker has two directions to choose from, north-south and east-west, while an LLM has one direction per parameter, billions of them. And the ground shifts: each step measures the slope on a different small sample of training data.

## Step 1: a neuron is a weighted sum with a bend

A neuron takes a few numbers in and produces one number out. It does three things:

1. Multiply each input by its own weight, a number that says how much that input matters (negative weights push the other way).
2. Add up the results, plus one extra number called the bias, which shifts the total up or down.
3. Pass the total through an activation function, a fixed rule that bends it.

Weights and biases are the network's parameters, the numbers training adjusts. One neuron with three inputs:

| Quantity | Input 1 | Input 2 | Input 3 |
|---|---|---|---|
| Input value | 1.0 | 2.0 | 3.0 |
| Weight | 0.5 | 0.25 | -0.1 |
| Input × weight | 0.5 | 0.5 | -0.3 |

The weighted sum is `0.5 + 0.5 - 0.3 = 0.7`. Add a bias of 0.1 and the total is 0.8.

Now the bend. The default recommendation in the [Deep Learning textbook](https://www.deeplearningbook.org/contents/mlp.html) by Goodfellow, Bengio and Courville is the ReLU (rectified linear unit): keep the number if it's positive, replace it with 0 if it's negative. Our 0.8 stays 0.8. GPT models use a smoother cousin called GELU (Gaussian error linear unit); [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) used it in 2018. GELU multiplies the input by the chance that a bell-curve random number falls below it, [as defined in the GELU paper](https://arxiv.org/abs/1606.08415), so small negative numbers leak through a little instead of being cut to zero:

| Input to activation | -3 | -1 | -0.5 | 0 | 0.5 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| ReLU | 0 | 0 | 0 | 0 | 0.5 | 1 | 3 |
| GELU | -0.004 | -0.159 | -0.154 | 0 | 0.346 | 0.841 | 2.996 |

Without the bend, stacking layers would gain nothing. Multiply by 3, then by 2, and you've just multiplied by 6, so two layers collapse into one. A weighted sum of weighted sums is still a weighted sum, and a network with no activation functions can only draw straight lines however deep it is. The bends let a deep network build curved functions out of simple pieces.

## Step 2: a layer is a matrix multiplication

A layer is many neurons reading the same inputs, each with its own weights. Add a second neuron to our example, with weights 0.4, -0.6 and 0.1 and no bias:

```text
inputs:      x = [1.0, 2.0, 3.0]
neuron A:    0.5×1.0 + 0.25×2.0 + (-0.1)×3.0 + 0.1  =  0.8   → ReLU →  0.8
neuron B:    0.4×1.0 + (-0.6)×2.0 + 0.1×3.0  + 0.0  = -0.5   → ReLU →  0.0
```

Stack the two rows of weights into a grid, a matrix, and the whole layer is one operation, `output = ReLU(W × x + b)`, where `W` is the 2-by-3 weight matrix, `x` the inputs and `b` the biases. Every weight is used exactly once, so that's 6 multiply-adds.

A layer with `n` inputs and `m` outputs holds `n × m` weights and costs `n × m` multiply-adds for each input it processes. In GPT-3, each token's vector has 12,288 numbers, and the first layer of each block's MLP (the small feed-forward network inside every Transformer block, introduced in Part 1) widens it to [4 × 12,288 = 49,152](https://arxiv.org/abs/2005.14165):

```text
one MLP matrix:   12,288 × 49,152          ≈ 604 million multiply-adds per token
both MLP matrices, all 96 blocks:          ≈ 116 billion multiply-adds per token
whole forward pass (≈ one per parameter):  ≈ 175 billion multiply-adds per token
```

Because each parameter is used about once per token, a forward pass costs about 2 FLOPs (floating-point operations, here one multiply and one add) per parameter. That's the rule in the [scaling-laws paper](https://arxiv.org/abs/2001.08361), and for GPT-3 it comes to about 350 billion FLOPs per token.

Those 604 million multiply-adds don't depend on each other. None waits for another's result, so they can all run at the same time, and that is why GPUs matter. In [NVIDIA's words](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html), a CPU (central processing unit) is designed to run "a serial sequence of operations" as fast as possible, while a GPU (graphics processing unit), born as a chip for 3D graphics, is designed to run "thousands of threads in parallel". Matrix multiplication is exactly that kind of work. While training Llama 3.1 405B, Meta's GPUs each sustained [380 to 430 trillion FLOPs per second](https://arxiv.org/abs/2407.21783) on it. [How GPUs Work for LLMs](/blog/gpus-for-llms/), in the LLM Inference & Optimization series, looks inside the chip.

## Step 3: measure the error, then walk downhill

A network with random weights produces junk. To improve it, you first need a number that says how bad the junk is: the loss, computed by a loss function.

Take the smallest possible network: one weight, `w`, and no bias. It predicts `prediction = w × x`. We have one training example: when `x = 2`, the right answer is 6. (So the best weight is 3, but the network doesn't know that.) We start at `w = 1`.

For the loss we'll use squared error, `loss = (prediction - target)²`. With `w = 1` the prediction is 2, it misses by 4, and the loss is 16. Squaring keeps the loss positive whichever way you miss and punishes big misses more than small ones.

### The slope, found by nudging

To find which way `w` should move, nudge it and watch the loss. No calculus needed.

| `w` | Prediction | Loss |
|---|---|---|
| 1.00 | 2.00 | 16.0000 |
| 1.01 | 2.02 | 15.8404 |

Raising `w` by 0.01 lowered the loss by 0.1596, a slope of about `-0.1596 / 0.01 ≈ -16`. Near `w = 1`, each unit added to `w` takes about 16 off the loss. Shrink the nudge toward zero and the slope settles on exactly -16; calculus calls that limit the derivative. For squared error it has a simple formula, `slope = 2 × (prediction - target) × x`, and `2 × (2 - 6) × 2 = -16` agrees.

A negative slope means raising `w` lowers the loss, so `w` should go up. The update rule `new w = w - learning rate × slope` handles the sign for you: subtracting a negative number moves `w` up.

### Three steps by hand

With a learning rate of 0.05:

| Step | `w` | Prediction | Loss | Slope | New `w` |
|---|---|---|---|---|---|
| 1 | 1.000 | 2.00 | 16.0000 | -16.00 | 1 + 0.05 × 16 = 1.800 |
| 2 | 1.800 | 3.60 | 5.7600 | -9.60 | 1.8 + 0.05 × 9.6 = 2.280 |
| 3 | 2.280 | 4.56 | 2.0736 | -5.76 | 2.28 + 0.05 × 5.76 = 2.568 |

The miss goes from 4 to 2.4 to 1.44, so each step removes 40% of what's left, and the loss falls from 16 to about 2 in two steps. The steps shrink on their own, because the ground flattens near the bottom and a smaller slope means a smaller step.

Here's the same loop in Python:

```python
# One input, one weight: the model predicts y = w * x.
# We want it to learn that x = 2 should give y = 6 (so w should be 3).
x, target = 2.0, 6.0
w = 1.0              # starting guess
lr = 0.05            # learning rate: how big a step to take

for step in range(1, 4):
    pred = w * x
    loss = (pred - target) ** 2          # squared error
    grad = 2 * (pred - target) * x       # slope of the loss with respect to w
    w = w - lr * grad                    # step downhill
    print(f"step {step}: pred={pred:.2f} loss={loss:.4f} slope={grad:.2f} new w={w:.3f}")
```

```text
step 1: pred=2.00 loss=16.0000 slope=-16.00 new w=1.800
step 2: pred=3.60 loss=5.7600 slope=-9.60 new w=2.280
step 3: pred=4.56 loss=2.0736 slope=-5.76 new w=2.568
```

With many weights, you get one slope per weight. That list of slopes is the gradient, and following it downhill is gradient descent.

### The learning rate

You choose the learning rate; the network doesn't learn it. Here's the same example with three values:

| Learning rate | `w` after 1 step | after 2 | after 3 | after 5 | Loss after 5 steps |
|---|---|---|---|---|---|
| 0.01 (too small) | 1.160 | 1.307 | 1.443 | 1.682 | 6.95 |
| 0.05 (about right) | 1.800 | 2.280 | 2.568 | 2.844 | 0.097 |
| 0.3 (too big) | 5.800 | -0.920 | 8.488 | 13.756 | 462.8 |

At 0.01 the network crawls. At 0.3 each step overshoots 3 by more than the last, and the loss climbs from 16 to 463. The knife edge here is 0.25, where `w` bounces between 1 and 5 forever. Real training fails the same ways: with too large a rate, the [Deep Learning book](https://www.deeplearningbook.org/contents/optimization.html) warns, "the learning curve will show violent oscillations."

## Step 4: backpropagation, the chain rule in reverse

Nudging works for one weight. GPT-3 has 175 billion, and nudging each one separately would take 175 billion extra passes through the model for a single update. Backpropagation gets every slope in one backward pass.

Add a second layer to our toy network. The input goes through weight `w1` and a ReLU to make a hidden value `h`, then through weight `w2` to make the prediction:

```text
h          = ReLU(w1 × x)
prediction = w2 × h
loss       = (prediction - target)²
```

Start with `x = 2`, `w1 = 0.5`, `w2 = 2`, target 6. The forward pass gives `h = ReLU(1) = 1`, a prediction of `2 × 1 = 2`, and a loss of 16.

Now walk backward one link at a time. At each link, ask a local question: if this value moved up a little, how much would the next value move? These are exchange rates.

- Loss per unit of prediction: `2 × (prediction - target) = 2 × (2 - 6) = -8`.
- Prediction per unit of `w2`: that's `h`, which is 1. So the slope for `w2` is `-8 × 1 = -8`.
- Prediction per unit of `h`: that's `w2`, which is 2. So the slope for `h` is `-8 × 2 = -16`.
- `h` per unit of `w1`: that's `x`, which is 2 (ReLU passes slopes through unchanged when its input is positive, and blocks them when it's negative). So the slope for `w1` is `-16 × 2 = -32`.

![Two panels. Forward pass: input x = 2, times w1 = 0.5 and ReLU, gives hidden h = 1; times w2 = 2 gives prediction 2; against target 6 the loss is (2 - 6)² = 16. Backward pass: the slope for the prediction is 2 × (2 - 6) = -8; it splits into the slope for w2, -8 × h = -8, and the slope for h, -8 × w2 = -16, which gives the slope for w1, -16 × x = -32.](/diagrams/f03-backprop.svg "The forward pass computes the loss. The backward pass multiplies local exchange rates from the loss back to every weight.")

Each weight's slope is the product of the exchange rates along the path from that weight to the loss.

That is the chain rule from calculus: when one change causes another, which causes another, the overall rate is the rates multiplied together. You can check it the slow way. Raising `w1` from 0.5 to 0.51 drops the loss to 15.6816, a slope of about -31.8, and raising `w2` from 2 to 2.01 drops it to 15.9201, a slope of about -8.0. Backpropagation got the same answers without nudging anything.

`w1` gets four times the push of `w2` because its effect passes through `w2 = 2`, which doubles every change in `h`. Take one gradient-descent step with a learning rate of 0.01: `w1` becomes 0.82 and `w2` becomes 2.08, the prediction rises to 3.41 and the loss falls from 16 to 6.70. After five such steps the loss is 0.033.

The saving comes from reuse: the slope for `h` was computed once and reused for everything that feeds into `h`. In a real network, each layer takes the slopes for its outputs, computes the slopes for its own weights, and passes slopes for its inputs down to the layer below, so one sweep from last layer to first fills in the whole gradient. The [scaling-laws paper](https://arxiv.org/abs/2001.08361) puts the backward pass at about twice the compute of the forward pass. That's where Part 1's `6 × N × D` training rule comes from: 2 FLOPs per parameter per token going forward and about 4 going back, or roughly 1 trillion FLOPs per training token for GPT-3.

The [1986 Nature paper](https://www.nature.com/articles/323533a0) noted that, trained this way, the hidden units in between come to represent important features of the task, an ability that set backpropagation apart from earlier, simpler learning methods.

## Step 5: from a toy to an LLM

An LLM is trained with this same loop. What changes is the loss, the amount of data per step, and the way the step is taken.

### The loss for next-token prediction

An LLM outputs a probability for every token in its vocabulary (the pieces of text from [Part 2](/blog/tokenization/)). The loss is cross-entropy: take the probability the model gave to the token that actually came next, and compute `-ln(probability)`, the negative of its natural logarithm. A confident right answer costs almost nothing; a surprise costs a lot. With the "The capital of France is" probabilities from Part 1:

| If the real next token was | Model's probability | Loss `-ln(p)` |
|---|---|---|
| Paris | 85.2% | 0.16 |
| a | 7.0% | 2.66 |
| Lyon | 0.95% | 4.66 |
| any token, if the model spread its bets evenly over 128,000 | 0.00078% | 11.76 |

The slopes at the output are simple. For the correct token's logit (its raw score before softmax), the slope is its probability minus 1; for every other token, it's just that token's probability. If the answer was Paris, the slope for Paris's logit is `0.852 - 1 = -0.148`, so gradient descent raises it; the slope for "a" is +0.070 and for "Lyon" +0.0095, so both get lowered a little. Backpropagation then carries those slopes back through all 96 blocks, down to the embedding table.

### Mini-batches

The exact slope would mean averaging over the whole training set, trillions of tokens, before every step. Instead, each step averages the slopes over a random sample called a mini-batch. Because the sample is random, the method is called stochastic gradient descent (SGD), and each step points only roughly downhill. The [Deep Learning book](https://www.deeplearningbook.org/contents/optimization.html) notes that most optimizers converge much faster in total computation with quick approximate gradients than with slow exact ones.

LLM mini-batches are measured in tokens, and they're big. GPT-3 used [3.2 million tokens per step](https://arxiv.org/abs/2005.14165). Llama 3.1 405B [started at 4 million](https://arxiv.org/abs/2407.21783), doubled to 8 million after 252 million tokens, and doubled again to 16 million after 2.87 trillion.

### Adam: a smarter step

Plain gradient descent uses one learning rate for every parameter, though some parameters have huge, steady slopes and others tiny, jittery ones. The Adam optimizer, from [Kingma and Ba](https://arxiv.org/abs/1412.6980), keeps two running averages for every parameter: the average of its recent slopes, which works like momentum and smooths out the noise, and the average of its recent squared slopes, which tracks how big its slopes usually are. Each step divides the first by the square root of the second.

As a result, every parameter's step comes out roughly the size of the learning rate, whatever the raw scale of its slope. The paper suggests a default learning rate of 0.001 and averaging factors of 0.9 and 0.999. With those, a slope of -16 and a slope of -0.16 both produce a first step of 0.001.

Adam costs memory. The [ZeRO paper](https://arxiv.org/abs/1910.02054) counts 16 bytes per parameter for the standard mixed-precision setup: the weights, the gradient, a full-precision copy of the weights and Adam's two averages. For a 1.5 billion-parameter model like GPT-2, that's at least 24 GB, against 3 GB to hold the weights alone. At the same 16 bytes, a 405 billion-parameter model would need about 6.5 TB before counting anything else, one reason training is spread over thousands of GPUs.

### Overfitting

What you want is low loss on text the model has never seen. When the gap between training error and error on new data grows too large, the [Deep Learning book](https://www.deeplearningbook.org/contents/ml.html) calls it overfitting: the model is memorizing examples where it should be learning the pattern.

The standard check is to track loss on held-out data the model never trains on. If training loss keeps falling while held-out loss rises, the model is memorizing. LLM training sets are big enough that much of the text is read once at most: GPT-3's training saw its filtered [Common Crawl data 0.44 times](https://arxiv.org/abs/2005.14165), less than once, and Wikipedia 3.4 times; the authors sampled data without replacement "to minimize overfitting".

## In the real world

- GPT-3 (2020) trained with Adam using averaging factors of 0.9 and 0.95, a learning rate of 0.6 × 10⁻⁴ for the 175B model, 3.2 million tokens per batch, and the overall size of each gradient capped at 1.0, a safeguard called gradient clipping ([GPT-3 paper](https://arxiv.org/abs/2005.14165), Appendix B).
- Llama 3.1 405B (2024) used AdamW, an Adam variant covered in Part 12, with a peak learning rate of 8 × 10⁻⁵. The rate ramped up over the first 8,000 steps, then decayed to 8 × 10⁻⁷ on a schedule laid out over 1.2 million steps, with batches growing from 4 million to 16 million tokens. Meta reports the recipe as "very stable", with few loss spikes ([Llama 3 paper](https://arxiv.org/abs/2407.21783)).
- The same Llama run kept its GPUs at 38 to 43% of their peak arithmetic rate, 380 to 430 trillion FLOPs per second each. That ratio is called MFU (model FLOPs utilization).

## Common misconceptions

- **"Artificial neurons work like brain neurons."** An artificial neuron is a weighted sum and a simple function like ReLU. The 1986 paper itself called them "neurone-like units". The name is borrowed loosely from biology.
- **"Backpropagation is the learning algorithm."** Backpropagation only computes the slopes. The optimizer, plain gradient descent or Adam, decides how far to move each weight, and backpropagation can feed any of them.
- **"Gradient descent finds the best possible weights."** It finds a low point near where it started, with no guarantee it's the lowest. The [Deep Learning book](https://www.deeplearningbook.org/contents/optimization.html) notes that experts now suspect most such low points in a large network have low loss anyway.
- **"A chatbot learns from each conversation."** Answering a prompt runs only the forward pass; no slopes are computed and no weights change. A model improves only when its maker runs another round of training.

## Key takeaways

- A neuron is a weighted sum plus a bias, bent by an activation function like ReLU or GELU. A layer of neurons is one matrix multiplication, and that's why GPUs run neural networks.
- A forward pass costs about one multiply-add per parameter per token. For GPT-3, that's 175 billion multiply-adds for every token.
- Training measures error with a loss, finds the slope of the loss for every weight, and steps each weight downhill. The learning rate sets the step size: too small crawls, too big blows up.
- Backpropagation finds all the slopes in one backward pass by multiplying local exchange rates, the chain rule, from the loss back to each weight. It costs about twice the forward pass.
- LLMs use exactly this loop, with cross-entropy on the next token as the loss, mini-batches of millions of tokens, and Adam-style optimizers, repeated across trillions of tokens.

## What's next

We've treated the network's inputs and outputs as plain lists of numbers. [Part 4](/blog/embeddings-and-logits/) looks at the two ends of an LLM: the embedding table that turns each token into a vector, and the final layer that turns a vector back into a score, a logit, for every token in the vocabulary. You'll see why words with related meanings end up pointing in similar directions.

## Further reading

- [Learning representations by back-propagating errors (Rumelhart, Hinton & Williams, 1986)](https://www.nature.com/articles/323533a0): the four-page Nature paper that described backpropagation for networks with hidden units.
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980): the optimizer used to train GPT-1 and GPT-3 (and, as AdamW, Llama 3), with its algorithm and default settings on one page.
- [Deep Learning, Chapter 6: Deep Feedforward Networks (Goodfellow, Bengio & Courville, 2016)](https://www.deeplearningbook.org/contents/mlp.html): neurons, activation functions and backpropagation, with the full math.
- [Deep Learning, Chapter 8: Optimization for Training Deep Models (Goodfellow, Bengio & Courville, 2016)](https://www.deeplearningbook.org/contents/optimization.html): mini-batches, SGD, momentum, Adam and why optimizing deep networks is hard.
- [Gaussian Error Linear Units (GELUs) (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415): the activation function used in GPT-style models.
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361): where the 2 FLOPs per parameter forward and 6 per parameter for training come from.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054): why Adam training needs 16 bytes per parameter and how to split that across GPUs.
- [Gradient descent, how neural networks learn (3Blue1Brown, 2017)](https://www.youtube.com/watch?v=IHZwWFHWa-w): an animated walk down the loss landscape.
- [Backpropagation, intuitively (3Blue1Brown, 2017)](https://www.youtube.com/watch?v=Ilg3gGewQ5U) and [Backpropagation calculus (3Blue1Brown, 2017)](https://www.youtube.com/watch?v=tIeHLnjs5U8): the best visual explanation of the backward pass, then the chain rule behind it.
- [The spelled-out intro to neural networks and backpropagation: building micrograd (Andrej Karpathy, 2022)](https://www.youtube.com/watch?v=VMj-3S1tku0): about two and a half hours building backpropagation from scratch in plain Python.
