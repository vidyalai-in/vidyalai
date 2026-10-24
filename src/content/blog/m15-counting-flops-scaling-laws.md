---
title: "Counting: Parameters, FLOPs, Memory and Scaling Laws"
description: "Exact parameter counts from a config file, 6ND derived with what it leaves out, KV cache bytes, log-log fits, the Chinchilla optimum and MoE routing math."
slug: "counting-flops-scaling-laws"
category: "llm-math"
tags: ["math", "scaling-laws", "training", "cost", "moe"]
series: "llm-math"
seriesOrder: 15
pubDate: 2026-10-24
---

## TL;DR

Most of what a language model costs can be worked out from its config file. This part counts a decoder's parameters exactly, checked against a published weight file, then derives the 6ND rule for training compute and the two things it leaves out, and turns compute into days and bytes. It fits power laws on log-log axes and derives the compute-optimal model size from Chinchilla's loss formula. That derivation gives a tokens-per-parameter ratio that a later replication traced to a fitting bug, and labs train small models far past either value anyway. It ends with mixture-of-experts arithmetic: total versus active parameters, the router's top-k softmax, and why the load-balancing loss is smallest when routing is even.

## Why it matters

Take the five fitted numbers from DeepMind's Chinchilla paper, plug them into one derivative, and you get the best model size for any training budget. At the budget Chinchilla itself used, that formula says to train about 93 tokens per parameter. The Chinchilla model was trained on 20.

Both numbers come from the same paper. A 2024 replication found out why they disagree: an optimizer that stopped too early. Following that story takes a few counts (parameters, FLOPs, bytes) and one optimization with a constraint.

The same counts tell you whether a model fits on your GPU, how long a run takes, and why Llama 3 8B was trained on 1,875 tokens per parameter. [Fundamentals Part 12](/blog/pretraining-scaling-laws/) told that story in words. This part derives it.

## The intuition

Picture a rectangle with a fixed area. One side is the number of parameters $$N$$, the other the number of training tokens $$D$$, and the area is the compute budget. A short side costs you: too few parameters and the model can't hold what it needs, too few tokens and it never learns it. Stretching one side squeezes the other, so the compute-optimal model is the rectangle with the smallest total penalty.

The analogy breaks in two places. The penalties aren't known in advance; they're fitted to hundreds of small training runs, and a fit can be wrong. And the area isn't the whole bill: a model that answers billions of questions pays again every time it runs.

## Step 1: counting parameters exactly

### A formula from the config file

Write $$V$$ for the vocabulary size, $$d$$ for the width of each token's vector, $$L$$ for the number of blocks, $$n_q$$ and $$n_{kv}$$ for the query and key-value heads, $$d_h$$ for the numbers per head and $$d_{ff}$$ for the MLP's inner width. One block of a Llama-style decoder holds:

- Attention: query and output projections of $$d \times n_q d_h$$ each, key and value projections of $$d \times n_{kv} d_h$$ each.
- The MLP: SwiGLU's gate, up and down matrices, $$3 \cdot d \cdot d_{ff}$$ weights.
- Two RMSNorms with $$d$$ gains each.

Around the blocks sit the embedding table ($$V \cdot d$$), an output table of the same size unless the two are tied, and a final RMSNorm. For an untied model:

$$
N = 2Vd + d + L\big(2d\,n_q d_h + 2d\,n_{kv} d_h + 3d\,d_{ff} + 2d\big)
$$

That's two tables, one final norm, and $$L$$ copies of attention, MLP and two norms. [Fundamentals Part 10](/blog/modern-llm-architecture/) filled it in line by line for Llama 3.1 8B and got 8,030,261,248. Qwen2.5-1.5B is a better test, with two quirks. It ties its tables, so $$2Vd$$ becomes $$Vd$$, and it adds a bias to every query, key and value number, $$(n_q + 2n_{kv}) d_h$$ more per block. Its [config](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json) gives $$V$$ = 151,936, $$d$$ = 1,536, $$L$$ = 28, 12 query heads, 2 key-value heads of 128, and $$d_{ff}$$ = 8,960.

A `.safetensors` file starts with 8 bytes giving the length of a JSON header that lists every tensor's shape, so the weight file can check the count without loading a weight:

```python
import json, struct
from math import prod

def decoder_params(V, d, L, n_q, n_kv, d_head, d_ff, tied, qkv_bias=False):
    attn = d * n_q * d_head + 2 * d * n_kv * d_head + n_q * d_head * d  # Q, K, V, output
    if qkv_bias:
        attn += (n_q + 2 * n_kv) * d_head    # one bias per query, key and value number
    block = attn + 3 * d * d_ff + 2 * d      # + SwiGLU MLP + two RMSNorm gains
    tables = V * d if tied else 2 * V * d    # embedding table (+ separate output table)
    return tables + L * block + d            # + final RMSNorm

print(f"{decoder_params(128256, 4096, 32, 32, 8, 128, 14336, tied=False):,}")   # Llama 3.1 8B
print(f"{decoder_params(151936, 1536, 28, 12, 2, 128, 8960, tied=True, qkv_bias=True):,}")  # Qwen2.5-1.5B

# Count every tensor listed in the safetensors header of the published weights
with open("model.safetensors", "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]   # first 8 bytes: header length
    header = json.loads(f.read(header_len))
header.pop("__metadata__", None)
print(f"{sum(prod(t['shape']) for t in header.values()):,} in {len(header)} tensors")
```

```text
8,030,261,248
1,543,714,304
1,543,714,304 in 338 tensors
```

The formula matches both models exactly. (We read Qwen2.5-1.5B-Instruct's file, which has the base model's shapes.) Qwen's file has no `lm_head` tensor, because the output layer reuses the embedding table, whose 233,373,696 parameters are 15.1% of the model. Small models tie their tables for exactly this reason.

### Twelve d squared, or thirteen

Divide a block's weight matrices by $$d^2$$ and one number sums up the design. The classic GPT block has four $$d \times d$$ attention matrices and an MLP from $$d$$ to $$4d$$ and back: $$4d^2 + 8d^2 = 12d^2$$. [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) write model size as

$$
N \approx 12\, n_{\text{layer}}\, d_{\text{model}}^2
$$

and leave out the embedding tables, which "produces significantly cleaner scaling laws." Llama 3.1 8B's grouped-query attention makes keys and values a quarter size, $$2.5d^2$$ for attention, and its three MLP matrices of width $$3.5d$$ add $$10.5d^2$$. The total is exactly $$13d^2$$, and $$13 \times 4096^2 + 2 \times 4096 = 218{,}112{,}000$$ per block. Qwen2.5-1.5B's wider MLP ($$d_{ff} = 5.83d$$) makes its block $$19.8d^2$$.

## Step 2: deriving 6ND

### Two FLOPs forward, four back

[Part 3](/blog/matrices-linear-maps/) counted a matrix multiply at $$2mkn$$ FLOPs (floating-point operations): per token, one multiply and one add for each weight. [Part 8](/blog/backpropagation-math/) showed the backward pass runs two more multiplies of the same size, for the input's gradient and the weights' gradient. That's 2 FLOPs forward and 4 backward per weight per token. Sum over the weights in matrices, $$N$$, and the training tokens, $$D$$:

$$
C \approx 6ND
$$

Here $$C$$ is the training compute in FLOPs. The rule is only as good as its $$N$$. An embedding lookup is a copy, free going forward, and its gradient is one row added back in.

- Llama 3.1 8B is untied, so its 525,336,576 embedding weights never enter a matmul. The honest $$N$$ is 7,504,658,432, and training costs 45.0 GFLOPs per token, not the 48.2 the headline 8.03 billion gives: 7% less.
- Qwen2.5-1.5B is tied. Its one table is looked up for free and then multiplied as the output layer, so $$6N$$ with all 1.54 billion parameters is right.

### What attention adds

The other thing 6ND leaves out is attention's token-to-token products, which use no weights. A token at position $$t$$ scores its query against $$t$$ keys in each head, then mixes $$t$$ values. With $$n_q d_h = d$$, each product costs $$2td$$ per layer, $$4td$$ for both, as [Part 11](/blog/attention-math/) counted per head.

With the causal mask, the average token in an $$n_{\text{ctx}}$$-token training sequence looks back about $$n_{\text{ctx}}/2$$ tokens. That makes the forward cost $$2 n_{\text{ctx}} d$$ per layer per token, the term in Kaplan et al.'s Table 1:

$$
C_{\text{forward}} \approx 2N + 2\, n_{\text{layer}}\, n_{\text{ctx}}\, d
$$

The backward pass triples both terms. Divide attention by weights, with $$N = 12 n_{\text{layer}} d^2$$, and almost everything cancels:

$$
\frac{2\, n_{\text{layer}} n_{\text{ctx}} d}{2 \cdot 12\, n_{\text{layer}} d^2} = \frac{n_{\text{ctx}}}{12\, d}
$$

Attention is a small extra while the context is short compared with $$12d$$, which is why Kaplan et al. dropped it for their 1,024-token runs. For Llama 3.1 8B, $$12d \approx 49{,}000$$:

| Training context | Weights, forward per token | Attention, forward per token | Attention's share |
|---|---|---|---|
| 2,048 tokens | 15.01 GFLOPs | 0.54 GFLOPs | 3.5% |
| 8,192 tokens | 15.01 GFLOPs | 2.15 GFLOPs | 12.5% |
| 131,072 tokens | 15.01 GFLOPs | 34.36 GFLOPs | 69.6% |

### Checking Llama 3.1 405B's budget

Meta's [Llama 3 paper](https://arxiv.org/abs/2407.21783) says the 405B model was "pre-trained using 3.8 × 10²⁵ FLOPs" on 15.6 trillion tokens, and $$6 \times 405 \times 10^9 \times 15.6 \times 10^{12} = 3.79 \times 10^{25}$$: the paper's figure is 6ND. Add attention with the paper's 126 layers and width 16,384. About 14.8 trillion tokens at 8,192 tokens of context add $$6 \times 126 \times 8192 \times 16384 \times 14.8 \times 10^{12} = 1.50 \times 10^{24}$$, or 4.0%. The final "approximately 800B training tokens," which raised the context to 128K in six stages, add at most $$1.30 \times 10^{24}$$ (3.4%), even if all of them ran at 131,072 tokens. So the true count is at most 4.07 × 10²⁵, 7% above 6ND, and less in practice because Llama 3 masks attention between documents packed into one sequence.

### From FLOPs to days

Compute turns into time with one division:

$$
\text{time} = \frac{C}{n_{\text{GPU}} \times \text{peak} \times \text{MFU}}
$$

Here $$n_{\text{GPU}}$$ is the number of GPUs, peak is one GPU's top FLOPs per second, and MFU (model FLOPs utilization) is the share of the peak spent on the model's own arithmetic. An H100's dense BF16 peak is 989 TFLOPS, half the [1,979 NVIDIA lists](https://www.nvidia.com/en-us/data-center/h100/) "with sparsity," and Meta reports an MFU of 38% to 43%. On 16,384 GPUs, 3.79 × 10²⁵ FLOPs take 63 to 71 days of pure compute. [Fundamentals Part 12](/blog/pretraining-scaling-laws/) compares that with the GPU-hours Meta reported.

Now a laptop. [GPU Part 4](/blog/gpu-profiling-roofline/) measured 25.3 BF16 TFLOPS on an RTX 4060 Laptop GPU. A model of Qwen2.5-1.5B's size at 20 tokens per parameter needs 30.9 billion tokens and $$6 \times 1.544 \times 10^9 \times 3.09 \times 10^{10} = 2.86 \times 10^{20}$$ FLOPs: 327 days at 40% MFU. Qwen2.5 was trained on 18 trillion tokens ([Qwen2.5 report](https://arxiv.org/abs/2412.15115)), which would take the laptop 522 years.

## Step 3: counting bytes

### Training memory and the KV cache

[Part 9](/blog/optimizers-adam/) counted mixed-precision Adam at 16 bytes per parameter. For Qwen2.5-1.5B that's 24.7 GB, three times the laptop's 8 GB, before any activations, which grow with every token in the batch ([Part 8](/blog/backpropagation-math/) gave the formula).

When a model generates text, it keeps every past token's keys and values ([Inference Part 2](/blog/kv-cache/)). In each of $$L$$ layers, a token's key has $$n_{kv}$$ heads of $$d_h$$ numbers, and so does its value. At $$b$$ bytes per number:

$$
\text{KV bytes per token} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot b
$$

The 2 is for keys and values. Qwen2.5-1.5B in 16-bit needs $$2 \times 28 \times 2 \times 128 \times 2 = 28{,}672$$ bytes, 28 KiB per token, or 896 MiB for a full 32,768-token context. The weights take $$N \cdot b$$ bytes, so the cache matches them at

$$
n_{\text{tokens}} = \frac{N}{2\, L\, n_{kv}\, d_h}
$$

with the bytes per number cancelled. For Llama 3.1 8B that's 122,532 tokens, inside its 128K window: one user at full context holds 16 GiB (17.2 GB) of cache, more than the model's 16.06 GB of weights.

### Arithmetic intensity, training versus decode

Arithmetic intensity is FLOPs per byte moved from memory. Pushing $$T$$ tokens through a $$k \times n$$ weight matrix does $$2Tkn$$ FLOPs and reads about $$2kn$$ bytes of weights, so one token gets about 1 FLOP per byte. An 8,192-token training sequence through Llama's 4,096 × 14,336 MLP matrix gets 2,294, counting inputs and outputs too. That's far above the ridge where a GPU turns compute-bound (about 102 FLOPs per byte on the laptop, 295 on an H100), so training time follows FLOPs, as the formula above assumes. Decoding one token at a time sits far below it, and its speed is bytes over bandwidth. [GPU Part 4](/blog/gpu-profiling-roofline/) plots both on a roofline, and [Inference Part 1](/blog/how-llm-inference-works/) shows how batching moves decode toward the ridge.

![A top-to-bottom map of how the counts connect. A config file gives the parameter count N and the KV cache per token, 2 times layers times key-value heads times head size times bytes. N leads to training FLOPs, 6ND plus attention, and to training memory, 16 bytes per parameter plus activations. Training FLOPs lead to training time, compute divided by GPUs times peak times MFU. Training FLOPs and a fitted loss, E plus A over N to the alpha plus B over D to the beta, together give the compute-optimal N and D. Adding inference cost leads to a smaller N trained on more data.](/diagrams/m15-counting-map.svg "Every count in this part starts from a config file or a fitted loss.")
## Step 4: power laws on log-log axes

### Straight lines in disguise

A power law says one quantity is a constant times another raised to a fixed power:

$$
y = a \cdot x^{b}
$$

Take logs of both sides. A product's log is a sum of logs, and a power comes down in front ([Part 4](/blog/probability-language-models/) introduced logarithms):

$$
\log y = \log a + b \log x
$$

That's a straight line with slope $$b$$ and intercept $$\log a$$. On log-log axes, where each grid step is a factor of 10, a power law is a straight line, and the exponent says how many powers of 10 $$y$$ moves per power of 10 in $$x$$.

Exponents read best as multipliers. Chinchilla's parameter term is $$A/N^{0.34}$$. Multiply $$N$$ by 10 and the term is multiplied by $$10^{-0.34} = 0.457$$: ten times the parameters removes 54% of the loss owed to being small. Doubling multiplies it by $$2^{-0.34} = 0.79$$. The data term $$B/D^{0.28}$$ shrinks by $$10^{-0.28} = 0.525$$ per factor of 10. The constant $$E$$ never goes away; [Part 6](/blog/cross-entropy-perplexity/) explained it as the entropy of text.

### Fitting a line by least squares

Least squares picks the line that makes the sum of squared vertical gaps to the points as small as possible; on log axes, you fit the logs. Here it is on Table 3 of the [Chinchilla paper](https://arxiv.org/abs/2203.15556), which lists compute-optimal sizes, budgets and token counts from its first method:

```python
import numpy as np

# Hoffmann et al. (2022), Table 3: parameters, FLOPs, tokens
N = np.array([0.4e9, 1e9, 10e9, 67e9, 175e9, 280e9, 520e9, 1e12, 10e12])
C = np.array([1.92e19, 1.21e20, 1.23e22, 5.76e23, 3.85e24, 9.90e24, 3.43e25, 1.27e26, 1.30e28])
D = np.array([8.0e9, 20.2e9, 205.1e9, 1.5e12, 3.7e12, 5.9e12, 11.0e12, 21.2e12, 216.2e12])

# A power law N = k * C^a is a straight line in logs: log N = log k + a log C
a, logk = np.polyfit(np.log10(C), np.log10(N), 1)
b, _ = np.polyfit(np.log10(C), np.log10(D), 1)
print(f"N_opt ∝ C^{a:.3f}   D_opt ∝ C^{b:.3f}")
print(f"6ND / C: {np.round(6 * N * D / C, 2)}")
print(f"D / N:   {np.round(D / N, 1)}")
```

```text
N_opt ∝ C^0.498   D_opt ∝ C^0.502
6ND / C: [1.   1.   1.   1.05 1.01 1.   1.   1.   1.  ]
D / N:   [20.  20.2 20.5 22.4 21.1 21.1 21.2 21.2 21.6]
```

`np.polyfit(..., 1)` fits a straight line and returns the slope first. Both slopes are about 0.50, the paper's value for this method, the budgets are 6ND, and the ratio stays between 20 and 22.4 over nine orders of magnitude of compute. That's where "20 tokens per parameter" comes from.

The full loss $$E + A/N^{\alpha} + B/D^{\beta}$$ isn't straight on log-log axes, because it flattens toward $$E$$. The Chinchilla authors fitted all five constants at once by minimizing a Huber loss (squared error that counts outliers less) on the log of the loss, with the L-BFGS optimizer started from a grid of guesses.

## Step 5: the compute-optimal N and D

### The derivation

Minimize

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

subject to $$6ND = C$$ for a fixed budget $$C$$. Substitute $$D = C/(6N)$$ so the loss depends on $$N$$ alone:

$$
L(N) = E + A N^{-\alpha} + B \left(\frac{C}{6}\right)^{-\beta} N^{\beta}
$$

The middle term falls as $$N$$ grows and the last rises, so $$L(N)$$ is U-shaped, lowest where its slope is zero. With [Part 7](/blog/derivatives-gradients/)'s power rule, the slope of $$N^k$$ is $$kN^{k-1}$$:

$$
\frac{dL}{dN} = -\alpha A N^{-\alpha-1} + \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta-1} = 0
$$

Multiply by $$N$$, move the first term across, and use $$(C/6)/N = D$$:

$$
\alpha \frac{A}{N^{\alpha}} = \beta \frac{B}{D^{\beta}}
$$

At the optimum, each penalty times its exponent is equal; otherwise moving compute toward the bigger one would pay. To solve for $$N$$, go back a step: $$N^{\alpha+\beta} = \frac{\alpha A}{\beta B}(C/6)^{\beta}$$. Take the $$(\alpha+\beta)$$-th root:

$$
N_{\text{opt}} = G \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}, \qquad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}
$$

$$
D_{\text{opt}} = \frac{C}{6 N_{\text{opt}}} = G^{-1} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}
$$

This is equation (4) of the Chinchilla paper. The exponents add to 1 because $$N \times D$$ must grow exactly as fast as $$C$$. Their ratio is

$$
\frac{D_{\text{opt}}}{N_{\text{opt}}} = G^{-2} \left(\frac{C}{6}\right)^{\frac{\alpha-\beta}{\alpha+\beta}}
$$

so tokens per parameter stay fixed only if $$\alpha = \beta$$. Otherwise the ratio drifts with the budget.

### Plugging in the numbers

The paper's fit is $$E = 1.69$$, $$A = 406.4$$, $$B = 410.7$$, $$\alpha = 0.34$$, $$\beta = 0.28$$. That gives $$N_{\text{opt}} \propto C^{0.452}$$, $$G = 1.345$$, and a ratio that grows 1.25 times per factor of 10 in compute. A check that the algebra is right:

```python
import numpy as np

E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28   # Hoffmann et al., eq. 10

def optimum(C):
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)

C = 1e24
N, D = optimum(C)
Ns = np.logspace(9, 12, 300_001)                       # brute force: 300,001 model sizes
loss = E + A / Ns**alpha + B / (C / (6 * Ns))**beta
print(f"closed form: N = {N/1e9:.2f}B, D = {D/1e12:.2f}T, D/N = {D/N:.0f}")
print(f"brute force: N = {Ns[loss.argmin()]/1e9:.2f}B")
```

```text
closed form: N = 41.30B, D = 4.04T, D/N = 98
brute force: N = 41.30B
```

At that point the two penalties are 0.0999 and 0.1213, and 0.34 × 0.0999 and 0.28 × 0.1213 both come to 0.0340, as the balance condition says. Across budgets, with this fit and with a refit discussed next:

| Budget (FLOPs) | N, Hoffmann fit | D/N, Hoffmann fit | N, Besiroglu refit | D/N, Besiroglu refit |
|---|---|---|---|---|
| 10²¹ | 1.8B | 50 | 2.8B | 21.6 |
| 10²³ | 14.6B | 78 | 29.5B | 19.2 |
| 5.76 × 10²³ (Chinchilla) | 32.2B | 93 | 72.2B | 18.4 |
| 10²⁵ | 117B | 122 | 312B | 17.1 |
| 3.8 × 10²⁵ (Llama 3.1 405B) | 213B | 139 | 619B | 16.5 |

For Chinchilla's budget the paper's formula says 32 billion parameters on 3 trillion tokens. DeepMind trained 70 billion on 1.4 trillion.

### What went wrong with the fit

The paper's Table 2 gives the exponent of $$N_{\text{opt}}$$ for each of its three methods: 0.50 from training curves, 0.49 from fixed-budget sweeps, and 0.46 from the fitted formula, with an 80% interval of only 0.454 to 0.455. The authors noted that the third method "predicts even smaller models being optimal at larger compute budgets."

[Besiroglu et al. (2024)](https://arxiv.org/abs/2404.10102) at Epoch AI rebuilt the data from the paper's figures and refitted it. They found the third method's estimates "inconsistent with their first two estimation methods," a poor fit to the extracted data, and intervals so narrow they "would require over 600,000 experiments, while they likely only ran fewer than 500." According to Besiroglu et al., one of the Chinchilla authors later traced the cause: the Huber losses were averaged instead of summed, so the optimizer stopped before converging. Rounding the printed constants made it worse. With the unrounded exponents (0.3392 and 0.2849), the ratio at Chinchilla's budget drops from 93 to 59.

The refit is $$E = 1.8172$$, $$A = 482.01$$, $$B = 2085.43$$, $$\alpha = 0.3478$$, $$\beta = 0.3658$$. Now $$\beta$$ is slightly larger than $$\alpha$$, so the ratio drifts down instead of up, from 21.6 to 16.5 across the table. That agrees with the first two methods and with how Chinchilla was actually trained.

### Paying for inference

Compute-optimal means the cheapest way to reach a loss in training. A served model also costs about $$2N$$ FLOPs per generated token. [Sardana et al. (2023)](https://arxiv.org/abs/2401.00448) added that cost: fix a target loss and minimize

$$
\text{total} = 6 N D_{\text{train}} + 2 N D_{\text{inference}}
$$

For each $$N$$, the target loss fixes the training tokens needed, so the search is over $$N$$ alone. With the Besiroglu refit, a target equal to the compute-optimal loss at 10²³ FLOPs (2.032), and illustrative inference volumes:

| Tokens served over the model's life | Best N | Training tokens | Tokens per parameter | Total compute vs compute-optimal model |
|---|---|---|---|---|
| 0 | 29.5B | 566B | 19 | same (1.0 × 10²³) |
| 10¹² | 19.7B | 907B | 46 | 8% less |
| 10¹³ | 11.4B | 2.43T | 214 | 43% less |
| 10¹⁴ | 7.5B | 9.48T | 1,269 | 68% less |

As demand grows, the best model shrinks and trains longer, even though training alone costs four times more in the last row. Sardana et al. conclude that labs "expecting reasonably large inference demand (~1B requests) should train models smaller and longer than Chinchilla-optimal," and they trained models out to 10,000 tokens per parameter. Llama 3 8B sits at about 1,875.

## Step 6: mixture of experts

### Total versus active parameters

A mixture-of-experts (MoE) layer replaces one MLP with $$E_{\text{n}}$$ smaller MLPs, the experts, plus a router that sends each token to $$k$$ of them ([Fundamentals Part 11](/blog/mixture-of-experts/)). With $$P_{\text{shared}}$$ for everything outside the experts and $$P_{\text{exp}}$$ for one expert:

$$
N_{\text{total}} = P_{\text{shared}} + L \cdot E_{\text{n}} \cdot P_{\text{exp}}
$$

$$
N_{\text{active}} = P_{\text{shared}} + L \cdot k \cdot P_{\text{exp}}
$$

Mixtral 8x7B's [config](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json) is Llama-shaped: 32 layers, $$d$$ = 4,096, a 32,000-token untied vocabulary, and 8 experts of $$3 \times 4096 \times 14336 = 176{,}160{,}768$$ weights, top-2, with an $$8 \times 4096$$ router per layer. The formulas give 46,702,792,704 total and 12,879,925,248 active, and the weight index lists `total_size` as 93,405,585,408 bytes, exactly 2 per parameter. DeepSeek-V3's [config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) gives about 671.0 billion total and 37.6 billion active by the same method, matching the [report](https://arxiv.org/abs/2412.19437)'s "671B total parameters, of which 37B are activated for each token."

Compute per token follows $$N_{\text{active}}$$, so 6ND for DeepSeek-V3 uses 37 billion. Memory follows $$N_{\text{total}}$$, because any token may pick any expert.

### The router's top-k softmax

Mixtral's [paper](https://arxiv.org/abs/2401.04088) writes its router as `Softmax(Top2(x · Wg))`: keep the two largest logits and softmax only those. The Hugging Face [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py) takes a softmax over all 8, keeps the top 2 probabilities, and divides by their sum. These always agree. Write $$S$$ for the sum of $$e^{z_j}$$ over all experts' logits $$z_j$$. A winner $$i$$ gets $$e^{z_i}/S$$, and dividing by the winners' total cancels $$S$$:

$$
\frac{e^{z_i}/S}{\sum_{j \in \text{top}} e^{z_j}/S} = \frac{e^{z_i}}{\sum_{j \in \text{top}} e^{z_j}}
$$

That's a softmax over the winners alone, and since softmax keeps the order, both pick the same winners. DeepSeek-V3 instead scores each expert with a sigmoid, $$1/(1+e^{-z})$$, and divides the top 8 by their sum.

### The load-balancing loss

A router that favors a few experts trains them more, favors them more, and leaves the rest idle. The [Switch Transformer](https://arxiv.org/abs/2101.03961) adds an auxiliary loss to each MoE layer:

$$
\text{loss}_{\text{bal}} = \alpha \cdot E_{\text{n}} \sum_{i=1}^{E_{\text{n}}} f_i \, P_i
$$

Here $$f_i$$ is "the fraction of tokens dispatched to expert $$i$$" in the batch, $$P_i$$ is the router probability for expert $$i$$ averaged over the batch, and $$\alpha$$ is a small weight (0.01 in the paper). The paper says it "is minimized under a uniform distribution," and high-school algebra shows why. When probabilities match where tokens go, $$f_i = P_i$$, and the sum is $$\sum P_i^2$$ for numbers that add to 1. Expand the squared gaps from the average $$1/E_{\text{n}}$$:

$$
\sum_i \left(P_i - \tfrac{1}{E_{\text{n}}}\right)^2 = \sum_i P_i^2 - \frac{2}{E_{\text{n}}}\sum_i P_i + \frac{E_{\text{n}}}{E_{\text{n}}^2} = \sum_i P_i^2 - \frac{1}{E_{\text{n}}}
$$

A sum of squares can't be negative, so $$E_{\text{n}} \sum P_i^2 \geq 1$$, with equality only when every $$P_i = 1/E_{\text{n}}$$. The factor $$E_{\text{n}}$$ makes the balanced value 1 for any number of experts, and the left side measures how far you are from it.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

# One token, 4 experts, top-2
z = np.array([2.0, 1.0, 0.5, -1.0])
p = softmax(z)
top = np.argsort(-p)[:2]
print("softmax over all:     ", p.round(3))
print("top-2 renormalized:   ", (p[top] / p[top].sum()).round(3))
print("softmax of top-2 only:", softmax(z[top]).round(3))

# Switch load-balancing loss without alpha: N * sum of f_i * P_i
def balance(f, P):
    return len(f) * np.dot(f, P)

print(f"uniform   {balance([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]):.2f}")
print(f"skewed    {balance([0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]):.2f}")
print(f"collapsed {balance([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]):.2f}")
```

```text
softmax over all:      [0.609 0.224 0.136 0.03 ]
top-2 renormalized:    [0.731 0.269]
softmax of top-2 only: [0.731 0.269]
uniform   1.00
skewed    1.20
collapsed 4.00
```

By hand, the skewed case is $$4 \times (0.16 + 0.09 + 0.04 + 0.01) = 1.20$$. The identity agrees: squared gaps of 0.0225, 0.0025, 0.0025 and 0.0225 sum to 0.05, plus 1/4 is 0.30, times 4 is 1.20. Collapse onto one expert gives the maximum, $$E_{\text{n}}$$.

Gradients flow only through $$P_i$$, since $$f_i$$ is a count. The loss's slope with respect to $$P_i$$ is $$\alpha E_{\text{n}} f_i$$, so the busiest expert's probability is pushed down hardest. (Hugging Face's top-k version counts each token $$k$$ times, so its balanced value is $$k$$.)

DeepSeek-V3 mostly drops the loss. Following [Wang et al. (2024)](https://arxiv.org/abs/2408.15664), it adds a bias to each expert's score only when choosing the top 8, and after each step lowers an overloaded expert's bias by $$\gamma$$ and raises an underloaded one's by $$\gamma$$. It used $$\gamma = 0.001$$ "for the first 14.3T tokens, and to 0.0 for the remaining 500B tokens," plus a tiny balance loss with $$\alpha = 0.0001$$. Fundamentals Part 11 works an example.

## In the real world

- Kaplan and Chinchilla mean different things by $$N$$. Kaplan counts non-embedding parameters only. Hoffmann et al. "count embeddings matrices in the total parameter count" and charge the embedding as a matmul; their detailed FLOP count lands within 0.99 to 1.10 times 6ND, and puts Chinchilla's own budget at 6.3 × 10²³ against 6ND's 5.76 × 10²³. [Pearce and Song (2024)](https://arxiv.org/abs/2406.12907) attribute most of the gap between the two papers' exponents to "counting non-embedding rather than total parameters," combined with Kaplan's smaller models.
- Meta fitted its own scaling law on Llama 3 data. Extrapolated to 3.8 × 10²⁵ FLOPs, it suggested "a 402B parameter model on 16.55T tokens," about 41 tokens per parameter.
- DeepSeek-V3 ships its weights in FP8, 1 byte per parameter, so about 671 GB, more than a server of eight 80 GB GPUs holds. [Inference Part 10](/blog/model-parallelism/) shows how expert parallelism spreads the experts across GPUs.

## Common misconceptions

- **"Compute per token is twice the parameter count."** Only for weights in a matmul. Llama 3.1 8B's embedding table is a free lookup, so 6 × 8.03 billion overstates its training FLOPs by 7%. An MoE model's compute follows its active parameters.
- **"6ND is the exact training cost."** Attention adds about $$n_{\text{ctx}}/(12d)$$ on top: 12.5% of the total for Llama 3.1 8B at 8,192 tokens, and more than the weights at 128K.
- **"Chinchilla's formula says 20 tokens per parameter."** Its first two methods do. Its printed formula, run through the derivation, gives 50 to 140. A replication traced that to an optimizer that stopped early, and its refit gives about 20.
- **"The compute-optimal model is the best one to train."** It's the cheapest way to a loss in training. Once serving costs count, the best model is smaller and trained much longer.
- **"The load-balancing loss splits each token evenly."** It evens out the batch-wide shares. Each token still goes to its top-k experts.

## Key takeaways

- A decoder's parameter count follows from its config: tables, a final norm, and per block attention, three MLP matrices and two norms. The formula matches Qwen2.5-1.5B's weight file exactly.
- Training costs 6 FLOPs per matmul weight per token, 2 forward and 4 backward. The embedding lookup is free, and attention adds about context over 12 d on top.
- The KV cache is 2 times layers times key-value heads times head size times bytes, per token. For Llama 3.1 8B it outgrows the weights at about 122,500 tokens.
- A power law is a straight line on log-log axes. Minimizing Chinchilla's loss with 6ND fixed gives a balance: alpha times the parameter penalty equals beta times the data penalty.
- Chinchilla's printed constants imply 50 to 140 tokens per parameter, a corrected fit about 20, and inference demand pushes the best model smaller still.

## What's next

The last part turns from counting to steering. [Part 16](/blog/alignment-math/), The math of alignment, starts from expected reward and derives the policy gradient behind reinforcement learning from human feedback, then works through PPO's clipped objective, the Bradley-Terry model of preferences, the optimum that DPO is built on, and GRPO's group baseline.

## Further reading

- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361): section 2.1 and Table 1 give the parameter and FLOP counts, including the attention term and the 6N rule.
- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556): the Chinchilla paper; equation (4) is the closed-form optimum, Table 2 the three methods' exponents, Appendix F the detailed FLOP count.
- [Chinchilla Scaling: A replication attempt (Besiroglu, Erdil, Barnett & You, 2024)](https://arxiv.org/abs/2404.10102): the refit of the third method and why its confidence intervals were too narrow.
- [Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws (Sardana, Portes, Doubov & Frankle, 2023)](https://arxiv.org/abs/2401.00448): adds inference cost to the optimization.
- [Reconciling Kaplan and Chinchilla Scaling Laws (Pearce & Song, 2024)](https://arxiv.org/abs/2406.12907): how counting parameters differently changes the fitted exponents.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): the 405B training budget, MFU, long-context stages and Meta's own scaling law.
- [Switch Transformers (Fedus, Zoph & Shazeer, 2021)](https://arxiv.org/abs/2101.03961): section 2.2 defines the load-balancing loss.
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (Wang et al., 2024)](https://arxiv.org/abs/2408.15664) and the [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437): bias-based balancing and the sigmoid router.
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) and the [Hugging Face Mixtral code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py): the router written both ways, and the balancing loss as implemented.
- [Qwen2.5-1.5B config](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json) and the [safetensors docs](https://huggingface.co/docs/safetensors/index): everything needed to repeat the exact count.
- [Transformer Math 101 (EleutherAI, 2023)](https://blog.eleuther.ai/transformer-math/): a practical reference for compute, memory and training-time arithmetic.
- [How To Scale Your Model (Google DeepMind)](https://jax-ml.github.io/scaling-book/): a free book that applies this kind of counting to training and serving on thousands of chips.
