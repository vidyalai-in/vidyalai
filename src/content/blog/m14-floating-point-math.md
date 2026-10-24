---
title: "Floating Point and Quantization: The Math of Fewer Bits"
description: "How computers store decimals, why 256 + 1 = 256 in BF16, why training keeps FP32 copies, and the error math of scales, groups, GPTQ and AWQ on real weights."
slug: "floating-point-math"
category: "llm-math"
tags: ["math", "quantization", "training", "gpu"]
series: "llm-math"
seriesOrder: 14
pubDate: 2026-10-24
---

## TL;DR

A floating-point number is a sign, a power of two and a fixed number of binary digits, so its error is a fixed percentage of the number's size. That one fact explains why BF16 has FP32's range but under 3 significant digits, why adding small numbers to a big total can do nothing, and why training keeps an FP32 copy of every weight. Quantization's error has a simple formula, the scale divided by √12, which reproduces the 12% to 13% weight error of 4-bit Qwen2.5-1.5B from Part 13. The way that error spreads out explains why rounding kept the model working while low-rank truncation broke it, and GPTQ and AWQ improve on rounding by using the statistics of real inputs.

## Why it matters

In BF16, the 16-bit format most LLMs are trained and served in, 256 + 1 = 256. Add a million random numbers between 0 and 1 one at a time in BF16 and the total gets stuck at 256, when the right answer is 500,159.

Yet the same model whose weights are stored in BF16 survives being squeezed into 4 bits. [Part 13](/blog/low-rank-svd/) rounded every attention and MLP weight in Qwen2.5-1.5B to 4 bits, with one scale per group of 128, and perplexity only went from 8.46 to 9.99. Cutting the same matrices to the same storage with the SVD sent it to about 45 million.

Both come down to what a format can represent and how rounding errors add up. [Inference Part 3](/blog/gpus-for-llms/) listed the formats and [Inference Part 8](/blog/quantization/) surveyed the methods; this part works out the math underneath.

## The intuition

Think of a price display with room for three significant digits and a movable decimal point. It can show 1.23, 12.3 or 12,300, a huge range. But after 12,300 the next price it can show is 12,400, so 12,345 is shown as 12,300, and adding a 40-cent item changes nothing. The error is always under about half a percent, but in money it grows with the price.

That is a floating-point number: the movable point is the exponent, the three digits are the mantissa, and "accurate to a percentage" is the rule behind everything below.

The analogy breaks in two places. Computers count in binary, so their "digits" are bits and familiar decimals like 0.1 can't be written exactly. And the exponent has a limited range too: push past it and you get infinity or zero, which [Part 5](/blog/softmax-math/) met as overflow and underflow.

## Step 1: how a computer stores a decimal

### Binary fractions, and why 0.1 is not exact

In binary the places after the point are halves, quarters, eighths and so on, so 0.375 is 0.011 in binary: 0 halves, 1 quarter, 1 eighth. To convert a decimal fraction, keep doubling it and write down the whole-number part each time. For 0.1: doubling gives 0.2, 0.4, 0.8, 1.6, then (dropping the 1) 1.2, 0.4, 0.8, 1.6, 1.2 and so on. The whole-number parts are 0, 0, 0, 1, 1, 0, 0, 1, 1, ... so

$$
0.1 = 0.0\,0011\,0011\,0011\ldots_2
$$

The block 0011 repeats forever, the way 1/3 = 0.333... does in decimal. Any finite number of bits has to cut it off, which is why 0.1 + 0.2 prints as 0.30000000000000004 in Python.

### Sign, exponent and mantissa

A float writes a number like scientific notation in base 2. Slide the binary point until exactly one 1 sits before it: 0.000110011... becomes 1.10011001... times $$2^{-4}$$. Then store three things:

$$
x = (-1)^{s} \times 1.m \times 2^{\,e - \text{bias}}
$$

Here $$s$$ is the sign bit (0 for positive), $$m$$ is the mantissa, the bits after the binary point, and $$e$$ is the exponent field, stored as a whole number with a fixed bias added so that negative powers need no sign of their own. The leading 1 is never stored, since it is always there: a free extra bit. FP32 (32-bit floating point) uses 1 sign bit, 8 exponent bits with a bias of 127, and 23 mantissa bits.

Python can show the actual bits:

```python
import struct
from fractions import Fraction

bits = struct.unpack(">I", struct.pack(">f", 0.1))[0]     # the 32 bits of float32(0.1)
b = f"{bits:032b}"
print("bits:    ", b[0], b[1:9], b[9:])
s, e, m = int(b[0]), int(b[1:9], 2), int(b[9:], 2)
print("sign", s, " exponent field", e, " mantissa field", m)
value = (-1)**s * (1 + m / 2**23) * 2.0**(e - 127)
print(f"value:    {value:.20f}")
print("exact:   ", Fraction(struct.unpack(">f", struct.pack(">f", 0.1))[0]))
```

```text
bits:     0 01111011 10011001100110011001101
sign 0  exponent field 123  mantissa field 5033165
value:    0.10000000149011611938
exact:    13421773/134217728
```

Decode it by hand. The sign is 0: positive. The exponent field is 123, and 123 − 127 = −4. The mantissa is the repeating 1001 pattern, its last bit rounded up from 0 to 1; as a fraction it's 5,033,165 / 8,388,608 = 0.6000000238. So the value is 1.6000000238 × 1/16 = 0.10000000149, off by about 1.5 parts in 100 million.

Two exponent patterns are reserved. All ones means infinity or NaN ("not a number"). All zeros switches off the hidden 1, giving subnormal numbers that fill the gap down to zero at reduced precision: they let FP16 reach $$2^{-24} \approx 6 \times 10^{-8}$$, a floor that comes back in Step 3.

## Step 2: the formats, side by side

### Percent-accurate numbers

Between two powers of two, a stretch called a binade, the representable numbers are evenly spaced: with $$m$$ mantissa bits there are $$2^m$$ of them, so between 1 and 2 the gap is $$2^{-m}$$, between 2 and 4 twice that. The gap doubles every time the number does, so gap divided by number stays between $$2^{-m-1}$$ and $$2^{-m}$$. Rounding to the nearest value is off by at most half a gap, so the worst relative error is

$$
u = 2^{-(m+1)}
$$

This $$u$$ is the unit roundoff. In words: a float is accurate to a fixed fraction of its own size, set only by the mantissa bits; the exponent bits set the range.

Here are the formats LLMs use, computed with PyTorch's `torch.finfo` (TF32 by formula; FP4 E2M1 from the [OCP Microscaling spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), which lists its bias as 1, its largest value as 6 and no infinities or NaN):

| Format | Sign / exponent / mantissa bits | Largest value | Smallest normal | Gap after 1.0 | Worst relative rounding error | Significant decimal digits |
|---|---|---|---|---|---|---|
| FP32 | 1 / 8 / 23 | 3.40 × 10^38 | 1.18 × 10^-38 | 1.19 × 10^-7 | 0.000006% | 7.2 |
| TF32 (math mode) | 1 / 8 / 10 | 3.40 × 10^38 | 1.18 × 10^-38 | 0.00098 | 0.049% | 3.3 |
| BF16 | 1 / 8 / 7 | 3.39 × 10^38 | 1.18 × 10^-38 | 0.0078 | 0.39% | 2.4 |
| FP16 | 1 / 5 / 10 | 65,504 | 6.10 × 10^-5 | 0.00098 | 0.049% | 3.3 |
| FP8 E4M3 | 1 / 4 / 3 | 448 | 0.0156 | 0.125 | 6.25% | 1.2 |
| FP8 E5M2 | 1 / 5 / 2 | 57,344 | 6.10 × 10^-5 | 0.25 | 12.5% | 0.9 |
| FP4 E2M1 | 1 / 2 / 1 | 6 | 1 | 0.5 | 25% | 0.6 |

The last column is $$\log_{10} 2^{m+1}$$, the number of decimal digits the mantissa (plus its hidden bit) is worth.

BF16 is laid out like the first 16 bits of FP32: the same 8 exponent bits, so the same range, and 7 mantissa bits, so about 2.4 significant digits. FP16 makes the opposite trade, more precision but overflow above 65,504, and the FP8 pair repeats it at 8 bits. [Micikevicius et al. (2022)](https://arxiv.org/abs/2209.05433) defined E4M3 without infinities and with only one NaN pattern, which buys one extra power of two of range: without that choice its largest value would be 240, not 448. FP4 E2M1 has only eight magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4 and 6.

Integers are the opposite design: INT8's 255 levels and INT4's 15 or 16 are evenly spaced, so small values get large relative errors. Step 4 gives them a scale.

### Rounding to nearest, ties to even

When a result falls exactly halfway between two neighbors, IEEE 754 arithmetic, the standard nearly all hardware follows, picks the one whose last mantissa bit is 0. In BF16 near 256 the gap is 2, so 257 becomes 256, 259 becomes 260 and 261 becomes 260. Rounding ties always up would push sums upward; ties to even goes each way equally often.

## Step 3: errors that add up

### 256 + 1 = 256

BF16 keeps 8 significant bits counting the hidden one, so from 256 = $$2^8$$ upward the gap is $$2^{8-7} = 2$$. Adding 1 lands exactly halfway and rounds back to 256. Adding anything smaller than 1 rounds back too. Here is what that does to a long sum:

```python
import numpy as np, torch

x = np.random.default_rng(0).random(1_000_000).astype(np.float32)   # a million numbers in [0, 1)
print(f"exact sum: {x.astype(np.float64).sum():,.2f}")

def running_sum(dtype):
    s = torch.zeros((), dtype=dtype)
    for v in torch.from_numpy(x).to(dtype):        # add one at a time, rounding after every add
        s = s + v
    return s.item()

for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    print(f"{str(dtype):15} {running_sum(dtype):,.2f}")
```

```text
exact sum: 500,159.26
torch.bfloat16  256.00
torch.float16   2,048.00
torch.float32   500,162.97
```

The BF16 total stalls at 256 and FP16, with 3 more mantissa bits, stalls at 2,048 = $$2^{11}$$, where its gap also reaches 2. That's 0.05% and 0.4% of the right answer. FP32 is off by 3.71, a relative error of $$7.4 \times 10^{-6}$$.

### Why errors grow like √n, not n

Each FP32 addition makes an error of at most half the gap at the current total. If all of them pushed the same way, they would add up in a straight line: for this run, a worst-case bound of 10,164. In practice they are as likely positive as negative and partly cancel, like the steps of a random walk. Each has a root-mean-square size of gap/√12 (Step 4 derives this), independent errors add in squares, and the typical total comes out at 6.69. The measured 3.71 is in that range.

In relative terms, adding $$n$$ numbers has a worst-case error of about $$n u$$ and a typical one of about $$\sqrt{n}\, u$$: for a million FP32 additions, 6% against 0.006%.

Two fixes are standard. One is a wider accumulator, a running total in FP32 even for 16-bit inputs. PyTorch's [numerical accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) say: "Half-precision GEMM operations are typically done with intermediate accumulations (reduction) in single-precision", GEMM being a general matrix multiply. The other is a better order: NumPy's `np.sum` uses what its [docs](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) call "partial pairwise summation", adding in pairs, then pairs of pairs, so no total gets huge next to what is added to it. Its FP32 error here is 0.006.

The same reasoning explains the RMSNorm detail from [Part 10](/blog/initialization-normalization/): Hugging Face's Qwen2 code converts each vector to FP32 before squaring and averaging its 1,536 numbers, because a 16-bit running total of squares can stop growing just as this one did. Softmax sums get the same treatment ([Part 5](/blog/softmax-math/)).

### Why training keeps an FP32 master copy

Training adds a small update to every weight at every step. Store the weight in BF16 and the update meets the same wall:

```python
import torch
w_bf16 = torch.tensor(1.0, dtype=torch.bfloat16)   # weight stored only in BF16
w_fp32 = torch.tensor(1.0)                          # FP32 master copy
for step in range(1000):
    w_bf16 += 1e-3                                  # learning rate x gradient
    w_fp32 += 1e-3
print(w_bf16.item(), w_fp32.item(), w_fp32.bfloat16().item())
```

```text
1.0 2.000046730041504 2.0
```

The BF16 gap above 1.0 is $$2^{-7} = 0.0078$$, so any update below half of it, 0.0039, is rounded away. A thousand updates of 0.001 do nothing, while the FP32 copy reaches 2.00005 and rounds to the correct 2.0 in BF16.

Only the ratio matters: in BF16 an update is lost when the weight is roughly 256 to 512 times bigger, depending on where it sits in its binade. Qwen's weights have a spread of about 0.03, where the BF16 gap is 0.000122. Adam moves each weight by roughly the learning rate ([Part 9](/blog/optimizers-adam/)), and Llama 3 405B's schedule ran from a peak of $$8 \times 10^{-5}$$ down to $$8 \times 10^{-7}$$. At the peak, a step of 0.00008 rounds up to a full gap, 53% too big; at the end, a step of 0.0000008 vanishes.

The fix, from [Micikevicius et al. (2017)](https://arxiv.org/abs/1710.03740), is "an FP32 master copy of weights" that the optimizer updates, with a 16-bit copy rounded from it for each forward and backward pass. Their FP16 version of the rule: an update is lost when the weight "is at least 2048 times larger". Without the master copy, their Mandarin speech model lost 80% of its relative accuracy. This copy is 4 of the 16 bytes per parameter that Part 9 counted for mixed-precision Adam.

![A loop of five steps. FP32 master weights are rounded to a BF16 copy. The forward and backward passes run in BF16 with sums kept in FP32. They produce 16-bit gradients; with FP16, the loss was multiplied by a scale S and the gradients are divided by S here. Adam updates the weights in FP32, where tiny steps survive, and the loop returns to the master weights for the next step.](/diagrams/m14-mixed-precision.svg "One step of mixed-precision training.")

Only the optimizer's weights and the sums need the extra bits; the expensive multiplications run in 16 bits.

### Loss scaling: rescuing tiny gradients in FP16

FP16 has a second problem: its range. Its smallest subnormal is $$2^{-24} \approx 6 \times 10^{-8}$$, and the same paper found that in one model "approximately 5% of weight gradient values have exponents smaller than −24", so they would become zero.

Loss scaling multiplies the loss by a constant $$S$$ before the backward pass. By the chain rule ([Part 8](/blog/backpropagation-math/)), every gradient is then $$S$$ times larger, so a gradient of $$10^{-8}$$, which is 0 in FP16, becomes $$1.03 \times 10^{-5}$$ with $$S = 1024$$. Dividing by $$S$$ in FP32 before the update gives back $$1.0012 \times 10^{-8}$$. The paper used factors from 8 to 32K, keeping the largest gradient under 65,504 and skipping any step that overflows anyway. PyTorch's `GradScaler` automates this and changes $$S$$ during training (defaults include `growth_factor=2.0` and `backoff_factor=0.5`).

BF16 mostly removes the need: its smallest normal number is $$1.2 \times 10^{-38}$$, the same as FP32, and it stores $$10^{-8}$$ directly as $$1.0012 \times 10^{-8}$$. That is one reason large models now train in BF16 rather than FP16.

## Step 4: quantization and the s/√12 rule

### Scales and zero points

Quantization stores a group of numbers as small integers plus a scale $$s$$, the distance between neighboring levels:

$$
q = \text{round}(x / s) + z, \qquad \hat{x} = s\,(q - z)
$$

Here $$x$$ is the original value, $$q$$ the stored integer, $$z$$ the zero point (an integer offset), and $$\hat{x}$$, "x hat", the value you get back. Symmetric INT4 sets $$z = 0$$ and $$s = \max|x| / 7$$; asymmetric stretches 16 levels from the group's minimum to its maximum, $$s = (\max - \min)/15$$, with $$z$$ putting the minimum on level 0. [Inference Part 8](/blog/quantization/) works both by hand.

### Deriving the error

Rounding to the nearest level leaves an error $$e$$ between $$-s/2$$ and $$+s/2$$. For values with no special relation to the grid, every error in that range is equally likely, so its mean square is the average of $$e^2$$ over the interval:

$$
\frac{1}{s}\int_{-s/2}^{s/2} e^2 \, de = \frac{1}{s}\cdot\frac{2}{3}\left(\frac{s}{2}\right)^3 = \frac{s^2}{12}
$$

So the root-mean-square (RMS) error is $$s/\sqrt{12} \approx 0.29\,s$$, and the average size of the error is $$s/4$$. Checked on a million random numbers:

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.standard_normal(1_000_000)
s = 0.1                                   # the scale: distance between levels
err = np.round(x / s) * s - x             # dequantized minus original
print(f"RMS error  {np.sqrt(np.mean(err**2)):.5f}   s/sqrt(12) = {s / np.sqrt(12):.5f}")
print(f"mean |err| {np.mean(np.abs(err)):.5f}   s/4        = {s / 4:.5f}")
```

```text
RMS error  0.02886   s/sqrt(12) = 0.02887
mean |err| 0.02498   s/4        = 0.02500
```

### Where 12% comes from

Now divide by the size of the weights. With $$s = \max|w| / 7$$, the relative error of a group is

$$
\frac{s/\sqrt{12}}{\text{rms}(w)} = \frac{1}{7\sqrt{12}} \cdot \frac{\max|w|}{\text{rms}(w)} \approx \frac{1}{24.2} \cdot \frac{\max|w|}{\text{rms}(w)}
$$

The second factor says how far a group's largest weight sticks out from a typical one. For 128 numbers from a bell curve it averages 2.83, which predicts 11.7%. In block 15 of Qwen2.5-1.5B it averages 2.91 to 3.18 depending on the matrix, and the formula predicts each matrix's error to within 0.1 percentage point: 13.36% against a measured 13.30% for the query projection, 13.49% against 13.41% for the MLP down projection. So Part 13's "12% to 13%" follows from the formula.

Each extra bit roughly halves $$s$$, and the error with it. INT8 per row gave 1.08% where INT4 per row gave 19.3%, a ratio of 17.9, close to 127/7 = 18.1.

### Outliers, group size and bits per weight

That ratio is why outliers hurt: one large weight sets $$s$$ for its whole group. Across a whole matrix of block 15, the largest weight is 6.4 to 39.7 times the typical size, against about 3 within a group of 128. Smaller groups confine each outlier, but every scale costs 16 more bits. The script reads block 15's seven matrices one at a time:

```python
import glob, os, torch
from safetensors import safe_open
hub = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")) + "/hub"
f = safe_open(glob.glob(hub + "/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors")[0], "pt")

def int4(W, group):                       # symmetric absmax: levels -7..7, one scale per group
    G = W.reshape(-1, group)              # rows of `group` consecutive weights
    s = G.abs().amax(1, keepdim=True) / 7
    return (torch.round(G / s) * s).reshape(W.shape), s

names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
for label in ["per-tensor", "per-row", "groups of 128", "groups of 32"]:
    err2 = pred2 = total2 = 0.0
    for n in names:                       # block 15 (index 14), one matrix at a time
        W = f.get_tensor(f"model.layers.14.{n}.weight").float()
        group = {"per-tensor": W.numel(), "per-row": W.shape[1], "groups of 128": 128, "groups of 32": 32}[label]
        Wq, s = int4(W, group)
        err2 += (Wq - W).pow(2).sum().item()
        pred2 += (s**2 / 12).sum().item() * group      # s^2/12 for every weight in the group
        total2 += W.pow(2).sum().item()
    print(f"{label:14} relative error {(err2 / total2)**0.5:6.1%}   s/sqrt(12) predicts {(pred2 / total2)**0.5:6.1%}")
```

```text
per-tensor     relative error  77.6%   s/sqrt(12) predicts  97.6%
per-row        relative error  19.3%   s/sqrt(12) predicts  19.6%
groups of 128  relative error  12.8%   s/sqrt(12) predicts  12.9%
groups of 32   relative error  10.2%   s/sqrt(12) predicts  10.4%
```

With one scale per matrix the step is so big that most weights round to 0, and a weight rounded to 0 has an error of exactly its own size, so the uniform-error rule overshoots. From one scale per row down, it holds.

With the scale overhead counted, and the two 4-bit float formats from [Inference Part 8](/blog/quantization/) simulated on the same weights:

| Scheme (block 15, all seven matrices) | Bits per weight | Relative weight error |
|---|---|---|
| INT4, one scale per matrix | 4.00 | 77.6% |
| INT4, one scale per row | about 4.01 | 19.3% |
| INT4, groups of 128 | 4 + 16/128 = 4.125 | 12.8% |
| INT4, groups of 128 with a zero point | 4.125 plus the zero points | 10.7% |
| MXFP4: FP4, 32 values share a power-of-two scale | 4 + 8/32 = 4.25 | 11.8% |
| INT4, groups of 32 | 4 + 16/32 = 4.5 | 10.2% |
| NVFP4: FP4, 16 values share an FP8 scale | 4 + 8/16 = 4.5 | 9.4% |
| FP8 E4M3, one scale per matrix | 8 | 2.7% |
| INT8, one scale per row | about 8.01 | 1.1% |

MXFP4's scale, an 8-bit power of two (E8M0 in the [OCP spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)), is the largest power of two at or below the block's maximum, divided by 4, so the block's largest value lands anywhere from 4 to just under 8 on the FP4 grid, and anything above 6 is clipped. [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)'s E4M3 scale, "with fractional precision", fits each block of 16 more tightly. FP8 gets away with one scale per matrix because, as a float, its error is a percentage of each value.

## Step 5: from weight error to output error

### Rounding error behaves like white noise

What matters is the layer's output, $$\mathbf{y} = W\mathbf{x}$$. Rounding changes $$W$$ by an error matrix $$E$$, so the output changes by $$E\mathbf{x}$$. If the entries of $$E$$ act like independent noise with spread $$\sigma_E$$, Part 10's variance rule says each output entry, a sum of $$d_{in}$$ terms $$E_{ij}x_j$$, has variance $$\sigma_E^2 \lVert\mathbf{x}\rVert^2$$. Adding over the $$d_{out}$$ outputs:

$$
\lVert E\mathbf{x} \rVert \approx \sigma_E \sqrt{d_{out}}\, \lVert \mathbf{x} \rVert
$$

In words: the output error depends on the input's length, not its direction. It behaves like white noise, the hiss of an untuned radio, which is equally loud at every pitch.

We tested this on block 15 with the real inputs from Part 13's Oz passage. For 4-bit rounding in groups of 128, the formula predicted each token's output error almost exactly: a median of 13.4% predicted against 13.3% measured for the query projection, 13.7% against 13.7% for the MLP up projection, 12.3% against 12.3% for the down projection. And 25% of the error's energy fell along the top quarter of the query projection's singular directions (Part 13's SVD), the share that random noise would put there.

### Why rounding beat truncation

Truncation behaves differently. Keeping 99% of each matrix's energy leaves a weight error of 10%, less than rounding's 12.8%, but all of it sits in the dropped directions, so the output error depends on where each input points. For the up projection the median token's error was 21%, and the passage's first token, the start-of-text marker, got 66%, against 25% under rounding.

The bigger difference builds up over the 28 blocks. Here is the error in the residual stream, the running sum each block adds to, for the median token:

| After block | 4-bit rounding, groups of 128 | SVD, 99% of energy kept |
|---|---|---|
| 1 | 17.1% | 32.6% |
| 5 | 23.3% | 52.3% |
| 10 | 24.0% | 72.1% |
| 15 | 24.9% | 87.1% |
| 20 | 25.0% | 97.5% |
| 28 | 25.5% | 82.3% |
| Perplexity at the end | 9.99 | 675 |

Rounding's error levels off at about 25%: each block adds fresh noise in random directions, which only partly adds to what came before, while the stream itself grows ([Part 10](/blog/initialization-normalization/)). Truncation's error climbs to 97.5% by block 20, the pattern of losses that build on each other, like Step 3's worst case. Giving the first token exact weights in every block cut the truncated model's perplexity from 675 to 292; doing the same for token 100 only cut it to 644. The first token explains part of the damage, the steady build-up the rest.

## Step 6: smarter rounding with GPTQ and AWQ

Round-to-nearest treats every weight alone. But the output only feels $$E\mathbf{x}$$, and inputs are not white noise: some channels are large, and many move together.

### GPTQ: let the next weight absorb the error

Take one output computed from two inputs that move together, with $$\mathbb{E}[x_1^2] = \mathbb{E}[x_2^2] = 1$$ and $$\mathbb{E}[x_1 x_2] = 0.9$$, where $$\mathbb{E}$$ means the average over many inputs. The weights are $$w = [0.3,\ 0.8]$$ and the only allowed values are multiples of 0.5.

Round-to-nearest gives [0.5, 1.0], errors of +0.2 and +0.2. Since the inputs move together, the two errors push the output the same way: its mean square is $$0.2^2 + 0.2^2 + 2 \times 0.2 \times 0.2 \times 0.9 = 0.152$$, an RMS error of 0.39.

Now quantize $$w_1$$ first, then ask how to change $$w_2$$ to cancel the +0.2. The output error is $$0.2\,x_1 + \delta\,x_2$$, and its mean square is smallest at

$$
\delta = -e_1 \frac{\mathbb{E}[x_1 x_2]}{\mathbb{E}[x_2^2]} = -0.2 \times 0.9 = -0.18
$$

where $$e_1 = +0.2$$ is the error already made. So $$w_2$$ becomes 0.62, which rounds to 0.5:

```python
import numpy as np
C = np.array([[1.0, 0.9],                 # E[x1^2], E[x1 x2]
              [0.9, 1.0]])                # E[x2 x1], E[x2^2]
w = np.array([0.3, 0.8])
grid = lambda v: np.round(v / 0.5) * 0.5  # allowed values: multiples of 0.5
rmse = lambda q: np.sqrt((q - w) @ C @ (q - w))   # RMS of the output error (q - w) . x

rtn = grid(w)
e1 = rtn[0] - w[0]                        # error on the first weight: +0.2
w2 = w[1] - e1 * C[0, 1] / C[1, 1]        # compensate: 0.8 - 0.2 * 0.9 = 0.62
comp = np.array([rtn[0], grid(w2)])
print("round to nearest:", rtn, f"output RMS error {rmse(rtn):.3f}")
print("compensated:     ", comp, f"output RMS error {rmse(comp):.3f}")
```

```text
round to nearest: [0.5 1. ] output RMS error 0.390
compensated:      [0.5 0.5] output RMS error 0.148
```

The second weight is now further off, 0.3 instead of 0.2, yet the output error is 2.6 times smaller: the two errors have opposite signs and cancel through the correlated inputs.

[GPTQ (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323) does this for whole matrices. It minimizes the output error on calibration inputs $$X$$ (one row per token). That error's Hessian, the matrix of second derivatives from [Part 9](/blog/optimizers-adam/), is $$H = 2X^{\top}X$$, a scaled version of the $$\mathbb{E}[x_i x_j]$$ table above. GPTQ quantizes one column of weights at a time and spreads its error over the columns not yet quantized using $$H^{-1}$$; with two weights, its formula reduces to the $$\delta$$ above. It adds 1% of the average diagonal to $$H$$ for stability and calibrates on "128 random 2048 token segments".

### AWQ: make the important weights relatively bigger

[AWQ (Lin et al., 2023)](https://arxiv.org/abs/2306.00978) starts from the s/4 average error above. Multiply one weight by $$s > 1$$ and divide its input by $$s$$; the product is unchanged. If the group's scale $$\Delta$$ (the paper's name for it) stays about the same, the rounding error on $$ws$$ is still about $$\Delta/4$$, but it now reaches the output divided by $$s$$:

$$
\text{error} \approx \frac{\Delta}{4} \cdot \frac{|x|}{s}
$$

That weight's error shrinks by a factor of $$s$$. But a large enough $$s$$ makes $$ws$$ the group's maximum and stretches $$\Delta$$ for its neighbors. So AWQ scales each input channel by its average activation size raised to a power $$\alpha$$ between 0 and 1, picked by a grid search. In the paper, scaling the 1% most important channels of OPT-6.7B by 2 at 3 bits took perplexity from 23.54 to 11.92.

### Both on Qwen

We ran simple versions of both on block 15, calibrating on 2,048 tokens of Chapter 2 of *The Wonderful Wizard of Oz* and testing on 1,705 tokens of Chapter 9:

| Matrix | Weight error: rounding / AWQ / GPTQ | Output error on new text: rounding / AWQ / GPTQ |
|---|---|---|
| q_proj | 13.3% / 15.1% / 21.0% | 13.1% / 10.0% / 9.1% |
| o_proj | 12.1% / 12.5% / 18.2% | 16.0% / 14.1% / 12.1% |
| up_proj | 12.5% / 13.1% / 18.0% | 13.8% / 11.9% / 11.6% |
| down_proj | 13.4% / 13.4% / 19.7% | 11.8% / 11.8% / 13.6% |

Both make the weights worse and the outputs better, by moving error into directions the inputs rarely use. The last row shows the limit: the down projection has 8,960 inputs and we calibrated on only 2,048 tokens, so GPTQ fit the calibration text (3.3% error there) and lost to plain rounding on new text, while AWQ's search chose $$\alpha = 0$$, no scaling. GPTQ's authors used a quarter of a million calibration tokens.

## In the real world

- DeepSeek-V3 trained in FP8 ([technical report](https://arxiv.org/abs/2412.19437)): E4M3 for all tensors, one scale per 1 × 128 tile of activations and per 128 × 128 block of weights (its checkpoint's config lists `"weight_block_size": [128, 128]`). FP8 matrix multiplies on H800 GPUs keep only "around 14 bits" while accumulating, a "maximum relative error of nearly 2%" for an inner dimension of 4,096, so every 128 elements the partial sums are copied to FP32 registers: Step 3's wider accumulator. Master weights and gradients stayed in FP32 and Adam's moments in BF16. Validated on two smaller models for about a trillion tokens, the loss stayed within 0.25% of BF16 training.
- OpenAI's gpt-oss stores its mixture-of-experts weights, "90+% of the total parameter count", in MXFP4 at "4.25 bits per parameter" ([model card](https://arxiv.org/abs/2508.10925)). Its config leaves attention, the router, the embeddings and the output layer unconverted.
- Qwen2.5-1.5B ships in BF16 (`"torch_dtype": "bfloat16"`), and its RMSNorm sums in FP32, as Part 10 showed.
- Tensor cores run FP8 on Ada, Hopper and Blackwell, and FP4 only on Blackwell ([Inference Part 3](/blog/gpus-for-llms/)). 4-bit integer formats like GPTQ's are unpacked to 16 bits inside the kernel, saving memory and bandwidth but not multiply time ([Inference Part 8](/blog/quantization/)).

## Common misconceptions

- **"BF16 is just a less accurate FP32."** It's less precise, but its range is the same: the smallest normal number is $$1.2 \times 10^{-38}$$ in both. FP16 is more precise than BF16 and overflows at 65,504, which is why it needs loss scaling and BF16 mostly doesn't.
- **"Quantization error is about how many bits you use."** Bits set the number of levels, but the scale sets their spacing, and one outlier can stretch it: the same 4 bits gave 77.6% error with one scale per matrix and 12.8% with one per 128 weights.
- **"Lower weight error means a better quantized model."** GPTQ raised Qwen's query-projection weight error from 13.3% to 21.0% and cut its output error from 13.1% to 9.1%, and truncation at 10% weight error did far more damage than rounding at 12.8%. What counts is error in the directions the inputs use.

## Key takeaways

- A float is a sign, a power of two and a fixed number of bits, so it's accurate to a fixed percentage: 0.39% for BF16, 6.25% for FP8 E4M3, 25% for FP4.
- Rounding errors add up. In BF16 a running sum stalls at 256, which is why matrix multiplies and norms add in FP32 and why training keeps FP32 master weights for its tiny updates.
- Quantization's error is the scale divided by the square root of 12. With a scale from the group's maximum, that predicts Qwen's 12.8% weight error at 4 bits and groups of 128.
- Rounding error acts like white noise, the same in every direction, so it doesn't pile up across blocks the way truncation's does.
- GPTQ and AWQ accept more weight error in exchange for less output error, by using the statistics of real inputs.

## What's next

You can now say how many bits each weight needs. [Part 15](/blog/counting-flops-scaling-laws/), Counting, adds everything else up: exact parameter counts per layer, the FLOPs of a training run, the memory for weights, optimizer states and the KV cache, and the scaling laws that decide how big a model should be for a given budget.

## Further reading

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg, 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html): the classic tutorial on rounding error, relative error and machine epsilon.
- [Mixed Precision Training (Micikevicius et al., 2017)](https://arxiv.org/abs/1710.03740): FP32 master weights, loss scaling and FP32 accumulation.
- [FP8 Formats for Deep Learning (Micikevicius et al., 2022)](https://arxiv.org/abs/2209.05433): the E4M3 and E5M2 definitions and the reasoning behind them.
- [OCP Microscaling Formats (MX) Specification v1.0 (Open Compute Project, 2023)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): FP4 E2M1, the E8M0 scale and how MX blocks are converted.
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference (NVIDIA, 2025)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/): 16-value blocks with FP8 scales and a per-tensor FP32 scale.
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323): error compensation with the inverse Hessian, column by column.
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (Lin et al., 2023)](https://arxiv.org/abs/2306.00978): the scaling argument and the search for α.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): section 3.3 on FP8 training with fine-grained scales and promoted accumulation.
- [gpt-oss-120b and gpt-oss-20b Model Card (OpenAI, 2025)](https://arxiv.org/abs/2508.10925): MXFP4 weights in a released model.
- [Numerical accuracy (PyTorch docs)](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) and [Automatic Mixed Precision (PyTorch docs)](https://docs.pytorch.org/docs/2.14/amp.html): how PyTorch accumulates half-precision matrix multiplies and scales gradients in practice.
