---
title: "Rotations, Complex Numbers and the Math of RoPE"
description: "Rotation matrices and complex numbers from scratch, a proof that RoPE scores depend only on distance, and the math of PI, NTK, YaRN and ALiBi."
slug: "rope-math"
category: "llm-math"
tags: ["math", "positional-encoding", "attention", "long-context", "transformers"]
series: "llm-math"
seriesOrder: 12
pubDate: 2026-10-24
---

## TL;DR

A rotation of the plane is a 2 × 2 matrix of cosines and sines. Rotations keep lengths and angles, add their angles when chained, and are undone by their transpose, and those facts prove that RoPE, the position scheme in Llama, Qwen, Gemma and DeepSeek, gives attention scores that depend only on how far apart two tokens are. Complex numbers shrink the proof to one line, because multiplying by a complex number of length 1 is a rotation. From there the article explains RoPE's 64 rotation speeds, why models break past their training length, and the formulas behind position interpolation, NTK-aware scaling, YaRN, Llama 3.1's `rope_scaling` and ALiBi.

## Why it matters

Llama 3.1 8B was pretrained on windows of 8,192 tokens, yet it reads 131,072. The part of its config file that stretches the window is five numbers long: `factor` 8, `low_freq_factor` 1, `high_freq_factor` 4, `original_max_position_embeddings` 8,192 and `rope_theta` 500,000. By the end of this article you'll be able to compute exactly what they do to each of the model's 64 rotation speeds.

The problem sits in the slow rotations. Inside every Llama 3 attention head, the slowest of those rotations turns just 1.15 degrees across the entire 8,192-token training window. Feed the model 131,072 tokens without changing anything and that rotation reaches 18 degrees, angles the model never saw in training.

[Fundamentals Part 7](/blog/positional-encoding/) showed in words what RoPE does. This part derives why it works, from the rotations of [Part 3](/blog/matrices-linear-maps/) and the dot product of [Part 2](/blog/vectors-dot-products/).

## The intuition

Put two arrows on a record player's turntable, one pointing at 2 o'clock and one at 4 o'clock. Spin the record. Both arrows swing around the room, but the angle between them never changes, because they turn together.

RoPE turns every query and key by a number of steps equal to its position. Two tokens 4 positions apart always end up 4 steps out of line, wherever they sit in the text, as if a shared turntable had carried both. The attention score sees only the angle between them, so it sees only the distance.

The analogy breaks in two places. A real head has 64 turntables spinning at different speeds, and the score adds them all up. And a model only knows what the arrows mean at angles it met in training, which is why stretching the context takes extra math.

## Step 1: rotations in the plane

### Where the basis vectors land

[Part 3](/blog/matrices-linear-maps/) showed that a 2 × 2 matrix's columns are where it sends the basis vectors [1, 0] and [0, 1]. So to build a rotation, ask where those two arrows go.

Turn [1, 0] counterclockwise by an angle $$\theta$$ (the Greek letter theta). Its tip stays on the circle of radius 1, and the school definition of cosine and sine puts it at $$(\cos\theta, \sin\theta)$$. The arrow [0, 1] starts a quarter turn further on and lands a quarter turn further on, at $$(-\sin\theta, \cos\theta)$$. Those are the columns:

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

For 30 degrees, $$\cos 30° = 0.866$$ and $$\sin 30° = 0.5$$. So [1, 0] lands on [0.866, 0.5] and [0, 1] on [−0.5, 0.866]. Rotating the point [2, 1] is a weighted sum of those columns, Part 3's recipe: `2 × [0.866, 0.5] + 1 × [-0.5, 0.866] = [1.232, 1.866]`.

Models measure angles in radians: a full turn is $$2\pi \approx 6.283$$ radians, so 30 degrees is 0.524 radians and 1 radian is about 57.3 degrees.

### Lengths and angles survive

The point [2, 1] has length $$\sqrt{5} = 2.236$$, and so does its rotated copy: $$1.232^2 + 1.866^2 = 1.518 + 3.482 = 5$$. For any point $$(x, y)$$, multiplying out $$(x\cos\theta - y\sin\theta)^2 + (x\sin\theta + y\cos\theta)^2$$ makes the cross terms cancel and leaves $$(x^2 + y^2)(\cos^2\theta + \sin^2\theta)$$, which is $$x^2 + y^2$$ because $$\cos^2\theta + \sin^2\theta = 1$$. Rotating never stretches anything.

Angles survive too, since two arrows turned by the same amount keep the angle between them. Part 2 showed that the dot product is $$\mathbf{a} \cdot \mathbf{b} = \lVert \mathbf{a} \rVert \lVert \mathbf{b} \rVert \cos\varphi$$, where $$\varphi$$ (phi) is that angle. Lengths and angle both stay, so the dot product stays. Check with [2, 1] and [1, 3]: their dot product is `2 + 3 = 5`, and after both are turned 30 degrees, to [1.232, 1.866] and [−0.634, 3.098], it's `-0.781 + 5.781 = 5.000`.

### Two turns make one turn

Turning by 30 degrees and then by 45 should be the same as turning by 75. Part 3 showed that doing one matrix after another is the matrix product, so the claim is $$R(45°)\,R(30°) = R(75°)$$. Multiply the top-left entries with the row-times-column rule:

```text
cos 30° cos 45° - sin 30° sin 45° = 0.866 × 0.7071 - 0.5 × 0.7071 = 0.2588 = cos 75°
```

The other three entries work the same way. In general:

$$
R(a)\,R(b) = R(a + b)
$$

Its four entries are the angle addition formulas from trigonometry class, such as $$\cos(a+b) = \cos a\cos b - \sin a\sin b$$. The matrix form is easier to remember: turns add.

### Undoing a turn is a transpose

To undo a turn by $$\theta$$, turn by $$-\theta$$. Since $$\cos(-\theta) = \cos\theta$$ and $$\sin(-\theta) = -\sin\theta$$, that matrix is

$$
R(-\theta) = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} = R(\theta)^{\top}
$$

It's the original matrix flipped over its diagonal, the transpose from [Part 1](/blog/math-for-llms/). For a rotation, the inverse is the transpose: $$R(\theta)^{\top} R(\theta) = I$$, the identity matrix. Matrices like this are called orthogonal matrices.

## Step 2: why RoPE scores depend only on distance

### By hand in two dimensions

Take a query $$\mathbf{q} = [2, 1]$$ and a key $$\mathbf{k} = [1, 3]$$, and say each position turns a vector by 30 degrees. (Real models use much smaller steps; 30 degrees keeps the arithmetic readable.) Without positions, the score is $$\mathbf{q} \cdot \mathbf{k} = 5$$.

Put the query at position 3 and the key at position 1. The query turns 90 degrees, to [−1, 2], and the key turns 30 degrees, to [−0.634, 3.098]. Their dot product is `0.634 + 6.196 = 6.830`. Now slide both along the text, keeping the query 2 positions after the key:

| Query position | Key position | Rotated query | Rotated key | Score |
|---|---|---|---|---|
| 3 | 1 | [−1, 2] | [−0.634, 3.098] | 6.830 |
| 5 | 3 | [−2.232, 0.134] | [−3, 1] | 6.830 |
| 10 | 8 | [1.866, −1.232] | [2.098, −2.366] | 6.830 |
| 3 | 3 | [−1, 2] | [−3, 1] | 5.000 |
| 1 | 3 | [1.232, 1.866] | [−3, 1] | −1.830 |

Geometry explains the 6.830. The query starts at 26.6 degrees and the key at 71.6, 45 degrees apart; the query then turns 60 degrees more than the key, leaving a 15-degree gap, and the score is $$\sqrt{5} \times \sqrt{10} \times \cos 15° = 7.071 \times 0.966 = 6.830$$. The last row shows that direction matters: a key 2 positions after the query gives a different score from a key 2 positions before it.

### The one-line proof

Write the dot product as a matrix product, $$\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^{\top}\mathbf{b}$$: a row times a column. Put the query at position $$m$$ and the key at position $$n$$, turn each by its position times $$\theta$$, and use the rule that transposing a product reverses it, $$(R\mathbf{q})^{\top} = \mathbf{q}^{\top}R^{\top}$$:

$$
\begin{aligned}
\big(R(m\theta)\,\mathbf{q}\big)^{\top} \big(R(n\theta)\,\mathbf{k}\big)
&= \mathbf{q}^{\top}\, R(m\theta)^{\top} R(n\theta)\, \mathbf{k} \\
&= \mathbf{q}^{\top}\, R(-m\theta)\, R(n\theta)\, \mathbf{k} \\
&= \mathbf{q}^{\top}\, R\big((n - m)\theta\big)\, \mathbf{k}
\end{aligned}
$$

The steps use "transpose undoes the turn" and "turns add", and the positions survive only as the difference $$n - m$$. In words: a rotated query against a rotated key scores the same as the original query against the key turned by the offset. This is equation 16 of the RoFormer paper, which calls the approach "multiplicative", unlike older schemes that added a position vector to each token.

### Many pairs at once

A real head has $$d$$ numbers per query and key ($$d = 128$$ in Llama and Qwen). RoPE splits them into $$d/2$$ pairs and gives pair $$j$$ its own step $$\theta_j$$, for $$j = 0, 1, \dots, d/2 - 1$$. The big rotation matrix is 64 little 2 × 2 rotations down the diagonal, zeros elsewhere. A dot product is a sum over pairs and the proof works pair by pair, so the whole score depends only on $$n - m$$. The snippet checks it at full size, with the original base of 10,000 (Step 4 covers the steps):

```python
import numpy as np

d, base = 128, 10_000
theta = base ** (-np.arange(0, d, 2) / d)    # 64 speeds, one per pair

def rope(x, pos):
    """Turn each pair (x[2j], x[2j+1]) by the angle pos * theta[j]."""
    c, s = np.cos(pos * theta), np.sin(pos * theta)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out

rng = np.random.default_rng(0)
q, k = rng.standard_normal(d), rng.standard_normal(d)
print(f"length of q: {np.linalg.norm(q):.6f} before, {np.linalg.norm(rope(q, 5000)):.6f} after")
for m, n in [(7, 3), (107, 103), (5_007, 5_003), (100_004, 100_000), (7, 4)]:
    print(f"query at {m:>7,}, key at {n:>7,}: score {rope(q, m) @ rope(k, n):.6f}")
```

```text
length of q: 10.783940 before, 10.783940 after
query at       7, key at       3: score 4.870427
query at     107, key at     103: score 4.870427
query at   5,007, key at   5,003: score 4.870427
query at 100,004, key at 100,000: score 4.870427
query at       7, key at       4: score 4.148108
```

The length is unchanged, and an offset of 4 gives the same score to six decimals near the start or 100,000 tokens in. An offset of 3 gives a different score.

## Step 3: complex numbers, rotation as multiplication

### A number with two parts

No ordinary number squares to −1. Complex numbers invent one anyway: a new number $$i$$ with the single rule $$i^2 = -1$$. A complex number is $$a + bi$$, a real part $$a$$ plus an imaginary part $$b$$ times $$i$$, best pictured as the point $$(a, b)$$ in the plane. Adding complex numbers adds the points like vectors. Multiplying uses ordinary algebra plus the one rule:

$$
(a + bi)(c + di) = (ac - bd) + (ad + bc)\,i
$$

The $$-bd$$ comes from $$bi \times di = bd\,i^2 = -bd$$. For example, $$(2 + i)(1 + i) = 2 + 2i + i + i^2 = 1 + 3i$$.

### Multiplying multiplies lengths and adds angles

As points, $$2 + i$$ has length $$\sqrt{5}$$ (written $$|2 + i|$$), $$|1 + i| = \sqrt{2}$$, and the product $$1 + 3i$$ has length $$\sqrt{10} = \sqrt{5} \times \sqrt{2}$$. Their angles from the horizontal axis are 26.57, 45 and 71.57 degrees, and $$26.57 + 45 = 71.57$$. Multiplying complex numbers multiplies lengths and adds angles.

So multiplying by a complex number of length 1 only adds an angle: it's a rotation. The complex number at angle $$\theta$$ on the unit circle is $$\cos\theta + i\sin\theta$$, and multiplying $$x + iy$$ by it gives

$$
(x + iy)(\cos\theta + i\sin\theta) = (x\cos\theta - y\sin\theta) + i\,(x\sin\theta + y\cos\theta)
$$

The real and imaginary parts are the two rows of $$R(\theta)$$ applied to $$[x, y]$$. With Step 1's numbers, $$(2 + i)(0.866 + 0.5i) = 1.232 + 1.866i$$, the same as rotating [2, 1] by 30 degrees.

### Euler's formula

Mathematicians write the unit-length complex number at angle $$\theta$$ as an exponential:

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

This is Euler's formula, where $$e \approx 2.718$$ is the base of natural logarithms from [Part 4](/blog/probability-language-models/). Rather than prove it, test it. The exponential has a series, $$e^x = 1 + x + x^2/2! + x^3/3! + \cdots$$, where $$3! = 3 \times 2 \times 1$$. Put in $$x = i\theta$$ and the powers of $$i$$ cycle through $$1, i, -1, -i$$, so the terms alternate between the real and imaginary parts. For 30 degrees, $$\theta = 0.5236$$:

| Terms added | Real part | Imaginary part |
|---|---|---|
| 1 | 1.000000 | 0.000000 |
| 2 | 1.000000 | 0.523599 |
| 3 | 0.862922 | 0.523599 |
| 4 | 0.862922 | 0.499674 |
| 5 | 0.866054 | 0.499674 |
| 6 | 0.866054 | 0.500002 |
| 8 | 0.866025 | 0.500000 |

The sums settle on $$\cos 30° = 0.866025$$ and $$\sin 30° = 0.5$$. Euler's formula earns its place because exponents add when you multiply, $$e^{ia}\,e^{ib} = e^{i(a+b)}$$, which is "turns add" in one symbol.

### RoPE in complex form

One more tool: the conjugate of $$z = a + bi$$, written $$z^*$$, is $$a - bi$$, its mirror image across the horizontal axis, with the angle negated. Write a 2-D query $$[q_1, q_2]$$ as $$q_1 + iq_2$$ and a key the same way, and

$$
(q_1 + iq_2)(k_1 - ik_2) = (q_1k_1 + q_2k_2) + i\,(q_2k_1 - q_1k_2)
$$

The real part is the dot product. With $$\mathbf{q} = [2, 1]$$ and $$\mathbf{k} = [1, 1]$$: $$(2 + i)(1 - i) = 3 - i$$, and indeed $$2 + 1 = 3$$.

RoPE in complex form multiplies the query at position $$m$$ by $$e^{im\theta}$$ and the key at position $$n$$ by $$e^{in\theta}$$. Writing $$\text{Re}[\cdot]$$ for "the real part of", the score is

$$
\text{Re}\big[\,q\,e^{im\theta}\,(k\,e^{in\theta})^*\,\big] = \text{Re}\big[\,q\,k^*\,e^{i(m - n)\theta}\,\big]
$$

The conjugate of $$e^{in\theta}$$ is $$e^{-in\theta}$$, the exponents add to $$(m - n)\theta$$, and the positions are gone except for their difference. That's RoFormer's equation 12, the whole of Step 2's proof in one line. For $$d$$ numbers, the head becomes $$d/2$$ complex numbers $$q_j$$ and $$k_j$$, one per pair, and the score is a sum (RoFormer's equation 35):

$$
\text{score}(m - n) = \text{Re} \sum_{j=0}^{d/2-1} q_j\,k_j^*\,e^{i(m - n)\theta_j}
$$

Meta's reference Llama 3 code works this way: it builds the rotations as complex numbers of length 1 with `torch.polar`, views queries and keys as complex numbers with `torch.view_as_complex`, and multiplies. In NumPy, with Step 2's `q`, `k` and `theta`:

```python
qc = q[0::2] + 1j * q[1::2]       # 64 complex numbers: one per pair
kc = k[0::2] + 1j * k[1::2]
m, n = 107, 103
rotated = np.sum(qc * np.exp(1j * m * theta) * np.conj(kc * np.exp(1j * n * theta)))
offset_only = np.sum(qc * np.conj(kc) * np.exp(1j * (m - n) * theta))
print(f"{rotated.real:.6f} {offset_only.real:.6f}")
```

```text
4.870427 4.870427
```

Both match the 4.870427 from the rotation matrices.

## Step 4: the ladder of frequencies

### Speeds and wavelengths

RoPE's steps follow the same ladder as the original Transformer's sine waves:

$$
\theta_j = \text{base}^{-2j/d}, \qquad j = 0, 1, \dots, d/2 - 1
$$

Pair 0 turns 1 radian per token. Each later pair is slower by the same factor, $$\text{base}^{2/d}$$: 1.155 for base 10,000 with $$d = 128$$, and 1.228 for base 500,000. A pair's wavelength, the number of tokens it takes to come full circle, is

$$
\lambda_j = \frac{2\pi}{\theta_j} = 2\pi \cdot \text{base}^{2j/d}
$$

where $$\lambda$$ is the Greek letter lambda. It runs from $$2\pi \approx 6.3$$ tokens for pair 0 up to almost $$2\pi \times \text{base}$$ for the last pair. For heads of 128 numbers:

| Pair j | Wavelength, base 10,000 (original RoPE) | Base 500,000 (Llama 3) | Base 1,000,000 (Qwen2.5) |
|---|---|---|---|
| 0 | 6.3 | 6.3 | 6.3 |
| 8 | 19.9 | 32.4 | 35.3 |
| 16 | 62.8 | 167.1 | 198.7 |
| 24 | 198.7 | 861.6 | 1,117.3 |
| 32 | 628.3 | 4,442.9 | 6,283.2 |
| 40 | 1,986.9 | 22,910.6 | 35,332.9 |
| 48 | 6,283.2 | 118,142.8 | 198,691.8 |
| 56 | 19,869.2 | 609,226.3 | 1,117,325.9 |
| 63 | 54,410.1 | 2,559,195.5 | 5,063,255.8 |

Fast pairs tell neighbors apart; slow pairs barely move. The base sets how slow the slowest is. [Llama 3](https://arxiv.org/abs/2407.21783) raised it to 500,000, and the [Qwen2.5 report](https://arxiv.org/abs/2412.15115) says Qwen2.5 went "from 10,000 to 1,000,000" when it extended its context from 4,096 to 32,768 tokens; its [config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json) lists `rope_theta: 1000000.0`.

### Long-term decay

RoFormer also argues that RoPE makes attention fade with distance, and the sum above shows why. If a query and a key match well, each term $$q_j k_j^*$$ points roughly the same way, and at offset 0 they all add up. As the offset grows, each term turns at its own speed, they point in different directions, and they partly cancel.

To see it, take a query equal to its key, every pair [1, 1], with base 10,000 and $$d = 128$$. Each term contributes $$2\cos(s\,\theta_j)$$, where $$s$$ is the offset:

| Offset s | 0 | 1 | 10 | 100 | 1,000 | 5,000 | 20,000 |
|---|---|---|---|---|---|---|---|
| Score | 128.0 | 124.2 | 85.6 | 61.1 | 20.4 | −0.9 | −13.6 |

A query matched with an identical key scores 128 at offset 0 and about 20 at 1,000 tokens apart; averaged over 20,000 random queries matched with themselves, the numbers agree to within 0.2. The fall isn't smooth (offset 200 scores 39.1, offset 500 scores 38.2), and it only describes matching pairs: for an unrelated random key the average score is about 0 at every distance.

## Step 5: the math of stretching a context window

### Why extrapolation fails

In training on windows of $$L$$ tokens, pair $$j$$ makes $$r_j = L/\lambda_j$$ turns. A pair with $$r_j$$ above 1 has gone all the way around, so every angle is familiar; a pair with $$r_j$$ below 1 has seen only a slice of its circle. For Llama 3 at $$L = 8{,}192$$, 29 of the 64 pairs never finish a turn, and the slowest covers 1.15 degrees. At 131,072 tokens it would cover 18.4 degrees, 16 times anything it was trained on.

The [position interpolation paper](https://arxiv.org/abs/2306.15595) (Chen et al., 2023) shows why that can be catastrophic. The score is a sum of 64 waves in the offset $$s$$, and such sums can fit almost any curve, including one that's tame up to $$L$$ and wild just past it. Fitting random points with LLaMA's $$d = 128$$, the paper's curve stayed "approximately within [−1, 1]" up to 2,048 and went "beyond 8000" past it. We repeated it with 300 random points: inside the window the fitted curves stayed within ±1.4, and between 2,048 and 4,096 they reached 1,363 to 3,492.

### Position interpolation: shrink the positions

Position interpolation (PI) divides positions down. For a target window $$L'$$ longer than the training window $$L$$, replace each position $$m$$ by

$$
m' = m \cdot \frac{L}{L'}
$$

In words: squeeze the long window back into the range the model knows. With the scale factor $$s = L'/L$$, every speed effectively becomes $$\theta_j / s$$, so no pair reaches an unseen angle; the curve is read between integer offsets instead of past the end.

The cost falls on the fast pairs. At $$s = 8$$, pair 0 turns 0.125 radians (7.2 degrees) between neighbors instead of 1, so adjacent tokens look more alike. The [YaRN paper](https://arxiv.org/abs/2309.00071) reports that PI fine-tunes reached only about $$s = 8$$ "before the LLM's outputs starts to degrade."

### NTK-aware scaling: change the base

The "NTK-aware" method (named after the neural tangent kernel theory that motivated it), first posted on Reddit by a user called bloc97 and written up in the YaRN paper's appendix, changes the base instead. It asks that the fastest pair stay as it was and the slowest pair be slowed by the full factor $$s$$, like PI.

The first comes for free, because $$\theta_0 = \text{base}^0 = 1$$ for any base. For the second, the last pair has exponent $$(d-2)/d$$, so the new base $$b'$$ must satisfy $$b'^{(d-2)/d} = s \cdot b^{(d-2)/d}$$. Solving:

$$
b' = b \cdot s^{\,d/(d-2)}
$$

For base 10,000, $$d = 128$$ and $$s = 4$$, the new base is 40,890. Pair $$j$$ is then slowed by $$s^{2j/(d-2)}$$, a smooth climb from 1 to 4: pair 32 turns 2.02 times slower. Code Llama's jump from base 10,000 to 1,000,000, which YaRN classifies as NTK-aware scaling, slows pair 32 tenfold and the slowest pair 93 times. The weakness, YaRN notes, is that the best base "usually has to be found empirically."

### NTK-by-parts and YaRN

YaRN's authors made the split explicit using the turn count $$r_j$$. Pick two thresholds, $$\alpha$$ (alpha) and $$\beta$$ (beta), and a ramp $$\gamma$$ (gamma) that is 0 below $$\alpha$$, 1 above $$\beta$$, and rises in a straight line in between:

$$
\gamma(r) = \frac{r - \alpha}{\beta - \alpha}, \quad \text{clipped to } [0, 1]
$$

The new speed blends the interpolated and original speeds:

$$
\theta_j' = (1 - \gamma)\,\frac{\theta_j}{s} + \gamma\,\theta_j
$$

A pair that turned more than $$\beta$$ times keeps its speed: it has seen every angle and carries the detail between neighbors. A pair that turned fewer than $$\alpha$$ times is interpolated like PI. For Llama models, the paper found $$\alpha = 1$$ and $$\beta = 32$$ worked well.

![A top-down flowchart for one RoPE pair j with speed theta j. First compute r, the turns it made in the original window L. If r is above beta, it's a fast pair: keep theta j unchanged. Otherwise, if r is below alpha, it's a slow pair: use theta j divided by s, interpolating like PI. Otherwise it's in between: mix theta j and theta j over s according to where r sits between alpha and beta. All three paths then lead to a final step that applies to YaRN only: multiply the attention scores by 1 over t.](/diagrams/m12-rope-by-parts.svg "NTK-by-parts, the rule YaRN and Llama 3.1 use to rescale each RoPE pair.")
YaRN adds one more piece: it multiplies the scores by $$1/t$$ before the softmax, where $$t$$ plays the role of the temperature from [Part 5](/blog/softmax-math/), to sharpen attention over the longer window. It scales queries and keys by $$\sqrt{1/t}$$ each, folded into the rotation's cosines and sines at no cost, and recommends

$$
\sqrt{1/t} = 0.1 \ln s + 1
$$

where $$\ln$$ is the natural logarithm. At $$s = 8$$ that's 1.208, as the paper computes, so the scores grow by $$1.208^2 = 1.46$$; at $$s = 32$$ they grow by 1.81.

### Llama 3.1's rope_scaling, computed

Now the five numbers from the opening. Hugging Face's `_compute_llama3_parameters`, which mirrors Meta's `apply_scaling`, leaves alone pairs with wavelengths under `8192 / high_freq_factor` = 2,048 tokens, divides by `factor` = 8 the speed of pairs with wavelengths over `8192 / low_freq_factor` = 8,192, and blends in between with `smooth = (8192 / wavelen - 1) / (4 - 1)`. Since $$8192/\lambda_j$$ is the turn count $$r_j$$, that's NTK-by-parts with $$s = 8$$, $$\alpha = 1$$ and $$\beta = 4$$, and no temperature (`attention_factor = 1.0`).

```python
import numpy as np

d, base = 128, 500_000                               # Llama 3.1 8B: head_dim, rope_theta
factor, low, high, L = 8, 1, 4, 8192                 # its rope_scaling block
theta = base ** (-np.arange(0, d, 2) / d)
r = L / (2 * np.pi / theta)                          # turns each pair makes in 8,192 tokens
gamma = np.clip((r - low) / (high - low), 0, 1)      # 0 = slow pair, 1 = fast pair
new = (1 - gamma) * theta / factor + gamma * theta   # blend of interpolated and original

for j in [0, 20, 28, 29, 30, 32, 34, 35, 63]:
    print(f"pair {j:2d}: {r[j]:9.3f} turns in 8K, gamma {gamma[j]:.3f}, "
          f"slowed {theta[j] / new[j]:.2f}x, {131072 * new[j] / (2 * np.pi):9.3f} turns in 128K")
```

```text
pair  0:  1303.797 turns in 8K, gamma 1.000, slowed 1.00x, 20860.757 turns in 128K
pair 20:    21.591 turns in 8K, gamma 1.000, slowed 1.00x,   345.463 turns in 128K
pair 28:     4.187 turns in 8K, gamma 1.000, slowed 1.00x,    66.993 turns in 128K
pair 29:     3.411 turns in 8K, gamma 0.804, slowed 1.21x,    45.196 turns in 128K
pair 30:     2.779 turns in 8K, gamma 0.593, slowed 1.55x,    28.619 turns in 128K
pair 32:     1.844 turns in 8K, gamma 0.281, slowed 2.69x,    10.949 turns in 128K
pair 34:     1.224 turns in 8K, gamma 0.075, slowed 5.26x,     3.724 turns in 128K
pair 35:     0.997 turns in 8K, gamma 0.000, slowed 8.00x,     1.993 turns in 128K
pair 63:     0.003 turns in 8K, gamma 0.000, slowed 8.00x,     0.006 turns in 128K
```

These match Hugging Face's own function to float32 rounding. Pairs 0 to 28 keep their speed, 29 to 34 are blended, and 35 to 63 turn 8 times slower. The factor is 8 while the window grew 16 times, so a slowed pair still covers twice its training range at 131,072 tokens. The model also met long inputs in training: Llama 3 increased its context "in six stages" from 8K to 128K.

## Step 6: ALiBi, a straight-line penalty

[ALiBi](https://arxiv.org/abs/2108.12409) (Attention with Linear Biases; Press, Smith and Lewis, 2021) skips rotations. Each head subtracts a penalty proportional to distance from every score:

$$
\text{softmax}\big(\mathbf{q}_i K^{\top} + m \cdot [-(i-1), \dots, -2, -1, 0]\big)
$$

Here $$\mathbf{q}_i$$ is the query at position $$i$$, $$K$$ holds the keys of positions 1 to $$i$$ as rows, and $$m$$ is the head's slope, fixed before training. For $$n$$ heads the slopes are a geometric sequence that "starts at $$2^{-8/n}$$ and uses that same value as its ratio": head $$h$$ gets $$2^{-8h/n}$$.

Push the penalty through the softmax and it becomes a multiplication. The weight on a key $$\Delta$$ tokens back, $$\Delta$$ standing for the distance, gets a factor $$e^{-m\Delta}$$, so each head forgets at a steady exponential rate. Its half-life, the distance at which a key keeps half the weight it would have had next door, is $$\ln 2 / m$$. With 32 heads, the slopes run from $$2^{-1/4} = 0.841$$ to $$2^{-8} = 0.0039$$:

| Head (of 32) | Slope m | Half-life in tokens | Weight factor 100 tokens back | 1,000 back |
|---|---|---|---|---|
| 1 | 0.841 | 0.8 | 3 × 10⁻³⁷ | 0 |
| 8 | 0.25 | 2.8 | 1.4 × 10⁻¹¹ | 3 × 10⁻¹⁰⁹ |
| 16 | 0.0625 | 11.1 | 0.0019 | 7 × 10⁻²⁸ |
| 24 | 0.0156 | 44.4 | 0.21 | 1.6 × 10⁻⁷ |
| 32 | 0.0039 | 177.4 | 0.68 | 0.020 |

Part 5's shift invariance gives a shortcut. Within one query's row, $$-m(i - j) = -mi + mj$$, and $$-mi$$ is the same for every key $$j$$, so the softmax ignores it: adding just $$m \cdot j$$ gives identical weights. Hugging Face's BLOOM code does exactly that, noting that it "relies on a translation invariance of softmax."

Both schemes depend only on the offset. ALiBi's dependence is a fixed straight line that always favors recent tokens. RoPE's is learned: the 64 waves get their sizes and phases $$q_j k_j^*$$ from the content, so a head can learn to look 1 token back, 50 back, or anywhere. That built-in recency suits inputs longer than the training length, which [Fundamentals Part 7](/blog/positional-encoding/) showed ALiBi handles well; the cost is that its steep heads give distant tokens almost no weight.

## In the real world

| Model | Position method | Base | Rotated numbers per head | Context extension |
|---|---|---|---|---|
| Llama 3.1 8B | RoPE | 500,000 | 128 of 128 | `llama3`, factor 8, 8,192 → 131,072 |
| Llama 4 Scout | RoPE in 3 of every 4 layers, none in the 4th | 500,000 | 128 of 128 | `llama3`, factor 16, from 8,192 |
| Qwen2.5-7B | RoPE | 1,000,000 | 128 of 128 | YaRN factor 4, 32,768 → 131,072, opt-in |
| Gemma 3 27B | RoPE | 1,000,000 global, 10,000 local | 128 of 128 | linear (PI), factor 8 on global layers |
| gpt-oss-120b | RoPE | 150,000 | 64 of 64 | YaRN factor 32, from 4,096 |
| DeepSeek-V3 | RoPE on a separate slice | 10,000 | 64 of 192 | YaRN factor 40, from 4,096 |
| GPT-NeoX-20B | RoPE on part of each head | 10,000 | 24 of 96 | none |
| BLOOM 176B | ALiBi | none | none | none |

Sources: the Hugging Face configs of [Llama 3.1 8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json), [Llama 4 Scout](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json), [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (model card), [Gemma 3 27B](https://huggingface.co/unsloth/gemma-3-27b-it/blob/main/config.json), [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json), [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b/blob/main/config.json) and [BLOOM](https://huggingface.co/bigscience/bloom/blob/main/config.json).

- Llama 4 Scout's `no_rope_layers` list has a 0 in every fourth entry, and Hugging Face's configuration code says "a `0` indicates that it's a NoPE layer", a layer with no position encoding at all.
- The [Gemma 3 report](https://arxiv.org/abs/2503.19786) raised the base to 1,000,000 only in its global layers and stretched them from 32K to 128K tokens "while rescaling RoPE (Chen et al., 2023). We find a scaling factor of 8 to work well in practice." That's plain position interpolation in a 2025 model.
- Partial RoPE rotates only some of each head's numbers. GPT-NeoX-20B sets `rotary_pct` to 0.25, [Phi-2](https://huggingface.co/microsoft/phi-2/blob/main/config.json) sets `partial_rotary_factor` to 0.4, and DeepSeek-V3 gives each query and key 128 unrotated numbers plus 64 rotated ones, for the compressed cache described in [The KV Cache](/blog/kv-cache/).
- Meta's reference code pairs neighbors, (0, 1), (2, 3) and so on. Hugging Face's [Llama code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) computes `q * cos + rotate_half(q) * sin`, where `rotate_half` returns `torch.cat((-x2, x1), dim=-1)`, so it pairs number $$i$$ with number $$i + d/2$$. That's the same math with the numbers reordered. The [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) reorders the rows of the query and key weights once with a `permute` function, and a dot product doesn't care how both vectors are reordered, so every score is unchanged. A check:

```python
import torch

def rotate_half(x):                       # copied from Hugging Face's Llama code
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

d, pos = 8, 3
inv_freq = 10_000 ** (-torch.arange(0, d, 2) / d)
ang = pos * inv_freq
x = torch.randn(d, generator=torch.Generator().manual_seed(0))

# Meta's layout: adjacent pairs (x0, x1), (x2, x3), ... as complex numbers
meta = torch.view_as_real(torch.view_as_complex(x.reshape(-1, 2)) * torch.polar(torch.ones(d // 2), ang)).flatten()

# Hugging Face's layout: pair i is (x[i], x[i + d/2])
perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])   # [0, 2, 4, 6, 1, 3, 5, 7]
xh = x[perm]
emb = torch.cat([ang, ang])
hf = xh * emb.cos() + rotate_half(xh) * emb.sin()

print(perm.tolist())
print(torch.allclose(hf, meta[perm], atol=1e-6))
```

```text
[0, 2, 4, 6, 1, 3, 5, 7]
True
```

## Common misconceptions

- **"RoPE adds a position vector to each token."** It multiplies: each pair is turned by an angle, so lengths stay the same. It has no learned parameters, and values are never rotated.
- **"RoPE needs complex arithmetic."** Complex numbers are bookkeeping for 2 × 2 rotations. Hugging Face's code uses real cosines and sines and gets the same numbers as Meta's complex version.
- **"Since the score depends only on distance, a RoPE model handles any length."** Offsets beyond the training window put the slow pairs at angles the model never saw, and a sum of waves fitted on the window can blow up just past it.
- **"Context extension squeezes every dimension."** Only position interpolation does. NTK-aware scaling, YaRN and Llama 3.1's scaling leave the fast pairs alone, because they carry the detail between neighbors.

## Key takeaways

- A rotation matrix holds cosines and sines because that's where the basis arrows land. Rotations keep lengths and angles, turns add, and the transpose undoes a turn.
- So a query rotated for position m and a key rotated for position n score like the original query against the key turned by n minus m. Only the offset survives.
- In complex numbers, multiplying by e to the i theta is a rotation, and the conjugate makes the two positions subtract.
- RoPE's 64 speeds run from one radian per token down to almost nothing. Pairs that never finish a turn in training break first when the context grows.
- Position interpolation slows every pair, NTK-aware scaling raises the base, and YaRN and Llama 3.1 slow only the slow pairs. ALiBi swaps all of it for a straight-line penalty.

## What's next

Rotations are one of the two building blocks in the most useful factorization in linear algebra. [Part 13](/blog/low-rank-svd/), Low rank and the SVD, shows that every matrix is a rotation, a stretch along the axes, and another rotation (allowing mirror flips), and uses that to explain why LoRA's updates can be small, how latent attention compresses the KV cache, and what a real weight matrix looks like when you take it apart.

## Further reading

- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864): the 2-D complex derivation (section 3.2), the general rotation matrix (equations 14 to 16) and the long-term decay argument (section 3.4.3).
- [Extending Context Window of Large Language Models via Positional Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595): the score as a sum of waves, why extrapolation blows up, and the interpolation bound.
- [YaRN: Efficient Context Window Extension of Large Language Models (Peng et al., 2023)](https://arxiv.org/abs/2309.00071): wavelengths, NTK-aware scaling (appendix A.2), NTK-by-parts and the attention temperature.
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (Press, Smith & Lewis, 2021)](https://arxiv.org/abs/2108.12409): ALiBi and its slopes.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): RoPE base 500,000 and the six-stage climb to 128K tokens.
- [Qwen2.5 Technical Report (Qwen Team, 2024)](https://arxiv.org/abs/2412.15115): raising the base to 1,000,000, and YaRN for 131,072 tokens.
- [Gemma 3 Technical Report (Gemma Team, 2025)](https://arxiv.org/abs/2503.19786): different bases for local and global layers, and position interpolation by a factor of 8.
- [Code Llama: Open Foundation Models for Code (Rozière et al., 2023)](https://arxiv.org/abs/2308.12950): raising the base from 10,000 to 1,000,000 for long inputs.
- [Hugging Face modeling_rope_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py): the `linear`, `dynamic`, `yarn` and `llama3` scalings in a few lines each.
- [Meta's Llama 3 reference model.py](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py): RoPE as complex multiplication, and `apply_scaling`.
- [Rotary Embeddings: A Relative Revolution (EleutherAI, 2021)](https://blog.eleuther.ai/rotary-embeddings/): an accessible walk through RoPE, with code.
- [Essence of linear algebra (3Blue1Brown)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab): animations of matrices as moves of the plane, including rotations.
