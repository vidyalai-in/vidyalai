---
title: "The Modern LLM Block: What Changed Since 2017"
description: "RMSNorm, SwiGLU, no biases, grouped-query attention and RoPE: GPT-2's block next to Llama 3's and Qwen3's, with a full 8.03B parameter count."
slug: "modern-llm-architecture"
category: "llm-fundamentals"
tags: ["transformers", "attention", "architecture", "normalization"]
series: "llm-fundamentals"
seriesOrder: 10
pubDate: 2026-03-26
---

## TL;DR

The block inside a 2025 language model has the same skeleton as GPT-2's: attention, then an MLP, each reading a normalized copy of the residual stream and adding its result back. What changed is the parts. Layer norm became RMSNorm, which skips the mean. The MLP became SwiGLU, which uses three matrices and a gate. Biases disappeared. Attention heads started sharing keys and values (grouped-query attention), or compressing them (multi-head latent attention), and positions moved inside attention as rotations (RoPE). By the end you'll know what each change does, why it was made, and you'll be able to count all 8,030,261,248 parameters of Llama 3.1 8B from its config file.

## Why it matters

Take GPT-2's 2019 block, change five things, and you have the block inside Llama 3 and Qwen3. Each change is small. One deletes a subtraction from the norm, another adds a third matrix to the MLP, and a third deletes every bias in the model, which in Llama 3.1 8B would have been 1.4 million numbers out of 8 billion.

Together, though, they affect how fast a model trains, whether a large training run blows up, and how many users fit on one GPU. Grouped-query attention alone cuts Llama 3.1 8B's key-value cache to a quarter of what full multi-head attention would need, and it cuts attention's weights by 37.5% on the way.

[Part 6](/blog/transformer-block/) built GPT-2's block and counted its 124 million parameters. [Part 8](/blog/attention-is-all-you-need/) listed which 2017 choices survived. This part opens the 2024 and 2025 blocks and goes through every line that differs.

## The intuition

Think of a car model that has been in production for years. The layout never changed: engine in front, four wheels, a driver's seat. But the carburetor became fuel injection, the steel panels became aluminium, and the bolts that turned out to be unnecessary are gone. A mechanic from the first year would recognize the car and still need a new manual.

The layout here is the pre-norm residual block: normalize, attend, add; normalize, MLP, add. The parts are the norm, the MLP's activation, the attention's head layout and the position scheme.

The analogy breaks on one point. Car parts get swapped because engineers understand why the new one is better. Several of these swaps were found by trying variants and keeping the winner. The paper that introduced SwiGLU ends with: "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

## Step 1: two blocks, side by side

Here are three models from three generations, line by line. Llama 3.1 8B and Qwen3-8B share the same width, 4,096, which makes them easy to compare.

| Component | GPT-2 small (2019) | Llama 3.1 8B (2024) | Qwen3-8B (2025) |
|---|---|---|---|
| Layers × width | 12 × 768 | 32 × 4,096 | 36 × 4,096 |
| Norm | LayerNorm (gain and bias) | RMSNorm (gain only) | RMSNorm, plus QK-norm in attention |
| Norm placement | Pre-norm | Pre-norm | Pre-norm |
| Attention | Multi-head, 12 heads | Grouped-query, 32 query heads, 8 key-value heads | Grouped-query, 32 query heads, 8 key-value heads |
| Position | Learned table of 1,024 slots, added at input | RoPE inside attention, base 500,000 | RoPE inside attention, base 1,000,000 |
| MLP | GELU, 768 → 3,072 → 768 (4×) | SwiGLU, three matrices, hidden 14,336 (3.5×) | SwiGLU, three matrices, hidden 12,288 (3×) |
| Biases | In every linear layer and norm | None | None |
| Vocabulary (embedding rows) | 50,257 | 128,256 | 151,936 |
| Input and output tables | Tied (one shared table) | Untied | Untied |
| Context | 1,024 tokens | 128K tokens | 32,768 native, 131,072 with YaRN |
| Parameters | 124.4M | 8.03B | 8.19B |

Sources: GPT-2's [config](https://huggingface.co/openai-community/gpt2/blob/main/config.json); Llama 3.1 8B's [reference code](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py), [model arguments](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py) and the [Llama 3 paper](https://arxiv.org/abs/2407.21783) Table 3; Qwen3-8B's [config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json), [model card](https://huggingface.co/Qwen/Qwen3-8B) and the [Qwen3 report](https://arxiv.org/abs/2505.09388) section 2. Parameter totals are our counts, and they match each model's published weight files to the parameter.

The Qwen3 report sums up the recipe in one sentence: its dense models use "Grouped Query Attention (GQA...), SwiGLU..., Rotary Positional Embeddings (RoPE...), and RMSNorm... with pre-normalization." Every item in that list was published between 2019 and 2023. Pre-norm and RoPE are covered in [Part 6](/blog/transformer-block/) and [Part 7](/blog/positional-encoding/). The rest of this article takes the other rows one at a time.

## Step 2: RMSNorm, and normalizing queries and keys

[Part 6](/blog/transformer-block/) worked layer norm on `x = [2, 4, 6, 8]`: subtract the mean (5), divide by the standard deviation (`√5 ≈ 2.236`), and get [-1.342, -0.447, 0.447, 1.342]. Then multiply by a learned gain and add a learned bias.

[Zhang and Sennrich (2019)](https://arxiv.org/abs/1910.07467) asked whether the first step, subtracting the mean, is needed at all. Layer norm makes a vector immune to two things: shifting every number by the same amount (re-centering) and multiplying every number by the same amount (re-scaling). They "hypothesize that re-centering invariance in LayerNorm is dispensable" and keep only the re-scaling. The result is root mean square layer normalization, RMSNorm:

```text
RMSNorm(x) = g × x / RMS(x),   where RMS(x) = √(mean of x²)
```

Square every number, average the squares, take the square root, and divide. No mean is subtracted and no bias is added; `g` is a learned gain per position, as in layer norm. Real code adds a tiny `ε` inside the square root to avoid dividing by zero (0.00001 in Llama 3.1 8B).

### The same vector, both ways

Take Part 6's vector again:

1. The squares are 4, 16, 36 and 64. Their mean is `120 / 4 = 30`.
2. `RMS = √30 ≈ 5.477`.
3. Divide: [0.365, 0.730, 1.095, 1.461].

| Input | LayerNorm output | RMSNorm output |
|---|---|---|
| [2, 4, 6, 8] | [-1.342, -0.447, 0.447, 1.342] | [0.365, 0.730, 1.095, 1.461] |
| [20, 40, 60, 80] (×10) | [-1.342, -0.447, 0.447, 1.342] | [0.365, 0.730, 1.095, 1.461] |
| [12, 14, 16, 18] (+10) | [-1.342, -0.447, 0.447, 1.342] | [0.791, 0.923, 1.055, 1.187] |
| [-3, -1, 1, 3] (mean 0) | [-1.342, -0.447, 0.447, 1.342] | [-1.342, -0.447, 0.447, 1.342] |

The outputs differ, and that's fine. Both ignore the overall size: multiplying by 10 changes nothing. Only layer norm ignores a shift. When the vector already has a mean of zero, the two agree exactly; in the paper's words, "When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm."

### What it saves

Dropping the mean saves one pass over the vector and a subtraction. Across their experiments, Zhang and Sennrich report that RMSNorm "achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models." The big speedups were in recurrent networks, which normalize at every time step. For a Transformer on English-German translation, the gain was smaller: 231 seconds per 1,000 training steps instead of 248, about 7% faster, with BLEU (a translation score) of 26.8 against layer norm's 26.6.

The gain is modest, but quality held, so [LLaMA](https://arxiv.org/abs/2302.13971) (2023) adopted it: "we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing function." Meta's Llama 3 [reference code](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py) writes the whole thing in one line, `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`, times a learned weight.

### QK-norm: one more norm, inside attention

Qwen3 adds a norm in a new place. Its report says it removed "QKV-bias used in Qwen2" and introduced "QK-Norm (Dehghani et al., 2023) to the attention mechanism to ensure stable training." QK-norm applies RMSNorm to each head's query and key vectors before their dot product. In Qwen3-8B that's two small norms of 128 gains per layer.

The problem it fixes showed up when Google scaled a vision Transformer. [Dehghani et al. (2023)](https://arxiv.org/abs/2302.05442) saw training loss diverge after a few thousand steps in an 8-billion-parameter model: "Without normalization, attention logits quickly grow to over 50000 in magnitude, resulting in one-hot attention weights after the softmax." A score of 50,000 next to scores near zero puts all the weight on a single token, and gradients through that softmax become unstable.

Normalization puts a ceiling on those scores. After RMSNorm, with the gains at their starting value of 1, a 128-number query or key has a length of `√128 ≈ 11.3`. Their dot product can be at most `128`, and after the usual division by `√128` the score is at most 11.3. The learned gains can raise that ceiling, but the scores can no longer run away on their own. [OLMo 2](https://arxiv.org/abs/2501.00656) uses QK-norm for the same reason: it "avoids attention logits being too large, which can lead to training loss divergence."

### No biases

The last change in this step is a deletion. GPT-2 adds a bias, a learned constant, after every matrix and inside every layer norm. Llama 3's reference code passes `bias=False` to every linear layer, and RMSNorm has no bias to begin with. Google's [PaLM](https://arxiv.org/abs/2204.02311) gave the reason: "No biases were used in any of the dense kernels or layer norms. We found this to result in increased training stability for large models."

The parameter savings are tiny. Biases on Llama 3.1 8B's seven matrices per block would add `43,008 × 32 = 1,376,256` numbers, 0.017% of the model.

## Step 3: SwiGLU, a gated MLP

GPT-2's MLP has two matrices with GELU between them: widen 768 numbers to 3,072, apply the activation to each, shrink back. Every middle number goes through the same fixed curve.

A gated linear unit (GLU), from [Dauphin et al. (2016)](https://arxiv.org/abs/1612.08083), computes two projections of the input instead of one and multiplies them number by number. One projection, the gate, passes through an activation and decides how much of the other, the "up" projection, gets through. [Shazeer (2020)](https://arxiv.org/abs/2002.05202) tried the gate with several activations inside the Transformer's MLP and named the variants: ReGLU (ReLU), GEGLU (GELU) and SwiGLU, which uses Swish. Swish with its default setting is also called SiLU (sigmoid linear unit): `SiLU(z) = z × sigmoid(z)`. It's close to ReLU, but smooth and slightly negative for small negative inputs. The full MLP is:

```text
FFN_SwiGLU(x) = (SiLU(x W_gate) ⊙ x W_up) W_down
```

`⊙` means multiply position by position. Llama's reference code is the same line: `self.w2(F.silu(self.w1(x)) * self.w3(x))`.

### A tiny worked example

Take a 2-number input `x = [1, 2]` and an MLP with 2 hidden units, small enough to do by hand. The matrices are hand-picked so each product is easy:

| Hidden unit | Gate `x · w_gate` | SiLU(gate) | Up `x · w_up` | Gate × up |
|---|---|---|---|---|
| 1 | `[1, 2] · [0, 1] = 2` | 1.762 | `[1, 2] · [1, 1] = 3` | 5.285 |
| 2 | `[1, 2] · [1, -1] = -1` | -0.269 | `[1, 2] · [1, 0] = 1` | -0.269 |

Unit 1's gate is open, so its up value of 3 passes at nearly double strength. Unit 2's gate is shut, so its up value is mostly blocked. `W_down` then maps [5.285, -0.269] back to the model's width. In GPT-2's MLP, how much a hidden unit fires depends on one projection of the input. In SwiGLU it depends on the product of two, so the input can switch a unit's output on or off in a way one curve can't.

### Did it help?

Shazeer tested each variant on T5's text-to-text setup, with the parameters and computation held equal. The score is log-perplexity on held-out text after pretraining: lower is better.

| MLP type | After 65,536 steps | After 524,288 steps |
|---|---|---|
| ReLU (baseline) | 1.997 | 1.677 |
| GELU | 1.983 | 1.679 |
| Swish | 1.994 | 1.683 |
| GLU (sigmoid gate) | 1.982 | 1.663 |
| Bilinear (no activation) | 1.960 | 1.648 |
| ReGLU | 1.953 | 1.645 |
| GEGLU | 1.942 | 1.633 |
| SwiGLU | 1.944 | 1.636 |

Source: [Shazeer (2020)](https://arxiv.org/abs/2002.05202), Table 1.

Every gated version beat every ungated one. At the end of training, SwiGLU's 1.636 is a perplexity of 5.13 against ReLU's 5.35, about 4% lower. After fine-tuning, SwiGLU's average GLUE score (a benchmark of language-understanding tasks) rose from 83.80 to 84.36. PaLM adopted SwiGLU because it was shown "to significantly increase quality compared to standard ReLU, GeLU, or Swish activations," and LLaMA took it from PaLM.

### Why the hidden width is 2/3 of 4d

A third matrix costs parameters. GPT-2's MLP has two matrices of `d × 4d`, which is `8d²`. SwiGLU has three matrices of `d × h`, which is `3dh`. Setting them equal:

```text
3 × d × h = 8 × d²   →   h = 8d / 3 = (2/3) × 4d
```

That's why Shazeer's T5 experiments cut the hidden width from 3,072 to 2,048, and why the LLaMA paper says "We use a dimension of 2/3 4d instead of 4d." For `d = 4,096`, `2/3 × 16,384 = 10,922.7`. GPU matrix kernels run fastest on round sizes, so the code rounds up to a multiple: LLaMA 7B's [code](https://github.com/meta-llama/llama/blob/llama_v1/llama/model.py) rounds up to a multiple of 256 and gets 11,008.

Llama 3.1 8B's 14,336 is not that number. Meta's reference code builds it in three steps from `4 × dim`:

```python
hidden_dim = int(2 * (4 * 4096) / 3)       # 10,922  (the 2/3 rule)
hidden_dim = int(1.3 * hidden_dim)         # 14,198  (ffn_dim_multiplier = 1.3)
hidden_dim = 1024 * ((hidden_dim + 1023) // 1024)   # 14,336  (multiple_of = 1024)
```

`14,336 / 4,096 = 3.5`, so the hidden layer is 3.5 times the width, not 4. Its MLP holds `3 × 4,096 × 14,336 = 176,160,768` parameters, 31% more than a GELU MLP of `8 × 4,096² = 134,217,728`. Llama 3 70B uses the same 3.5 ratio (28,672 for a width of 8,192), and 405B uses 3.25 (53,248 for 16,384). Qwen3-8B uses exactly 3 (12,288). The "2/3 of 4d" rule is a starting point, and each lab tunes the multiplier.

## Step 4: attention that shares, compresses and windows

The attention formula from [Part 5](/blog/attention-explained/) hasn't changed. What changed is how many key and value vectors each layer produces, because during generation every one of them is stored in the [KV cache](/blog/kv-cache/) for every earlier token. Four designs, all sized here for a body with 32 query heads of 128 numbers:

| Design | How keys and values are shared | Numbers cached per token per layer | Used in | Paper |
|---|---|---|---|---|
| Multi-head attention (MHA) | Each of 32 heads has its own key and value | 8,192 | GPT-2, Llama 2 7B | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) |
| Multi-query attention (MQA) | All 32 query heads share 1 key and 1 value | 256 | PaLM, Falcon 7B | [Shazeer, 2019](https://arxiv.org/abs/1911.02150) |
| Grouped-query attention (GQA) | Groups of 4 query heads share one pair: 8 pairs | 2,048 | Llama 3, Qwen3, Mistral 7B, Gemma 3 | [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245) |
| Multi-head latent attention (MLA) | Keys and values rebuilt from one cached 512-number latent, plus a 64-number position key | 576 | DeepSeek-V2, DeepSeek-V3 | [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434) |

The MLA row uses DeepSeek's own sizes. [The KV cache](/blog/kv-cache/) works through what each design means in gigabytes and in users per GPU, and how MLA folds its rebuild step into other matrices.

Two things are easy to miss. First, GQA shrinks the weights as well as the cache. Llama 3.1 8B's key and value matrices are `4,096 × 1,024` instead of `4,096 × 4,096`, so attention holds 41.9 million parameters per block instead of 67.1 million, 37.5% fewer. Second, the MLP now dominates even more: attention is 19% of Llama's block, against 33% of GPT-2's.

### Sliding windows and local-global layers

Another way to bound the cache is to stop each token looking all the way back. In sliding window attention, each token attends only to the last `W` tokens. [Mistral 7B](https://arxiv.org/abs/2310.06825) used `W = 4,096`. Information still travels further, one window per layer, so after 32 layers a token can in principle draw on about 131K tokens back (`4,096 × 32`). Its cache only ever holds the last 4,096 positions, which Mistral reports "reduces the cache memory usage by 8x" on 32K-token sequences, "without impacting the model quality."

Google's Gemma models mix the two kinds of layer. [Gemma 2](https://arxiv.org/abs/2408.00118) alternated local layers with a 4,096-token window and global layers that see everything. [Gemma 3](https://arxiv.org/abs/2503.19786) pushed the ratio to "1 global for every 5 local layers" and shrank the window to "only 1024 tokens," so that "only the global layers attend to long context." By our arithmetic, at 128K tokens of context and with every layer caching the same amount per token, that pattern stores about 17% of what all-global layers would, roughly 5.8 times less. Serving these patterns well is the subject of the Inference series' part on long-context inference.

### Position: RoPE, in one paragraph

GPT-2 added a learned position vector to each token at the input. Llama, Qwen3, Gemma and DeepSeek use RoPE (rotary position embedding), which rotates each query and key by an angle proportional to its position, so the attention score depends on how far apart two tokens are. [Part 7](/blog/positional-encoding/) works through it by hand. The base, which sets how slowly the slowest rotations turn, keeps rising for longer contexts: 10,000 in the original paper, 500,000 in Llama 3, 1,000,000 in Qwen3 and in Gemma 3's global layers. MLA needs a special case: compressed keys can't carry the rotation, so DeepSeek caches a separate 64-number key just for position, the extra 64 in the table above.

## Step 5: the whole block, and every parameter

![A left-to-right flow through one Llama 3.1 8B block. The residual stream comes in and splits. One path goes through RMSNorm and grouped-query attention with 32 query heads, 8 key-value heads and RoPE on queries and keys, then reaches an add node, where the skip path joins it unchanged. From that sum, one path goes through a second RMSNorm and splits into a gate projection and an up projection, each from 4,096 to 14,336 numbers. The gate passes through SiLU, the two are multiplied position by position, and a down projection maps 14,336 numbers back to 4,096. A second add node joins a second skip path, giving the residual stream out.](/diagrams/f10-modern-block.svg "One Llama 3.1 8B block: pre-RMSNorm, grouped-query attention, and a SwiGLU MLP with gate, up and down projections.")

The skeleton is GPT-2's; the MLP now has two parallel paths that meet at a multiply.

In code, the block is still two lines. From Meta's [Llama 3 reference code](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py):

```python
h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
out = h + self.feed_forward(self.ffn_norm(h))
```

Set it beside the nanoGPT block in Part 6 and the structure is identical: normalize, sublayer, add, twice.

### Counting Llama 3.1 8B

Everything needed is in the [config](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json): `hidden_size` 4,096, `num_hidden_layers` 32, `num_attention_heads` 32, `num_key_value_heads` 8, `head_dim` 128, `intermediate_size` 14,336, `vocab_size` 128,256, `tie_word_embeddings: false`. One block:

| Part of the block | Shape | Parameters |
|---|---|---|
| RMSNorm before attention | 4,096 gains | 4,096 |
| Query projection | 4,096 × (32 × 128) | 16,777,216 |
| Key projection | 4,096 × (8 × 128) | 4,194,304 |
| Value projection | 4,096 × (8 × 128) | 4,194,304 |
| Output projection | 4,096 × 4,096 | 16,777,216 |
| RMSNorm before the MLP | 4,096 gains | 4,096 |
| Gate projection | 4,096 × 14,336 | 58,720,256 |
| Up projection | 4,096 × 14,336 | 58,720,256 |
| Down projection | 14,336 × 4,096 | 58,720,256 |
| **One block** | | **218,112,000** |

Attention is 41,943,040 (19.2%), the MLP 176,160,768 (80.8%), and the two norms 8,192. Now the whole model:

| Part of the model | Parameters | Share |
|---|---|---|
| 32 blocks × 218,112,000 | 6,979,584,000 | 86.9% |
| Token embeddings, 128,256 × 4,096 | 525,336,576 | 6.5% |
| Output table (untied), 128,256 × 4,096 | 525,336,576 | 6.5% |
| Final RMSNorm | 4,096 | 0.00005% |
| **Total** | **8,030,261,248** | |

That's the "8B". The published weights are 16,060,522,496 bytes in 16-bit format, exactly two bytes for each of these parameters. There's no position table at all, since RoPE has no learned weights. The same count for Qwen3-8B, with 36 blocks, a 12,288-wide MLP, QK-norm and a 151,936-row vocabulary, gives 8,190,735,360, which also matches its weight files exactly.

### Two choices that vary

Tied or untied tables. [Part 4](/blog/embeddings-and-logits/) showed that one table can serve as both the input embeddings and the output layer. GPT-2 ties them; Llama 3.1 8B doesn't, and spends 525 million parameters, 6.5% of the model, on the second table. Qwen3 ties them in its 0.6B, 1.7B and 4B models and unties them from 8B up ([Qwen3 report](https://arxiv.org/abs/2505.09388), Table 1).

Parallel layers. The block above is serial: the MLP reads the stream after attention has added to it. GPT-J (2021) and PaLM run both halves on the same normalized input and add both results at once. PaLM writes the two forms as:

```text
serial:    y = x + MLP(LayerNorm(x + Attention(LayerNorm(x))))
parallel:  y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x))
```

PaLM reports that "the parallel formulation results in roughly 15% faster training speed at large scales, since the MLP and Attention input matrix multiplications can be fused." The catch: "Ablation experiments showed a small quality degradation at 8B scale but no quality degradation at 62B scale." Llama 3 and Qwen3 stayed serial.

## In the real world

- Norm placement is moving again. [OLMo 2](https://arxiv.org/abs/2501.00656) normalizes each sublayer's output instead of its input, `h = x + RMSNorm(Attention(x))`, and adds QK-norm. Together, the two changes cut a measure of gradient spikes from 0.108 to 0.069; "in isolation, neither of these changes yield good results." [Gemma 3](https://arxiv.org/abs/2503.19786) uses RMSNorm both before and after each sublayer, and replaced Gemma 2's soft-capping with QK-norm.
- MLP ratios vary by lab and size. Llama 3 uses 3.5 times the width at 8B and 70B and 3.25 at 405B ([Llama 3 paper](https://arxiv.org/abs/2407.21783), Table 3). Qwen3 uses 3 at 8B and 5 at 32B (25,600 for a width of 5,120, [config](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json)).
- Mistral 7B is shaped like Llama, plus a window. Its paper lists 32 layers, width 4,096, 32 query heads, 8 key-value heads, a hidden MLP width of 14,336 and a 4,096-token sliding window ([Mistral 7B](https://arxiv.org/abs/2310.06825), Table 1).
- DeepSeek-V3 pairs MLA with a different kind of MLP. Its [config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) has 61 layers of width 7,168 with 128 attention heads and RMSNorm. The first 3 layers use an ordinary dense MLP of 18,432; the other 58 replace it with 256 small experts and 1 shared expert, 8 chosen per token. The [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) gives 671 billion parameters, with 37 billion active for each token.
- Attention libraries support these variants directly. The [FlashAttention](https://github.com/Dao-AILab/flash-attention) kernels accept fewer key-value heads than query heads for MQA and GQA, and a `window_size` argument for sliding window attention. [FlashAttention](/blog/flashattention/) explains what those kernels do.

## Common misconceptions

- **"RMSNorm is just a faster layer norm that gives the same output."** It gives the same output only when the vector's mean is already zero. On [2, 4, 6, 8], layer norm gives [-1.342, -0.447, 0.447, 1.342] and RMSNorm gives [0.365, 0.730, 1.095, 1.461]. Models are trained with one or the other; they aren't interchangeable after training.
- **"SwiGLU's MLP is smaller because its hidden layer is narrower."** It has three matrices instead of two. Llama 3.1 8B's 14,336-wide SwiGLU MLP holds 31% more parameters than a 16,384-wide GELU MLP at the same model width.
- **"Grouped-query attention only changes the cache."** It also removes weights. Llama 3.1 8B's attention holds 41.9 million parameters per block, against 67.1 million for full multi-head attention at the same width.
- **"Dropping biases was about saving memory."** Biases would add 0.017% to Llama 3.1 8B. PaLM dropped them because it "found this to result in increased training stability for large models."
- **"Modern LLMs use a new architecture."** The block is still normalize, attend, add, then normalize, MLP, add, the pre-norm layout GPT-2 used in 2019. Only the parts inside changed.

## Key takeaways

- A 2025 block has the same skeleton as GPT-2's. The parts changed: RMSNorm, a SwiGLU MLP, no biases, grouped-query attention and rotary positions.
- RMSNorm divides by the root mean square and skips the mean. It's slightly cheaper and works as well. QK-norm applies it to queries and keys to stop attention scores from running away.
- SwiGLU multiplies a gated projection by a second projection. It needs a third matrix, so the hidden width drops from 4 times the model width to about 2.7 times; Llama 3 then scaled it up to 3.5.
- MHA, MQA, GQA and MLA differ in how many keys and values each token stores. GQA cuts Llama 3.1 8B's cache by 4 times and its attention weights by 37.5%.
- Llama 3.1 8B is 32 blocks of 218 million parameters plus two embedding tables of 525 million: 8,030,261,248 in all, 81% of each block in the MLP.

## What's next

In Llama 3.1 8B, the MLP holds 81% of every block, and every token passes through all of it. [Part 11](/blog/mixture-of-experts/) covers mixture of experts, which replaces that one MLP with many smaller ones and a router that picks a few for each token, the design behind DeepSeek-V3's 671 billion parameters with only 37 billion used per token.

## Further reading

- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467): RMSNorm, its invariance properties and its measured speedups.
- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202): SwiGLU, GEGLU and ReGLU, with the parameter-matched comparison in Table 1. Four pages.
- [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2016)](https://arxiv.org/abs/1612.08083): where gated linear units come from.
- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971): section 2.2 lists pre-norm with RMSNorm, SwiGLU at 2/3 of 4d, and RoPE, with where each came from.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): Table 3 gives the 8B, 70B and 405B shapes.
- [Llama 3 reference model code (Meta)](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py): the whole block, RMSNorm and SwiGLU in about 300 lines, with `bias=False` throughout.
- [Qwen3 Technical Report (Qwen Team, 2025)](https://arxiv.org/abs/2505.09388): section 2 describes the architecture, QK-norm and the removal of QKV biases.
- [PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311): section 2 explains SwiGLU, parallel layers, multi-query attention and no biases in a few bullet points.
- [Scaling Vision Transformers to 22 Billion Parameters (Dehghani et al., 2023)](https://arxiv.org/abs/2302.05442): the runaway attention scores that QK-norm fixes.
- [OLMo 2 Furious (OLMo Team, 2024)](https://arxiv.org/abs/2501.00656): section 3.3 tests norm placement and QK-norm for training stability.
- [Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019)](https://arxiv.org/abs/1911.02150), [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) and [DeepSeek-V2 (DeepSeek-AI, 2024)](https://arxiv.org/abs/2405.04434): multi-query, grouped-query and multi-head latent attention.
- [Mistral 7B (Jiang et al., 2023)](https://arxiv.org/abs/2310.06825) and [Gemma 3 Technical Report (Gemma Team, 2025)](https://arxiv.org/abs/2503.19786): sliding window attention and the 5-to-1 local-global pattern.
- [GPT-J (Wang & Komatsuzaki, 2021)](https://github.com/kingoflolz/mesh-transformer-jax): the open model PaLM credits for the parallel attention and MLP block.
- [nanoGPT model.py (Andrej Karpathy)](https://github.com/karpathy/nanoGPT/blob/master/model.py): GPT-2's block, to read side by side with Llama's.
