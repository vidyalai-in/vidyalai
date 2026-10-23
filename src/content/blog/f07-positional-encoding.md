---
title: "Positional Encoding: How Transformers Know Word Order"
description: "Attention can't see word order on its own. How sinusoids, learned tables, RoPE and ALiBi add it, and how models stretch their context past training length."
slug: "positional-encoding"
category: "llm-fundamentals"
tags: ["transformers", "attention", "long-context", "positional-encoding"]
series: "llm-fundamentals"
seriesOrder: 7
pubDate: 2026-10-23
---

## TL;DR

Attention compares every token with every other token, but nothing in that comparison says which token came first. Shuffle the words and each one gets exactly the same output, so position has to be added from outside. The original Transformer added a fixed pattern of sine waves to each token's vector, and GPT-2 added a learned vector for each of its 1,024 slots. Most models today use RoPE, which rotates queries and keys by an angle that grows with position, so attention scores depend on how far apart two tokens are. By the end you'll have checked that with a 2-D example by hand, and seen how models trained on 8,000 tokens get stretched to 128,000.

## Why it matters

"Dog bites man" and "man bites dog" use the same three words. To the attention mechanism from [Part 5](/blog/attention-explained/), with nothing added, they're the same input: each word gets the same weights and the same output, just listed in a different order.

Position also decides how much text a model can read. In 2023, Meta researchers took the original LLaMA 7B model, trained on 2,048-token sequences, and gave it 4,096 tokens. Its perplexity, a measure of how surprised a model is by the real next token (lower is better), went from 7.20 to over 1,000. It had never seen positions past 2,048. A small change to how positions are fed in, plus at most 1,000 training steps, got the same model working at 32,768 tokens.

Both problems come down to how a model encodes position, and that choice also sets its context window: the longest input it can handle.

## The intuition

Imagine writing a speech on index cards and dropping the stack. Every card is still there, but the order is gone. Attention sees text the same way: a pile of cards, all visible at once, with no page numbers.

You could write a page number on every card: that's absolute position. You could write "two cards after the one about the budget": that's relative position. Or you could turn each card by an angle that depends on its page number, so when you hold two cards side by side, the difference in their angles tells you how far apart they are. That last one is RoPE.

The analogy breaks in two places. The model can't read a page number as a number: position arrives as more numbers mixed into the vector that carries the word's meaning, and the model has to learn what they mean. And a model can only reliably use page numbers it saw in training, which is why "page 4,097" can break a model trained on 4,096.

## Step 1: attention can't see order

Take the four-token example from Part 5, "The river bank flooded", with the same 3-number token vectors and the same query, key and value matrices. Without the causal mask, the output for "bank" was `[0.944, 0.028]`: almost pure water signal, pulled from "river" and "flooded".

Now shuffle the sentence into nonsense, "flooded The bank river", and run the same calculation:

| Token | Output in "The river bank flooded" | Output in "flooded The bank river" |
|---|---|---|
| The | [0.427, 0.286] | [0.427, 0.286] |
| river | [0.804, 0.098] | [0.804, 0.098] |
| bank | [0.944, 0.028] | [0.944, 0.028] |
| flooded | [0.754, 0.123] | [0.754, 0.123] |

Every token gets the same output. "bank" still gives 47.2% of its attention to "river" and 47.2% to "flooded"; those tokens just sit in different rows, so the model can't tell the sentence from word salad.

The formula, `softmax(Q Kᵀ / √d_k) V`, explains why. A token's query, key and value come only from its own vector, and its score with another token depends only on the two vectors. Softmax and the weighted sum don't care what order the terms are added in, so no step ever asks where a token sits. This is called permutation equivariance: shuffle the inputs and the outputs come out shuffled the same way. [Shaw et al. (2018)](https://arxiv.org/abs/1803.02155) put it plainly: without position information, the Transformer is "entirely invariant to sequence ordering."

The rest of the block doesn't help. The MLP (multi-layer perceptron, the feed-forward network in each block from [Part 6](/blog/transformer-block/)) processes each token on its own. There's one partial exception, the causal mask, which comes back in Step 4.

## Step 2: stamping each position with sine waves or a learned vector

The 2017 paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al.) built a position vector the same size as the token vectors and added the two together at the bottom of the model: "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence."

### The sinusoidal formula

The position vector is built from sine and cosine waves:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Here `pos` is the token's position (0, 1, 2, ...), `d` is the vector size (512 in the paper), and `i` numbers the pairs of dimensions from 0 to `d/2 - 1`. Each pair holds a sine and a cosine of the same wave. Pair 0 advances 1 radian per token and repeats every 2π ≈ 6.28 tokens. Each later pair is slower by the same factor, so "the wavelengths form a geometric progression from 2π to 10000 · 2π." The slowest of the 256 pairs repeats every 60,611 tokens, just under 10000 · 2π ≈ 62,832.

Sine values for a few positions in four of the 512 dimensions:

| Position | Dim 0 (repeats every 6.3) | Dim 128 (every 63) | Dim 256 (every 628) | Dim 510 (every 60,611) |
|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.841 | 0.100 | 0.010 | 0.000 |
| 2 | 0.909 | 0.199 | 0.020 | 0.000 |
| 10 | -0.544 | 0.841 | 0.100 | 0.001 |
| 100 | -0.506 | -0.544 | 0.841 | 0.010 |
| 1,000 | 0.827 | -0.506 | -0.544 | 0.103 |

Think of a clock with 256 hands spinning at different speeds: fast hands tell neighbors apart, slow hands tell position 100 from 1,000. Notice the diagonal: dim 128 at position 10 reads 0.841, the same as dim 0 at position 1, because its wave is exactly 10 times slower.

### Why sine waves

The authors "hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE(pos+k) can be represented as a linear function of PE(pos)."

You can check this on one pair. A sine and cosine pair is a point on a circle, and moving `k` positions turns that point by `k` times the pair's speed. For pair 0 at position 3, the point is `(sin 3, cos 3) = (0.141, -0.990)`. Rotating it by 2 radians gives `(-0.959, 0.284)`, which is exactly `(sin 5, cos 5)`. A rotation is a matrix multiplication, so "two tokens later" is the same fixed matrix wherever you start.

It follows that the dot product of two position vectors depends only on the gap. With `d = 512`, `PE(10) · PE(13)`, `PE(50) · PE(53)` and `PE(500) · PE(503)` all come to 211.749, and a gap of 10 gives 173.79 wherever it falls. RoPE builds its whole design on this idea.

They also hoped sine waves "may allow the model to extrapolate to sequence lengths longer than the ones encountered during training."

### Learned position embeddings

The simpler option is a second lookup table, like the token embedding table from [Part 4](/blog/embeddings-and-logits/), with one learned row per position. The 2017 paper tried it and "found that the two versions produced nearly identical results": its Table 3 shows the same perplexity, 4.92, for both, and a translation score (BLEU) of 25.7 for learned positions against 25.8 for sine waves.

GPT-2 went the learned route. Its [configuration](https://huggingface.co/openai-community/gpt2/blob/main/config.json) sets `n_positions` to 1,024, and the [model code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) adds row `pos` of that table to each token's embedding. The table holds 1,024 × 768 = 786,432 numbers, about 0.6% of the smallest GPT-2. [BERT](https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json) has 512 learned positions.

A learned table sets a hard limit. GPT-2 has no row 1,025, so it can't take a 1,025th token. As the ALiBi paper puts it, the learned approach "does not have a way to encode positions greater than L; it therefore has no ability to extrapolate." Sine waves have no missing row, but the extrapolation hope didn't hold up: in tests covered in Step 4, a sinusoidal model trained on 512 tokens kept improving for only about 20 extra tokens, and by 1,012 tokens its perplexity had more than doubled.

## Step 3: rotating queries and keys with RoPE

Absolute positions leave the model to work out distances for itself, though in language "two words back" usually matters more than "word number 4,117". Shaw and colleagues put relative position straight into attention: each layer learns a vector for each distance between two tokens (clipped at 16 in their base model) and uses it when computing scores. On English-to-German and English-to-French translation, that beat absolute positions by 1.3 and 0.3 BLEU.

Rotary position embedding, or RoPE, from [Su et al. (2021)](https://arxiv.org/abs/2104.09864), gets relative positions with no extra learned vectors. It "encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation." Rotate each token's query and key by an angle set by its position, and their dot product depends only on the distance.

### A 2-D example by hand

Take a query `q = [1, 2]` and key `k = [3, 1]`. Unrotated, their dot product is `1 × 3 + 2 × 1 = 5`. Pick a rotation speed of `θ = 0.5` radians per position (an illustrative value).

Put the query at position 5, so it turns by `5 × 0.5 = 2.5` radians. Rotating `[x, y]` by angle `a` gives `[x cos a - y sin a, x sin a + y cos a]`. With `cos 2.5 = -0.801` and `sin 2.5 = 0.598`, the query becomes `[-1.998, -1.004]`. Put the key at position 3, a turn of 1.5 radians, and it becomes `[-0.785, 3.063]`. Their dot product is -1.506.

Now move both tokens down the text, keeping the gap at 2:

| Query position | Key position | Gap | Rotated query | Rotated key | Score |
|---|---|---|---|---|---|
| 5 | 3 | 2 | [-1.998, -1.004] | [-0.785, 3.063] | -1.506 |
| 12 | 10 | 2 | [1.519, 1.641] | [1.810, -2.593] | -1.506 |
| 102 | 100 | 2 | [-0.598, 2.155] | [3.157, 0.178] | -1.506 |
| 1,002 | 1,000 | 2 | [1.909, -1.164] | [-2.184, -2.287] | -1.506 |
| 5 | 5 | 0 | [-1.998, -1.004] | [-3.002, 0.994] | 5.000 |
| 6 | 3 | 3 | [-1.272, -1.839] | [-0.785, 3.063] | -4.634 |

The rotated vectors all differ, yet the score depends only on the gap. Turning both vectors by the same extra amount leaves the angle between them unchanged, so only the difference, `(m - n) × θ`, survives. The paper states the goal as a score that depends on the two tokens' content and "their relative position m − n".

One pair has a weakness: it repeats. At 0.5 radians per position it comes full circle every 12.6 tokens, and a gap of 47 scores 4.681, close to a gap of 0. Real models fix this the way the sine waves did, with many pairs at many speeds.

### The general form

A query or key with `d` numbers is split into `d/2` pairs, and pair `i` turns at its own speed:

```text
θ_i = base^(-2i/d)      for i = 0, 1, ..., d/2 - 1
angle for pair i at position m = m × θ_i
```

With the original base of 10,000, this is the sinusoidal schedule from Step 2, used to turn vectors instead of being added to them. Three practical details:

- RoPE runs inside every attention layer, on queries and keys only. Values are never rotated, so position changes the attention weights and leaves the mixed-in content alone.
- `d` is the size of one attention head: 128 in Llama 3 8B, which splits 4,096 numbers across 32 heads.
- Keys are rotated before they're stored, so the [KV cache](/blog/kv-cache/) holds keys that already carry their position. Meta's [Llama 3 reference code](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py) rotates, then writes to the cache.

### Why Llama 3 raised the base to 500,000

The base sets how slowly the slowest pairs turn. [Llama 3](https://arxiv.org/abs/2407.21783) raised it from 10,000 to 500,000: "This enables us to better support longer contexts." For a head of 128 numbers:

| RoPE base | Fastest pair repeats every | Slowest pair repeats every | Pairs (of 64) slower than 32,768 tokens |
|---|---|---|---|
| 10,000 (original RoPE) | 6.3 tokens | 54,410 tokens | 4 |
| 500,000 (Llama 3) | 6.3 tokens | 2.56 million tokens | 22 |
| 1,000,000 (Qwen3) | 6.3 tokens | 5.06 million tokens | 24 |

With base 10,000, every pair has come full circle by about 54,000 tokens, so in a 128K-token window even the slowest pairs repeat their angles. A larger base keeps more pairs on their first lap, which gives distant positions distinct angles.

## Step 4: penalizing distance with ALiBi, or adding nothing

[Press, Smith and Lewis (2021)](https://arxiv.org/abs/2108.12409) dropped position vectors entirely. ALiBi, short for Attention with Linear Biases, "biases query-key attention scores with a penalty that is proportional to their distance." Nothing is learned. Each head subtracts `slope × distance` from every score before softmax:

```text
scores for query i = q_i · K + m × [-(i-1), ..., -2, -1, 0]
```

The slope `m` is fixed per head. With 8 heads, the paper uses 1/2, 1/4 and so on down to 1/256, so some heads look locally and others far:

| Head slope | Penalty 1 token back | 10 back | 100 back | 1,000 back | Weight kept at 100 back |
|---|---|---|---|---|---|
| 1/2 | -0.5 | -5 | -50 | -500 | about 0 |
| 1/16 | -0.0625 | -0.625 | -6.25 | -62.5 | 0.2% |
| 1/256 | -0.004 | -0.039 | -0.39 | -3.9 | 68% |

The last column is `e^penalty`: if a far key scored the same as the current token, the share of weight it keeps. The steepest head sees only a few tokens; the gentlest still reads 1,000 tokens back at 2% strength.

In the paper's main result, a 1.3 billion parameter model trained on 1,024-token inputs and run on 2,048 matched a sinusoidal model trained on 2,048, "but training 11% faster and using 11% less memory." The paper's Table 2 tests models trained on 512 tokens of the WikiText-103 dataset on longer inputs (perplexity, lower is better):

| Input length at test | Sinusoidal | Rotary | T5 bias | ALiBi |
|---|---|---|---|---|
| 512 (training length) | 20.05 | 20.07 | 19.65 | 19.73 |
| 1,012 | 43.54 | 21.37 | 18.79 | 18.73 |
| 3,512 | 178.97 | 35.54 | 22.91 | 18.40 |
| 7,512 | 293.02 | 51.12 | 39.29 | 18.33 |

The sinusoidal model fell apart, RoPE degraded more slowly, and ALiBi was still improving at nearly 15 times its training length. ("T5 bias" is the learned distance bias from Google's T5 model.) [BLOOM](https://arxiv.org/abs/2211.05100), the 176-billion-parameter open multilingual model, used ALiBi. Remember the penalty table, though: ALiBi doesn't break on long inputs, but its steep heads give distant tokens almost no weight.

![A left-to-right flow in two groups. In the group labeled "Once, at the input", token embeddings get a position vector added, either sinusoidal or learned. That feeds the group labeled "Inside every attention layer": make queries, keys and values; rotate queries and keys by position (RoPE); score each query-key pair; subtract slope times distance from each score (ALiBi); then mask, softmax and mix the values.](/diagrams/f07-where-position-enters.svg "Three places position can enter a Transformer.")

Absolute methods touch the input once; RoPE and ALiBi act inside attention in every layer, which is how they express distance directly.

### No position encoding (NoPE)

The Step 1 shuffle had no causal mask. Decoder models, the kind that generate text, hide future tokens, and that alone leaks position. [Kazemnejad et al. (2023)](https://arxiv.org/abs/2305.19466) note that decoders with a causal mask "are not permutation invariant and can model sequences even without explicit position information."

The proof is short. Make every key identical, so each query spreads its attention evenly over the tokens it can see. Give the first token a value of 1 and every other token 0. The token at position `t` sees `t` tokens and gets `1/t`: 1.0 at position 1, 0.5 at position 2, 0.333 at position 3. Its position has appeared from nothing but the mask and softmax.

In their experiments, small models of about 107 million parameters with no position encoding (NoPE), trained from scratch on reasoning and arithmetic tasks, handled longer test inputs better than models with any of the encodings above. Those were small models on synthetic tasks; every large model in this article uses explicit positions.

## Step 5: stretching the context window

RoPE shares the weakness of every scheme above: a model trained up to 2,048 positions has never seen the angles its slow pairs reach at position 3,000. That's the jump to perplexity over 1,000 from the opening. There are two ways out: make long positions look like short ones, or train on longer sequences. Real models do both.

### Position interpolation

[Chen et al. (2023)](https://arxiv.org/abs/2306.15595) proposed position interpolation (PI): divide every position by the stretch factor. For a 4× stretch from 2,048 to 8,192, position 6,000 goes in as 1,500. Every angle now falls in the range the model trained on, and RoPE handles fractional positions fine. Going past the trained range, the paper warns, "may lead to catastrophically high attention scores that completely ruin the self-attention mechanism," and its error bound for interpolation is "at least ∼600× smaller" than for extrapolation.

PI took LLaMA models to 32,768 tokens with "minimal fine-tuning (within 1000 steps)". Results on the PG19 book dataset for the 7B model:

| LLaMA 7B variant | Perplexity at 2,048 tokens | At 4,096 | At 8,192 | At 32,768 |
|---|---|---|---|---|
| Original, 2,048-token window | 7.20 | over 1,000 | over 1,000 | over 1,000 |
| Fine-tuned to 8,192 without PI | 7.21 | 7.34 | 7.69 | not tested |
| PI to 8,192 | 7.13 | 6.96 | 6.95 | not tested |
| PI to 32,768 | 7.23 | 7.04 | 6.91 | 6.77 |

Plain fine-tuning adapted slowly: after more than 10,000 batches, the effective window had grown only "from 2048 to 2560." With PI, perplexity falls as the window grows, so the model is using the extra text.

### NTK-aware scaling and YaRN

PI squeezes every pair equally, including the fast pairs that tell neighbors apart. At a 4× stretch the fastest pair turns 0.25 radians per token instead of 1, so adjacent tokens look more alike. The [YaRN paper](https://arxiv.org/abs/2309.00071) (Peng et al., 2023) reports that PI fine-tunes reached only about 8× "before the LLM's outputs starts to degrade."

The next fix, "NTK-aware" scaling, was first posted online in 2023 by a user called bloc97 and is described in the YaRN paper: raise the RoPE base instead, which slows the slow pairs a lot and the fast pairs barely at all. (NTK stands for neural tangent kernel, the theory that motivated it.) Meta's [Code Llama](https://arxiv.org/abs/2308.12950) raised its base from 10,000 to 1,000,000 while fine-tuning on 16,384-token sequences, and handled inputs of up to 100,000 tokens.

YaRN ("Yet another RoPE extensioN method") makes the split explicit. Pairs that turn many times within the original window are left alone, pairs that never finish a turn are interpolated like PI, and pairs in between get a blend. It also sharpens the attention softmax slightly to offset scores spreading out over longer inputs. The paper claims "10x less tokens and 2.5x less training steps than previous methods," using less than about 0.1% of the original pretraining data. Stretching Llama 2 7B by 32× from 4,096 tokens took 600 fine-tuning steps and reached perplexity 2.37 at 131,072 tokens on Proof-pile documents (a different dataset from the PI table).

## In the real world

| Model | Position method | Native training window | Final window and how |
|---|---|---|---|
| GPT-2 | Learned, added to input | 1,024 | 1,024: hard limit |
| BERT base | Learned, added to input | 512 | 512: hard limit |
| BLOOM 176B | ALiBi | 2,048 | 2,048 |
| Llama 3 | RoPE, base 500,000 | 8,192 | Llama 3.1: 128K, long-context pretraining |
| Qwen3-8B | RoPE, base 1,000,000 | 32,768 | 131,072 with YaRN (factor 4) |
| DeepSeek-V3 | RoPE on a small slice of Q and K | 4,096 | 128K with YaRN (factor 40) |

Sources: the Hugging Face configurations, papers and model card linked in this article.

### Llama 3.1: from 8K to 128K

The [Llama 3 paper](https://arxiv.org/abs/2407.21783) trained at 8K first, because "the compute in self-attention layers grows quadratically in the sequence length." It then "increased context length gradually in six stages, starting from the original 8K context window and ending in the final 128K context window," using about 800 billion tokens. Each stage ran until short-context scores had "recovered completely" and the model "perfectly solves 'needle in a haystack' tasks", which hide one fact in a long document. Meta's [model list](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py) also turns on a RoPE scaling function for Llama 3.1 that Llama 3 didn't use. In the [code](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py), pairs that repeat in under 2,048 tokens are untouched, pairs slower than 8,192 tokens turn 8 times slower, and the rest are blended. With base 500,000 and 128-number heads, that's 29 pairs untouched, 29 slowed and 6 blended: the YaRN idea in a few lines.

### Qwen3: YaRN switched on by configuration

The [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) says "Qwen3 natively supports context lengths of up to 32,768 tokens" and was validated "up to 131,072 tokens using the YaRN method," switched on with a `rope_scaling` block set to `factor: 4.0` and `original_max_position_embeddings: 32768`. The card warns that open-source frameworks apply the same scaling to every input, "potentially impacting performance on shorter texts," and recommends enabling it only for long contexts.

### DeepSeek-V3: YaRN plus two short training phases

The [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) applies RoPE only to a small extra slice of each query and to one "decoupled" key shared by all heads, a consequence of the compressed attention described in [The KV Cache](/blog/kv-cache/). After pretraining at 4K, it applied YaRN with scale 40 and ran "two additional training phases, each comprising 1000 steps," from 4K to 32K, then to 128K.

### A long window isn't used evenly

[Liu et al. (2023)](https://arxiv.org/abs/2307.03172) found that performance "can degrade significantly when changing the position of relevant information," and is usually best when the needed fact sits near the start or end, a pattern called "lost in the middle". The memory and compute cost of serving 128K-token prompts is the subject of Inference Part 11.

## Common misconceptions

- **"Transformers read text left to right, so they know the order."** Attention sees every token at once; without position information, shuffling the input just shuffles the outputs. The causal mask leaks some order in decoders, but every large model in this article adds position explicitly.
- **"Position information costs a lot of parameters."** Sinusoidal encodings, RoPE and ALiBi have none. GPT-2's learned table is 786,432 numbers, about 0.6% of the smallest GPT-2.
- **"RoPE only knows relative positions."** The score depends only on the gap, but each key is rotated by its absolute position. The YaRN paper argues that slow pairs, which never finish a turn during training, let the model work out absolute position, which is part of why RoPE models break past their training length.
- **"A model's context window is fixed by its architecture."** For learned tables it is. For RoPE models it mostly depends on which angles the model saw in training, which is why Llama 3.1, Qwen3 and DeepSeek-V3 could stretch theirs with interpolation, extra training, or both.

## Key takeaways

- Attention on its own is order-blind: shuffle the tokens and each one gets the same output. Position has to be added.
- The original Transformer added fixed sine waves to each token's vector, from one turn every 6.3 tokens to one every 60,000 or so. Learned tables, as in GPT-2, worked as well but stop at a hard limit: 1,024 tokens for GPT-2.
- RoPE rotates queries and keys by an angle that grows with position, so the attention score depends only on the distance between two tokens. Llama, Qwen and DeepSeek all use it.
- ALiBi subtracts a distance penalty from every score instead. It held up best when inputs grew far past training length.
- Context windows are stretched by remapping positions: position interpolation squeezes all of them, while NTK-aware scaling and YaRN slow mainly the slow pairs. A short round of training on long text finishes the job.

## What's next

You've now seen every piece of the Transformer: tokens, embeddings, attention, the block and position. [Part 8](/blog/attention-is-all-you-need/) goes back to where it started, with a guided reading of "Attention Is All You Need": the original encoder-decoder design, the cross-attention that connects its two halves, how the authors trained it, and which of their choices survived into today's models.

## Further reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): section 3.5 defines sinusoidal positional encodings; Table 3 row (E) compares them with learned ones.
- [Self-Attention with Relative Position Representations (Shaw et al., 2018)](https://arxiv.org/abs/1803.02155): the first relative-position scheme for Transformers.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864): the RoPE paper, with the 2-D derivation in section 3.2.
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (Press et al., 2021)](https://arxiv.org/abs/2108.12409): ALiBi, and extrapolation tests for sinusoidal, rotary and T5 positions.
- [The Impact of Positional Encoding on Length Generalization in Transformers (Kazemnejad et al., 2023)](https://arxiv.org/abs/2305.19466): NoPE, and the proof that a causal mask leaks position.
- [Extending Context Window of Large Language Models via Positional Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595): position interpolation for RoPE models.
- [YaRN: Efficient Context Window Extension of Large Language Models (Peng et al., 2023)](https://arxiv.org/abs/2309.00071): NTK-aware scaling, NTK-by-parts and YaRN, explained through RoPE wavelengths.
- [Code Llama: Open Foundation Models for Code (Rozière et al., 2023)](https://arxiv.org/abs/2308.12950): raising the RoPE base to 1,000,000 for 100K-token inputs.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): RoPE base 500,000 and the six-stage climb from 8K to 128K tokens.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): section 4.3 on extending context with YaRN.
- [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model (BigScience, 2022)](https://arxiv.org/abs/2211.05100): a large model built on ALiBi.
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172): why a long window isn't used evenly.
- [Qwen3-8B model card (Qwen team)](https://huggingface.co/Qwen/Qwen3-8B): native 32,768-token context and the YaRN setting for 131,072.
- [Rotary Embeddings: A Relative Revolution (EleutherAI, 2021)](https://blog.eleuther.ai/rotary-embeddings/): an accessible walk through RoPE, with code.
- [Transformer Architecture: The Positional Encoding (Amirhossein Kazemnejad, 2019)](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/): a visual explanation of the sinusoidal scheme.
