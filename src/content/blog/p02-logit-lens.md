---
title: "The Residual Stream and the Logit Lens"
description: "Decode a model's half-finished predictions layer by layer with the logit lens, train a tuned lens on a laptop, and see where the simple version breaks."
slug: "logit-lens"
category: "interpretability"
tags: ["interpretability", "logit-lens", "tuned-lens", "residual-stream", "transformers"]
series: "llm-interpretability"
seriesOrder: 2
pubDate: 2026-10-29
---

## TL;DR

Every block of a Transformer adds its output to one running vector per token, the residual stream, so you can stop the model partway and ask what that vector would predict. The logit lens does this by pushing an intermediate vector through the model's own final layer norm and unembedding. On GPT-2 small it works, roughly: after block 6 of 12, its top guess matches the model's final answer on 15% of tokens, and after block 11 on 59%. A tuned lens adds one small trained translator per layer; ours, trained in 6 minutes on a laptop GPU, raised those numbers to 45% and 81%. On Pythia-160M and Qwen2.5-0.5B the logit lens barely works until the last few blocks, yet a tuned lens recovers Pythia's predictions from the middle layers. By the end you'll know what each lens measures, how to train one, and why a lens reading is evidence about the model's computation that still needs an intervention to confirm.

## Why it matters

Stop GPT-2 halfway through "Neil Armstrong was the first person to walk on the" and ask what it's thinking. Through block 8 of its 12 blocks, the logit lens says " same". After block 9 it says " planet". From block 10 on it says " moon", which is what the full model says too.

So the answer forms late, and in steps you can watch. Reading the residual stream directly lets you see a prediction take shape, where normally you'd only see the output. It's also the cheapest tool in interpretability: it needs no training and no labels, just one line of code.

The catch is that the early layers were never trained to be read this way. A tuned lens trained on 1 million tokens sees " moon" as the top guess four blocks earlier, after block 6. Which reading should you trust, and what does either one tell you about how the model works? [Fundamentals Part 16](/blog/llm-interpretability/) introduced the logit lens on a single prompt. This article measures both lenses on 155,040 predictions from held-out text, on three models, with confidence intervals.

## The intuition

Picture a report that moves down a line of twelve editors. Nobody rewrites it. Each editor reads the current draft and adds notes in the margin, and at the end a final editor reads everything and writes a one-word headline. That headline is the next token.

The logit lens is what you get if you grab the draft halfway down the line and hand it straight to the final editor. If the earlier editors write in the same shorthand the final editor reads, you learn what the headline would be so far. If they use their own shorthand, the final editor misreads it, and you learn about the misreading, not the draft. The tuned lens hires a translator for each position in the line, trained to rewrite that stage's shorthand into the final editor's.

The analogy breaks in two places. The notes aren't separate: every block adds a vector to the same vector, so they overlap. And the translator learns only from the final headline, so it can show you a prediction that is forming, but nothing the model computes and later throws away.

## Step 1: the residual stream is a running sum

[Fundamentals Part 6](/blog/transformer-block/) built the Transformer block: attention and an MLP (multi-layer perceptron), each reading a normalized copy of a token's vector and adding its result back. That running vector is the residual stream. Write `h_0` for the token's embedding plus its position embedding, and `h_ℓ` for the vector after block `ℓ`:

```text
h_ℓ = h_0 + (attn_1 + mlp_1) + (attn_2 + mlp_2) + ... + (attn_ℓ + mlp_ℓ)
```

Nothing is ever overwritten. GPT-2 small has 12 blocks, so its final vector `h_12` is the embedding plus 24 additions, and every one of them is a 768-number vector in the same space. [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html) call the stream a communication channel: layers talk to later layers by writing directions into it and reading directions out.

At the very end, the model turns `h_12` into a prediction in two steps. A last layer norm, `ln_final`, rescales the vector to mean 0 and spread 1 and applies learned gains, and the unembedding `W_U` (from [Fundamentals Part 4](/blog/embeddings-and-logits/)) multiplies it into one score per vocabulary token, 50,257 of them. Softmax turns the scores into probabilities.

### The stream grows as it goes

Because blocks only add, the vector gets longer. We measured the length (the norm) of the stream at every point on 304 held-out documents, averaging over 510 positions per document:

| Read after | Mean norm, all positions but the first | Norm at the first position |
|---|---|---|
| Embeddings | 4.6 | 10.3 |
| Block 1 | 51.0 | 165 |
| Block 3 | 61.7 | 2,580 |
| Block 6 | 79.0 | 3,029 |
| Block 9 | 120.3 | 3,144 |
| Block 11 | 219.0 | 3,159 |
| Block 12 (the output) | 448.0 | 418 |

From block 1 to block 11 the typical vector grows about 16% per block, and the last block doubles it. [Heimersheim and Turner (2023)](https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward) found the same exponential pattern in other models, about 4.5% per layer in the 48-layer GPT-2 XL. Size alone doesn't change what a lens reads, because `ln_final` divides by the vector's spread before decoding, so only the vector's direction reaches the readout.

The first position is a special case. Each of our 512-token chunks starts with GPT-2's `<|endoftext|>` token, and by block 3 its vector is 40 times longer than a typical token's. The first position is also where many heads park their attention when nothing else is relevant, the attention sink from [Fundamentals Part 5](/blog/attention-explained/) ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)); we didn't measure attention here. We leave the first position out of every lens measurement below.

## Step 2: the logit lens on GPT-2 small

The logit lens applies the final readout to any point in the stream:

```text
logit_lens(h_ℓ) = W_U · ln_final(h_ℓ)
```

For `ℓ = 12` it gives exactly the model's own output. For smaller `ℓ` it asks: if every block after `ℓ` added nothing, what would the model say? Here it is with [TransformerLens](https://transformerlensorg.github.io/TransformerLens/), a library that loads a model and records every intermediate activation in a cache:

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")  # fp32, weights as published
prompt = "Neil Armstrong was the first person to walk on the"
logits, cache = model.run_with_cache(prompt)

# The residual stream at 13 points: the embeddings, then after each of the 12 blocks
points = [cache[f"blocks.{l}.hook_resid_pre"] for l in range(12)]
points.append(cache["blocks.11.hook_resid_post"])

for l, h in enumerate(points):
    probs = model.unembed(model.ln_final(h[:, -1:]))[0, -1].softmax(-1)  # the logit lens
    top = probs.argmax()
    print(l, repr(model.to_string(top)), round(probs[top].item(), 3))
```

`hook_resid_pre` of block `l` is the stream going into that block, which is the output of the block before. We load with `from_pretrained_no_processing`, which keeps the weights exactly as published. TransformerLens's default loader folds the layer norms' gains into the next weights and centers every weight that writes to the stream, so each stream vector has mean zero. That changes the numbers inside the stream, though not this lens's output. Every measurement here uses unprocessed weights, so "the residual stream" means the same hidden state the Hugging Face model computes.

The run prints " destro" for the embeddings, " same" for blocks 1 to 8, " planet" for block 9 and " moon" from block 10 on. Next to it, the tuned lens we train in Step 3:

| Read after | Logit lens: top guess (probability) | Rank of " moon" | Tuned lens: top guess (probability) | Rank of " moon" |
|---|---|---|---|---|
| Embeddings | " destro" (0.88) | 40,394 | " same" (0.01) | 1,560 |
| Block 2 | " same" (0.87) | 775 | " first" (0.01) | 32 |
| Block 4 | " same" (0.92) | 419 | " road" (0.02) | 3 |
| Block 6 | " same" (0.88) | 112 | " moon" (0.15) | 1 |
| Block 8 | " same" (0.47) | 2 | " moon" (0.55) | 1 |
| Block 9 | " planet" (0.54) | 2 | " moon" (0.52) | 1 |
| Block 10 | " moon" (0.65) | 1 | " moon" (0.68) | 1 |
| Block 11 | " moon" (0.94) | 1 | " moon" (0.68) | 1 |
| Block 12 (output) | " moon" (0.70) | 1 | " moon" (0.70) | 1 |

The logit lens is confidently wrong for a long time: 88% on " same" after block 6, when the model will end up at 70% on " moon". And after block 11 it's more confident in " moon" than the model itself, so the last block lowers confidence, the same pattern Fundamentals Part 16 saw with " Paris". The tuned lens starts out unsure and becomes confident as the answer forms. The embeddings row is a curiosity: " destro" is a rare token fragment, and the logit lens returned it at the embeddings for four of the six prompts we tried, which makes it a quirk of the unembedding rather than a guess about this sentence.

### One prompt is an example, not a result

To measure the lens, we ran GPT-2 small on 304 documents from FineWeb-Edu (a filtered web-text corpus, ODC-By 1.0 license) that no lens was trained on. Each document gives 510 next-token predictions, 155,040 in all. At each point we compared the lens's distribution to the model's final one in two ways.

Top-1 agreement is the share of positions where the lens's top token is the final output's top token. KL divergence (Kullback-Leibler divergence, from [Math Part 6](/blog/cross-entropy-perplexity/)) measures how far one probability distribution is from another. In bits, it's the extra cost per token of encoding the final model's predictions with the lens's distribution instead. Zero means identical. Intervals are 95% bootstrap confidence intervals over documents.

| Read after | KL to final, logit lens (bits) | Top-1 agreement, logit lens |
|---|---|---|
| Embeddings | 72.3 [72.0, 72.7] | 0.01% |
| Block 1 | 10.7 [10.6, 10.8] | 7.4% [7.2, 7.6] |
| Block 3 | 8.37 [8.32, 8.43] | 8.3% [8.1, 8.5] |
| Block 6 | 6.81 [6.76, 6.85] | 15.0% [14.8, 15.3] |
| Block 9 | 4.67 [4.63, 4.71] | 34.1% [33.7, 34.6] |
| Block 10 | 3.69 [3.66, 3.73] | 44.9% [44.4, 45.4] |
| Block 11 | 1.95 [1.92, 1.97] | 59.2% [58.8, 59.7] |

The lens converges toward the output, as nostalgebraist reported for the largest GPT-2 in 2020: the decoded guesses "gradually converge to the final distribution over the layers of the network." But on ordinary text, most of the convergence is in the last four blocks. Through block 6, the logit lens matches the model's final top token on fewer than 1 in 6 positions.

Is that because the early stream holds no information about the next token yet, or because the logit lens reads it badly? Training a better reader answers the question.

## Step 3: training a tuned lens

[Belrose et al. (2023)](https://arxiv.org/abs/2303.08112) give two reasons the logit lens misreads early layers. First, it's biased: setting the later blocks' contributions to zero ignores the fact that they add a large average vector, so the lens leans toward some tokens at every layer. Second, representations drift: the directions that carry a given feature at block 3 aren't the directions `ln_final` and `W_U` expect at block 12.

Their fix is a translator per layer, an affine map (a matrix multiply plus a constant vector) applied before the usual readout:

```text
tuned_lens(h_ℓ) = W_U · ln_final(h_ℓ + A_ℓ h_ℓ + b_ℓ)
```

`A_ℓ` is a 768 by 768 matrix and `b_ℓ` a 768-number vector, 590,592 parameters per layer, all starting at zero, so training starts from the logit lens. The model itself stays frozen. The loss is the KL divergence from the model's final distribution to the lens's distribution, so the translator learns to predict what the model will say, not what the text says. The authors call it a distillation loss, and note that it keeps the lens from learning anything "over and above what the model has learned."

![A top-to-bottom flow. Tokens become h0, the embeddings. Blocks 1 to l add attention and MLP outputs, giving h-l, the stream after block l. From there the main path continues through later blocks, which keep adding, to h12, the final vector, then the final norm and unembedding, then next-token probabilities. Two dotted side paths leave h-l: the logit lens applies the final norm and unembedding to h-l directly, and the tuned lens first translates h-l with a trained A-l and b-l. Both side paths lead to an early guess at block l.](/diagrams/p02-lens-readout.svg "Both lenses reuse the model's own final readout; the tuned lens adds one trained translator per layer.")

The translators are the only new parameters: `ln_final`, `W_U` and all 12 blocks stay exactly as the model shipped.

### Our run

We trained the 12 translators (one per point from the embeddings to block 11; block 12 is the output itself) on 1,024,000 tokens of FineWeb-Edu: 250 steps of 8 chunks of 512 tokens, from 7,971 training documents with the 20 that also appeared in the held-out set removed. The matrices used the Muon optimizer, which orthogonalizes each update matrix so that no single direction dominates, and the biases used Adam. The learning rate (0.02 for Muon, 0.05 for the biases) came from a short sweep on 50 separate development documents, and decayed linearly to zero. Training took 352 seconds on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), at a peak of 7.1 GB of GPU memory. GPT-2 ran in fp32 with TF32 matrix multiplies; during training the lens's readout ran in bf16, and every evaluation ran in fp32 with TF32.

For comparison we trained two more lenses on the same data. One uses the paper's original recipe: SGD with Nesterov momentum, learning rate 1.0, gradient clipping at 1 and weight decay of 0.001. The other trains only the biases `b_ℓ`, which the paper calls the debiased logit lens, for 150 steps.

| Read after | Logit lens: KL (bits) / top-1 | Bias only | Tuned lens, SGD | Tuned lens, Muon |
|---|---|---|---|---|
| Embeddings | 72.3 / 0.01% | 10.4 / 0.9% | 5.30 / 15.4% | 4.69 / 23.0% |
| Block 3 | 8.37 / 8.3% | 5.45 / 18.1% | 3.12 / 30.9% | 2.73 / 34.7% |
| Block 6 | 6.81 / 15.0% | 4.18 / 27.7% | 2.18 / 41.5% | 1.93 / 45.2% |
| Block 9 | 4.67 / 34.1% | 2.84 / 46.1% | 0.97 / 60.6% | 0.88 / 63.1% |
| Block 11 | 1.95 / 59.2% | 1.36 / 68.5% | 0.25 / 79.2% | 0.22 / 80.6% |

Every 95% interval in this table is within ±0.5 percentage points and ±0.1 bits, except the logit lens at the embeddings (±0.4 bits). The full tuned lens beats the logit lens at every layer: 3.5 times lower KL after block 6, and three times the agreement. The bias alone closes about half of the KL gap at block 6 and 40% of the agreement gap, so the logit lens's bias is a real part of the problem, and the change of basis is the rest. Muon beat SGD at every point; the paired gap at block 6 is 3.7 percentage points [3.5, 3.9] of agreement.

![A line chart of KL divergence to the final output, in bits on a log scale from 0.1 to 100, against the block after which the stream is read, from the embeddings to block 11, for GPT-2 small. The logit lens starts at 72 bits at the embeddings, drops to about 10.7 after block 1, and falls slowly to 1.95 after block 11. The bias-only lens runs below it, from 10.4 to 1.36. The tuned lens is lowest everywhere, from 4.7 bits at the embeddings to 1.9 after block 6 and 0.22 after block 11. Confidence bands are too narrow to see.](/diagrams/p02-lens-kl.svg "GPT-2 small, 304 held-out documents: the tuned lens sits 3 to 15 times closer to the final output than the logit lens at every block.")

With 155,040 predictions the confidence bands are thinner than the lines, so what limits the tuned lens here is how long it was trained.

The training curve tells a similar story. Averaged over the 12 layers, the KL on 24 development documents fell from 4.05 bits after 41,000 training tokens to 2.49 bits after 410,000 and 2.16 bits after 1.02 million, and it was still falling. Belrose et al.'s original lenses used 250 steps of 262,144 tokens each, 65.5 million in all, and their 2025 revision says those SGD-trained lenses "were severely undertrained" next to Muon-trained ones. Ours is smaller still, so read our tuned-lens numbers as a floor on what a translator can recover.

### What the tuned lens is, and isn't

At the embeddings, the tuned lens agrees with the final output on 23% of tokens. There, the vector holds only the current token and its position, so the translator has learned a bigram model: a table of likely next tokens given the current one. Its cross-entropy on the real next token is 6.44 nats, a perplexity of about 625, against 3.17 nats (perplexity 24) for the full model.

The same caveat applies at every layer. A tuned lens is a probe: a trained model of its own, with 590,592 parameters per layer, that reads the stream. It shows that information about the final prediction is linearly readable at block 6. It doesn't show that block 7 uses it, or that the model computes its answer the way the lens's trajectory suggests. Belrose et al. tested this with a causal method they call causal basis extraction: they found the directions the tuned lens relies on most, erased them from the model's own stream, and found that the model's output changed disproportionately. That's evidence the lens reads features the model uses. Part 5 covers that kind of intervention in depth.

## Step 4: other models

GPT-2 small is the model the logit lens was discovered on. We ran the same evaluation, on the same 304 held-out documents, on two newer models:

| Model | Hugging Face ID, license | Blocks, width, vocabulary | Precision here |
|---|---|---|---|
| GPT-2 small (2019) | openai-community/gpt2, MIT | 12, 768, 50,257 | fp32 |
| Pythia-160M (2023) | EleutherAI/pythia-160m, Apache-2.0 | 12, 768, 50,304 | fp32 |
| Qwen2.5-0.5B (2024) | Qwen/Qwen2.5-0.5B, Apache-2.0 | 24, 896, 151,936 | bf16 (as published) |

Pythia comes from a suite of "16 LLMs all trained on public data seen in the exact same order" by [Biderman et al. (2023)](https://arxiv.org/abs/2304.01373). Its blocks run attention and the MLP in parallel, both reading the same normalized input, and its unembedding is a separate matrix from its input embedding. Qwen2.5-0.5B is the smallest of a family whose pretraining data the [Qwen2.5 report](https://arxiv.org/abs/2412.15115) says was scaled "to 18 trillion tokens", and it uses RMSNorm instead of layer norm. Both are still sums of block outputs, so both lenses apply unchanged.

| Model, read after | Logit lens: KL (bits) / top-1 | Tuned lens: KL (bits) / top-1 |
|---|---|---|
| GPT-2 small, block 6 of 12 | 6.81 / 15.0% | 1.93 / 45.2% |
| GPT-2 small, block 11 of 12 | 1.95 / 59.2% | 0.22 / 80.6% |
| Pythia-160M, block 6 of 12 | 18.6 / 6.0% | 2.33 / 50.2% |
| Pythia-160M, block 11 of 12 | 9.01 / 38.5% | 1.03 / 72.1% |
| Qwen2.5-0.5B, block 12 of 24 | 15.2 / 1.1% | not trained |
| Qwen2.5-0.5B, block 20 of 24 | 5.60 / 32.4% | not trained |
| Qwen2.5-0.5B, block 23 of 24 | 1.24 / 62.4% | not trained |

The 95% intervals are again within ±0.7 percentage points and ±0.7 bits. On Pythia-160M the logit lens barely works. After block 6 its cross-entropy on the real next token is still 16.1 nats, worse than the 10.8 nats you'd get by guessing uniformly over Pythia's 50,304 tokens. On Qwen2.5-0.5B it fails for longer: its top guess matches the final output on fewer than 2% of tokens through block 14 of 24, and it only passes 30% at block 20.

![A line chart of the share of positions where the lens's top guess matches the model's final output, from 0% to 100%, against depth as a fraction of the blocks run, for 304 held-out documents. The tuned lens on Pythia-160M is highest, rising from 25% at the embeddings to 50% at the halfway point and 72% at the second-to-last block. The logit lens on GPT-2 small rises from 7% after block 1 to 15% at the halfway point and 59% at the second-to-last block. The logit lens on Pythia-160M stays below 11% until two-thirds of the way and reaches 39%. The logit lens on Qwen2.5-0.5B stays below 2% until 58% depth, then climbs steeply to 62% after its second-to-last block. All lines reach 100% at the last block.](/diagrams/p02-model-compare.svg "The logit lens works worst on the two newer models; a tuned lens trained the same way recovers Pythia-160M's predictions from the middle layers.")

Is the information missing from Pythia's middle layers, or is the reader wrong? The tuned lens answers that. Trained with the same recipe as GPT-2's (1,024,000 tokens, 457 seconds, no separate learning-rate sweep), it matches the final top token on 50% of positions after block 6. That's slightly more than GPT-2 small's tuned lens at the same depth, and eight times Pythia's logit lens. The next-token prediction is about as readable halfway through Pythia as halfway through GPT-2. What differs is whether the final readout can decode it without a translator.

This fits Belrose et al.'s account, where the logit lens fails on some families because their intermediate representations drift away from the basis the unembedding expects. Our experiment doesn't isolate why Pythia and Qwen drift more than GPT-2. A tuned lens for Qwen would have cost roughly seven times GPT-2's training compute (twice the blocks, three times the vocabulary), more than our budget, so for Qwen we can only say the logit lens fails, not what a translator would recover.

![A line chart of the mean residual-stream norm on a log scale, from 0.3 to 1,000, against depth as a fraction of the blocks run, excluding the first position. GPT-2 small rises from 4.6 at the embeddings to 51 after block 1, then steadily to 219 after block 11 and 448 after its last block. Pythia-160M rises from 0.7 to 9 after block 1, to 31 after block 11 and 113 after its last block. Qwen2.5-0.5B rises from 0.4 to 5.5 after block 1 and grows steadily to 80 at the last block.](/diagrams/p02-norm-growth.svg "The residual stream grows with depth in all three models, by different amounts.")

All three streams grow with depth, Pythia's and Qwen's from much smaller embeddings. In each model the first position behaves differently again: in Qwen2.5-0.5B its norm sits between 1,396 and 1,421 from block 4 to block 21, while a typical position's grows from 11 to 57 over the same blocks.

## In the real world

- nostalgebraist introduced the logit lens in August 2020 on the largest GPT-2 (1.5 billion parameters, 48 layers), and noticed that "the inputs are immediately converted to a very different representation, which is smoothly refined into the final prediction." An edit to the post says the lens was later tried on models from 125M to 2.7B parameters, including GPT-Neo.
- Belrose et al. trained tuned lenses on models up to 20 billion parameters, across the Pythia, GPT-NeoX, GPT-2, GPT-Neo, OPT and BLOOM families. They report that for BLOOM and OPT 125M, the logit lens's top guess "is often the input token, rather than any plausible continuation token, in more than half the layers." They used the trajectory of tuned-lens predictions across layers to flag prompt injection attacks, "often with near-perfect accuracy", and found that lenses trained on a base model transfer to fine-tuned versions of it without retraining. Not reproduced here.
- Halawi, Denain and Steinhardt (2023) decoded intermediate layers while giving models few-shot examples with wrong labels. Early layers behaved the same with right and wrong demonstrations; after a "critical layer", accuracy under wrong demonstrations fell. They traced part of this "overthinking" to "false induction heads" that copy the wrong labels, "whose ablation reduces overthinking." Not reproduced here.
- Wendler et al. (2024) used the logit lens on Llama-2 while it translated between non-English languages. In the middle layers the right concept was already decodable, but in its English form, before the last layers switched to the target language. Their reading is that the model's concept space "lies closer to English than to other languages." Not reproduced here.

## Common misconceptions

- **"The logit lens shows what the model is thinking at each layer."** It shows what the model's final readout would make of an intermediate vector, a readout that was trained only on the last layer. On GPT-2 small it misses the final top token on 85% of positions after block 6, while a trained translator recovers three times as many.
- **"If a lens shows garbage at a layer, the information isn't there yet."** A bad reading can mean a bad reader. On Pythia-160M the logit lens matches the final top token on 6% of positions after block 6; a tuned lens reading the same vectors matches 50%.
- **"The tuned lens is ground truth because it's more accurate."** It's a trained probe with 590,592 parameters per layer. At the embeddings it acts as a learned bigram table, which says more about the lens than about the model. Agreement with the output shows readability, not use.
- **"The residual stream's growing norm means later layers matter more."** Every lens and every block reads the stream through a layer norm, which divides out the size. In GPT-2 small the first token's vector reaches a norm above 3,000 while carrying no information about the text, because it's the same `<|endoftext|>` token in every chunk.

## Key takeaways

- The residual stream is a running sum: the embedding plus every attention and MLP output so far, all in one space. That's why you can decode it at any layer.
- The logit lens needs no training. On GPT-2 small it matches the final top token on 15% of positions after block 6 of 12 and 59% after block 11, so most of the visible convergence happens late.
- A tuned lens adds one affine translator per layer, trained to match the model's own output. Ours, trained on 1 million tokens in 6 minutes, reached 45% after block 6 and 81% after block 11.
- The simple lens doesn't transfer. On Pythia-160M and Qwen2.5-0.5B the logit lens is close to useless until the last blocks, but a tuned lens shows Pythia's prediction is as readable at mid-depth as GPT-2's.
- Both lenses show what is readable from the stream, not what the model uses. Turning a lens reading into a claim about mechanism takes an intervention.

## What's next

The lens can only ask one question: which token comes next? [Part 3](/blog/linear-probes/) asks any question you like. A linear probe is a small classifier trained on a layer's activations to read out a property, such as a word's part of speech or whether a statement is true. We'll train probes on every layer of GPT-2 small, and test them against the control tasks that show when a probe has learned the task itself instead of reading the model.

## Further reading

- [interpreting GPT: the logit lens (nostalgebraist, 2020)](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens): the original post, on GPT-2 1.5B, with the observation that predictions converge smoothly across layers.
- [Eliciting Latent Predictions from Transformers with the Tuned Lens (Belrose et al., 2023)](https://arxiv.org/abs/2303.08112): the tuned lens, why the logit lens is biased and brittle, and (in the 2025 revision) the finding that SGD-trained lenses were undertrained and Muon trains them better.
- [A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)](https://transformer-circuits.pub/2021/framework/index.html): the residual stream as a shared communication channel that every layer reads from and writes to.
- [Residual stream norms grow exponentially over the forward pass (Heimersheim and Turner, 2023)](https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward): norm growth of about 4.5% per layer in GPT-2 XL, and where it comes from.
- [Overthinking the Truth: Understanding how Language Models Process False Demonstrations (Halawi et al., 2023)](https://arxiv.org/abs/2307.09476): decoding intermediate layers to find the critical layer where wrong demonstrations take over.
- [Do Llamas Work in English? On the Latent Language of Multilingual Transformers (Wendler et al., 2024)](https://arxiv.org/abs/2402.10588): the logit lens on Llama-2 finds English in the middle layers during translation.
- [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling (Biderman et al., 2023)](https://arxiv.org/abs/2304.01373): the model family behind our second model.
- [Qwen2.5 Technical Report (Qwen team, 2024)](https://arxiv.org/abs/2412.15115): our third model.
- [Muon: An optimizer for hidden layers in neural networks (Keller Jordan, 2024)](https://kellerjordan.github.io/posts/muon/): the optimizer that trained our tuned lens.
- [tuned-lens (AlignmentResearch)](https://github.com/AlignmentResearch/tuned-lens): the authors' MIT-licensed library, with pretrained lenses for GPT-2, Pythia and other models.
- [TransformerLens documentation](https://transformerlensorg.github.io/TransformerLens/): the hooks and activation cache used for every experiment here.
