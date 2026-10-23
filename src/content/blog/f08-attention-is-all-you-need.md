---
title: "Reading \"Attention Is All You Need\", the 2017 Transformer Paper"
description: "A guided reading of the 2017 Transformer paper: encoder, decoder and cross-attention, how it was trained, what it scored, and which choices survived."
slug: "attention-is-all-you-need"
category: "llm-fundamentals"
tags: ["transformers", "attention", "cross-attention", "architecture", "training"]
series: "llm-fundamentals"
seriesOrder: 8
pubDate: 2026-10-23
---

## TL;DR

"Attention Is All You Need", the 2017 paper that introduced the Transformer, runs 15 pages including its appendix of attention plots. It describes a translation model in two halves: an encoder that reads the whole source sentence at once, and a decoder that writes the translation one token at a time, joined by cross-attention, where the decoder's queries search the encoder's output. This article reads the paper section by section, works a cross-attention example by hand, computes its learning-rate schedule, and checks its results. It ends with a table of which 2017 choices are still in today's models.

## Why it matters

In 2016, Google's production translation system took "around 6 days to train a basic model using 96 NVIDIA K80 GPUs" on English-to-French. A year later, eight researchers working at Google trained a model on one machine with 8 GPUs for 3.5 days and beat every earlier single-model English-to-French result.

GPT, Llama, Claude and Gemini are all built on that model's design, yet the paper never mentions chatbots or language models trained on the web. It's a translation paper, and some of its parts, including the encoder and the cross-attention layers, are missing from most of the models it inspired.

[Parts 4 to 7](/blog/embeddings-and-logits/) took the pieces apart one at a time. This part reads the paper as written, so you can see the whole machine in its original form and spot what changed.

## The intuition

Picture a translator working on a German sentence. She reads the whole sentence and writes a note next to each word, recording what it means in context. Then she writes the English one word at a time. Before each word, she rereads what she has written so far, then looks back at her notes on the German to find what to say next.

Reading and annotating is the encoder. Rereading her own draft is the decoder's masked self-attention: she can only see words she has already written. Looking back at the notes is cross-attention. The notes don't change while she writes, and she can consult any of them at any point.

The analogy breaks in two places. During training, the model doesn't write one word at a time: it's shown the correct translation and predicts every next word at once, with a mask hiding the future. And the "notes" are lists of 512 numbers per token, not words. Nobody tells the model what to write in them; training sets them.

## Step 1: the problem the paper set out to solve

In 2017, the best translation systems were sequence-to-sequence (seq2seq) models built from recurrent neural networks (RNNs). [Sutskever, Vinyals and Le (2014)](https://arxiv.org/abs/1409.3215) used one LSTM (long short-term memory, an RNN design that is better at holding information over many steps) "to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector." Their model had 4 layers of 1,000 cells and took about ten days to train on an 8-GPU machine.

[Bahdanau, Cho and Bengio (2014)](https://arxiv.org/abs/1409.0473) let the decoder look back at every source word instead of one summary vector, which is where attention came from ([Part 5](/blog/attention-explained/) tells that story). Google's 2016 production system, [GNMT](https://arxiv.org/abs/1609.08144), stacked 8 LSTM layers in the encoder and 8 in the decoder with attention between them.

The trouble was speed. An RNN builds its state for word `t` from its state for word `t - 1`. The paper's introduction says so directly: "This inherently sequential nature precludes parallelization within training examples."

Two convolutional models tried to fix that. [ByteNet](https://arxiv.org/abs/1610.10099) from DeepMind and [ConvS2S](https://arxiv.org/abs/1705.03122) from Facebook AI Research used convolutions, which slide a small window across the sentence and can process every position at the same time. ConvS2S beat GNMT's accuracy "at an order of magnitude faster speed." But a window only sees nearby words, so linking two distant words takes a stack of layers, a cost the Transformer paper says grows "linearly for ConvS2S and logarithmically for ByteNet."

### Section 4: "Why Self-Attention", in numbers

Section 4 compares the three layer types on three measures, summarized in Table 1. Here `n` is the number of tokens, `d` the size of each token's vector, and `k` the convolution's window width.

| Layer type | Work per layer | Steps that must run in order | Longest path between two tokens |
|---|---|---|---|
| Self-attention | `O(n² · d)` | `O(1)` | `O(1)` |
| Recurrent | `O(n · d²)` | `O(n)` | `O(n)` |
| Convolutional | `O(k · n · d²)` | `O(1)` | `O(log_k(n))` |

Plug in a 50-token sentence and the paper's `d = 512`:

- **Work per layer.** Self-attention compares every pair of tokens: `50² × 512 = 1,280,000` multiply-adds. A recurrent layer multiplies a 512 by 512 matrix at each of 50 steps: `50 × 512² = 13,107,200`, about 10 times more. A convolution with a 3-token window does 3 times the recurrent work, 39,321,600.
- **Steps in order.** Self-attention computes all 50 positions at once: 1 step. The RNN needs 50 steps, one per token.
- **Path length.** For token 1 to influence token 50, self-attention needs 1 hop. The RNN passes the signal through 49 updates. Convolutions with a 3-token window need about 17 stacked layers, or 4 if the windows are spread out (dilated), as in ByteNet, since `log₃(50) ≈ 3.6`.

The paper argues that shorter paths make long-range relationships "easier to learn," since the training signal passes through fewer steps.

Look again at the first column, though. Self-attention is cheaper only "when the sequence length n is smaller than the representation dimensionality d," which was true for sentences in 2017. The two costs are equal at `n = 512`. At 4,000 tokens, self-attention does 7.8 times the work of a recurrent layer of the same width. The authors saw this coming and suggested restricting attention to a neighborhood "to improve computational performance for tasks involving very long sequences." Long prompts today run to hundreds of thousands of tokens, which is why so much engineering, from [FlashAttention](/blog/flashattention/) to sliding windows, targets that `n²`.

## Step 2: the architecture, section by section

### The abstract's claims

The abstract claims a model "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely," that is better, "more parallelizable" and faster to train. It reports 28.4 BLEU on English-to-German, over 2 points above earlier results "including ensembles," and 41.8 on English-to-French. BLEU (bilingual evaluation understudy) scores a translation from 0 to 100 by its word overlap with human translations.

The English-to-French number moved between versions. The first arXiv version, posted 12 June 2017, gave 41.0. The current version's abstract and Table 2 say 41.8, while the running text of section 6.1 still says 41.0.

### The encoder

Section 3.1: "The encoder is composed of a stack of N = 6 identical layers." Each layer has two sub-layers: multi-head self-attention, then a feed-forward network (FFN) that processes each token on its own. Every sub-layer is wrapped as `LayerNorm(x + Sublayer(x))`, the post-norm layout from [Part 6](/blog/transformer-block/), and everything is 512 numbers wide.

The encoder has no mask. Every German token can attend to every other, before and after it, because the whole source sentence is known up front. Its output is one 512-number vector per source token. Section 3.2.3 calls the keys and values made from it "memory keys and values", so this output is often called the memory.

### The decoder

The decoder is also 6 layers, but each has three sub-layers:

1. Masked self-attention over the English written so far. The mask from [Part 5](/blog/attention-explained/) stops position `i` from seeing later positions, and the target is "offset by one position," so the prediction for position `i` "can depend only on the known outputs at positions less than i."
2. Encoder-decoder attention, now usually called cross-attention, "which performs multi-head attention over the output of the encoder stack."
3. The same kind of FFN as the encoder.

### The three uses of attention

Section 3.2.3 lists them in one short paragraph:

| Where | Queries come from | Keys and values come from | Mask |
|---|---|---|---|
| Encoder self-attention | the previous encoder layer | the previous encoder layer | none |
| Decoder self-attention | the previous decoder layer | the previous decoder layer | hides later positions |
| Encoder-decoder (cross) attention | the previous decoder layer | the encoder's final output | none |

Cross-attention is the only place where the two halves meet. In the paper's words, it "allows every position in the decoder to attend over all positions in the input sequence," mimicking the attention in earlier seq2seq models.

![Two groups side by side. The encoder side: German tokens plus positions feed self-attention, which sees the whole source, then a feed-forward network; the pair repeats 6 times. The decoder side: the English written so far, shifted right, plus positions, feeds masked self-attention, then cross-attention with queries from the decoder, then a feed-forward network; the three repeat 6 times, and a linear layer plus softmax gives the next English token. An arrow labeled "keys and values, to every decoder layer" runs from the encoder's output into cross-attention.](/diagrams/f08-encoder-decoder.svg "The original Transformer: the encoder's output becomes the keys and values for cross-attention in every decoder layer.")

The encoder's final output is reused by all 6 decoder layers; residual connections and layer norms, which wrap every sub-layer, are left out of the picture.

### Cross-attention by hand

Translate "Die Katze schläft" ("The cat sleeps"), with made-up numbers small enough to follow. The formula is the scaled dot-product attention from Part 5, `softmax(Q Kᵀ / √d_k) V`, with `d_k = 2`. The only change is where `Q`, `K` and `V` come from.

The encoder has read the German, and each source token's output has been turned into a key and a value. Pretend the key's two numbers mean "is a noun" and "is a verb", and the value's three numbers mean "cat", "sleep" and "article":

| Source token | Key (noun, verb) | Value (cat, sleep, article) |
|---|---|---|
| Die | [0.5, 0] | [0, 0, 1] |
| Katze | [2, 0] | [1, 0, 0] |
| schläft | [0, 2] | [0, 1, 0] |

The decoder has written "The" and is about to predict the next word. Its vector at "The" produces the query [2, 0]: roughly, "find me the noun". Dot it with each key: 1, 4 and 0. Divide by `√2`: 0.707, 2.828, 0. Softmax turns those into weights of 0.102, 0.848 and 0.050, so the output is mostly Katze's value: [0.848, 0.050, 0.102]. That's a strong "cat" signal, added to the decoder's stream right before it predicts "cat".

One step later, at "cat", the query is [0, 2], "find me the verb". The scores are 0, 0 and 4, the weights 0.053, 0.053 and 0.894, and the output [0.053, 0.894, 0.053] points at "sleep".

| Decoder position | Weight on Die | Katze | schläft |
|---|---|---|---|
| The (predicting "cat") | 0.102 | 0.848 | 0.050 |
| cat (predicting "sleeps") | 0.053 | 0.053 | 0.894 |

Three things differ from self-attention. The grid isn't square: it has one row per English token and one column per German token. There's no mask, because the whole German sentence already exists. And the keys and values come from the encoder's output, which doesn't change while the decoder writes, so they can be computed once per sentence and reused at every step.

### Embeddings and softmax

Section 3.4 uses one learned matrix three times: for the encoder's input embeddings, the decoder's input embeddings and "the pre-softmax linear transformation" that turns the decoder's final vectors into logits, the weight tying from [Part 4](/blog/embeddings-and-logits/). That only works because English and German share one vocabulary of about 37,000 tokens. The embeddings are also multiplied by `√d_model`, about 22.6, a detail the paper states without explaining.

### One sentence through the whole model

Here's "Die Katze schläft ." going in and "The cat sleeps ." coming out during training, pretending each word and the period is one token. The decoder gets the correct English shifted right by one, starting with a start token `<s>`, and must predict each next token.

| Step | What happens | Shape |
|---|---|---|
| Source tokens | Die, Katze, schläft, . | 4 |
| Source embeddings | rows of the shared 37,000 × 512 table, times 22.6, plus sine-wave positions ([Part 7](/blog/positional-encoding/)) | 4 × 512 |
| Encoder, 6 layers | self-attention (8 heads, each a 4 × 4 grid of scores), then FFN (512 → 2,048 → 512) | 4 × 512 |
| Memory | the encoder's output, sent to all 6 decoder layers | 4 × 512 |
| Target input | `<s>` The cat sleeps . | 5 |
| Target embeddings | same table, times 22.6, plus positions | 5 × 512 |
| Masked self-attention | 8 heads, each a 5 × 5 grid with the upper triangle masked | 5 × 512 |
| Cross-attention | queries from 5 decoder rows, keys and values from 4 memory rows: 8 heads of 5 × 4 grids | 5 × 512 |
| FFN | 512 → 2,048 → 512, each row alone | 5 × 512 |
| Decoder, 6 layers | the three sub-layers above, 6 times | 5 × 512 |
| Logits | multiply by the shared table, transposed | 5 × 37,000 |
| Softmax and loss | compare with "The cat sleeps . `</s>`" | 1 number |

All five predictions come out of one pass, which is why training parallelizes. When translating new text, the decoder instead runs once per output token, feeding each choice back in.

Counting parameters from these sizes, with a 37,000-token vocabulary, gives 18.9 million in the shared table, 18.9 million in the encoder and 25.2 million in the decoder: 63.1 million, against the 65 million in the paper's Table 3. The same count for the big model gives 214 million against the paper's 213 million.

## Step 3: how they trained it

### Data and batches

Section 5.1: English-German used WMT 2014 (the 2014 Workshop on Statistical Machine Translation dataset), "about 4.5 million sentence pairs," split with byte-pair encoding ([Part 2](/blog/tokenization/)) into "a shared source-target vocabulary of about 37000 tokens." English-French used 36 million sentences and a 32,000-token word-piece vocabulary, Google's own subword method from GNMT. Sentences of similar length were batched together, with "approximately 25000 source tokens and 25000 target tokens" per batch.

### Hardware and schedule

Section 5.2: one machine with 8 NVIDIA P100 GPUs. The base model took about 0.4 seconds per step for 100,000 steps, 12 hours in total. The big model took 1.0 second per step for 300,000 steps, 3.5 days. At 25,000 target tokens a step, that's 2.5 billion target tokens for the base model and 7.5 billion for the big one.

### The optimizer and the warm-up schedule

Section 5.3 uses Adam (the optimizer from [Part 3](/blog/neural-networks-basics/)) with `β1 = 0.9`, `β2 = 0.98` and `ε = 10⁻⁹`, and a learning rate that changes every step:

```text
lrate = d_model^-0.5 × min(step^-0.5, step × warmup_steps^-1.5)
```

For the first `warmup_steps` steps, the second term is smaller, so the rate climbs in a straight line. After that, the first term takes over and the rate falls with one over the square root of the step number. The two meet at `step = warmup_steps`, which the paper set to 4,000.

```python
def lrate(step, d_model=512, warmup=4000):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)

lrate(1000)    # 0.000175  (a quarter of the way up)
lrate(4000)    # 0.000699  (the peak)
lrate(16000)   # 0.000349  (4x the steps, half the rate)
lrate(100000)  # 0.000140  (end of base training)
```

The peak is about 7 × 10⁻⁴. The big model, with `d_model = 1,024`, peaks lower, at 4.9 × 10⁻⁴. The model needed that warm-up: [Part 6](/blog/transformer-block/) showed that the post-norm layout used here trains badly without one.

### Regularization: dropout and label smoothing

Section 5.4 announces "three types of regularization" and then names two. (Section 6.3 later mentions tuning "attention" dropout too.)

- **Residual dropout.** Dropout randomly zeroes a share of the numbers during training so the network can't lean on any single one. The paper applies it to each sub-layer's output and to the embedding-plus-position sums, at a rate of 0.1 for the base model. The big English-German model used 0.3.
- **Label smoothing.** The normal training target says the correct next token has probability 1 and every other token 0. Label smoothing, from [Szegedy et al. (2015)](https://arxiv.org/abs/1512.00567), mixes in a little of a uniform distribution: `q'(k) = (1 − ε) × [k is correct] + ε / K`. With `ε = 0.1` and K = 37,000 tokens, the target becomes about 0.900003 for the correct token and 0.0000027 for each of the others. A model that matches that target perfectly is 90% sure, never 100%.

The paper names the cost: "This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score." Table 3 shows exactly that. Without smoothing, perplexity improves from 4.92 to 4.67, and BLEU drops from 25.8 to 25.3.

### Checkpoint averaging and beam search

Section 6.1 adds two tricks for producing translations. The final base model is the average of the weights in its last 5 saved checkpoints, written 10 minutes apart; the big model averaged its last 20. Translations come from beam search with "a beam size of 4 and length penalty α = 0.6." Beam search keeps the 4 most probable partial translations at each step instead of only the single best one.

The length penalty comes from GNMT. A sentence's probability is a product of many numbers below 1, so longer translations always score lower, and plain beam search prefers short ones. GNMT divides each candidate's log-probability by `lp(Y) = ((5 + |Y|) / 6)^α`, where `|Y|` is its length in tokens. With `α = 0.6`, a 10-token candidate is divided by 1.73 and a 20-token one by 2.35. A 10-token translation with log-probability -6.0 scores -3.46; a 20-token one at -8.0 scores -3.40 and wins, though it's less probable in raw terms.

## Step 4: the results

### Table 2: better and cheaper

| Model | BLEU, EN-DE | BLEU, EN-FR | Training cost, EN-DE (FLOPs) | Training cost, EN-FR (FLOPs) |
|---|---|---|---|---|
| ByteNet | 23.75 | | | |
| GNMT + RL | 24.6 | 39.92 | 2.3 × 10¹⁹ | 1.4 × 10²⁰ |
| ConvS2S | 25.16 | 40.46 | 9.6 × 10¹⁸ | 1.5 × 10²⁰ |
| MoE | 26.03 | 40.56 | 2.0 × 10¹⁹ | 1.2 × 10²⁰ |
| GNMT + RL ensemble | 26.30 | 41.16 | 1.8 × 10²⁰ | 1.1 × 10²¹ |
| ConvS2S ensemble | 26.36 | 41.29 | 7.7 × 10¹⁹ | 1.2 × 10²¹ |
| Transformer (base) | 27.3 | 38.1 | 3.3 × 10¹⁸ | 3.3 × 10¹⁸ |
| Transformer (big) | 28.4 | 41.8 | 2.3 × 10¹⁹ | 2.3 × 10¹⁹ |

FLOPs (floating-point operations) measure total training arithmetic. The paper estimated them as training time × number of GPUs × an assumed sustained speed per GPU, 9.5 trillion operations per second for a P100. The arithmetic checks out: 12 hours × 8 GPUs × 9.5 × 10¹² gives 3.3 × 10¹⁸, and 3.5 days gives 2.3 × 10¹⁹.

The base model beat every earlier English-German result, ensembles included, for about a third of the compute of a single ConvS2S model. The big model's 41.8 on English-French topped the ConvS2S ensemble, which cost about 52 times more. "Ensemble" means several models voting together, a standard way to buy a little extra accuracy.

### Table 3: what each part is worth

Table 3 changes one thing at a time in the base model and scores it on a separate development set (newstest2013, without checkpoint averaging), so its BLEU numbers sit lower than Table 2's. The base model scores 25.8. [Part 6](/blog/transformer-block/) covered the head-count rows (A), where 1 head scored 24.9 and 8 heads 25.8, and [Part 7](/blog/positional-encoding/) covered row (E), where learned positions matched sine waves. Three other rows:

| Change to the base model | BLEU (dev) | Parameters |
|---|---|---|
| Base: 6 layers, `d_model` 512, `d_k` 64, dropout 0.1 | 25.8 | 65M |
| Key size `d_k` cut to 16 (B) | 25.1 | 58M |
| 2 layers instead of 6 (C) | 23.7 | 36M |
| 8 layers (C) | 25.5 | 80M |
| `d_model` 1,024, `d_k` 128 (C) | 26.0 | 168M |
| FFN width 4,096 (C) | 26.2 | 90M |
| No dropout (D) | 24.6 | 65M |
| Big model, 300K steps | 26.4 | 213M |

Shrinking the key size hurt, which the authors read as a sign "that determining compatibility is not easy." Bigger models did better, "as expected," and turning off dropout cost 1.2 BLEU, which shows how much this 65-million-parameter model was at risk of memorizing its training data.

### Section 6.3: parsing

To test whether the model could do more than translate, the authors tried English constituency parsing: turning a sentence into a tree of nested phrases (noun phrase, verb phrase and so on), written out as a string much longer than the input. They trained a 4-layer Transformer with `d_model = 1,024` on about 40,000 sentences from the Wall Street Journal section of the Penn Treebank, and on about 17 million sentences in a semi-supervised setting.

Scores are F1, a 0 to 100 measure of how many phrase brackets match the reference trees. Trained on the 40,000 sentences alone, the Transformer scored 91.3, behind only one model in that group (91.7) and ahead of the Berkeley Parser (90.4), with only a few settings tuned. With extra data it scored 92.7.

## In the real world

### What survived, and what changed

Scaled dot-product attention, multiple heads, residual connections, normalization and a feed-forward network between attention layers are in every model this series has covered. Most of what surrounds them has changed.

| 2017 choice | Typical choice today | Where in this series |
|---|---|---|
| Encoder plus decoder, joined by cross-attention | One decoder-only stack for chat LLMs; encoder-decoder survives in models like T5 | Part 9 |
| 6 + 6 layers, 65M or 213M parameters | 126 layers and 405B parameters in Llama 3.1 405B, about 6,200 times the base model | [Part 1](/blog/how-llms-work/) |
| Post-norm: `LayerNorm(x + Sublayer(x))` | Pre-norm, often with RMSNorm (Llama 2 and 3) | [Part 6](/blog/transformer-block/), Part 10 |
| Sine-wave positions added at the input | RoPE inside attention (Llama 3, base 500,000) | [Part 7](/blog/positional-encoding/) |
| ReLU in the FFN | GELU (GPT-1), then SwiGLU (Llama 2 and 3) | [Part 6](/blog/transformer-block/), Part 10 |
| 8 heads, each with its own keys and values | Grouped-query attention: Llama 3 shares 8 key-value heads among 32 to 128 query heads | [The KV cache](/blog/kv-cache/), Part 10 |
| Adam, `β2 = 0.98`, 4,000 warm-up steps, then 1/√step decay | Adam or AdamW, `β2 = 0.95`, warm-up, then cosine decay (GPT-3, Llama 2, Llama 3) | Part 12 |
| Dropout 0.1 | No dropout in pretraining (PaLM) | Part 12 |
| Beam search, 4 candidates | Sampling with temperature and top-p for chat | Part 9 |
| BPE, about 37,000 tokens shared by two languages | 128,000 tokens in Llama 3 | [Part 2](/blog/tokenization/) |
| 8 P100 GPUs for 3.5 days | Up to 16,000 H100 GPUs for Llama 3 405B | Part 12 |

Sources: [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-3](https://arxiv.org/abs/2005.14165) appendix B, [Llama 2](https://arxiv.org/abs/2307.09288) section 2.2, [Llama 3](https://arxiv.org/abs/2407.21783) Table 3 and sections 3.3 to 3.4, [PaLM](https://arxiv.org/abs/2204.02311) section 5, [T5](https://arxiv.org/abs/1910.10683) section 3.1.

The split into families happened fast. A year after the paper, OpenAI's GPT-1 used "a multi-layer Transformer decoder," a "12-layer decoder-only transformer with masked self-attention heads," with no encoder and no cross-attention. The same year, [BERT](https://arxiv.org/abs/1810.04805) kept only the other half: "a multi-layer bidirectional Transformer encoder based on the original implementation." Google's T5 (2019) kept both, describing its model as "a standard encoder-decoder Transformer as proposed by Vaswani et al. (2017)."

A few training choices were dropped as models grew. Google's 540-billion-parameter PaLM states: "The model was trained without dropout, although dropout of 0.1 is used for finetuning in most cases." The GPT-3 paper adds that its data were "sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting." A model that sees most of its text once has little chance to memorize it. The optimizer barely changed in kind: Llama 3 405B still uses Adam's descendant AdamW, a linear warm-up (8,000 steps), and a peak learning rate of 8 × 10⁻⁵, about a ninth of the 2017 base model's.

### The paper itself

The eight authors are Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin, from Google Brain and Google Research; Gomez was at the University of Toronto and did the work at Google Brain. A footnote marks them all as equal contributors: "Listing order is random." It also records who did what: Jakob "proposed replacing RNNs with self-attention," and Noam "proposed scaled dot-product attention, multi-head attention and the parameter-free position representation."

It was presented at NIPS 2017 (now NeurIPS) in Long Beach, California. The code was released in Google's tensor2tensor library, and the paper has seven versions on arXiv, the last from August 2023.

## Common misconceptions

- **"The Transformer was invented for chatbots."** It was a translation model, with an encoder for the source language and a decoder for the target. The decoder-only design behind chat models came a year later, with GPT-1.
- **"This paper invented attention."** Attention between a decoder and an encoder dates to Bahdanau et al. in 2014. The paper's contribution was dropping recurrence entirely and building the whole model from attention and small feed-forward networks.
- **"Today's LLMs are the 2017 Transformer, made bigger."** They keep its core block but drop the encoder and cross-attention, move the layer norm, swap sine-wave positions for RoPE, ReLU for SwiGLU, and full multi-head attention for grouped-query attention.
- **"Self-attention is always cheaper than recurrence."** Only when the sequence is shorter than the vector size. At 50 tokens and `d = 512` it's 10 times cheaper per layer; at 4,000 tokens it's 7.8 times more expensive. The paper's advantage was parallel steps and short paths, not raw arithmetic.
- **"Cross-attention is just self-attention."** The formula is the same, but the queries come from one sequence and the keys and values from another. The score grid is rectangular, target length by source length, and needs no causal mask.

## Key takeaways

- The 2017 Transformer was a translation model: a 6-layer encoder reads the whole source sentence, and a 6-layer decoder writes the translation one token at a time.
- Cross-attention joins the halves. The decoder's queries search the encoder's output, which supplies the keys and values, through a rectangular grid of weights with no mask.
- The training recipe was Adam with a 4,000-step warm-up peaking near 0.0007, dropout 0.1, label smoothing 0.1, checkpoint averaging, and beam search with 4 candidates.
- The big model hit 28.4 BLEU on English-German and 41.8 on English-French with 8 GPUs in 3.5 days, at a fraction of earlier models' compute.
- The core block survived. The encoder, post-norm, sine waves, ReLU and dropout mostly didn't, and the model grew from 65 million parameters to hundreds of billions.

## What's next

Within a year, the Transformer split in two: BERT kept the encoder to understand text, and GPT kept the decoder to write it. [Part 9](/blog/encoders-decoders-gpt/) compares encoder-only, decoder-only and encoder-decoder models, then covers how a decoder actually picks its next token, from greedy choice and beam search to temperature, top-k and top-p sampling.

## Further reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): the paper itself. Sections 3.1 and 3.2.3 describe the encoder, decoder and the three uses of attention; section 5 the training recipe; Tables 2 and 3 the results.
- [Sequence to Sequence Learning with Neural Networks (Sutskever, Vinyals & Le, 2014)](https://arxiv.org/abs/1409.3215): the LSTM encoder-decoder the Transformer replaced.
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau, Cho & Bengio, 2014)](https://arxiv.org/abs/1409.0473): the first encoder-decoder attention, the ancestor of cross-attention.
- [Google's Neural Machine Translation System (Wu et al., 2016)](https://arxiv.org/abs/1609.08144): GNMT, the 16-layer LSTM system in Table 2, and the source of the length penalty and word-piece vocabulary.
- [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](https://arxiv.org/abs/1705.03122) and [Neural Machine Translation in Linear Time (Kalchbrenner et al., 2016)](https://arxiv.org/abs/1610.10099): ConvS2S and ByteNet, the convolutional rivals.
- [Rethinking the Inception Architecture for Computer Vision (Szegedy et al., 2015)](https://arxiv.org/abs/1512.00567): section 7 defines label smoothing.
- [Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): GPT-1, a decoder-only Transformer pretrained as a language model.
- [BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) and [T5 (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683): the encoder-only and encoder-decoder branches.
- [PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311): section 5 notes that pretraining used no dropout.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): a 2024 recipe to set beside the 2017 one.
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/): the paper line by line, with working PyTorch code for every section.
- [The Illustrated Transformer (Jay Alammar, 2018)](https://jalammar.github.io/illustrated-transformer/): the encoder-decoder model drawn step by step.
