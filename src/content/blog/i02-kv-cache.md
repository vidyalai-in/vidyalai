---
title: "The KV Cache: What It Stores, What It Costs, How to Shrink It"
description: "What the KV cache holds, how big it gets on real models, and how MQA, GQA and DeepSeek's MLA shrink it so more users fit on one GPU."
slug: "kv-cache"
category: "inference-optimization"
tags: ["caching", "gpu", "serving", "latency"]
series: "llm-inference"
seriesOrder: 2
pubDate: 2026-02-20
---

## TL;DR

The KV cache keeps the attention keys and values of every earlier token, in every layer, so they are computed once instead of at every step. Without it, a 500-token answer from an 8 billion-parameter model would take about 1,250 times more arithmetic. The cost moves to memory instead: Llama 3.1 8B needs 128 KiB per token, so one user with a 128K-token context needs more memory than the model's weights. This article shows what gets stored, sizes the cache for real models, and explains how multi-query, grouped-query and multi-head latent attention shrink it by 4 to 57 times. By the end you'll be able to estimate how many users fit on a GPU from four numbers in a model's config file.

## Why it matters

Llama 3.1 8B fits in 16 GB. Give one user its full 128K-token context, and that user's KV cache needs another 17 GB, more than the model itself.

Now take DeepSeek-V3, a model with 671 billion parameters, about 84 times larger. Its cache for the same 128K tokens is about 9 GB, roughly half of Llama's. The difference comes from how each model's attention layer stores keys and values.

[Part 1](/blog/how-llm-inference-works/) showed that decode, the phase that writes the answer one token at a time, is limited by memory bandwidth. The KV cache is the part of memory that grows with every token and every user. It decides how many people one GPU can serve and how fast tokens stream once a conversation gets long.

## The intuition

Imagine reading a long mystery novel and keeping index cards. For each sentence you've read, you write two cards. One is a label card: a short note on what the sentence is about ("the butler's alibi"). The other is a content card: what you'd want to remember from it ("he was in the garden at 9").

When a new sentence arrives, you don't reread the book. You scan the label cards to see which earlier sentences matter now, then pull their content cards and blend them. Once a card is written, it never changes, because later pages can't change what earlier sentences said.

Keys are the label cards, values are the content cards, and the new token's question ("what's relevant to me?") is the query.

The analogy breaks in two places. First, the model doesn't keep one stack of cards: it keeps a separate stack for every layer and every key-value head, so Llama 3.1 8B writes 256 pairs of cards for each token. Second, the model never summarizes or throws cards away, so the stack only grows.

## Step 1: what exactly gets stored

Attention gets its own part in the Fundamentals series; here is the short version, building on [Fundamentals Part 1](/blog/how-llms-work/). Each token builds a query (Q, "what am I looking for?"), a key (K, "what do I contain?") and a value (V, "what do I pass on?"). A new token compares its query with the keys of every earlier token, turns the scores into weights, and takes a weighted mix of their values.

Two facts make caching possible:

- LLMs use **causal attention**: a token can only look backward. So when token 1,001 arrives, the keys and values of tokens 1 to 1,000 are exactly what they were a step ago.
- Only the newest token's query is ever needed, and its attention scores are cheap to compute fresh.

Here's the full list for one token:

| Thing | Kept in the cache? | Why |
|---|---|---|
| Keys, every layer, every key-value head | Yes | Every future token compares its query against them |
| Values, every layer, every key-value head | Yes | Every future token takes a weighted mix of them |
| Queries of past tokens | No | Only the newest token's query is used |
| Attention scores | No | Recomputed each step, one row for the new token |
| Hidden states (the token's vector between layers) | No | Later tokens only need its keys and values |

For Llama 3.1 8B, which has 32 layers and 8 key-value heads of 128 numbers each, that's:

```text
numbers per token = 32 layers × 8 heads × (128 for K + 128 for V) = 65,536
bytes per token   = 65,536 × 2 bytes (16-bit) = 131,072 bytes = 128 KiB
```

A token of English text takes a few bytes to store as text. Its cache entry takes 128 KiB.

## Step 2: decode with and without the cache

Start with a 4-token prompt and a 4-token answer, and count how many tokens go through the whole model at each step:

| Step | Produces | Tokens through the model, no cache | With cache |
|---|---|---|---|
| Prefill | Token 5 | 4 | 4 |
| Decode 1 | Token 6 | 5 | 1 |
| Decode 2 | Token 7 | 6 | 1 |
| Decode 3 | Token 8 | 7 | 1 |
| Total | | 22 | 7 |

Without a cache, each step reruns every earlier token through all the layers to rebuild keys and values it already had. With one, each step runs only the new token.

Now real numbers. Llama 3.1 8B, a 1,000-token prompt, a 500-token answer. Prefill produces the first output token, so decode runs 499 steps. A forward pass costs about 2 floating-point operations (FLOPs) per parameter per token, so:

```text
with cache:     2 × 8.03 billion × 1 token        ≈ 16 GFLOPs per step
                × 499 steps                        ≈ 8 TFLOPs in total
without cache:  last step reruns 1,499 tokens:
                2 × 8.03 billion × 1,499           ≈ 24 TFLOPs for that one step
                sum over all 499 steps             ≈ 10,000 TFLOPs
ratio           ≈ 1,250×
```

The last uncached step alone takes about 24 ms of pure math at the H100's 989 TFLOPs peak. In general:

```text
without cache: FLOPs ≈ 2N × G × (P + G/2)
with cache:    FLOPs ≈ 2N × G   (plus a small attention term)
```

Here N is the parameter count, P the prompt length and G the number of generated tokens. The uncached cost grows with the square of the length; the cached cost grows linearly.

Attention itself still costs something. Each step, the new token's query meets every cached key and value. For Llama 3.1 8B at 1,500 tokens of context, that's `4 × 32 query heads × 128 × 1,500 tokens × 32 layers ≈ 0.79 GFLOPs`: two matrix products (query times keys, then attention weights times values), each costing 2 FLOPs per multiply-add. That's about 5% of the 16 GFLOPs for the weights. The expensive part is that every step has to read the whole cache from memory.

## Step 3: how big the cache gets

[Part 1](/blog/how-llm-inference-works/) gave the formula:

```text
KV bytes per token = 2 (K and V) × layers × KV heads × head size × bytes per number
```

Four numbers from a model's config file are enough. Six real models, all in 16-bit:

| Model | Attention | Layers | KV heads × head size | Per token | 4,096-token chat | 128K tokens |
|---|---|---|---|---|---|---|
| GPT-3 175B (2020) | MHA | 96 | 96 × 128 | 4.5 MiB | 18 GiB* | 576 GiB* |
| Llama 2 7B (2023) | MHA | 32 | 32 × 128 | 512 KiB | 2 GiB | 64 GiB* |
| Falcon 7B (2023) | MQA | 32 | 1 × 64 | 8 KiB | 32 MiB* | 1 GiB* |
| Llama 3.1 8B (2024) | GQA | 32 | 8 × 128 | 128 KiB | 512 MiB | 16 GiB |
| Llama 3.1 70B (2024) | GQA | 80 | 8 × 128 | 320 KiB | 1.25 GiB | 40 GiB |
| DeepSeek-V3 (2024) | MLA | 61 | 512 latent + 64 RoPE key | 68.6 KiB | 274.5 MiB | 8.6 GiB |

\*GPT-3 and Falcon 7B were trained on 2,048-token sequences and Llama 2 7B on 4,096, so starred cells are past what those models can read; they show what the design would cost at that length.

Configurations come from the [GPT-3 paper](https://arxiv.org/abs/2005.14165) (Table 2.1), the [LLaMA paper](https://arxiv.org/abs/2302.13971) (whose 7B shape Llama 2 7B keeps), [Falcon-7B's config](https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json) (71 query heads of 64 numbers, `multi_query: true`), the [Llama 3 paper](https://arxiv.org/abs/2407.21783) (Table 3) and [DeepSeek-V3's config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json). For DeepSeek-V3 the per-token count is `(512 + 64) × 61 layers × 2 bytes = 70,272 bytes`.

Model size doesn't predict cache size. Llama 2 7B needs 4 times more per token than Llama 3.1 8B, and DeepSeek-V3 needs about half as much as an 8B model. At long contexts the cache outgrows everything else: GPT-3's design at 128K tokens would need 576 GiB for one user.

[Pope et al. (2022)](https://arxiv.org/abs/2211.05102), the Google paper that laid out the economics of serving large Transformers, gives an example at scale: for a 500B+ model with multi-head attention, at batch size 512 and context length 2,048, "the KV cache totals 3TB, which is 3 times the size of the model's parameters."

## Step 4: sharing keys and values: MHA, MQA, GQA and MLA

The Attention column explains most of the gaps. In the original Transformer design, called **multi-head attention (MHA)**, attention is split into several heads that each look for different patterns, and every head has its own keys and values. That's why GPT-3 stores 96 key-value pairs per layer. Later designs ask whether the heads need separate keys and values, or only separate queries.

![Four panels, each with 32 query heads. Multi-head attention: each head has its own key and value, 32 pairs or 8,192 numbers per token per layer. Grouped-query attention: groups of 4 query heads share a pair, 8 pairs or 2,048 numbers. Multi-query attention: all 32 heads share one pair, 256 numbers. Multi-head latent attention: all heads rebuild keys and values from one 512-number latent plus a 64-number RoPE key, 576 numbers.](/diagrams/i02-kv-sharing.svg "What each attention design caches per token, per layer, for a model with 32 query heads of 128 numbers.")

All four designs keep 32 query heads; they differ only in what they cache.

### Multi-query attention (MQA)

Noam Shazeer's [2019 paper](https://arxiv.org/abs/1911.02150) proposed **multi-query attention (MQA)**: keep many query heads but share a single key head and a single value head across all of them. The cache shrinks by the number of heads. On a translation task, decoding time per token fell from 46 to 3.8 microseconds, while the quality score (BLEU) on the development set moved only from 26.7 to 26.5.

Google's PaLM used it, and Pope et al. showed that, split across chips the right way, it let PaLM 540B handle up to 32 times longer contexts on the same 64 chips. The catch is quality: the [Llama 2 paper](https://arxiv.org/abs/2307.09288) compared the options at 30B parameters and found MQA behind on average. MQA also splits awkwardly across GPUs: with one key-value head and 8 GPUs, each GPU needs its own copy.

### Grouped-query attention (GQA)

[Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245) proposed the middle ground: **grouped-query attention (GQA)**, where query heads are split into groups and each group shares one key-value pair. They showed GQA reaches "quality close to multi-head attention with comparable speed to MQA," and that an existing multi-head model can be converted with about 5% of its original training compute.

GQA became the default. Llama 2 adopted it for its 34B and 70B models after a 30B-parameter test in which it "performs comparably to the MHA baseline on most evaluation tasks." Every Llama 3 model, from 8B to 405B, uses [8 key-value heads](https://arxiv.org/abs/2407.21783).

### Multi-head latent attention (MLA)

DeepSeek took a different route in [DeepSeek-V2](https://arxiv.org/abs/2405.04434). **Multi-head latent attention (MLA)** doesn't share heads. Instead, it compresses each token's keys and values into one small vector, a **latent** of 512 numbers, and caches only that. At attention time, the model rebuilds per-head keys and values from the latent. The rebuild matrices can be folded into the query and output projections, so full keys and values are never written out.

Position adds a complication. Llama-style models mark each token's position with **RoPE** (rotary position embedding), which rotates queries and keys by an amount that depends on their position. DeepSeek found that "RoPE is incompatible with low-rank KV compression," so MLA also caches one small 64-number key that carries position, shared by all heads. Total: 576 numbers per layer. The paper puts it as equal to "GQA with only 2.25 groups," while reporting quality stronger than MHA.

DeepSeek-V3 keeps the same sizes with 128 heads of 128 numbers each. Had it used plain MHA, its cache would be `2 × 61 × 128 × 128 × 2 bytes ≈ 3.8 MiB` per token. MLA stores 1.76% of that, about 57 times less.

### How many users fit on one GPU

For a fair comparison, put all four designs in the same body: Llama 3.1 8B's 32 layers and 32 query heads of 128 numbers. The MLA row is illustrative: it borrows DeepSeek's 576-number cache, since no 8B Llama uses MLA. The GPU is an [H100](https://www.nvidia.com/en-us/data-center/h100/) with 80 GB; the 16-bit weights take 16.06 GB, leaving about 63.9 GB for the cache (ignoring the working memory a real server also needs).

| Design | Numbers per token per layer | Per token | Per 4,096-token user | Users at 4,096 tokens | Users at 128K tokens |
|---|---|---|---|---|---|
| MHA (32 KV heads) | 8,192 | 512 KiB | 2 GiB | 29 | 0 (needs 64 GiB) |
| GQA (8 KV heads) | 2,048 | 128 KiB | 512 MiB | 119 | 3 |
| MLA (576, illustrative) | 576 | 36 KiB | 144 MiB | 423 | 13 |
| MQA (1 KV head) | 256 | 16 KiB | 64 MiB | 952 | 29 |

The same GPU and the same weights serve anywhere from 29 to 952 users, depending only on how attention shares its keys and values. MQA is the smallest but loses some quality. MLA's cache is 2.25 times MQA's, and DeepSeek reports quality above MHA.

## Step 5: one long user versus many short ones

Go back to the real Llama 3.1 8B on an 80 GB H100:

```text
weights                    = 16.06 GB
one 128K-token user's cache = 131,072 × 128 KiB = 16 GiB ≈ 17.2 GB
```

The 63.9 GB left after the weights fits 3 such users, or 119 at 4,096 tokens: one 128K-token conversation takes the memory of 32 ordinary chats.

Long contexts also slow every token. Each decode step reads the weights plus the user's whole cache:

```text
step time  ≈ (16.06 GB + 17.2 GB) ÷ 3.35 TB/s ≈ 9.9 ms
best case  ≈ 1 ÷ 9.9 ms ≈ 100 tokens per second (vs about 208 with an empty cache)
```

At 128K tokens of context, a single user's stream runs at half the speed of a fresh chat, before any batching. For Llama 3.1 70B the weights take about 140 GB, and one 128K user adds another 43 GB (40 GiB).

Part 11 covers the techniques that compress or skip parts of a long context's cache: sliding windows, sparse attention and cache eviction.

## In the real world

### DeepSeek's numbers

Compared with its earlier dense DeepSeek 67B, DeepSeek-V2 "reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times." Not all of that is MLA. For deployment, DeepSeek also converted the weights to FP8 and quantized the cache "into 6 bits on average." One 8-GPU H800 node generated over 50,000 tokens per second. DeepSeek-V4 pushes further: the [DeepSeek-V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) reports needing only 10% of DeepSeek-V3.2's KV cache at a one-million-token context, using new compressed attention designs.

### Storing the cache in fewer bits

The other lever is fewer bytes per number. vLLM's `kv_cache_dtype="fp8"` option stores keys and values as 8-bit floating-point numbers, which the [vLLM docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html) say "enables you to store more tokens in memory, leading to improved throughput and support for longer context windows." One byte instead of two halves Llama 3.1 8B's cache to 64 KiB per token and a 128K user to 8 GiB. The question is how much accuracy you lose, and Part 8 on quantization covers how to check.

### When the cache runs out

Requests grow one token at a time, so a server can admit a batch that later outgrows memory. The [vLLM paper](https://arxiv.org/abs/2309.06180) describes two recovery options:

- **Swapping**: copy a paused request's cache to CPU memory and bring it back later.
- **Recomputation**: throw the cache away, and when the request resumes, run its prompt plus the tokens generated so far as one new prefill. The vLLM paper notes this can be much faster than the original generation, because all those tokens go through in one parallel pass.

Current vLLM [defaults to recomputation](https://docs.vllm.ai/en/latest/configuration/optimization.html) and logs a warning that a request "is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space." Frequent preemption is a sign to give the cache more memory or admit fewer requests.

Layout is a separate problem: early systems lost most of their cache memory to fragmentation, which PagedAttention fixes (Part 5). Keeping caches around between requests, and spilling them to CPU memory or SSD, is Part 6. The [System Design post on serving open models](/blog/serving-open-llms-at-scale/) shows the same math for sizing a 70B fleet.

## Common misconceptions

- **"The KV cache stores the conversation."** It stores per-layer keys and values: 128 KiB per token for Llama 3.1 8B, against a few bytes for the token's text.
- **"Bigger models have bigger caches."** Cache size depends on layers, key-value heads and head size, not total parameters. DeepSeek-V3, with 671B parameters, stores 68.6 KiB per token; Llama 2 7B stores 512 KiB.
- **"With a cache, long contexts are cheap."** The cache removes recomputation, not reading. Every decode step still reads every cached key and value, which is why a 128K-token context halves Llama 3.1 8B's top speed on an H100.
- **"Sharing keys and values always hurts quality."** MQA can lose some quality. GQA with 8 groups is close to MHA in the Llama 2 and GQA papers, and DeepSeek reports MLA beating MHA.
- **"The KV cache and prompt caching are the same thing."** The KV cache belongs to one request and is normally freed when it ends. Prompt caching (or prefix caching) keeps it afterwards so the next request with the same opening can skip that part of prefill. That's Part 6.

## Key takeaways

- The KV cache keeps every earlier token's keys and values, in every layer, so the model never reruns the whole conversation. For a 500-token answer on an 8B model, that's about 1,250 times less arithmetic.
- Its size is `2 × layers × KV heads × head size × bytes` per token: 128 KiB for Llama 3.1 8B, 4.5 MiB for GPT-3's design, 68.6 KiB for DeepSeek-V3.
- One 128K-token user on Llama 3.1 8B needs about 17 GB of cache, more than the 16 GB of weights, and halves that user's top decode speed.
- Multi-query, grouped-query and multi-head latent attention shrink the cache by sharing or compressing keys and values. On one 80 GB GPU, the same 8B body serves 29 users at 4,096 tokens with MHA, 119 with GQA and over 400 with MLA-sized caches.
- Storing the cache in 8 bits halves it again, and when it still runs out, servers pause requests and recompute or swap their caches.

## What's next

The whole story so far has been about bytes moving between memory and the chip. [Part 3](/blog/gpus-for-llms/) opens up the GPU itself: why its fast on-chip memory (SRAM) is tiny compared with its main memory (HBM), what tensor cores do, and how number formats from 32-bit down to 4-bit change what fits and how fast it runs.

## Further reading

- [Efficiently Scaling Transformer Inference (Pope et al., 2022)](https://arxiv.org/abs/2211.05102): the analytical model of inference cost, the 3 TB KV cache example and multi-query attention at PaLM scale.
- [Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019)](https://arxiv.org/abs/1911.02150): the paper that introduced multi-query attention.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245): grouped-query attention and converting existing models to it.
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (DeepSeek-AI, 2024)](https://arxiv.org/abs/2405.04434): multi-head latent attention, decoupled RoPE and the cache comparison table.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): MLA at 671B parameters, with the dimensions used in this article.
- [Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288): a direct MHA vs MQA vs GQA comparison at 30B parameters.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180): KV cache waste, swapping and recomputation.
- [vLLM: Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html) and [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization.html): FP8 cache settings and how preemption works in practice.
- [Understanding and Coding the KV Cache in LLMs from Scratch (Raschka, 2025)](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms): a readable PyTorch implementation, with and without the cache.
