---
title: "Long-Context Inference: Sliding Windows, Sparse Attention, KV Eviction"
description: "What 128K to 1M-token prompts cost on a GPU, and how sliding windows, sparse attention, KV cache eviction and hybrids cut attention and memory."
slug: "long-context-inference"
category: "inference-optimization"
tags: ["long-context", "attention", "caching", "latency", "gpu"]
series: "llm-inference"
seriesOrder: 11
pubDate: 2026-10-23
---

## TL;DR

Long prompts cost twice. Prefill arithmetic grows with the square of the prompt length, and the KV cache grows linearly, so every decode step reads more memory: on Llama 3.1 8B, a 128K-token prompt spends about two thirds of its prefill on attention and needs 17 GB of cache. Real models cut these costs with sliding windows and local-global layers, which cap how far most layers look back, and with sparse attention, which picks a few thousand tokens per query. Servers can also evict or compress cache entries, and hybrid models replace most attention layers with a fixed-size state. Every one of these gives up some ability to use the whole context, so the article ends with how to measure what a model can really use.

## Why it matters

Picture a one-million-token prompt (about 2,000 pages of text) going into a model shaped like Llama 3.1 8B on one NVIDIA H100. At the GPU's peak speed, prefill alone takes about 5 minutes before the first token appears, and 94% of that time is attention. The KV cache for that prompt would be 137 GB, bigger than the 80 GB GPU.

Context windows of that size now ship: DeepSeek-V4 and current Claude models offer one million tokens. To make that affordable, model designers change which tokens each query looks at, how many layers look far back, and what gets stored.

[Fundamentals Part 5](/blog/attention-explained/) showed why attention scales with the square of the length, and [Part 7](/blog/flashattention/) showed that FlashAttention fixes the memory traffic of that computation but not its arithmetic. This part is about doing less of it.

## The intuition

Imagine a lawyer answering questions about a 2,000-page case file. Full attention is rereading the entire file before writing every sentence: thorough, and absurdly slow.

Real lawyers take shortcuts. They keep the last few pages they read fresh in mind (a sliding window). They remember the cover page, because they keep glancing at it to orient themselves (attention sinks). They use an index to jump to the dozen pages that matter for the current question (sparse attention). And they throw away notes they haven't looked at in a while (cache eviction).

The analogy breaks here: a lawyer decides what's relevant by understanding the question. A model decides with learned scores or fixed rules, and a rule that throws away page 1,412 can't know the next question is about page 1,412.

## Step 1: what 128K to 1M tokens actually costs

Use [Part 1](/blog/how-llm-inference-works/)'s model: an H100 does 989 trillion 16-bit floating-point operations (FLOPs) per second at peak and reads memory at 3.35 TB/s. Prefill costs about 2 FLOPs per parameter per token for the weights, plus attention. Attention costs about `4 × n² × head size` FLOPs per head per layer, halved by the causal mask (each token only looks backward).

### Prefill: attention takes over

For Llama 3.1 8B (32 layers, 32 query heads of 128 numbers) and Llama 3.1 70B (80 layers, 64 query heads of 128), at the H100's peak:

| Model and prompt | Weight FLOPs | Attention FLOPs | Attention share of prefill | Prefill time on one H100 at peak |
|---|---|---|---|---|
| 8B, 4,096 tokens | 6.6 × 10¹³ | 4.4 × 10¹² | 6% | 0.07 s |
| 8B, 32,768 | 5.3 × 10¹⁴ | 2.8 × 10¹⁴ | 35% | 0.8 s |
| 8B, 131,072 | 2.1 × 10¹⁵ | 4.5 × 10¹⁵ | 68% | 6.7 s |
| 8B shape, 1,048,576 | 1.7 × 10¹⁶ | 2.9 × 10¹⁷ | 94% | 308 s |
| 70B, 131,072 | 1.9 × 10¹⁶ | 2.3 × 10¹⁶ | 55% | 41 s |
| 70B shape, 1,048,576 | 1.5 × 10¹⁷ | 1.4 × 10¹⁸ | 91% | 1,607 s (27 min) |

Llama 3.1's window ends at 128K, so the 1M rows are the same shapes stretched, to show the trend. Eight GPUs working together divide those times by at most 8: about 5 seconds for the 70B at 128K and over 3 minutes at 1M, before real-world efficiency losses. Weight FLOPs grow 8 times from 128K to 1M; attention FLOPs grow 64 times.

### The KV cache: memory per user

[Part 2](/blog/kv-cache/) sized the cache at 128 KiB per token for Llama 3.1 8B and 320 KiB for the 70B:

| Context | 8B cache | 70B cache |
|---|---|---|
| 4,096 tokens | 0.5 GB | 1.3 GB |
| 32,768 | 4.3 GB | 10.7 GB |
| 131,072 | 17.2 GB | 42.9 GB |
| 1,048,576 | 137 GB | 344 GB |

### Decode: every token rereads the whole cache

Each decode step reads the weights (16.06 GB for the 8B) plus that user's whole cache. For one user on one H100:

| Context already in the cache | Bytes read per step | Step time | Tokens per second | Cache share of the reads |
|---|---|---|---|---|
| Empty | 16.1 GB | 4.8 ms | 209 | 0% |
| 4,096 | 16.6 GB | 5.0 ms | 202 | 3% |
| 32,768 | 20.4 GB | 6.1 ms | 165 | 21% |
| 131,072 | 33.2 GB | 9.9 ms | 101 | 52% |
| 524,288 (hypothetical) | 84.8 GB | 25.3 ms | 40 | 81% |
| 1,048,576 (hypothetical) | 153 GB | 45.8 ms | 22 | 90% |

The last two rows don't fit on one 80 GB GPU; they show where the curve goes. Decode attention is still memory-bound: at 1M tokens it's about 550 billion FLOPs per token, under a millisecond of math against 46 ms of reading.

Prefill is dominated by `n²` arithmetic; decode and capacity, by a cache that grows with `n`. The techniques below attack one or both.

## Step 2: sliding windows and local-global layers

The simplest fix is to stop looking all the way back. In **sliding-window attention**, each token attends only to the most recent `W` tokens.

![Four panels, each showing a query at token 100,000. Full attention reads all 100,000 earlier tokens and the cache keeps all 100,000. Sliding window, as in Mistral 7B, reads only the last 4,096 tokens and the cache keeps 4,096. Sinks plus window, as in StreamingLLM, reads the first 4 tokens plus the last 4,092 and the cache keeps 4,096. Top-k sparse attention, as in DeepSeek Sparse Attention, uses an indexer to pick the top 2,048 tokens to read, but the cache keeps all 100,000.](/diagrams/i11-attention-patterns.svg "Which past tokens one query reads, and what the cache must keep, under four attention patterns.")

Windows shrink both the reads and the cache. Sparse top-k attention shrinks only the reads.

### Longformer and Mistral 7B

[Longformer](https://arxiv.org/abs/2004.05150) (Beltagy, Peters and Cohan, 2020) used "a fixed-size window attention surrounding each token," with cost `O(n × w)`, "which scales linearly with input sequence length." Stacking layers widens what the model can reach: "the receptive field size at the top layer is ℓ × w." A few "global" tokens, such as the question in question answering, attend to everything and are attended to by everything.

[Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023) brought the idea to a decoder chat model, with a window of 4,096 tokens in all 32 layers. Information still hops one window per layer, so "at the last layer, using a window size of W = 4096, we have a theoretical attention span of approximately 131K tokens" (32 × 4,096 = 131,072). Each hop squeezes old information into a newer token's vector, and nothing guarantees it survives.

The serving win is the **rolling buffer cache**. The keys and values for position `i` go into slot `i mod W`, so once the buffer is full, new tokens overwrite the oldest ones and "the size of the cache stops increasing." Mistral 7B has the same cache shape as Llama 3.1 8B (32 layers, 8 key-value heads of 128), so its cache stops at 4,096 × 128 KiB = 512 MiB per user. At 32K tokens that's the paper's "8x" saving, and at 1M tokens it would be 256 times. Prefill attention shrinks the same way: with a 4,096 window, the 8B shape's attention at 1M tokens falls from 2.9 × 10¹⁷ to 2.3 × 10¹⁵ FLOPs, 128 times less.

### Interleaving local and global layers

Pure windows lose long-range recall, so recent models mix layer types. Gemma 2 alternated: a 4,096-token sliding window in every other layer, full attention in the rest. The [Gemma 3 report](https://arxiv.org/abs/2503.19786) (2025) went further: "1 global for every 5 local layers," with local layers given "a smaller span of only 1024 tokens." Only the global layers carry the 128K context. The report found "minimal impact on perplexity" from the ratio change (perplexity measures how surprised the model is by real text; lower is better), and at 32K tokens it measured the KV cache's memory overhead falling from 60% with global-only attention to under 15% with 1,024-token local windows.

Gemma 3 27B's configuration shows the size of the saving: 62 layers, 16 key-value heads of 128, a 1,024-token window, every sixth layer global. In 16-bit numbers, at 128K tokens:

```text
per layer per token = 2 (K and V) × 16 heads × 128 × 2 bytes = 8 KiB
all 62 layers global: 62 × 8 KiB × 131,072        = 62.0 GiB
10 global + 52 local: 10 × 8 KiB × 131,072
                      + 52 × 8 KiB × 1,024        = 10.4 GiB
```

That's about six times less cache, and almost all of what's left sits in the 10 global layers.

OpenAI's [gpt-oss](https://arxiv.org/abs/2508.10925) models (2025) use the same idea more aggressively: "attention blocks alternate between banded window and fully dense patterns, where the bandwidth is 128 tokens." gpt-oss-120b has 36 layers, 18 of each kind, with 8 key-value heads of 64 numbers. At 128K tokens, the dense layers need 4.5 GiB and the banded layers 4.5 MiB in total, half of what all-dense attention would need.

### Attention sinks: windows for endless streams

A plain sliding window fails in a known way. [Fundamentals Part 5](/blog/attention-explained/) described **attention sinks**: models park a lot of attention on the first few tokens, whatever they say. Evict those tokens and the model breaks. [StreamingLLM](https://arxiv.org/abs/2309.17453) (Xiao et al., 2023) keeps "4 initial tokens" plus a rolling window of recent ones, and gave Llama-2, MPT, Falcon and Pythia "stable and efficient language modeling with up to 4 million tokens," up to 22.2 times faster per token than recomputing a sliding window from scratch.

Positions are assigned by slot in the cache, not by place in the original text, so the model never sees a position it wasn't trained on. The paper is blunt about its limits: "it does not extend the models' context window or enhance their long-term memory capabilities." It suits endless chats and agent logs, not questions about the middle of a book. gpt-oss bakes the idea into training, with "a learned bias in the denominator of the softmax" for each head that lets attention "pay no attention to any tokens."

## Step 3: sparse attention, reading only what matters

Windows assume the recent past matters most. **Sparse attention** lets each query choose: it scores the past cheaply, then runs real attention only on the winners.

### Block selection: NSA and MoBA

DeepSeek's [Native Sparse Attention](https://arxiv.org/abs/2502.11089) (NSA; Yuan et al., 2025) runs three branches per query and blends them with a learned gate: "compressed attention for coarse-grained patterns, selected attention for important token blocks, and sliding attention for local context." The coarse branch summarizes every 32 tokens; its scores pick 16 blocks of 64 tokens (always including the first block and 2 local ones), and a 512-token window covers the recent past. At 64K tokens, a query loads about 5,632 tokens' worth of keys and values instead of 65,536.

"Native" means the model is trained with the sparsity from the start, not pruned afterwards. On a 27-billion-parameter model (3 billion active per token), NSA beat full attention on 7 of 9 general benchmark metrics, retrieved the needle "across all positions" in a 64K-token needle-in-a-haystack test, and ran up to 9.0 times faster forward and 6.0 times faster backward at 64K on A100 GPUs. Its 11.6 times decoding figure is an estimate from memory traffic, not a timing. The paper won a best paper award at ACL 2025.

Moonshot AI's [MoBA](https://arxiv.org/abs/2502.13189) (Mixture of Block Attention, 2025) treats blocks of context like experts in a mixture-of-experts model: each query scores every block by its average key and attends to the top few, plus its own block. It "has already been deployed to support Kimi's long-context requests," and was up to 6.5 times faster at prefilling 1M tokens.

### Token selection: DeepSeek Sparse Attention

[DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) (September 2025) made exactly one architectural change to its predecessor: **DeepSeek Sparse Attention (DSA)**. A small **lightning indexer**, with 64 heads of 128 numbers running in 8-bit (FP8), scores every past token for each query, and the main attention reads only the top 2,048 key-value entries. In the report's words, DSA "reduces the core attention complexity of the main model from O(L²) to O(Lk)," where `L` is the context length and `k` is 2,048. The indexer is still `O(L²)`, but much cheaper per pair.

At 128K tokens, each query's main attention touches 2,048 of 131,072 entries, 1.6% of them. The model was trained into it in two stages: 2.1 billion tokens to teach the indexer to mimic full attention, then 943.7 billion tokens of sparse training. The [model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) reports "performance on par with V3.1-Terminus," and DeepSeek [cut API prices by "50%+"](https://api-docs.deepseek.com/news/news250929) at launch. The report's cost chart, from its H800 GPU service, shows the gap widening with position; reading it at 128K, prefill falls from about $0.65 to about $0.20 per million tokens and decode from about $2.15 to under $0.30.

DSA doesn't save memory. The cache still holds every token, because any of them might be picked at the next step, so DeepSeek-V4 adds compression on top. Its [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) describes "a hybrid attention mechanism combining Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA)," and says that at a 1M-token context, DeepSeek-V4-Pro needs "only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2."

## Step 4: shrinking the cache, and checking what you lost

The methods above are baked in by training. A second family works on any model at serving time by dropping or squeezing cache entries.

### Eviction: keep the heavy hitters

[H2O](https://arxiv.org/abs/2306.14048) (Heavy-Hitter Oracle; Zhang et al., 2023) observed that a small set of tokens collects most of the attention. It keeps those "heavy hitters," ranked by accumulated attention score, plus the most recent tokens, and evicts the rest. With a 20% cache budget, a five-times memory cut, it reported quality "comparable" to the full cache on OPT, LLaMA and GPT-NeoX, and up to 29 times the throughput of DeepSpeed Zero-Inference and Hugging Face Accelerate on OPT-6.7B and OPT-30B.

[SnapKV](https://arxiv.org/abs/2404.14469) (Li et al., 2024) compresses once, right after prefill. It looks at how the last few prompt tokens, an "observation window," attend to the rest, and keeps the positions they vote for in each head, plus their neighbors. The question usually sits at the end of the prompt, so this keeps what the question needs. At 16K-token inputs it reported 3.6 times faster generation and 8.2 times better memory efficiency, and it handled 380K tokens on one A100 80 GB with "only a negligible accuracy drop" in needle-in-a-haystack tests.

### Compression: fewer bits per entry

[Quantization](/blog/quantization/) makes each entry smaller instead of removing it. An 8-bit cache halves Llama 3.1 8B's 128K-token cache from 17.2 to 8.6 GB, and research methods such as KIVI go to 2 bits. The two stack: evict half, store the rest in 8 bits, and you hold four times as many tokens.

### The catch: a fact you evicted is gone

Eviction is a bet on the future. H2O ranks tokens by the attention they've received so far, SnapKV by what the current question needs. Ask a follow-up about something neither kept and the model can't answer it, and won't say so. Multi-turn chat over one document is where this bites.

The classic **needle-in-a-haystack** test, which hides one sentence in filler and asks for it back, is too easy to catch this. [RULER](https://arxiv.org/abs/2404.06654) (Hsieh et al., NVIDIA, 2024) found that "despite achieving nearly perfect accuracy in the vanilla NIAH test, almost all models exhibit large performance drops as the context length increases." RULER adds harder tasks (multiple needles, tracing variables through chains, aggregating word counts, question answering) and defines a model's **effective context length** as the longest length at which it still scores above Llama-2-7B's 85.6% at 4K. Of 17 models that "all claim context sizes of 32K tokens or greater, only half of them can maintain satisfactory performance at the length of 32K." Selected rows:

| Model | Claimed context | Effective context on RULER |
|---|---|---|
| Llama 3.1 8B | 128K | 32K |
| Llama 3.1 70B | 128K | 64K |
| GPT-4 (1106 preview) | 128K | 64K |
| Gemini 1.5 Pro | 1M | over 128K (longest tested) |
| LWM | 1M | under 4K |

Run RULER-style tests, not just a single needle, at your real prompt lengths before and after turning on any cache compression. The ["lost in the middle"](/blog/positional-encoding/) effect, where facts in the middle of a long prompt get used less, adds a second reason to check by position.

### Hybrids: no growing cache at all

The most radical option replaces most attention layers. A **state-space model** such as [Mamba](https://arxiv.org/abs/2312.00752) (Gu and Dao, 2023) reads text like a recurrent network, folding each token into a fixed-size state, so generation "does not require a cache of previous elements." It reported 5 times the inference throughput of a Transformer of similar size. Pure state-space models struggle with exact recall, so production designs mix the two. AI21's [Jamba](https://arxiv.org/abs/2403.19887) uses one attention layer for every seven Mamba layers; at 256K tokens its cache is 4 GB, against 32 GB for Mixtral and 128 GB for Llama-2 6.7B.

## In the real world

### Spreading one prompt across GPUs

A 1M-token prompt is too much prefill and too much cache for one GPU. **Context parallelism** splits the sequence itself across GPUs; [Part 10](/blog/model-parallelism/) covers the other ways of splitting a model. [Ring Attention](https://arxiv.org/abs/2310.01889) (Liu, Zaharia and Abbeel, 2023) arranges GPUs in a ring: each holds one chunk of queries, keys and values, and passes its key-value block to the next GPU while computing on the one it just received. The maximum length grows with the number of devices; the paper reached over 1 million tokens for a 7B model on 32 A100s. Meta trained Llama 3's 128K stage with context parallelism across 16 GPUs per group, using an all-gather of keys and values rather than a ring, because with grouped-query attention the keys and values are small.

### Chunked prefill and prefix caching

A 128K-token prompt shouldn't freeze everyone else's stream. [Chunked prefill](/blog/continuous-batching/) feeds it in slices alongside other users' decode steps: at a 2,048-token budget per step, that's about 64 steps. Each chunk rereads the keys and values of all earlier chunks, so chunking adds memory traffic for very long prompts.

The bigger saving comes from never prefilling the same document twice. When many questions hit the same long document, [prefix caching](/blog/prefix-caching/) keeps its cache: DeepSeek [reported](https://api-docs.deepseek.com/news/news0802) a 128K-token prompt's first-token latency falling from 13 seconds to 500 ms on a cache hit. Windowed layers complicate this, because their caches only hold the last window. DeepSeek's DeepInfer engine handles it with "SWA bounded replay": on a hit it reuses the stored full-context cache and recomputes "only the most recent 128 tokens" for the sliding-window layers, which makes that cost "independent of context length and layer count." Part 14 looks at DeepInfer in detail.

### What long prompts cost on APIs, as of October 2026

Some providers charge a higher per-token rate once a prompt crosses a length threshold:

| Provider and model | Threshold | Input price per million tokens, below / above | Output price per million tokens, below / above |
|---|---|---|---|
| [Google Gemini 3.1 Pro Preview](https://ai.google.dev/gemini-api/docs/pricing) | 200K tokens | $2.00 / $4.00 | $12.00 / $18.00 |
| [OpenAI gpt-5.5](https://developers.openai.com/api/docs/pricing) | 272K tokens | $5 / $10 | $30 / $45 |
| [Anthropic Claude 4.6 and later](https://platform.claude.com/docs/en/about-claude/pricing) (except Haiku 5.5) | None | Flat across the 1M window | Flat |
| [Anthropic Claude Haiku 5.5](https://platform.claude.com/docs/en/about-claude/pricing) | 100K tokens | $0.10 / $0.50 | $0.50 / $2.50 |

Anthropic's page puts the flat rule plainly: "A 900k-token request is billed at the same per-token rate as a 9k-token request." Gemini's newest flagship, Gemini 4 Argon, has no length tier.

### When retrieval beats a long context

Long context doesn't replace retrieval-augmented generation (RAG). A 10-million-document corpus is about 25 billion tokens, as [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/) works out, which no context window holds. Even when the data fits, cost scales with it: one 1M-token prompt at Gemini 3.1 Pro's long-context rate is $4.00 of input, while a RAG prompt of 10,000 retrieved tokens at the standard rate is $0.02, 200 times less, and it prefills in a fraction of the time.

Long context wins when the material fits, needs to be read as a whole (a contract's cross-references, a codebase's call graph) and gets reused through prefix caching. Retrieval wins when the corpus is large or changes often, or when answers must cite sources. Many systems do both: retrieve generously, then let a long window hold everything retrieved.

## Common misconceptions

- **"A 1M-token window means the model uses 1M tokens well."** RULER found Llama 3.1 8B's effective context to be 32K out of a claimed 128K, and only half the models it tested held up at 32K. Test at the lengths you plan to use.
- **"FlashAttention made long context cheap."** It removed the memory traffic of the n×n score grid, not the arithmetic. At 128K tokens, attention is still about two thirds of Llama 3.1 8B's prefill FLOPs.
- **"Sliding-window models can't see past the window."** Information hops forward one window per layer, which is how Mistral 7B claims a theoretical span of about 131K tokens. Gemma 3 and gpt-oss keep some full-attention layers for exactly this reason.
- **"Sparse attention shrinks the KV cache."** Top-k methods like DSA keep every token's cache entry, because any of them might be chosen next. They cut reads and arithmetic. Shrinking memory takes windows, compression or eviction.
- **"StreamingLLM gives a model infinite context."** It lets a model keep generating over an endless stream, but anything that leaves the window is gone, as its authors stress.

## Key takeaways

- Long prompts cost twice: prefill attention grows with the square of the length, and the KV cache grows linearly and is reread at every step. A 128K-token prompt on Llama 3.1 8B needs 6.7 seconds of H100 prefill at peak and 17 GB of cache, and halves decode speed.
- Sliding windows cap both costs. Mistral 7B's 4,096-token rolling buffer stops the cache from growing; Gemma 3's five local layers per global layer cut a 128K cache about six times.
- Sparse attention picks what to read. DeepSeek's DSA reads 2,048 tokens per query at any length but keeps the whole cache, which is why DeepSeek-V4 adds compression and reports 10% of V3.2's cache at 1M tokens.
- Eviction and quantization shrink the cache of any model at serving time, but an evicted fact can't come back. Measure with RULER-style tests, not a single needle.
- In production, long context is a system problem: context parallelism for prefill, chunked prefill for fairness, prefix caching for reuse, and retrieval when the data is bigger than any window.

## What's next

Long prompts make prefill heavy and compute-bound, while decode stays light and memory-bound, and putting both on the same GPU makes them fight. Part 12 covers disaggregated serving: running prefill and decode on separate groups of GPUs, shipping the KV cache between them, and how llm-d schedules it all on Kubernetes.

## Further reading

- [Longformer: The Long-Document Transformer (Beltagy, Peters and Cohan, 2020)](https://arxiv.org/abs/2004.05150): sliding-window, dilated and global attention, and the receptive-field argument.
- [Mistral 7B (Jiang et al., 2023)](https://arxiv.org/abs/2310.06825): a 4,096-token window and the rolling buffer cache in a production model.
- [Gemma 3 Technical Report (Gemma Team, 2025)](https://arxiv.org/abs/2503.19786): five local layers per global layer, 1,024-token windows and the KV cache memory ablations.
- [gpt-oss-120b and gpt-oss-20b Model Card (OpenAI, 2025)](https://arxiv.org/abs/2508.10925): alternating 128-token banded and dense attention, with learned attention sinks.
- [Efficient Streaming Language Models with Attention Sinks (Xiao et al., 2023)](https://arxiv.org/abs/2309.17453): StreamingLLM, and why the first four tokens must stay.
- [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (Yuan et al., 2025)](https://arxiv.org/abs/2502.11089): compressed, selected and sliding branches trained from scratch.
- [DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention (DeepSeek-AI, 2025)](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf): the lightning indexer, top-2,048 selection and the cost curves.
- [MoBA: Mixture of Block Attention for Long-Context LLMs (Lu et al., 2025)](https://arxiv.org/abs/2502.13189): block-level top-k attention deployed in Kimi.
- [DeepSeek-V4-Pro model card (DeepSeek-AI, 2026)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro): compressed sparse and heavily compressed attention at 1M tokens.
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models (Zhang et al., 2023)](https://arxiv.org/abs/2306.14048): evicting all but the most-attended and most recent tokens.
- [SnapKV: LLM Knows What You are Looking for Before Generation (Li et al., 2024)](https://arxiv.org/abs/2404.14469): compressing the prompt's cache by what the question attends to.
- [RULER: What's the Real Context Size of Your Long-Context Language Models? (Hsieh et al., 2024)](https://arxiv.org/abs/2404.06654) and its [GitHub leaderboard](https://github.com/NVIDIA/RULER): effective context length, measured.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu and Dao, 2023)](https://arxiv.org/abs/2312.00752) and [Jamba: A Hybrid Transformer-Mamba Language Model (Lieber et al., 2024)](https://arxiv.org/abs/2403.19887): state-space layers and the hybrid route.
- [Ring Attention with Blockwise Transformers for Near-Infinite Context (Liu, Zaharia and Abbeel, 2023)](https://arxiv.org/abs/2310.01889): splitting one sequence across a ring of GPUs.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): context parallelism in practice for the 128K training stage.
- [DeepInfer: KV cache hits and recomputation (DeepSeek, 2026)](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md): how a production engine combines prefix caching with sliding-window layers.
