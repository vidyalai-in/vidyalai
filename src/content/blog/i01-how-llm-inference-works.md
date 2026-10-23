---
title: "How LLM Inference Works: Prefill, Decode and the Memory Wall"
description: "Why a GPU that can do a quadrillion operations a second spends most of its time waiting when it writes your answer, and what that means for speed and cost."
slug: "how-llm-inference-works"
category: "inference-optimization"
tags: ["serving", "latency", "gpu", "caching"]
series: "llm-inference"
seriesOrder: 1
pubDate: 2026-10-23
---

## TL;DR

Inference is the work a trained model does to answer a request. It runs in two phases. **Prefill** reads your whole prompt in one parallel pass and is limited by how fast the GPU can calculate. **Decode** then writes the answer one token at a time, and every single token requires reading all of the model's weights from memory, so it's limited by memory bandwidth, not by calculation. For an 8 billion-parameter model on an NVIDIA H100, that caps one user at about 208 tokens per second while the GPU's arithmetic units sit more than 99% idle. Almost every inference optimization, from the KV cache and batching to quantization, speculative decoding and disaggregated serving, is a way around that memory wall. This article builds the mental model the rest of the series uses.

## Why it matters

An NVIDIA H100 can perform about [989 trillion](https://www.nvidia.com/en-us/data-center/h100/) 16-bit multiply-adds per second. A model with 8 billion parameters needs about 16 billion operations to produce one token. Divide one by the other and the GPU should write about 60,000 tokens per second.

Serving one user, it manages about 200, roughly 300 times fewer. Better software won't close that gap. The limit is the time it takes to move the model's weights from memory to the chip.

That gap shapes most of how LLMs are run. It's why providers batch many users onto the same GPUs, why long conversations get slower, and why projects like [vLLM](https://docs.vllm.ai/en/latest/), [llm-d](https://llm-d.ai/) and DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer) exist. To follow any of them, you need two ideas: prefill and decode, and the difference between compute-bound and memory-bound work.

This series assumes you know the basic generation loop: a model predicts one token, appends it, and runs again. If that's new, read [How a Large Language Model Works](/blog/how-llms-work/) first.

## The intuition

Picture a chef in a small kitchen with a recipe book so large it lives in a warehouse down the street. The chef is fast. The trip to the warehouse is slow.

- Prefill is like reading the whole order at once. The chef fetches the book once and cooks every line of the order with it open.
- Decode is like being asked for one dish, then the next dish only after the first is served, and having to fetch the entire book again for each dish. The chef spends almost all day walking.

The fix: if you have to fetch the book anyway, cook for many tables at once. That's batching, and it's the first big lever in LLM serving. Horace He's essay [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) uses a similar picture: compute is a factory, memory is the warehouse, and the road between them is usually the bottleneck.

The analogy breaks in one way: the chef's book never changes, but in an LLM each table's order also grows as the meal goes on. That growing part is the KV cache, and we'll get to it.

## Step 1: what happens to one request

When you send a prompt to an inference server, it goes through four stages:

1. Tokenize: the text becomes token IDs.
2. Prefill: the model processes all prompt tokens together and produces the first output token.
3. Decode: the model produces one token per step, each step feeding the new token back in, until it emits an end token or hits a length limit.
4. Detokenize and stream: each new token is turned back into text and sent to you as it's produced.

![Prefill takes a 1,000-token prompt through one parallel pass that writes the KV cache and produces the first output token, ending the time to first token. Decode then loops: read all weights plus the KV cache, produce one new token, append its key and value to the cache, and stream the token to the user, until a stop condition.](/diagrams/i01-prefill-decode.svg "Prefill runs once per request. Decode runs once per output token, and each step re-reads the weights and the KV cache.")

The two phases look similar from the outside (both run the same model), but they stress the hardware in opposite ways.

## Step 2: prefill does a lot of math at once

In prefill, every prompt token goes through the model at the same time. The core of the work is matrix multiplication: the weights are a big matrix, the prompt is a stack of token vectors, and the GPU multiplies them in one go.

A forward pass costs about [2 floating-point operations (FLOPs) per parameter per token](https://arxiv.org/abs/2001.08361): one multiply and one add. Take Llama 3.1 8B, which has 8.03 billion parameters, and a 1,000-token prompt:

```text
FLOPs  = 2 × 8.03 billion parameters × 1,000 tokens ≈ 16 trillion
time   = 16 trillion ÷ 989 trillion FLOPs per second ≈ 16 ms (at the H100's peak)
```

Real systems run below peak, and attention adds work that grows with prompt length. Still, a long prompt is a lot of arithmetic, and the GPU does it efficiently. The weights are read from memory once and used for all 1,000 tokens. Prefill is **compute-bound**: the GPU's arithmetic units are the limit.

Prefill sets the **TTFT** (time to first token): how long you wait before anything appears. [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) defines it as "how quickly users start seeing the model's output after entering their query." Long documents in the prompt, retrieval results and chat history all make prefill, and so TTFT, longer. [How to Reduce Time to First Token](/blog/reduce-time-to-first-token/) covers that in depth.

## Step 3: decode reads everything to make one token

After the first token, the model has to produce the rest one at a time. Token 2 depends on token 1, token 3 on token 2. There's nothing to parallelize within a single request.

Each step pushes just one new token through the model. But to do it, the GPU must stream every weight from its high-bandwidth memory (HBM) into its compute units. For Llama 3.1 8B stored in 16-bit numbers:

```text
weights        = 8.03 billion × 2 bytes ≈ 16.06 GB
H100 bandwidth = 3.35 TB per second
one step       ≥ 16.06 GB ÷ 3.35 TB/s ≈ 4.8 ms
best case      ≈ 1 ÷ 4.8 ms ≈ 208 tokens per second
```

That's the ceiling for a single user, before counting anything else. The math for that step takes about 0.02 ms; the memory transfer takes 4.8 ms. Decode is **memory-bound**: the GPU spends almost all its time waiting for data. As [Databricks puts it](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices), "Generating the first token is typically compute-bound, while subsequent decoding is memory-bound."

Decode sets the **TPOT** (time per output token), also called **ITL** (inter-token latency): the gap between tokens as they stream in. Together they give the total response time:

```text
total latency = TTFT + TPOT × number of output tokens
example:        0.2 s + 0.02 s × 500 tokens = 10.2 s
```

For a 500-token answer, decode is 98% of the wait. That's why a fast-feeling chat app cares about TPOT at least as much as TTFT, and why a model that "thinks" for thousands of tokens before answering is expensive to serve.

## Step 4: the memory wall, in one number

The useful way to compare the two phases is **arithmetic intensity**: how many FLOPs you do for each byte you read from memory.

A GPU has a balance point, sometimes called the ridge point of its roofline. Below it, you're memory-bound; above it, compute-bound. For the H100:

```text
ridge point = 989 trillion FLOPs/s ÷ 3.35 trillion bytes/s ≈ 295 FLOPs per byte
```

Work with fewer than 295 FLOPs per byte can't keep the H100's arithmetic busy. Now compare:

- Decode, one user: each weight (2 bytes) is used for one multiply-add (2 FLOPs). That's **1 FLOP per byte**. The GPU is running at about 1/295 of its arithmetic capacity: more than 99.6% idle.
- Prefill, 1,000 tokens: each weight is used 1,000 times per read. About 1,000 FLOPs per byte, well past the ridge.

So what if decode served many users at once? The weights are read once per step no matter how many users share it. Each extra user adds math but almost no extra weight traffic. Here's Llama 3.1 8B on one H100, counting weights only:

| Users per step (batch) | Memory time | Math time | Step time | Total tokens/s | Arithmetic busy |
|---|---|---|---|---|---|
| 1 | 4.79 ms | 0.02 ms | 4.79 ms | 209 | 0.3% |
| 8 | 4.79 ms | 0.13 ms | 4.79 ms | 1,669 | 2.7% |
| 64 | 4.79 ms | 1.04 ms | 4.79 ms | 13,350 | 22% |
| 295 | 4.79 ms | 4.79 ms | 4.79 ms | 61,535 | 100% |
| 512 | 4.79 ms | 8.31 ms | 8.31 ms | 61,582 | 100% |

Up to about 295 users, each one gets the same 4.79 ms step, and total throughput grows almost for free. Past that, the GPU becomes compute-bound, and more users just make everyone slower. That's why LLM serving runs on batching: one user per GPU is very expensive, and hundreds of users per GPU is how providers make the numbers work.

Real systems never hit those ideal numbers, and the biggest reason is the next step.

## Step 5: the KV cache, the part that grows

Recall from [Part 1 of the Fundamentals series](/blog/how-llms-work/) that attention lets each new token look back at every earlier token. To do that, the model needs two vectors for every earlier token in every layer: a key (K) and a value (V). Recomputing them for the whole conversation at every step would waste enormous work, so servers compute them once and keep them in GPU memory. That store is the **KV cache**.

Its size per token is simple to work out. For each layer, you store a key and a value for each key-value head:

```text
KV bytes per token = 2 (K and V) × layers × KV heads × head size × bytes per number
Llama 3.1 8B       = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB per token
```

[NVIDIA](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) and [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) give the same formula. Now scale it:

- One 4,096-token conversation: 128 KiB × 4,096 = **512 MiB**.
- One full 128K-token context: **16 GiB**, as much as the model's weights.
- An 80 GB H100 with 16 GB of weights has about 64 GB left, room for about **488,000 tokens** of cache in total: around 120 users at 4,096 tokens each, before counting any other memory the server needs.

Every decode step also reads each user's whole cache. Take 64 users at 4,096 tokens each:

```text
weights read per step  = 16.06 GB
KV cache read per step = 64 × 4,096 tokens × 128 KiB ≈ 34.4 GB
step time              ≈ (16.06 + 34.4) GB ÷ 3.35 TB/s ≈ 15 ms
throughput             ≈ 64 ÷ 15 ms ≈ 4,250 tokens/s (vs 13,350 with weights alone)
```

At this point the cache is more than twice the traffic of the weights. This is why long conversations slow down, why memory, not compute, decides how many users fit on a GPU, and why so many optimizations target the KV cache. Part 2 is all about it.

## How serving is measured

Four numbers come up in every inference discussion:

| Metric | What it measures | Who feels it |
|---|---|---|
| TTFT (time to first token) | Prompt arrival to first token out | The user waiting for anything to happen |
| TPOT / ITL (time per output token) | Average gap between tokens | The user reading the stream |
| Throughput | Total tokens per second across all users | Whoever pays for the GPUs |
| Goodput | Requests per second that meet both the TTFT and TPOT targets | Both: speed that actually counts |

Goodput comes from the [DistServe paper](https://arxiv.org/abs/2401.09670). It's needed because the targets pull against each other. Bigger batches raise throughput but stretch every user's TPOT, and mixing long prefills into a decode batch stalls everyone's stream. A system tuned only for throughput can look great on a dashboard while every user waits.

## In the real world

### The optimizations in this series

Each later part takes on one part of the wall:

| Part | Technique | The bottleneck it attacks |
|---|---|---|
| 2 | KV cache (and shrinking it with GQA and MLA) | Recomputing past tokens; cache size |
| 3 | Continuous batching, chunked prefill | Idle compute in decode; prefill stalling decode |
| 4 | PagedAttention and vLLM | KV cache memory wasted by fragmentation |
| 5 | FlashAttention | Memory traffic inside the attention computation |
| 6 | Quantization (8-bit, 4-bit) | Bytes per weight and per cached value |
| 7 | Speculative decoding | One token per expensive weight read |
| 8 | Tensor, pipeline and expert parallelism | Models too big for one GPU's memory |
| 9 | Disaggregated serving and llm-d | Prefill and decode fighting over the same GPU |
| 10 | DeepSeek's inference stack and DeepInfer | All of the above, at production scale |

### How much the fixes are worth

The vLLM team found that serving systems before it used only [20.4% to 38.2%](https://arxiv.org/abs/2309.06180) of their KV cache memory for actual tokens; the rest was lost to fragmentation and over-reservation. Their fix, PagedAttention, improved throughput [2 to 4 times](https://arxiv.org/abs/2309.06180) at the same latency. Continuous batching alone has been measured at [up to 23 times](https://www.anyscale.com/blog/continuous-batching-llm-inference) the throughput of naive batching.

### The engines

Today most open models are served by a handful of engines:

- [vLLM](https://docs.vllm.ai/en/latest/), "a fast and easy-to-use library for LLM inference and serving," started at UC Berkeley and introduced PagedAttention.
- [SGLang](https://arxiv.org/abs/2312.07104) added RadixAttention, which reuses the KV cache across requests that share a prefix.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) is NVIDIA's engine, tuned for its own GPUs.
- [llm-d](https://llm-d.ai/) sits a level above: "an open-source, Kubernetes-native stack for distributed LLM inference" that runs vLLM or SGLang underneath and adds cache-aware routing and prefill/decode disaggregation. Red Hat [launched it in May 2025](https://www.redhat.com/en/about/press-releases/red-hat-launches-llm-d-community-powering-distributed-gen-ai-inference-scale) with CoreWeave, Google Cloud, IBM Research and NVIDIA as founding contributors.
- [DeepInfer](https://github.com/deepseek-ai/DeepInfer), open-sourced by DeepSeek in October 2026, is "the high-performance inference framework that powers DeepSeek's large-scale production services." It was originally built primarily on vLLM.

### DeepSeek at production scale

DeepSeek's [published overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) of its V3/R1 serving system shows the prefill/decode split in practice. Each 8-GPU H800 node handled about **73,700 input tokens per second** during prefill (counting tokens served from cache) but about **14,800 output tokens per second** during decode, roughly five times fewer. Prefill and decode ran on separate groups of machines, an idea Part 9 covers.

### Newer GPUs

The [H200](https://www.nvidia.com/en-us/data-center/h200/) keeps the H100's compute but raises bandwidth to 4.8 TB/s and memory to 141 GB, which lifts the single-user ceiling for our 8B model from about 208 to about 300 tokens per second.

## Common misconceptions

- **"A faster GPU means faster chat."** For a single user, decode speed tracks memory bandwidth, not FLOPs. Two GPUs with the same FLOPs and different bandwidth will stream tokens at different speeds.
- **"Long prompts only cost time at the start."** They also grow the KV cache, and every later decode step has to read it. Long contexts slow the whole answer and reduce how many users fit on the GPU.
- **"Batching makes each user wait longer."** In the memory-bound region, extra users ride along almost free: in the table above, 64 users get the same 4.79 ms step as one. The cost shows up only past the ridge point, or when the KV cache traffic grows.
- **"Throughput is the number that matters."** Throughput without latency targets is how you build a system that's cheap and miserable. Goodput, requests that meet both TTFT and TPOT targets, is the honest measure.
- **"Prefill and decode are the same work."** They run the same model but hit opposite limits, which is why DeepSeek's production system and llm-d run them on different machines.

## Key takeaways

- Inference has two phases: prefill processes the prompt in parallel and is compute-bound; decode writes one token per step and is memory-bound.
- Every decode step reads all of the model's weights. For Llama 3.1 8B on an H100, that alone caps a single user at about 208 tokens per second.
- The H100 needs about 295 FLOPs per byte to stay busy; single-user decode does 1. Batching many users per step is the main way to close that gap.
- The KV cache stores past keys and values so they aren't recomputed. It costs 128 KiB per token for Llama 3.1 8B and quickly outgrows the weights in memory traffic.
- TTFT, TPOT, throughput and goodput are the four numbers to watch. Every later part of this series improves at least one of them.

## What's next

The KV cache makes generation fast, then fills the GPU's memory. Part 2 opens it up: what exactly gets stored, how grouped-query attention and DeepSeek's multi-head latent attention shrink it, and why a single long-context user can cost as much memory as the model itself.

## Further reading

- [LLM Inference Performance Engineering: Best Practices (Agarwal et al., Databricks, 2023)](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices): defines TTFT and TPOT and shows why decode is memory-bound.
- [Mastering LLM Techniques: Inference Optimization (Verma and Vaidya, NVIDIA, 2023)](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/): prefill, decode and the KV cache size formula with worked examples.
- [Making Deep Learning Go Brrrr From First Principles (Horace He)](https://horace.io/brrr_intro.html): the clearest intuition for compute-bound versus memory-bound work.
- [Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180): the KV cache fragmentation problem and the fix behind vLLM.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (Zhong et al., 2024)](https://arxiv.org/abs/2401.09670): the goodput metric and why splitting the phases helps.
- [LLM Inference Unveiled: Survey and Roofline Model Insights (Yuan et al., 2024)](https://arxiv.org/abs/2402.16363): roofline analysis applied across many inference techniques.
- [How continuous batching enables 23x throughput in LLM inference (Daniel et al., Anyscale, 2023)](https://www.anyscale.com/blog/continuous-batching-llm-inference): a benchmark-driven explanation of batching.
- [DeepSeek-V3/R1 Inference System Overview (DeepSeek, 2025)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md): real production numbers for disaggregated, expert-parallel serving.
- [DeepInfer (DeepSeek, 2026)](https://github.com/deepseek-ai/DeepInfer): DeepSeek's open-sourced production inference engine.
- [llm-d](https://llm-d.ai/): the Kubernetes-native distributed inference stack built on vLLM.
