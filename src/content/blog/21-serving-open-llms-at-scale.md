---
title: "Serving Open-Weight LLMs at Scale: Batching, KV Cache, Quantization"
description: "How to size and run an open-weight LLM fleet: prefill vs decode, KV-cache math, continuous batching, FP8/FP4 quantization and goodput per dollar."
slug: "serving-open-llms-at-scale"
category: "cost-latency-serving"
tags: ["serving", "gpu", "latency", "cost", "caching"]
seriesOrder: 21
pubDate: 2026-10-16
sourceDoc: "15jgtoXCDflquvlaEjUw7LYBq2BmcoregHbUPWEtQCxw"
---

## TL;DR

LLM inference has two phases that behave very differently. Prefill (processing the prompt) is compute-bound; decode (generating one token at a time) is memory-bandwidth-bound and limited by how much KV cache fits in GPU memory. Serving at scale is therefore about keeping HBM (high-bandwidth memory) busy without breaking latency SLOs (service-level objectives). The standard design is a continuous-batching engine with a paged KV cache (vLLM, SGLang or TensorRT-LLM), plus chunked prefill and prefix caching, with weights quantized to FP8 (or FP4 on Blackwell) and the KV cache to FP8 so more sequences fit per GPU. Use tensor parallelism inside an NVLink domain and data-parallel replicas across nodes, size the fleet from tokens per second per replica measured at your TTFT and TPOT targets, and judge success by goodput: requests per second that meet the SLO, per dollar.

## Requirements to pin down first

- **Which model, and what architecture?** A dense model (for example Llama 3.3 70B) and a mixture-of-experts (MoE) model (for example DeepSeek-V4, Qwen3-235B-A22B, Kimi K2 or gpt-oss-120b) need different parallelism plans. MoE adds expert parallelism and changes the memory math. Also check the attention layout: the GQA (grouped-query attention) ratio, or a compressed scheme like MLA (multi-head latent attention), sets the KV-cache size per token.
- **What does the traffic look like?** Peak QPS, the input and output token distributions, and how much prompt prefix is shared (system prompts, RAG context, multi-turn chat history). Shared prefixes decide whether prefix caching is a 10% or a 50%+ win.
- **What are the SLOs?** TTFT (time to first token) at p50 and p99, for example under 500 ms and under 2 s. TPOT (time per output token, also called inter-token latency), for example under 50 ms, which is at least 20 tok/s per user and faster than people read. Plus availability.
- **Interactive, batch, or both?** Offline jobs can run at very large batch sizes with no latency constraint. Mixing them with interactive traffic requires priority scheduling, or the batch jobs will eat the latency budget.
- **How much quality loss is acceptable?** Decide up front whether quantization must stay within, say, 0.5 points on your eval suite, and which evals count. This decides between BF16, FP8 and 4-bit.

## Where the bottleneck sits: a worked example

The arithmetic below uses Llama 3.3 70B because its configuration is public and simple: 80 layers, 64 query heads, 8 KV heads, head dimension 128 (the same architecture as Llama 3.1 70B). The method carries over to any model; plug in its config.

### Weights

70B parameters take about 140 GB in BF16, about 70 GB in FP8, and about 35-38 GB in 4-bit formats (INT4 AWQ/GPTQ with group scales, or NVFP4/MXFP4 on Blackwell-class GPUs).

### KV cache per token

For every token in context, each layer stores a key and a value vector per KV head:

```text
KV bytes/token = 2 (K and V) × layers × kv_heads × head_dim × bytes_per_value
               = 2 × 80 × 8 × 128 × 2 bytes   = 327,680 bytes ≈ 320 KiB  (BF16)
               = 2 × 80 × 8 × 128 × 1 byte    = 163,840 bytes ≈ 160 KiB  (FP8)
```

Without GQA (all 64 heads storing K and V) the same model would need 2 × 80 × 64 × 128 × 2 = 2.6 MB per token, 8x more. That is why GQA, MLA and newer compressed-attention schemes matter so much for serving. DeepSeek-V4's model card, for example, reports needing 10% of the KV cache of DeepSeek-V3.2 at 1M-token context.

### Capacity on one replica (4x H100 80 GB, TP=4)

- Total HBM: 4 × 80 GB = 320 GB. At about 90% usable, that is 288 GB.
- Subtract FP8 weights (70 GB) and activations and workspace (about 15 GB): roughly 200 GB left for KV cache.
- 200 GB ÷ 163,840 bytes per token ≈ 1.2M tokens of FP8 KV cache.
- At a 4K-token average context: 1.2M ÷ 4,096 ≈ 300 concurrent sequences.

### Decode step time

Every decode step reads all the weights plus the KV cache of every active sequence.

- Weights: 70 GB ÷ 4 GPUs = 17.5 GB per GPU. At 3.35 TB/s (H100 SXM HBM3), that is a floor of about 5.2 ms per step.
- KV reads at batch 128: 128 sequences × 4,096 tokens × 163,840 bytes ≈ 86 GB, or about 21.5 GB per GPU, which is another 6.4 ms.
- The bandwidth floor is therefore about 11.6 ms. Add the all-reduce traffic of tensor parallelism, kernel launch overhead and sampling, and a realistic step is 20-30 ms.
- At batch 128 and 20-30 ms per step, the replica produces 128 ÷ 0.025 s ≈ 4-6k output tok/s, or roughly 33-50 tok/s per user.

Note the gap between capacity (about 300 sequences fit) and the batch you run (128). Larger batches raise throughput but also raise step time, and the TPOT SLO caps the batch long before memory does.

### Prefill throughput

- Prefill costs about 2 FLOPs per parameter per token: 2 × 70B = 140 GFLOP per token (ignoring attention FLOPs, which are small at 4K context).
- An H100's dense FP8 peak is about 1,979 TFLOPS (NVIDIA lists 3,958 with sparsity). At about 40% MFU (model FLOPs utilization) that is roughly 800 TFLOPS per GPU, or 3.2 PFLOPS per 4-GPU replica.
- 3.2 PFLOPS ÷ 140 GFLOP ≈ 23k prefill tok/s per replica (call it 20-25k).
- A 4K-token prompt costs 4,096 × 140 GFLOP ÷ 3.2 PFLOPS ≈ 180 ms of compute before any queueing.

### The same model on Blackwell

On B200 (180 GB HBM3e and about 8 TB/s per GPU, from NVIDIA's DGX B200 figures of 1,440 GB and 64 TB/s across 8 GPUs), the same model fits on TP=2:

- Memory: 2 × 180 GB × 90% = 324 GB; minus 70 GB weights and about 15 GB workspace leaves about 240 GB, or about 1.46M FP8 KV tokens (about 350 sequences at 4K).
- Decode floor at batch 128: 35 GB of weights per GPU ÷ 8 TB/s ≈ 4.4 ms, plus 43 GB of KV per GPU ÷ 8 TB/s ≈ 5.4 ms, so about 9.8 ms.
- Prefill: about 4.5 PFLOPS dense FP8 per GPU (half of the DGX B200's 72 PFLOPS sparse figure, divided by 8). At 40% MFU, two GPUs give about 3.6 PFLOPS, or about 26k prefill tok/s.

So two B200s do roughly the work of four H100s for this model. Blackwell also adds native FP4 (NVFP4), which halves weight bytes again. Compare hardware on cost per million tokens at your SLO, not on GPU count.

### MoE models change the picture

For an MoE model, each token only activates a fraction of the parameters (DeepSeek-V4-Pro: 1.6T total, 49B active; DeepSeek-V4-Flash: 284B total, 13B active; Kimi K2: about 1T total, 32B active; gpt-oss-120b: 117B total, 5.1B active). Per-token compute follows the active count. But the full weights still have to live in HBM, and at large batch nearly every expert is touched on every step, so per-step weight traffic approaches the full model size spread across the expert-parallel group. A 1.6T-parameter model in mixed FP4/FP8 is on the order of 1 TB of weights, which is why such models are served across multi-node groups or rack-scale NVLink domains like GB200 NVL72 (72 GPUs in one NVLink domain, 13.4 TB of HBM3e).

## Batching and scheduling

![A request flows from clients to a cache-aware router, which picks one of several replicas. Inside a replica, admission control and priority queues feed a continuous-batching scheduler that splits each step's token budget between prefill chunks (compute-bound) and a decode batch (bandwidth-bound). Both run on four GPUs holding sharded FP8 weights, which read and write a paged KV cache with 16-token blocks and prefix reuse. Idle KV can be offloaded to host DRAM or SSD.](/diagrams/q21-serving-replica.svg "One serving replica: the scheduler mixes prefill chunks and decode steps over a shared paged KV cache.")

The router decides which replica holds the useful cache; the scheduler inside the replica decides how each GPU step is spent.

### Continuous batching

**Static batching** wastes slots: the whole batch waits for its longest sequence, and new requests wait for the whole batch. **Continuous (iteration-level) batching**, introduced by Orca, admits and evicts sequences at every decode step. It typically gives 3-10x more throughput than static batching at the same latency.

### Chunked prefill

A long prompt processed in one shot blocks every decoding user for its whole duration. A 32K-token prompt on the replica above costs 32,768 × 140 GFLOP ÷ 3.2 PFLOPS ≈ 1.4 s, and every other user sees a 1.4 s gap in their stream, which shows up as p99 TPOT spikes. **Chunked prefill** (Sarathi-Serve) splits long prompts into chunks of about 512-2,048 tokens and mixes them into decode steps, so prefill progresses without stalling decode.

### Scheduler policy

- Set a max-tokens-per-step budget that bounds step time, and therefore TPOT.
- Use priority classes: interactive over batch, paid over free.
- When KV runs out, preempt by recompute (drop the KV and redo prefill later) or swap (copy KV to host memory).
- Admission control rejects or queues new requests early instead of admitting them and then thrashing on preemption.

## KV cache management

- **PagedAttention** stores KV in fixed-size blocks (16 tokens in vLLM) addressed through per-sequence block tables, like virtual memory pages. The vLLM paper reports that existing systems wasted 60-80% of KV memory to fragmentation and over-reservation, versus under 4% with paging. Copy-on-write blocks also make beam search and parallel sampling cheap.
- **Prefix caching** (vLLM automatic prefix caching, SGLang RadixAttention) reuses KV for shared system prompts and multi-turn history. With a 2K-token system prompt on every request, it can cut prefill compute by 50% or more. It only pays off if the router is cache-aware: requests that share a prefix must land on the replica that holds it. Use consistent hashing on a prefix hash with bounded load, so a hot prefix cannot overload one replica.
- **KV quantization** to FP8 halves KV bytes and so roughly doubles concurrency, at a small quality cost. Validate it on long-context evals, because errors accumulate over long sequences.
- **Offload tiers.** Keep active KV in HBM, spill idle multi-turn sessions to host DRAM, then to SSD (the design Mooncake uses for Kimi). Restoring a session's KV over PCIe is far cheaper than recomputing its prefill.

## Quantization

- **Weight-only 4-bit** (AWQ, GPTQ) mainly helps decode, which is bandwidth-bound: about 2-3x faster at small batch, because each step reads a quarter of the BF16 bytes. Gains shrink at large batch, where decode becomes compute-bound and dequantization overhead shows up.
- **FP8 W8A8** (8-bit weights and activations) on Hopper and Blackwell speeds up both prefill and decode, and is usually within noise on quality for 70B-class and larger models. It is the sensible production default.
- **FP4 on Blackwell.** NVFP4 is a 4-bit floating-point format with FP8 scale factors per small block of values; MXFP4 is the OCP microscaling variant. Both run natively on Blackwell tensor cores, so unlike INT4 weight-only they speed up compute as well as bandwidth. Some open models now ship in FP4: gpt-oss-120b's MoE weights are post-trained in MXFP4 and fit on a single 80 GB GPU, and DeepSeek-V4 ships FP4 experts with FP8 elsewhere.
- **Validation gate.** Run the full eval suite, not only perplexity: long-context retrieval, code, tool-call JSON validity and multilingual. Small models (7-8B) and low-resource languages degrade first.

## Parallelism and topology

- Keep tensor parallelism (TP) inside an NVLink domain: TP=2, 4 or 8 inside a node, or wider inside a rack-scale domain like GB200 NVL72. TP does an all-reduce every layer, which is too chatty for the network between nodes.
- Use pipeline parallelism (PP) or independent replicas across nodes. For large MoE models, add **expert parallelism (EP)**, which places different experts on different GPUs and routes tokens to them with all-to-all communication.
- **Speculative decoding** (a small draft model, or Medusa or EAGLE heads that propose several tokens per step for the big model to verify in one pass) gives 1.5-3x lower latency at low batch. The gain disappears at high batch, where decode is no longer bandwidth-starved and the rejected draft tokens are wasted compute. Enable it adaptively based on load.

### Prefill/decode disaggregation

![A router sends requests to a prefill pool sized for input tokens per second. The prefill pool ships the finished KV blocks over RDMA to a separate decode pool sized for output tokens per second, which streams tokens back to the client.](/diagrams/q21-disaggregated.svg "Disaggregated serving: prefill and decode run on separate pools connected by an RDMA KV transfer.")

Disaggregation removes prefill-decode interference at the cost of moving KV between pools.

At large scale, run prefill and decode on separate pools (DistServe, Splitwise) and ship the KV cache from one to the other over RDMA. Each pool can then be sized and tuned independently, and long prefills never stall decoding users. Open-source orchestration layers package this pattern: NVIDIA Dynamo runs vLLM, SGLang or TensorRT-LLM as backends, and llm-d sits above vLLM and SGLang. It is worth the extra failure modes once p99 TPOT is dominated by prefill interference, or when the input-to-output ratio is very skewed (long RAG prompts, short answers).

## Fleet sizing: a worked example

Assume a peak of 1,000 requests/s, with 1,500 input and 300 output tokens per request, on the 4x H100 replica above.

1. **Decode:** 1,000 × 300 = 300k output tok/s. At about 5k tok/s per replica, that needs 60 replicas.
2. **Prefill:** 1,000 × 1,500 = 1.5M input tok/s. With a 50% prefix-cache hit rate, 750k tok/s must be computed. At about 22k tok/s per replica, that is about 34 replicas' worth of compute.
3. **Combine:** with chunked prefill, prefill work fills compute that bandwidth-bound decode steps leave idle, so the two do not simply add. Budget 80-90 replicas (60 + 34 = 94 is the no-overlap upper bound).
4. **Headroom:** add 30% for failures, deploys and forecast error: about 110 replicas × 4 GPUs ≈ 450 H100s.

### Cost per token

At an assumed $2.50 per GPU-hour (rental prices vary widely by provider and commitment):

```text
450 GPUs × $2.50/GPU-hr × 24 h = $27,000/day
```

How many output tokens does that buy? The fleet is sized for peak, so it depends on how busy the day is:

```text
At peak all day:      300,000 tok/s × 86,400 s = 25.9B output tokens/day
                      $27,000 ÷ 25.9M (thousands of tokens) ≈ $0.0010 per 1K output tokens
At 1/3 of peak avg:   8.6B output tokens/day
(3x diurnal swing)    $27,000 ÷ 8.6M ≈ $0.0031 per 1K output tokens
```

So the all-in cost is about $0.001-0.003 per 1K output tokens ($1-3 per million), and that price also covers the 5x larger volume of input tokens. Utilization over the day therefore matters as much as per-replica efficiency, which is why off-peak batch work and autoscaling pay for themselves.

The same throughput on B200 would need roughly half as many GPUs (about 110 replicas of 2), at a higher hourly price per GPU.

## How to measure it

- Goodput at the SLO: requests per second that meet both the TTFT and the TPOT target. Never quote peak tok/s without the latency attached.
- Latency distributions for TTFT and TPOT, split by input-length bucket, so long prompts do not hide inside an average.
- Engine health: KV-cache utilization, preemption rate, prefix-cache hit rate, queue depth, MFU and memory-bandwidth utilization. GPU "utilization %" from `nvidia-smi` only says a kernel was running, not how much of the chip it used, so do not use it as a capacity metric.
- Cost per million tokens, input and output, per model and per tenant.
- Quantization quality: the full eval suite (long context, code, tool-call JSON validity, multilingual) for every new precision or calibration.
- Load tests that replay production traces, not uniform synthetic prompts. Fixed-length prompts hide tail latency and prefix-sharing effects.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| BF16, TP=8 | Maximum quality, simple | 2x memory and cost vs FP8; less room for KV | Quality-critical evals, small fleets |
| FP8 W8A8 + FP8 KV | About 2x capacity, faster prefill and decode, near-lossless at 70B+ | Needs Hopper or newer; calibration; long-context drift | Default for production on H100, H200, B200 |
| FP4 (NVFP4/MXFP4) | Another ~2x fewer weight bytes, native compute on Blackwell | Needs Blackwell for speedups; larger quality risk; model support varies | Blackwell fleets, very large MoE models, cost-driven deployments |
| INT4 weight-only (AWQ/GPTQ) | Fits 70B on 1-2 GPUs; fast low-batch decode | Quality loss on small models; weaker at high batch | Cost-sensitive, latency-bound, smaller deployments |
| Colocated prefill + decode with chunked prefill | One pool, simpler ops, good utilization | Residual interference in p99 TPOT | Most deployments up to hundreds of GPUs |
| Disaggregated prefill/decode | Independent scaling; best p99 TPOT | RDMA KV transfer complexity; more failure modes | Very large fleets, long prompts, strict TPOT |
| Speculative decoding | 1.5-3x lower latency at low load | Wasted compute at high batch; draft model to maintain | Latency-sensitive, low-to-medium concurrency |

- Throughput vs latency is a single dial: max batch size. Choose it from the SLO and maximize goodput under that constraint.
- Cache-aware routing fights even load balancing. Prefix-hit locality wants requests on the same replica; load spreading wants them anywhere. Bounded-load consistent hashing is the compromise.
- Build vs buy the engine: use vLLM or SGLang and contribute upstream. Custom kernels only pay off for the top two or three hot paths.

## Common mistakes

- **Ignoring that decode is memory-bound.** Adding FLOPs, or reading `nvidia-smi` utilization as capacity, leads to fleets that look busy and still miss SLOs.
- **Getting the KV-cache math wrong.** KV grows with context length × concurrency, and a few long-context users can starve everyone else. Without per-tenant token budgets, one tenant's 128K prompts evict everyone's cache.
- **Treating quantization as free.** No eval gate, only perplexity checked, and no checks on tool calling, JSON validity or long context means quality regressions ship silently.
- **Missing head-of-line blocking from long prefills, and preemption storms when KV runs out.** Both show up as p99 collapse under load, not in averages.
- **Load-testing with fixed-length synthetic prompts.** This hides real tails and prefix-sharing effects, so capacity plans come out wrong in both directions.
- **Forgetting cold start.** Loading 70-140 GB of weights takes minutes. Without pre-warmed pools, streamed safetensors loading, and autoscaling on queue depth and KV utilization (not CPU), a traffic spike becomes an outage.

## Further reading

- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180): the paged KV cache behind vLLM and most modern engines.
- [Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level (continuous) batching.
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (Agrawal et al., 2024)](https://arxiv.org/abs/2403.02310): chunked prefill and stall-free scheduling.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (Zhong et al., 2024)](https://arxiv.org/abs/2401.09670): disaggregation and the goodput metric.
- [AWQ: Activation-aware Weight Quantization (Lin et al., 2023)](https://arxiv.org/abs/2306.00978) and [GPTQ (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323): the two standard 4-bit weight-only methods.
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference (NVIDIA, 2025)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/): the Blackwell FP4 format and its block scaling.
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192): the original speculative decoding paper.
- [SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2023)](https://arxiv.org/abs/2312.07104): RadixAttention, a radix-tree prefix cache.
- [Efficiently Scaling Transformer Inference (Pope et al., 2022)](https://arxiv.org/abs/2211.05102): partitioning strategies and latency/MFU tradeoffs for inference.
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (Qin et al., 2024)](https://arxiv.org/abs/2407.00079): KV offload tiers and disaggregation in production at Kimi.
- [DeepSeek-V4 model card (DeepSeek, 2026)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro): parameter counts, FP4/FP8 weights and the KV-cache reduction from compressed attention.
- [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) and [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/): memory, bandwidth and NVLink-domain specs used in the Blackwell arithmetic.
