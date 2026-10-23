---
title: "Case Study: DeepSeek's Inference Stack and DeepInfer"
description: "How DeepSeek serves its models so cheaply: MLA, FP8 kernels, wide expert parallelism, MTP, disaggregation and disk caching, and what DeepInfer releases."
slug: "deepseek-inference-deepinfer"
category: "inference-optimization"
tags: ["serving", "gpu", "cost", "caching", "latency"]
series: "llm-inference"
seriesOrder: 14
pubDate: 2026-10-23
---

## TL;DR

DeepSeek runs one of the cheapest frontier-scale APIs, and it has published more about how than almost anyone else. Its stack chains the techniques of this series together: a model designed for a small KV cache, 8-bit and 4-bit arithmetic with its own kernels, experts spread across dozens of GPUs, extra tokens per step from a built-in drafter, prefill and decode on separate machines, and a KV cache kept on SSDs for days. Each one removes a bottleneck that would otherwise cap the others. In October 2026 DeepSeek open-sourced DeepInfer, the engine at the center of that stack. This part walks through the chain with DeepSeek's own numbers, then looks at what DeepInfer contains, what it leaves out, and how to try it.

## Why it matters

In February 2025, DeepSeek [published a day of its books](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md). Serving its V3 and R1 models for 24 hours took an average of 226.75 nodes of 8 H800 GPUs. At an assumed rental price of $2 per GPU-hour, that cost $87,072. Had every token been billed at R1's prices, the day would have brought in $562,027, which DeepSeek called "a cost profit margin of 545%."

DeepSeek was careful to say its real revenue was "substantially lower": V3 cost less than R1, the web and app were free, and night-time discounts applied. So 545% is an upper bound. The cost figure, on the other hand, is measured. Spread over 43,536 GPU-hours, that theoretical revenue comes to about $12.90 per GPU-hour on a GPU that rents for $2.

Twenty months later, with the [DeepInfer release](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md), DeepSeek reported that "a single cluster currently serves 155 trillion tokens per day, with overall revenue of US$38 per GPU-hour (at valley pricing)." That's about 1.8 billion tokens every second from one cluster.

DeepSeek got there with a chain of the techniques from this series, each with its own published numbers. [Part 1](/blog/how-llm-inference-works/) listed this final part as "all of the above, at production scale."

## The intuition

Picture a parcel sorting hub. Every parcel (a token) must visit 8 of 256 specialist desks (experts) and then leave. To run the hub cheaply, you shrink the parcels so more fit on each conveyor. You give each desk its own room so it handles a big pile at once instead of a trickle. You keep the conveyors moving while desks work, add spare desks for the busiest destinations, and keep the parcels that come back every day pre-packed in a warehouse instead of packing them again.

That's most of DeepSeek's stack. MLA and low-precision formats shrink the parcels. Wide expert parallelism gives each expert a big pile. Dual-batch overlap keeps the conveyors moving. Redundant experts are the spare desks, and the disk cache is the warehouse.

The analogy breaks in one important place. A sorting hub can't redesign its parcels. DeepSeek can, and many of its biggest wins came from changing the model itself so that serving it got cheaper.

## Step 1: shrink what every token carries

### A small KV cache by design

[Part 2](/blog/kv-cache/) showed that multi-head latent attention (MLA) stores one compressed latent of 576 numbers per token per layer instead of full per-head keys and values. For DeepSeek-V3 that's 70,272 bytes per token in 16-bit precision. DeepSeek's [ISCA 2025 paper](https://arxiv.org/abs/2505.09343) puts it next to Qwen-2.5 72B at 327,680 bytes (4.66 times more) and Llama 3.1 405B at 516,096 bytes (7.3 times more).

Later models shrink it much further. [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), the model DeepInfer serves, combines layers that share keys and values, sparse attention (covered in [Part 11](/blog/long-context-inference/)) and an FP4 main KV cache to reach "890 bytes per token." Its model card calls that a 437-fold reduction from DeepSeek's first models. The arithmetic checks out against [DeepSeek LLM 67B's config](https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/blob/main/config.json), with its 95 layers and 8 key-value heads of 128 numbers:

```python
ctx = 131_072                                # a 128K-token conversation
per_token = {
    "DeepSeek LLM 67B (GQA, 2023)": 2 * 95 * 8 * 128 * 2,   # 389,120 bytes
    "DeepSeek-V3 (MLA, 2024)":      70_272,
    "DeepSeek-V4.1-Flash (2026)":   890,
}
for name, b in per_token.items():
    gb = b * ctx / 1e9
    print(f"{name}: {b:,} B/token, {gb:.2f} GB per conversation, "
          f"{gb / 40 * 1000:,.0f} ms to read at 40 GB/s")
# 67B:   389,120 B/token, 51.00 GB, 1,275 ms
# V3:     70,272 B/token,  9.21 GB,   230 ms
# V4.1F:     890 B/token,  0.12 GB,     3 ms
```

The 40 GB/s is the read speed DeepSeek recommends per KV cache storage server. At 890 bytes per token, a whole 128K-token conversation loads from SSD in less time than one decode step takes. That's what makes Step 3's disk cache work.

### FP8 and FP4 arithmetic, with custom kernels

DeepSeek's production V3 and R1 ran "with precision consistent with training": matrix multiplications and expert dispatch in FP8, the core attention math and the combine step in BF16. [Part 8](/blog/quantization/) explains the formats. On Hopper GPUs ([Part 3](/blog/gpus-for-llms/)), FP8 doubles tensor-core throughput over BF16 and halves the bytes read per weight.

DeepSeek's FP8 recipe uses fine-grained scales, one for each 1×128 tile of activations and each 128×128 block of weights, and it wrote its own kernels to run it fast. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) launched in February 2025 as an FP8 matrix-multiplication library with its core kernel in "~300 lines of code" and "up to 1350+ FP8 TFLOPS on Hopper GPUs"; an April 2025 update reached "up to 1550 TFLOPS on H800." It now also covers FP4 weights and "Mega MoE," one kernel that fuses expert dispatch, both expert matrix multiplications, the activation and combine.

V4.1-Flash goes further down. The DeepInfer Q&A lists FP4 for the main KV cache, the sparse-attention indexer keys and the routed-expert weights.

### Attention kernels: FlashMLA

[Part 7](/blog/flashattention/) showed why attention lives or dies by memory traffic. MLA needs its own kernels, because the latent is shared by all heads. [FlashMLA](https://github.com/deepseek-ai/FlashMLA), released on the first day of DeepSeek's February 2025 open-source week, used a paged KV cache with blocks of 64 tokens ([Part 5](/blog/pagedattention-vllm/)) and reached "3000 GB/s memory-bound" and "BF16 580 TFLOPS compute-bound on H800"; a later version reported 660 TFLOPS. Today's FlashMLA serves V4.1's sparse attention on Blackwell (B200) GPUs and Huawei Ascend 950 chips, and in September 2026 it dropped Hopper support entirely.

## Step 2: spread the experts wide

### Why wide

DeepSeek-V3 has 256 routed experts per layer and each token uses 8 ([Mixture of Experts](/blog/mixture-of-experts/) explains the design). The February 2025 overview gives two reasons for large-scale expert parallelism (EP) across nodes: it "significantly scales the batch size," which keeps expert matrix multiplications efficient, and each GPU holds only a few experts, which cuts memory reads and latency.

The production layouts:

| Phase | GPUs per unit | Routed experts per GPU | Shared expert per GPU | Redundant experts |
|---|---|---|---|---|
| Prefill | 32 (4 nodes), EP32 | 9 | 1 | 32 |
| Decode | 144 (18 nodes), EP144 | 2 | 1 | 32 |

Attention runs data-parallel (DP32 and DP144): each GPU keeps full attention weights and its own requests' cache, as [Part 10](/blog/model-parallelism/) explained for MLA models.

A memory budget shows why decode uses 144 GPUs rather than 8. Each V3 expert has 44 million parameters (three 7,168 × 2,048 matrices), across 58 MoE layers. Subtracting all the experts from 671 billion leaves about 14.5 billion parameters of attention, dense layers and embeddings, which every GPU keeps:

```python
expert = 3 * 7168 * 2048                 # 44.0 million parameters
moe_layers = 58
non_expert = 14.5e9                       # replicated on every GPU (data-parallel attention)

def gb_per_gpu(routed_per_gpu):          # FP8: 1 byte per parameter
    return (non_expert + (routed_per_gpu + 1) * moe_layers * expert) / 1e9   # +1 shared expert

print(gb_per_gpu(32))    # EP8:   98.8 GB, more than an 80 GB H800 holds
print(gb_per_gpu(2))     # EP144: 22.2 GB, leaving about 58 GB for KV cache

free = 80e9 - gb_per_gpu(2) * 1e9
print(free / 70_272 / 4_989)    # about 165 requests per GPU, 16-bit cache, 4,989 tokens each

# Tokens each expert copy sees per decode step, at 128 requests per GPU:
print(8 * 128 * 8 / 256)         # one 8-GPU node:   32 tokens per expert
print(144 * 128 * 8 / 288)       # EP144, 288 copies: 512 tokens per expert
```

On one node, V3 doesn't even fit. Spread over 144 GPUs, each holds 22 GB of weights and has room for a large batch, and each expert copy sees 16 times more tokens per step. The 4,989 tokens is the average cache length per output token from DeepSeek's statistics, and the 128 requests per GPU is the batch in DeepSeek's [published decode profile](https://github.com/deepseek-ai/profile-data).

### Moving tokens: DeepEP

Every MoE layer sends each token to its experts' GPUs (dispatch) and back (combine). [DeepEP](https://github.com/deepseek-ai/DeepEP) has two kinds of kernels for this. Its [original benchmarks](https://github.com/deepseek-ai/DeepEP/tree/82dcf48fd315d7b83f2cd2b4f1d1f1fda6af8ed2) on H800s, each with a 400 Gb/s InfiniBand card, show both:

| Kernel | Setting | Result |
|---|---|---|
| Normal (prefill), within a node | EP8, 4,096 tokens | 153 GB/s dispatch, 158 GB/s combine over NVLink |
| Normal (prefill), across nodes | EP32 | 44 GB/s dispatch, 47 GB/s combine over RDMA |
| Low-latency (decode), pure RDMA | EP128, 128 tokens | 192 µs dispatch, 369 µs combine |

The normal kernels move big batches through NVLink and the network at close to the link speed. The low-latency kernels send small decode batches straight over RDMA (remote direct memory access, where one machine's network card writes into another's memory), trading bandwidth for a fixed, short delay.

Even short delays add up. At EP128, dispatch plus combine take 561 µs per MoE layer, so 58 layers spend 32.5 ms per decode step just waiting on the network. Part 10 estimated about 25 ms from first principles; DeepEP's measured numbers land in the same place. At DeepSeek's reported 20 to 22 tokens per second per user, with MTP (Step 3) adding almost a token per step, a step lasts roughly 85 to 95 ms. Left exposed, the network wait would take up about a third of it.

### Hiding it: dual-batch overlap

DeepSeek's answer is **dual-batch overlap**: split each batch into two micro-batches and run one's computation while the other's tokens are on the wire. In prefill, the two simply alternate. In decode, attention takes longer than the rest, so DeepSeek splits attention into two steps and runs "a 5-stage pipeline" to keep the GPU and the network busy at once. The decode profile shows a detail: once the RDMA messages are issued, "all GPU SMs are freed," so communication doesn't steal compute.

The ISCA paper works out the limit. With 32 tokens per GPU and each token sent to 9 experts, one dispatch and combine over a 50 GB/s link takes `(1 + 2 bytes) × 32 × 9 × 7,000 ÷ 50 GB/s = 120.96 µs`. Two micro-batches per layer over 61 layers gives 14.76 ms per output token, about 67 tokens per second, even if compute were free. On an NVLink rack such as NVIDIA's GB200 NVL72, the same estimate drops to 0.82 ms. On H800s, the network sets V3's speed limit.

### Keeping the load even: EPLB

Some experts are always busier than others, and every GPU waits for the slowest one at each layer. DeepSeek's fix is **redundant experts**: extra copies of the hottest experts, placed by its Expert Parallelism Load Balancer, [EPLB](https://github.com/deepseek-ai/EPLB). The open-source algorithm has two policies: a hierarchical one for prefill, which first balances expert groups across nodes to limit cross-node traffic, and a global one for decode's wider layout. The [V3 report](https://arxiv.org/abs/2412.19437) re-chose redundant experts from live statistics "every 10 minutes."

DeepInfer reports how close V4.1-Flash gets, as `moe_lb`, the slowest GPU's expert load divided by the mean. Prefill starts from 4 redundant experts per GPU and re-assigns every 1,000 steps, reaching 1.074 at EP4. Decode starts from 1 per GPU and re-assigns every 10,000 steps, reaching 1.090; at EP64 that rounds up to 7 experts per GPU (384 + 64 = 448 copies). In other words, the busiest GPU does about 9% more expert work than average.

## Step 3: more tokens per step, fewer prefills

### Multi-token prediction

[Part 9](/blog/speculative-decoding/) covered how a drafter can propose tokens that the model checks in one pass. DeepSeek-V3 trained its own: one multi-token prediction (MTP) module whose guess for the token after next was accepted 85% to 90% of the time, for "1.8 times TPS" (tokens per second). The ISCA paper adds a serving benefit: more tokens per step means bigger batches for the experts.

V4.1-Flash replaces it with **DSpark**, which drafts 5 tokens at a time and adds a confidence head that predicts how likely each drafted token is to be accepted. DeepInfer then verifies only the most promising prefix of each draft. At DeepSeek's production operating point of about 60 sequences per GPU, this cut the tokens verified per step from 5.00 to 3.58 while the average accepted length barely moved, from 3.30 to 3.21, for an estimated 5.4% more throughput.

### Prefill and decode on different machines

[Part 12](/blog/disaggregated-serving-llm-d/) used DeepSeek's daily statistics to show how prefill and decode nodes are sized and why the split pays off. V4.1-Flash's system keeps the idea and adds a third role, a separate vision encoder for images, so DeepSeek calls it E/P/D. The parallelism is very different from V3's:

| Phase | V3/R1 on H800 (2025) | V4.1-Flash (2026) |
|---|---|---|
| Prefill | EP32 over 4 nodes | EP2 to EP8: "the smallest EP/DP that HBM allows" |
| Decode | EP144 over 18 nodes | EP64 to EP128 on 64- to 128-GPU "supernodes" |
| Smallest efficient deployment | Not stated | 64 prefill GPUs plus 64 decode GPUs |

Decode stays as wide as the hardware allows, because wide EP "substantially reduces per-step memory traffic in the MoE expert part." What changed is the fabric: expert parallelism now runs "within a (super)node," a scale-up domain where dozens of GPUs share one fast interconnect, so expert traffic no longer crosses InfiniBand, the bottleneck the ISCA paper calculated.

![A left-to-right flow of DeepSeek's V4.1-Flash serving system. An API request goes to DeepSeek Recipe, which turns text into tokens, then to the Entrypoint, which matches the prompt against the prefix cache. The request passes to a prefill load balancer, marked not released, and on to the prefill engine, marked DeepInfer, running EP2 to EP8 with FlashMLA and DeepGEMM. A KV cache store on 3FS SSDs, marked not released, supplies cached prefixes to the prefill engine. The prefill engine sends the request to a decode load balancer, marked not released, and sends the KV cache directly to the decode engine, marked DeepInfer, running EP64 to EP128 with DeepEP and DSpark, which returns tokens to the user.](/diagrams/i14-deepseek-serving-stack.svg "DeepSeek's V4.1-Flash serving path. DeepInfer is the two engines; the load balancers and the KV cache store are not in the release.")

The open-source release covers the engines in the middle, and the parts that tie a fleet together stay in-house.

### A KV cache that lasts for days

[Part 6](/blog/prefix-caching/) covered DeepSeek's context caching on disk, which its API has offered since 2024; Step 1 showed why MLA's small cache makes cheap disks fast enough. In the February 2025 statistics, 342 billion of 608 billion input tokens, 56.3%, were served from that cache.

The October 2026 system aims to "retain KV cache for more than 72 hours." The cache lives on a dedicated cluster running [3FS](https://github.com/deepseek-ai/3FS), DeepSeek's distributed file system, with about 100 PiB of SSD. Entries expire after 3 days for agent traffic and 7 days for chatbot traffic, since a few chatbot prefixes are read very many times. Data is stored with 10+2 erasure coding, so any 10 of 12 pieces rebuild it, and a group survives two failed storage nodes. Because any prefill machine can read any prefix from shared storage, the prefill load balancer needs no cache-aware routing at all.

Customers see the result in the price. On DeepSeek's [pricing page](https://api-docs.deepseek.com/quick_start/pricing) for V4.1-Flash, a million input tokens cost $0.30 on a cache miss and $0.006 on a hit at peak hours: a cached token costs 2% as much.

## Step 4: DeepInfer, the engine DeepSeek open-sourced

### Why it took until 2026

In April 2025, DeepSeek [explained](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md) why it would not release its engine. It was "based on an early fork of vLLM from over a year ago," tightly tied to internal infrastructure, and the team lacked "bandwidth to maintain a large open-source project." It would contribute components and optimizations to existing projects instead.

In October 2026 it released [DeepInfer](https://github.com/deepseek-ai/DeepInfer) anyway, under the Apache 2.0 license, as "the high-performance inference framework that powers DeepSeek's large-scale production services." Its README says it "was originally built primarily on vLLM, with additional inspiration from SGLang and TensorRT-LLM." The code shows that heritage: core files carry the header "Derived from vLLM; modified by DeepSeek," but DeepInfer doesn't install or import vLLM. It's a standalone descendant of vLLM that you build and run on its own.

### What's in it

- It serves one model, DeepSeek-V4.1-Flash, with 552 billion backbone parameters, 8 billion active per token in prefill and 16 billion in decode, and a 1-million-token context.
- It runs on NVIDIA SM100 GPUs (the Blackwell B200 generation) and Huawei Ascend 950 NPUs. Hopper GPUs such as the H800 and H100 are not supported.
- Its kernels come from DeepSeek's own libraries (FlashMLA, DeepGEMM, DeepEP, TileKernels, DeepSelect), plus custom kernels written in the TileLang language.
- It splits work with data-parallel attention, balanced context parallelism, and MoE expert parallelism with EPLB.
- DSpark speculative decoding, asynchronous scheduling that plans step N+1 on the CPU while the GPU runs step N (the scheduler loop from [Part 4](/blog/continuous-batching/)), batch-invariant inference, deterministic sampling and structured outputs.
- "SWA bounded replay": on a cache hit, the sliding-window layers recompute only the last 128 tokens, a cost "independent of context length and layer count," as Part 11 described.

### What's not

The README lists three things "required for a complete inference service" that the release leaves out: "a KV cache system," "VE/Prefill/Decode disaggregation (most required engine functionality is already implemented)," and "load balancing across data-parallel replicas for both prefill and decode." DeepSeek says its KV cache management and load balancing are "tightly coupled to our internal infrastructure." The release also "provides offline inference": you run batches from Python, with no HTTP server. For prompt formatting, DeepSeek released [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) alongside it, a Rust library that converts API requests to and from DeepSeek's token format.

DeepInfer, then, is the engine of the production system without the services around it. Most of what Steps 1 to 3 described runs inside it. The caching tier, the hand-off between prefill and decode machines, and the routing above the engines don't ship.

### Trying it

You need Linux, Python 3.12, GCC 13 or later, and CUDA 13 (or CANN 9.2 on Ascend). The default example runs expert parallelism across 8 accelerators:

```bash
export DEEPINFER_ACCELERATOR=nvidia
./build.sh
python examples/offline_inference.py --ep 8
```

The first run downloads `deepseek-ai/DeepSeek-V4.1-Flash-1017` from Hugging Face. The repository's benchmark charts, all with a 131,072-token context, show what the engine does on 4 SM100 GPUs for prefill and 64 for decode, "each within one scale-up domain." Read off the charts: prefill reaches about 87,000 uncached input tokens per second per GPU at 32,768-token batches, and decode about 15,700 output tokens per second per GPU at 64 requests per GPU, with each user getting about 246 tokens per second. The decode figures assume 3 draft tokens accepted at 75% each, so they are projections from an assumed acceptance rate.

## In the real world

- SGLang reproduced DeepSeek's design on [96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/) in May 2025, with prefill/decode disaggregation and large-scale EP over DeepEP. It reached 52,300 input and 22,300 output tokens per second per node for 2,000-token inputs, "the first open-source implementation to nearly match the throughput reported in the official DeepSeek blog," at an estimated $0.20 per million output tokens.
- The vLLM team [reported](https://vllm.ai/blog/2025-12-17-large-scale-serving) about 2,200 tokens per second per H200 for DeepSeek models with wide EP, dual-batch overlap and EPLB, and llm-d packages the same pattern for Kubernetes, as Parts 10 and 12 described.
- V4.1 itself runs in both open engines. SGLang tracks its DeepSeek-V4.1 work in a [public roadmap](https://github.com/sgl-project/sglang/issues/42170) that plans to adopt the latest DeepGEMM and DeepEP releases, and vLLM has merged DSpark work such as [adaptive verification on AMD GPUs](https://github.com/vllm-project/vllm/pull/52362). DeepSeek's April 2025 plan to contribute components shows up in both projects.
- DeepEP's README now lists community ports to AMD GPUs and other network cards, and DeepGEMM, FlashMLA and DeepEP ship Ascend versions, in line with DeepSeek's stated goal to "support accelerators from multiple vendors."

## Common misconceptions

- **"DeepSeek makes a 545% profit on inference."** That figure assumed every token was billed at R1 prices and counted only GPU rental at $2 per hour. DeepSeek itself said actual revenue was "substantially lower."
- **"DeepInfer is a drop-in replacement for vLLM."** It serves one model, on Blackwell or Ascend 950 hardware, as offline inference. It has no KV cache system, no disaggregation and no load balancers. vLLM and SGLang remain the way to serve DeepSeek models behind an API.
- **"Wide expert parallelism is about spreading one request across many GPUs."** It's about the experts. On 144 GPUs, each holds 2 or 3 experts per layer instead of 32, which frees memory for a large batch and gives every expert hundreds of tokens per step instead of dozens.
- **"DeepSeek's low prices come from clever software alone."** The model was shaped for serving: MLA, cross-layer KV sharing and an FP4 cache cut the KV cache per token 437-fold since DeepSeek's first model, MTP and DSpark add tokens per step, and V4.1's encoder-decoder layout activates half as many parameters during prefill as during decode.
- **"You need DeepSeek's code to serve DeepSeek models efficiently."** SGLang nearly matched DeepSeek's published throughput on 96 H100s with open-source software, and vLLM and llm-d run the same wide-EP pattern.

## Key takeaways

- No single trick explains DeepSeek's costs. A small KV cache, low-precision kernels, wide expert parallelism, a built-in drafter, split prefill and decode, and a disk cache each remove the limit the next one would hit.
- The model and the serving system were designed together. MLA, MTP, FP4 caches and sparse attention exist partly to make serving cheaper.
- Wide expert parallelism lets each GPU hold a few experts and a big batch. DeepEP moves the tokens, dual-batch overlap hides the wait, and redundant experts keep the slowest GPU within about 9% of average.
- DeepInfer open-sources DeepSeek's production engine for V4.1-Flash on Blackwell and Ascend. The KV cache tier, disaggregation and load balancing stay in-house.
- The whole series' toolkit appears here: prefill and decode, the KV cache, GPUs and number formats, batching, paging, prefix caching, fused kernels, quantization, speculative decoding, parallelism, long context and disaggregation.

## What's next

That's the end of the Inference series, and the full reading order is on the [series page](/series/llm-inference/). To see these ideas inside complete products, the [LLM System Design series](/series/llm-system-design/) starts from engineering problems, such as [serving open LLMs at scale](/blog/serving-open-llms-at-scale/). And if you want to go back to what the model is doing between all these kernels, the [LLM Fundamentals series](/series/llm-fundamentals/) explains it from the first token up.

## Further reading

- [DeepSeek-V3/R1 Inference System Overview (DeepSeek, 2025)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md): production parallelism, dual-batch overlap, load balancers, and the cost and revenue statistics with their caveats.
- [DeepInfer (DeepSeek, 2026)](https://github.com/deepseek-ai/DeepInfer) and its [deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md): the engine, the V4.1-Flash serving system, EPLB settings, KV cache storage and DSpark measurements.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): FP8 training, prefill and decode deployment units, redundant experts and MTP acceptance rates.
- [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures (Zhao et al., ISCA 2025)](https://arxiv.org/abs/2505.09343): KV cache comparisons and the network-bound speed limit of expert parallelism.
- [DeepSeek-V4.1-Flash model card (DeepSeek-AI, 2026)](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): the encoder-decoder layout, 890-byte KV cache and DSpark.
- [The Path to Open-Sourcing the DeepSeek Inference Engine (DeepSeek, 2025)](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md): why DeepSeek first chose to contribute components instead of its engine.
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [DeepEP](https://github.com/deepseek-ai/DeepEP), [EPLB](https://github.com/deepseek-ai/EPLB), [profile-data](https://github.com/deepseek-ai/profile-data) and [3FS](https://github.com/deepseek-ai/3FS): the open-source kernels, load balancer, overlap traces and storage behind the stack.
- [DeepSeek API: Context Caching](https://api-docs.deepseek.com/guides/kv_cache) and [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing): how the disk cache behaves and what a cache hit costs.
- [Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs (LMSYS, 2025)](https://lmsys.org/blog/2025-05-05-large-scale-ep/): an open-source reproduction of DeepSeek's design in SGLang.
- [vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP (vLLM team, 2025)](https://vllm.ai/blog/2025-12-17-large-scale-serving): the same techniques in vLLM.
