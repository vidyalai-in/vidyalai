---
title: "Tensor, Pipeline and Expert Parallelism: One Model, Many GPUs"
description: "How to split a model too big for one GPU: tensor, pipeline, data and expert parallelism, what each costs in communication, and why NVLink decides the layout."
slug: "model-parallelism"
category: "inference-optimization"
tags: ["gpu", "serving", "latency", "cost"]
series: "llm-inference"
seriesOrder: 10
pubDate: 2026-03-24
---

## TL;DR

When a model doesn't fit in one GPU's memory, or one GPU can't read its weights fast enough, you split it across several. Tensor parallelism slices every weight matrix so all GPUs work on every token together, at the price of two all-reduce exchanges per layer, which only fast links like NVLink can absorb. Pipeline parallelism gives each GPU a range of layers and passes one small vector per token between them, so it works across slow network links, but it raises throughput, not single-user speed. Mixture-of-experts models add expert parallelism, which spreads experts over GPUs and ships each token to the few experts it needs with all-to-all messages. By the end you'll be able to compute what each split sends over the wire per token and pick a layout for a given model and cluster.

## Why it matters

Llama 3.1 405B in BF16, the usual 16-bit format, is 812 GB of weights. An NVIDIA H100 holds 80 GB. Even a whole server of eight H100s, 640 GB, falls short, and Meta's own paper says so: the model "does not fit in the GPU memory of a single machine with 8 Nvidia H100 GPUs."

The KV cache, the per-token memory from [Part 2](/blog/kv-cache/), needs room on top of the weights. Here is what three large open models need just for weights, worked from their configs, in BF16 and in 8-bit FP8 ([Part 8](/blog/quantization/)):

| Model | Parameters | Weights in BF16 | Weights in FP8 | Smallest NVIDIA setup that holds the FP8 weights |
|---|---|---|---|---|
| Llama 3.1 70B | 70.6 billion | 141 GB | 71 GB | 1 H100 (80 GB), with little room left for cache |
| Llama 3.1 405B | 405.9 billion | 812 GB | 406 GB | 8 H100s (640 GB) |
| DeepSeek-V3 | 671 billion | 1,342 GB | 671 GB | 8 H200s (1,128 GB); 8 H100s fall 31 GB short |

Speed is the second reason. [Part 1](/blog/how-llm-inference-works/) showed that decode is limited by how fast a GPU reads its weights. Spread Llama 3.1 70B over 8 H100s and each GPU reads 17.6 GB per step instead of 141 GB, about 5.3 ms. The catch is that the eight GPUs have to talk to each other, and how they talk decides whether the split pays off.

## The intuition

Picture a banquet kitchen with one enormous recipe, too long for any single cook's counter.

You could give every cook a slice of every step: each chops an eighth of the onions, stirs an eighth of the sauce. The work per step shrinks, but after every step the cooks have to huddle and combine their bowls before anyone can start the next one. That's **tensor parallelism**.

You could run an assembly line instead: cook 1 does steps 1 to 40, then slides the plate to cook 2, who does steps 41 to 80. Only the plate moves between them, so they can stand in different rooms. But while cook 2 works on a plate, cook 1 is idle unless there's a second plate to start. That's **pipeline parallelism**.

Or, if the recipe calls for specialists, you could seat 256 of them around the hall and send each dish to the 8 specialists it needs. That's **expert parallelism**, and it only applies to mixture-of-experts models.

The analogy breaks on scale: the tensor-parallel huddles happen 160 times for every token Llama 3.1 70B writes, and each must finish in microseconds. And in a real MoE model a learned router picks the specialists, so some get swamped while others sit idle.

![Three side-by-side panels. Tensor parallelism: GPU 1 holds half of every matrix and GPU 2 the other half, linked by an all-reduce twice per layer. Pipeline parallelism: GPU 1 holds layers 1 to 40 and sends activations of 16 KiB per token to GPU 2, which holds layers 41 to 80. Expert parallelism: a token's home GPU runs attention and the router, dispatches the token to GPU 1 holding experts 1 to 128 and GPU 2 holding experts 129 to 256, and both send their outputs back in a combine step to the home GPU, which adds them.](/diagrams/i10-three-splits.svg "Three ways to split one model across GPUs, shown with two GPUs each, and what each sends between them.")

Tensor parallelism exchanges data inside every layer, pipeline parallelism only at stage boundaries, and expert parallelism only in the MoE layers, to the GPUs holding the chosen experts.

## Step 1: tensor parallelism slices every layer

Tensor parallelism (TP) comes from NVIDIA's [Megatron-LM paper](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019). Its trick is to split matrices so that GPUs can go as long as possible without talking.

Take the MLP, the feed-forward part of each Transformer block. In Megatron's model it's two matrices (Llama's version adds a second input matrix, split the same way). The first matrix is split by columns: each of 8 GPUs gets one eighth of the output columns, computes its eighth of the hidden activations and applies the activation function on its own. The paper explains why this split comes first: the activation function is nonlinear, so splitting the other way would force a "synchronization point before the GeLU function." The second matrix is split by rows, so each GPU multiplies its eighth of the activations by its eighth of the rows. Each GPU now holds a partial sum of the full output. One **all-reduce**, a collective operation that adds up a tensor across all GPUs and gives every GPU the total, finishes the layer.

Attention splits along its heads. Queries, keys and values are computed column-parallel, "such that the matrix multiply corresponding to each attention head is done locally on one GPU." The output projection is row-parallel, and one more all-reduce closes the sub-layer. The paper sums it up: all the matrix multiplies in a layer need "only two all-reduces in the forward path." Inference has no backward pass, so that's the whole bill: two all-reduces per Transformer block.

### What it costs per token

For Llama 3.1 70B at TP = 8, each all-reduce adds up one vector of the model's hidden size, 8,192 numbers, per token:

```text
size of one all-reduce        = 8,192 × 2 bytes (BF16)  = 16 KiB per token
all-reduces per token         = 2 per layer × 80 layers = 160
data reduced per token        = 160 × 16 KiB            = 2.5 MiB
```

The standard way to run an all-reduce is a ring: the GPUs pass slices around a circle, first adding and then distributing. NVIDIA's [nccl-tests documentation](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) gives its best-case time for `S` bytes on `N` GPUs, each with bandwidth `B`:

```text
t = (S / B) × 2(N − 1) / N
```

Each GPU sends almost twice the message size (for 8 GPUs, 2 × 7/8 = 1.75 times), because every slice goes around the ring once to be summed and once to be shared. Here is that formula, times 160, for three batch sizes, using the link speeds from [Part 3](/blog/gpus-for-llms/):

| Tokens in the step | Data each GPU sends per all-reduce | 160 all-reduces over NVLink 4 (450 GB/s) | Over PCIe Gen5 (64 GB/s) | Over a 400 Gb/s network (50 GB/s) |
|---|---|---|---|---|
| 1 (one user decoding) | 0.029 MB | 0.01 ms | 0.07 ms | 0.09 ms |
| 64 (a batched decode step) | 1.8 MB | 0.65 ms | 4.6 ms | 5.9 ms |
| 4,096 (one prompt's prefill) | 117 MB | 42 ms | 294 ms | 376 ms |

Compare those with the work they interrupt. A decode step at TP = 8 reads 17.6 GB of weights per GPU, about 5.3 ms on an H100. A 4,096-token prefill is about 578 trillion floating-point operations, about 73 ms across 8 H100s at their peak of 989 TFLOPS. Over NVLink, communication is a modest tax. Over PCIe, a single prefill spends four times longer talking than computing.

### Small messages are about latency, not bandwidth

The single-user row isn't free, because every all-reduce also pays a fixed cost to launch and synchronize, and a ring of `N` GPUs takes `2(N − 1)` steps, 14 for 8 GPUs. If each all-reduce took 10 microseconds end to end (an illustrative figure; real ones depend on the GPUs, links and library), 160 of them would add 1.6 ms to a 5.3 ms step. NVIDIA's [MultiShot](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/) all-reduce uses NVSwitch to finish in "two communication steps" regardless of GPU count, for up to 3 times lower latency.

NVIDIA's [numbers](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference), modeled from internal measurements, for Llama 3.1 70B at TP = 2 on H200s show both effects. Comparing a full NVSwitch fabric with slower point-to-point links, throughput per GPU was the same at batch size 1 (26 versus 25 tokens per second) and 1.5 times higher at batch size 32 (168 versus 112), because small batches are limited by latency and only big ones by bandwidth.

This is why the rule from Part 3 holds: keep tensor parallelism inside one NVLink domain. Engines such as vLLM move this data with NCCL, the NVIDIA Collective Communications Library, which picks the fastest path it can find between GPUs. vLLM's docs warn that if NCCL falls back to plain TCP sockets across nodes, it "is not efficient for cross-node tensor parallelism."

### The KV cache splits too, until it doesn't

Splitting attention heads also splits the KV cache. Llama 3.1 70B has 64 query heads but only 8 key-value heads (grouped-query attention, from Part 2), and 320 KiB of cache per token. At TP = 8, each GPU gets 8 query heads and exactly 1 KV head, so it stores 40 KiB per token.

At TP = 16 there aren't enough KV heads to go around. vLLM's code handles it directly: when the TP size is at least the number of KV heads, each GPU keeps one KV head and the heads are replicated, here on 2 GPUs each. Each GPU still stores 40 KiB per token, but the group stores 640 KiB, twice the real cache. TP = 16 also needs two 8-GPU servers, which puts all 160 all-reduces on the network. vLLM also requires the TP size to divide the number of attention heads, so 64 heads allow 1, 2, 4, 8, 16, 32 or 64.

## Step 2: pipeline and data parallelism

### Pipeline parallelism splits the layers

Pipeline parallelism (PP), popularized by Google's [GPipe](https://arxiv.org/abs/1811.06965) (Huang et al., 2018), gives each GPU a contiguous range of layers, called a stage. With 2 stages, Llama 3.1 70B puts layers 1 to 40 on one GPU and layers 41 to 80 on the other. The only traffic is the hidden-state vector crossing each boundary, once per token:

```text
70B:  8,192 × 2 bytes  = 16 KiB per token per boundary
405B: 16,384 × 2 bytes = 32 KiB per token per boundary
```

That's small enough for an ordinary network. A full 4,096-token prompt through Llama 3.1 405B moves 134 MB across a boundary, about 2.7 ms at 50 GB/s. Split the same 405B model with tensor parallelism across two servers instead, and every token pays 2 × 126 = 252 all-reduces, each one crossing the network.

The cost is idle time. A token has to finish stage 1 before stage 2 can start on it, so with one batch in flight, only one stage works at a time. GPipe calls the idle time the **bubble** and fills it by splitting a batch into `M` micro-batches that follow each other through `K` stages. The idle fraction is `(K − 1) / (M + K − 1)`:

| Stages K | 1 micro-batch | 4 micro-batches | 8 micro-batches | 32 micro-batches |
|---|---|---|---|---|
| 2 | 50% idle | 20% | 11% | 3% |
| 4 | 75% idle | 43% | 27% | 9% |
| 8 | 88% idle | 64% | 47% | 18% |

GPipe found the bubble "negligible when M ≥ 4 × K" in training. Inference is easier still. The [Llama 3 paper](https://arxiv.org/abs/2407.21783) notes bubbles "are not an issue during inference, since inference does not involve a backward pass that requires a pipeline flush," and Meta used micro-batching to keep stages busy. A serving engine running continuous batching ([Part 4](/blog/continuous-batching/)) can keep several batches in flight, one per stage.

None of that makes a single request faster. Each token still passes through every stage in sequence, and each stage has only its own GPU's bandwidth. Take Llama 3.1 405B in FP8 on 8 H200s (4.8 TB/s each):

```text
weights per GPU                 = 406 GB ÷ 8          = 50.7 GB
one GPU reads its share         = 50.7 GB ÷ 4.8 TB/s  = 10.6 ms
TP = 8: all GPUs read at once   ≈ 10.6 ms per token   → at most 95 tokens/s
PP = 8: stages read in turn     ≈ 8 × 10.6 = 85 ms    → at most 12 tokens/s
```

With tensor parallelism all eight GPUs read their shares at the same moment, so each token gets the bandwidth of the whole group. With pipeline parallelism a token only ever uses one GPU's bandwidth at a time; what you gain is room for the weights and throughput from keeping every stage busy.

### Data parallelism copies the whole thing

The simplest way to use more GPUs is **data parallelism** (DP): run complete, independent copies of the model, called replicas, and send each request to one of them. Llama 3.1 8B (16 GB) fits on one H100, so an 8-GPU server runs 8 replicas with no communication between them during inference. Throughput scales almost linearly; single-user speed doesn't change. vLLM exposes it as `--data-parallel-size`, which combines with TP: `--data-parallel-size 4 --tensor-parallel-size 2` runs 4 replicas of 2 GPUs each. What's left is routing requests well across replicas, which [Part 6](/blog/prefix-caching/) and [Serving Open LLMs at Scale](/blog/serving-open-llms-at-scale/) cover.

## Step 3: expert parallelism for mixture-of-experts models

A **mixture-of-experts** (MoE) model replaces the single MLP in most layers with many smaller MLPs, the experts, plus a router that picks a few of them for each token ([Mixture of Experts](/blog/mixture-of-experts/) explains the design). DeepSeek-V3's [config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) has 61 layers; the first 3 are dense and the other 58 each hold 256 routed experts plus 1 shared expert, and the router picks 8 routed experts per token. The [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) counts 671 billion parameters, of which 37 billion are used for each token.

**Expert parallelism** (EP) puts different experts on different GPUs. With EP = 32, each GPU holds 8 of the 256 routed experts. Each MoE layer then runs in three moves:

1. **Dispatch**: each GPU sends each of its tokens to the GPUs holding that token's chosen experts.
2. **Compute**: each GPU runs its experts on whatever tokens arrived.
3. **Combine**: the results go back to each token's home GPU, which adds them up, weighted by the router.

Dispatch and combine are **all-to-all** operations: every GPU may send something different to every other GPU.

### What it costs per token

DeepSeek sends dispatch data in FP8 (1 byte per number) and keeps combine in BF16 (2 bytes). With a hidden size of 7,168 and 8 experts per token:

```text
dispatch per token per layer = 8 × 7,168 × 1 byte  = 57,344 bytes
combine per token per layer  = 8 × 7,168 × 2 bytes = 114,688 bytes
total per token per layer    = 168 KiB
over 58 MoE layers           ≈ 10 MB per token
```

That's an upper bound, since experts on the token's own GPU need no transfer. Now give each GPU an illustrative 128 tokens per decode step: 22 MB leaves each GPU per MoE layer. Over a 50 GB/s InfiniBand link (the H800 cluster's figure in DeepSeek's report), that's 0.44 ms per layer and about 25 ms per step across 58 layers, unless it overlaps with computation. That's why DeepSeek splits each batch into two micro-batches and runs one's attention while the other's dispatch and combine are on the wire.

Per layer, though, expert parallelism moves less data than tensor parallelism would. An all-reduce over `N` GPUs moves `2(N − 1)` times the message in total, so tensor-parallel layers for DeepSeek-V3's hidden size across 32 GPUs would move about 1.8 MB per token per layer, against EP's 168 KiB. [llm-d's docs](https://github.com/llm-d/llm-d/blob/main/docs/architecture/disaggregation/wide-expert-parallelism.md) make the same point: dispatch and combine "consume much less bandwidth than the all-reduces of TP setups," so they work over InfiniBand or RoCE (RDMA over Converged Ethernet), the networks between GPU servers.

### Load imbalance

The router picks experts by content, so popular experts get more tokens, and every GPU waits for the slowest one at each MoE layer. If one GPU receives twice the average number of tokens, the whole layer takes about twice as long (illustrative). vLLM's docs note that in practice "the distribution of tokens across experts can be highly skewed."

DeepSeek's answer is redundancy: it duplicates high-load experts, detected from live traffic and "adjusted periodically (e.g., every 10 minutes)," with 32 redundant experts in prefill. vLLM's version is the Expert Parallel Load Balancer, `--enable-eplb`, which tracks load and moves experts while serving. Each extra copy costs memory: about 2.4 GB per redundant expert per GPU for DeepSeek-V3, by vLLM's estimate (44 million parameters per expert, times 58 layers).

### Attention runs data-parallel

The attention layers of DeepSeek-V3 use multi-head latent attention (MLA, from Part 2), which caches one shared 576-number latent per token per layer instead of per-head keys and values. That latent can't be split by heads. As the team behind the SGLang serving engine [put it](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), DeepSeek models "only have one KV head," so tensor parallelism "will lead to duplicated KV cache." At TP = 8, every GPU stores the same 68.6 KiB per token, 549 KiB in total.

So MLA models run attention with data parallelism: each GPU keeps full attention weights (small next to the experts) and its own requests' cache, and the GPUs join forces only in the MoE layers. vLLM's [data parallel docs](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) recommend exactly this for "MoE models, particularly those like DeepSeek that employ MLA," with expert parallelism for the expert layers. Spread over many nodes, the pattern is called **wide expert parallelism**.

Fast all-to-all kernels are what make this work at scale. DeepSeek open-sourced [DeepEP](https://github.com/deepseek-ai/DeepEP), which provides "high-throughput and low-latency expert-parallel (EP) all-to-all GPU kernels for MoE dispatch and combine, including FP8 dispatch." vLLM offers it as `deepep_high_throughput` for prefill and `deepep_low_latency` for decode. Part 14 returns to DeepEP and DeepSeek's load balancer in depth.

## Step 4: choosing a layout

The layout follows from how many bytes each split sends and how fast the links between the GPUs are. If the model fits, copy it; if it doesn't, use the splits that send the most data on the fastest links.

| Model and hardware | Layout | Why |
|---|---|---|
| Fits on one GPU (Llama 3.1 8B on H100s) | DP replicas only | No communication; throughput scales with GPU count |
| Too big for one GPU, fits one NVLink server (Llama 3.1 70B in BF16 on 8 H100s) | TP = 2, 4 or 8 inside the server, DP replicas for more throughput | Two all-reduces per layer stay on NVLink; higher TP cuts latency |
| Too big for one server (Llama 3.1 405B in BF16 on H100s) | TP = 8 inside each server, PP across servers | One small transfer per token crosses the network |
| GPUs without NVLink (for example L40S) | PP instead of TP | vLLM's advice: "higher throughput and lower communication overhead" |
| Large MoE with MLA on one server (DeepSeek-V3 in FP8 on 8 H200s) | DP attention + EP experts | No duplicated cache; experts spread over GPUs |
| Large MoE across many servers | Wide EP with DP attention, prefill and decode on separate groups | Sparse all-to-all works over InfiniBand or RoCE |
| Very long prompts | Add context parallelism | Splits one prompt's tokens and cache across GPUs |

vLLM's [parallelism guide](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) states the dense-model rule directly: if a model is "too large for a single node, combine tensor parallelism with pipeline parallelism," setting the TP size to the GPUs per node and the PP size to the number of nodes. In flags:

```bash
# Llama 3.1 70B: tensor parallelism across one 8-GPU node
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 8

# Llama 3.1 405B in BF16: TP inside each node, PP across 2 nodes (16 GPUs)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 --pipeline-parallel-size 2

# DeepSeek-V3: data-parallel attention, expert-parallel MoE on one 8-GPU H200 node
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel
```

With `--enable-expert-parallel`, vLLM sizes the expert group as `TP × DP`, 8 in the last example. Multi-node runs need one command per node and a fast network.

### Context parallelism for very long prompts

One more split targets long inputs. **Context parallelism** (CP) divides one prompt's tokens across GPUs, so each GPU holds and computes attention for a slice of the sequence. [Ring Attention](https://arxiv.org/abs/2310.01889) (Liu, Zaharia and Abbeel, 2023) keeps the result exact by passing blocks of keys and values around a ring of GPUs, overlapping that traffic with the attention math. Meta adapted it for serving: on 128 H100s across 16 nodes, Llama 3 405B prefilled a 1-million-token prompt in 77 seconds and a 128K prompt in 3.8 seconds, where one 8-GPU host needs about 60 seconds for 128K. Part 11 covers the rest of long-context inference.

## In the real world

- Meta's Llama 3 405B. The [Llama 3 paper](https://arxiv.org/abs/2407.21783) served the BF16 model on "16 GPUs on two machines": tensor parallelism inside each machine thanks to "the high NVLink bandwidth," and pipeline parallelism across them because cross-node links have "lower bandwidth and higher latency." Its FP8 version fits on one machine and gave up to 50% higher prefill throughput.
- TP versus PP, measured. NVIDIA [ran Llama 3.1 405B](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-throughput-by-another-1-5x-on-nvidia-h200-tensor-core-gpus-and-nvlink-switch) in FP8 on an 8-GPU HGX H200. In the minimum-latency scenario, TP produced 56 output tokens per second against PP's 10. At maximum throughput, PP produced 764 against TP's 506, 1.5 times more. Our ceilings above, 95 and 12 tokens per second, predict the same ordering.
- DeepSeek-V3 in production. The [technical report](https://arxiv.org/abs/2412.19437) describes a decode unit of 40 nodes and 320 H800 GPUs with EP = 320, one expert per GPU. DeepSeek's later [February 2025 overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) lists prefill at EP32 with attention at DP32 over 4 nodes, and decode at EP144 with DP144 over 18 nodes, 2 routed experts per GPU. Each 8-GPU node averaged about 73,700 input tokens per second in prefill and 14,800 output tokens per second in decode.
- Wide EP in vLLM and llm-d. The vLLM team [reported](https://vllm.ai/blog/2025-12-17-large-scale-serving) community benchmarks of DeepSeek at "2.2k tokens/s per H200 GPU in production-like, multi-node deployments" with wide EP, dual-batch overlap and EPLB. llm-d's [wide-EP guide](https://github.com/llm-d/llm-d/tree/main/guides/wide-ep) deploys DeepSeek-R1-0528 on 32 H200s: a prefill group and a decode group, each DP16 with EP across two nodes, DeepEP for the all-to-all and the NIXL transfer library to move the KV cache between them. Part 12 covers that prefill and decode split.
- Rack-scale NVLink. NVIDIA's [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) is a "72-GPU NVIDIA NVLink domain that acts as a single, massive GPU," with 130 TB/s of NVLink bandwidth and 13.4 TB of HBM3e memory. DeepSeek-V3's 671 GB fits 20 times over, and expert traffic moves at NVLink 5's 900 GB/s each way instead of 50 GB/s over InfiniBand, 18 times faster. Our 25 ms of per-step expert traffic drops to about 1.4 ms.

## Common misconceptions

- **"Eight GPUs make a model eight times faster."** Tensor parallelism cuts each GPU's weight reads by 8 but adds 160 all-reduces per token for Llama 3.1 70B, each with a fixed latency. As NVIDIA puts it, "Amdahl's Law limits the speed-up possible with each additional GPU."
- **"Pipeline parallelism lowers latency."** It lets a model fit and raises throughput, but each token still goes through every stage in turn: 10 tokens per second for PP against 56 for TP in NVIDIA's test.
- **"Tensor parallelism across servers is fine with a fast network."** A 400 Gb/s port moves 50 GB/s each way, a ninth of NVLink 4's 450 GB/s. A 4,096-token prefill of Llama 3.1 70B at TP = 8 would spend about 376 ms on all-reduces over such a link, five times its compute time.
- **"Every GPU in an MoE deployment does the same amount of work."** The router decides, and real traffic is skewed. Without redundant experts and rebalancing, the busiest GPU sets the pace for all of them.
- **"More tensor parallelism always shrinks the KV cache per GPU."** Only until TP reaches the number of KV heads. Past that, heads are replicated, and for MLA models, whose cache has no heads to split, every GPU in the group keeps the same copy.

## Key takeaways

- You split a model for two reasons: its weights and KV cache don't fit in one GPU, or one GPU can't read the weights fast enough for the latency you want.
- Tensor parallelism slices every layer and needs two all-reduces per block: 160 per token for Llama 3.1 70B, each 16 KiB per token. That's cheap over NVLink and expensive over anything else, so keep TP inside one NVLink domain.
- Pipeline parallelism gives each GPU a range of layers and sends just 16 to 32 KiB per token across each boundary, so it works between servers. It raises throughput, not single-request speed.
- Data parallelism runs independent replicas with no communication. It's the first choice whenever the model fits.
- Expert parallelism spreads an MoE model's experts across GPUs and sends each token only where it's routed, about 168 KiB per token per layer for DeepSeek-V3. Load balancing and data-parallel attention make it work at scale.

## What's next

Parallelism spreads a model's weights across GPUs, but a single long conversation can still swamp the KV cache and make attention the slowest part of every step. [Part 11](/blog/long-context-inference/) looks at long-context inference: sliding-window attention, sparse attention and ways to compress or evict the KV cache, so a million-token context doesn't need a million tokens' worth of memory.

## Further reading

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053): column- and row-parallel splits and the two all-reduces per layer.
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Huang et al., 2018)](https://arxiv.org/abs/1811.06965): pipeline stages, micro-batches and the bubble.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783): serving 405B with TP inside nodes and PP across them, and why bubbles matter less in inference.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): node-limited routing, FP8 dispatch, redundant experts and the prefill and decode deployment units.
- [DeepSeek-V3/R1 Inference System Overview (DeepSeek, 2025)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md): large-scale EP with data-parallel attention, with production throughput.
- [DeepEP (DeepSeek)](https://github.com/deepseek-ai/DeepEP): the open-source all-to-all kernels for MoE dispatch and combine.
- [Ring Attention with Blockwise Transformers for Near-Infinite Context (Liu, Zaharia and Abbeel, 2023)](https://arxiv.org/abs/2310.01889): splitting a sequence across devices in a ring.
- [Context Parallelism for Scalable Million-Token Inference (Yang et al., 2024)](https://arxiv.org/abs/2411.01783): ring attention adapted for serving, with 1M-token prefill on 128 H100s.
- [vLLM: Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/), [Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) and [Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/): the flags, layouts and EPLB settings.
- [NCCL User Guide (NVIDIA)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html) and [nccl-tests performance notes](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md): the collective operations and the all-reduce time formula.
- [NVIDIA NVLink and NVSwitch Supercharge Large Language Model Inference (NVIDIA, 2024)](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference): how much TP traffic a 70B query generates and what the link does to throughput.
- [Boosting Llama 3.1 405B Throughput by Another 1.5x on NVIDIA H200 (NVIDIA, 2024)](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-throughput-by-another-1-5x-on-nvidia-h200-tensor-core-gpus-and-nvlink-switch): TP versus PP for latency and throughput, measured.
- [SGLang v0.4 (LMSYS, 2024)](https://lmsys.org/blog/2024-12-04-sglang-v0-4/): why MLA models use data-parallel attention.
- [vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP (vLLM team, 2025)](https://vllm.ai/blog/2025-12-17-large-scale-serving) and [llm-d: Wide Expert Parallelism](https://github.com/llm-d/llm-d/blob/main/docs/architecture/disaggregation/wide-expert-parallelism.md): wide EP, dual-batch overlap and load balancing in open-source stacks.
