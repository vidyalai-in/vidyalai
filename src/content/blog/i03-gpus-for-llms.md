---
title: "How GPUs Work for LLMs: Cores, Memory and Number Formats"
description: "What is inside an H100: thousands of cores, tensor cores, a memory hierarchy from 256 KB to 80 GB, number formats down to FP4, and NVLink."
slug: "gpus-for-llms"
category: "inference-optimization"
tags: ["gpu", "serving", "latency", "cost"]
series: "llm-inference"
seriesOrder: 3
pubDate: 2026-10-23
---

## TL;DR

A GPU is thousands of simple arithmetic units organized into blocks called streaming multiprocessors, plus special matrix engines called tensor cores that do most of an LLM's math. The hard part is feeding them: on an H100 the fast on-chip memory holds about 118 MB, while the model lives in 80 GB of slower high-bandwidth memory beside the chip. Smaller number formats (BF16, then FP8, then FP4) double the math rate at each step and halve the bytes to move. When a model doesn't fit on one GPU, NVLink connects several GPUs at about 7 times the speed of the ordinary PCIe slot. By the end you can work out with a pencil whether a model fits on a GPU and how fast it can decode at best.

## Why it matters

The same NVIDIA H100 does about [67 trillion](https://www.nvidia.com/en-us/data-center/h100/) ordinary 32-bit operations per second, and about 989 trillion 16-bit operations per second when the work is a matrix multiplication. That's a 15x difference on the same chip, and LLMs are built almost entirely out of the second kind of work.

Yet in [Part 1](/blog/how-llm-inference-works/) we saw that an H100 serving one user leaves more than 99% of that matrix power idle. The bottleneck was memory bandwidth: 3.35 TB/s between the GPU and its memory.

A GPU is an enormous calculator fed through a narrow pipe. FlashAttention, quantization, tensor parallelism and the rest of this series each use the calculator better or the pipe less, and they make sense once you can picture what's inside the chip.

## The intuition

Part 1 compared inference to a chef whose recipe book is stored in a warehouse down the street. Now picture the kitchen itself. A CPU is a small kitchen with a few master chefs who can each cook anything. A GPU is a vast canteen kitchen with thousands of line cooks. None of them is clever, but they all chop at the same time, and some stations have a machine that chops a whole crate in one motion. Those machines are the tensor cores.

Each station has a small counter right next to it (fast on-chip memory, called SRAM). The canteen has one shared pantry (the L2 cache). The bulk supplies sit in the warehouse (HBM, the GPU's main memory) at the end of a wide but finite road. Cooks working from the counter are fast; cooks waiting for a truck are idle.

The analogy breaks in one place: real line cooks can each do something different, but GPU cores work in lockstep groups of 32, all running the same instruction on different data. A GPU is fast when the work is the same simple operation repeated millions of times, which is exactly what a matrix multiplication is.

## Step 1: thousands of simple cores

A CPU spends its chip area on making a few cores fast at anything, with branch prediction, instruction reordering and large caches. NVIDIA's own [Grace CPU Superchip](https://www.nvidia.com/en-us/data-center/grace-cpu/), a two-chip server CPU, has 144 cores. A GPU spends its area on arithmetic.

The H100's cores are grouped into **streaming multiprocessors** (SMs). An SM is a self-contained block with its own cores, its own fast memory and its own scheduler. A GPU program (a "kernel") is split into many small blocks of work, and the GPU hands them out to SMs. Here's what the H100 SXM contains, from NVIDIA's [Hopper architecture blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/):

| Part | Per SM | Whole H100 SXM (132 SMs) |
|---|---|---|
| FP32 cores (CUDA cores) | 128 | 16,896 |
| Tensor cores (4th generation) | 4 | 528 |
| Register file | 256 KB | about 34 MB |
| Shared memory and L1 cache (combined) | 256 KB | about 34 MB |
| Threads that can be resident at once | 2,048 | 270,336 |

The full GH100 chip has 144 SMs; the H100 SXM product enables 132 of them. Threads run in groups of 32 called **warps**: one instruction, 32 pieces of data. Each SM can hold up to 64 warps, so when one stalls waiting for memory, the scheduler switches to another that's ready. That hides some of the memory delay, provided there's enough work to go around.

### Tensor cores: the matrix machines

An ordinary core does one multiply-add on a pair of numbers. A **tensor core** multiplies two small matrices and adds the result to a third in a single operation. A Transformer is mostly matrix multiplications (every projection in attention, every layer of the MLP), so nearly all of its arithmetic lands on tensor cores. From the [H100 SXM datasheet](https://www.nvidia.com/en-us/data-center/h100/):

```text
FP32 on ordinary cores      =  67 TFLOPS
BF16 on tensor cores        = 989 TFLOPS (dense)
ratio                       ≈ 15x
```

Horace He makes the same point about the previous generation in [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html): on an A100, "if you aren't doing matrix multiplication, you'll only be able to achieve 19.5 teraflops instead of the stated 312." Softmax, layer norm and activation functions run on the ordinary cores. They're a small share of the FLOPs and slow per FLOP, and as the next step shows, memory usually limits them anyway.

## Step 2: the memory hierarchy

Cores can only work on numbers that are close to them, so a GPU has several layers of memory.

![The H100 memory hierarchy. On the chip, in SRAM: registers of 256 KB per SM, shared memory and L1 cache of 256 KB per SM, and a 50 MB L2 cache shared by all SMs. Below the chip, 80 GB of HBM3 memory connected at 3.35 TB/s. From HBM, links go to other GPUs in the server over NVLink at 450 GB/s each way, to the CPU and its memory over PCIe Gen5 at 64 GB/s each way, and to other servers over the network at 50 GB/s each way.](/diagrams/i03-memory-hierarchy.svg "The H100's memory, from the fastest and smallest at the top to the slowest links at the bottom.")

Each level down is bigger and slower, and the biggest drop in speed is between the chip and its HBM. From the top:

- **Registers** hold the numbers a thread is working on right now. 256 KB per SM.
- **Shared memory and L1 cache** are 256 KB of SRAM (static random-access memory, fast memory built into the chip) per SM, which a kernel can partly use as a scratchpad it controls itself. Up to 228 KB of it can be shared memory.
- **L2 cache** is 50 MB of SRAM shared by all 132 SMs.
- **HBM** (high-bandwidth memory) is 80 GB of DRAM, in five stacks of memory chips mounted next to the GPU die, delivering 3.35 TB/s. This is where the model's weights and the KV cache live.

Add up the SRAM and you get about 34 + 34 + 50 ≈ 118 MB. The HBM is about 680 times larger. Llama 3.1 8B's 16 GB of weights is 321 times the size of the L2 cache, so the weights can't stay on the chip. Every decode step streams them in from HBM: the memory wall from Part 1.

How much faster is SRAM? NVIDIA's spec pages don't list an SRAM bandwidth, but the [FlashAttention paper](https://arxiv.org/abs/2205.14135) estimates it for the A100: HBM at 1.5 to 2.0 TB/s, on-chip SRAM at "around 19TB/s." In its words, SRAM "is an order of magnitude faster than HBM but many orders of magnitude smaller in size."

### Why this decides how kernels are written

Suppose a model applies three simple element-by-element operations in a row (say, scale, add a bias, then an activation function) to a 1 GB tensor. Done naively, each operation reads the tensor from HBM and writes the result back:

```text
unfused: 3 operations × (1 GB read + 1 GB write) = 6 GB of HBM traffic
         6 GB ÷ 3.35 TB/s ≈ 1.8 ms
fused:   1 GB read + 1 GB write, all 3 operations done in SRAM = 2 GB
         2 GB ÷ 3.35 TB/s ≈ 0.6 ms
```

The math is the same in both cases; the fused version just makes a third of the trips to HBM. This is **kernel fusion**, which Horace He calls "the most important optimization in deep learning compilers." The numbers are illustrative. FlashAttention applies the same idea to attention, computing it in tiles that fit in shared memory so the large intermediate matrix never reaches HBM. Part 7 covers it.

## Step 3: number formats, from FP32 to FP4

Every weight is stored as a floating-point number: a sign bit, some **exponent** bits that set how big or small the number can be (its range), and some **mantissa** bits that set how finely it's resolved (its precision). Fewer bits means less memory, less bandwidth and, on tensor cores, faster math, at the cost of precision. In the table, "gap after 1.0" is the distance to the next number the format can represent, a simple measure of precision:

| Format | Sign / exponent / mantissa bits | Largest value | Gap after 1.0 | π is stored as |
|---|---|---|---|---|
| FP32 | 1 / 8 / 23 | about 3.4 × 10^38 | 0.00000012 | 3.1415927 |
| TF32 | 1 / 8 / 10 | same as FP32 | 0.00098 | 3.140625 |
| BF16 | 1 / 8 / 7 | about 3.4 × 10^38 | 0.0078 | 3.140625 |
| FP16 | 1 / 5 / 10 | 65,504 | 0.00098 | 3.140625 |
| FP8 E4M3 | 1 / 4 / 3 | 448 | 0.125 | 3.25 |
| FP8 E5M2 | 1 / 5 / 2 | 57,344 | 0.25 | 3.0 |
| FP4 E2M1 | 1 / 2 / 1 | 6 | 0.5 | 3.0 |

- **BF16** ("brain floating point," from Google Brain) keeps FP32's 8 exponent bits, so, as [Google puts it](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus), its dynamic range "is identical to that of FP32." It gives up precision instead. FP16 makes the opposite trade and tops out at 65,504. Large models train and serve in BF16 because overflow does more damage than a little rounding.
- **TF32** isn't a storage format. It's a tensor-core math mode: FP32 inputs are rounded to 19 bits for the multiply and the results are added up in FP32. [NVIDIA](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/) describes it as "only exposed as a Tensor Core operation mode, not a type."
- **FP8** has two variants. [NVIDIA's Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html) say E4M3 "can store values up to +/-448" and E5M2 "up to +/-57344," with E4M3 typically used for weights and activations and E5M2 for gradients in training. The formats are defined in [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433).
- **FP4** can represent only 0, 0.5, 1, 1.5, 2, 3, 4 and 6, plus their negatives. That's far too coarse on its own, so Blackwell's **NVFP4** format gives every block of 16 values its own FP8 scale factor, plus one FP32 scale per tensor. [NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) counts this as about 4.5 bits per value and about 3.5 times less memory than FP16.

### Halve the bits, double the math

Each halving of precision doubles tensor-core throughput. The datasheet numbers, converted to dense (explained below):

| Format | H100 SXM (dense TFLOPS) | B200 (dense TFLOPS per GPU) |
|---|---|---|
| TF32 | 495 | 1,125 |
| BF16 / FP16 | 989 | 2,250 |
| FP8 | 1,979 | 4,500 |
| FP4 | not supported | 9,000 |

H100 figures are half the "with sparsity" numbers on the [H100 page](https://www.nvidia.com/en-us/data-center/h100/). The [HGX B200 specs](https://www.nvidia.com/en-us/data-center/hgx/) list sparse totals for 8 GPUs, so those are divided by 2, then by 8. FP4 tensor cores are new in Blackwell.

Fewer bits also means fewer bytes. Llama 3.1 70B has 70.55 billion parameters:

```text
FP32: 70.55 billion × 4 bytes   ≈ 282 GB
BF16: 70.55 billion × 2 bytes   ≈ 141 GB
FP8:  70.55 billion × 1 byte    ≈ 71 GB
FP4:  70.55 billion × 0.5 bytes ≈ 35 GB  (about 40 GB with NVFP4's scale factors)
```

Decode time depends on how many bytes cross from HBM each step, so halving the bytes roughly halves the time per token. Part 8 covers how to shrink a model without wrecking its answers.

### The sparsity asterisk

Many datasheets print a number twice as large as the one in this article. The H100 page lists 1,979 TFLOPS for BF16 with a footnote: "With sparsity." That figure assumes **2:4 structured sparsity**, where, in NVIDIA's [description](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/), "Among each group of four contiguous values, at least two must be zero." The hardware then skips the zeros. Ordinary LLM weights don't follow that pattern, so for inference the dense number is the realistic one, and Part 1's ridge point used 989 TFLOPS for that reason.

## Step 4: what fits on one GPU, and linking several

You now have enough to size a deployment. The GPU's memory has to hold the weights, then the KV cache, which [Part 2](/blog/kv-cache/) worked out per token: 128 KiB for Llama 3.1 8B and 320 KiB for Llama 3.1 70B (in BF16).

```text
memory left for cache = GPU memory − weights
tokens of cache       = memory left ÷ KV bytes per token
```

Part 1 did this for the 8B model on an H100: 64 GB left, about 488,000 tokens. Here is the 70B model:

| Setup | GPU memory | Weights | Left for cache | Tokens of cache | Users at 4,096 tokens |
|---|---|---|---|---|---|
| BF16 on 1 × H100 | 80 GB | 141.1 GB | doesn't fit | none | none |
| BF16 on 1 × H200 | 141 GB | 141.1 GB | doesn't fit (0.1 GB short) | none | none |
| BF16 on 2 × H100 | 160 GB | 141.1 GB | 18.9 GB | 57,656 | 14 |
| BF16 on 1 × B200 | 180 GB | 141.1 GB | 38.9 GB | 118,691 | 29 |
| FP8 on 1 × H100 | 80 GB | 70.6 GB | 9.4 GB | 28,828 | 7 |
| FP8 on 1 × H200 | 141 GB | 70.6 GB | 70.4 GB | 214,985 | 52 |
| FP8 on 1 × B200 | 180 GB | 70.6 GB | 109.4 GB | 334,004 | 81 |

These are upper bounds. A real server also needs memory for activations, temporary buffers and the runtime, so it fits fewer users. Still, the same model goes from not fitting at all to 81 users depending on the GPU and the number format. Bandwidth sets each user's speed limit, as in Part 1:

```text
70B in FP8 on an H100: 70.6 GB ÷ 3.35 TB/s ≈ 21 ms per step ≈ 47 tokens/s at most
70B in FP8 on a B200:  70.6 GB ÷ 8 TB/s    ≈ 8.8 ms per step ≈ 113 tokens/s at most
```

### When one GPU isn't enough

If a model doesn't fit, you split it across GPUs. The most common split, tensor parallelism, cuts every weight matrix into slices, gives each GPU one slice, and has the GPUs exchange partial results at every layer. Llama 3.1 70B has 80 layers, so the exchange repeats many times for every token, and the speed of the link between GPUs shows up directly in latency.

Inside a server, NVIDIA GPUs talk over **NVLink**, a direct GPU-to-GPU link. The H100 has 18 NVLink connections with "900 GB/sec total bandwidth," according to the [Hopper blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/). **NVSwitch** chips in the server connect every GPU to every other at that full speed. Links that leave the GPU any other way are much slower. Time to move 1 GB in one direction:

| Link | Bandwidth each way | Time to move 1 GB |
|---|---|---|
| HBM, inside one H100 | 3.35 TB/s (total) | 0.3 ms |
| NVLink 5, B200 to B200 | 900 GB/s (half of 1.8 TB/s total) | 1.1 ms |
| NVLink 4, H100 to H100 | 450 GB/s (half of 900 GB/s total) | 2.2 ms |
| PCIe Gen5 x16, GPU to CPU | 64 GB/s | 15.6 ms |
| 400 Gb/s network port, server to server | 50 GB/s | 20 ms |

H100 NVLink is 7 times faster than its PCIe Gen5 connection (900 versus 128 GB/s, both counted in both directions). Blackwell doubles NVLink to [1.8 TB/s per GPU](https://www.nvidia.com/en-us/data-center/hgx/), while the [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) connects to other servers through 400 Gb/s ports. Hence the rule of thumb: keep tensor parallelism inside one NVLink domain and use other strategies across servers, as Part 10 explains. NVIDIA's [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) stretches the domain to a whole rack: 72 GPUs on one NVLink fabric with 130 TB/s of total bandwidth.

## In the real world

### The accelerators people serve models on

Dense, per-chip figures come from each vendor's own pages. The last two columns use this series' formulas: the ridge point is dense BF16 FLOPs divided by bandwidth, and the decode ceiling is one user on Llama 3.1 8B in BF16 (16.06 GB of weights per step).

| Accelerator | Memory | Bandwidth | BF16 (dense TFLOPS) | FP8 (dense TFLOPS) | Ridge point (FLOPs per byte) | 8B decode ceiling |
|---|---|---|---|---|---|---|
| [NVIDIA H100 SXM](https://www.nvidia.com/en-us/data-center/h100/) | 80 GB HBM3 | 3.35 TB/s | 989 | 1,979 | 295 | 209 tokens/s |
| [NVIDIA H200 SXM](https://www.nvidia.com/en-us/data-center/h200/) | 141 GB HBM3e | 4.8 TB/s | 989 | 1,979 | 206 | 299 tokens/s |
| [NVIDIA B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) (DGX B200 ÷ 8) | 180 GB HBM3e | 8 TB/s | 2,250 | 4,500 | 281 | 498 tokens/s |
| [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) | 192 GB HBM3 | 5.3 TB/s | 1,307 | 2,615 | 247 | 330 tokens/s |
| [AMD Instinct MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html) | 256 GB HBM3E | 6 TB/s | 1,307 | 2,615 | 218 | 374 tokens/s |
| [Google TPU7x (Ironwood)](https://docs.cloud.google.com/tpu/docs/tpu7x) | 192 GiB HBM | 7.37 TB/s | 2,307 | 4,614 | 313 | 459 tokens/s |

The H200 makes a clean comparison: the H100's tensor cores and FLOPs, with 76% more memory and 43% more bandwidth. NVIDIA reports it running Llama 2 70B inference [1.9 times faster](https://www.nvidia.com/en-us/data-center/h200/) than the H100, and the footnote shows how: batch size 32 on the H200 against 8 on the H100. The extra memory let four times as many users share each weight read.

Every ridge point in the table sits between about 200 and 315 FLOPs per byte. Vendors raise compute and bandwidth together, so single-user decode is memory-bound on all of them and batching is still how you use the hardware.

### What an hour of GPU time buys

Cloud provider [Lambda](https://lambda.ai/pricing) lists an H100 SXM at $3.99 per GPU-hour and a B200 at $6.69 per GPU-hour on its 8-GPU on-demand instances (October 2026). Using Part 1's throughput numbers for Llama 3.1 8B:

```text
1 user,   208 tokens/s:   0.75 million tokens/hour → $3.99 ÷ 0.75  ≈ $5.32 per million tokens
64 users, 4,250 tokens/s: 15.3 million tokens/hour → $3.99 ÷ 15.3  ≈ $0.26 per million tokens
```

Batching alone makes each token 20 times cheaper on the same GPU and model. The B200 costs 1.68 times as much per hour as the H100 but has 2.39 times the bandwidth, so for memory-bound decode it costs less per token. Prices change often and vary by provider and commitment; use them to reason with. [How to Cut LLM Cost by 70%](/blog/cut-llm-cost-70-percent/) and [Serving Open LLMs at Scale](/blog/serving-open-llms-at-scale/) apply the same math to real fleets.

## Common misconceptions

- **"A GPU is just a faster CPU."** The designs differ. NVIDIA's Grace CPU Superchip has 144 cores, each tuned for varied, branching work. An H100 has 16,896 simple cores and 528 tensor cores that only run fast when thousands of threads do the same thing on different data.
- **"Bigger TFLOPS means faster chat."** For one user, decode speed follows bandwidth. The H200 has exactly the H100's FLOPs, yet its single-user ceiling on our 8B model is 299 tokens per second versus 209.
- **"The datasheet number is what you'll get."** Many headline numbers assume 2:4 sparsity, which ordinary LLM weights don't have. Divide by two for the dense figure, then expect real kernels to reach only part of that.
- **"More GPUs always means more speed."** Splitting a model adds communication at every layer. Over NVLink that's cheap; over PCIe or the network it can cost more than it saves.

## Key takeaways

- A GPU is 132 streaming multiprocessors on an H100, each with ordinary cores and 4 tensor cores. Tensor cores do matrix math about 15 times faster than the ordinary cores, and an LLM is mostly matrix math.
- Memory is a hierarchy: about 118 MB of fast on-chip SRAM, then 80 GB of HBM at 3.35 TB/s. Model weights never fit on the chip, so decode streams them from HBM every step.
- Each halving of precision, from BF16 to FP8 to FP4, doubles tensor-core throughput and halves the bytes per weight. When comparing chips, use the dense figures.
- To size a deployment: weights plus KV cache must fit in memory, and bytes per step divided by bandwidth gives the speed limit. Llama 3.1 70B in BF16 doesn't fit on one H100; in FP8 it fits with room for about 7 users at 4,096 tokens each.
- NVLink moves data between GPUs about 7 times faster than PCIe, which is why tensor parallelism stays inside one server's NVLink domain.

## What's next

Batching keeps a GPU busy because the hardware has math to spare and bandwidth to protect. Requests, though, don't arrive in neat batches. They start and finish at different times, and their prompts range from ten tokens to a hundred thousand. Part 4 covers continuous batching and chunked prefill, the scheduling tricks that keep a GPU full while keeping every user's stream smooth.

## Further reading

- [NVIDIA Hopper Architecture In-Depth (NVIDIA, 2022)](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/): the H100's SMs, tensor cores, memory hierarchy, NVLink and NVSwitch, from the people who built it.
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/), [H200](https://www.nvidia.com/en-us/data-center/h200/), [HGX](https://www.nvidia.com/en-us/data-center/hgx/) and [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) product pages: the official specs, with the sparsity footnotes.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135): the clearest statement of why SRAM versus HBM matters, with A100 numbers.
- [FP8 Formats for Deep Learning (Micikevicius et al., 2022)](https://arxiv.org/abs/2209.05433): the definition of the E4M3 and E5M2 formats.
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference (NVIDIA, 2025)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/): how 4-bit floating point stays accurate with per-block scales.
- [Using FP8 and FP4 with Transformer Engine (NVIDIA)](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html): FP8, MXFP8 and NVFP4 ranges and scaling in practice.
- [Making Deep Learning Go Brrrr From First Principles (Horace He)](https://horace.io/brrr_intro.html): compute, memory bandwidth and overhead, and why operator fusion matters.
