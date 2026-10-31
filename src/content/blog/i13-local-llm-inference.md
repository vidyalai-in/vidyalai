---
title: "Running LLMs on Your Own Machine: llama.cpp, GGUF and Ollama"
description: "How fast a model can run on your laptop and what fits: bandwidth math, unified memory, llama.cpp, GGUF quants, Ollama, MLX and measured numbers."
slug: "local-llm-inference"
category: "inference-optimization"
tags: ["local", "quantization", "latency", "gpu", "cost"]
series: "llm-inference"
seriesOrder: 13
pubDate: 2026-04-06
updatedDate: 2026-10-23
---

## TL;DR

On your own machine, one number sets how fast a model can talk: memory bandwidth divided by the bytes the model reads per token. A 4-bit Llama 3.1 8B is a 4.9 GB file, so a laptop with 90 GB/s of memory bandwidth tops out near 18 tokens per second and a 614 GB/s Mac near 125. Memory size decides what fits, which is why a Mac or mini PC with 128 GB of unified memory can run a 70B model that a 32 GB graphics card can't hold. llama.cpp, its GGUF file format and the apps built on it (Ollama, LM Studio) make all of this a one-line install. By the end you'll be able to size a model and a quantization for your own hardware, and you'll know why long prompts are the part laptops find hardest.

## Why it matters

On a gaming laptop, the same 4.9 GB model writes 11 tokens per second on its CPU and 45 on its 8 GB graphics card. Reading a 4,096-token prompt before the first word of the answer takes 82 seconds on the CPU and 2.3 seconds on the GPU. Both figures come from the test laptop described in Step 2, and the gap between them is what this article explains.

People run models locally for plain reasons. Prompts and documents never leave the machine. It works on a plane. There's no per-token bill once you own the hardware, and no network round trip before the first token. And you can try any open-weight model, quantization or setting you like.

The rest of this series ran on data center GPUs serving many users at once. This part is the opposite case: one user, one machine, and memory that's smaller and slower than an H100's.

## The intuition

Picture a cook who has to reread an entire recipe binder before writing each word of a menu. The cook's pace depends on how fast they can flip through the binder and how far away it sits. A binder on the counter (GPU memory) is quick to read. One in the basement (system RAM) is slow. A binder too thick for the counter has to be split, and then every word means a trip downstairs.

That's local inference. The binder is the model's weights, the counter is fast memory, and the trip is memory bandwidth. Quantization makes the binder thinner. Unified memory, as on Apple silicon, is a kitchen where the whole counter is fast and very large.

The analogy breaks in one place. Before writing anything, the cook reads your order, the prompt, in one go. That part is limited by how fast the cook can think (compute), not by reading speed, and it's where laptops struggle most.

## Step 1: the speed limit is bandwidth divided by bytes

[Part 1](/blog/how-llm-inference-works/) showed that each decode step, which produces one new token, reads every weight of the model once. **Memory bandwidth** is how many bytes per second the processor can pull from memory. So for one user:

```text
tokens per second (ceiling) ≈ memory bandwidth ÷ bytes read per token
```

For a 4-bit Llama 3.1 8B, the bytes per token are roughly the file size, or slightly less, since only one row of the word-embedding table is read per token: 4.92 GB for the Q4_K_M version (more on that name in Step 3). On a PC with two sticks of DDR5-5600 memory (DDR5 is today's standard PC memory), bandwidth is 5,600 million transfers per second × 8 bytes per transfer × 2 channels = 89.6 GB/s. So:

```text
89.6 GB/s ÷ 4.92 GB per token ≈ 18 tokens per second
```

Real speeds come in below that ceiling, because no chip reads memory at its full rated speed.

### Real machines, as of October 2026

Bandwidth and memory come from each vendor's spec page. The two right-hand columns apply the formula to Llama 3.1 8B at Q4_K_M (4.92 GB) and Llama 3.1 70B at Q4_K_M (43.1 GB, from [llama.cpp's quantize docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)):

| Machine | Memory | Bandwidth | 8B Q4_K_M ceiling | 70B Q4_K_M ceiling |
|---|---|---|---|---|
| Typical PC, dual-channel DDR5-5600 | Whatever is installed | 89.6 GB/s (computed) | 18 tokens/s | 2.1 tokens/s, if it fits |
| [MacBook Air or Pro, M5](https://www.apple.com/macbook-pro/specs/) | 16 to 32 GB unified | 153 GB/s | 31 tokens/s | Doesn't fit |
| [AMD Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) | Up to 128 GB LPDDR5x | 256 GB/s (computed) | 52 tokens/s | 5.9 tokens/s |
| [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) | 128 GB unified | 273 GB/s | 56 tokens/s | 6.3 tokens/s |
| [MacBook Pro, M5 Pro](https://www.apple.com/macbook-pro/specs/) | Up to 64 GB unified | 307 GB/s | 62 tokens/s | 7.1 tokens/s |
| [MacBook Pro or Mac Studio, M5 Max (40-core GPU)](https://www.apple.com/mac-studio/specs/) | Up to 128 GB unified | 614 GB/s | 125 tokens/s | 14 tokens/s |
| [Mac Studio, M5 Ultra](https://www.apple.com/mac-studio/specs/) | Up to 512 GB unified | 1.2 TB/s | 244 tokens/s | 28 tokens/s |
| [NVIDIA GeForce RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/) | 32 GB GDDR7 | 1,792 GB/s | 364 tokens/s | Doesn't fit |

AMD lists a "256-bit LPDDR5x" interface at "LPDDR5x-8000," so its bandwidth is 8,000 million transfers per second × 32 bytes = 256 GB/s. The RTX 5090 is the fastest on the list, yet it can't hold the 70B model.

### What fits: weights, KV cache and everything else

Speed is moot if the model doesn't fit. Memory has to hold the weights, which take about the file size (4.9 GB for an 8B model at Q4_K_M, 43.1 GB for a 70B), plus the operating system, your browser and the app itself.

It also has to hold the KV cache: the keys and values stored for every token of the conversation so far ([Part 2](/blog/kv-cache/)). Llama 3.1 8B needs 128 KiB per token in 16-bit numbers, so an 8,192-token conversation adds 1 GiB, and its full 128K window adds 16 GiB, more than three times the weights.

On a graphics processor (GPU), only its own memory (**VRAM**, video RAM) counts for speed. Anything that spills into system RAM runs at system RAM's speed.

## Step 2: unified memory, and splitting a model in two

A desktop with a graphics card has two separate pools of memory. The GPU has its own fast VRAM, and the CPU has slower system RAM, connected to the GPU through a PCIe (PCI Express) slot that [Part 3](/blog/gpus-for-llms/) measured at 64 GB/s each way on an H100 server. A model has to fit in VRAM to run at the GPU's speed.

Apple silicon works differently. When Apple introduced the M1 in 2020, it described a "unified memory architecture that brings together high-bandwidth, low-latency memory into a single pool," which "allows all of the technologies in the SoC [system on a chip] to access the same data without copying it between multiple pools of memory." The GPU reads the same RAM as the CPU, at the full rated bandwidth. A Mac with 128 GB has, roughly, a GPU with 128 GB of memory.

### The 70B example

Take Llama 3.1 70B at Q4_K_M, 43.1 GB, and an RTX 5090 in a desktop with DDR5. The card holds 32 GB. Say 29 GB of layers go on the card, leaving room for the cache, and the remaining 14.1 GB sit in system RAM:

```text
RAM part:  14.1 GB ÷ 89.6 GB/s  = 157 ms
GPU part:  29 GB   ÷ 1,792 GB/s =  16 ms
per token: 173 ms                → about 5.8 tokens per second
```

On an M5 Max with 128 GB, the whole model sits in unified memory: 43.1 ÷ 614 = 70 ms per token, a ceiling of about 14 tokens per second.

The graphics card has three times the Mac's bandwidth and still loses, because the slowest memory sets the pace. The AMD Ryzen AI Max+ 395 and the DGX Spark follow a similar design. AMD's chip has its Radeon 8060S graphics built in, sharing the same pool of up to 128 GB. NVIDIA calls the Spark's memory "coherent unified system memory" and says one unit handles models of "up to 200B parameters." Both fit the 70B model easily. At 256 and 273 GB/s, though, they run it at about 6 tokens per second.

### Splitting layers: what it costs

When a model doesn't fit in VRAM, llama.cpp (Step 3) can put some layers on the GPU and run the rest on the CPU. The `-ngl` flag (number of GPU layers) sets how many layers go to the GPU.

![A flowchart. One GGUF file contains metadata, the tokenizer and the chat template, plus the quantized weights, for example Q4_K_M. llama.cpp loads it and places N layers, set by the -ngl flag, in GPU memory, which is fast and small, and the remaining layers in system RAM, which is slow and large. Each token reads every layer once from both places, produces the next token, and repeats.](/diagrams/i13-layer-split.svg "llama.cpp splits a model's layers between GPU memory and system RAM; every token still reads all of them.")

Every token passes through both halves, so the slow half sets the speed.

Here's what that looked like on the test laptop: an AMD Ryzen 9 8945HS with two 16 GB sticks of DDR5-5600, an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB, Windows with WSL2 (Ubuntu), and llama.cpp build b11836 running Llama 3.1 8B Instruct at Q4_K_M. Llama 3.1 8B has 32 repeated layers plus an output layer, 33 in all. Measured with `llama-bench`, generating 128 tokens:

| Layers on the GPU (`-ngl`) | Share of weights on the GPU | Measured tokens/s |
|---|---|---|
| 0 | 0% | 10.1 |
| 16 | about half | 13.3 |
| 24 | about three quarters | 22.9 |
| All 33 | 100% | 41 to 45, across runs |

Moving half the model to a GPU with about three times the rated bandwidth gained only 30%. Most of the benefit arrives with the last few layers.

### Mixture-of-experts models split better

[Mixture-of-experts](/blog/mixture-of-experts/) (MoE) models change the math. Each token uses only a few experts, so most weights go unread on any given token. llama.cpp has flags for this: `--cpu-moe` keeps all the expert weights in system RAM, `--n-cpu-moe N` does it for the first N layers, and `-ot` (override tensor) places any tensor by name.

OpenAI's [gpt-oss-120b](https://arxiv.org/abs/2508.10925) shows why that works. It has 116.8 billion parameters, 114.7 billion of them in the feed-forward (MLP) layers that hold the experts, but only 5.1 billion are active per token. Its experts are stored in MXFP4, a 4-bit floating-point format with shared scales, at 4.25 bits per value. Put the experts in system RAM and the attention layers on the GPU, and each token reads only about 3.6 billion expert parameters, roughly 1.9 GB, from RAM. On 89.6 GB/s of DDR5, that's a ceiling of about 47 tokens per second for a 117B model. The experts still need about 61 GB of system RAM, but that RAM can be slow.

## Step 3: llama.cpp, GGUF and the quantization menu

[llama.cpp](https://github.com/ggml-org/llama.cpp) is the open-source engine under most local LLM tools. Its README states the goal as "LLM (and VLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware," with a "Plain C/C++ implementation without any dependencies." It treats Apple silicon as "a first-class citizen" through Metal, and its backends include CUDA for NVIDIA GPUs, HIP for AMD GPUs, Vulkan for most GPUs, SYCL for Intel GPUs and optimized CPU code for x86 and Arm.

### GGUF: the model in one file

llama.cpp reads models in **GGUF**, a format defined in the [ggml docs](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). The spec's goals include "single-file deployment" and "full information: all information needed to load a model is contained in the model file." It's also built for `mmap`, which lets the operating system map the file straight into memory instead of copying it.

Dumping the header of the 4.92 GB file used here shows what that means. It starts with the four bytes `GGUF`, then 36 key-value metadata entries and 292 tensors. The metadata holds the architecture (32 layers, 32 query heads, 8 key-value heads, a 131,072-token context), the full tokenizer with its 128,256 tokens, and the chat template that formats a conversation for the model. A typical Hugging Face checkpoint keeps the configuration, tokenizer and weights in separate files; in GGUF, one file is all you download.

### K-quants: why Q4_K_M is 4.89 bits

GGUF quantization types have names like Q4_K_M. The number is the main bit width, K marks the **k-quant** family, and the last letter is the mix: S, M or L for small, medium and large. In the [pull request that introduced k-quants](https://github.com/ggml-org/llama.cpp/pull/1684), Q4_K stores weights in "super-blocks containing 8 blocks, each block having 32 weights," with each block's scale and minimum themselves stored in 6 bits, for 4.5 bits per weight. These are [Part 8](/blog/quantization/)'s scales and groups, nested one level deeper.

The M mix then spends extra bits where they matter most: Q4_K_M uses Q6_K "for half of the `attention.wv` and `feed_forward.w2` tensors." The dump of the test file shows it, with layer 0's value projection and feed-forward output in Q6_K and the rest in Q4_K. That's how Q4_K_M lands at 4.89 bits per weight, not 4.5.

The menu for Llama 3.1 8B, with bits per weight and sizes from llama.cpp's [quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md). The quality column is perplexity on the original LLaMA 7B, from the k-quants pull request (2023). Perplexity measures how surprised a model is by real text; lower is better.

| Type | Bits per weight | Llama 3.1 8B size | LLaMA 7B perplexity (F16 = 5.9066) |
|---|---|---|---|
| F16 | 16.00 | 14.96 GiB | 5.9066 |
| Q8_0 | 8.50 | 7.95 GiB | Not in the table |
| Q6_K | 6.56 | 6.14 GiB | 5.9110 (+0.07%) |
| Q5_K_M | 5.70 | 5.33 GiB | 5.9208 (+0.24%) |
| Q4_K_M | 4.89 | 4.58 GiB | 5.9601 (+0.91%) |
| IQ4_XS | 4.46 | 4.17 GiB | Not in the table |
| Q3_K_M | 4.00 | 3.74 GiB | 6.1503 (+4.1%) |
| Q2_K | 3.16 | 2.95 GiB | 6.7764 (+14.7%) |

The pull request notes that the "6-bit quantized perplexity is within `0.1%` or better from the original `fp16` model." Q4_K_M costs under 1%, and quality drops quickly below 4 bits. That's why Q4_K_M is the usual default.

The **IQ** types (i-quants) push lower, down to about 2 bits. They lean on an **importance matrix** (imatrix), statistics collected by running sample text through the model, so rounding protects the weights that matter most. The quantize README says accuracy loss "can be minimized by using a suitable imatrix file." The test file's metadata records that its quantizer used one.

### Other flags worth knowing

- `-c 8192` sets the context size, and `--cache-type-k q8_0 --cache-type-v q8_0` stores the KV cache in 8-bit blocks, about 8.5 bits per value instead of 16, so Llama 3.1 8B's 128K-token cache shrinks from 16 GiB to about 8.5 GiB. The allowed types go down to `q4_0`.
- Recent builds default `-ngl` to `auto` and turn on `--fit`, which adjusts unset arguments "to fit in device memory."
- `llama-server` supports [speculative decoding](/blog/speculative-decoding/), with a small draft model given by `-md` or self-drafting types chosen with `--spec-type`, including `draft-eagle3`, `draft-mtp` and n-gram lookup. It suits local use: with one user, the GPU's arithmetic mostly waits on memory, and that idle arithmetic is what speculative decoding puts to work.

## Step 4: running one yourself, and where laptops struggle

You don't need to compile anything.

[Ollama](https://github.com/ollama/ollama) wraps model download, storage and serving into one command. It has a model library, its own HTTP API and an OpenAI-compatible API (application programming interface) at `http://localhost:11434/v1/`. A **Modelfile**, "the blueprint to create and share customized models," sets a base model (or a local GGUF file), parameters and a system prompt. Ollama 0.30 added "GGUF model compatibility through llama.cpp," which "augments Ollama's MLX engine on Apple silicon." Its FAQ is direct about privacy: "Ollama runs locally. We don't see your prompts or data when you run locally."

[LM Studio](https://lmstudio.ai/docs/app) is a desktop app for Mac, Windows and Linux with a model browser and chat window. It runs GGUF models through llama.cpp, and "on Apple Silicon Macs, LM Studio also supports running LLMs using Apple's MLX." Its local server speaks the OpenAI API at `http://localhost:1234/v1`.

[MLX](https://github.com/ml-explore/mlx) is Apple's own "array framework for machine learning on Apple silicon," in which "arrays in MLX live in shared memory." [mlx-lm](https://github.com/ml-explore/mlx-lm) runs and fine-tunes LLMs on it. Apple's own footnotes for its MacBook Pro LLM benchmarks name mlx-lm as the software used.

### The commands

With Ollama:

```bash
ollama run llama3.1:8b          # downloads the 4.9 GB Q4_K_M build, then opens a chat
ollama ps                       # shows whether it loaded on the GPU, the CPU, or split
```

With llama.cpp's prebuilt binaries from its [releases page](https://github.com/ggml-org/llama.cpp/releases):

```bash
# serve a local GGUF file: 8,192-token context, every layer on the GPU
llama-server -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 8192 -ngl 99 --port 8080

# or fetch a model from Hugging Face by name
llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 8192 --port 8080

# measure your own machine: prompt reading (pp) and generation (tg) speed
llama-bench -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -p 512,4096 -n 128 -ngl 99
```

On a Mac, with mlx-lm:

```bash
pip install mlx-lm
mlx_lm.generate --prompt "How tall is Mt Everest?"
```

Once `llama-server` is running, any OpenAI-style client can talk to `http://localhost:8080/v1/chat/completions`. On the test laptop, a 49-token question came back at 43.8 tokens per second.

### Measured on the test laptop

`llama-bench` results for Llama 3.1 8B Instruct Q4_K_M (4.92 GB) on the hardware described in Step 2:

| Where it ran | Bandwidth | Ceiling | Measured generation | Prompt reading, 4,096 tokens | Time to read a 4,096-token prompt |
|---|---|---|---|---|---|
| CPU only, 8 threads | 89.6 GB/s | 18.2 tokens/s | 10.9 tokens/s | 50 tokens/s | 82 s |
| RTX 4060 Laptop GPU | 256 GB/s | 52 tokens/s | 44.7 tokens/s | 1,813 tokens/s | 2.3 s |

The GPU number lands close to its ceiling. The driver reports a 128-bit memory bus at up to 8,001 MHz, which works out to 256 GB/s. During a repeat run, the memory clock sat at 7,000 MHz, about 224 GB/s, while the memory controller reported 100% utilization and the card drew about 47 W against a 50 W limit. Against that lower figure, 44.7 tokens per second is essentially the ceiling. The CPU reached about 60% of its theoretical bandwidth.

### Prefill: where laptops struggle most

The biggest gap in that table is in reading the prompt. Decode is memory-bound, so the GPU was 4.1 times faster than the CPU, in line with the memory bandwidth each actually achieved (about 220 against 54 GB/s). **Prefill**, reading the prompt in one pass, is compute-bound, as [Part 1](/blog/how-llm-inference-works/) showed: about 2 floating-point operations per parameter per token. For a 4,096-token prompt on an 8B model, that's 2 × 8.03 billion × 4,096 ≈ 66 trillion operations. The CPU managed about 0.8 trillion per second, and the GPU about 29 trillion. That's a 36-fold gap, and it decides how long you wait for the **time to first token** (TTFT).

That's why pasting a long document or codebase feels slow locally even when chat feels fast. Apple added "a Neural Accelerator in each GPU core" of the M5 family and says it speeds up "large language model (LLM) prompt processing." Its MacBook Pro page claims the M5 Max reaches time to first token 6.7 times faster than the M1 Max, against 1.7 times for the M4 Max, which works out to about 3.9 times faster than the M4 Max on a 16K-token prompt. [Long-context inference](/blog/long-context-inference/) covers why long prompts cost more still.

### Batch of one

A data center GPU earns its keep by serving many users per weight read; [Part 1](/blog/how-llm-inference-works/) showed 64 users sharing one step. Your laptop serves one person, so every token pays the full cost of reading the weights. Per token, local inference is rarely the efficient option; it makes sense when you already own the hardware or when privacy is the point. The test GPU used about 1 joule per token (47 W ÷ 44.7 tokens per second), or about 0.3 kWh of GPU power per million tokens, before counting the rest of the laptop.

### Choosing a model and quantization

Sizes below are weights only, at Q4_K_M unless noted (4.89 bits per weight, or the published file size). Leave room for the KV cache and for the operating system; as a rule of thumb, plan on a few gigabytes of headroom.

| Memory available | Comfortable choices | Weights | Notes |
|---|---|---|---|
| 8 GB | 3B to 4B models; 8B with short context | about 2.4 GB for 4B; 4.9 GB for 8B | An 8K-token cache adds 1 GiB for Llama 3.1 8B |
| 16 GB | 8B at Q8_0, 14B, or gpt-oss-20b | 8.5 GB; 8.6 GB; 13.7 GB | OpenAI says gpt-oss-20b runs "with as little as 16GB memory" |
| 32 GB | 32B | about 19.6 GB | Room for a long context with an 8-bit cache |
| 64 GB | 70B | 43.1 GB | 32K tokens of 16-bit cache adds 10 GiB for Llama 3.1 70B |
| 128 GB | 70B at Q8_0, or gpt-oss-120b | about 74 GB; 65.3 GB | MoE models run much faster than their size suggests |
| 512 GB | 671B MoE models, such as DeepSeek-V3 at about 4.9 bits | about 410 GB | Only 37B parameters are active per token |

For speed, look up your bandwidth in Step 1's table. On a Mac, macOS limits how much memory the GPU may claim; [mlx-lm's README](https://github.com/ml-explore/mlx-lm) shows how to raise it with `sudo sysctl iogpu.wired_limit_mb=N`. Ollama also picks a default context from your VRAM: 4K tokens under 24 GiB, 32K from 24 to 48 GiB, and 256K at 48 GiB and above, per its [context-length docs](https://docs.ollama.com/context-length).

## In the real world

- Apple benchmarks its Macs on LLMs: its MacBook Pro page measures time to first token on a 16K-token prompt with a 14B 4-bit model in mlx-lm, and its [August 2026 Mac Studio announcement](https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/) says the M5 Ultra's 512 GB lets users "run enormous LLMs entirely on device."
- DeepSeek-V3 has [671 billion parameters with 37 billion active per token](https://arxiv.org/abs/2412.19437). At Q4_K_M's 4.89 bits, the weights would take about 410 GB, which fits in an M5 Ultra's 512 GB. Each token would read about 22.6 GB, for a ceiling near 53 tokens per second at 1.2 TB/s.
- Ollama's [March 2026 MLX preview](https://ollama.com/blog/mlx) reported Qwen3.5-35B-A3B decoding at 112 tokens per second with MLX and an NVFP4 model, against 58 on its previous engine with a Q4_K_M model, and prefill at 1,810 against 1,154 tokens per second. It didn't name the chip, and it recommended "a Mac with more than 32GB of unified memory."
- OpenAI shipped gpt-oss-20b at 12.8 GiB and gpt-oss-120b at 60.8 GiB, with MXFP4 experts. Ollama lists them as 14 GB and 65 GB downloads.
- llama.cpp's quantize README reports Llama 3.1 8B generating 29.2 tokens per second in F16 and 71.9 at Q4_K_M on its test machine, 2.5 times faster for a file 3.3 times smaller.

## Common misconceptions

- **"A faster GPU always runs models faster."** Only if the model fits in its memory. An RTX 5090 has three times the bandwidth of an M5 Max, but a 70B Q4_K_M model spills out of its 32 GB, and the part in system RAM drags it to a ceiling near 6 tokens per second, against 14 for the Mac.
- **"128 GB of memory means big models run fast."** Capacity decides what fits; bandwidth decides speed. The DGX Spark and Ryzen AI Max+ 395 hold a 70B model easily, but at 256 to 273 GB/s it tops out near 6 tokens per second.
- **"Offloading half the layers to the GPU doubles the speed."** The slow half still sets the pace. On the test laptop, half the layers on a GPU with three times the rated bandwidth gave a 30% speedup.
- **"If chat is fast, long documents will be fast too."** Generation is limited by bandwidth; reading the prompt is limited by compute. The test laptop's CPU generated at a usable 10.9 tokens per second but needed 82 seconds to read a 4,096-token prompt.
- **"A 4-bit model is a quarter of the 16-bit size."** Q4_K_M uses 4.89 bits per weight, because of its block scales and the layers it keeps at 6 bits. Llama 3.1 8B is 4.58 GiB at Q4_K_M, about 31% of its 14.96 GiB F16 size.

## Key takeaways

- Your machine's speed limit for one user is memory bandwidth divided by the bytes read per token. A 4.9 GB model on 90 GB/s of DDR5 tops out near 18 tokens per second, and on a 614 GB/s M5 Max near 125.
- Memory size decides what fits: weights, plus the KV cache, plus the operating system. Unified memory on Apple silicon, and the shared pools of the Ryzen AI Max+ 395 and the DGX Spark, give the GPU 128 GB or more to work with.
- When a model doesn't fit in VRAM, the part left in system RAM sets the pace. Mixture-of-experts models split far better, because each token reads only its active experts.
- GGUF packs the weights, tokenizer and chat template into one file. Q4_K_M, at 4.89 bits per weight, cost LLaMA 7B under 1% in perplexity and is the usual default.
- Reading a long prompt is compute-bound, and that's where laptops struggle most: 82 seconds for 4,096 tokens on a laptop CPU, 2.3 seconds on its GPU.

## What's next

This part shrank inference down to one machine and one user. The final part goes the other way: [Part 14](/blog/deepseek-inference-deepinfer/) is a case study of DeepSeek's inference stack and its DeepInfer engine, where MLA, expert parallelism, FP8 kernels, multi-token prediction and [disaggregated serving](/blog/disaggregated-serving-llm-d/) all run together at production scale.

## Further reading

- [llama.cpp (ggml-org)](https://github.com/ggml-org/llama.cpp): the engine, its backends and prebuilt binaries.
- [GGUF specification (ggml docs)](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): the file layout, metadata keys and naming convention.
- [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md): every quantization type with bits per weight, sizes and speeds for Llama 3.1 8B.
- [k-quants pull request #1684 (Iwan Kawrakow, 2023)](https://github.com/ggml-org/llama.cpp/pull/1684): how the Q2_K to Q6_K formats and their S, M and L mixes are built, with perplexity results.
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): every `llama-server` flag, including offload, KV cache types and speculative decoding.
- [gpt-oss-120b and gpt-oss-20b Model Card (OpenAI, 2025)](https://arxiv.org/abs/2508.10925): total and active parameter counts and MXFP4 checkpoint sizes.
- [Apple unleashes M1 (Apple, 2020)](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/): Apple's description of unified memory.
- [Apple debuts M5 Pro and M5 Max (Apple, 2026)](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/) and the [Mac Studio](https://www.apple.com/mac-studio/specs/) and [MacBook Pro](https://www.apple.com/macbook-pro/specs/) spec pages: memory sizes and bandwidth.
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) and [AMD Ryzen AI Max+ 395](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html): official specs for two unified-memory alternatives to a Mac.
- [Ollama docs](https://docs.ollama.com/faq): FAQ, [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [Modelfile reference](https://docs.ollama.com/modelfile) and [context length](https://docs.ollama.com/context-length).
- [LM Studio docs](https://lmstudio.ai/docs/app): the desktop app, its llama.cpp and MLX engines, and its OpenAI-compatible server.
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm): Apple's array framework and its LLM package.
