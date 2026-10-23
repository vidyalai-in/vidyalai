---
title: "Quantization: Running LLMs in 8 and 4 Bits"
description: "How LLMs run in 8 or 4 bits: scales and outliers, why decode speeds up, GPTQ, AWQ, SmoothQuant, FP8, FP4, KV cache quantization and checking quality."
slug: "quantization"
category: "inference-optimization"
tags: ["quantization", "gpu", "latency", "cost", "serving"]
series: "llm-inference"
seriesOrder: 8
pubDate: 2026-10-23
---

## TL;DR

Quantization stores a model's numbers in fewer bits: 8 or 4 instead of 16. Each group of numbers gets a scale factor, and each number is rounded to one of a small set of levels: at most 256 for 8 bits and 16 for 4 bits. Because decode is limited by how many bytes it reads from memory, halving the bytes roughly doubles the speed limit: Llama 3.1 8B on an H100 goes from about 209 tokens per second in 16 bits to about 417 in 8 bits and about 809 in 4 bits. The hard part is a handful of unusually large values, called outliers, which methods like AWQ, LLM.int8() and SmoothQuant each handle differently. By the end you'll know what the main formats are, why they speed things up (and when they don't), and how to check a quantized model still gives good answers.

## Why it matters

Llama 3.1 70B doesn't fit on an 80 GB NVIDIA H100 in its native 16-bit form: its weights alone take 141 GB. Store each weight in 4 bits and the same model takes about 36 GB, with room left over for users.

OpenAI went further with gpt-oss-120b. Its 116.8 billion parameters would take about 234 GB in 16 bits. OpenAI shipped the checkpoint at [60.8 GiB](https://arxiv.org/abs/2508.10925), small enough to run on a single 80 GB GPU, by storing more than 90% of its weights in a 4-bit format.

Fewer bits also means faster answers, because [Part 1](/blog/how-llm-inference-works/) showed that every decode step reads every weight from memory. [Part 7](/blog/flashattention/) cut the bytes attention moves; this part cuts the bytes of the numbers themselves, and shows how to check the model survives.

## The intuition

Think of a tape measure. A carpenter's tape marked every millimeter measures a 2-meter plank to the nearest millimeter. Give the same carpenter a tape with only 16 marks along its length, and every measurement snaps to the nearest mark, about 13 centimeters apart. That's 4-bit quantization: 16 allowed values, and every number rounds to one of them.

What makes it usable is choosing the tape for the job. To measure a bookshelf you'd pick a short tape, so its 16 marks are close together. Quantization does the same with a **scale**: each block of numbers gets a tape just long enough to reach its largest value, so the marks are as close as they can be.

The analogy breaks in two places. A carpenter's errors stay in one plank, but a model's rounding errors pass through dozens of layers and can add up. And a single very long plank (an outlier) forces a long tape on everything measured with it, and most of the methods below exist to deal with that.

## Step 1: levels, scales and outliers

A model's **weights** are its learned numbers, stored in [BF16](/blog/gpus-for-llms/) (16-bit brain floating point) by default. Its **activations** are the numbers computed as a token passes through the layers. Quantizing either one means mapping it to a small integer or low-precision float, plus a scale that converts it back.

The simplest recipe is **symmetric absmax** quantization (absmax means the largest absolute value):

```text
scale        = largest |value| ÷ largest integer   (127 for INT8, 7 for INT4)
integer      = round(value ÷ scale)
dequantized  = integer × scale
```

Here's a group of eight weights, quantized to INT8 (8-bit integers, levels −127 to 127) and INT4 (4-bit integers, levels −7 to 7). The largest absolute value is 1.27, so the INT8 scale is 1.27 ÷ 127 = 0.01 and the INT4 scale is 1.27 ÷ 7 ≈ 0.181:

| Weight | INT8 integer | INT8 dequantized | INT4 integer | INT4 dequantized |
|---|---|---|---|---|
| 0.423 | 42 | 0.42 | 2 | 0.363 |
| −1.270 | −127 | −1.27 | −7 | −1.270 |
| 0.081 | 8 | 0.08 | 0 | 0.000 |
| 0.914 | 91 | 0.91 | 5 | 0.907 |
| −0.337 | −34 | −0.34 | −2 | −0.363 |
| 0.562 | 56 | 0.56 | 3 | 0.544 |
| −0.708 | −71 | −0.71 | −4 | −0.726 |
| 1.049 | 105 | 1.05 | 6 | 1.089 |

The average weight here is 0.668 in size. INT8's average error is 0.002, about 0.3% of that. INT4's is 0.031, about 4.7%, and the small weight 0.081 rounds all the way to zero. The same weights in FP8 E4M3 (an 8-bit float from Part 3), scaled so 1.27 maps to its maximum of 448, have an average error of 0.013, about 1.9%. INT8 wins here because its 255 levels are evenly spaced, while FP8 spends its bits on range: fine steps near zero, coarse steps near the top.

In code:

```python
import numpy as np

w = np.array([0.423, -1.270, 0.081, 0.914, -0.337, 0.562, -0.708, 1.049])
for bits in (8, 4):
    qmax = 2 ** (bits - 1) - 1                 # 127 or 7
    scale = np.abs(w).max() / qmax             # 0.01 or 0.181
    q = np.clip(np.round(w / scale), -qmax, qmax)
    print(bits, q.astype(int), np.abs(q * scale - w).mean())   # mean error 0.0020, 0.0311
```

### Zero points: when the values are lopsided

Symmetric quantization wastes levels when the values aren't centered on zero. If a group runs from −0.4 to 1.1, symmetric INT4 needs a scale of 1.1 ÷ 7 ≈ 0.157, and the levels below −0.4 go unused. **Asymmetric** quantization adds a **zero point**, an integer offset, so the 16 levels cover exactly the range:

```text
scale      = (1.1 − (−0.4)) ÷ 15 = 0.1
zero point = 4                     (so −0.4 maps to 0 and 1.1 maps to 15)
integer    = round(value ÷ 0.1) + 4
```

The step between levels shrinks from 0.157 to 0.1. The cost is one more stored number per group and a subtraction during dequantization.

### One outlier, and why scales get finer

Now change the fourth weight from 0.914 to 9.0, a single **outlier**. With one scale for the whole tensor (**per-tensor** scaling), the INT4 scale becomes 9.0 ÷ 7 ≈ 1.286, and the ordinary weights collapse:

| Scheme | Scale(s) | Integers | Average error on the 7 ordinary weights |
|---|---|---|---|
| Per-tensor INT4 | 1.286 | 0, −1, 0, 7, 0, 0, −1, 1 | 0.319 (about 50% of their size) |
| Groups of 4, INT4 | 1.286 and 0.150 | 0, −1, 0, 7, −2, 4, −5, 7 | 0.091 (about 14%) |

Per-tensor, four of the eight weights round to zero. Splitting into two groups of four, each with its own scale, rescues the group without the outlier: its average error drops to 0.029. Only the outlier's own group still suffers.

That's why real schemes use finer scales:

- **Per-channel**: one scale per row of the weight matrix (one output channel). Cheap and common for 8-bit.
- **Per-group**: one scale per run of consecutive weights, typically 128. A 16-bit scale per 128 weights adds 16 ÷ 128 = 0.125 bits per weight, so INT4 with group size 128 costs about 4.125 bits per weight.

Outliers in real models are worse than in this toy. Tim Dettmers and colleagues found in [LLM.int8()](https://arxiv.org/abs/2208.07339) that activations contain "large features with magnitudes up to 20x larger than in other dimensions." In the models they studied, "at around 6.7B parameters, a phase shift occurs": these outliers show up in every layer, yet they're concentrated in only 6 feature dimensions. Setting those dimensions to zero degraded perplexity by 600 to 1,000%, so they can't simply be rounded away.

## Step 2: why fewer bits make decode faster, and when they don't

[Part 1](/blog/how-llm-inference-works/) worked out that a single user's decode step reads every weight once, so the speed limit is bytes divided by bandwidth. For Llama 3.1 8B (8.03 billion parameters) on an H100 at 3.35 TB/s:

| Weight format | Bits per weight | Bytes per step | Step time | Ceiling, one user |
|---|---|---|---|---|
| BF16 | 16 | 16.06 GB | 4.79 ms | 209 tokens/s |
| FP8 or INT8 | 8 | 8.03 GB | 2.40 ms | 417 tokens/s |
| INT4, group size 128 | 4.125 | 4.14 GB | 1.24 ms | 809 tokens/s |

These ceilings assume every weight is quantized. Real checkpoints often keep the embedding and output layers in 16 bits, so their files are somewhat larger.

Measured numbers follow the same direction. Red Hat's [quantization study](https://arxiv.org/abs/2411.02355) served Llama 3.1 8B in vLLM on one NVIDIA A6000, one request at a time, on a code-completion task. Latency fell from 24.5 seconds in BF16 to 15.9 seconds with 8-bit weights and activations, and to 9.7 seconds with 4-bit weights: 2.5 times faster. For Llama 3.1 70B on H100s, 4-bit weights on one GPU (28.6 seconds) beat BF16 spread over two (31.3 seconds).

### Room for the KV cache

Smaller weights also free memory for the KV cache, the stored keys and values from [Part 2](/blog/kv-cache/). Llama 3.1 70B in FP8 leaves 9.4 GB free on an H100, room for 7 users at 4,096 tokens with a BF16 cache, as [Part 3](/blog/gpus-for-llms/) computed. Quantize the cache to FP8 too, and the same 9.4 GB holds 14. More users per step means more tokens per weight read, the batching effect from Part 1.

### When quantization doesn't speed things up

Bytes are only the bottleneck when the work is memory-bound. Two cases break the rule:

- **Prefill.** Prefill (reading the prompt) is compute-bound. If only the weights are quantized, the multiplications still run in BF16, so prefill gets no faster. In the Red Hat paper's words, "Weight quantization primarily accelerates decode by reducing memory movement, whereas weight-and-activation quantization improves computational efficiency in prefill."
- **Dequantization overhead.** 4-bit weights must be unpacked and multiplied by their scales before BF16 math can use them; the GPTQ authors note that "dequantization consumes extra compute." Under heavy load, when decode is no longer purely memory-bound, the weight-only advantage shrinks. In the same study's throughput test on one A6000, 8B with INT8 weights and activations ran 1.38 times faster than BF16, but with 4-bit weights only 1.08 times.

## Step 3: weight-only quantization

There are two ways to run a quantized model, depending on whether the activations are quantized too:

![A flowchart. Offline, once: BF16 weights are calibrated on sample prompts and quantized to integers plus scales, then stored in HBM at 4 or 8 bits per weight. From there, two paths. Weight-only, for example W4A16: 4-bit weights are loaded into SRAM, dequantized to BF16 on the chip, and multiplied with BF16 activations in a BF16 matmul. Weights and activations, for example W8A8: activations are quantized to FP8 or INT8, multiplied with the 8-bit weights in an FP8 or INT8 matmul on the tensor cores, and the result is multiplied by the scales and output in BF16.](/diagrams/i08-quantized-matmul.svg "Weight-only quantization saves bytes; quantizing activations as well also uses the faster low-precision tensor cores.")

Both paths read fewer bytes from HBM; only the second also does its arithmetic in fewer bits.

The shorthand **W4A16** means 4-bit weights and 16-bit activations; **W8A8** means 8 bits for both. Weight-only quantization is the easier problem, since weights are fixed and can be studied offline. The methods here are **post-training quantization**: they take a finished model and a small **calibration set** of sample texts, used to see what typical activations look like, with no retraining.

### GPTQ: fix each rounding error as you go

Rounding each weight to the nearest level (round-to-nearest) treats every error as independent. **GPTQ** ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323)) quantizes the weights of a layer one column at a time, and after each step it adjusts the not-yet-quantized weights to cancel the error just introduced. It decides how to adjust them from second-order information, a measure of how the layer's output changes with each weight, estimated from the calibration activations.

The paper reports that GPTQ "can quantize GPT models with 175 billion parameters in approximately four GPU hours," down to 3 or 4 bits "with negligible accuracy degradation." Generating one token at a time, a 3-bit OPT-175B on one A100 ran 3.24 times faster per token than FP16 spread over five, and 4.5 times faster on two A6000s against eight (589 ms down to 130 ms per token).

### AWQ: protect the weights that matter

**AWQ**, activation-aware weight quantization ([Lin et al., 2023](https://arxiv.org/abs/2306.00978)), starts from an observation: "Protecting only 1% salient weights can greatly reduce quantization error." The salient weights are the ones multiplied by large activations, so AWQ finds them from activation statistics, not from the weights themselves.

Keeping 1% of weights in 16 bits would need awkward mixed-precision kernels. Instead, AWQ multiplies each salient channel by a factor greater than 1 before quantizing, and divides the matching activations by the same factor, so the layer computes the same result while those weights lose less to rounding. On OPT-6.7B at 3 bits with group size 128, WikiText perplexity was 10.86 in FP16, 23.54 with round-to-nearest, and 11.39 with AWQ. AWQ won the best paper award at MLSys 2024.

The Red Hat study tuned both methods and found them nearly identical on academic benchmarks, but "GPTQ exhibits notable gains over AWQ in real-world tasks, particularly coding."

### NF4 and bitsandbytes

**NF4** (4-bit NormalFloat), from the [QLoRA paper](https://arxiv.org/abs/2305.14314), spaces its 16 levels so each holds an equal share of a bell-shaped distribution, which the paper calls "information theoretically optimal for normally distributed weights." It ships in the bitsandbytes library alongside LLM.int8() and is mostly used to fine-tune on a quantized base: [LoRA fine-tuning with 5,000 examples](/blog/lora-fine-tuning-5k-examples/) works through the memory math.

## Step 4: quantizing activations and the KV cache

Activations are harder than weights: they change with every input, and they carry the outliers LLM.int8() found. The payoff is faster math: an H100 does about 989 trillion dense BF16 operations per second on its tensor cores, and twice that in FP8 or INT8.

### SmoothQuant: move the difficulty into the weights

**SmoothQuant** ([Xiao et al., 2022](https://arxiv.org/abs/2211.10438)) starts from two facts. Activation outliers are "∼100× larger than most of the activation values," and they "persist in fixed channels." Weights, meanwhile, are flat and easy to quantize. So SmoothQuant divides each activation channel by a smoothing factor `s` and multiplies the matching weights by the same `s`. The product doesn't change, because `(X ÷ s) × (s × W) = X × W`.

The paper sets the factor per channel as `s = max|X|^α ÷ max|W|^(1−α)` and calls α = 0.5 "a well-balanced point." With α = 0.5, both sides end up with the same maximum. Take a channel whose activations reach 64 and whose weights reach 0.5:

```text
s                 = √64 ÷ √0.5 ≈ 11.3
activation max    = 64 ÷ 11.3  ≈ 5.66
weight max        = 0.5 × 11.3 ≈ 5.66
```

The factor is folded into the previous layer's weights offline, so it costs nothing at run time. With every matrix multiplication in INT8 (W8A8), SmoothQuant reported up to 1.56 times faster inference than FP16 for OPT-30B on one A100-80GB, about half the memory, and a 530B model served within one 8-GPU node.

### FP8: the default on Hopper and Blackwell

FP8 needs less help than INT8, because its exponent bits give it range. Inference uses E4M3, which reaches ±448 ([Part 3](/blog/gpus-for-llms/)), with a scale mapping each tensor into that range. vLLM's [FP8 W8A8 scheme](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/) uses "static, per-channel quantization on the weights" and "dynamic, per-token quantization on the activations," computing each token's scale on the fly. The docs promise "a 2x reduction in model memory requirements" and "up to a 1.6x improvement in throughput with minimal impact on accuracy," and note that FP8 math needs compute capability 8.9 or newer (Ada Lovelace, Hopper, Blackwell). Older Ampere GPUs can still load FP8 weights, but only as weight-only W8A16.

### FP4: NVFP4 and MXFP4 on Blackwell

FP4 E2M1 has only 8 magnitudes (0, 0.5, 1, 1.5, 2, 3, 4 and 6), so it depends entirely on small groups:

- **MXFP4** comes from the [OCP Microscaling (MX) specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) of 2023, written by Microsoft, AMD, Arm, Intel, Meta, NVIDIA and Qualcomm. Every 32 values share one 8-bit scale of the form E8M0, a pure power of two: 4 + 8 ÷ 32 = 4.25 bits per value.
- **NVFP4**, [NVIDIA's format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), uses blocks of 16 with an FP8 E4M3 scale, plus one FP32 scale per tensor: 4.5 bits per value. Smaller blocks and non-power-of-two scales fit each block's range more closely. NVIDIA reports "1% or less accuracy degradation" on key language tasks when quantizing DeepSeek-R1-0528 from FP8 to NVFP4.

Blackwell's tensor cores handle these block scales in hardware, so FP4 speeds up the math as well as the memory: 9 dense PFLOPS per B200, double its FP8 rate. gpt-oss is the best-known example: OpenAI "post-trained the models with quantization of the MoE weights to MXFP4 format," at 4.25 bits per parameter.

### Quantizing the KV cache

The KV cache is the other big read in every decode step, and in Part 1's example of 64 users at 4,096 tokens it was more than twice the size of the weights. vLLM's `kv_cache_dtype="fp8"` [option](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html) stores it in FP8, with one scale per K and V tensor. Without calibration, "all quantization scales are set to `1.0`," so calibrated scales are recommended. The docs also warn that some layers, such as sliding-window attention, "are more sensitive to KV-cache quantization."

Put FP8 weights and an FP8 cache together, and Part 1's 64-user step shrinks:

```text
BF16 weights + BF16 cache:  (16.06 + 34.36) GB ÷ 3.35 TB/s ≈ 15.1 ms → about 4,250 tokens/s
FP8 weights + FP8 cache:    ( 8.03 + 17.18) GB ÷ 3.35 TB/s ≈  7.5 ms → about 8,500 tokens/s
```

Research pushes the cache further. **KIVI** ([Liu et al., 2024](https://arxiv.org/abs/2402.02750)) stores keys and values in 2 bits. Its analysis found keys have outliers in fixed channels and values don't, so it quantizes keys per channel and values per token, in groups of 32, and keeps the most recent 128 tokens at full precision. On Llama-2-7B on one A100 80GB, it used 2.6 times less peak memory, allowed up to 4 times larger batches and delivered 2.35 to 3.47 times the throughput.

## Step 5: checking the model still gives good answers

Three kinds of check, from cheapest to most telling:

- **Perplexity.** Perplexity measures how surprised a model is by real text: the exponential of the average negative log-probability it assigned to each actual next token. Lower is better. It's quick and sensitive, and it's the yardstick GPTQ and AWQ report, as in AWQ's 10.86 against 23.54 above. llama.cpp's docs also use KL divergence, which compares the quantized model's full next-token probabilities with the original's.
- **Task benchmarks.** Knowledge and reasoning tests (MMLU, GSM8K, GPQA), code (HumanEval), long context (RULER) and chat quality judged by another model (Arena-Hard). These catch damage perplexity misses.
- **Your own evals.** The tasks your users actually run, with the same prompts and sampling settings, compared side by side with the unquantized model. [Building an eval suite for a support bot](/blog/support-bot-eval-suite/) shows how to build one.

### What the evidence says

The Red Hat team ran over 500,000 evaluations on the Llama 3.1 family at 8B, 70B and 405B. Their summary: "FP8 (W8A8-FP) is effectively lossless across all model scales," "well-tuned INT8 (W8A8-INT) achieves surprisingly low (1-3%) accuracy degradation," and "INT4 weight-only (W4A16-INT) is more competitive than expected, rivaling 8-bit quantization." Average score recovery, the quantized score as a percentage of BF16's, on the harder Open LLM Leaderboard V2:

| Model | FP8 (W8A8) | INT8 (W8A8) | INT4 (W4A16) |
|---|---|---|---|
| Llama 3.1 8B Instruct | 101.2% | 101.5% | 96.1% |
| Llama 3.1 70B Instruct | 100.0% | 97.3% | 97.4% |
| Llama 3.1 405B Instruct | 99.9% | 98.3% | 98.9% |

Small models and hard reasoning are where losses show. On reasoning benchmarks (AIME, MATH-500, GPQA-Diamond), the same study found DeepSeek-R1-Distill-Llama-70B in 4 bits kept 98.3% of its score, while the 1.5B Qwen distill kept 93.5%. A separate study, [Quantization Hurts Reasoning?](https://arxiv.org/abs/2504.04823) (Liu et al., 2025), found 4-bit weights cost a 1.5B distilled Qwen model 2.1% against 0.4% for the 32B, and that "harder tasks (e.g., AIME-120) suffer up to 4× greater degradation than simpler ones (e.g., GSM8K)." A 4-bit KV cache cost its larger models less than 1%, even on answers over 10,000 tokens long.

## In the real world

### The formats you'll meet

Memory for Llama 3.1 70B (70.55 billion parameters), assuming every weight is quantized:

| Format | Bits per weight | Llama 3.1 70B weights | Native math on |
|---|---|---|---|
| BF16 (baseline) | 16 | 141.1 GB | All recent data center GPUs |
| FP8 W8A8 | 8 | 70.6 GB | NVIDIA Ada, Hopper, Blackwell; AMD MI300X |
| INT8 W8A8 (SmoothQuant) | 8 | 70.6 GB | NVIDIA Turing through Hopper, and B200 |
| INT4 W4A16 (GPTQ, AWQ), group 128 | about 4.125 | 36.4 GB | None: dequantized to 16-bit in the kernel |
| MXFP4 | 4.25 | 37.5 GB | NVIDIA Blackwell |
| NVFP4 | 4.5 | 39.7 GB | NVIDIA Blackwell |
| GGUF Q4_K_M | about 4.89 | 43.1 GB | None: dequantized in the kernel |

Hardware support comes from [vLLM's quantization table](https://docs.vllm.ai/en/latest/features/quantization/) and NVIDIA's spec pages; GPUs without native math for a format can often still run it weight-only. The Q4_K_M size is from [llama.cpp's quantize docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md); GGUF, the file format of llama.cpp, is the usual way to run models on a laptop or Mac, and Part 13 covers it.

NVIDIA's newest chips favor floating point. Its [HGX page](https://www.nvidia.com/en-us/data-center/hgx/) lists 72 POPS of INT8 (with sparsity) for an 8-GPU B200 system, but only 3 POPS for the newer B300, while dense FP4 rises from 72 to 108 PFLOPS.

### In production

- **gpt-oss.** The MoE (mixture-of-experts) weights, "responsible for 90+% of the total parameter count," ship in MXFP4. The [model card](https://arxiv.org/abs/2508.10925) says this lets gpt-oss-120b "fit on a single 80GB GPU" and gpt-oss-20b run "on systems with as little as 16GB memory."
- **DeepSeek-V3** was trained in FP8. Its [technical report](https://arxiv.org/abs/2412.19437) scales activations in 1×128 tiles and weights in 128×128 blocks, and reports training loss within 0.25% of BF16 when the recipe was validated on smaller models.
- **Fleet sizing.** [Serving Open-Weight LLMs at Scale](/blog/serving-open-llms-at-scale/) sizes its 70B fleets with FP8 weights and an FP8 KV cache, about 70 GB of weights and 160 KiB of cache per token.

## Common misconceptions

- **"Quantization always makes a model faster."** Weight-only 4-bit mostly helps memory-bound decode. Prefill still runs 16-bit math, and at high load the dequantization work eats into the gain: 1.08 times in one A6000 test, against 2.5 times for a single user.
- **"8-bit integers and 8-bit floats are the same thing."** They spend their bits differently. INT8 has 255 evenly spaced levels; FP8 E4M3 trades precision for range. On the eight weights above, INT8's average error was 0.3% and FP8's 1.9%, yet FP8's range copes better with values of very different sizes, and in the Red Hat study FP8 W8A8 was lossless where INT8 lost 1 to 3%.
- **"If perplexity barely moves, the model is fine."** Perplexity is one average over ordinary text. Accuracy losses vary by task: up to 4 times larger on AIME problems than on grade-school math in one study. Test on your own tasks.
- **"4 bits means a quarter of the memory."** Scales and zero points add bits: 4.125 for INT4 with groups of 128, 4.25 for MXFP4, 4.5 for NVFP4, about 4.89 for GGUF Q4_K_M. Layers kept in 16 bits add more.

## Key takeaways

- Quantization rounds each number to one of a few levels and stores a scale to undo it: at most 256 levels for 8 bits, 16 for 4.
- Outliers do the most damage. Finer scales (per channel, per group of 128, per block of 16 or 32) contain the damage, and methods like LLM.int8(), AWQ and SmoothQuant each give outliers special treatment.
- Fewer bytes per weight raise decode's ceiling: Llama 3.1 8B on an H100 goes from about 209 to 417 to 809 tokens per second as weights go from 16 to 8 to 4 bits.
- Weight-only formats like GPTQ and AWQ save memory and speed up decode. Quantizing activations too, in FP8 on Hopper or FP4 on Blackwell, also speeds up the math.
- FP8 is close to lossless on Llama 3.1. 4-bit is usually close, but small models and hard reasoning lose more, so always check with your own evals.

## What's next

Quantization shrinks every byte a decode step reads, but the model still produces one token per step. [Part 9](/blog/speculative-decoding/) covers speculative decoding: a small, fast model guesses several tokens ahead, and the big model checks all of them in a single pass, so one read of the weights can produce several tokens at once.

## Further reading

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323): 3- and 4-bit weights with error compensation, and the A100 and A6000 speedups.
- [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration (Lin et al., 2023)](https://arxiv.org/abs/2306.00978): protecting salient weights by scaling, MLSys 2024 best paper.
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., 2022)](https://arxiv.org/abs/2208.07339): the discovery of emergent outlier features and the mixed-precision fix.
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models (Xiao et al., 2022)](https://arxiv.org/abs/2211.10438): moving activation outliers into the weights for W8A8.
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314): NF4 and double quantization.
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (Liu et al., 2024)](https://arxiv.org/abs/2402.02750): per-channel keys, per-token values and a 2-bit cache.
- ["Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization (Kurtić et al., 2024)](https://arxiv.org/abs/2411.02355): FP8, INT8 and INT4 on the whole Llama 3.1 family, with vLLM speed and cost on A6000, A100 and H100.
- [Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models (Liu et al., 2025)](https://arxiv.org/abs/2504.04823): how model size and task difficulty change the damage.
- [OCP Microscaling Formats (MX) Specification v1.0 (Open Compute Project, 2023)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): the definition of MXFP4, MXFP8 and their shared E8M0 scales.
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference (NVIDIA, 2025)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/): NVFP4's two-level scaling and how it compares with MXFP4.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): FP8 training at scale with fine-grained scales.
- [gpt-oss-120b and gpt-oss-20b Model Card (OpenAI, 2025)](https://arxiv.org/abs/2508.10925): MXFP4 weights in a released model.
- [vLLM docs: quantization](https://docs.vllm.ai/en/latest/features/quantization/), [FP8 W8A8](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/) and [quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html): supported formats and hardware, and how scales are set in practice.
- [llama.cpp quantize tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md): GGUF quantization types with bits per weight and sizes for Llama 3.1.
