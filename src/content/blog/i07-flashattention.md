---
title: "FlashAttention and Fused Kernels: Attention Without the n×n Matrix"
description: "Why standard attention is memory-bound, how tiling and online softmax compute it exactly in on-chip SRAM, FlashAttention 1 to 4, and fused kernels."
slug: "flashattention"
category: "inference-optimization"
tags: ["attention", "gpu", "latency", "serving", "long-context"]
series: "llm-inference"
seriesOrder: 7
pubDate: 2026-10-23
---

## TL;DR

The textbook way to compute attention builds an n×n grid of scores, writes it to the GPU's main memory, reads it back for the softmax, writes the result, and reads that back again. For long prompts that traffic sets the speed, and the arithmetic waits on it. FlashAttention computes the same answer in small tiles that fit in the GPU's fast on-chip memory, using a running ("online") version of softmax, so the n×n grid never leaves the chip. The result is exact, its memory grows linearly with length, and on an A100 it ran attention up to 7.6 times faster. Four versions later it reaches 71 to 75% of a modern GPU's peak, and the same idea, fusing work into fewer kernels, runs through every serving engine alongside CUDA graphs and torch.compile.

## Why it matters

Take Llama 3.1 8B and a 32,768-token prompt. Computed the textbook way, the attention scores for a single layer, across its 32 heads, take 68.7 GB in 16-bit numbers. That's for one of its 32 layers and a single prompt, on a GPU that holds 80 GB in total.

No serving engine stores that grid. They all run some descendant of **FlashAttention**, a 2022 algorithm from Tri Dao and colleagues at Stanford and the University at Buffalo that never materializes it. On GPT-2 medium, the paper measured attention moving 40.3 GB through memory the standard way and 4.4 GB with FlashAttention, finishing in 7.3 ms instead of 41.7 ms on an NVIDIA A100. It got there while doing *more* arithmetic than the standard version.

[Part 3](/blog/gpus-for-llms/) showed why: on-chip memory is about an order of magnitude faster than main memory. This part shows how attention was rewritten to live there, and why the result is exact.

## The intuition

Picture a judge scoring a thousand contestants. The result is a weighted average, where each contestant's weight depends on how their score compares with everyone else's. The obvious method: write all thousand scores on a board in a hall down the corridor, walk back to compute the weights, walk again to average. Most of the day goes on walking.

A smarter judge sees contestants in groups of ten and keeps three numbers on a notepad: the best score so far, a running total of weights and a running weighted sum. When a group contains a new best score, the judge shrinks the old totals by the right factor and carries on. After the last group, one division gives the exact answer, and nobody walks to the hall.

The board is the GPU's high-bandwidth memory (HBM), the notepad is SRAM (fast memory on the chip), and the rescaling trick is online softmax. The analogy breaks in one place: the smart judge still meets every contestant. FlashAttention removes the walking, not the meeting, so the arithmetic still grows with the square of the length.

## Step 1: why standard attention is memory-bound

[Fundamentals Part 5](/blog/attention-explained/) built attention as `softmax(Q Kᵀ / √d) V`. For one head, with `n` tokens and head size `d`, the FlashAttention paper writes the standard implementation as three steps, each a separate GPU **kernel** (a program the GPU runs across thousands of threads):

1. Read Q and K from HBM, compute the scores `S = Q Kᵀ` (n×n), write S to HBM.
2. Read S, compute `P = softmax(S)` row by row, write P to HBM.
3. Read P and V, compute the output `O = P V` (n×d), write O to HBM.

S and P each make a round trip: four passes over an n×n matrix. Count the bytes for one head with `d = 128` (Llama 3.1 8B's head size) in 16-bit numbers, and the arithmetic, which is about `4 × n² × d` floating-point operations (FLOPs) for the two matrix multiplications:

| Prompt length | One n×n matrix | S and P traffic (4 passes) | Q, K, V in and O out | FLOPs | FLOPs per byte |
|---|---|---|---|---|---|
| 4,096 tokens | 33.6 MB | 134 MB | 4.2 MB | 8.6 billion | 62 |
| 32,768 tokens | 2.1 GB | 8.6 GB | 34 MB | 550 billion | 64 |

The last column is **arithmetic intensity**, the work done per byte moved. [Part 1](/blog/how-llm-inference-works/) worked out that an H100 needs about 295 FLOPs per byte to keep its tensor cores busy. Standard attention offers about 64, whatever the length, so it's memory-bound:

```text
32K tokens, one head, H100:
memory: 8.62 GB ÷ 3.35 TB/s       ≈ 2.57 ms
math:   550 GFLOP ÷ 989 TFLOPS     ≈ 0.56 ms
```

Most of the time goes on moving matrices that exist only to be thrown away. The softmax kernel alone is worse: a few operations per score while reading 2 bytes and writing 2, close to 1 FLOP per byte.

Compare the floor: read Q, K and V once and write O once. At 4,096 tokens that's 4.2 MB for the same 8.6 billion FLOPs, about 2,048 FLOPs per byte, far past the ridge. Every byte between that floor and 138 MB is S and P. The goal is to get rid of them without changing the answer.

### Why not just fuse the three kernels?

Kernel fusion, doing several steps in one pass so intermediate results stay on the chip, was Part 3's main trick. Here it hits a wall: softmax needs the maximum and the sum over *all* n scores in a row before it can produce a single weight. At 32K tokens one row is 64 KB, a kernel works on many rows at once, and an H100 SM (streaming multiprocessor) has at most 228 KB of shared memory. The rows can't stay on the chip, so softmax seems to force a trip to HBM. Online softmax removes that requirement.

## Step 2: tiling and the online softmax

The fix comes from a 2018 NVIDIA paper by Maxim Milakov and Natalia Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), which computes softmax "with fewer memory accesses" by keeping running statistics as it streams through the input.

Real softmax code subtracts the row's maximum before taking exponentials, because exponentials overflow fast. `e^11.1` is already 66,171, past FP16's largest value of 65,504 from [Part 3](/blog/gpus-for-llms/). Subtracting the max keeps every exponential at 1 or below and gives the same weights. The problem is that you don't know the max until you've seen the whole row.

Online softmax handles that by keeping three running numbers for each query row:

- `m`, the largest score seen so far
- `l`, the sum of `e^(score − m)` so far
- `acc`, the sum of `e^(score − m) × value` so far

When a block arrives with a bigger max, multiplying the old `l` and `acc` by `e^(m_old − m_new)` puts them on the new footing, and the block's own terms are added. At the end, `acc ÷ l` is the output.

### A worked example

One query, four keys, scores `[2, 1, 4, 3]`, and to keep it readable each value is a single number: `[10, 20, 30, 40]`. Full softmax gives weights 0.0871, 0.0321, 0.6439 and 0.2369, and the output `0.0871 × 10 + 0.0321 × 20 + 0.6439 × 30 + 0.2369 × 40 = 30.31`.

Now stream it in two blocks of two:

| Step | Scores seen | Running max `m` | Rescale old totals by | Running sum `l` | Running output `acc` | `acc ÷ l` |
|---|---|---|---|---|---|---|
| Start | none | minus infinity | nothing to rescale | 0 | 0 | none |
| Block 1 | 2, 1 | 2 | nothing to rescale | 1 + 0.368 = 1.368 | 10 × 1 + 20 × 0.368 = 17.36 | 12.69 |
| Block 2 | 4, 3 | 4 | e^(2 - 4) = 0.135 | 1.368 × 0.135 + 1 + 0.368 = 1.553 | 17.36 × 0.135 + 30 × 1 + 40 × 0.368 = 47.06 | 30.31 |

After block 1, the partial answer, 12.69, is correct for the first two keys alone. Block 2 brings a bigger max, the old totals shrink by `e^(−2)`, and the result is 30.31, matching the full softmax. In code:

```python
import numpy as np

s = np.array([2.0, 1.0, 4.0, 3.0])      # one query's four scores
v = np.array([10.0, 20.0, 30.0, 40.0])  # one number per value, to keep it small

# Standard: see the whole row, then normalize
w = np.exp(s - s.max())
w /= w.sum()
print(w @ v)                            # 30.3054

# Online: two blocks of two, never holding the whole row
m, l, acc = -np.inf, 0.0, 0.0           # running max, running sum, running output
for sb, vb in [(s[:2], v[:2]), (s[2:], v[2:])]:
    m_new = max(m, sb.max())
    scale = np.exp(m - m_new)           # shrink old totals to the new max
    p = np.exp(sb - m_new)
    l = l * scale + p.sum()
    acc = acc * scale + p @ vb
    m = m_new
print(acc / l)                          # 30.3054
```

### Why it's exact

Nothing is approximated. `e^(a − m_old) × e^(m_old − m_new)` equals `e^(a − m_new)` exactly, so rescaling just rewrites the same sum against a different reference point. The only differences from the standard version come from floating-point rounding, the kind you'd get from adding the same numbers in a different order. This is why the FlashAttention paper calls itself "exact attention", and why its GPT-2 models reached the same perplexity (18.2 for GPT-2 small) as the standard implementations.

In a real kernel, "value" is a vector of 128 numbers and a block is a tile of perhaps 128 keys, but the bookkeeping is the same: two numbers per query row, plus an output row.

## Step 3: FlashAttention, one fused kernel

With online softmax, attention fits into one kernel. The paper splits Q, K and V into tiles sized to fit in SRAM, loads them from HBM, and does everything else on the chip:

![Two rows. The top row shows standard attention as three kernels: Q, K and V in HBM go into kernel 1, which computes S equals Q times K transpose and writes the n by n matrix S to HBM; kernel 2 reads S, computes P equals softmax of S and writes the n by n matrix P to HBM; kernel 3 reads P, computes O equals P times V and writes O to HBM. The bottom row shows FlashAttention as one fused kernel: tiles of Q, K and V are loaded from HBM into SRAM, one Q tile stays while K and V tiles stream in, each K and V tile updates the scores, the online softmax and the output, and O is written to HBM once.](/diagrams/i07-hbm-vs-tiles.svg "Standard attention sends two n×n matrices through HBM; FlashAttention keeps every tile on the chip and writes only the output.")

The n×n matrices disappear from HBM traffic, and everything that touches them happens in SRAM.

For each tile of query rows, the kernel walks through the tiles of K and V. It computes a small block of scores with the tensor cores, updates the online-softmax statistics, rescales the output so far, and adds the new block's contribution. Masking (the causal mask from Fundamentals Part 5) and dropout happen in the same pass. When the last tile is done, it divides by `l` and writes the output row to HBM. The paper calls this design "IO-aware": it's planned by counting reads and writes between levels of memory.

### How much traffic it saves

The paper proves (Theorem 2) that standard attention needs `Θ(Nd + N²)` HBM accesses and FlashAttention needs `Θ(N²d²/M)`, where `N` is the sequence length, `d` the head size and `M` the SRAM size. Two things follow:

- FlashAttention's traffic still grows with N², because each tile of queries must see every key and value, but it's divided by `M/d²`. With `d` of 64 to 128 and `M` "around 100KB", in the paper's words, that factor is large.
- No exact attention algorithm can do asymptotically better across all SRAM sizes; the paper proves that too.

The paper measured the difference on GPT-2 medium (1,024 tokens, head size 64, 16 heads, batch 64), forward and backward on an A100:

| Measure | Standard attention | FlashAttention |
|---|---|---|
| Arithmetic | 66.6 GFLOPs | 75.2 GFLOPs |
| HBM reads and writes | 40.3 GB | 4.4 GB |
| Runtime | 41.7 ms | 7.3 ms |

FlashAttention does 13% more math, moves 9 times fewer bytes and runs 5.7 times faster.

### Memory that grows linearly

Since S and P are never stored, the extra memory per query row is the two statistics, `m` and `l`. At 32K tokens, the standard approach needs a 2.1 GB score matrix per head; FlashAttention needs 32,768 × 2 numbers, a few hundred kilobytes. The paper reports its memory footprint "grows linearly with sequence length" and is up to 20 times smaller than exact-attention baselines on an A100.

### Training: recompute instead of store

Training's backward pass, which computes gradients, normally reads P back from HBM. FlashAttention keeps only O and the per-row statistics and recomputes S and P tile by tile. That's the 75.2 versus 66.6 GFLOPs in the table: extra arithmetic instead of storing two n×n matrices. The backward pass still got faster, because it reads so much less from HBM.

## Step 4: decoding is a different problem

During decode (Part 1), each request adds one token per step, so attention is one query row against `n` cached keys and values. There's no n×n matrix to avoid: the score row is just `1 × n`. The cost is reading the KV cache. For Llama 3.1 8B, which shares each key-value head among 4 query heads ([Part 2](/blog/kv-cache/)), that's about 4 FLOPs per byte at any context length. Decode attention is memory-bound by nature, and the job is to read the cache at full bandwidth.

The original FlashAttention kernel struggled with that. It spreads work across the GPU by giving each batch item, head and tile of queries its own **thread block** (a group of threads that runs on one SM). With one query per request and a small batch, there are few thread blocks to hand out. The Flash-Decoding authors put it bluntly: "With a batch size of 1, FlashAttention will use less than 1% of the GPU!" Long contexts make it worse, since they force smaller batches to fit in memory.

### Flash-Decoding: split the cache, not the queries

[Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) (Tri Dao, Daniel Haziza, Francisco Massa and Grigory Sizov, 2023) adds a new direction to parallelize over: the keys and values. In three steps:

1. Split the cached keys and values into chunks.
2. Run FlashAttention on each chunk in parallel, each producing a partial output plus "1 extra scalar per row and per split: the log-sum-exp of the attention values."
3. Combine the partial outputs, weighting each by its log-sum-exp.

Step 3 is online softmax again, run in parallel instead of in sequence. In the worked example, block 1 alone gives (max 2, sum 1.368, output 12.69) and block 2 alone gives (max 4, sum 1.368, output 32.69). Rescale both to the larger max and combine: `(17.36 × 0.135 + 44.72) ÷ (1.368 × 0.135 + 1.368) = 47.06 ÷ 1.553 = 30.31`. Same answer, but the two blocks could have run on different SMs.

Splitting a 64K-token cache into chunks of 1,024 tokens (an illustrative choice) turns one head's work into 64 independent pieces. The authors measured decoding on CodeLlama-34B at lengths from 512 to 64K tokens and report "up to 8x speedups in decoding speed for very large sequences," with the attention step itself up to 50 times faster than FlashAttention in micro-benchmarks on an A100. The post doesn't name the GPU for the end-to-end test. The FlashAttention library adopted the split in version 2.2, through `flash_attn_with_kvcache`.

### Paged caches

[Part 5](/blog/pagedattention-vllm/) showed vLLM storing the cache in blocks found through a block table, and decode kernels now read that layout directly. FlashAttention 2.5 added a `block_table` argument for a "paged KV cache." **FlashInfer** (Ye et al., MLSys 2025) is a kernel library that treats a paged cache as a block-sparse matrix, so one kernel design handles pages, shared prefixes and sparse patterns. Plugged into the SGLang engine, it cut inter-token latency by 29 to 69% against SGLang's Triton backend, on Llama 3.1 8B (one H100) and 70B (four H100s). Its cascade attention for shared prefixes pairs naturally with [prefix caching](/blog/prefix-caching/).

## Step 5: fusing the rest of the model

Attention is one kernel per layer among many: normalization, the projections, rotary position encoding, the MLP's activation, the residual adds. Two costs pile up.

### Fusing small operations

Each small kernel reads its input from HBM and writes its output back. Serving engines fuse neighbors that touch the same data. vLLM, for example, has a [`fused_add_rms_norm`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py) that adds the residual and applies RMSNorm in one pass, and a [`SiluAndMul`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py) kernel for the SwiGLU activation that "computes x -> silu(x[:d]) * x[d:]" in one pass.

Here's what the second one saves for Llama 3.1 8B, whose MLP is 14,336 numbers wide, on an 8,192-token chunk of prefill. Each 14,336 × 8,192 tensor in 16-bit numbers is 235 MB:

```text
unfused: silu(gate)  read 235 MB, write 235 MB
         × up        read 470 MB, write 235 MB
         total 1.17 GB → 351 µs at 3.35 TB/s
fused:   read gate and up 470 MB, write 235 MB
         total 0.70 GB → 210 µs
```

That's 40% less traffic for one operation in one layer, with identical math.

### Launch overhead and CUDA graphs

The second cost is on the CPU, which spends a few microseconds preparing and submitting each kernel. Many decode kernels finish in about that time, so the GPU can end up waiting. NVIDIA's [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) measured 20 short kernels of 2.9 µs each on a V100: launched one by one with a sync after each, they took 9.6 µs per kernel; with launches overlapped, 3.8 µs; as a **CUDA graph**, 3.4 µs.

A CUDA graph records a whole sequence of kernels once, then replays it "through a single CPU operation." The catch is that a graph is fixed: same kernels, same shapes, same memory addresses. Serving engines handle that by capturing one graph per batch size and padding each real batch up to the nearest captured size.

### torch.compile and Triton

Writing fused kernels by hand for every model doesn't scale, so compilers do much of it. [torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler.html), introduced in PyTorch 2, captures a model's Python code as a graph with TorchDynamo and passes it to TorchInductor, which fuses operations and generates code. For NVIDIA, AMD and Intel GPUs, Inductor "leverages OpenAI Triton as the key building block."

**Triton** is a language and compiler for writing GPU kernels in terms of tiles, which its [docs](https://triton-lang.org/) describe as "a Python-based programming environment" for custom kernels. Philippe Tillet, H. T. Kung and David Cox introduced it in a [2019 paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) with matrix multiplication kernels "on par with hand-tuned vendor libraries (cuBLAS / cuDNN)." FlashAttention-2 credits Triton's fused-attention tutorial as the first place its loop ordering was tried.

## In the real world

### Four generations of FlashAttention

Each version re-fits the same algorithm to a newer GPU:

| Version | GPU | What changed | Reported result |
|---|---|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135) (2022) | A100 | Tiling, online softmax, recomputation, one fused kernel | 7.6x faster attention on GPT-2; 3x faster GPT-2 training, 15% faster BERT-large training |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023) | A100 | Fewer non-matmul FLOPs, parallelism over sequence length, better split of work between warps | About 2x over FlashAttention; 50 to 73% of peak FLOPs |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) (2024) | H100 | Asynchronous data movement (TMA), warp specialization, FP8 | 1.5 to 2.0x over FlashAttention-2 (FP16 forward pass); 740 TFLOPS in FP16 (75%); close to 1.2 PFLOPS in FP8 |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451) (2026) | B200 | Pipelines for Blackwell, software exponentials, conditional rescaling | Up to 1,613 TFLOPS in BF16 (71%); 1.3x over cuDNN 9.13 |

The first paper's training runs used 8 A100s: BERT-large hit its target accuracy in 17.4 minutes against 20.0 for NVIDIA's MLPerf 1.1 record.

FlashAttention-2 went after work that wasn't matrix multiplication. On an A100, "each non-matmul FLOP is 16× more expensive than a matmul FLOP" (312 TFLOPS on tensor cores against 19.5 elsewhere). It cut those operations, gave each thread block its own tile of queries so even one head spreads across many SMs, and split work among warps so they stopped swapping partial results through shared memory. It reached up to 230 TFLOPS on an A100 80GB.

FlashAttention-3 started from a problem: FlashAttention-2 reached only 35% utilization on an H100. Hopper adds the Tensor Memory Accelerator (TMA), a unit that copies tiles between HBM and shared memory on its own. FlashAttention-3 splits each thread block's warps into producers, which only issue TMA loads, and consumers, which only compute, so the next tile loads while the current one is multiplied. It also overlaps softmax with the matrix multiplications, because exponentials are slow: the H100 SXM5 does 989 TFLOPS of FP16 matrix math but only 3.9 TFLOPS of special functions such as the exponential. Its FP8 mode has 2.6 times lower numerical error than a baseline FP8 attention.

FlashAttention-4 targets Blackwell, where tensor-core throughput doubled (2.25 PFLOPS on a B200 against about 1 on an H100) but shared memory bandwidth and the exponential units didn't keep pace. It computes some exponentials with a polynomial on the ordinary arithmetic units, and it rescales the online softmax only when the running max jumps by more than a threshold (a factor of 256), tracking the scaling so the answer stays exact. The [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) ships it as `flash-attn-4`.

### In a serving engine

vLLM turns all of this on by default. When you don't choose an attention backend, it takes the first compatible one from a priority list: [FlashAttention on Ampere and Hopper GPUs, FlashInfer on Blackwell](https://docs.vllm.ai/en/latest/design/attention_backends.html).

"In vLLM's V1 architecture, `torch.compile` is enabled by default," its [design docs](https://docs.vllm.ai/en/latest/design/torch_compile.html) say. The model graph is split at each attention operation, Inductor compiles the pieces in between, and the results are cached on disk for later starts. CUDA graphs [default to](https://docs.vllm.ai/en/latest/design/cuda_graphs.html) `FULL_AND_PIECEWISE` mode: "full CUDA Graph for uniform decode, piecewise CUDA Graphs for others." Graphs are captured for batch sizes up to 512 ([1,024 on data center Blackwell GPUs](https://docs.vllm.ai/en/latest/configuration/engine_args.html)).

The flag `--enforce-eager` "disables both torch.compile and CUDA graphs." It's meant for debugging, and running with and without it shows what the two are worth on your own hardware.

### What FlashAttention doesn't change

FlashAttention removes the memory traffic, not the arithmetic. Every query still meets every key, so prefill attention costs `4 × n² × d` FLOPs per head (about half that with the causal mask): 550 billion at 32K tokens and 8.8 trillion at 128K, in each of Llama 3.1 8B's 1,024 heads. Fundamentals Part 5 showed that at 128K tokens this comes to about twice the cost of all the model's weights. Shrinking that number takes changing what gets computed, with sliding windows or sparse attention, which is Part 11's subject.

## Common misconceptions

- **"FlashAttention is an approximation."** It computes exactly the same attention. The online-softmax rescaling is exact algebra, and results differ from the standard version only by floating-point rounding. Approximate methods, like the block-sparse variant in the same paper, are a separate option.
- **"FlashAttention makes attention linear in sequence length."** Its memory is linear. Its FLOPs are still quadratic, and its HBM traffic is still `Θ(N²d²/M)`, just with a much smaller constant than the standard version.
- **"FlashAttention speeds up decoding the same way it speeds up prefill."** Decode has no n×n matrix to avoid. It's limited by reading the KV cache, and the original kernel couldn't spread one query across the GPU. Flash-Decoding and paged kernels fix that by splitting the cache.

## Key takeaways

- Standard attention writes and rereads two n×n matrices in main memory. At about 64 FLOPs per byte against the H100's 295, it's memory-bound.
- Online softmax keeps a running max, sum and output, and rescales them when a bigger score arrives. Block by block, it gives exactly the same answer.
- FlashAttention computes attention tile by tile in on-chip SRAM, in one fused kernel. On GPT-2 medium, HBM traffic fell from 40.3 GB to 4.4 GB.
- Each version re-fits the idea to new hardware, up to 740 TFLOPS on the H100 and 1,613 TFLOPS on the B200.
- Decoding needs a different split: Flash-Decoding divides the KV cache across thread blocks and merges the pieces with the same rescaling math.
- The quadratic arithmetic is still there. Cutting it means computing less attention, which is Part 11's subject.

## What's next

FlashAttention cuts the bytes attention moves around. The weights and the KV cache are still read from memory at 16 bits per number on every decode step. Part 8 covers quantization: storing weights, and the KV cache itself, in 8 or 4 bits, so each step moves half or a quarter of the bytes, and how to check the model still gives good answers.

## Further reading

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135): the original algorithm, the IO-complexity proofs and the A100 measurements.
- [Online normalizer calculation for softmax (Milakov and Gimelshein, 2018)](https://arxiv.org/abs/1805.02867): the running-max softmax that FlashAttention builds on.
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)](https://arxiv.org/abs/2307.08691): fewer non-matmul FLOPs and better use of thread blocks and warps.
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah et al., 2024)](https://arxiv.org/abs/2407.08608): warp specialization, TMA and FP8 on the H100.
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling (Zadouri et al., 2026)](https://arxiv.org/abs/2603.05451): adapting the kernel to Blackwell's faster tensor cores and slower everything else.
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving (Ye et al., 2025)](https://arxiv.org/abs/2501.01005): paged and block-sparse attention kernels for serving engines.
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (Tillet, Kung and Cox, 2019)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf): the tile-based GPU language behind torch.compile's GPU code.
- [FlashAttention repository (Dao-AILab)](https://github.com/Dao-AILab/flash-attention): code for all four versions, including `flash_attn_with_kvcache` and paged KV cache support.
- [vLLM design docs: torch.compile integration](https://docs.vllm.ai/en/latest/design/torch_compile.html), [CUDA graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html) and [attention backends](https://docs.vllm.ai/en/latest/design/attention_backends.html): how a serving engine puts these pieces together by default.
- [torch.compiler documentation (PyTorch)](https://docs.pytorch.org/docs/stable/torch.compiler.html): TorchDynamo, TorchInductor and the Triton backend.
- [Flash-Decoding for long-context inference (Dao, Haziza, Massa and Sizov, 2023)](https://crfm.stanford.edu/2023/10/12/flashdecoding.html): splitting the KV cache to parallelize decoding.
- [Getting Started with CUDA Graphs (Alan Gray, NVIDIA, 2019)](https://developer.nvidia.com/blog/cuda-graphs/): launch overhead and how graphs remove it, with measurements.
- [Making Deep Learning Go Brrrr From First Principles (Horace He)](https://horace.io/brrr_intro.html): compute, bandwidth and overhead, the three costs this article keeps returning to.
