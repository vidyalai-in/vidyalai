---
title: "Writing an Attention Kernel: FlashAttention and Decode in Triton"
description: "Build a causal FlashAttention kernel and a split-KV, paged decode kernel in Triton, then time them against naive attention and PyTorch's SDPA."
slug: "attention-kernel"
category: "gpu-kernels"
tags: ["gpu", "attention", "triton", "long-context", "latency"]
series: "llm-gpu"
seriesOrder: 10
pubDate: 2026-06-02
---

## TL;DR

Attention needs two different kernels. Prefill, which processes the whole prompt at once, is a big compute-bound problem. Decode, which adds one token at a time, is a memory-bound read of the KV cache (the keys and values stored for every earlier token). This part writes both in Triton on a laptop RTX 4060. A FlashAttention-style prefill kernel of about 45 lines reached 24 TFLOPS, 90% of the tensor-core peak and, from 4,096 tokens up, within 2 to 3% of the FlashAttention-2 kernel inside PyTorch; textbook attention ran out of memory at 16,384 tokens. A decode kernel with one program (thread block) per head used 2 of the GPU's 24 SMs (streaming multiprocessors) and moved 60 GB/s. Splitting the cache into chunks (Flash-Decoding) lifted it to 246 GB/s, 99% of the measured ceiling, and reading the cache through a vLLM-style block table cost under 1% at long contexts.

## Why it matters

For one layer of a 1.5-billion-parameter model and an 8,192-token prompt, textbook attention asked this laptop's GPU for 8.1 GB of scratch memory, more than the 7.4 GB that was free. The allocation still went through, the overflow landed outside the GPU's own memory, and the call took 359 milliseconds. A Triton kernel computed the same answer in 8.7 ms with 26 MB.

[Inference Part 7](/blog/flashattention/) explained why FlashAttention wins: it never writes the n × n score matrix to memory. This part writes one, measures it, then turns to the case it handles badly, generating one token at a time. It uses the online softmax that [Part 8](/blog/gpu-reductions/) said pays off "when data outgrows every cache, the normal case in attention."

## The intuition

Picture a library with 24 reading tables. Prefill is 4,096 readers, each needing every book shelved before their own seat number. The slow way photocopies every page for every reader and piles the copies in the corridor. The fast way seats 128 readers at a table, wheels carts of books past them, and lets each reader keep a running note. Nothing goes to the corridor.

Decode is one reader and the whole library. Wheeling carts past one table leaves 23 tables empty. So you split the shelves into chapters: eight tables each read one, write a summary plus a weight for how much their chapter matters, and a ninth person merges the summaries.

Here the analogy breaks, because real summaries lose information. As [Math Part 5](/blog/softmax-math/) proved, merging partial softmax results by their log-sum-exp gives exactly the one-pass answer.

## Step 1: textbook attention, measured

All measurements use Qwen2.5-1.5B's attention shapes: 12 query heads of dimension 128 sharing 2 key-value heads, so each key-value head serves 6 query heads. That's **grouped-query attention** (GQA, [Math Part 11](/blog/attention-math/)). Batch size is 1 and the numbers are BF16 (16-bit brain floating point).

Here is attention the way Hugging Face Transformers' `eager_attention_forward` writes it:

```python
def naive_attn(q, k, v, mask):
    """Textbook attention, step for step as Hugging Face Transformers' eager_attention_forward does it."""
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)                       # GQA: copy each KV head to its query heads
    v = v.repeat_interleave(group, dim=1)
    s = torch.matmul(q, k.transpose(2, 3)) * q.shape[-1] ** -0.5  # (B, H, n, n) scores in BF16
    s = s + mask                                                # causal mask
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(p, v)
```

The **causal mask** is an n × n tensor of zeros with minus infinity above the diagonal, so each token attends only to itself and earlier tokens. It's built once and shared by every layer.

### How it was timed

Everything ran on [Part 1](/blog/gpu-from-python-to-silicon/)'s machine: an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, 24 SMs, compute capability 8.9) under WSL2, driver 592.82, PyTorch 2.11.0 with CUDA 12.8, and Triton 3.6.0. Prefill times come from `triton.testing.do_bench`, which flushes the 32 MB L2 cache before every call and times each call with CUDA events: 0.2 s of warm-up, 0.8 s of timed calls, the median, then the median of three rounds. `nvidia-smi` logged clocks and power every 200 ms. The SM clock sat at 2,175 MHz in almost every run (1,950 to 2,175 MHz at 32,768 tokens), the memory clock switched between 7,000 and 8,000 MHz, and the GPU drew up to 50 W against an enforced power limit of 44.5 to 50 W.

FLOPs (floating-point operations) are counted the way the [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) counts them: `4 · n² · d` per head for the two matrix multiplies, divided by 2 with the causal mask "to account for the fact that approximately only half of the entries are calculated." For 12 heads, d = 128 and n = 4,096, that's 51.5 billion. The peak for BF16 with FP32 accumulation is `24 SMs × 512 FLOPs per clock × clock`, from the [Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) as Part 1 derived it: 26.7 TFLOPS at 2,175 MHz. [Part 4](/blog/gpu-profiling-roofline/) measured a practical roof of 25.3 TFLOPS.

### Where the time and memory go

| Tokens | Time | TFLOPS | Extra GPU memory |
|---|---|---|---|
| 512 | 0.30 ms | 2.7 | 43 MB |
| 1,024 | 1.88 ms | 1.7 | 132 MB |
| 2,048 | 7.50 ms | 1.7 | 516 MB |
| 4,096 | 31.1 ms | 1.7 | 2,038 MB |
| 8,192 | 359 ms | 0.6 | 8,103 MB |
| 16,384 | out of memory | none | none |

Memory grows fourfold each time the prompt doubles. At 4,096 tokens there are `12 × 4,096² = 201 million` scores, and the peak holds them three times (BF16 scores, an FP32 copy for the softmax, the FP32 result): 10 bytes per score, 2.0 GB. At 8,192 tokens that's 8.1 GB with 7.4 GB free. The allocations succeeded anyway, but `nvidia-smi` never showed more than 7,833 MiB in use on the GPU: the rest sat in system memory, which the Windows driver behind WSL2 can lend the GPU. The call took 11.5 times as long as at 4,096 tokens for 4 times the work. At 16,384 tokens even that failed.

The profiler splits the 31 ms at 4,096 tokens into nine kernels:

| Kernel | Time | What it does |
|---|---|---|
| `cunn_SoftMaxForwardReg` | 7.06 ms | Softmax in FP32 |
| BF16 copy | 5.22 ms | Casts the probabilities back to BF16 |
| direct copy | 5.15 ms | Casts the scores to FP32 for the softmax |
| `CUDAFunctor_add` | 5.02 ms | Adds the mask |
| `AUnaryFunctor` | 3.64 ms | Multiplies by `1/√d` |
| `ampere_bf16_s1688gemm` | 2.51 ms | `S = Q Kᵀ` on tensor cores |
| `cutlass ... s16816gemm` | 2.26 ms | `O = P V` on tensor cores |
| two copies | 0.29 ms | `repeat_interleave` of K and V |

The two matrix multiplies, the only real arithmetic, take 4.8 ms, 15% of the total; the other 85% shuttles the score matrix through memory. Nsight Compute (`ncu`), which locks clocks at base but counts bytes exactly, measured 3.67 GB read from DRAM and 3.14 GB written. That's 7.6 FLOPs per byte for 51.5 GFLOP, far below Part 4's ridge point of 102, so textbook attention is memory-bound.

## Step 2: a FlashAttention kernel in Triton

The kernel follows [Inference Part 7](/blog/flashattention/)'s recipe with FlashAttention-2's loop order, in the Triton style of [Part 7](/blog/triton-kernels/). The grid has one **program** (Triton's unit of work, one CUDA thread block running on one SM) per query block and head: for 4,096 tokens and 128-query blocks, `32 × 12 = 384` programs. Each program:

1. Loads its 128 × 128 tile of Q once.
2. Walks over 64-token key blocks: K tile in, scores with `tl.dot` on tensor cores, online-softmax update, V tile in, output updated with a second `tl.dot`.
3. Divides by the running sum and writes 128 output rows and their log-sum-exp.

Grouped-query attention costs one line: query head `h` reads key-value head `h // 6`, whose tiles the other five programs have usually pulled into L2 already.

### The causal mask as a loop bound

A query block never needs keys past its own last row, so key blocks fall into three kinds:

- Blocks entirely before the query block: no masking needed.
- The block on the diagonal: some scores must be set to minus infinity.
- Blocks entirely after it: skipped, never loaded.

So the kernel calls its inner loop twice: from key 0 to its own query block with masking compiled out, then over the diagonal with masking on. The FlashAttention-2 paper does the same and reports that skipping about half the blocks gives "around 1.7-1.8× speedup compared to attention without the causal mask." Here is the inner loop, the whole online softmax:

```python
@triton.jit
def _attn_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, offs_m, offs_n, lo, hi, qk_scale,
                stride_n, BLOCK_N: tl.constexpr, MASKED: tl.constexpr):
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_n)                 # BLOCK_N x D tile of K
        s = tl.dot(q, tl.trans(k)) * qk_scale                    # scores x log2(e)/sqrt(d), in FP32
        if MASKED:                                               # only the diagonal block needs the mask
            s = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))                    # running max
        alpha = tl.exp2(m_i - m_new)                             # rescale factor for the old state
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)                         # running sum
        v = tl.load(v_ptrs + start_n * stride_n)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])     # running output, rescaled
        m_i = m_new
    return acc, l_i, m_i
```

The scale folds in `log2(e) = 1.4427`, so the exponentials become `exp2`, the GPU's native base-2 instruction (the compiled PTX, NVIDIA's virtual assembly, holds 68 `ex2.approx`), as in Triton's fused-attention tutorial. The running max `m_i`, sum `l_i` and output `acc` stay in FP32; only the probabilities `p` drop to BF16 for the second multiply. The division by `l_i` happens once, after the loop, the FlashAttention-2 change [Math Part 5](/blog/softmax-math/) describes.

The outer kernel just computes pointers and runs the two loops. It compiled to 256 `mma.sync` tensor-core instructions, 64 `cp.async` copies for pipelining and 152 `ldmatrix` loads, with 96 KB of shared memory and 206 registers per thread, no spills.

### Is it right?

```python
import torch, torch.nn.functional as F
from flash import flash_attn
torch.manual_seed(0)
q = torch.randn(1, 12, 4096, 128, device="cuda", dtype=torch.bfloat16)   # 12 query heads
k = torch.randn(1, 2, 4096, 128, device="cuda", dtype=torch.bfloat16)    # 2 KV heads (GQA)
v = torch.randn_like(k)
ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True, enable_gqa=True)
sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
mine = flash_attn(q, k, v)
print(f"max error vs FP64: mine {(mine.double() - ref).abs().max():.2e}, "
      f"SDPA {(sdpa.double() - ref).abs().max():.2e}, mine vs SDPA {(mine - sdpa).abs().max():.2e}")
```

```text
max error vs FP64: mine 7.47e-03, SDPA 7.47e-03, mine vs SDPA 3.91e-03
```

Both kernels land the same distance from a 64-bit reference, and that distance is BF16 rounding of the output: near 1.0, neighboring BF16 values are 0.0078 apart. The naive version was off by up to 0.0196, because it rounds the scores to BF16 before the softmax. A first draft of this kernel scaled Q in BF16 and had errors up to 30% larger; scaling the FP32 scores instead fixed it.

### Against PyTorch's SDPA

PyTorch's `scaled_dot_product_attention` (SDPA) "attempts to automatically select the most optimal implementation based on the inputs," per its [docs](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html), and the `torch.nn.attention.sdpa_kernel` context manager pins a backend. Here the default was `flash_fwd_kernel` with traits `<128, 64, 64, 4>` (head dimension 128, 64 × 64 tiles, 4 warps): PyTorch's bundled FlashAttention-2, in CUDA. The memory-efficient backend refused `enable_gqa=True` in PyTorch 2.11, so its runs expand K and V inside the timed call. TFLOPS, median of three rounds:

| Tokens | Naive | My Triton kernel | SDPA, FlashAttention-2 (default) | SDPA, cuDNN | SDPA, memory-efficient |
|---|---|---|---|---|---|
| 512 | 2.7 | 10.2 (38%) | 12.8 (48%) | 10.0 | 6.8 |
| 1,024 | 1.7 | 15.7 (59%) | 17.8 (66%) | 14.9 | 10.3 |
| 2,048 | 1.7 | 20.0 (75%) | 21.2 (79%) | 19.0 | 12.9 |
| 4,096 | 1.7 | 22.6 (85%) | 23.1 (86%) | 21.5 | 14.5 |
| 8,192 | 0.6 | 23.7 (89%) | 24.2 (90%) | 22.9 | 15.5 |
| 16,384 | out of memory | 24.1 (90%) | 24.5 (92%) | 23.6 | 15.9 |
| 32,768 | out of memory | 23.8 (92%) | 24.4 (93%) | 23.9 | 16.2 |

Percentages are of the peak at each run's SM clock. From 4,096 tokens up, the Triton kernel ran at 97 to 98% of FlashAttention-2's speed, ahead of cuDNN at every length but 32,768, and at 16,384 tokens it reached 95% of Part 4's measured 25.3 TFLOPS roof. At 4,096 tokens it needed 13 MB (output and log-sum-exp) against naive's 2,038 MB and ran 13.6 times faster; at 8,192, 41 times faster.

Below 2,048 tokens every kernel falls off, and not for lack of bandwidth. At 512 tokens there are only `4 × 12 = 48` programs, and lopsided ones: the last query block of each head loops over 8 key tiles, the first over 2. With 96 KB of shared memory each, one program fits per SM, so 24 SMs work through them in two uneven rounds. SDPA's 64-row tiles make twice as many programs and balance better.

Nsight Compute confirmed the bytes. At 4,096 tokens the Triton kernel read 16.8 MB from DRAM, exactly Q, K and V once each, and its 12.6 MB of output was still in L2 when it ended: 405 times less traffic than naive. At 32,768 tokens it read 145 MB against a minimum of 134 MB, because K and V (33.6 MB) no longer fit in the 32 MB L2 and some tiles are fetched twice.

### Tuning, briefly

Tile sizes, warps per program and pipeline stages, swept at 8,192 tokens with the SM clock at 2,175 MHz:

| Tile (queries × keys) | Warps | Stages | TFLOPS |
|---|---|---|---|
| 128 × 64 | 8 | 3 | 23.3 |
| 128 × 64 | 8 | 2 | 23.3 |
| 128 × 32 | 4 | 2 | 23.3 |
| 128 × 128 | 8 | 2 | 21.6 |
| 64 × 64 | 4 | 1 | 21.2 |
| 128 × 64 | 4 | 3 | 19.1 |
| 64 × 32 | 8 | 1 | 9.4 |
| 128 × 64 | 8 | 4 | did not fit: over 99 KB of shared memory |

The best and worst configurations that fit differ by 2.5 times. Bigger query tiles reuse each K and V tile across more rows; eight warps help a 128-row tile and starve a 64-row one. Each extra stage holds another K and V tile in shared memory (16 KB each), and Ada allows a block 99 KB, which rules out deep pipelines with big tiles. Launching the query blocks in reverse gained another 2 to 4%: the longest programs start first and the short ones fill the gaps at the end.

## Step 3: why prefill is compute-bound and decode isn't

Compare the minimum traffic (Q, K and V in, the output out) with the FLOPs. For n tokens with these shapes the causal FLOPs are `12 × 2 × n² × 128` and the bytes are `(12 + 2 + 2 + 12) × n × 128 × 2`, so the ratio is about `0.43 × n` FLOPs per byte:

| Work | FLOPs per byte | Ridge (Part 4) | Bound by |
|---|---|---|---|
| Prefill, 512 tokens | 219 | 102 | compute |
| Prefill, 4,096 tokens | 1,755 | 102 | compute |
| Naive prefill, 4,096 tokens (measured bytes) | 7.6 | 102 | memory |
| Decode, any context length | 6 | 102 | memory |

Prefill's intensity grows with length because every key is reused by every later query. Decode has one query per sequence, so each key and value is read once and used by six query heads: 512 bytes of K and V per token and key-value head buy `6 × 4 × 128 = 3,072` FLOPs, 6 FLOPs per byte at any context length. Batching doesn't help, since every sequence has its own cache; only more query heads per key-value head would.

A decode kernel should therefore be judged in GB/s, against [Part 3](/blog/gpu-memory-hierarchy/)'s measured read ceiling of 248.7 GB/s.

## Step 4: decode attention and Flash-Decoding

The decode kernel takes one query vector per sequence and a cache laid out as (batch, token, key-value head, dimension). One program owns one key-value head and its 6 query heads, padded to 16 rows because `tl.dot` needs at least 16, and loops over the cache in 64-token tiles with the same online softmax. For batch 1 that's 2 programs.

### The timing harness

At 4,096 tokens a decode call reads 4 MB, which fits in L2, in about 20 µs, so a careless benchmark measures the cache or the launch. Each measurement here is a CUDA graph ([Part 9](/blog/kernel-fusion-cuda-graphs/)) of L calls, one per "layer," each with its own cache (at least 128 MB in all), so every call reads cold from DRAM as in a real model and no launch gaps remain. Replays ran for about a second; time per call is the median replay divided by L, and tables give the median of three rounds. The memory clock held 8,000 MHz in all of my kernel's runs, so the ceiling is 248.7 GB/s.

### Two programs, 22 idle SMs

This kernel moved 57 to 60 GB/s at every context length from 4,096 to 131,072 tokens, 24% of the ceiling: about 30 GB/s per program, all one SM manages with the loads it keeps in flight. One program per query head did no better (59 GB/s): six programs read each cache, and 12 programs still leave half the SMs idle. The Flash-Decoding post says it bluntly: "With a batch size of 1, FlashAttention will use less than 1% of the GPU!"

### Split the cache

[Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) (Tri Dao, Daniel Haziza, Francisco Massa and Grigory Sizov, 2023) adds a grid dimension over the sequence: "First, we split the keys/values in smaller chunks," attend to each in parallel, and "Finally, we compute the actual output by reducing over all the splits, using the log-sum-exp to scale the contribution of each split." In the kernel, `program_id(1)` picks a chunk, the loop covers that chunk only, and each program stores its normalized partial output `o_i` and log-sum-exp `LSE_i` in FP32. A second kernel merges them:

```python
@triton.jit
def decode_combine(O_part, LSE_part, O, NUM_SPLITS: tl.constexpr, S_PAD: tl.constexpr, HEAD_DIM: tl.constexpr):
    bh = tl.program_id(0)                        # one program per (sequence, query head)
    s = tl.arange(0, S_PAD)
    d = tl.arange(0, HEAD_DIM)
    lse = tl.load(LSE_part + bh * NUM_SPLITS + s, mask=s < NUM_SPLITS, other=float("-inf"))
    m = tl.max(lse, 0)
    w = tl.exp2(lse - m)                         # each split's share of the softmax mass ...
    w = w / tl.sum(w, 0)                         # ... normalized: exp(LSE_i - LSE_total)
    o = tl.load(O_part + (bh * NUM_SPLITS + s[:, None]) * HEAD_DIM + d[None, :], mask=(s < NUM_SPLITS)[:, None], other=0.0)
    tl.store(O + bh * HEAD_DIM + d, tl.sum(w[:, None] * o, 0).to(O.dtype.element_ty))
```

The weight `exp(LSE_i − LSE)` is the share of the total softmax mass that falls in chunk i, the merge [Math Part 5](/blog/softmax-math/) derived. Against a 64-bit reference, the 16-chunk kernel's largest error at 33,000 tokens was 8.6 × 10⁻⁵, against 9.1 × 10⁻⁵ unsplit.

![A left-to-right flowchart. A KV cache stored in 16-token pages scattered in GPU memory feeds a block table that maps logical pages to physical pages. The block table's page lookups and the new token's query, 12 heads with 6 per KV head, feed kernel 1, which runs one program per KV head and chunk: chunk 1 of 8 and chunk 2 of 8 each run the online softmax over their tokens, and chunks 3 to 8 run the same loop in parallel. Kernel 1 writes partial results, an output o_i and a log-sum-exp LSE_i per chunk. Kernel 2 combines them, weighting each o_i by exp of LSE_i minus LSE, and produces the attention output for 12 heads.](/diagrams/g10-split-kv.svg "Decode attention with split-KV and a paged cache: many programs read separate chunks, and a small second kernel merges their partial results.")

Each chunk runs on its own SM, and only a few kilobytes of partial results per head cross between the two kernels.

How many chunks? GB/s at batch 1 (programs = 2 × chunks):

| Chunks | 1,024 tokens | 4,096 tokens | 16,384 tokens | 131,072 tokens |
|---|---|---|---|---|
| 1 | 50 | 57 | 59 | 59 |
| 2 | 86 | 108 | 115 | 117 |
| 4 | 122 | 189 | 221 | 233 |
| 8 | 127 | 199 | 233 | 247 |
| 16 | 110 | 187 | 227 | 243 |
| 32 | none | 177 | 230 | 246 |

Bandwidth doubles with each doubling of programs up to 8, then flattens by 16: 16 busy SMs saturate the bus. Eight chunks was best at every length; more add partial results to write and merge, which shows at short contexts. (131,072 tokens is beyond Qwen2.5-1.5B's 32,768-token window; it stands in for long-context models with the same head layout.)

### The full comparison

| Context (tokens) | PyTorch, unfused | SDPA (FlashAttention-2 split-KV) | Mine, one program per KV head | Mine, 8 chunks | Mine, 8 chunks, paged | Time per call, 8 chunks |
|---|---|---|---|---|---|---|
| 1,024 | 49 GB/s | 85 | 51 | 127 | 111 | 8.3 µs |
| 4,096 | 88 | 135 | 57 | 199 | 188 | 21.1 µs |
| 16,384 | 81 | 195 | 59 | 233 | 229 | 71.9 µs |
| 32,768 | 82 | 224 | 60 | 239 | 237 | 140 µs |
| 131,072 | 82 | 230 | 60 | 246 | 246 | 545 µs |
| 4,096, batch 16 | 48 | 230 | 240 (no split) | 242 (2 chunks) | 239 | 277 µs |

The split kernel reached 96% of the 248.7 GB/s ceiling at 32,768 tokens and 99% at 131,072. Under SDPA the profiler showed `flash_fwd_splitkv_kernel` followed by `flash_fwd_splitkv_combine_kernel`: PyTorch's bundled FlashAttention-2 already does Flash-Decoding for single-token queries. It ran 6% behind at 32,768 and 131,072 tokens and 16% behind at 16,384. Short contexts are latency-bound for everyone: 1 MB at 248.7 GB/s takes 4.2 µs, about what two kernels spend ramping up and draining.

The PyTorch version (`matmul`, softmax, `matmul`) launched six kernels and read the cache at about 80 GB/s; at batch 16 it first copied the whole K cache into a contiguous layout, 985 µs on its own. At batch 16, my kernel's 32 programs fill the GPU without splitting; the split pays off for one long sequence, the case Flash-Decoding was built for.

Qwen2.5-1.5B has 28 layers. At its full 32,768-token window, attention reads 28 KB of cache per token of context, 0.94 GB per generated token, 3.9 ms at the split kernel's speed. That's a third of the 12.4 ms it takes to read the model's 3.09 GB of weights. With one program per head it would be 15.7 ms, more than the weights.

## Step 5: paged KV and the block table

[Inference Part 5](/blog/pagedattention-vllm/) explained vLLM's paged cache: memory comes in fixed blocks of 16 tokens, and each sequence has a **block table** that maps its logical blocks to physical ones wherever they landed. The kernel change is the address arithmetic:

```python
        if PAGED:                                # block table lookup: logical page -> physical page
            page = tl.load(BT + b * max_pages + tok // PAGE, mask=tok_ok, other=0)
            row = (page.to(tl.int64) * PAGE + tok % PAGE) * KVH + kvh
        else:                                    # contiguous cache, strides in rows of HEAD_DIM
            row = b.to(tl.int64) * sb + tok * st + kvh * sh
```

The paged cache uses the layout FlashAttention's README gives for one, (num_blocks, page_block_size, nheads_k, headdim), with 16-token pages scattered at random. A 64-token tile spans four pages, so each tile does one small load from the block table, then the K and V loads at the addresses it returns.

At long contexts that cost almost nothing: 0.8% at 32,768 tokens and 0.4% at 131,072. At short ones it cost more, 14% at 1,024 tokens and 6% at 4,096. Pages of 256 tokens cost the same as pages of 16, within 3%, so the scattering wasn't the problem; the extra lookup was. A pipelining test at 4,096 tokens with 4 chunks showed why:

| Pipeline stages | Contiguous | Paged |
|---|---|---|
| 1 | 115 GB/s | 106 GB/s |
| 2 | 133 GB/s | 133 GB/s |
| 3 | 190 GB/s | 132 GB/s |

With 3 stages the paged kernel ran like the contiguous one with 2. Triton's software pipelining issues loads for future iterations during the current one, but an address that depends on another load must wait for it; the numbers fit the paged kernel losing one stage of lookahead. Adding a stage needs smaller tiles to fit in shared memory: 32-token tiles with 5 stages brought the paged kernel to 127, 199, 233 and 239 GB/s at 1,024 to 32,768 tokens, level with contiguous. The PagedAttention paper measured its kernel 20 to 26% slower than FasterTransformer's contiguous one ([Inference Part 5](/blog/pagedattention-vllm/)); here, once the loop hides its latency, the indirection costs well under 1%.

## In the real world

[FlashAttention-2](https://arxiv.org/abs/2307.08691) (Tri Dao, 2023) is the design copied here. It parallelizes "even for a single head, across different thread blocks," one per block of query rows, and inside a block it splits Q across warps rather than K and V, because splitting K forced warps to "write their intermediate results out to shared memory, synchronize, then add up the intermediate results." It reported 50 to 73% of the A100's theoretical peak, credits the loop order to Phil Tillet's Triton implementation, and tuned block sizes by hand from "{64, 128} × {64, 128}", the space swept above.

Triton's [fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html), "a Triton implementation of the Flash Attention v2 algorithm," uses the same `exp2` scaling and two-stage causal loop ("off-band" and "on-band") and adds the backward pass.

[FlashAttention-3](https://arxiv.org/abs/2407.08608) (Shah et al., 2024) starts from FlashAttention-2 "achieving only 35% utilization on the H100 GPU." It uses Hopper's TMA copy engine and warp specialization to "overlap overall computation and data movement," interleaves matmul and softmax, and adds FP8, reaching "up to 740 TFLOPs/s (75% utilization)" in FP16. Ada has neither TMA nor `wgmma` ([Part 6](/blog/tensor-cores/)), so none of that was measured here.

The backward pass, which training needs, isn't written here. The [original FlashAttention paper](https://arxiv.org/abs/2205.14135) avoids storing the n × n probabilities by keeping "the output O and the softmax normalization statistics (m, ℓ)" and recomputing scores tile by tile, which is why the forward kernel writes a log-sum-exp per row. FlashAttention-2 counts the backward pass as 2.5 times the forward FLOPs, since recomputation makes it 5 matrix multiplies against the forward pass's 2.

The Flash-Decoding post reports "up to 8x faster generation for very long sequences"; the method shipped in FlashAttention 2.2 and xFormers 0.0.22. The [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) added a paged KV cache through `block_table` in version 2.5; its README requires `page_block_size` to be "a multiple of 256." vLLM's [attention backend docs](https://docs.vllm.ai/en/latest/design/attention_backends.html) allow block sizes in multiples of 16 for its `FLASH_ATTN` backend, which runs "FA4 on SM100+ (Blackwell), FA3 on SM90 (Hopper), FA2 otherwise." By default vLLM tries `FLASH_ATTN`, then `FLASHINFER`, then `TRITON_ATTN` on compute capability 8.x and 9.x, which includes this laptop's Ada GPU, and tries `FLASHINFER` first on Blackwell.

[FlashInfer](https://arxiv.org/abs/2501.01005) (Ye et al., MLSys 2025) stores every cache layout in a block-sparse format and adds "load-balanced scheduling" that stays "compatible with CUDAGraph which requires static configuration": Step 4's split-count question, answered at runtime for batches of mixed lengths. It reports 29 to 69% lower inter-token latency than compiler backends in serving benchmarks.

## Common misconceptions

- **"A good prefill kernel is a good decode kernel."** They hit different limits. The prefill kernel ran at 92% of the compute peak; the same loop with one program per head moved 60 GB/s in decode, 24% of the memory ceiling. Splitting the grid over the sequence fixed it with no change to the math.
- **"Naive attention is slow because it does more math."** It does the same 51.5 GFLOP at 4,096 tokens. Its matrix multiplies took 4.8 of 31 ms; the rest moved the score matrix, 6.81 GB of DRAM traffic against the fused kernel's 16.8 MB.
- **"Out of memory is the worst case."** On this WSL2 laptop the 8,192-token naive run didn't fail: memory beyond the 7.4 GB free came from system memory, and the call took 359 ms instead of the 124 ms that four times the 4,096-token time predicts, with no error to flag it.
- **"Paged attention is slow because pages are scattered."** Pages of 256 tokens cost the same as pages of 16. The cost came from the dependent block-table load shortening the pipeline, and one more stage removed it.

## Key takeaways

- Textbook attention spent 85% of its time moving the n × n scores through memory, and it needed 2 GB of scratch at 4,096 tokens and more than the GPU had at 8,192.
- A Triton FlashAttention kernel, with tiles, online softmax in `exp2`, and causal block skipping, reached 24 TFLOPS, 90% of peak and 97 to 98% of PyTorch's FlashAttention-2 from 4,096 tokens up, using megabytes instead of gigabytes.
- Prefill attention gains intensity with length, about 0.43 FLOPs per byte per token here. Decode attention stays at 6 FLOPs per byte, so bandwidth sets its speed.
- One program per head used 2 of 24 SMs and moved 60 GB/s. Splitting the cache into 8 chunks and merging by log-sum-exp reached 239 to 246 GB/s at long contexts, 96 to 99% of the ceiling.
- A block table costs one dependent load per tile: under 1% at long contexts, and nothing once the pipeline gets one more stage.

## What's next

Decode now reads the KV cache and the weights at full bandwidth, so going faster means reading fewer bytes. [Part 11](/blog/low-precision-kernels/), Low-precision kernels, does that: weight-only INT4 and INT8 matrix multiplies that dequantize inside the kernel, FP8 with scaling factors, the microscaling formats, and measurements of when quantization actually speeds things up on this laptop.

## Further reading

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Tri Dao, 2023)](https://arxiv.org/abs/2307.08691): the loop order, the parallelism over query blocks, causal block skipping and the FLOP-counting convention used here.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135): tiling, online softmax and recomputation in the backward pass.
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah et al., 2024)](https://arxiv.org/abs/2407.08608): warp specialization, TMA and FP8 on Hopper.
- [Flash-Decoding for long-context inference (Dao, Haziza, Massa and Sizov, 2023)](https://crfm.stanford.edu/2023/10/12/flashdecoding.html): split-KV decoding and the log-sum-exp merge.
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving (Ye et al., 2025)](https://arxiv.org/abs/2501.01005): block-sparse cache formats and load-balanced decode scheduling.
- [Fused Attention tutorial (Triton docs)](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html): a complete FlashAttention-2 forward and backward pass in Triton.
- [torch.nn.functional.scaled_dot_product_attention (PyTorch docs)](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html): SDPA's backends, `sdpa_kernel` and `enable_gqa`.
- [FlashAttention repository (Dao-AILab)](https://github.com/Dao-AILab/flash-attention): `flash_attn_with_kvcache`, paged KV through `block_table`, and the Triton backend for AMD GPUs.
- [vLLM design docs: attention backends](https://docs.vllm.ai/en/latest/design/attention_backends.html): which kernel a serving engine picks on each GPU, and the block sizes each supports.
- [NVIDIA Ada GPU Architecture whitepaper (NVIDIA)](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf): the tensor-core rates behind the peaks quoted here.
