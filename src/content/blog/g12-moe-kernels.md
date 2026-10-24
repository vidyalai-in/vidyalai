---
title: "Mixture-of-Experts Kernels: Routing, Grouped GEMM and Load Imbalance"
description: "Route tokens, sort them by expert and run 128 experts in one grouped GEMM: a Triton MoE kernel timed against loops, padding and torch.grouped_mm."
slug: "moe-kernels"
category: "gpu-kernels"
tags: ["gpu", "moe", "triton", "latency"]
series: "llm-gpu"
seriesOrder: 12
pubDate: 2026-10-24
---

## TL;DR

A mixture-of-experts (MoE) layer sends each token to a few of its many expert networks, so the GPU faces dozens of small matrix multiplications of different sizes instead of one big one. This article runs one real layer, Qwen3-30B-A3B's 128 experts with 8 chosen per token, four ways on a laptop RTX 4060: a Python loop over experts, padded batches, PyTorch's `grouped_mm`, and a Triton grouped GEMM in the style of vLLM's fused MoE kernel, which sorts token-expert pairs by expert and runs every expert in one launch per matrix. In a CUDA graph it ran 64 tokens 12.6 times faster than the loop, at 95% of the memory-bandwidth limit, and with SwiGLU fused in it ran 8,192 tokens at 77% of the measured tensor-core limit. You'll see why an MoE layer stays memory-bound up to about 1,600 tokens, why skewed routing barely slows one GPU but made the busiest GPU of a simulated expert-parallel group take 6.8 times as long as balanced routing, and how DeepGEMM and DeepEP handle this on data center GPUs.

## Why it matters

Here is one mixture-of-experts layer of Qwen3-30B-A3B, run for 64 tokens on a laptop GPU the way the eager code in Hugging Face Transformers 5.20.2 writes it: a Python loop over the experts. It takes 64 ms, and the GPU is busy for only 14 ms of that. Rewritten as one GPU kernel per matrix multiply, the same layer takes 5.1 ms, and 4.8 ms of that is the time needed to read 1.2 GB of expert weights from memory once.

Most of the largest open models are MoE models ([Mixture of Experts](/blog/mixture-of-experts/) explains the design). Qwen3-30B-A3B stores 30.5 billion parameters but uses 3.3 billion per token, per its [model card](https://huggingface.co/Qwen/Qwen3-30B-A3B). The GPU only sees that saving if it runs 128 differently sized matrix multiplies about as efficiently as one large one, which is the job of this part's kernels.

## The intuition

Think of a parcel depot with 128 delivery vans, where each parcel goes onto 8 particular vans. The slow way calls each van in turn and has it search the whole pile. A second way gives every van a crate as big as the busiest van's load and fills the gaps with empty boxes. Real depots use a sorting belt: parcels are sorted by van onto one conveyor, with a divider between sections, and all the vans load at once.

Those are the Python loop, the padded batch and the grouped GEMM. Sorting is cheap next to driving, and one shared belt keeps every loading bay busy.

The analogy breaks on what is heavy. In a depot the parcels are the cargo. On a GPU the expensive thing to move is often the van itself: each expert's weights must be read from memory before any token can use them, and with few tokens that read costs far more than the arithmetic.

## Step 1: one MoE layer and its router

The shapes come from Qwen3-30B-A3B's [config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json): `hidden_size` 2,048, `num_experts` 128, `num_experts_per_tok` 8, `moe_intermediate_size` 768 and `norm_topk_prob` true. Each expert is a SwiGLU MLP, the gated design from [Fundamentals Part 10](/blog/modern-llm-architecture/): a gate and an up projection from 2,048 to 768 numbers, `silu(gate) × up`, and a down projection back to 2,048.

```text
one expert:     3 × 2,048 × 768        = 4,718,592 weights = 9.44 MB in BF16
one layer:      128 experts            = 1.21 GB
whole model:    48 layers              = 58 GB   (too big for 8 GB, so one layer, random weights)
per token:      8 experts × 2 FLOPs × 4,718,592 = 75.5 million FLOPs
```

A FLOP is one floating-point operation; [Math Part 15](/blog/counting-flops-scaling-laws/) derives the two FLOPs per weight per token. The weights are random BF16 tensors with these shapes, stored as Transformers stores them: one `(128, 1,536, 2,048)` tensor for all gate and up projections and one `(128, 2,048, 768)` tensor for the down projections.

### The router

The router is a 2,048 × 128 matrix. Transformers' `Qwen3MoeTopKRouter` turns its output into expert choices like this:

```python
def router(x, wr, k=TOPK):
    logits = F.linear(x, wr)                                   # (T, E)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    top_w, top_i = torch.topk(probs, k, dim=-1)                # (T, k)
    top_w /= top_w.sum(dim=-1, keepdim=True)                   # norm_topk_prob
    return top_w.to(x.dtype), top_i
```

`top_i` holds 8 expert numbers per token and `top_w` their weights, renormalized to sum to 1 ([Fundamentals Part 11](/blog/mixture-of-experts/) works one through by hand).

Every number here comes from an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, 24 SMs) under WSL2, driver 592.82, PyTorch 2.11.0 with CUDA 12.8, and Triton 3.6.0. Each call was timed with CUDA events after flushing the 32 MB L2 cache and synchronizing, so the CPU's launch work counts, as in a real forward pass; times are medians of 15 to 400 calls after warm-up. Kernel counts come from `torch.profiler` and DRAM bytes from Nsight Compute 2025.1.1 (which locks clocks, so it wasn't used for timing). Small batches ran at a 2,175 MHz SM clock with memory mostly at 8,000 MHz; heavy loads at 1,800 to 2,175 MHz with memory at 7,000 MHz, against a 47 to 50 W power limit. The reference roofs are [Part 4](/blog/gpu-profiling-roofline/)'s measured 248.7 GB/s and 25.3 BF16 TFLOPS (trillions of FLOPs per second).

The router took 134 µs (microseconds) for one token, but its 8 kernels ran for only 19 µs: the rest was the CPU launching them ([Part 1](/blog/gpu-from-python-to-silicon/)). A Triton kernel doing softmax, top-k and renormalization in one pass cut it to 3 kernels and 114 µs, with identical choices; vLLM ships that fusion as a CUDA kernel, `csrc/moe/topk_softmax_kernels.cu`, adapted from TensorRT-LLM. At 8,192 tokens the router took 0.61 ms, under 2% of the layer.

## Step 2: sorting tokens by expert

The router's output is `T × 8` token-expert pairs, and each expert needs its pairs side by side. A toy layer with 4 tokens, top-2 of 4 experts, padded to blocks of 2 rows:

| Pair id (token × 2 + slot) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Token | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| Expert chosen | 2 | 0 | 1 | 2 | 2 | 3 | 0 | 2 |

Sorted by expert, each group padded to a multiple of 2 with the dummy id 8, the code below produces:

```text
sorted_ids  [1, 6 | 2, 8 | 0, 3 | 4, 7 | 5, 8 | 8, 8]
expert_ids  [0,     1,     2,     2,     3,     3   ]    padded total: 10 rows
```

Expert 0 gets pairs 1 and 6 (tokens 0 and 3), expert 2 two full blocks, and every block belongs to exactly one expert; the last block lies past the padded total and is skipped. A histogram counts pairs per expert, a prefix sum turns counts into starting rows, and a sort orders the pairs. In PyTorch, mirroring vLLM's `moe_align_block_size`:

```python
def align_block_size(top_i, block_m):
    flat = top_i.reshape(-1)
    S = flat.numel()
    dev = flat.device
    e_sorted, order = torch.sort(flat, stable=True)
    counts = torch.zeros(E, device=dev, dtype=torch.int64).scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + block_m - 1) // block_m * block_m
    pad_end = torch.cumsum(padded, 0)
    pad_start = pad_end - padded
    start = torch.cumsum(counts, 0) - counts
    dest = pad_start[e_sorted] + torch.arange(S, device=dev) - start[e_sorted]
    max_pad = S + E * (block_m - 1)
    sorted_ids = torch.full((max_pad,), S, device=dev, dtype=torch.int32)
    sorted_ids[dest] = order.to(torch.int32)
    blocks = torch.arange(max_pad // block_m, device=dev) * block_m
    expert_ids = torch.searchsorted(pad_end, blocks, right=True).clamp_(max=E - 1).to(torch.int32)
    return sorted_ids, expert_ids, pad_end[-1:].to(torch.int32)
```

Every output has a fixed size, `S + E × (block_m − 1)`, the most padding possible, so the CPU never reads a count back from the GPU. That would force a synchronization, a stall until the GPU catches up, and rule out CUDA graphs, which replay recorded launches with fixed sizes ([Part 9](/blog/kernel-fusion-cuda-graphs/)).

### What moving the tokens costs

Permutation copies each token's row into expert order (a gather); after the experts, each output row goes back to its token, times its routing weight, and the 8 rows per token are added (unpermute and combine). As separate passes over memory:

| Step | 2,048 tokens | 8,192 tokens |
|---|---|---|
| Permute: sort, gather 65,536 rows, histogram, prefix sum | 0.61 ms | 2.32 ms |
| Unpermute with `index_add_` (a scatter-add with atomic adds) | 2.46 ms | 11.6 ms |
| Unpermute as a gather back to token order, then a sum | 1.58 ms | 6.01 ms |
| The two grouped matrix multiplies, for comparison (profiler) | 12.3 ms | 31.0 ms |

With `index_add_`, moving rows cost 25% (2,048 tokens) to 45% (8,192 tokens) on top of the matrix multiplies; with the gather-and-sum version, 18% to 27%. Step 4's kernel removes almost all of it by reading token rows in place through `sorted_ids` and writing each output row straight to its token-and-slot position. Only the sort (0.16 ms of GPU time at 8,192 tokens) and the final sum (1.4 ms) remain.

![A left-to-right flow. Tokens, T by 2,048, go to the router, which computes logits, softmax, top-8 and renormalized weights. The top-8 ids and weights go to an align step that sorts token-expert pairs by expert and pads each group to BLOCK_M. Its sorted_ids and expert_ids feed a box labeled grouped GEMMs, one launch each: GEMM 1 gathers token rows, computes gate and up and applies SwiGLU in its epilogue, and GEMM 2 does the down projection and multiplies by the routing weight. A second arrow from the tokens to GEMM 1 is labeled rows read in place. GEMM 2 outputs T times 8 rows to a combine step that sums 8 rows per token, giving the output, T by 2,048.](/diagrams/g12-moe-pipeline.svg "One MoE layer as a fused pipeline: route, sort, two grouped GEMMs, combine.")

The data never moves into expert order: the GEMMs reach into the original token rows through the sorted index list.

## Step 3: four ways to run the experts

Each version computes the same output (all agreed with an FP32 reference to within BF16 rounding, about 1% of the largest value) from the same router choices:

- (a) A Python loop over the experts hit, copied from Transformers' `Qwen3MoeExperts.forward`: `torch.where` to find the tokens, two `F.linear` calls, `index_add_` into the output.
- (b) Pad every expert to the busiest one's count in a `(128, cap, 2,048)` buffer and run two `torch.bmm` batched multiplies over all 128 experts.
- (c) Step 4's Triton grouped GEMM, structured as vLLM does it: align, one launch per matrix, SwiGLU between them, a sum at the end. Also timed as a CUDA graph.
- (d) `torch.nn.functional.grouped_mm` with sort, gather and `index_add_` around it, as in Transformers' `grouped_mm` option.

| Tokens | Experts hit | (a) loop | (b) padded `bmm` | (c) Triton | (c) in a CUDA graph | (d) `grouped_mm` | Floor |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 4.14 ms | 5.55 ms | 0.98 ms | 0.46 ms | 1.02 ms | 0.30 ms (memory) |
| 4 | 30 | 15.1 | 5.60 | 1.76 | 1.29 | 1.91 | 1.14 (memory) |
| 16 | 83 | 41.1 | 6.36 | 3.79 | 3.31 | 4.53 | 3.15 (memory) |
| 64 | 127 | 63.9 | 6.58 | 5.53 | 5.06 | 6.63 | 4.82 (memory) |
| 512 | 128 | 64.7 | 8.02 | 6.00 | | 7.71 | 4.87 (memory) |
| 2,048 | 128 | 72.5 | 17.7 | 10.8 | | 16.8 | 6.11 (compute) |
| 8,192 | 128 | 102 | 63.4 | 35.4 | | 49.3 | 24.4 (compute) |

The floor is the larger of two times: the bytes that must move (the weights of every expert hit, plus input and output) at 248.7 GB/s, and the FLOPs at 25.3 TFLOPS.

### Why the loop loses

The loop launched 120 kernels for one token, 1,786 for 64 and 1,928 for 8,192, about 14 per expert, and `torch.where` with one argument forces a synchronization for every expert. For 64 tokens its kernels ran for 14.3 ms of its 63.9 ms, so the GPU idled 78% of the time; in the 512-token run the idle GPU even dropped its clock to 1,365 MHz. This is [Part 1](/blog/gpu-from-python-to-silicon/)'s launch-overhead problem multiplied by 128.

### Why padding loses

The padded version launches only 27 to 36 kernels, but `bmm` multiplies all 128 experts whether or not they got tokens. For one token it read all 1.21 GB of weights to use 75.5 MB of them, 16 times more traffic than needed, and took 5.55 ms. With many tokens it wastes arithmetic on padding instead: at 8,192 tokens the busiest expert had 590 pairs against an average of 512, so the padded buffer held 15% more rows than there were pairs.

### Distinct experts and the bandwidth floor

For small batches what matters is how many different experts the batch touches, at 9.44 MB each. With random routing, `T` tokens should touch `128 × (1 − (120/128)^T)` experts: 29 for 4 tokens, 82 for 16, 126 for 64, against 30, 83 and 127 measured. By 64 tokens the batch reads nearly the whole layer, so the time hardly moves from 64 to 512 tokens (5.53 and 6.00 ms). Nsight Compute confirmed the bytes: the two Triton kernels read 75.5 MB from DRAM for one token, exactly 8 experts, and 1.20 GB for 64.

The Triton version in a CUDA graph ran at 95% of the bandwidth floor from 16 tokens up. For one token it reached 66%. There are only 8 real rows, and in the run profiled with Nsight Compute the first GEMM launched just 96 programs (Triton's blocks of work) for 24 SMs, too few to keep enough memory requests in flight.

### When does it become compute-bound?

Once every expert is hit, the layer reads 1.21 GB and does `T × 75.5` million FLOPs, about `T/16` FLOPs per byte. Part 4 put this GPU's ridge point, where the compute and bandwidth limits meet, at 102 FLOPs per byte, so the layer turns compute-bound near `T = 1,600` tokens. A dense layer does 1 FLOP per byte per token and crosses at about 100. The factor of 16 is `128 / 8`: each expert sees one-sixteenth of the batch, which is why MoE serving wants big batches.

### What `grouped_mm` does on this GPU

`torch.nn.functional.grouped_mm` ran correctly on this compute-capability 8.9 GPU, but not as one kernel. In PyTorch 2.11's [GroupedBlas.cpp](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/GroupedBlas.cpp), the fast path (commented "no d2h sync needed") is enabled only for SM90 and SM100, Hopper and Blackwell. Everything else takes `_grouped_mm_fallback`, which copies the group offsets to the CPU and calls cuBLAS once per group: 254 GEMM launches for 64 tokens. It beat the loop everywhere and padding at all but one size, but can't go in a CUDA graph.

## Step 4: a grouped GEMM in Triton

A **grouped GEMM** is many independent matrix multiplies (GEMMs) in one launch. In an MoE layer all experts share N and K, the output and input widths; only M, the row count, differs. vLLM's `fused_moe_kernel` in [vllm/model_executor/layers/fused_moe/fused_moe.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) exploits that: sorting and padding `sorted_token_ids` "ensures divisibility by BLOCK_SIZE_M," its docstring says, so each block of rows has one expert, and `expert_ids` holds "the indices of the expert for each block." Here is a simplified version, without quantization, bias or expert-parallel masking, built like [Part 7](/blog/triton-kernels/)'s matmul:

```python
@triton.jit
def moe_gemm_kernel(a_ptr, b_ptr, c_ptr, w_ptr, sorted_ids_ptr, expert_ids_ptr, n_padded_ptr,
                    N, K, num_valid, stride_am, stride_be, stride_bn, stride_cm, M_HINT,
                    TOP_K: tl.constexpr, MUL_W: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)                                      # N tiles vary fastest: programs that run
    pid_m = tl.program_id(1)                                      # together share one block of token rows
    if pid_m * BLOCK_M >= tl.load(n_padded_ptr):
        return                                                    # past the last padded block
    rows = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))   # (token, slot) pair ids
    valid = rows < num_valid                                      # padding slots hold num_valid
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)              # every row in a block has one expert
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (rows // TOP_K).to(tl.int64)[:, None] * stride_am + rk[None, :]   # gathered token rows
    b_ptrs = b_ptr + e * stride_be + rn[None, :] * stride_bn + rk[:, None]            # expert e's weights, K x N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
        acc = tl.dot(a, tl.load(b_ptrs), acc)                     # tensor cores
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    if MUL_W:                                                     # routing weight, fused into the epilogue
        acc *= tl.load(w_ptr + rows, mask=valid, other=0.0)[:, None]
    c_ptrs = c_ptr + rows.to(tl.int64)[:, None] * stride_cm + rn[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=valid[:, None])
```

Each program reads `BLOCK_M` pair ids, looks up their expert, and from then on is an ordinary tiled matmul whose B pointer starts at that expert's weights. Three lines do the MoE-specific work:

- `rows // TOP_K` turns pair ids back into token rows, so the first GEMM gathers tokens straight from the input. The second passes `TOP_K=1`, since its input already has one row per pair.
- `valid` masks the padding rows, which load zeros and store nothing.
- The store writes each row at the pair's own position, in token-and-slot order, ready for the final sum.

The grid has one program per column tile and row block, capped at `T × 8 × BLOCK_M` rows as in vLLM; programs past the padded total return at once. `M_HINT`, the token count rounded up to a power of two, is only the autotuning key, as vLLM keys its configs on batch size. `N` and `K` must be multiples of the tile sizes, true for 768, 1,536 and 2,048.

### Padding to the block size

Every expert's group is rounded up to a multiple of `BLOCK_M`, and padded rows cost full arithmetic. The tile height trades that padding against efficiency:

| Tokens | `BLOCK_M` | Rows after padding | Time | TFLOPS |
|---|---|---|---|---|
| 2,048 | 16 | 1.06 × | 12.4 ms | 12.5 |
| 2,048 | 32 | 1.12 × | 10.7 ms | 14.5 |
| 2,048 | 128 | 1.48 × | 11.7 ms | 13.2 |
| 8,192 | 16 | 1.01 × | 43.8 ms | 14.1 |
| 8,192 | 64 | 1.06 × | 34.5 ms | 17.9 |
| 8,192 | 128 | 1.13 × | 34.3 ms | 18.0 |

At 2,048 tokens each expert averages 128 pairs, so 128-row tiles leave many experts with a second, nearly empty block. vLLM's default picks 128 above 512 tokens; this article uses 16, 32 or 64 by batch size. vLLM's tuned H200 file for this shape, `E=128,N=768,device_name=NVIDIA_H200.json`, picks 16 × 64 tiles with 5 pipeline stages for one token and 128 × 256 tiles with 8 warps for 4,096 tokens.

### Fusing more into the kernel

Three fusions, each measured:

- SwiGLU in the epilogue, the code after the main loop. A variant computes gate and up tiles for the same columns in two accumulators and stores `silu(gate) × up`, half the bytes of the plain GEMM's output:

```python
    h = acc_g * tl.sigmoid(acc_g) * acc_u                          # silu(gate) * up, in FP32
    c_ptrs = c_ptr + rows.to(tl.int64)[:, None] * stride_cm + rn[None, :]
    tl.store(c_ptrs, h.to(c_ptr.dtype.element_ty), mask=valid[:, None])
```

  At 8,192 tokens the first GEMM plus a separate SwiGLU took about 22.3 + 2.2 ms of GPU time; fused, 19.9 ms. The whole layer went from 35.4 to 31.8 ms, and from 10.8 to 10.0 ms at 2,048 tokens. Up to 64 tokens the two versions were within 0.1 ms of each other, because reading weights dominates.
- The routing weight in the second GEMM's epilogue (`MUL_W`), which saves a pass over a 65,536 × 2,048 tensor. Doing it and the SwiGLU in FP32 before rounding likely explains why the fused versions' largest errors were 0.4 to 0.8 times the others'.
- The whole layer in a CUDA graph. With no host synchronization, capture just works; for one token it cut 0.98 ms to 0.46 ms. Most of the 34 kernels replayed are the PyTorch align step, which vLLM does in one CUDA kernel (`csrc/moe/moe_align_sum_kernels.cu`).

At 8,192 tokens the fused version ran at 19.5 TFLOPS, 80% of the tensor-core peak at its 1,980 MHz clock. cuBLAS, given the same FLOPs as two dense matmuls (65,536 rows through one expert), took 26.3 ms against the grouped kernels' 29.8 ms: the gather, the padding and the down projection's short `K = 768` loop cost about 13%.

## Step 5: load imbalance

A real router doesn't spread tokens evenly. To test skew, the same 4,096 tokens were routed six ways, with 8 distinct experts per token every time: perfectly balanced, the random-weight router, Zipf distributions (expert `r` chosen with probability proportional to `1/r^s`), 4 hot experts that every token picks, and full collapse onto experts 0 to 7. "Busiest / mean" is the busiest expert's pair count over the average of 256.

| Routing | Busiest / mean | (b) padded `bmm` | (c) Triton, one GPU | Simulated 8 GPUs: busiest group | Mean group |
|---|---|---|---|---|---|
| Balanced | 1.00 | 28.9 ms | 15.3 ms | 2.22 ms | 2.21 ms |
| Router, random inputs | 1.23 | 32.2 | 16.7 | 2.68 | 2.50 |
| Zipf, s = 0.5 | 5.31 | 93.3 | 17.0 | 3.21 | 2.52 |
| Zipf, s = 1.0 | 13.15 | 260 | 17.5 | 4.02 | 2.51 |
| 4 hot experts | 16.00 | 5,261 | 17.4 | 3.53 | 2.55 |
| Collapse onto 8 experts | 16.00 | 5,261 | 14.4 | 15.07 | 1.88 |

On one GPU the grouped kernel barely noticed: 15.3 to 17.5 ms, with collapse the fastest because only 8 experts' weights had to be read. The busiest expert's tiles are just more tiles in one shared grid, and the 24 SMs take whatever block is next, from any expert. The small losses likely come from padding each group to 64 rows and from wave quantization, a last round of blocks that fills only part of the 24 SMs.

The padded version's work is 128 experts times the busiest expert's count, so a 13-fold imbalance made it 9 times slower. At 4,096 pairs for one expert its buffers needed about 7.5 GB on top of 1.2 GB of weights, more than the card holds, and each call took 5.3 seconds (consistent with the Windows driver spilling into system memory; not investigated further).

Expert parallelism behaves differently. The last two columns simulate 8 GPUs holding 16 experts each, running each GPU's share alone on this GPU, back to back so the clock stayed up. A layer finishes when the busiest GPU does. Zipf with s = 1.0 put 1.79 times the average load on one group and took 1.81 times as long as balanced routing; collapse onto one GPU's experts took 6.8 times as long while the other seven idled. That is the effect [Inference Part 10](/blog/model-parallelism/) described, measured.

Training fights skew with the load-balancing loss ([Math Part 15](/blog/counting-flops-scaling-laws/); Qwen3-30B-A3B sets `router_aux_loss_coef` to 0.001). The [Switch Transformer](https://arxiv.org/abs/2101.03961) also capped each expert at a fixed **capacity**, `(tokens / experts) × capacity factor`, because, in the paper's words, "all of our tensor shapes are statically determined at compilation time." Tokens past the cap are dropped: "computation is skipped and the token representation is passed directly to the next layer through the residual connection." That is the padded `bmm` with a bounded `cap`, trading dropped tokens for fixed shapes; grouped GEMMs remove the need for it on one GPU. Across GPUs, serving systems copy hot experts instead ([Inference Part 14](/blog/deepseek-inference-deepinfer/) covers DeepSeek's EPLB).

## In the real world

- vLLM's Triton `fused_moe_kernel` is the design measured here, plus FP8, INT8 and INT4 variants. Its `configs` folder holds 337 tuned JSON files, named by expert count, intermediate size and GPU, that map batch size to tiles, warps and stages; [benchmark_moe.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) `--tune` generates them. None is for an RTX 4060, so vLLM would use `get_default_config` here.
- SGLang carries the same `fused_moe_kernel` in [fused_moe_triton_kernels.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py), with its own tuning script.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) requires an "NVIDIA SM90 or SM100 architecture GPU", so it couldn't run here. Unlike CUTLASS's general grouped GEMMs, it "groups only the M-axis, while N and K must remain fixed." Its **contiguous** layout concatenates each expert's tokens for prefill, and "each expert segment must be aligned to the GEMM M block size," which is the padding of Step 2. Its **masked** layout is for decode, "when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives"; a mask tensor tells the kernel which portions are valid, so the CPU never needs the counts. Its Mega MoE fuses dispatch, both GEMMs, SwiGLU and combine into one kernel, overlapping NVLink traffic with tensor-core work.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) provides "high-throughput and low-latency expert-parallel (EP) all-to-all GPU kernels for MoE dispatch and combine," the cross-GPU version of Step 2's permutation; its low-latency output feeds DeepGEMM's masked layout. [Part 13](/blog/nccl-collectives/) covers all-to-all communication.
- [MegaBlocks](https://arxiv.org/abs/2211.15841) (Gale et al., 2022) reformulated MoE "in terms of block-sparse operations" because frameworks "must choose between dropping tokens from the computation or wasting computation and memory on padding." It "never drops tokens" and reported training "up to 40%" faster than Tutel.
- On Hopper and Blackwell, PyTorch's `grouped_mm` takes its single-kernel path. Transformers 5.20.2 lets you choose `eager`, `batched_mm` or `grouped_mm` experts per model, and defaults to `grouped_mm` when it can dispatch it, falling back to the eager loop otherwise. On this laptop that default means version (d): PyTorch's per-group fallback.

## Common misconceptions

- **"An MoE layer with 8 of 128 experts active reads 1/16 of its weights."** Per token, yes. A batch reads every expert any token chose: 64 tokens touched 127 of 128 experts and read 1.20 GB, all but one expert of the layer.
- **"Running each expert as its own matmul is fine."** In eager PyTorch that loop launched 1,786 kernels for 64 tokens and synchronized per expert, leaving the GPU idle 78% of the time.
- **"Padding to equal sizes is the GPU-friendly way."** Padding to the busiest expert made skewed routing 9 times slower (and 180 times once its buffers outgrew GPU memory), and read all 128 experts for one token. Padding only to a tile height of 16 to 64 rows cost 1% to 24% extra rows.
- **"Load imbalance slows down every MoE kernel."** On one GPU, the grouped kernel lost at most 14% under extreme skew. Imbalance hurts across GPUs, where the busiest one sets the pace.
- **"Permutation is negligible next to the GEMMs."** Done as separate passes, gather and scatter-add cost 25% to 45% on top of the GEMMs. Folded into the GEMM's loads and stores, it nearly disappears.

## Key takeaways

- An MoE layer is a router, a sort of token-expert pairs by expert, two grouped GEMMs and a weighted sum. Sorting is cheap; moving rows in separate passes cost up to 45% of the GEMM time.
- A grouped GEMM runs every expert in one launch: each block of rows has one expert, and the kernel gathers token rows on load and scatters results on store.
- With small batches, time follows the distinct experts touched, 9.44 MB each. By 64 tokens the layer reads all 1.2 GB, and it stays memory-bound to about 1,600 tokens, 16 times longer than a dense layer.
- The Triton kernel reached 95% of the bandwidth floor at 16 to 64 tokens in a CUDA graph and 80% of the tensor-core peak at 8,192 tokens with SwiGLU fused in.
- Skewed routing barely slows a grouped GEMM on one GPU but destroys padded batching, and with expert parallelism the busiest GPU sets the step time.

## What's next

Expert parallelism only works if tokens can travel to their experts' GPUs and back fast enough. Part 13, Many GPUs: NVLink, NCCL and collectives, covers how GPUs exchange data: ring and tree all-reduce and their cost model, the all-to-all that MoE dispatch needs, NVLink and NVSwitch against PCIe and InfiniBand, and overlapping communication with computation.

## Further reading

- [vLLM fused_moe.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) and [moe_align_block_size.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/moe_align_block_size.py) (vLLM): the Triton fused MoE kernel, its default configs and the sort-and-pad step, with a worked example in the docstring.
- [vLLM tuned MoE configs](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe/configs) and [benchmark_moe.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py): per-GPU tile choices by batch size, and how to generate them.
- [DeepGEMM (DeepSeek)](https://github.com/deepseek-ai/DeepGEMM): contiguous and masked grouped GEMMs for Hopper and Blackwell, and Mega MoE.
- [DeepEP (DeepSeek)](https://github.com/deepseek-ai/DeepEP): all-to-all dispatch and combine kernels for expert parallelism.
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (Gale, Narayanan, Young and Zaharia, 2022)](https://arxiv.org/abs/2211.15841): block-sparse MoE kernels without token dropping or padding.
- [Switch Transformers (Fedus, Zoph and Shazeer, 2021)](https://arxiv.org/abs/2101.03961): expert capacity, capacity factor and dropped tokens on static-shape hardware.
- [PyTorch GroupedBlas.cpp, v2.11.0](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/GroupedBlas.cpp): where `grouped_mm` picks its fast path or the per-group fallback.
- [Transformers MoE integrations](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/moe.py) and [Qwen3-MoE modeling code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py): the eager loop, `batched_mm` and `grouped_mm` experts.
- [Qwen3-30B-A3B config.json (Qwen)](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json): the shapes used throughout.
- [SGLang fused MoE Triton kernels](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py): the same design in another serving engine.
