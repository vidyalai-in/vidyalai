---
title: "Triton: Writing GPU Kernels in Python, Measured"
description: "Write vector add, fused softmax and a tensor-core matmul in Triton, autotune them, read the PTX they compile to, and time them against PyTorch."
slug: "triton-kernels"
category: "gpu-kernels"
tags: ["gpu", "cuda", "latency", "triton"]
series: "llm-gpu"
seriesOrder: 7
pubDate: 2026-10-24
---

## TL;DR

Triton is a Python language and compiler for GPU kernels. You write the code for one block of work, a program, as operations on whole tensors of indices, and the compiler decides what each thread does. On a laptop RTX 4060, a six-line Triton vector add moved 231 GB/s, 90% of the rated bandwidth and as fast as `torch.add`. A 12-line fused softmax ran 1.4 to 1.8 times faster than `torch.softmax` on the BF16 shape [Part 4](/blog/gpu-profiling-roofline/) flagged as slow, and fell to a quarter of its speed once a row no longer fit in registers. A 30-line tensor-core matmul beat cuBLAS at small sizes and matched it at large ones. You'll see how autotuning picks tile sizes, how changing the order of programs cut DRAM traffic 6.7 times, and what the compiler wrote at each stage, down to the `mma.sync` instructions.

## Why it matters

[Part 5](/blog/cuda-matmul-optimization/) took eight versions of a CUDA kernel to match cuBLAS in FP32. Here is a matmul that runs on tensor cores, pipelines its memory loads and orders its blocks for cache reuse, in 30 lines of Python. At n = 8,192 in BF16 it ran at 99.0% of the tensor-core peak at its clock, against cuBLAS's 98.5%.

`torch.compile` writes Triton kernels for you, and LLM training libraries ship hand-written ones. Knowing what Triton does for you, and where it stops, tells you when Python is enough and when to drop to CUDA.

## The intuition

CUDA is managing a team by writing instructions for each worker: "worker 37, fetch item 37, add, put it back." You also decide when they wait for each other and where they put things on the shared table.

Triton is writing one instruction for the whole team: "this team handles items 1,024 to 2,047: fetch, add, store." The compiler splits the work among the members, picks how many items each carries per trip, and puts in the waits.

The analogy breaks at one point. A real manager can still step in and direct one worker. In Triton you can't address a single thread or warp, so tricks that rely on threads trading values directly, or on a hand-chosen layout in shared memory, are out of reach.

## Step 1: programs, offsets and masks

A **Triton kernel** is a Python function decorated with `@triton.jit`. Launching it starts a grid of **programs**, each running as one CUDA thread block. There are no thread indices: a program asks which one it is with `tl.program_id`, builds a tensor of the indices it owns with `tl.arange`, and loads, computes and stores whole tensors. Philippe Tillet, H. T. Kung and David Cox introduced this model in a [2019 paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf): it is CUDA's, except that "each kernel is single-threaded," yet "automatically parallelized."

Here is Part 2's vector add in Triton, close to the official tutorial:

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                          # which block this program owns
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # a tensor of BLOCK_SIZE indices
    mask = offsets < n                                   # the last block may overhang n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

n = 1 << 26                                              # 67,108,864 floats, as in Part 2
x = torch.rand(n, device="cuda"); y = torch.rand(n, device="cuda"); out = torch.empty_like(x)
grid = (triton.cdiv(n, 1024),)                           # 65,536 programs
k = add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
assert torch.equal(out, x + y)
ms = triton.testing.do_bench(lambda: add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024),
                             warmup=1000, rep=1000, return_mode="median")  # 1 s warm-up, 1 s timed
print(f"{ms:.3f} ms, {3 * n * 4 / (ms * 1e6):.1f} GB/s, {k.n_regs} registers per thread")
```

```text
3.488 ms, 230.9 GB/s, 26 registers per thread
```

`x_ptr + offsets` is a tensor of 1,024 pointers. The **mask**, 1,024 true-or-false values, switches off loads and stores past the end of the array, the job `if (i < n)` did in CUDA. `BLOCK_SIZE` is a `tl.constexpr`, a compile-time constant, because tensor shapes must be known when the kernel is compiled. The first launch compiles the kernel for these argument types and constants and caches it on disk.

### How it was timed

Everything ran on [Part 1](/blog/gpu-from-python-to-silicon/)'s machine: an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, 24 SMs, compute capability 8.9) under WSL2, driver 592.82, PyTorch 2.11.0 (CUDA 12.8) and Triton 3.6.0. Times come from `triton.testing.do_bench`, which in Triton 3.6 defaults to 25 ms of warm-up calls and 100 ms of timed calls, zeroes a 256 MB buffer before each timed call to flush the 32 MB L2 cache, times each call with its own CUDA events, and returns the mean. Every number here is the median over 0.5 to 1 second, after a second of warm-up (Part 2 found the GPU needs about 0.6 seconds to leave its low-power state). `nvidia-smi` logged clocks and power every 200 ms.

### Against CUDA and PyTorch

The same 805 MB of traffic (two 268 MB reads, one write), median of three rounds:

| Kernel | Time | GB/s | % of 256 GB/s rated |
|---|---|---|---|
| Part 2's CUDA kernel, 256 threads per block | 3.555 ms | 226.5 | 88.5% |
| `torch.add(x, y, out=out)` | 3.487 ms | 231.0 | 90.2% |
| Triton, `BLOCK_SIZE` = 1,024 | 3.485 ms | 231.1 | 90.3% |

Memory stayed at 8,000 MHz, so the roof was the rated `8,000 MHz × 2 × 16 bytes = 256 GB/s` ([Part 4](/blog/gpu-profiling-roofline/)'s read-only test reached 248.7); the SM ran at 1,700 to 2,190 MHz, drawing 42 to 50 W. The CUDA kernel was timed as in Part 2, the median of 51 runs.

The compiled code differs. Part 2's kernel loads one float per thread, `LDG.E` in the machine code. Triton gave each thread 4 consecutive floats and emitted `LDG.E.128`, a 16-byte load, twice per input: the `float4` trick Part 5 wrote by hand. It could, because it noticed that every pointer and `n` were multiples of 16 (`tt.divisibility = 16` in its IR), so no 16-byte access straddles an alignment boundary.

### The block-size sweep

`BLOCK_SIZE` counts elements, not threads; the default is 4 warps (128 threads) per program. Median of three rounds:

| `BLOCK_SIZE` | 64 | 128 | 256 | 1,024 | 4,096 | 8,192 | 16,384 |
|---|---|---|---|---|---|---|---|
| GB/s | 239.0 | 227.0 | 229.5 | 231.1 | 231.7 | 235.9 | 231.6 |
| % of 256 GB/s | 93.3% | 88.7% | 89.7% | 90.3% | 90.5% | 92.1% | 90.5% |

Every size landed within 5% of the others, with no trend; 64, which compiles to 4-byte loads, was the fastest. A kernel this simple has one limit, DRAM, and almost any choice reaches it.

## Step 2: a fused softmax

Softmax turns each row into probabilities: subtract the row's maximum, exponentiate, divide by the sum ([Math Part 5](/blog/softmax-math/) derives it). As separate PyTorch operations it's five kernels, each reading its input from DRAM and writing its output back. The Triton tutorial counts "5MN + 2M elements" read and "3MN + 2M elements" written for an M × N input, against MN each way for one pass.

A **fused** kernel does every step in one pass, keeping the row on chip. With one program per row:

```python
@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)                               # one program per row
    offs = tl.arange(0, BLOCK_SIZE)                      # BLOCK_SIZE >= n_cols, a power of 2
    mask = offs < n_cols
    x = tl.load(in_ptr + row * n_cols + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)                            # subtract the row max (stability)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * n_cols + offs, y.to(out_ptr.dtype.element_ty), mask=mask)
```

The launcher rounds `BLOCK_SIZE` up to a power of two (`tl.arange` requires one) and picks 4, 8 or 16 warps as rows grow. Masked lanes load minus infinity, which adds `exp(-inf) = 0` to the sum. For `tl.max` and `tl.sum` the compiler writes the warp shuffles and shared-memory exchanges that [Part 8](/blog/gpu-reductions/) writes by hand.

### Measured against PyTorch

Every case has 2^25 (33.5 million) elements, Part 4's 8,192 × 4,096 shape and the same total at other widths, so each moves 134 MB in BF16 and 268 MB in FP32. The table is the first round, when the power limit sat near 49 W and memory stayed at 8,000 MHz (a 256 GB/s roof):

| Columns | BF16 `torch.softmax` | BF16 Triton | FP32 `torch.softmax` | FP32 Triton |
|---|---|---|---|---|
| 512 | 221.4 GB/s | 225.2 GB/s | 225.6 GB/s | 226.2 GB/s |
| 2,048 | 221.8 | 227.2 | 227.1 | 224.6 |
| 4,096 | 163.4 | 224.8 | 229.1 | 223.9 |
| 8,192 | 195.3 | 223.3 | 224.6 | 223.5 |
| 16,384 | 227.2 | 219.9 | 227.6 | 223.3 |
| 65,536 | 222.9 | 220.3 | 229.5 | 219.6 |
| 131,072 | 221.8 | 55.0 | 210.2 | 73.3 |

In two more rounds the laptop had cut its power limit to 40 W, pulling the SM clock down to 1,000 to 1,900 MHz. The Triton kernel stayed at 86% to 89% of the roof at the logged memory clock, at every width up to 65,536.

At 4,096 BF16 columns, the shape Part 4 measured at 63% of the roof, the 12-line kernel ran 1.38 times faster than `torch.softmax` (597 µs against 821 µs), then 1.53 and 1.78 times faster in the 40 W rounds. Part 4's profile showed why: PyTorch gives each row a 1,024-thread block, only one such block fits on an SM, and every row waits at block-wide barriers for its max and sum. Barrier waits tie a kernel to the SM clock, so at 40 W PyTorch's kernel fell to 49% to 53% of the roof, while the Triton kernel, three 512-thread rows per SM, stayed memory-bound. Elsewhere the two were within 4% in the first round and mostly within 6% at 40 W.

The unfused version took 4.05 times as long as the fused one in BF16 (2,416 µs) and 3.79 times in FP32. It moves 8 bytes for every 2 (5 reads and 3 writes of the matrix), and time follows bytes.

### Where it stops winning

The kernel holds a whole row in registers, so the row has to fit. At 65,536 columns each of 512 threads holds 128 values, and the compiler spilled 26 registers to local memory (thread-private memory in DRAM, cached in L1 and L2) at no visible cost. At 131,072 it spilled 472 and ran 4 times slower than PyTorch, which for rows this long switches to a kernel (`cunn_SoftMaxForward`) that streams the row three times, for the max, the sum and the output, and lets the L2 cache absorb the re-reads. More warps help but don't fix it: at 32 warps, the limit of 1,024 threads, spills fell to 174 and speed rose to 82 to 88 GB/s.

The fix is algorithmic: walk the row in chunks with a running max and sum, rescaling the sum when the max grows. That's the **online softmax** from [Math Part 5](/blog/softmax-math/), which [Part 8](/blog/gpu-reductions/) builds.

## Step 3: a matmul on tensor cores, and autotuning it

[Part 5](/blog/cuda-matmul-optimization/) tiled by hand at three levels: block tiles in shared memory, thread tiles in registers, and double buffering. In Triton you write only the first. Each program computes one `BLOCK_M × BLOCK_N` tile of C, walking along K one `BLOCK_K` slice at a time, and `tl.dot` multiplies two tiles:

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr, PRECISION: tl.constexpr):
    # Which output tile is this program's? Grouped ordering: walk GROUP_M tile-rows at a time.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * num_pid_n)
    first_m = group * GROUP_M
    group_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % (GROUP_M * num_pid_n)) % group_m
    pid_n = (pid % (GROUP_M * num_pid_n)) // group_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak    # BLOCK_M x BLOCK_K tile of A
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn    # BLOCK_K x BLOCK_N tile of B
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=rk[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=rk[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, input_precision=PRECISION)              # tensor cores
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))
```

`rm[:, None] * stride_am + rk[None, :]` is NumPy-style broadcasting: a column of row offsets plus a row of column offsets makes a 2D tile of pointers. The accumulator is FP32, and for FP16 or BF16 inputs `tl.dot` compiles to **tensor-core** instructions, which multiply small matrices in one step ([Part 6](/blog/tensor-cores/) covers them). Thread tiles, shared-memory staging and load pipelining are the compiler's job.

### Grouped ordering and the L2 cache

The first block of lines picks this program's tile. In row order, programs running at the same time share one strip of A but need every column strip of B. **Grouped ordering** walks down `GROUP_M` rows of tiles before moving right, so concurrent programs cover a squarer patch of C and reuse both inputs while they're in L2: in the tutorial's 9 × 9 example, the first 9 output tiles need 90 input tiles in row order and 54 grouped. CUTLASS calls this threadblock rasterization (Part 5).

Here it is for a BF16 matmul with n = 8,192: 4,096 programs, 128 × 128 tiles with 32-wide K slices, 4 stages, 4 warps. DRAM bytes and L2 hit rate come from Nsight Compute; times and clocks are medians of six runs with normal clocks under the 40 W limit:

| Ordering | DRAM read | L2 hit rate | Time under `ncu` | Time | TFLOPS | SM clock | % of peak at that clock |
|---|---|---|---|---|---|---|---|
| Row by row (`GROUP_M` = 1) | 9.02 GB | 48.0% | 68.66 ms | 61.9 ms | 17.8 | 1,462 MHz | 98.7% |
| Grouped (`GROUP_M` = 8) | 1.35 GB | 92.2% | 68.67 ms | 48.8 ms | 22.5 | 1,852 MHz | 99.0% |
| cuBLAS (`torch.matmul`) | 1.51 GB | n/a | 68.68 ms | 52.2 ms | 21.1 | 1,740 MHz | 98.5% |

Grouping cut DRAM reads 6.7 times (the minimum is 0.27 GB). Under Nsight Compute, which locks clocks at base, both orders took 68.7 ms: tensor cores were the limit, not DRAM. With normal clocks grouping was 1.27 times faster, and the clock accounts for the gap: both ran at 99% of the peak at their own clock and drew the full 40 W, but at 1,462 against 1,852 MHz. Pulling 7.7 GB more through DRAM costs power, and on a power-capped laptop that power comes out of the SMs' share. At n = 4,096 the effect was smaller: 417 against 168 MB, 1.07 times faster.

### Against cuBLAS

`@triton.autotune` (below) picked the tiles for each shape. Median of three rounds against `torch.matmul`, which calls cuBLAS; the peak is `24 SMs × 512 FLOPs per clock × SM clock` for BF16 with FP32 accumulation, from the [Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) as Part 4 derived:

| n | cuBLAS kernel | cuBLAS | Triton | Triton's tiles (M × N × K), stages, warps |
|---|---|---|---|---|
| 512 | CUTLASS `s16816gemm` | 10.1 TFLOPS (39%) | 13.1 TFLOPS (53%) | 32 × 64 × 32, 5, 2 |
| 1,024 | `ampere_bf16_s1688gemm` 128 × 64 | 17.8 (75%) | 19.1 (83%) | 64 × 64 × 64, 3, 4 |
| 2,048 | `ampere_bf16_s1688gemm` 128 × 128 | 22.4 (94%) | 21.3 (89%) | 128 × 128 × 32, 2 or 4, 4 |
| 4,096 | same | 20.5 (99%) | 22.0 (94%) | 128 × 128 × 32, 2, 4 |
| 8,192 | same | 21.1 (98.5%) | 22.5 (99.0%) | 128 × 128 × 32, 4, 4 (set by hand) |

Percentages are of the peak at each run's clock; FP16 was within 3% of BF16 throughout, and results matched cuBLAS within BF16 rounding. Triton won outright at 512 and 1,024. At 2,048 and 4,096 cuBLAS did more per clock cycle (Triton finished first at 4,096 only by running 200 MHz higher), and at 8,192 they matched within half a point.

### FP32 inputs: TF32 by default

`tl.dot` on FP32 tensors doesn't do FP32 math by default. Triton 3.6's docstring for `input_precision` lists `"tf32"`, `"tf32x3"` and `"ieee"`, with "Default: "tf32"". **TF32** keeps FP32's 8-bit exponent but only 10 mantissa bits, so tensor cores can multiply it ([Math Part 14](/blog/floating-point-math/) covers the formats). PyTorch leaves `torch.backends.cuda.matmul.allow_tf32` off by default. At n = 4,096 (median of three rounds; error is the largest difference from a float64 result, relative to its largest entry):

| Kernel | Time | TFLOPS | % of peak at its clock | Max relative error |
|---|---|---|---|---|
| `torch.matmul`, TF32 off (CUDA cores) | 27.8 ms | 4.95 | 62% | 2.9 × 10⁻⁶ |
| `torch.matmul`, TF32 on | 13.8 ms | 9.96 | 94% | 2.8 × 10⁻⁴ |
| Triton, default (`"tf32"`) | 12.8 ms | 10.77 | 90% | 7.8 × 10⁻⁴ |
| Triton, `input_precision="ieee"` | 24.6 ms | 5.60 | 58% | 2.9 × 10⁻⁶ |
| Triton, `"tf32x3"` | 38.4 ms | 3.58 | 29% | 5.6 × 10⁻⁷ |

On this GeForce chip the TF32 tensor-core peak equals the FP32 CUDA-core peak, 256 FLOPs per SM per clock (from the whitepaper's 82.6 dense TF32 TFLOPS for the RTX 4090). TF32 still ran twice as fast, because tensor-core kernels get near their peak and FP32 kernels don't (Part 5 found them held to 60% to 70% by instruction issue), at the price of 100 to 270 times more error. With `"ieee"`, Triton reached 58% of peak, behind cuBLAS's 62% and Part 5's 70%. `"tf32x3"` splits each input into a TF32 "big" part and a "small" remainder and adds three TF32 products (big × big, big × small, small × big), beating plain FP32's accuracy here at three times the cost.

### Autotuning

Tile sizes, warps and pipeline depth interact with the shape, data type and GPU, so Triton lets you list candidates and time them all:

```python
def cfg(bm, bn, bk, stages, warps, g=8):
    return triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": g}, num_stages=stages, num_warps=warps)

CONFIGS = [  # the first eight are the Triton tutorial's CUDA configs (fp16/bf16 list)
    cfg(128, 256, 64, 3, 8), cfg(64, 256, 32, 4, 4), cfg(128, 128, 32, 4, 4), cfg(128, 64, 32, 4, 4),
    cfg(64, 128, 32, 4, 4), cfg(128, 32, 32, 4, 4), cfg(64, 32, 32, 5, 2), cfg(32, 64, 32, 5, 2),
    cfg(128, 128, 64, 3, 8), cfg(128, 128, 32, 2, 4), cfg(128, 128, 32, 1, 4), cfg(64, 64, 64, 3, 4),
]
matmul_autotuned = triton.autotune(configs=CONFIGS, key=["M", "N", "K", "PRECISION"])(matmul_kernel)
```

`num_warps` sets threads per program. `num_stages` is, in the [docs](https://triton-lang.org/main/python-api/generated/triton.Config.html)' words, "the number of stages that the compiler should use when software-pipelining loops": how many K slices are in flight at once, Part 5's double buffering generalized. The first time the kernel sees new values of the `key` arguments, the autotuner compiles every config, times each with `do_bench` (25 ms warm-up, 100 ms of runs, median) and keeps the fastest.

All twelve, timed one at a time at n = 4,096 in BF16 (median of three rounds):

| Tiles (M × N × K) | Stages | Warps | Shared memory | Registers per thread | Time | TFLOPS | % of peak at its clock |
|---|---|---|---|---|---|---|---|
| 128 × 128 × 32 | 4 | 4 | 48 KB | 231 | 6.09 ms | 22.6 | 98.0% |
| 128 × 256 × 64 | 3 | 8 | 96 KB | 255 | 6.11 ms | 22.5 | 91.8% |
| 128 × 128 × 64 | 3 | 8 | 64 KB | 168 | 6.12 ms | 22.5 | 90.2% |
| 128 × 128 × 32 | 2 | 4 | 16 KB | 225 | 6.49 ms | 21.2 | 91.9% |
| 64 × 128 × 32 | 4 | 4 | 36 KB | 150 | 6.50 ms | 21.2 | 97.9% |
| 128 × 64 × 32 | 4 | 4 | 36 KB | 157 | 6.76 ms | 20.3 | 95.1% |
| 64 × 256 × 32 | 4 | 4 | 60 KB | 229 | 6.88 ms | 20.0 | 91.8% |
| 128 × 128 × 32 | 1 | 4 | 16 KB | 220 | 6.99 ms | 19.7 | 82.7% |
| 64 × 64 × 64 | 3 | 4 | 32 KB | 126 | 7.43 ms | 18.5 | 94.7% |
| 128 × 32 × 32 | 4 | 4 | 30 KB | 95 | 7.76 ms | 17.7 | 97.3% |
| 32 × 64 × 32 | 5 | 2 | 24 KB | 96 | 7.80 ms | 17.6 | 95.5% |
| 64 × 32 × 32 | 5 | 2 | 24 KB | 96 | 8.20 ms | 16.8 | 89.1% |

The slowest took 1.35 times as long as the fastest. Small tiles reached a high share of peak but ran at about 1,500 MHz against 1,850 to 2,050 for big ones; they reread the inputs more often, and the grouping test showed what memory traffic does to the clock. Pipelining mattered: the same 128 × 128 × 32 tile reached 82.7% of peak with 1 stage, 91.9% with 2 and 98.0% with 4.

Tuning isn't free. With an empty cache, the first call at n = 4,096 took 6.1 seconds to compile and time twelve variants; with them compiled, a new process still spent 2.1 seconds re-timing, since results are saved only with `cache_results=True`. And the pick is only as steady as the clocks: those two runs chose different winners, and all three size sweeps chose 2 stages at n = 4,096, 7% slower than 4 stages timed for a full second.

## Step 4: how it compiles, and what it hides

The first launch runs a compiler. Triton 3.6's NVIDIA backend has five stages, `ttir`, `ttgir`, `llir`, `ptx` and `cubin`, and keeps each one's output:

![A left-to-right flowchart. A Python function with @triton.jit is traced, with its argument types and constants, into Triton IR, tensors of pointers. With num_warps and num_stages it is lowered to TritonGPU IR, where layouts, coalescing, tensor cores and pipelining are decided. That becomes LLVM IR, per-thread code, which LLVM's NVPTX backend turns into PTX, virtual GPU assembly. These four stages sit inside a box labeled Triton compiler, first launch. ptxas turns PTX into a cubin with SASS machine code, which is stored in the on-disk cache at ~/.triton/cache and launched on the GPU.](/diagrams/g07-triton-compile.svg "What Triton does on the first launch of a kernel: four intermediate forms, then NVIDIA's assembler.")

**Triton IR** (TTIR) is the function as tensor operations, with no threads. **TritonGPU IR** (TTGIR) gives every tensor a **layout**, which thread holds which elements, and its passes coalesce accesses, map `tt.dot` to tensor cores and pipeline loops. Then come LLVM IR, **PTX** (Parallel Thread Execution, NVIDIA's virtual instruction set, which `nvcc` also emits) and, from NVIDIA's assembler `ptxas`, **SASS**, the real machine code. Every stage is a string on the object a launch returns:

```python
import re                                   # continuing the vector add listing, where k is the launch's result
print(list(k.asm.keys()))
print(k.asm["ttgir"].splitlines()[0])
print(sorted(set(re.findall(r"(?:ld|st)\.global\.\S+", k.asm["ptx"]))))
```

```text
['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
['ld.global.v4.b32', 'st.global.v4.b32']
```

That layout line is the vector add's whole thread plan: 4 consecutive elements per thread, 32 threads per warp, 4 warps, so 512 elements per pass and two passes for 1,024. Four consecutive floats per thread is what allowed `ld.global.v4.b32`, the 16-byte load. The same files sit in the cache directory (`~/.triton/cache`, or `TRITON_CACHE_DIR`), and `MLIR_ENABLE_DUMP=1` prints the IR after every compiler pass.

### Inside the matmul

For the BF16 matmul (128 × 128 × 32 tiles, 4 warps, 4 stages), 210 lines of TTIR became 1,341 lines of PTX and 848 SASS instructions. The TTGIR shows the compiler's choices:

```text
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
%a = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xbf16, #shared, #smem, mutable>
```

The four warps split the tile two by two, 64 × 64 each, with Ampere-style tensor-core instructions. The A and B tiles sit in shared memory in a **swizzled** layout, addresses permuted to avoid bank conflicts ([Part 3](/blog/gpu-memory-hierarchy/); [Part 6](/blog/tensor-cores/) explains swizzling), three copies of each: `num_stages = 4` became three slices loading while a fourth is computed, 48 KB in all. The loop body:

| Operation | PTX | SASS | Count | What it does |
|---|---|---|---|---|
| Tensor-core multiply | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `HMMA.16816.F32.BF16` | 64 | 16 × 8 × 16 multiply-accumulate per warp |
| Async copy to shared memory | `cp.async.cg.shared.global` | `LDGSTS.E.BYPASS.128` | 32 | 16 bytes from DRAM to shared memory, skipping registers |
| Shared memory to registers | `ldmatrix.sync.aligned.m8n8.x4` (plain and `.trans`) | `LDSM.16.M88.4`, `LDSM.16.MT88.4` | 16 | Loads tensor-core operands in their required layout |
| Barrier | `bar.sync` | `BAR.SYNC` | 8 | Waits for all four warps |

The count checks out. A 128 × 128 × 32 slice is `128 × 128 × 32 × 2 = 1,048,576` FLOPs and each `m16n8k16` does `16 × 8 × 16 × 2 = 4,096`, so a slice needs 256: 64 per warp. The `cp.async` copies are the pipelining, issued for slices ahead and waited on later (`cp.async.wait_group`): Part 5's double buffering without the trip through registers. The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) lists `cp.async` as requiring sm_80 (Ampere) or newer.

### What Triton hides from you

The automation is also the limit. The 2019 paper said "CUDA-like concurrency primitives (shared memory synchronization, inter-thread communication, etc.) are inexistent," and in ordinary Triton they still are:

- No thread or warp index: no warp shuffles, and no giving warps different jobs (warp specialization) unless the compiler does it.
- No explicit shared memory: the compiler picks what goes there, its layout and the barriers.
- Power-of-two tiles held in registers, which the softmax outgrew at 131,072 columns.
- Compile time: seconds per variant, times every autotuning config and new shape.

The Triton project's answer is **Gluon**, a lower-level language on the same compiler, shipped in Triton 3.6 as `triton.experimental.gluon`. Its tutorial says Triton "defers to the compiler to manage tile layouts, memory allocation, data movement, and asynchronity," that "it can be beaten by hand-tuned low-level code," and that in Gluon "these details are exposed to the user." Below that is CUDA C++ with NVIDIA's CUTLASS and CuTe. Hopper's TMA (tensor memory accelerator) and wgmma instructions, which this laptop lacks, appear in Triton's compiler passes and Gluon's tutorials; they weren't measured here.

## In the real world

PyTorch's [compiler documentation](https://docs.pytorch.org/docs/stable/torch.compiler.html) says TorchInductor, the backend behind `torch.compile`, "leverages OpenAI Triton as the key building block" on NVIDIA, AMD and Intel GPUs: it fuses chains of operations and emits Triton source for them. [Inference Part 7](/blog/flashattention/) introduced that pipeline; [Part 9](/blog/kernel-fusion-cuda-graphs/) reads the Triton that Inductor generates for LLM layers.

Triton's "Fused Attention" tutorial is "a Triton implementation of the Flash Attention v2 algorithm," and the FlashAttention repository ships a Triton backend that "supports AMD's CDNA (MI200, MI300) and RDNA GPUs." Part 10 writes a FlashAttention-style kernel in Triton on this laptop, with the online softmax Step 2 lacked.

LinkedIn's [Liger-Kernel](https://arxiv.org/abs/2410.10989), Triton kernels for LLM training such as RMSNorm, RoPE, SwiGLU and fused cross-entropy, reports "on average a 20% increase in training throughput and a 60% reduction in GPU memory usage" against Hugging Face implementations. [Unsloth](https://github.com/unslothai/unsloth)'s fine-tuning library keeps Triton kernels for RMSNorm, LayerNorm, RoPE, SwiGLU and cross-entropy in its `unsloth/kernels` folder. Both target the memory-bound operations around the matmuls, where Step 2 showed fusion pays.

Triton's sweet spot is what this article measured: memory-bound fused kernels and standard tiled matmuls. Kernels that need warp-level control, unusual shared-memory layouts or the newest data center instructions are still written in CUDA C++, often with CUTLASS, or in Gluon. Triton also has an AMD backend, and its matmul tutorial claims performance "on par with cuBLAS or rocBLAS," AMD's library; Part 14 returns to AMD GPUs.

## Common misconceptions

- **"Python kernels must be slower than CUDA."** The Python is only a description: the vector add compiled to 16-byte loads and matched `torch.add`, and the matmul compiled to `HMMA` tensor-core instructions.
- **"Fused is always faster."** The fused softmax tied PyTorch at most widths and lost four times over at 131,072 columns, where a row no longer fit in registers. Fusion saves bytes; it can't fix a tile that spills.
- **"`tl.dot` on FP32 inputs is FP32 math."** Triton 3.6 defaults to TF32, with 10 mantissa bits. Ask for `input_precision="ieee"` for FP32 accuracy; PyTorch's default is the opposite.
- **"The autotuner finds the best config."** It finds the fastest of your configs in a short timing window; its usual pick at n = 4,096 was 7% slower than the best.
- **"L2 reuse only matters for memory-bound kernels."** Grouped ordering didn't change the matmul's time at locked clocks, but it cut DRAM traffic 6.7 times and, on a 40 W laptop, let the SM clock run 27% higher.

## Key takeaways

- A Triton kernel describes one program's work as operations on tensors of indices and pointers, with masks for the edges. The compiler decides what each thread does: vector width, layouts and synchronization.
- For memory-bound work, a few lines reach the roof. The vector add hit 231 GB/s, and the fused softmax ran at 86% to 89% of the bandwidth roof from 512 to 65,536 columns, up to 1.8 times faster than `torch.softmax` on Part 4's slow shape.
- `tl.dot` compiles to `mma.sync` tensor-core instructions, `num_stages` becomes `cp.async` pipelining, and grouped program order cut DRAM reads from 9.0 GB to 1.35 GB.
- Autotuning picks tiles, warps and stages per shape, at a cost of seconds per new shape, and on a laptop its pick is only as steady as the clocks.
- Triton hides threads, warps and shared memory. When a kernel needs them, drop to Gluon, CUDA or CUTLASS.

## What's next

The softmax here gave up once a row outgrew one program's registers, and its reductions were a `tl.max` and a `tl.sum` the compiler expanded for us. [Part 8](/blog/gpu-reductions/), Reductions and normalization kernels, opens those up: reduction trees, warp shuffles and block-level reductions, softmax, RMSNorm and LayerNorm kernels for any row length, numerical stability on the GPU, and the sampling kernels that pick each next token.

## Further reading

- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (Tillet, Kung and Cox, MAPL 2019)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf): the original paper, its programming model and its compiler passes for coalescing and shared memory.
- [Triton tutorials (Triton docs)](https://triton-lang.org/main/getting-started/tutorials/index.html): vector add, fused softmax, matrix multiplication with grouped ordering, and fused attention, with benchmarks on data center GPUs.
- [triton.Config](https://triton-lang.org/main/python-api/generated/triton.Config.html) and [triton.autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html) (Triton docs): `num_warps`, `num_stages`, autotuning keys and `cache_results`.
- [triton.language.dot (Triton docs)](https://triton-lang.org/main/python-api/generated/triton.language.dot.html): `input_precision` and the TF32 default.
- [Introduction to Gluon (Triton tutorials source)](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/01-intro.py): what Triton leaves to the compiler, and the lower-level language that exposes it.
- [PTX ISA (NVIDIA)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): `mma.sync`, `ldmatrix` and `cp.async`, the instructions Triton emitted here.
- [Liger Kernel: Efficient Triton Kernels for LLM Training (Hsu et al., 2024)](https://arxiv.org/abs/2410.10989): fused Triton kernels for LLM training and their measured savings.
- [torch.compiler documentation (PyTorch)](https://docs.pytorch.org/docs/stable/torch.compiler.html): TorchDynamo, TorchInductor and its Triton code generation.
- [NVIDIA Ada GPU Architecture whitepaper (NVIDIA)](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf): the tensor-core rates per format behind this article's peaks.
