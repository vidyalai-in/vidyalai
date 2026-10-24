---
title: "Reductions and Normalization Kernels: Summing Across Threads"
description: "Sum 67 million floats on a laptop GPU step by step, then write softmax, RMSNorm, LayerNorm and argmax kernels that stay fast and numerically stable."
slug: "gpu-reductions"
category: "gpu-kernels"
tags: ["gpu", "cuda", "latency", "serving"]
series: "llm-gpu"
seriesOrder: 8
pubDate: 2026-10-24
---

## TL;DR

A reduction turns many numbers into one: a sum, a maximum, the position of the largest value. GPUs do it as a tree: each thread sums a slice in a register, each warp of 32 threads combines its values with shuffle instructions, each block combines its warps through shared memory, and the blocks meet in a final step. On a laptop RTX 4060 it summed 67 million floats at 247.5 GB/s, 99.6% of the measured read ceiling, with a more accurate answer than a plain loop, which returned half the right total. With the same pieces, a softmax without block-wide barriers ran 1.41 times faster than `torch.softmax` on the shape [Part 4](/blog/gpu-profiling-roofline/) flagged, a fused RMSNorm ran 8.7 times faster than Hugging Face's eight-kernel eager version, and Welford's algorithm kept LayerNorm from returning NaN where the textbook variance formula failed. Picking the next token is a reduction small enough that launch overhead dominates it.

## Why it matters

Add up 67,108,864 random numbers between 0 and 1, one after another, in 32-bit floating point (FP32). The answer should be about 33.55 million. A plain C++ loop returns 16,777,216, exactly 2^24, and stops there. The GPU, adding the same numbers in a different order, returns 33,552,476, the closest FP32 value to the exact sum, in 1.08 milliseconds.

LLMs reduce constantly: every RMSNorm averages a row's squares, every softmax finds a row's maximum and sums exponentials, and every token starts with a search for the largest of 151,936 scores. [Part 1](/blog/gpu-from-python-to-silicon/) counted 58 reduction kernels in one decode step of Qwen2.5-1.5B, and RMSNorm alone launched 456 of the step's 1,143 kernels. [Part 4](/blog/gpu-profiling-roofline/) found PyTorch's softmax at 63% of the bandwidth roof and promised kernels that avoid the problem. [Part 7](/blog/triton-kernels/)'s Triton softmax fixed that shape, then fell to a quarter of PyTorch's speed on rows of 131,072 values. This part writes those kernels in CUDA.

## The intuition

Think of counting votes on election night. One clerk adding 67 million ballots to a single tally is slow, and once the total is large, a few more votes stop changing what fits on the sheet. So the hall splits the work. Each table counts its own stack, then neighboring tables combine totals in pairs, then pairs of pairs, until one number reaches the front. No running total gets huge next to what is added to it.

The GPU has four levels of tables: a thread, a warp, a block and the whole grid. The analogy breaks in two places. Tellers within a warp read each other's tally sheets directly, without walking to a shared desk. And with floating-point numbers, the order in which totals arrive changes the last digits of the result, which never happens with whole votes.

## Step 1: summing 67 million floats, step by step

A **reduction** combines a list into one value with an associative operation (one where grouping doesn't matter on paper, like addition or maximum). A sum does one addition per 4-byte float read, so the yardstick is GB/s against the bandwidth roof.

### How it was timed

Everything ran on [Part 1](/blog/gpu-from-python-to-silicon/)'s machine: an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, 24 SMs, compute capability 8.9) under WSL2, driver 592.82, CUDA 12.8 with CUB 2.7.0, PyTorch 2.11.0 and Triton 3.6.0. CUDA kernels were timed with CUDA events after about 0.4 seconds of warm-up, as the median of back-to-back runs filling 1.2 seconds; Python-side kernels used Part 7's `triton.testing.do_bench` setup (L2 flushed before each call, median). Tables give the median of three rounds unless noted. `nvidia-smi` logged clocks every 200 ms: the power limit sat between 40 and 50 W, the SM clock between 1,515 and 2,175 MHz, and the memory clock at 8,000 MHz for every timed result unless noted, so the roof is the rated 256 GB/s. [Part 4](/blog/gpu-profiling-roofline/)'s read-only `float4` kernel reached 248.5 GB/s in the same runs. Nsight Compute 2025.1.1 had hardware counters this time.

### Reductions 1 and 2: a tree in shared memory

The classic starting point is Mark Harris's NVIDIA talk [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf). Each block of 256 threads copies 256 values into **shared memory**, the on-chip scratchpad a block's threads share ([Part 3](/blog/gpu-memory-hierarchy/)), and halves them in eight rounds. `__syncthreads()` is a **barrier**: no thread passes it until all threads in the block reach it.

```cuda
__global__ void r1_interleaved(const float* in, float* out, int n) {
  __shared__ float s[T];                       // T = 256 threads per block
  unsigned t = threadIdx.x, i = blockIdx.x * T + t;
  s[t] = i < n ? in[i] : 0.0f;
  __syncthreads();
  for (unsigned st = 1; st < T; st *= 2) {
    if (t % (2 * st) == 0) s[t] += s[t + st];  // only every (2*st)-th thread works
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = s[0];          // one partial sum per block
}
```

The host relaunches it on the 262,144 partial sums until one value remains. Harris's slides flag the `if`: "highly divergent warps are very inefficient, and % operator is very slow." **Divergence** means threads of one warp take different branches, so the warp runs both paths ([Part 2](/blog/cuda-programming-model/)). His fix is sequential addressing, `if (t < st) s[t] += s[t + st]` with `st` counting down from 128, so the working threads stay contiguous and whole warps finish early.

### Reductions 3 to 5: fewer barriers, more work per thread

Three more steps follow Harris, with one modern substitution:

- Reduction 3 adds two values while loading (`in[i] + in[i + 256]`), so half as many blocks run the same tree.
- Reduction 4 replaces the shared-memory tree with **warp shuffles**, which the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html) says "exchange a value between non-exited threads within a warp without the use of shared memory." Harris instead relied on a warp's 32 threads running in lockstep, which NVIDIA's Lin and Grover [called unsafe in 2018](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/): "implicit lock-step execution is not guaranteed." The `_sync` shuffles are the supported replacement.
- Reduction 5 launches only the 144 blocks that fit on the GPU at once (six per SM) and uses a **grid-stride loop**: each thread walks the array in steps of the whole grid, adding about 455 16-byte `float4` loads into one register. The 144 block totals are combined with one `atomicAdd` per block or with a second, one-block pass.

Here is the warp step. `__shfl_down_sync(mask, v, off)` returns the `v` held by the thread `off` lanes higher (a **lane** is a thread's position, 0 to 31, in its warp):

```cuda
__device__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffff, v, off);   // lane i adds lane i+off
  return v;                                      // lane 0 holds the warp's total
}
```

Five steps cover 32 lanes. Follow lane 0:

| Step | Offset | Lane 0 adds the value from | Lane 0 now holds the sum of |
|---|---|---|---|
| 1 | 16 | lane 16 | lanes 0 and 16 |
| 2 | 8 | lane 8 (which already added 24) | lanes 0, 8, 16, 24 |
| 3 | 4 | lane 4 | 8 lanes, every 4th |
| 4 | 2 | lane 2 | 16 lanes, every 2nd |
| 5 | 1 | lane 1 | all 32 lanes |

`__shfl_xor_sync` instead pairs lane i with lane i XOR offset, which the guide calls "a butterfly addressing pattern"; every lane ends with the total, as the softmax kernels below need. A block of 8 warps then needs one barrier: lane 0 of each warp writes its total to shared memory, and warp 0 shuffles those 8 values.

![A top-to-bottom flowchart of the grid-stride reduction. Global memory holds 2^26 floats, 268 MB. A grid-stride loop feeds each of 36,864 threads, which keeps a running sum of about 455 float4 loads in a register. Each warp of 32 threads combines its values in 5 shuffle steps with offsets 16, 8, 4, 2 and 1. Lane 0 of each warp writes its total into shared memory, so each block of 8 warps holds 8 warp sums. After one barrier, warp 0 shuffles again, leaving 144 block sums for the whole grid. These are combined either by atomicAdd into one float, which is fast but adds in a varying order, or by a second pass in one block, which uses a fixed order and gives the same bits every run.](/diagrams/g08-reduction-levels.svg "The reduction hierarchy: registers, then shuffles, then shared memory, then one global step.")

Each level hands the next 32 or more times fewer values, so the shared levels see almost no traffic.

### The climb, measured

Every version sums the same 2^26 floats (268.4 MB) and includes all its passes. This is the first round, with memory at 8,000 MHz throughout; the other rounds were within 1%, except reduction 3 at 190 to 200 GB/s:

| Version | Time | GB/s | % of 256 GB/s |
|---|---|---|---|
| R1 interleaved addressing (divergent) | 2.472 ms | 108.6 | 42.4% |
| R2 sequential addressing | 2.431 ms | 110.4 | 43.1% |
| R3 first add during load | 1.331 ms | 201.6 | 78.8% |
| R4 warp shuffles | 1.094 ms | 245.5 | 95.9% |
| R5 grid-stride, one `atomicAdd` per block | 1.088 ms | 246.6 | 96.3% |
| R5 grid-stride, second pass | 1.084 ms | 247.5 | 96.7% |
| `cub::DeviceReduce::Sum` | 1.084 ms | 247.5 | 96.7% |
| `torch.sum` (separate run, same timing) | 1.096 ms | 245.0 | 95.7% |
| Read-only ceiling (`float4` reads, no math) | 1.080 ms | 248.5 | 97.1% |

On Harris's 2007 G80 GPU these steps were worth a factor of 30: his first kernel moved 2.083 GB/s against an 86.4 GB/s peak, his last 73 GB/s on 32 million elements. Here the climb is worth 2.3 times, and sequential addressing, which doubled his speed, gained 2%.

Nsight Compute explains why. R1 and R2 kept the L1/shared-memory pipeline 91% to 92% busy while DRAM throughput sat at 27%, so shared memory and barriers limit them, not the memory bus. The compiler had also turned the `if` into **predication**: every thread issues the instruction and a per-thread flag switches off those that shouldn't act. Divergence becomes cheap, but R2's idle warps keep issuing too; both kernels averaged about 17 of 32 useful threads per instruction. When the memory clock dropped to 7,000 MHz during R1 and R2 in later rounds, their speed didn't move. R4 cut the shared-memory pipeline to 66% busy and pushed DRAM to 97%. From there the bus is the limit, and R5 adds 1%.

### The library answer

NVIDIA's CUB library, shipped with the CUDA toolkit, provides this as [`cub::DeviceReduce`](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html), "device-wide, parallel operations for computing a reduction across a sequence of data items." The first call, with a null pointer, only reports how much scratch memory it needs (3,327 bytes):

```cuda
size_t tmp_bytes = 0; void* tmp = nullptr;
cub::DeviceReduce::Sum(tmp, tmp_bytes, d, b0, n);   // first call: ask how much scratch it needs
cudaMalloc(&tmp, tmp_bytes);
cub::DeviceReduce::Sum(tmp, tmp_bytes, d, b0, n);   // second call: two kernels, partials then total
```

It tied the hand-written two-pass version to the microsecond, and `torch.sum`'s single `reduce_kernel` was within 1%.

## Step 2: what the tree does to the answer

### Order changes the result

Floating-point addition rounds after every step, so `(a + b) + c` and `a + (b + c)` can differ ([Math Part 14](/blog/floating-point-math/)). The same 2^26 uniform random floats, summed four ways and checked against a 64-bit (FP64) sum on the CPU:

```text
FP64 serial  33552475.8093
FP32 serial  16777216.0000  rel err -5.00e-01
FP32 R4 tree  33552476.0000  rel err 5.68e-09
FP32 R5 two-pass 33552476.0000  rel err 5.68e-09
FP32 CUB      33552476.0000  rel err 5.68e-09
```

The loop stalls at 2^24 for the reason Math Part 14 showed for BF16 at 256: from 16,777,216 up, neighboring FP32 values are 2 apart, so adding anything below 1 rounds back to the same total. The tree never puts a big partial sum next to a small input: in R5 each thread's total stays under about 1,000, and each level then adds numbers of similar size. That's **pairwise summation**, the order NumPy uses, whose error grows far more slowly than a running total's. Its worst case over 26 levels is about 26 times FP32's unit roundoff of 6 × 10^-8. The measured error, 0.19, is smaller than the gap of 2 between neighboring FP32 values here, so the GPU returned the correctly rounded answer.

### Atomics make it nondeterministic

An **atomic** operation is a read-modify-write the hardware performs indivisibly, so blocks adding to one address can't lose an update. But blocks finish in a different order each run, and so do the 144 additions. Twenty runs of each:

```text
atomicAdd  values over 20 runs: 33552466 (x2) 33552468 (x3) 33552470 (x1) 33552472 (x4) 33552474 (x1) 33552476 (x5) 33552478 (x2) 33552482 (x2)
two-pass   values over 20 runs: 33552476 (x20)
CUB        values over 20 runs: 33552476 (x20)
```

Eight different answers. Justin Luitjens's 2014 NVIDIA post on [shuffle reductions](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) warned that with atomics "floating point reductions will not be bit-wise exact from run to run." The spread is tiny, but it makes same-seed runs drift apart and exact-output tests fail at random. Here the deterministic second pass was even 4 µs faster.

Since CCCL 3.1 (newer than the CUB 2.7 measured here), CUB lets you choose. Its [determinism docs](https://nvidia.github.io/cccl/unstable/cub/determinism.html) make `run_to_run` the default; per NVIDIA's [CCCL post](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl), `not_guaranteed` uses atomics for a single pass, and `gpu_to_gpu` gives identical bits on any GPU at a cost of "20% to 30% for large problem sizes." PyTorch has `torch.use_deterministic_algorithms`, and its [reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html) flag FlashAttention's backward pass for using "non-deterministic atomic operations by default."

## Step 3: softmax, one row at a time

LLM-shaped reductions are thousands of rows of a few thousand values, each with its own maximum and sum, so the question is how many threads to give each row.

The test shape is Part 4's: 8,192 rows of 4,096 BF16 values (16-bit brain floating point), 67.1 MB in and 67.1 MB out, with FP32 arithmetic. The layouts:

- Warp per row, in registers: each lane loads 128 values as 16 chunks of 8, takes an XOR-shuffle maximum and sum, and writes. No shared memory, no `__syncthreads()`.
- Block per row, three passes: 256 threads stream the row from memory three times, for the max, the sum of exponentials, and the output.
- Block per row, online: the first pass keeps a running max `m` and sum `l` together, rescaling `l` when `m` grows. That's the **online softmax** of Milakov and Gimelshein, derived in [Math Part 5](/blog/softmax-math/). It reads the row twice instead of three times.
- `torch.softmax`, and Part 7's 12-line Triton kernel.

The online kernel's core merges two (max, sum) states, per chunk and again across lanes:

```cuda
__device__ void ms_merge(float& m, float& l, float m2, float l2) {
  float nm = fmaxf(m, m2);
  l = l * __expf(m - nm) + l2 * __expf(m2 - nm);   // rescale both sums to the new max
  m = nm;
}
```

Median of three rounds:

| Kernel, 8,192 × 4,096 BF16 | Time | GB/s | % of 256 GB/s |
|---|---|---|---|
| `torch.softmax` | 842.8 µs | 159.3 | 62.2% |
| Triton ([Part 7](/blog/triton-kernels/)) | 592.9 µs | 226.4 | 88.4% |
| CUDA, warp per row, registers | 606.2 µs | 221.4 | 86.5% |
| CUDA, block per row, three passes | 598.0 µs | 224.4 | 87.7% |
| CUDA, block per row, online | 597.0 µs | 224.8 | 87.8% |

Every hand-written layout beat PyTorch by 1.39 to 1.41 times and landed within 2.2% of Triton. All matched an FP32 reference to within 0.002.

### Why PyTorch's kernel is slow here

Nsight Compute (clocks locked at 1.32 GHz SM, 7,000 MHz memory) shows two ways to be fast:

| Kernel (Nsight Compute, locked clocks) | Threads per block | Registers per thread | Theoretical occupancy | DRAM throughput | Barrier stall per instruction |
|---|---|---|---|---|---|
| `torch.softmax` (`cunn_SoftMaxForwardReg`) | 1,024 | 30 | 66.7% | 44% | 2.6 cycles |
| CUDA, block per row, online | 256 | 26 | 100% | 91% | 3.1 cycles |
| CUDA, warp per row, registers | 128 | 255 | 16.7% | 89% | 0 |

**Occupancy** is the share of an SM's 48 warp slots that resident warps fill ([Part 2](/blog/cuda-programming-model/)). PyTorch gives each row a 1,024-thread block and only one fits per SM, so each SM alternates between loading a row and waiting at barriers for that row's maximum and sum. With six 256-thread blocks per SM, one block's reduction overlaps other blocks' loads, and DRAM stays busy.

The warp-per-row kernel uses all 255 registers a thread may have, so only 8 warps fit per SM, yet it nearly matches. Each lane issues 16 independent 16-byte loads before it needs any: 64 KB in flight per SM, enough to keep the bus busy, and with no barriers no warp waits for another. Occupancy is one way to hide memory latency; independent loads within a thread are another.

### Rows that don't fit, and the read the online version saves

The block-per-row kernels stream each row, so they handle any length. Same 2^25 elements as 256 rows of 131,072:

| Kernel, 256 × 131,072 BF16 | Time | GB/s | % of roof | DRAM bytes read (Nsight Compute) |
|---|---|---|---|---|
| `torch.softmax` | 608.3 µs | 220.7 | 86.2% | 67.2 MB |
| Triton (Part 7) | 2,439 µs | 55.0 | 21.5% | not profiled |
| CUDA three passes, 256 threads | 1,032 µs | 130.0 | 50.8% | 175.7 MB |
| CUDA online, 256 threads | 864.3 µs | 155.3 | 60.7% | 133.7 MB |
| CUDA three passes, 1,024 threads | 599.0 µs | 224.1 | 87.5% | 67.1 MB |
| CUDA online, 1,024 threads | 596.0 µs | 225.2 | 88.0% | 67.1 MB |

The input is 67.1 MB, so the last column counts trips to DRAM. At 4,096 columns every version read the input exactly once: an 8 KB row read again is still in the L1 or L2 cache, so extra passes were free. At 131,072 columns a row is 256 KB, and six 256-thread blocks per SM keep 144 rows (36 MB) in flight, more than the 32 MB L2. Re-reads spill to DRAM: the three-pass kernel read 2.6 times the input, the online one 2.0 times, and the saved read was worth 1.19 times the speed.

With 1,024-thread blocks only 24 rows (6 MB) are in flight, every re-read hits L2, and both versions reach 88%, as PyTorch does with the same block size. The online algorithm pays off when data outgrows every cache, the normal case in attention, which Part 10 builds.

## Step 4: RMSNorm and LayerNorm

### RMSNorm, fused

RMSNorm divides each row by the square root of its mean square, then multiplies by a learned weight ([Math Part 10](/blog/initialization-normalization/)). Hugging Face's Qwen2 code, run eagerly, takes eight kernels, each reading from DRAM and writing back: cast to FP32, square, mean, add ε, reciprocal square root, multiply, cast back to BF16, multiply by the weight.

The fused kernel is the warp-per-row softmax with different arithmetic: sum squares in FP32, one `rsqrtf`, scale. It rounds where Hugging Face's code does and matched its output bit for bit on 99.999% of elements, one BF16 step off elsewhere. PyTorch 2.11's [`torch.nn.functional.rms_norm`](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.rms_norm.html) is fused too; the profiler showed one kernel, `vectorized_layer_norm_kernel` with its RMS flag set.

| RMSNorm, 8,192 × 4,096 BF16 | Kernels | Time | GB/s (134 MB moved) | % of roof |
|---|---|---|---|---|
| Hugging Face eager composite | 8 | 5,214 µs | 25.7 | 10.1% |
| `F.rms_norm` | 1 | 601.1 µs | 223.3 | 87.2% |
| CUDA, warp per row | 1 | 603.1 µs | 222.5 | 86.9% |

Per byte, the eager version is fine: its eight kernels move about 1,150 MB, mostly FP32 intermediates, which at 5.2 ms is 221 GB/s. It moves 8.6 times more data than it needs, and that accounts for the 8.7 times. [Part 9](/blog/kernel-fusion-cuda-graphs/) has torch.compile generate such fused kernels.

### LayerNorm and catastrophic cancellation

LayerNorm subtracts the row's mean before dividing by the standard deviation, so it needs a variance. The one-pass shortcut uses variance = mean of squares minus square of the mean, `E[x²] − E[x]²`: accumulate `Σx` and `Σx²` together and subtract at the end. When the mean is large next to the spread, that subtracts two nearly equal big numbers, leaving mostly their rounding errors. This is **catastrophic cancellation**.

**Welford's algorithm** ([Welford, 1962](https://doi.org/10.1080/00401706.1962.10490022)) keeps a running mean and a running sum of squared deviations from it, updating both per value. Partial results from two threads merge with the pairwise formula of [Chan, Golub and LeVeque](https://doi.org/10.1080/00031305.1983.10483115), so it runs as a tree:

```cuda
for (int v = 0; v < V; v++) {                     // per thread: Welford over its values
  cnt += 1.0f; float d = f[v] - mu; mu += d / cnt; m2 += d * (f[v] - mu);
}
for (int off = 16; off > 0; off >>= 1) {          // across lanes: Chan et al.'s merge
  float mu2 = __shfl_xor_sync(0xffffffff, mu, off), m22 = __shfl_xor_sync(0xffffffff, m2, off);
  float d = mu2 - mu;                             // equal counts in every lane
  mu += 0.5f * d; m2 += m22 + d * d * 0.5f * cnt; cnt *= 2.0f;
}
```

To test it, feed 1,024 rows of 4,096 FP32 values, each an offset plus normal noise with standard deviation 1, and compare each kernel's variance with an FP64 one. Median relative error:

| Offset (mean of the row) | One pass, `E[x²] − E[x]²` | Welford | Two passes over registers |
|---|---|---|---|
| 0 | 4.0 × 10^-8 | 3.9 × 10^-8 | 3.4 × 10^-8 |
| 100 | 7.6 × 10^-4 | 3.6 × 10^-7 | 3.3 × 10^-8 |
| 1,000 | 5.8 × 10^-2 | 3.2 × 10^-6 | 5.2 × 10^-8 |
| 10,000 | 8.8 (357 rows negative, output NaN) | 4.9 × 10^-5 | 3.8 × 10^-7 |

At an offset of 10,000, `E[x²]` is about 10^8, where neighboring FP32 values are 8 apart, while the true variance is 1. The shortcut returned negative variances for 357 rows, whose square roots are NaN (not a number), and those rows of output became NaN. Welford stayed accurate. When the row already sits in registers, a third option is best: compute the mean, then the mean of `(x − mean)²` from the same registers, at no extra memory cost. PyTorch's [`layer_norm_kernel.cu`](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu) uses Welford, in `cuWelfordOnlineSum` and `cuWelfordCombine`.

Stability cost nothing. On BF16 data at 8,192 × 4,096, `F.layer_norm` took 599.0 µs, the one-pass kernel 593.9, Welford 598.0 and two passes 600.1: all 87% to 88% of the roof, because the kernel waits on memory, not arithmetic. Real rows rarely have a mean 10,000 times their spread, but some activations reach large magnitudes ([Inference Part 8](/blog/quantization/) covers these outliers).

## Step 5: picking the next token

**Greedy decoding** picks the token with the largest **logit**, the model's raw score for each vocabulary entry. That's an argmax over Qwen2.5's 151,936 entries, a reduction over (value, index) pairs, with ties going to the lower index as in `torch.argmax`:

```cuda
__device__ void am_merge(float& v, int& i, float v2, int i2) {
  if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
}
```

At batch 1 the logits are 608 KB of FP32. One 1,024-thread block took 14.5 µs, leaving 23 SMs idle; 24 blocks plus a one-warp merge kernel took 4.2 µs, and `torch.argmax` 7.6 µs.

None of that is what a decode loop pays. In the table, kernel times come from `torch.profiler` in a separate run; the CPU columns are medians of three rounds, each the median of five loops of 2,000 queued calls, or of 500 single calls each followed by `torch.cuda.synchronize()`:

| Operation on 151,936 FP32 logits | Batch | Kernels | GPU kernel time (profiler) | CPU time per call, queued | One call + synchronize |
|---|---|---|---|---|---|
| `torch.argmax` | 1 | 2 | 7.6 µs | 18.6 µs | 34.7 µs |
| CUDA argmax, 48 blocks per row | 1 | 2 | 5.5 µs | 18.0 µs | 29.2 µs |
| `torch.topk(k=50)` | 1 | 22 | 63.5 µs | 182 µs | 224 µs |
| Top-p: sort, softmax, cumsum, sample | 1 | 42 | 311 µs | 434 µs | 441 µs |
| `torch.argmax` | 64 | 2 | 171 µs | 152.5 µs | 196.6 µs |
| `torch.topk(k=50)` | 64 | 22 | 1,074 µs | 1,015 µs | 1,085 µs |
| Top-p: sort, softmax, cumsum, sample | 64 | 48 | 10,455 µs | 10,164 µs | 10,203 µs |

At batch 1 the argmax kernel is a fraction of the call: Python and the launch take 18 µs, and waiting for the result brings it to about 30 µs, the launch-bound regime Parts 1 and 4 measured. A faster argmax saves 2 µs out of 30.

**Top-k sampling** keeps the k most likely tokens and samples among them. PyTorch's [`TensorTopK.cu`](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/TensorTopK.cu) finds them by **radix select**: count how many values share each possible top 8 bits, keep the bucket holding the k-th largest, repeat on the next 8 bits, and never sort. The profiler showed four passes, one per byte of a float. **Top-p (nucleus) sampling** keeps the smallest set of tokens whose probabilities add up to p; Hugging Face's `TopPLogitsWarper` sorts the whole vocabulary and takes a cumulative sum. At batch 64 that path took 10 to 11 ms of GPU time, close to the 13.8 ms of GPU work in a whole batch-1 decode step of Qwen2.5-1.5B (Part 1). At batch 1, each of its three softmax launches ran as one block on one SM, 53 µs apiece.

FlashInfer's [sorting-free sampling kernels](https://flashinfer.ai/2025/03/10/sampling.html) skip the sort. One fused kernel samples a token while ignoring tokens below a pivot, checks the top-p (or top-k) condition, and if it fails, raises the pivot and tries again; a dual-pivot version gives "logarithmic (in worst case) time complexity." The authors report it cut vLLM's "overall sampling time by more than 50%" on one H100 (not measured here).

## In the real world

vLLM's [`layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu) runs RMSNorm as one block per token: threads sum squares in FP32, `cub::BlockReduce<float, 1024>` combines them, and one thread computes `rsqrtf(variance / hidden_size + epsilon)`. Its `fused_add_rms_norm_kernel` also folds in the residual add.

[Liger Kernel](https://github.com/linkedin/Liger-Kernel), LinkedIn's Triton training kernels, has a Hugging Face-compatible RMSNorm whose forward pass is `tl.sum(X_row * X_row, axis=0) / n_cols` and an `rsqrt`; the project says its kernels together "increase multi-GPU training throughput by 20% and reduce memory usage by 60%." NVIDIA's [Apex](https://github.com/NVIDIA/apex) ships `FusedLayerNorm` and `FusedRMSNorm`, and torch.compile writes fused Triton kernels for norm-plus-elementwise chains ([Part 9](/blog/kernel-fusion-cuda-graphs/)).

Sampling kernels read the whole vocabulary for every token of every sequence: 151,936 entries for Qwen2.5, and "a vocabulary with 128K tokens" for Llama 3, per [its paper](https://arxiv.org/abs/2407.21783).

## Common misconceptions

- **"GPU reductions are less accurate than a simple loop."** A tree is pairwise summation: on 2^26 values it returned the correctly rounded FP32 sum, while the loop stopped at half the answer. Atomics add nondeterminism, not error.
- **"Higher occupancy is always faster."** The warp-per-row softmax ran at 16.7% occupancy and 89% of DRAM throughput, with 16 loads in flight per thread. PyTorch's kernel had four times the occupancy and half the throughput.
- **"Online softmax is always faster than three passes."** It saves a DRAM read only once the row has left the cache. At 4,096 columns the two ran within 0.2%; at 131,072 columns with 256-thread blocks, online read 24% fewer bytes and ran 1.19 times faster.
- **"`E[x²] − E[x]²` is just a faster variance."** In FP32 it gave negative variances for 357 of 1,024 rows with a mean of 10,000, and NaN outputs. Welford ran at the same speed.
- **"Faster sampling kernels speed up batch-1 decode."** The argmax kernel takes 5 to 8 µs; the call around it takes 30. Sorting is the real cost: about 11 ms per step for sort-based top-p at batch 64.

## Key takeaways

- A GPU reduction is a tree: a register per thread, shuffles within a warp, shared memory within a block, and one step across blocks. Each level shrinks the data 32 times or more.
- Shuffles plus a grid-stride loop summed 268 MB at 247.5 GB/s, 99.6% of the measured read ceiling, tying CUB and `torch.sum`.
- The tree is also the accurate order. One `atomicAdd` per block gave eight different sums in twenty runs; a second pass gave the same bits every time.
- Row-wise kernels win by avoiding block-wide barriers and extra memory trips: 1.41 times faster than `torch.softmax` on Part 4's shape, 8.7 times faster than eager RMSNorm. Online softmax saves a read once rows outgrow the cache.
- Sampling is a reduction too. At batch 1 launch overhead dominates it; at batch 64, sorting the vocabulary for top-p cost about 11 ms.

## What's next

The fused RMSNorm here won by moving fewer bytes. [Part 9](/blog/kernel-fusion-cuda-graphs/), Kernel fusion, torch.compile and CUDA graphs, makes that the main subject: why elementwise operations are memory-bound, fusing SwiGLU, residual add and norm, the Triton code TorchInductor writes for you, and CUDA graphs removing the launch overhead that dominated the argmax here.

## Further reading

- [Optimizing Parallel Reduction in CUDA (Mark Harris, NVIDIA)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf): the seven-step reduction this part retraces, with G80 numbers.
- [CUDA Programming Guide: C++ language extensions (NVIDIA)](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html): warp shuffle functions and atomics.
- [Using CUDA Warp-Level Primitives (Yuan Lin and Vinod Grover, NVIDIA, 2018)](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/): why implicit warp-synchronous code is unsafe and how the `_sync` primitives replace it.
- [cub::DeviceReduce (NVIDIA CCCL docs)](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html) and [CUB determinism](https://nvidia.github.io/cccl/unstable/cub/determinism.html): the library reduction and its run-to-run and GPU-to-GPU guarantees.
- [Controlling Floating-Point Determinism in NVIDIA CCCL (Al Awar and Singanaboina, NVIDIA, 2026)](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl): what each determinism level does and costs.
- [Note on a Method for Calculating Corrected Sums of Squares and Products (B. P. Welford, Technometrics, 1962)](https://doi.org/10.1080/00401706.1962.10490022): the one-pass mean and variance update.
- [Algorithms for Computing the Sample Variance: Analysis and Recommendations (Chan, Golub and LeVeque, The American Statistician, 1983)](https://doi.org/10.1080/00031305.1983.10483115): the pairwise merge that parallelizes Welford, with an error analysis of the alternatives.
- [Online normalizer calculation for softmax (Milakov and Gimelshein, 2018)](https://arxiv.org/abs/1805.02867): the online softmax.
- [PyTorch `layer_norm_kernel.cu`](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu) and [`TensorTopK.cu`](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/TensorTopK.cu): Welford LayerNorm and RMSNorm, and radix-select top-k, as PyTorch 2.11 ships them.
- [Sorting-Free GPU Kernels for LLM Sampling (Xing, Ye, Hou, Ceze and Chen, FlashInfer blog, 2025)](https://flashinfer.ai/2025/03/10/sampling.html): rejection-based top-k and top-p without a sort.
- [vLLM `layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu) and [Liger Kernel](https://github.com/linkedin/Liger-Kernel): production RMSNorm kernels in CUDA and Triton.
- [Faster Parallel Reductions on Kepler (Justin Luitjens, NVIDIA, 2014)](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/): the first shuffle-based reductions, and the warning about atomics.
