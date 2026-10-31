---
title: "The CUDA Programming Model: Threads, Warps, Blocks and SMs"
description: "Write, compile and time a first CUDA kernel, see it hit the memory-bandwidth limit, and learn what warps, occupancy and grid size do to its speed."
slug: "cuda-programming-model"
category: "gpu-kernels"
tags: ["gpu", "cuda", "latency"]
series: "llm-gpu"
seriesOrder: 2
pubDate: 2026-04-30
---

## TL;DR

A CUDA kernel is the code for one thread; a launch starts millions of copies, grouped into blocks of up to 1,024 threads that form a grid. The hardware runs threads in warps of 32 that share one instruction stream, and runs each block on one streaming multiprocessor (SM). On a laptop RTX 4060, a four-line vector-add kernel moves 805 MB in 3.55 ms, 227 GB/s, as fast as NVIDIA's own `cudaMemcpy` and 89% of the rated 256 GB/s. By the end you'll know how to write, time and debug a kernel like that, why branching inside a warp can nearly double a kernel's time, why 24 blocks on a 24-SM GPU still leave it a quarter short of full speed, and how to catch the errors CUDA reports late.

## Why it matters

Here is a puzzle. A kernel that adds two arrays of 67 million numbers runs at 238 GB/s when launched as 72 blocks of 256 threads. Launch the same kernel as one block and it crawls at 10 GB/s. Launch it as 24 blocks, one per SM on this GPU, and it reaches 177 GB/s. Launch 24 blocks of 1,024 threads and it hits 235 GB/s, but add a 25th block and it drops to 195.

None of that is visible in the kernel's code. It comes from how CUDA maps threads onto hardware, and the rules are few enough to learn in one sitting.

Every kernel in an LLM follows the same rules. [Part 1](/blog/gpu-from-python-to-silicon/) traced one decode step of a 1.5-billion-parameter model and found 1,143 kernel launches, most of them small elementwise operations and copies like this vector add. This part shows how their block and grid sizes get chosen, on a kernel small enough to read in full.

## The intuition

Picture a parade-ground drill. A sergeant (the warp scheduler) shouts one order at a time to a squad of 32 soldiers (a warp), and every soldier carries it out on their own crate. Squads are grouped into companies (thread blocks), and each company is posted to one barracks (an SM) for its whole job. The army (the grid) can be far bigger than the barracks hold, so companies wait their turn.

If an order says "even numbers, paint your crate; odd numbers, sand yours," the sergeant shouts both orders in turn while half the squad stands idle each time. That's warp divergence. And when a squad is waiting for a delivery from memory, the sergeant turns to another squad that's ready. The more squads a barracks holds, the less often the sergeant has nobody to shout at. That's occupancy.

The analogy breaks in two places. Since compute capability 7.0 (the Volta generation), NVIDIA GPUs track a program position for every thread, so lockstep is the programming model's description, and the hardware may depart from it. And a soldier isn't a core: an SM keeps up to 1,536 threads on hand but has only 128 FP32 lanes to run them on.

## Step 1: threads, blocks and grids

[Part 1](/blog/gpu-from-python-to-silicon/) introduced the vocabulary: the CPU is the **host**, the GPU is the **device**, and a **kernel** is a function that runs on the GPU. [Inference Part 3](/blog/gpus-for-llms/) described the **SM** (streaming multiprocessor), the GPU's unit of cores, registers and on-chip memory. This NVIDIA GeForce RTX 4060 Laptop GPU has 24.

In the words of the CUDA Programming Guide's [programming model chapter](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html), a kernel is launched "with many threads, often millions of threads. These threads are organized into blocks," called **thread blocks**, and "thread blocks are organized into a grid." Every thread runs the same code. Each reads its position from built-in variables: `threadIdx.x` is its index inside its block, `blockIdx.x` is its block's index in the grid, and `blockDim.x` is the number of threads per block. The [guide's vector-add example](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) combines them into one global index:

```text
i = blockIdx.x * blockDim.x + threadIdx.x
```

With 256 threads per block, thread 5 of block 2 gets `i = 2 × 256 + 5 = 517`. Block 0 covers elements 0 to 255, block 1 covers 256 to 511, and so on: one thread per element.

The length rarely divides evenly. A million elements in blocks of 256 need `(1,000,000 + 255) / 256 = 3,907` blocks (adding 255 makes integer division round up). That's 1,000,192 threads, so the last block has 64 threads with work and 192 without, and the kernel needs a guard, `if (i < n)`.

One more rule: "it be possible to execute thread blocks in any order, in parallel or in series." Blocks can't depend on each other, which is what lets the same kernel run on a 24-SM laptop and a 132-SM H100.

### A complete first kernel

Here's the whole program: allocate, copy the inputs over, launch, time, check, free. `__global__` marks a kernel, and `<<<blocks, threads>>>` launches it, grid size first.

```cpp
// vadd.cu: c = a + b on the GPU, timed with CUDA events.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
  printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  exit(1); } } while (0)

__global__ void vadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's element
  if (i < n) c[i] = a[i] + b[i];                  // the last block may overhang n
}

int main() {
  const int n = 1 << 26;                          // 67,108,864 floats
  const size_t bytes = n * sizeof(float);         // 256 MiB per array
  std::vector<float> ha(n), hb(n), hc(n);
  for (int i = 0; i < n; i++) { ha[i] = i % 1000; hb[i] = 0.5f; }

  float *a, *b, *c;                               // device (GPU) pointers
  CHECK(cudaMalloc(&a, bytes)); CHECK(cudaMalloc(&b, bytes)); CHECK(cudaMalloc(&c, bytes));
  CHECK(cudaMemcpy(a, ha.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b, hb.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = 256, blocks = (n + threads - 1) / threads;
  cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  std::vector<float> ms(51);
  for (int r = -150; r < 51; r++) {               // 150 warm-up runs, then 51 timed
    CHECK(cudaEventRecord(start));
    vadd<<<blocks, threads>>>(a, b, c, n);
    CHECK(cudaGetLastError());                    // catches launch errors
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));            // catches errors while running
    if (r >= 0) CHECK(cudaEventElapsedTime(&ms[r], start, stop));
  }
  std::sort(ms.begin(), ms.end());
  printf("%d blocks x %d threads: median %.3f ms, %.1f GB/s\n",
         blocks, threads, ms[25], 3.0 * bytes / (ms[25] * 1e6));

  CHECK(cudaMemcpy(hc.data(), c, bytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; i++)
    if (hc[i] != ha[i] + hb[i]) { printf("wrong result at %d\n", i); return 1; }
  printf("all %d results correct\n", n);
  CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(c));
}
```

`nvcc` is NVIDIA's CUDA compiler. `-arch=sm_89` compiles for compute capability 8.9, NVIDIA's version number for the Ada generation's features and limits:

```text
$ nvcc -O3 -arch=sm_89 -o vadd vadd.cu
$ ./vadd
262144 blocks x 256 threads: median 3.554 ms, 226.6 GB/s
all 67108864 results correct
```

The kernel is two lines; the rest is bookkeeping every CUDA program carries.

## Step 2: timing it against the bandwidth limit

Each element costs one addition and 12 bytes: 4 read from `a`, 4 from `b`, 4 written to `c`. That's `3 × 67,108,864 × 4 = 805 MB` per run. The kernel is **memory-bound**: its limit is how fast DRAM (the GPU's main memory) delivers bytes.

All numbers come from [Part 1](/blog/gpu-from-python-to-silicon/)'s machine: the RTX 4060 Laptop GPU (8 GB, compute capability 8.9) under WSL2, driver 592.82, CUDA 12.8 (`nvcc` 12.8.93), PyTorch 2.11.0. Times are from CUDA events, the median of 51 runs after warm-up. During the memory-bound runs, `nvidia-smi` showed the SM clock at 2,175 to 2,190 MHz, the memory clock at its full 8,000 MHz and power up to about 41 W, under an enforced power limit that varied from 39 to 50 W.

The rated bandwidth is 256 GB/s: a 128-bit bus at 16 gigabits per second per pin (`128 × 16 / 8 = 256`). To see what's reachable, compare kernels that only move memory:

| Kernel (n = 2^26 floats per array) | Bytes moved | Time | GB/s | Share of 256 GB/s |
|---|---|---|---|---|
| Read two arrays, write nothing | 537 MB | 2.150 ms | 249.7 | 98% |
| `cudaMemset` (write only) | 268 MB | 1.081 ms | 248.2 | 97% |
| Vector add, 4 floats per thread (`float4`) | 805 MB | 3.493 ms | 230.6 | 90% |
| Vector add, the kernel above | 805 MB | 3.550 ms | 226.8 | 89% |
| `cudaMemcpy` device to device | 537 MB | 2.373 ms | 226.3 | 88% |
| Copy kernel, `c[i] = a[i]` | 537 MB | 2.381 ms | 225.5 | 88% |

Pure reads and pure writes come within 3% of the rating. Any mix, including NVIDIA's `cudaMemcpy`, levels off near 226 GB/s; [Part 3](/blog/gpu-memory-hierarchy/) looks at why. So the first kernel you write already matches NVIDIA's copy routine. That's normal for simple memory-bound kernels: the gains come from not running them at all, by fusing them into neighbors ([Part 9](/blog/kernel-fusion-cuda-graphs/)).

NVIDIA's Nsight Compute profiler (`ncu`, covered in [Part 4](/blog/gpu-profiling-roofline/)) confirms the byte count: 537.0 MB read from DRAM and 254.7 MB written, against 536.9 MB and 268.4 MB expected. The missing 14 MB of writes was likely still in the 32 MB L2 cache when the kernel ended.

One timing trap: in a fresh process, the first 0.2 seconds of runs measured 198 GB/s before stepping up to 227 GB/s. Touching the memory first didn't change that, and `nvidia-smi` showed the GPU moving from its P3 to its P0 performance state as the load began, so it looks like power-management warm-up. Hence the 150 warm-up runs (about 0.6 seconds) in the listing.

### How big should a block be?

Sweeping the block size at the same array size:

| Threads per block | 32 | 64 | 128 | 256 | 512 | 1,024 |
|---|---|---|---|---|---|---|
| Blocks | 2,097,152 | 1,048,576 | 524,288 | 262,144 | 131,072 | 65,536 |
| GB/s | 227.5 | 226.8 | 226.9 | 226.9 | 227.3 | 243.9 |

From 32 to 512 threads, block size barely matters. The 1,024-thread result is 7% faster, and Step 4 suggests why. NVIDIA's [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) suggests "between 128 and 256 threads per block is a good initial range," always a multiple of 32.

### How big should the array be?

Now shrink the array, with the L2 cache overwritten by 64 MB of other data before each run so the test measures DRAM:

| Elements | Bytes moved | Time | GB/s |
|---|---|---|---|
| 2^10 (1,024) | 12.3 KB | 5.1 µs | 2.4 |
| 2^14 (16,384) | 197 KB | 6.1 µs | 32 |
| 2^18 (262,144) | 3.1 MB | 19.5 µs | 162 |
| 2^20 (1,048,576) | 12.6 MB | 63.5 µs | 198 |
| 2^22 (4,194,304) | 50 MB | 240 µs | 210 |
| 2^24 (16,777,216) | 201 MB | 904 µs | 223 |
| 2^26 (67,108,864) | 805 MB | 3,558 µs | 226 |

Below a few hundred thousand elements, a fixed cost of roughly 5 µs per launch dominates, as Part 1 found for small matrix multiplies. Above 16 million, the kernel sits at the bandwidth limit. Skip the cache flush and the 2^20 case, which fits in L2, runs at 768 GB/s, three times what DRAM allows; [Part 3](/blog/gpu-memory-hierarchy/) measures that hierarchy.

## Step 3: warps, SIMT and divergence

The block is what you choose; the **warp** is what the hardware runs. In the guide's words, "threads are organized into groups of 32 threads called warps. A warp executes the kernel code in a Single-Instruction Multiple-Threads (SIMT) paradigm." Threads 0 to 31 form warp 0, 32 to 63 warp 1, so a 256-thread block is 8 warps. A 100-thread block still occupies 4 warps, with 28 lanes idle, which is why block sizes are multiples of 32.

**SIMT** (single instruction, multiple threads) means "all threads in the warp execute the same instruction simultaneously," each on its own data. Unlike a CPU's SIMD (single instruction, multiple data) vector units, you write code for one scalar thread, and each thread may branch on its own. In hardware, per the [Ada architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), each SM is "divided into four processing blocks (or partitions)," each with "one warp scheduler" and 32 FP32 lanes. Each cycle, a scheduler picks a ready warp and issues its next instruction.

![A diagram in two rows. The top row, what you launch: a grid of 262,144 blocks contains blocks of 256 threads, each block contains warps of 32 threads that execute one instruction at a time, and each warp contains threads that each handle one element through their own index i. The bottom row, where it runs on the RTX 4060 Laptop: the GPU has 24 SMs, each SM holds up to 6 such blocks or 48 warps at once, and each SM has 4 warp schedulers with 32 FP32 lanes each. Dashed arrows connect the rows: the grid's blocks spread over the GPU, each block stays on one SM, and each warp is issued by one warp scheduler.](/diagrams/g02-thread-hierarchy.svg "What you launch, and where each level runs.")

### What happens at a branch

If threads in a warp disagree at an `if`, "the threads which do not follow the branch will be masked off while the threads which follow the branch are executed," which the guide calls **warp divergence**. Different warps are independent: "Branch divergence occurs only within a warp" ([SIMT execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)).

To test it, branch on `threadIdx.x % 2`, which splits every warp in half, and on `(threadIdx.x / 32) % 2`, which sends whole warps one way or the other. First with the vector add's tiny body, adding on one side and subtracting on the other:

| Memory-bound body, n = 2^24 | Time | GB/s |
|---|---|---|
| No branch taken differently | 0.897 ms | 224.4 |
| Split inside every warp | 0.849 ms | 237.2 |
| Split by whole warps | 0.898 ms | 224.2 |

Divergence costs nothing here; the in-warp split even ran a few percent faster, consistently. The compiled code shows why: the compiler turned the branch into **predication**, issuing both the subtract-and-store and the add-and-store and switching each thread's copy on or off with a one-bit flag. Two extra instructions are invisible when warps spend their time waiting for memory.

Now give each path real work: 1,024 dependent steps per thread, a fused multiply-add per step on path A and a multiply-add plus a multiply on path B.

| Compute-bound body, n = 2^24 | Time | Active threads per instruction (ncu) | Warp instructions (ncu) |
|---|---|---|---|
| All threads take path A | 3.607 ms | 32 | 660 million |
| All threads take path B | 5.812 ms | 32 | 1,200 million |
| Split inside every warp | 8.933 ms | 16.1 | 1,848 million |
| Split by whole warps | 4.858 ms | 32 | 932 million |

Split inside every warp, the kernel takes about as long as path A followed by path B (8.9 ms against 3.6 + 5.8 = 9.4), and Nsight Compute counts 16.1 active threads per issued instruction instead of 32: half of every warp sits idle. Split by whole warps, each warp runs one path at full width, the time lands between the two, and it's 1.8 times faster than the in-warp split. Path A alone reaches 9.5 TFLOPS, 71% of the 13.4 TFLOPS FP32 peak at 2,175 MHz (`24 SMs × 128 lanes × 2 operations × 2.175 GHz`). This run pushed power to 48 W and the SM clock dipped to about 1,900 MHz at times; Nsight Compute locks clocks to base while profiling, so only its counts are quoted.

So when the two sides of a branch do real work, branch on something that's the same across a warp.

## Step 4: SMs, occupancy and grid size

A block runs on one SM from start to finish, and the SM reserves its resources up front: thread slots, registers and shared memory (on-chip scratch memory a block's threads share). The limits for compute capability 8.9, from the guide's [compute capabilities tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) and the [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html), match what `cudaGetDeviceProperties` reports here:

| Per-SM resource (compute capability 8.9) | Limit |
|---|---|
| Resident threads | 1,536 (48 warps) |
| Resident blocks | 24 |
| 32-bit registers | 65,536 (64K) |
| Registers per thread | 255 at most |
| Shared memory | 100 KB per SM; up to 99 KB per block (above 48 KB the kernel must opt in) |
| Threads per block | 1,024 |

**Occupancy** is "the ratio of the number of active warps to the maximum number of active warps supported by the SM," set by whichever resource runs out first. It matters because "switching between warps incurs no cost": while one warp waits for memory, the scheduler issues from another.

### A worked example

A kernel that keeps 64 values per thread in registers uses 72 registers per thread, per `nvcc --resource-usage`. With 256 threads per block:

```text
registers per block = 256 threads × 72 = 18,432
blocks per SM       = 65,536 ÷ 18,432 = 3.56, so 3 blocks
resident warps      = 3 blocks × 8 warps = 24 of 48
occupancy           = 50%
```

Thread slots would allow 6 blocks, but registers allow 3. CUDA's occupancy API, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, returns the same 3. The vector add needs only 12 registers and reaches 6 blocks, 100%.

Two edge cases: with 32-thread blocks, the 24-block limit bites first (768 threads, 50%), and with 1,024-thread blocks only one fits in 1,536 slots (67%). Ask for 1,024 threads of the 72-register kernel and the launch fails: 73,728 registers don't fit in 65,536, and `cudaGetLastError` returns "too many resources requested for launch."

### Does occupancy make it faster?

The Best Practices Guide suggests an experiment: reserve shared memory the kernel never uses, which caps blocks per SM without changing the code. For the vector add (256 threads per block), with Nsight Compute's achieved occupancy:

| Unused shared memory per block | Blocks per SM | Theoretical occupancy | Achieved (ncu) | GB/s |
|---|---|---|---|---|
| 0 | 6 | 100% | 81% | 226.6 |
| 19 KB | 5 | 83% | 65% | 227.6 |
| 24 KB | 4 | 67% | 56% | 241.8 |
| 32 KB | 3 | 50% | 40% | 242.1 |
| 48 KB | 2 | 33% | 27% | 224.5 |
| 60 KB | 1 | 17% | 16% | 141.5 |

Too little hurts: with one 8-warp block per SM, bandwidth falls to 142 GB/s, because 8 warps can't keep enough loads in flight to cover memory latency. But the fastest runs came at 50 to 67%, not 100%, like the 1,024-thread blocks in Step 2 (also 67%). The Best Practices Guide says as much: "higher occupancy does not always equate to higher performance," but "low occupancy always interferes with the ability to hide memory latency." Why full occupancy is slower here isn't settled by these counters; more warps means more interleaved access streams for DRAM to serve, and [Part 3](/blog/gpu-memory-hierarchy/) looks at access patterns.

### Too few blocks

Fewer blocks than SMs leaves SMs idle. To test it, rewrite the kernel as a **grid-stride loop**, a pattern [Mark Harris described in 2013](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/): each thread steps through the array by the total number of threads, so any grid covers any array.

```cpp
__global__ void vadd_gs(const float* a, const float* b, float* c, int n) {
  int stride = gridDim.x * blockDim.x;                 // threads in the whole grid
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    c[i] = a[i] + b[i];
}
```

Then launch it with 1 to 1,152 blocks, n = 2^26:

| Blocks | 256 threads per block: GB/s | 1,024 threads per block: GB/s |
|---|---|---|
| 1 | 10.3 | 38.6 |
| 4 | 38.8 | 136.9 |
| 12 | 108.4 | 231.2 |
| 24 | 177.4 | 235.2 |
| 25 | 181.5 | 195.5 |
| 48 | 231.0 | 236.4 |
| 72 | 237.7 | 234.9 |
| 1,152 | 236.9 | 234.2 |

With 256-thread blocks, bandwidth grows almost in step with the block count up to 24, as each block lands on an idle SM. But one 8-warp block per SM reaches only 177 GB/s, 75% of the best; it takes 48 to 72 blocks to saturate memory. With 1,024-thread blocks, each block brings 32 warps, and 12 blocks already reach 231 GB/s.

The 25-block entry shows a different effect. A 1,024-thread block takes 1,024 of an SM's 1,536 thread slots, so no SM can hold two. Blocks 1 to 24 finish their 24/25 of the data in about 3.3 ms; block 25 waits, then runs alone at single-block speed, 38.6 GB/s, for another 0.83 ms. The predicted 4.1 ms matches the measured 4.118 ms. NVIDIA's [matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) calls this **wave quantization**: blocks run in waves of what the GPU holds at once, and a small "tail" wave leaves most SMs idle. [Part 5](/blog/cuda-matmul-optimization/) meets it again in matrix multiplies.

At 72 blocks the grid-stride version reached 238 GB/s, 5% faster than one thread per element, which fits Harris's advice to launch "a number of blocks that is a multiple of the number of multiprocessors on the device."

## Step 5: when it goes wrong

Kernel launches don't return an error code, and per the guide, a clean error state right after a launch "does not mean the kernel has executed successfully or even started execution." Two kinds of error show up at different times:

```cpp
#include <cstdio>
__global__ void write_one(float* p) { p[threadIdx.x] = 1.0f; }
int main() {
  float* buf; cudaMalloc(&buf, 4096);
  // 1) a bad launch configuration: the kernel never starts
  write_one<<<1, 2048>>>(buf);                       // 2048 > 1024 threads per block
  printf("after bad launch, cudaGetLastError: %s\n", cudaGetErrorString(cudaGetLastError()));
  printf("next call, cudaMalloc:              %s\n", cudaGetErrorString(cudaMalloc(&buf, 4096)));
  // 2) a bad pointer: the launch succeeds, the kernel fails later on the GPU
  write_one<<<1, 32>>>((float*)0x10);                // not a device allocation
  printf("after launch, cudaGetLastError:     %s\n", cudaGetErrorString(cudaGetLastError()));
  printf("cudaDeviceSynchronize:              %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
  printf("cudaGetLastError:                   %s\n", cudaGetErrorString(cudaGetLastError()));
  printf("next call, cudaMalloc:              %s\n", cudaGetErrorString(cudaMalloc(&buf, 4096)));
}
```

```text
after bad launch, cudaGetLastError: invalid configuration argument
next call, cudaMalloc:              no error
after launch, cudaGetLastError:     no error
cudaDeviceSynchronize:              an illegal memory access was encountered
cudaGetLastError:                   an illegal memory access was encountered
next call, cudaMalloc:              an illegal memory access was encountered
```

A bad configuration is caught at launch, and the program can carry on. A bad memory access happens later, on the GPU, after the launch reported success. It surfaces at the next call that waits for the GPU and never goes away: even an unrelated `cudaMalloc` fails. The [CUDA runtime documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html) says this error "leaves the process in an inconsistent state," and "the process must be terminated and relaunched." Hence `vadd.cu` checks `cudaGetLastError()` after every launch and checks the call that synchronizes.

### Bugs that don't crash

Worse are bugs that trip nothing. Change the guard to `i <= n`, a classic off-by-one, and run a million elements:

```cpp
__global__ void vadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) c[i] = a[i] + b[i];          // bug: should be i < n
}
```

```text
$ nvcc -O3 -arch=sm_89 -lineinfo -o oob oob.cu && ./oob
kernel finished: no error
```

Thread 1,000,000 reads and writes one element past the end of each array, and nothing complains, most likely because the address just past each array still falls inside memory the allocator reserved. In a real program that write could silently corrupt a neighboring buffer.

**compute-sanitizer**, part of the CUDA toolkit, is built to catch it. Its [documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) describes the memcheck tool as "capable of precisely detecting and attributing out of bounds and misaligned memory access errors." You run the program under it, `compute-sanitizer --tool memcheck ./oob`, and compiling with `-lineinfo` lets it name the source line of the bad access. On this laptop it wouldn't start: under WSL2 it needs NVIDIA's GPU debugger interface, which is switched off in Windows by default, so there's no captured output here. On native Linux it runs as is.

The suite also includes racecheck for shared-memory data races, initcheck for reads of uninitialized device memory, and synccheck for misused synchronization.

## In the real world

PyTorch's elementwise operations are this kernel with more engineering. `torch.add` goes through **TensorIterator**, which PyTorch's [source](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/TensorIterator.h) calls "a helper class for element-wise operations" that "handles broadcasting and type conversions of operands." The math itself is a one-line lambda, `a + b`, handed to `gpu_kernel`.

The launch logic is in [CUDALoops.cuh](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/CUDALoops.cuh). For contiguous tensors, `launch_vectorized_kernel` loads up to 16 bytes per instruction (4 floats; an 8-wide version is compiled only for Hopper and Blackwell) and gives each thread 8 elements, in blocks of `num_threads() = C10_WARP_SIZE * 4`, 128 threads ([thread_constants.h](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/thread_constants.h)). Non-contiguous tensors take a slower path that computes each element's address.

The profiler confirms it here: `torch.add` on 2^26 floats launched `vectorized_elementwise_kernel<4, CUDAFunctor_add<float>, ...>` as 65,536 blocks of 128 threads, 1,024 elements per block, and ran at 233.4 GB/s, 3% faster than the hand-written kernel.

LLM kernels are the same model at larger scale. [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023) describes the original FlashAttention's grid: "We use 1 thread block to process one attention head, and there are overall batch size · number of heads thread blocks," which "is efficient when this number is large (say ≥ 80)" on an A100's 108 SMs. With long sequences and few heads, SMs sat idle, the problem the 24-block test shows, so FlashAttention-2 also splits the sequence across blocks, which "helps improve occupancy," and uses "4 or 8 warps per thread block." Decode is harsher still, with one query token per request, which is why Flash-Decoding splits the KV cache across blocks ([Inference Part 7](/blog/flashattention/)).

## Common misconceptions

- **"A thread is a core."** An Ada SM has 128 FP32 lanes and keeps up to 1,536 threads resident, switching between warps at no cost. The first kernel here launched 67 million threads.
- **"Bigger blocks are faster."** Block sizes from 32 to 512 all ran within 1% of 227 GB/s. The 1,024-thread blocks were faster, and 256-thread blocks matched them once occupancy was capped at the same 67%.
- **"One block per SM keeps the GPU busy."** With 24 blocks of 256 threads every SM had work, yet bandwidth reached only 177 GB/s: 8 warps per SM can't hide memory latency. It took 48 to 72 blocks.
- **"Branches are slow on GPUs."** Only branches that split a warp, and only when the split code does real work. Splitting every warp took 1.8 times as long as splitting by whole warps with a compute-heavy body, and cost nothing with a memory-bound one.
- **"No error means the kernel worked."** An illegal memory access shows up at the next synchronizing call, and an off-by-one write may never show up without compute-sanitizer.

## Key takeaways

- You write a CUDA kernel for one thread. Each thread finds its element with `blockIdx.x * blockDim.x + threadIdx.x` and needs a guard for the last, partly filled block.
- A simple vector add already hits the memory limit: 227 GB/s on this laptop, the same as `cudaMemcpy` and 89% of the rated 256 GB/s. Count the bytes, divide by the time, and compare.
- Threads run in warps of 32. A branch that splits a warp makes it run both sides, which nearly doubled a compute-heavy kernel's time and cost nothing in a memory-bound one.
- Occupancy is resident warps over the SM's maximum, limited by threads, registers, shared memory and blocks per SM. Too little starves memory; past about half, more didn't help here.
- Launch enough blocks: a few per SM, not one, and watch for a last round of leftover blocks. Check errors after every launch and every synchronize, and run compute-sanitizer before you trust a kernel.

## What's next

The vector add touches each byte once, so all it needs from memory is raw bandwidth. Real kernels reuse data, and that's where the GPU's memory hierarchy comes in. [Part 3](/blog/gpu-memory-hierarchy/) measures it level by level, from registers and shared memory to the L2 cache and DRAM, and shows how the order in which threads touch memory can cost you most of that bandwidth.

## Further reading

- [CUDA Programming Guide (NVIDIA)](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html): the official reference, especially [the programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) (blocks, grids, warps, SIMT), [writing SIMT kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) (occupancy, worked examples), [intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) (error handling) and [compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) (per-SM limits).
- [CUDA C++ Best Practices Guide (NVIDIA)](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): occupancy, block-size heuristics, and the shared-memory trick used above to vary occupancy.
- [NVIDIA Ada GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html): the compute capability 8.9 limits in one page.
- [NVIDIA Ada GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf): the SM's four partitions, warp schedulers and FP32 lanes.
- [Compute Sanitizer documentation (NVIDIA)](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html): memcheck, racecheck, initcheck and synccheck.
- [CUDA Runtime API: error types (NVIDIA)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html): what each `cudaError_t` means and which ones end the process's use of CUDA.
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)](https://arxiv.org/abs/2307.08691): how a production kernel chooses its grid and warps.
- [PyTorch CUDALoops.cuh (PyTorch source, v2.11.0)](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/CUDALoops.cuh): the vectorized elementwise kernel behind `torch.add`.
- [Matrix Multiplication Background User's Guide (NVIDIA)](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html): wave quantization and tail waves, with A100 examples.
- [CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops (Mark Harris, NVIDIA, 2013)](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/): the grid-stride pattern and why to size grids by SM count.
