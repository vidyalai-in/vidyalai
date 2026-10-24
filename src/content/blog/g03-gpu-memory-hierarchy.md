---
title: "The GPU Memory Hierarchy: Registers, Caches and DRAM, Measured"
description: "Measure every level of a GPU's memory, from registers to DRAM, and see how access order, bank conflicts and spills cost most of the bandwidth."
slug: "gpu-memory-hierarchy"
category: "gpu-kernels"
tags: ["gpu", "cuda", "latency", "caching"]
series: "llm-gpu"
seriesOrder: 3
pubDate: 2026-10-24
---

## TL;DR

A GPU's memory is a ladder: registers inside each SM, then a per-SM L1 cache that doubles as programmer-managed shared memory, then an L2 cache shared by the whole chip, then DRAM off the chip. On a laptop RTX 4060, the same read loop runs at about 6,000 GB/s when its data fits in L1, 1,440 GB/s from L2 and 253 GB/s from DRAM, and one load takes 40, 264 and about 610 clock cycles at those levels. The order in which threads touch memory matters as much as the level: reading every eighth float delivers 44 GB/s of useful data instead of 224, because memory moves in 32-byte sectors. You'll also see why a matrix transpose needs shared memory plus one padding column, what a register spill costs, and why LLM decode is stuck at the bottom of the ladder.

## Why it matters

Here is one kernel that reads 2.15 GB of floats on a laptop GPU. Between runs, only the size of the buffer it loops over changes. With a 64 KB buffer it finishes in 0.36 ms. With a 256 MB buffer it takes 8.5 ms, 24 times longer, for exactly the same number of bytes read.

Change something else instead: keep the buffer large and make each thread read every eighth float instead of the next one. The amount of data the kernel uses stays the same, but the time grows fivefold.

Neither effect shows in the code. Both come from the memory hierarchy, and they decide the speed of nearly every kernel in an LLM. [Part 2](/blog/cuda-programming-model/) ended with a vector add at 227 GB/s, 89% of the rated bandwidth, and two loose ends: why every kernel that both reads and writes stopped near 226 GB/s, and what made the cached runs three times faster. This part answers both, level by level.

## The intuition

Picture a workshop. The tools in your hands are registers: instant, but you hold only a few. The workbench beside you is L1 and shared memory: small, a step away, and you can lay it out however you like. The storeroom down the corridor is L2, shared by every workbench in the building. The warehouse across town is DRAM: huge, but everything comes by truck.

The truck never brings one screw. It brings a whole crate (a 32-byte sector), and the storeroom shelves hold rows of four crates (a 128-byte cache line). If 32 workers each want one screw from the same crate, one trip serves them all. If each wants a screw from a different crate, the truck brings 32 crates and most of every crate goes unused. That's the difference between coalesced and strided access.

The analogy breaks in two places. A GPU's DRAM keeps hundreds of trucks on the road at once, so how many crates arrive per second and how long one trip takes are very different numbers. And the workbench has 32 drawers (banks), each opened once per clock; workers who reach into the same drawer must queue.

## Step 1: the hierarchy on this GPU

The machine is [Part 1](/blog/gpu-from-python-to-silicon/)'s: an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, AD107, compute capability 8.9) under WSL2, driver 592.82, CUDA 12.8 (`nvcc -O3 -arch=sm_89`). Times are CUDA events, the median of 5 to 51 runs after about 0.6 seconds of warm-up (Part 2 found the GPU needs that long to leave its low-power state). `nvidia-smi` showed the SM clock at 2,175 to 2,190 MHz (briefly 1,820), the memory clock at 8,000 MHz except where noted, and up to 43 W under a 41 to 50 W limit.

The CUDA runtime describes the hierarchy through `cudaGetDeviceProperties`. A short program prints the relevant fields:

```text
$ ./props
NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, 24 SMs
registers per SM:        65536 x 32-bit = 256 KB
shared memory per SM:    100 KB (per block: 48 KB default, 99 KB opt-in)
L1 caches global/local: 1/1
L2 cache:                32 MB (persisting max 22 MB)
constant memory:         64 KB
DRAM:                    8.59 GB, 128-bit bus, memory clock 8001 MHz
```

From the top, with NVIDIA's descriptions:

- Registers hold each thread's working values: "64K 32-bit registers per SM," per the [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).
- L1 cache and **shared memory**, scratch space that the threads of one block share and the kernel manages itself. They are one block of SRAM (static RAM, the fast memory built into the chip): "the combined L1 cache capacity is 128 KB," and Ada "supports shared memory capacity of 0, 8, 16, 32, 64 or 100 KB per SM," leaving the rest as L1.
- L2 cache, 32 MB here, which the [Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) says "is shared among all the SMs."
- DRAM (dynamic RAM): 8 GB of GDDR6 graphics memory on a 128-bit bus, per NVIDIA's [laptop specs](https://www.nvidia.com/en-us/geforce/laptops/compare/). Kernels see it as **global memory**, and the [Nsight Compute guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) notes that "all GPU units communicate to main memory through the Level 2 cache."

### The bandwidth limit, from the clock

DRAM's theoretical bandwidth comes from two of those numbers. NVIDIA's [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) computes it as the memory clock, times 2 for double data rate (two transfers per clock), times the bus width in bytes; its worked example gives a V100 898 GB/s. For this GPU:

```text
8,001 MHz × 2 transfers per clock × 128 bits ÷ 8 bits per byte = 256.0 GB/s
```

That matches the 16 gigabits per second per pin that Part 2 used: `128 × 16 ÷ 8 = 256`.

![A vertical diagram of the memory hierarchy on the RTX 4060 Laptop GPU. Inside each of the 24 SMs: registers, 256 KB per SM, about 4.6 cycles for a dependent operation; below them, L1 cache plus shared memory, 128 KB per SM, 30 to 40 cycles, about 6 TB/s in total across the chip. A link labeled 32-byte sectors leads to the L2 cache, 32 MB shared by all SMs, 264 cycles, about 1.4 TB/s. A link labeled 128-bit bus leads to GDDR6 DRAM off the chip, 8 GB holding global and local memory, about 610 cycles and about 250 GB/s.](/diagrams/g03-memory-levels.svg "The memory hierarchy of the RTX 4060 Laptop GPU, with the latencies and bandwidths measured in this article.")

From L1 down to DRAM, latency grows about 15-fold and bandwidth falls about 24-fold. The rest of this article measures every number in that picture.

## Step 2: bandwidth and latency, level by level

### A copy against the spec

Start at the bottom, with arrays far too big for the caches:

| Kernel (64 to 128 MiB arrays) | GB/s | Share of 256 GB/s |
|---|---|---|
| Read only (4 arrays per thread) | 248.7 | 97% |
| Write only (4 arrays per thread) | 243.6 | 95% |
| `cudaMemcpy`, device to device | 227.0 | 89% |
| Copy kernel, one `float` per thread | 227.6 | 89% |
| Copy kernel, one `float4` per thread | 226.6 | 89% |

Loading four floats per instruction (`float4`) made no difference, because DRAM, not instruction count, is the limit. A second run landed within 2% of every number.

So pure reads reach 97% of the rating, and anything that mixes reads and writes, NVIDIA's own `cudaMemcpy` included, stops at 89%. To see why, vary the mix:

| Per thread | GB/s |
|---|---|
| Read 4, write 0 | 248.7 |
| Read 4, write 1 | 244.9 |
| Read 2, write 1 | 224.7 |
| Read 1, write 1 | 224.1 |
| Read 1, write 2 | 229.7 |
| Read 1, write 4 | 233.7 |
| Read 0, write 4 | 243.6 |

The closer the mix gets to even, the lower the bandwidth. That fits how DRAM works: the same data pins carry reads and writes, and switching direction leaves the bus idle for some cycles. A [Rambus article](https://www.design-reuse.com/article/58224-how-memory-architectures-affect-system-performance/) notes that "bus turnaround by itself can create very long periods of inactivity on the data channel." NVIDIA doesn't publish the GDDR6 timings, so this is the likely explanation rather than a proof. In practice, a kernel that reads and writes equal amounts should expect about 225 GB/s here, not 256.

### The working-set sweep

To measure the caches, every thread block reads the same buffer over and over, each block starting at its own offset and wrapping around at the end. Every launch reads 2.15 GB in total, and only the buffer size changes:

```cpp
__global__ void read_ws(const float4* __restrict__ buf, unsigned mask, unsigned per_block, float* out) {
  unsigned start = blockIdx.x * ((mask + 1) / gridDim.x);       // each block starts elsewhere
  float4 a0 = {0, 0, 0, 0}, a1 = a0;
  for (unsigned j = threadIdx.x; j < per_block; j += 2 * blockDim.x) {
    float4 x = buf[(start + j) & mask];
    float4 y = buf[(start + j + blockDim.x) & mask];
    a0.x += x.x; a0.y += x.y; a0.z += x.z; a0.w += x.w;
    a1.x += y.x; a1.y += y.y; a1.z += y.z; a1.w += y.w;
  }
  float s = a0.x + a0.y + a0.z + a0.w + a1.x + a1.y + a1.z + a1.w;
  if (s == -1.0f) out[0] = s;                                    // never true; keeps the loads alive
}
```

It runs 6 blocks of 256 threads per SM, with the split set to favor L1, so each SM's working set is the buffer size. Nsight Compute (`ncu`, NVIDIA's kernel profiler, covered in [Part 4](/blog/gpu-profiling-roofline/)) reported the hit rates:

| Buffer | Time | GB/s | L1 hit rate | L2 hit rate | Read from DRAM |
|---|---|---|---|---|---|
| 16 KB | 0.357 ms | 6,008 | 99.98% | n/a | 0 |
| 64 KB | 0.358 ms | 5,990 | 99.93% | n/a | 0 |
| 128 KB | 0.551 ms | 3,897 | 88.7% | 99.9% | 0 |
| 1 MB | 0.987 ms | 2,175 | 43.4% | 99.9% | 1 MB |
| 4 MB | 1.292 ms | 1,661 | 20.6% | 99.8% | 4 MB |
| 16 MB | 1.483 ms | 1,448 | 5.4% | 99.2% | 17 MB |
| 32 MB | 1.494 ms | 1,437 | 2.2% | 98.4% | 34 MB |
| 64 MB | 5.674 ms | 378 | 5.4% | 35.1% | 1.36 GB |
| 128 MB | 7.761 ms | 277 | 0.9% | 11.2% | 1.89 GB |
| 256 MB | 8.504 ms | 253 | 0.2% | 3.0% | 2.08 GB |

("n/a": almost nothing reached L2, so its rate is noise.) Three plateaus stand out:

1. Up to 64 KB, every load hits L1: about 6,000 GB/s (5,600 in a second run), 24 times the DRAM rate.
2. From 128 KB to 32 MB, data spills into L2. The L1 hit rate falls gradually, and once it's near zero the L2 alone delivers about 1,440 GB/s, 5.7 times DRAM. That's Part 2's second loose end: its vector add ran at 768 GB/s when its 12.6 MB stayed in L2.
3. Past 32 MB, nearly every byte comes from DRAM: 253 GB/s at 256 MB, 99% of the rating.

Shared memory, read the same way from a 16 KB tile, ran at 5,600 to 6,300 GB/s, like L1, as you'd expect from the same SRAM. Step 4 measures its limit at 128 bytes per clock per SM, `24 × 128 bytes × 2.19 GHz = 6.7 TB/s`.

### Latency: chasing pointers

Bandwidth is bytes per second with thousands of loads in flight; **latency** is how long one load takes. To measure it, one thread follows a chain where each load returns the address of the next, so loads can't overlap. The chain visits 128-byte lines in random order, and `clock64()` reads the SM's cycle counter:

```cpp
__global__ void chase(const unsigned* __restrict__ next, int steps, unsigned* out, long long* cycles) {
  unsigned j = 0;
  for (int k = 0; k < steps; k++) j = next[j];          // first lap: warm the caches
  long long t0 = clock64();
  for (int k = 0; k < steps; k++) j = next[j];          // timed lap
  long long t1 = clock64();
  out[0] = j; cycles[0] = t1 - t0;
}
```

| Where the data was | Buffer | Cycles per load | At 2,175 MHz |
|---|---|---|---|
| Register (dependent fused multiply-add, FMA) | n/a | 4.6 | 2 ns |
| Shared memory | 16 KB | 30 | 14 ns |
| L1 cache | 4 to 64 KB | 40 | 18 ns |
| L2 cache | 256 KB to 16 MB | 264 | 121 ns |
| DRAM | 64 to 256 MB | 611 to 616 | 281 to 283 ns |

A load from DRAM takes 15 times as long as one from L1. CUDA events timing the whole kernel gave 19 to 20 ns, 122 ns and 270 to 287 ns per load, close to the cycle counts. The register row is a chain of fused multiply-adds (one multiply and one add in a single instruction), each waiting for the previous result.

The test times one load on an otherwise idle GPU, plus a cycle or two of loop arithmetic. Under heavy load, requests queue and latency rises. Random addresses over 64 to 256 MB may also miss the GPU's address-translation caches, and with a single thread running, the memory clock sat at 7,000 MHz for about half the samples. These are floors, not what a busy kernel sees.

### Why both numbers matter

The two measurements are linked by **Little's law**: the bytes in flight equal bandwidth times latency. To sustain 253 GB/s with 281 ns per load, the GPU needs `253 GB/s × 281 ns ≈ 71 KB` of reads outstanding at every moment, about 3 KB per SM. That's 185 `float4` loads or 740 `float` loads per SM, in flight all the time. It's why Part 2's kernel with one 8-warp block per SM stalled at 177 GB/s: its 256 threads couldn't keep enough loads in the air.

## Step 3: coalescing, or why access order matters

When a warp executes a load, the hardware combines its 32 addresses into as few transactions as it can. The Programming Guide explains that "global memory is accessed via 32-byte memory transactions," and "if consecutive threads in the warp request consecutive 4-byte words in memory, then the warp will request 128 bytes." Nsight Compute's guide defines the units: a **sector** is an "aligned 32 byte-chunk of memory," and "an L1 or L2 cache line is four sectors, i.e. 128 bytes." A warp load whose addresses fall in the fewest possible sectors is **coalesced**.

The test: each thread reads one float, `stride` floats apart, and writes it contiguously. Writing 64 MB of other data before each run flushes the L2, so reads come from DRAM:

```cpp
__global__ void strided(const float* __restrict__ in, float* __restrict__ out, int stride, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[(size_t)i * stride];
}
```

With n = 2^21 floats, the kernel uses 8.4 MB of input whatever the stride. "Useful GB/s" counts only those bytes plus the 8.4 MB written; the last three columns come from Nsight Compute:

| Stride (floats) | Time | Useful GB/s | Sectors per warp load | Read from DRAM | Bytes used per sector |
|---|---|---|---|---|---|
| 1 | 0.075 ms | 224.4 | 4 | 8.4 MB | 100% |
| 2 | 0.117 ms | 143.7 | 8 | 16.8 MB | 50% |
| 4 | 0.203 ms | 82.7 | 16 | 33.6 MB | 25% |
| 8 | 0.382 ms | 43.9 | 32 | 67.1 MB | 12.5% |
| 16 | 0.405 ms | 41.4 | 32 | 67.1 MB | 12.5% |
| 32 | 0.401 ms | 41.8 | 32 | 67.1 MB | 12.5% |

At stride 1, a warp's 32 floats fill exactly 4 sectors. At stride 2 they spread over 8 sectors and half of every sector is wasted, which is the Best Practices Guide's "a stride of 2 results in a 50% of load/store efficiency." At stride 8, 32 bytes apart, every thread lands in its own sector, so the warp pulls 32 sectors, 1,024 bytes, to use 128. Past that it can't get worse, since each thread already costs one whole sector. DRAM was still busy, moving 176 GB/s (67.1 MB in 0.382 ms), but seven-eighths of it was thrown away. This is the cost of access order that Part 2 promised: the same useful data at a fifth of the speed.

### Misalignment is nearly free

The Best Practices Guide also covers a warp reading 128 contiguous bytes from a misaligned address, where "five 32-byte segments will be requested" instead of four. Offsetting each read (`out[i] = in[i + offset]`, 64 MB each way):

| Offset (floats) | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| GB/s | 225.6 | 230.0 | 230.0 | 230.0 | 230.0 | 229.5 | 227.6 |

Every offset ran within 2%. At offset 1, Nsight Compute shows 5 sectors per warp load in L1, but only 3% more sectors reaching L2 and no extra DRAM traffic: the fifth sector is the one the neighboring warp needs next, and the caches catch it.

## Step 4: shared memory and bank conflicts

Per the Programming Guide, "shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks": word 32 is back in bank 0. "When multiple threads in the same warp attempt to access different elements in the same bank, a **bank conflict** occurs," and the accesses are serialized. Threads reading the same word don't conflict; that's a broadcast.

To measure it, each lane of a warp reads word `lane × stride` of a shared array, 16,384 times in a dependent chain, in 48 warps per SM:

| Stride (words) | Lanes per bank | Time | Relative |
|---|---|---|---|
| 1 | 1 | 0.743 ms | 1.0× |
| 2 | 2 | 0.736 ms | 1.0× |
| 4 | 4 | 1.459 ms | 2.0× |
| 8 | 8 | 2.907 ms | 3.9× |
| 16 | 16 | 5.801 ms | 7.8× |
| 32 | 32 | 11.591 ms | 15.6× |
| 33 | 1 | 0.735 ms | 1.0× |

From stride 4 on, each doubling of lanes per bank doubles the time. (Stride 2 is free only because the stride-1 run used about half of shared memory's throughput.) Stride 33 spreads the lanes over all 32 banks again, so it costs nothing.

The 32-way case also measures shared memory's ceiling. Each SM ran 786,432 warp loads, each split into 32 separate accesses (wavefronts, in Nsight Compute's terms): 25.2 million wavefronts in 25.4 million cycles at 2.19 GHz. That's one wavefront per clock, and an unconflicted wavefront delivers one 4-byte word from each of 32 banks: 128 bytes per clock per SM.

### A transpose, three ways

A transpose reads rows and writes columns, so one side is always strided. Mark Harris's [2013 transpose post](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) shows the fix: read a 32 × 32 tile into shared memory with coalesced loads, synchronize, and write it out transposed, also coalesced. Blocks of 32 × 8 threads handle four tile rows each:

```cpp
template <int PAD> __global__ void tiled(const float* in, float* out, int N) {
  __shared__ float tile[TILE][TILE + PAD];
  int x = blockIdx.x * TILE + threadIdx.x, y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];  // coalesced read
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x; y = blockIdx.x * TILE + threadIdx.y;          // swap block coordinates
  for (int j = 0; j < TILE; j += ROWS) out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];  // column of the tile
}
```

The last line reads a column of the tile. With 32 floats per row, a whole column sits in one bank: a 32-way conflict. Making the tile 33 wide (`PAD = 1`) shifts each row by one bank, as the Programming Guide recommends: "pad the shared memory by adding one to the column dimension."

Bandwidth counts one read and one write of the matrix. The 2,048 case fits both matrices (16 MB each) in L2 and times 20 back-to-back launches:

| Kernel | 4,096 × 4,096 (DRAM) | 2,048 × 2,048 (L2) |
|---|---|---|
| Copy, same access pattern, no transpose | 222.9 GB/s | 954 GB/s |
| Naive transpose (strided writes) | 101.7 GB/s | 101.3 GB/s |
| Tiled, `tile[32][32]` | 217.7 GB/s | 332 GB/s |
| Tiled, `tile[32][33]` | 221.0 GB/s | 932 GB/s |

All three transposes were checked element by element against the input. The L2 copy varied between 820 and 954 GB/s across runs; the transposes stayed within 1%. Nsight Compute, on the 4,096 case, explains the rows:

```text
naive:     global store sectors per request          32   (copy: 4)
tile[32]:  shared loads 524,288 instructions -> 16,874,002 wavefronts, 16,252,928 bank conflicts
tile[33]:  shared loads 524,288 instructions ->    525,227 wavefronts,          0 bank conflicts
```

The naive version's writes cost 32 sectors per warp, 8 times the copy's 4, and it runs at half speed or less at both sizes. Its DRAM writes were only 52 MB, though: the L2 merged the scattered writes into full sectors, so the waste is in the traffic between the SMs and L2.

From DRAM, the conflicted tile runs as fast as the padded one, because DRAM is slow enough that shared memory can serialize 32 ways and still keep up. From L2 the conflicts show, and padding makes it 2.8 times faster. (Under Nsight Compute, which locks clocks at base frequency, the conflict cost 36% even at 4,096.) So a bank conflict costs time only once nothing slower is hiding it, which is exactly the state an optimized kernel reaches.

## Step 5: registers, spills and the special paths

Registers are the fastest storage and the scarcest. When a kernel needs more than it can get, the compiler **spills** values to **local memory**, and the name misleads: per the Programming Guide, "the physical location of local memory is in the global memory space. The 'local' label refers to its logical scope." It's cached in L1 and L2, but it's DRAM underneath.

The compiler tells you when it happens. This kernel keeps 64 accumulators live across a loop; compile it normally, then cap registers at 32 per thread with `-maxrregcount=32`:

```text
$ nvcc -O3 -arch=sm_89 -Xptxas -v -o spill2 spill2.cu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 72 registers, used 0 barriers, 376 bytes cmem[0]
$ nvcc -O3 -arch=sm_89 -Xptxas -v -maxrregcount=32 -o spill2 spill2.cu
    232 bytes stack frame, 380 bytes spill stores, 380 bytes spill loads
ptxas info    : Used 32 registers, used 0 barriers, 232 bytes cumulative stack size, 376 bytes cmem[0]
```

The 72-register version ran in 3.75 ms; the spilling version took 77.5 ms, 21 times longer. It had twice the occupancy (32 registers per thread let 6 blocks fit per SM instead of 3), and it didn't matter. Nsight Compute counted 1.93 billion sectors of local loads; 77% hit in L1, and 76 GB passed through L2, for a kernel whose real input is 16 MB.

An array can land in local memory with no register pressure at all. Registers can't be indexed by a value computed at run time, so `a[perm[k]]` on a 32-float array put it on the stack (128 bytes stack frame, 0 spills) and made a small kernel 5.5 times slower (3.07 ms against 0.56 ms) than indexing with constants. `-Xptxas -v` is the first thing to check when a kernel is mysteriously slow.

Two other paths deserve a short note. **Constant memory** is 64 KB, read-only and cached; per the Best Practices Guide, "if all threads of a warp access the same location, then constant memory can be as fast as a register access," while different addresses within a warp "are serialized." It suits values every thread needs, like a scale factor. The **read-only path** (`const __restrict__` pointers or `__ldg`) loads through the texture cache, which on Ada is part of the same unified L1 SRAM.

## In the real world

### Why decode lives at the bottom of the ladder

Generating one token at batch size 1 reads every weight exactly once (Part 1 measured 3.09 GB of weights streamed at 251 GB/s on this laptop). Caches only help when data is used again soon, and within a decode step no weight is. Compare Llama 3.1 8B's 16.06 GB of BF16 weights ([Inference Part 3](/blog/gpus-for-llms/)) with the L2:

```text
32 MiB ÷ 16.06 GB = 33.5 MB ÷ 16.06 GB = 0.2% of the weights
one MLP weight matrix: 4,096 × 14,336 × 2 bytes = 117 MB, 3.5 times the L2
```

The dimensions are from the [Llama 3 paper](https://arxiv.org/abs/2407.21783). Not one weight matrix fits, so decode runs at the DRAM row of the sweep table, and the levers are fewer bytes ([quantization](/blog/quantization/)) or more users per byte read (batching).

### Why tiling is the core of fast kernels

The transpose used shared memory to fix an access pattern; matrix multiplication uses it to reuse data. In an N × N matmul each input element is needed N times. Load 32 × 32 tiles into shared memory and each element fetched from DRAM or L2 is used 32 times, a 32-fold cut in global traffic. Part 5 builds that kernel step by step and measures it against cuBLAS. [FlashAttention](/blog/flashattention/) is the same idea applied to attention: tiles of queries, keys and values sized to fit shared memory, so the n × n score matrix never leaves the chip. On this GPU, shared memory delivered about 24 times DRAM's bandwidth; the [FlashAttention paper](https://arxiv.org/abs/2205.14135) estimated about 10 times on the A100 ("around 19TB/s" against 1.5 to 2.0 TB/s of HBM).

### Data center GPUs

Data center GPUs have the same rungs, all larger. Per NVIDIA's [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), the H100 has 256 KB of combined L1 and shared memory per SM (up to 228 KB shared) and a 50 MB L2; the [H100 SXM](https://www.nvidia.com/en-us/data-center/h100/) has 80 GB of HBM3 at 3.35 TB/s, 13 times this laptop's bandwidth. Blackwell keeps 228 KB of shared memory per SM, and "the NVIDIA GB200 GPU increases the L2 cache capacity to 126 MB" ([Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)). A [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) lists 64 TB/s of HBM3e across 8 GPUs, 8 TB/s each. Even 126 MB is under 1% of an 8B model, so decode is DRAM-bound there too.

Hopper also added a hardware copy engine between the levels. The **Tensor Memory Accelerator** (TMA), per the Hopper guide, "allows applications to transfer 1D and up to 5D tensors between global memory and shared memory" asynchronously: one thread describes a tile, and the hardware does the address arithmetic and the copy while warps compute, saving the registers and instructions the transpose above spent on indexing. Ada has no TMA, so it isn't measured here; Part 6 covers it.

## Common misconceptions

- **"Bandwidth is bandwidth."** The same read loop ran at 6,000 GB/s, 1,440 GB/s or 253 GB/s depending only on whether its data fit in L1, L2 or neither.
- **"Bytes the kernel uses are bytes the GPU moves."** Reading every eighth float moved 67 MB from DRAM to deliver 8.4 MB, because the smallest unit is a 32-byte sector. Count sectors, not floats.
- **"Misaligned access is expensive."** On this GPU, offsetting a contiguous read by 1 to 32 floats changed bandwidth by under 2%; the extra sector per warp is caught by the caches. Strided access is what hurts.
- **"Bank conflicts always slow a kernel down."** A 32-way conflict cost nothing measurable when the transpose was waiting on DRAM, and 2.8 times when its data was in L2.
- **"Local memory is on-chip."** It's per-thread in scope only; physically it's DRAM, cached in L1 and L2. A forced spill made a kernel 21 times slower even though it doubled occupancy.

## Key takeaways

- The ladder on this laptop GPU: registers in about 4.6 cycles, L1 and shared memory in 30 to 40 cycles at about 6 TB/s, a 32 MB L2 in 264 cycles at about 1.4 TB/s, and DRAM in about 610 cycles at 253 GB/s.
- Compute DRAM bandwidth from the memory clock and bus width, then measure: pure reads hit 97% of 256 GB/s, and copies that mix reads and writes stop near 89%.
- Memory moves in 32-byte sectors. Keep the 32 threads of a warp on consecutive addresses; reading every eighth float cut useful bandwidth from 224 to 44 GB/s.
- Shared memory has 32 banks. Make the threads of a warp hit different banks, often with one column of padding, which made a cached transpose 2.8 times faster.
- Check `-Xptxas -v` for stack frames and spills: local memory lives in DRAM. And remember that decode reads weights once per token, so no cache saves it.

## What's next

You now have numbers for every level of the ladder, but a real kernel uses several at once and you need a way to tell which one is holding it back. [Part 4](/blog/gpu-profiling-roofline/) covers profiling and the roofline: Nsight Systems and Nsight Compute in depth, the PyTorch profiler, and plotting this series' kernels against the 4060's limits for compute and bandwidth, along with the clocks, power limits and throttling that move those limits on a laptop.

## Further reading

- [CUDA C++ Best Practices Guide (NVIDIA)](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): theoretical and effective bandwidth, coalescing with misaligned and strided examples, shared-memory bank conflicts, local memory, constant and texture memory.
- [CUDA Programming Guide: writing CUDA kernels (NVIDIA)](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html): the memory spaces, coalesced access, bank conflicts and padding, and register spilling.
- [NVIDIA Ada GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html): the 128 KB unified L1/shared memory, its shared-memory carveouts, and the compute capability 8.9 limits.
- [Nsight Compute Kernel Profiling Guide (NVIDIA)](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): sectors, cache lines, requests and wavefronts, and the hardware model behind every counter quoted here.
- [Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) and [Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) tuning guides (NVIDIA): shared memory, L2 and TMA on data center GPUs.
- [An Efficient Matrix Transpose in CUDA C/C++ (Mark Harris, NVIDIA, 2013)](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/): the tiled transpose and the padding trick, measured on older GPUs.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135): tiling into SRAM as an algorithm design principle.
- [How memory architectures affect system performance (Victor Echevarria, Rambus, 2005)](https://www.design-reuse.com/article/58224-how-memory-architectures-affect-system-performance/): bus turnaround and other reasons DRAM falls short of its peak.
