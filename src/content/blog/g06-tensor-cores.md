---
title: "Tensor Cores: Matrix Instructions, Swizzling and FP8, Measured"
description: "What one tensor-core instruction computes, how to feed it from shared memory without bank conflicts, and what FP16, BF16 and FP8 cost in speed and accuracy."
slug: "tensor-cores"
category: "gpu-kernels"
tags: ["gpu", "cuda", "latency", "quantization"]
series: "llm-gpu"
seriesOrder: 6
pubDate: 2026-10-24
---

## TL;DR

A tensor core multiplies two small matrices in one instruction: on this laptop's Ada GPU, one `mma.sync` computes a 16 × 8 block of results from a 16 × 16 and a 16 × 8 input, 4,096 floating-point operations, with operands spread across a warp's 32 threads in a documented layout. This article checks that layout on the GPU, then rebuilds [Part 5](/blog/cuda-matmul-optimization/)'s matmul around the instruction. Fed from shared memory through `ldmatrix`, with an XOR swizzle that removes a 4-way bank conflict, a BF16 kernel reached 19.7 TFLOPS, 96% of the peak at the clock it ran at, against cuBLAS's 20.8 and Part 5's 5.5 TFLOPS on CUDA cores. FP16 accumulation doubles the peak rate and FP8 doubles it again; error measurements against FP64 show what each costs, including an FP8 accumulator that keeps only about 14 significant bits. Hopper's `wgmma` and TMA and Blackwell's `tcgen05` come from NVIDIA's documentation at the end.

## Why it matters

Same laptop, same 40-watt power limit, same 4,096 × 4,096 matrix multiply. Part 5's best kernel needed 25.1 milliseconds on the CUDA cores. A kernel of about the same length built on tensor-core instructions needed 7.0, and in FP8 it took 2.3.

One tensor-core instruction does the work of 64 warp-wide fused multiply-adds, so the kernel issues 24 times fewer instructions per FLOP. LLMs run almost all their arithmetic this way, in BF16 or FP8. Writing the kernel by hand shows what cuBLAS and FlashAttention spend their effort on: getting operands into the right registers fast enough.

## The intuition

Go back to Part 5's hall of clerks. Each clerk takes one number from row A and one from column B, multiplies, adds to a running total, and repeats. A tensor core is a stamping press: two small sheets go in, one 16 × 16 and one 16 × 8, and a 16 × 8 sheet of finished dot products comes out in one stroke, 2,048 multiply-adds at once.

The press only accepts sheets held in a particular grip, each of 32 workers holding specific cells. And it is so fast that the workers spend most of their time fetching and arranging sheets, the subject of most of this article.

The analogy breaks on precision. A clerk adds every product exactly in FP32. The press decides internally how many bits its sums keep, and for FP8 on this GPU the answer turns out to be fewer than the instruction's name suggests.

## Step 1: what one MMA instruction computes

A **tensor core** is a matrix unit inside each SM (streaming multiprocessor); Ada has four per SM ([Inference Part 3](/blog/gpus-for-llms/)). Programs reach them through **MMA** (matrix multiply-accumulate) instructions. In **PTX**, NVIDIA's portable assembly language that `nvcc` emits, the instruction is `mma.sync`. The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) defines it as "a MxNxK matrix multiply and accumulate operation, D = A*B+C, where the A matrix is MxK, the B matrix is KxN, and the C and D matrices are MxN."

For 16-bit inputs the main shape is `m16n8k16`: A is 16 × 16, B is 16 × 8, C and D are 16 × 8. That's `16 × 8 × 16 = 2,048` multiply-adds, 4,096 FLOPs, issued by one warp. No single thread holds a matrix. Each holds a **fragment**: 8 elements of A in four 32-bit registers (two BF16 values per register), 4 of B in two registers, and 4 FP32 values of C. The PTX ISA gives the mapping in terms of `groupID = lane >> 2` and `threadID_in_group = lane % 4`. For A, elements a0 and a1 sit at row `groupID`, columns `2 × threadID_in_group` and the next one; a2 and a3 are 8 rows lower; a4 to a7 repeat that 8 columns to the right.

Here's one instruction with fragments loaded by those formulas, in inline PTX (from `mma_demo.cu`, which checks all 128 outputs against the CPU):

```cpp
__global__ void one_mma(const __nv_bfloat16* A, const __nv_bfloat16* B, float* D) {
  int lane = threadIdx.x, g = lane >> 2, t = lane % 4;          // groupID, threadID_in_group
  __nv_bfloat162 a[4], b[2];                                    // A 16x16 row-major, B 16x8 row-major
  a[0] = {A[g * 16 + 2 * t],           A[g * 16 + 2 * t + 1]};
  a[1] = {A[(g + 8) * 16 + 2 * t],     A[(g + 8) * 16 + 2 * t + 1]};
  a[2] = {A[g * 16 + 2 * t + 8],       A[g * 16 + 2 * t + 9]};
  a[3] = {A[(g + 8) * 16 + 2 * t + 8], A[(g + 8) * 16 + 2 * t + 9]};
  b[0] = {B[(2 * t) * 8 + g],     B[(2 * t + 1) * 8 + g]};      // rows (k) 2t, 2t+1, column g
  b[1] = {B[(2 * t + 8) * 8 + g], B[(2 * t + 9) * 8 + g]};
  float d[4] = {0, 0, 0, 0};
  const unsigned* ra = reinterpret_cast<const unsigned*>(a);
  const unsigned* rb = reinterpret_cast<const unsigned*>(b);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]));
  // d[0], d[1]: row g, columns 2t and 2t+1;  d[2], d[3]: row g+8
  D[g * 8 + 2 * t] = d[0];       D[g * 8 + 2 * t + 1] = d[1];
  D[(g + 8) * 8 + 2 * t] = d[2]; D[(g + 8) * 8 + 2 * t + 1] = d[3];
}
```

```text
mma.m16n8k16: 0 of 128 outputs differ from the CPU
D[0][0] = -28, held by lane 0;  D[9][5] = -3, held by lane 6
```

The inputs are small integers, exact in BF16, so the check is exact. Element (9, 5) of D belongs to lane 6: row 9 is `groupID + 8` with groupID 1, column 5 belongs to threadID_in_group 2, and `4 × 1 + 2 = 6`.

### The WMMA API

The higher-level route is **WMMA** (warp matrix multiply-accumulate), functions in `nvcuda::wmma` that, per the [CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#warp-matrix-functions), "leverage Tensor Cores to accelerate matrix problems of the form D=A*B+C":

```cpp
__global__ void one_wmma(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);
  wmma::load_matrix_sync(a, A, 16);          // leading dimension: 16 elements per row
  wmma::load_matrix_sync(b, B, 16);
  wmma::mma_sync(c, a, b, c);                // C = A x B + C, the whole warp at once
  wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
  if (threadIdx.x == 0) printf("WMMA fragment sizes per thread: a %d, b %d, c %d elements\n", a.num_elements, b.num_elements, c.num_elements);
}
```

```text
WMMA fragment sizes per thread: a 8, b 8, c 8 elements
wmma 16x16x16: 0 of 256 outputs differ from the CPU
```

It's simpler, but the guide warns that "the mapping of matrix elements into fragment internal storage is unspecified and subject to change in future architectures," so you can only load and store whole tiles. WMMA also has no FP8 type; FP8 needs `mma.sync`.

## Step 2: a tensor-core matmul, built on Part 5

The test is Part 5's: C = A × B for 4,096 × 4,096 matrices, 137.4 billion FLOPs. One change: B is stored N × K, the way PyTorch stores `nn.Linear` weights, so both operands have K contiguous, the layout the instruction's `.row.col` suffix asks for. Inputs are uniform in [-1, 1), rounded to BF16; outputs are FP32.

The machine is Part 1's NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, 24 SMs) under WSL2, driver 592.82, CUDA 12.8 with cuBLAS 12.8.4, PyTorch 2.11.0, `nvcc -O3 -arch=sm_89`. Each kernel ran in its own process: a 0.6-second warm-up, then back-to-back launches for 3 seconds, each bracketed by CUDA events, keeping the median of the last three quarters. The tables give medians of three rounds, which agreed within 2% (9% for the padded kernels). `nvidia-smi` logged clocks and power every 100 ms; the power limit sat at 40 W, below Parts 4 and 5's 45 to 50 W, and the GPU drew all of it.

Peak rates come from the [Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)'s RTX 4090 column divided by its 128 SMs and 2.52 GHz boost clock, as in Part 1. Per SM per clock: 256 FP32 FLOPs on CUDA cores, 512 for BF16 or FP16 with FP32 accumulation, 1,024 for FP16 with FP16 accumulation or FP8 with FP32 accumulation, and 2,048 for FP8 with FP16 accumulation. "Share of peak" below uses `24 SMs × that rate × the clock the kernel ran at`.

### Four kernels

In the textbook kernel, each warp owns a 16 × 16 tile of C and loads WMMA fragments straight from global memory. It ran at 5.29 TFLOPS, level with Part 5's best CUDA-core kernel but only 25% of the tensor-core peak: every warp fetches every fragment it needs from global memory, Part 5's naive kernel again.

The fix is Part 5's structure: a 128 × 128 block tile of C, 8 warps each owning a 64 × 32 piece, and 32-wide slices of K staged in shared memory. Two things are new. Copies use `cp.async`, an Ampere instruction that moves 16 bytes from global to shared memory without passing through registers, so the next slices load while the current one computes; with three slices in flight, a **3-stage pipeline**, the kernel needs 48 KB of shared memory. And warps read operands with `ldmatrix`, which the PTX ISA describes as a way to "collectively load one or more matrices from shared memory for mma instruction": with `.x4`, each thread supplies one row address and the warp receives four 8 × 8 matrices in fragment order, a whole A fragment.

The inner loop for one slice, simplified from `tc.cu`:

```cpp
for (int ks = 0; ks < 2; ks++) {
  unsigned a[4][4], b[4][2];
  for (int i = 0; i < 4; i++)                       // A: four 16 x 16 fragments, one ldmatrix.x4 each
    ldsm_x4(a[i], smem_u32(sA + soff<L>(wm * 64 + i * 16 + (lane & 15), ks * 2 + (lane >> 4))));
  for (int j = 0; j < 2; j++) {                     // B: each ldmatrix.x4 gives two 16 x 8 fragments
    unsigned r[4];
    ldsm_x4(r, smem_u32(sB + soff<L>(wn * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1))));
    b[2 * j][0] = r[0]; b[2 * j][1] = r[1]; b[2 * j + 1][0] = r[2]; b[2 * j + 1][1] = r[3];
  }
  for (int i = 0; i < 4; i++)                       // 16 MMAs: a 64 x 32 piece of C per warp
    for (int j = 0; j < 4; j++) mma<BF16>(acc[i][j], a[i], b[j]);
}
```

`soff<L>(row, chunk)` gives the byte offset of a 16-byte chunk in the shared-memory tile; Step 3 is about its three versions.

| Kernel | Time | TFLOPS | SM clock | Share of peak at that clock |
|---|---|---|---|---|
| Part 5, kernel 8: FP32 on CUDA cores | 25.14 ms | 5.47 | 1,275 MHz | 70% (of the FP32 peak) |
| cuBLAS `cublasSgemm`, FP32 | 26.92 ms | 5.11 | 1,350 MHz | 62% (of the FP32 peak) |
| WMMA, fragments from global memory | 25.99 ms | 5.29 | 1,740 MHz | 25% |
| WMMA, shared-memory tiles, 3 stages | 7.82 ms | 17.57 | 1,582 MHz | 90% |
| `ldmatrix` + `mma.sync`, same tiles | 7.42 ms | 18.54 | 1,695 MHz | 89% |
| `ldmatrix` + `mma.sync`, swizzled tiles | 6.98 ms | 19.69 | 1,672 MHz | 96% |
| cuBLAS `cublasGemmEx`, BF16 in, FP32 out | 6.62 ms | 20.77 | 1,740 MHz | 97% |

Part 5's FP32 kernels repeat their 70% and 62% at this lower power limit. The swizzled kernel is 3.6 times faster than kernel 8: a factor of 2 from the peak rate (512 against 256 FLOPs per SM per clock), 1.37 from efficiency (96% against 70%) and 1.31 from clock. Under the same 40 W, the tensor-core kernel held 1,672 MHz while the FP32 kernel fell to 1,275, so it also did 3.6 times more arithmetic per joule.

All the BF16 kernels matched cuBLAS bit for bit, with a relative error (root-mean-square error over root-mean-square value) of 4.7 × 10⁻⁶ against an FP64 product of the same inputs.

## Step 3: bank conflicts and the XOR swizzle

The swizzled kernel is 6% faster than the plain one because of [Part 3](/blog/gpu-memory-hierarchy/)'s banks. Shared memory has 32 banks of 4 bytes, so one wavefront (one pass through the banks) delivers 128 bytes. Grouped in fours, each 16-byte chunk lives in one of 8 bank groups, `(byte offset ÷ 16) mod 8`.

An `ldmatrix` 8 × 8 matrix is 8 rows of 16 bytes, so it takes one wavefront only if the 8 rows hit 8 different groups. In the plain layout a tile row is 32 BF16 values, 64 bytes, so row r starts at byte `64r`. The groups that rows 0 to 7 hit, from `banks.py`:

```text
none chunk 0: rows 0-7 -> bank groups [0, 4, 0, 4, 0, 4, 0, 4]  passes needed: 4
pad  chunk 0: rows 0-7 -> bank groups [0, 5, 2, 7, 4, 1, 6, 3]  passes needed: 1
swz  chunk 0: rows 0-7 -> bank groups [0, 4, 1, 5, 2, 6, 3, 7]  passes needed: 1
```

Two 64-byte rows fill one 128-byte bank row, so rows 0, 2, 4 and 6 all land in group 0: a 4-way conflict. **Padding**, Part 3's fix, adds 16 unused bytes per row, so row r starts at `80r` and the rows spread over all 8 groups. **Swizzling** keeps rows packed and permutes the 16-byte chunks within each row: chunk c of row r goes to position `c XOR ((r >> 1) & 3)`.

```cpp
template <int L> __device__ __forceinline__ int soff(int row, int chunk) {   // byte offset of a 16-byte chunk
  if (L == PAD) return row * 80 + chunk * 16;                                  // 80-byte rows
  if (L == SWZ) return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);           // XOR the chunk index with row bits
  return row * 64 + chunk * 16;                                                // 64-byte rows
}
```

Row 2's chunk 0 moves to position 1, row 4's to position 2, and so on. The same XOR applies when `cp.async` writes the tile and when `ldmatrix` reads it, and no bytes are wasted. Nsight Compute (clocks locked at their 1,320 MHz base, so only ratios and counts are quoted) confirmed the arithmetic:

| Layout | Wavefronts per `ldmatrix.x4` | Bank conflicts on loads | Shared memory, 3 stages | Blocks per SM | Share of peak, 3 stages | Share of peak, 2 stages |
|---|---|---|---|---|---|---|
| Plain, 64-byte rows | 16.0 | 151 million | 48 KB | 2 | 89% | 84% |
| Padded, 80-byte rows | 4.0 | 0 | 60 KB | 1 | 78% | 92% |
| Swizzled, 64-byte rows | 4.0 | 0.04 million | 48 KB | 2 | 96% | 90% |

Four 8 × 8 matrices need at least 4 wavefronts; the plain layout took 16, the predicted 4-way conflict. With two stages, both fixes beat the plain layout, padding slightly ahead. With three, padding's extra 25% of shared memory made each block 60 KB, only one fit in the SM's 100 KB, occupancy halved, and the padded kernel was the slowest. Only the swizzled layout ran a 3-stage pipeline at two blocks per SM.

Production libraries swizzle for this reason. CUTLASS's `Swizzle<BBits, MBase, SShift>` in [swizzle.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp) XORs `BBits` address bits, taken `SShift` positions higher, into the bits above the lowest `MBase`; in BF16 element offsets, the layout above is `Swizzle<2,3,3>`, which `banks.py` checked against `soff` for all 4,096 elements of a tile. Hopper's TMA copy engine swizzles in hardware: the PTX ISA's "64 byte swizzle mode" table, in 16-byte cells, reads `0 1 2 3 4 5 6 7`, then `1 0 3 2 5 4 7 6`, then `2 3 0 1 6 7 4 5`, this same XOR.

## Step 4: reading PTX and SASS

`nvcc -ptx` shows the PTX, and `cuobjdump -sass` shows **SASS**, the machine code `ptxas` produces from it. Each PTX operation in the inner loop becomes one SASS instruction:

```text
PTX:   ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r93,%r94,%r95,%r96}, [%r97];
SASS:  LDSM.16.M88.4 R28, [R94] ;
PTX:   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f616,%f615,%f614,%f613}, {%r93,%r94,%r95,%r96}, {%r113,%r114}, {%f616,%f615,%f614,%f613};
SASS:  HMMA.16816.F32.BF16 R48, R36.reuse, R28, R48 ;
PTX:   cp.async.cg.shared.global [%r22], [%rd10], 16;
SASS:  LDGSTS.E.BYPASS.128 [R113], [R4.64] ;
```

FP16 accumulation compiles to `HMMA.16816.F16`, FP8 to `QMMA.16832.F32.E4M3.E4M3`, and a WMMA 16 × 16 × 16 to two `HMMA.16816`, so WMMA and `mma.sync` share hardware. cuBLAS's BF16 kernel uses the smaller m16n8k8 shape and issued 67.1 million tensor instructions to this kernel's 33.6 million.

Per 32-wide slice, each warp executes 96 instructions in the main loop: 32 `HMMA`, 12 `LDSM`, 4 `LDGSTS`, 3 for synchronization, and 45 integer and branch instructions, mostly addresses. (`cuobjdump` lists 99; three are `@!PT LDS RZ, [RZ]`, predicated never to run.)

Compare Part 5's kernel 8: per 8-wide slice, 577 instructions, 512 of them `FFMA`.

```text
kernel 8 (FP32):        512 FFMA × 64 FLOPs per warp        = 32,768 FLOPs in 577 instructions =    57 FLOPs per instruction
swizzled (BF16 MMA):    32 HMMA × 4,096 FLOPs per warp      = 131,072 FLOPs in 96 instructions  = 1,365 FLOPs per instruction
```

That's 24 times more work per instruction. Part 5's FP32 kernel was limited by issue slots; here the schedulers issued on only 9.5% of cycles. The limit is the tensor pipe, busy 48.6% of the time against cuBLAS's 49.6%. That's Part 4's 49% headline: Nsight Compute measures against the FP16-accumulate rate of 24,576 tensor operations per cycle, and BF16 with FP32 accumulation runs at half of it, so 48.6% is 97% of what this format can reach.

## Step 5: FP16, BF16, TF32 and FP8 on Ada

[Math Part 14](/blog/floating-point-math/) covered the formats: BF16 has 8 exponent and 7 mantissa bits, FP16 5 and 10, FP8 E4M3 4 and 3. Tensor cores multiply in the input format and add into an **accumulator**, the running sum, in FP32 or FP16. Ada GeForce runs FP16 accumulation at twice the FP32-accumulate rate, and FP8 at twice the FP16 rate. Here's the same matmul through PyTorch:

```python
torch.backends.cuda.matmul.allow_tf32 = True                   # FP32 tensors, TF32 multiplies
c = torch.mm(a16, b16.T, out_dtype=torch.float32)              # BF16 or FP16 inputs, FP32 output
torch.backends.cuda.matmul.allow_fp16_accumulation = True      # FP16 inputs, FP16 sums
c = torch._scaled_mm(a8, b8.T, scale_a=one, scale_b=one,       # FP8 E4M3 inputs (private API)
                     out_dtype=torch.float32, use_fast_accum=False)
```

Errors are relative RMS against an FP64 product: "total" from the original FP32 inputs, "from the sums" from the already-rounded inputs, isolating what the multiply-accumulate adds.

| Format (PyTorch, cuBLAS) | TFLOPS | SM clock | Share of peak at that clock | Error, total | Error from the sums |
|---|---|---|---|---|---|
| FP32 on CUDA cores | 4.82 | 1,320 MHz | 60% | 1.2 × 10⁻⁶ | 1.2 × 10⁻⁶ |
| TF32 (FP32 inputs, rounded inside the tensor core) | 9.99 | 1,725 MHz | 95% | 2.6 × 10⁻⁴ | n/a |
| BF16, FP32 accumulate | 20.52 | 1,755 MHz | 95% | 2.1 × 10⁻³ | 4.6 × 10⁻⁶ |
| FP16, FP32 accumulate | 20.02 | 1,710 MHz | 95% | 2.6 × 10⁻⁴ | 6.6 × 10⁻⁶ |
| FP16, FP16 accumulate (FP16 output) | 31.72 | 1,440 MHz | 90% | 3.3 × 10⁻³ | 3.3 × 10⁻³ |
| FP8 E4M3, `_scaled_mm` | 39.19 | 1,770 MHz | 91% | 3.3 × 10⁻² | 1.2 × 10⁻⁴ |
| FP8 E4M3, `_scaled_mm`, fast accumulate | 40.70 | 1,815 MHz | 91% | 3.3 × 10⁻² | 1.2 × 10⁻³ |

With FP32 accumulation, nearly all the error is input rounding: 0.2% for BF16's 7 mantissa bits, 0.026% for FP16's 10, 3.3% for FP8's 3. FP16 accumulation was 1.58 times faster, but its sums alone added 13 times FP16's total error with FP32 sums: a 16-bit total of 4,096 products loses low bits at every add, Math Part 14's "256 + 1 = 256."

The hand-written kernels, identical except for the instruction, show a problem specific to FP8:

| Hand-written kernel | TFLOPS | SM clock | Share of peak at that clock | Error from the sums |
|---|---|---|---|---|
| BF16, FP32 accumulate | 19.69 | 1,672 MHz | 96% | 4.7 × 10⁻⁶ |
| FP16, FP32 accumulate | 19.30 | 1,635 MHz | 96% | 7.2 × 10⁻⁶ |
| FP16, FP16 accumulate | 25.94 | 1,290 MHz | 82% | 3.3 × 10⁻³ |
| FP16, FP16 accumulate, 64 × 64 warp tiles | 27.46 | 1,268 MHz | 89% | 3.3 × 10⁻³ |
| FP8 E4M3, "FP32" accumulate | 41.40 | 1,860 MHz | 91% | 2.9 × 10⁻³ |
| FP8 E4M3, partial sums promoted to FP32 | 40.30 | 1,860 MHz | 88% | 7.0 × 10⁻⁵ |
| FP8 E4M3, FP16 accumulate, 64 × 64 warp tiles | 60.40 | 1,590 MHz | 77% | 3.8 × 10⁻³ |
| cuBLAS, FP16 accumulate | 32.37 | 1,418 MHz | 93% | 3.3 × 10⁻³ |

The FP8 kernel's sums were 620 times less accurate than the BF16 kernel's, with the same FP32 accumulator. A one-instruction probe (`accbits.cu`) shows why: it sets C to 2ᵉ and adds a product of 1 × 1, and FP32's 24 significant bits hold 2ᵉ + 1 exactly up to e = 23:

```text
BF16 mma, FP32 accumulator    : ... 21:1 22:1 23:1 24:0 25:0
  largest C = 2^e to which the product 1.0 was added exactly: e = 23
FP8 E4M3 mma, FP32 accumulator: ... 12:1 13:1 14:0 15:0 16:0
  largest C = 2^e to which the product 1.0 was added exactly: e = 13
```

The FP8 instruction keeps about 14 significant bits, though the PTX ISA promises accumulation "with at least single precision." The [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) found the same on Hopper: "the accumulation precision of FP8 GEMM on NVIDIA H800 GPUs is limited to retaining around 14 bits." Their fix, "promotion to CUDA Cores for higher precision," moves partial results into FP32 registers at intervals. The promoted kernel does that every 64 values of K: two FP8 MMAs into a zeroed partial sum, then ordinary FP32 adds. Its error from the sums fell 42 times, for 3% of the speed. cuBLAS promotes by default; its [docs](https://docs.nvidia.com/cuda/cublas/) say fast accumulation, PyTorch's `use_fast_accum`, is faster "at the cost of lower accuracy because intermediate results will not periodically be promoted to a higher precision," on "Ada, Hopper." The PyTorch table shows that tenfold difference.

The FP16-accumulate kernel reached 82%, against cuBLAS's 93%: at twice the math rate, `ldmatrix` traffic starts to limit. Making each warp's tile 64 × 64 cut `ldmatrix` per MMA from 0.375 to 0.25 and reached 89%, the same lever as Part 5's register blocking. The FP8 FP16-accumulate kernel stayed at 77%; closing that gap would take more tuning than this article does.

## Step 6: Hopper and Blackwell, from the documentation

On Ada, every warp loads its own fragments and every thread helps issue copies. Since Hopper, NVIDIA's data center GPUs move both jobs to dedicated hardware. Nothing in this section was measured; this laptop can't run these instructions.

![Three paths from global memory to tensor-core results. Ada, sm_89, measured on this laptop: cp.async copies 16 bytes per thread into shared memory with swizzled tiles, ldmatrix loads fragments into registers, and mma.sync runs per warp with sums in registers. Hopper, sm_90a, from the docs: a TMA copy issued by one thread moves a whole tile into shared memory, and wgmma runs across four warps, reading B from shared memory. Blackwell, sm_100a, from the docs: a TMA copy fills shared memory, and tcgen05.mma, issued by one thread, keeps its sums in tensor memory.](/diagrams/g06-tensor-core-paths.svg "How operands reach the tensor cores on Ada, Hopper and Blackwell.")

**TMA**, the Tensor Memory Accelerator that [Part 3](/blog/gpu-memory-hierarchy/) promised, replaces `cp.async`. The [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) says it "allows applications to transfer 1D and up to 5D tensors between global memory and shared memory," and that "a single thread can issue large data movement instructions to the TMA unit." The tensor's shape and strides live in a **tensor map**, "a 128-byte opaque object" per the PTX ISA, and the copy instruction is `cp.async.bulk.tensor`. TMA writes tiles in the swizzle modes from Step 3, so the 45 integer instructions per slice in this article's kernel leave the SM's schedulers.

**wgmma**, the warpgroup MMA, replaces `mma.sync`. A **warpgroup**, per the PTX ISA, "is a set of four contiguous warps." Its 16-bit shapes run up to m64n256k16, a 64 × 256 tile per instruction. "The input matrix A of the warpgroup wide MMA operations can be either in registers or in the shared memory. The input matrix B of the warpgroup wide MMA operations must be in the shared memory," described by a matrix descriptor rather than loaded into registers with `ldmatrix`. It's asynchronous: a kernel issues `wgmma.mma_async`, commits a group, and waits with `wgmma.wait_group` later. It requires `sm_90a`, a target whose features, per the PTX ISA, "are supported on the specified architecture only," so the code can't run on later generations.

**tcgen05**, Blackwell's fifth-generation tensor core instructions, go further. They add **tensor memory**, which the PTX ISA describes as "dedicated on-chip memory that is specialized for use by TensorCore operations": on sm_100a, "512 columns and 128 rows per CTA, with each cell being 32-bits in size," where a CTA (cooperative thread array) is a thread block: 256 KB per block. The accumulator D lives there instead of in registers, A can come from tensor memory or shared memory, and B from shared memory. One thread issues `tcgen05.mma` for the whole block, with shapes up to 128 × 256 × 16, or 256 × 256 × 16 when two blocks pair up.

CUTLASS 3 and its layout library **CuTe** hide these differences. A CuTe layout is "a pair of `IntTuple`s: the `Shape` and the `Stride`," written `Shape:Stride`, per its [layout docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md). This article's plain shared-memory tile is `(128,32):(32,1)`, 128 rows of 32 elements, 32 apart; composed with `Swizzle<2,3,3>` it becomes the swizzled tile. Step 1's thread-to-fragment formulas become layouts too, so one GEMM can target `mma.sync`, `wgmma` or `tcgen05.mma` by swapping the MMA "atom."

## In the real world

Kernel names show these instructions at work. `torch.matmul` in BF16 launched `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn`: the m16n8k8 shape (`1688`), 128 × 128 block tiles and the "TN" layout chosen in Step 2. FP8 `_scaled_mm` launched `sm89_xmma_gemm_e4m3f32_e4m3f32_f32_tn_n_tilesize128x64x64_stage4_warpsize2x2x1_tensor16x8x32`, with 4 pipeline stages and the m16n8k32 FP8 instruction.

FlashAttention-3 is built on Hopper's asynchronous path. Its [paper](https://arxiv.org/abs/2407.08608) describes exploiting the "asynchrony of the Tensor Cores and TMA" to "overlap overall computation and data movement via warp-specialization," with FP16 "reaching up to 740 TFLOPs/s (75% utilization)" on an H100, and FP8 "close to 1.2 PFLOPs/s." [Inference Part 7](/blog/flashattention/) explains the algorithm; Part 10 of this series writes an attention kernel.

DeepSeek-V3 trained in FP8 with Step 5's promotion trick; without it, its report measured "a maximum relative error of nearly 2%" for a K = 4,096 GEMM ([Inference Part 14](/blog/deepseek-inference-deepinfer/) covers their serving stack). That's why LLMs train and serve in BF16 with FP32 accumulation, and move to FP8 only with scaling factors and periodic promotion: BF16's range matches FP32's ([Math Part 14](/blog/floating-point-math/)), its tensor-core rate on data center GPUs is 15 times FP32's ([Inference Part 3](/blog/gpus-for-llms/)), and FP8 doubles it again. [Inference Part 8](/blog/quantization/) covers FP8 and lower-bit weights, and Part 11 measures quantized kernels.

## Common misconceptions

- **"Tensor cores are 15 times faster than CUDA cores."** That's the H100's ratio. On this GeForce chip, BF16 with FP32 accumulation peaks at 2 times FP32 per clock. The measured 3.6 times also came from higher efficiency (96% against 70%) and a higher clock under the same power cap.
- **"An FP32 accumulator means FP32 accuracy."** For BF16 and FP16, yes: the probe kept all 24 bits. For FP8 on this GPU, the instruction kept about 14, and the error from the sums was 620 times worse until partial sums were promoted to real FP32 every 64 values of K.
- **"Padding is the way to avoid bank conflicts."** Padding and swizzling both turned 16 wavefronts per `ldmatrix` into 4. But padding cost 25% more shared memory, which at 3 stages meant one block per SM instead of two, and made that kernel the slowest. Swizzling costs no memory.
- **"WMMA and mma.sync are different hardware."** Both compile to `HMMA.16816`. WMMA hides the fragment layout and has no FP8; `mma.sync` exposes both.
- **"FP16 accumulation is a free 2x."** It was 1.58 times faster through PyTorch and cuBLAS, and its error from the sums alone, 3.3 × 10⁻³, was larger than all of BF16's input rounding.

## Key takeaways

- One `mma.sync.m16n8k16` is 4,096 FLOPs from one warp, with each thread holding a documented fragment of A, B and C. Feeding those fragments is the whole job of a tensor-core kernel.
- Part 5's tiles still apply: shared-memory tiles, a multi-stage `cp.async` pipeline and `ldmatrix` took a BF16 kernel from 25% of peak to 96%, 19.7 TFLOPS against cuBLAS's 20.8 and 3.6 times Part 5's best.
- `ldmatrix` reads 8 rows of 16 bytes; on 64-byte rows that was a 4-way bank conflict. An XOR swizzle fixed it without the extra shared memory padding costs, and TMA does the same in hardware.
- On Ada, FP16 accumulation runs at twice the FP32-accumulate rate and FP8 at twice again, measured at 32.4 and 60.4 TFLOPS. The FP8 accumulator kept only about 14 bits, and promotion to FP32 every 64 values of K fixed it for 3% of the speed.
- Hopper and Blackwell move the work out of warps: TMA copies whole tiles, `wgmma` reads B from shared memory across four warps, and `tcgen05.mma` is issued by one thread into tensor memory.

## What's next

Everything here was placed by hand: fragment layouts, swizzles, pipeline stages. [Part 7, Triton](/blog/triton-kernels/), writes the same matmul in Python at the level of blocks, leaves those details to a compiler, and measures the result against PyTorch.

## Further reading

- [PTX ISA (NVIDIA)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): `mma.sync` and its fragment layouts, `ldmatrix`, `cp.async`, tensor maps and swizzle modes, `wgmma` and the tcgen05 family with tensor memory.
- [CUDA C++ Programming Guide, Warp Matrix Functions (NVIDIA, CUDA 12.8)](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#warp-matrix-functions): the WMMA API, supported types and shapes, and the unspecified fragment mapping.
- [NVIDIA Ada GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf): tensor-core rates for FP8, FP16, BF16 and TF32 with FP16 and FP32 accumulation.
- [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html): the Tensor Memory Accelerator and thread block clusters.
- [cuBLAS documentation (NVIDIA)](https://docs.nvidia.com/cuda/cublas/): `cublasGemmEx`, FP8 types and the fast-accumulation flag.
- [CuTe quickstart](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md), [layouts](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md) and [swizzle.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp) (NVIDIA CUTLASS): layouts as shape and stride, and swizzles as functions on addresses.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): FP8 training, the 14-bit accumulation finding and promotion to CUDA cores.
- [FlashAttention-3 (Shah et al., 2024)](https://arxiv.org/abs/2407.08608): attention on Hopper with `wgmma`, TMA and warp specialization.
- [How to Optimize a CUDA Matmul Kernel (Simon Boehm, 2022)](https://siboehm.com/articles/22/CUDA-MMM): the CUDA-core progression that Part 5 and this article build on.
