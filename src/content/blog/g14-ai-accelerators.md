---
title: "Beyond NVIDIA: AMD, Apple, TPUs and Other AI Accelerators"
description: "How AMD GPUs, Apple silicon, Google TPUs, AWS Trainium, Groq and Cerebras map onto the same ideas: threads, on-chip memory, matrix units, bandwidth and links."
slug: "ai-accelerators"
category: "gpu-kernels"
tags: ["gpu", "architecture", "serving", "latency"]
series: "llm-gpu"
seriesOrder: 14
pubDate: 2026-06-09
updatedDate: 2026-10-24
---

## TL;DR

Every AI chip has the parts this series measured on an NVIDIA GPU: lanes that run in lockstep, a small fast memory on the chip, a matrix unit, a big slow memory off it, a compiler, and links to other chips. AMD's Instinct GPUs map almost one to one onto NVIDIA's, and on this laptop one Triton matmul compiled to both NVIDIA and AMD matrix instructions. Google's TPUs and AWS Trainium use systolic arrays, which a NumPy simulation shows need hundreds of input rows per weight tile to stay busy. Groq and Cerebras keep weights in on-chip SRAM, which trades the bandwidth limit for a capacity one: about 614 Groq chips or four Cerebras wafers to hold a 70B model in 16 bits. None of this hardware is on the laptop; every non-NVIDIA number comes from vendor documents or papers, as of October 2026.

## Why it matters

Google trained Gemini 2.5 on chips with no warps, no CUDA cores and no CUDA. Anthropic says it uses "over one million Trainium2 chips to train and serve Claude," on top of Google TPUs, NVIDIA GPUs and a 2026 deal for AMD's. OpenAI has agreements for 6 gigawatts of AMD GPUs and 750 megawatts of Cerebras wafers.

Thirteen parts of this series ran on one NVIDIA GPU. Most of it carries over, because the roofline, the bandwidth ceiling on decode, tiling, low precision and the cost of collectives come from the workload. What changes is who schedules the work, how big each memory is, and the shape of the matrix unit.

None of these chips is in the laptop, so two experiments stand in: a cycle-by-cycle systolic array simulation on the CPU, and one Triton kernel cross-compiled for NVIDIA and AMD. Everything else comes from vendor pages, whitepapers and papers, linked where used.

## The intuition

Think of every accelerator as a factory. There's a warehouse (the big off-chip memory), a workbench next to the machines (on-chip SRAM), the machines themselves (matrix units), a foreman who decides what happens when, and roads to other factories (chip-to-chip links). Making one token of LLM output means hauling every model weight from the warehouse to the bench once, so decode speed depends on the road from the warehouse far more than on the machines.

The proportions differ. Apple's factory shares its warehouse with the CPU next door. Google's machine is a conveyor grid. Groq's factory has no warehouse, only a big bench, so one model needs hundreds of factories. And on a GPU the foreman schedules thousands of threads on the fly, while on a TPU or a Groq chip the compiler writes the whole schedule in advance.

The analogy breaks there: a compiler-scheduled chip has no foreman at run time at all. Every data movement is fixed before the program starts, which is why those chips need tensor shapes in advance.

## Step 1: one vocabulary, six vendors

Here are the series' ideas side by side. Each cell is the vendor's own term, from the documents cited in the steps below.

| Idea (series part) | NVIDIA (this series) | AMD Instinct | Apple M5 GPU | Google TPU | AWS Trainium | Groq / Cerebras |
|---|---|---|---|---|---|---|
| Basic compute unit (Part 2) | SM | Compute unit (CU); work group processor on MI455X | GPU core | TensorCore, 1 or 2 per chip | NeuronCore, 8 per chip | Groq: one chip of functional slices; Cerebras: 900,000 cores per wafer |
| Lanes in lockstep (Part 2) | Warp of 32 threads | Wavefront of 64 (CDNA 1 to 4); wave32 on CDNA 5 | SIMD group of 32 | None: vector and matrix ops scheduled by the compiler | None: engines run compiled instruction streams | None: compiler-scheduled |
| On-chip memory the program manages (Part 3) | Shared memory, up to 100 KB per SM on Ada | LDS, 64 KB per CU (CDNA 3), 160 KB (CDNA 4) | Threadgroup memory, 32 KB | Vector memory (VMEM) | SBUF, 28 MiB per NeuronCore-v3, plus 2 MiB PSUM | All of it: 230 MB per GroqChip, 44 GB per wafer |
| Matrix unit (Part 6) | Tensor cores, `mma.sync` | Matrix cores, MFMA instructions | Neural Accelerator in each core | MXU, a systolic array | Tensor Engine, a systolic array | Groq MXM; Cerebras per-core FP16 SIMD |
| Off-chip memory (Part 3) | GDDR6 here, HBM in data centers | HBM3E, HBM4 | Unified LPDDR shared with the CPU | HBM | HBM | None for inference |
| Kernel language (Parts 5, 7) | CUDA C++, Triton | HIP, Triton | Metal Shading Language, MLX custom kernels | Pallas, via JAX and XLA | NKI | Compiler only |
| Collectives and links (Part 13) | NCCL over NVLink | RCCL over Infinity Fabric | Not a scale-out part | XLA collectives over ICI (torus) | NeuronLink | RealScale links (Groq) |

GPUs expose threads and let hardware interleave them to hide memory latency. TPUs, Trainium and Groq expose a few large operations and let a compiler plan every byte's movement.

## Step 2: AMD, the closest cousin

### Compute units and wavefronts

AMD's [HIP hardware docs](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html) describe a compute unit (CU) much as [Part 2](/blog/cuda-programming-model/) described an SM, running warps "each containing 32 (RDNA) or 64 (CDNA) threads." CDNA is AMD's data center architecture, RDNA its graphics line. AMD calls a warp a **wavefront**, and on Instinct GPUs up to the MI355X it has 64 threads.

That number leaks into code. HIP's [language reference](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html) warns that "warpSize should not be assumed to be a specific value," since "AMD devices return 64 for gfx9 and 32 for gfx10 and above." A [Part 8](/blog/gpu-reductions/) reduction that hard-codes five shuffle steps (32 = 2⁵) drops half of a 64-wide wavefront's values. The newest chip changes the answer again: AMD's [MI400 page](https://www.amd.com/en/products/accelerators/instinct/mi400.html) lists the MI455X with "256 Work Group Processors with Wave32 execution."

The on-chip scratchpad is the **LDS** (local data share), AMD's name for shared memory. The [CDNA 4 whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) says CDNA 3's LDS held "a total of 64KB of data," while "the LDS in the AMD CDNA 4 architecture is 160KB," with read bandwidth doubled "to 256 bytes per clock."

### Matrix cores and MFMA

AMD's tensor cores are **matrix cores**, driven by **MFMA** (matrix fused multiply-add) instructions. Like `mma.sync` in [Part 6](/blog/tensor-cores/), they're cooperative: AMD's [matrix core guide](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html) says "all work-items (threads) within a wavefront collectively perform a single MFMA operation." The shapes are bigger. A BF16 MFMA of shape 32 × 32 × 8 does `2 × 32 × 32 × 8 = 16,384` FLOPs per instruction, four times Ada's 16 × 8 × 16 `mma.sync` (4,096 FLOPs). CDNA 4 adds 32 × 32 × 16 for BF16, again doubling the work per instruction.

### Same Triton source, three targets

[Part 7](/blog/triton-kernels/) promised a return to Triton's AMD backend. Triton's [README](https://github.com/triton-lang/triton) supports "NVIDIA GPUs (Compute Capability 8.0+), AMD GPUs (ROCm 6.2+)," and the AMD backend lowers Triton's IR through LLVM to AMDGCN, AMD's GPU assembly, on any machine. So one tile-matmul kernel compiles here for an NVIDIA target and two AMD targets, with no AMD GPU present:

```python
import re, triton, triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def mm_tile(a_ptr, b_ptr, c_ptr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)        # one program = one BM x BN tile of C
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    N = tl.num_programs(1) * BN
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)                  # the matrix unit, whichever chip
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))

src = ASTSource(fn=mm_tile,
                signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "K": "i32",
                           "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
                constexprs={"BM": 128, "BN": 128, "BK": 32})
for name, tgt in {"RTX 4060 Laptop (sm_89)": GPUTarget("cuda", 89, 32),
                  "MI300X (gfx942)": GPUTarget("hip", "gfx942", 64),
                  "MI355X (gfx950)": GPUTarget("hip", "gfx950", 64)}.items():
    k = triton.compile(src, target=tgt, options={"num_warps": 4})
    asm = k.asm["ptx"] if tgt.backend == "cuda" else k.asm["amdgcn"]
    ops = re.findall(r"\b(mma\.sync\.\S+|v_mfma_\w+)", asm)
    print(f"{name}: {len(ops)} matrix instructions, e.g. {sorted(set(ops))[0]}; "
          f"{4 * tgt.warp_size} threads, {k.metadata.shared // 1024} KB shared memory")
```

With Triton 3.6.0 it prints:

```text
RTX 4060 Laptop (sm_89): 64 matrix instructions, e.g. mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32; 128 threads, 32 KB shared memory
MI300X (gfx942): 16 matrix instructions, e.g. v_mfma_f32_32x32x8_bf16; 256 threads, 8 KB shared memory
MI355X (gfx950): 16 matrix instructions, e.g. v_mfma_f32_32x32x16_bf16; 256 threads, 16 KB shared memory
```

Each loop iteration covers a 128 × 128 × 32 block, `2 × 128 × 128 × 32 = 1,048,576` FLOPs. On Ada that's 64 `mma.sync` per warp, all inside the loop. On the MI300X it's 16 MFMAs per wavefront. The MI355X's MFMA does twice the work, so 8 per wavefront suffice; the compiler software-pipelined the loop, putting 8 of its 16 MFMAs in the loop body and 8 after it. `num_warps=4` means 128 threads on NVIDIA and 256 on AMD, and the AMD metadata says `.wavefront_size: 64`. Whether the AMD code is fast can't be measured here.

### HIP, HIPIFY and the libraries

For C++, AMD offers **HIP**, "a C++ runtime API and kernel programming language designed for AMD GPUs" with "an interface closely aligned with NVIDIA CUDA" ([docs](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html)). The [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html) tools translate CUDA source with renames such as "cudaMalloc => hipMalloc" ([porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)); kernels keep `__global__`, `__shared__` and `threadIdx`. AMD states the limits: HIP "is not intended to be a drop-in replacement for NVIDIA CUDA" ([What is HIP?](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html)).

**ROCm**, AMD's software platform, has a library for each NVIDIA one this series used. [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html) and [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html) play cuBLAS and cuBLASLt. [Composable Kernel](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/index.html), a C++ template library for "performance critical kernels," plays CUTLASS. [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/index.html) has "an API aligned to NCCL," per AMD's [collectives guide](https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/collective-operations.html). [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) supports "MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950)," and SGLang has an [AMD guide](https://docs.sglang.io/docs/hardware-platforms/amd_gpu).

### The chips and their links

The [MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) (CDNA 4, launched June 2025) has 256 CUs, 288 GB of HBM3E at 8 TB/s, and 2.5 PFLOPS of dense BF16 (5 PFLOPS "with Structured Sparsity"), doubling with each step down to FP8 and to MXFP4 ([Part 11](/blog/low-precision-kernels/)). The [MI455X](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html) (CDNA 5, launched July 23, 2026) lists 5 PFLOPS dense BF16, 40.3 PFLOPS MXFP4, and 432 GB of HBM4 at 23.3 TB/s. AMD's 72-GPU Helios rack design has "volume deployments expected in 2H 2026."

An 8-GPU MI355X server has no switch: each GPU has seven Infinity Fabric links, one per peer, at "76.8GB/s in each direction" (whitepaper). That's 537.6 GB/s each way in total, more than an H100's 450 GB/s into NVSwitch ([Part 13](/blog/nccl-collectives/)), but a copy between two particular GPUs gets one link unless the collective spreads traffic across all seven. The MI455X lists 3.6 TB/s of bidirectional scale-up bandwidth per GPU over UALoE (UALink over Ethernet).

## Step 3: Apple, one pool of memory

[Inference Part 13](/blog/local-llm-inference/) covered why Macs run big models: the GPU reads the CPU's unified memory, up to 512 GB on an M5 Ultra at 1.2 TB/s.

Apple's GPU is a SIMT machine too. Apple's [WWDC22 compute talk](https://developer.apple.com/videos/play/wwdc2022/10159/) says the SIMD width "on all Apple GPUs ... is equal to 32" and that "threads can share up to 32K of threadgroup memory": a warp and shared memory by other names. Kernels are written in the Metal Shading Language, and PyTorch's `mps` device uses Apple's [Metal Performance Shaders](https://docs.pytorch.org/docs/stable/notes/mps.html).

The M5 (October 2025) added the matrix unit. Apple's [announcement](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/) describes "a Neural Accelerator in each core," programmed "using Tensor APIs in Metal 4." They're separate from the Neural Engine, a dedicated AI block ("up to 38 trillion operations per second" on the [M4](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/)) that apps reach through Core ML.

**MLX**, Apple's array framework, is lazy ("Arrays are only materialized when needed") and its arrays "live in shared memory," usable on CPU or GPU "without performing data copies" ([docs](https://ml-explore.github.io/mlx/build/html/index.html)). Laziness lets MLX see a chain of operations before running it, as torch.compile does in [Part 9](/blog/kernel-fusion-cuda-graphs/).

Apple publishes no GPU TFLOPS, so its chips get no ridge point, but its own measurements show the roofline. Its [machine learning research team](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) compared six LLMs in MLX on an M5 and an M4. Time to first token on a 4,096-token prompt, which is compute-bound, improved 3.33 to 4.06 times. Generation, which is bandwidth-bound, improved only 19% to 27%, "thanks to its greater memory bandwidth (120GB/s for the M4, 153GB/s for the M5, which is 28% higher)." That's [Part 4](/blog/gpu-profiling-roofline/)'s roofline: compute helps the compute-bound phase, bandwidth the memory-bound one.

## Step 4: TPUs and Trainium, systolic arrays

### A grid that passes data along

Google's TPU (tensor processing unit) has no warps. Its matrix unit, the **MXU**, is a **systolic array**: a grid of multiply-accumulate cells (processing elements, PEs) where data pulses from neighbor to neighbor instead of each cell fetching its own operands. Google's [system architecture page](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) says "An MXU is composed of either 256 x 256 (TPU v6e and TPU7x) or 128 x 128 (TPU versions prior to v6e) multiply-accumulators in a systolic array." A chip has one or two TensorCores, each with MXUs, a vector unit and a scalar unit.

The original [TPU paper](https://arxiv.org/abs/1704.04760) (Jouppi et al., ISCA 2017) describes the dataflow: "data flows in from the left, and the weights are loaded from the top. A given 256-element multiply-accumulate operation moves through the matrix as a diagonal wavefront. The weights are preloaded." This is **weight-stationary**: each PE keeps one weight, activations stream past, and partial sums accumulate on their way out.

![A flowchart of a weight-stationary systolic array. A weight tile from HBM is loaded once and shifted into the array, with the next tile preloading. Activation rows from an on-chip buffer enter at the left edge, with row k starting k cycles late. Inside the array of K rows by N columns of processing elements, the PE at row k, column j keeps one weight and each cycle adds weight times x to the partial sum; x moves right to the PE at column j plus 1, and the partial sum moves down to the PE at row k plus 1. The bottom edge emits outputs, column j one cycle after column j minus 1, into accumulators and then the vector unit.](/diagrams/g14-systolic-array.svg "A weight-stationary systolic array: weights stay put, activations move right, partial sums move down.")

### Simulating one, cycle by cycle

This NumPy function models a K × N weight-stationary array. Every cycle, activations shift one PE right, partial sums shift one PE down, and every PE does one multiply-add. Row k receives feature k of input vector m at cycle `m + k`, the skew that makes the diagonal wavefront.

```python
import numpy as np

def systolic_matmul(X, W, show=False):
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    a = np.zeros((K, N))            # activation register in each PE
    p = np.zeros((K, N))            # partial-sum register in each PE
    tag = -np.ones((K, N), int)     # which input row m each PE is working on (-1 = idle)
    Y = np.zeros((M, N))
    t, done, busy = 0, 0, 0
    while done < M * N:
        # activations shift right; row k's left edge receives x[t-k, k] (skewed input)
        a[:, 1:], tag[:, 1:] = a[:, :-1].copy(), tag[:, :-1].copy()
        for k in range(K):
            m = t - k
            a[k, 0], tag[k, 0] = (X[m, k], m) if 0 <= m < M else (0.0, -1)
        # partial sums shift down and every PE does one multiply-add
        p = np.vstack([np.zeros((1, N)), p[:-1]]) + W * a
        busy += int((tag >= 0).sum())
        for j in range(N):          # bottom row emits finished outputs
            m = t - (K - 1) - j
            if 0 <= m < M:
                Y[m, j] = p[K - 1, j]; done += 1
        if show is not False:
            show.append(tag.copy())
        t += 1
    return Y, t, busy
```

The script's driver runs it on a 4 × 4 array with four input vectors (x0 to x3) of small random integers and prints which vector each PE is working on at each cycle:

```text
cycle 0         cycle 1         cycle 2         cycle 3         cycle 4
x0  .  .  .    x1 x0  .  .    x2 x1 x0  .    x3 x2 x1 x0     . x3 x2 x1
 .  .  .  .    x0  .  .  .    x1 x0  .  .    x2 x1 x0  .    x3 x2 x1 x0
 .  .  .  .     .  .  .  .    x0  .  .  .    x1 x0  .  .    x2 x1 x0  .
 .  .  .  .     .  .  .  .     .  .  .  .    x0  .  .  .    x1 x0  .  .
cycle 5         cycle 6         cycle 7         cycle 8         cycle 9
 .  . x3 x2     .  .  . x3     .  .  .  .     .  .  .  .     .  .  .  .
 . x3 x2 x1     .  . x3 x2     .  .  . x3     .  .  .  .     .  .  .  .
x3 x2 x1 x0     . x3 x2 x1     .  . x3 x2     .  .  . x3     .  .  .  .
x2 x1 x0  .    x3 x2 x1 x0     . x3 x2 x1     .  . x3 x2     .  .  . x3
matches X @ W: True
cycles = 10 (formula M + K + N - 2 = 10); PE utilization = 64/160 = 40%
```

Each input vector crosses the array as a diagonal band, and the result matches `X @ W` exactly. The 64 multiply-adds took 10 cycles on 16 PEs, 40% busy: `K + N − 2` cycles of filling and draining surround M cycles of streaming. The same formula (checked against the simulation for 32 vectors on a 16 × 16 array: 62 cycles) gives the busy fraction of one tile with its weights loaded:

| Input rows M (tokens in the batch) | 128 × 128 array | 256 × 256 array |
|---|---|---|
| 1 | 0.4% | 0.2% |
| 8 | 3.1% | 1.5% |
| 128 | 33.5% | 20.1% |
| 1,024 | 80.1% | 66.8% |
| 8,192 | 97.0% | 94.1% |

Real chips overlap one tile's drain with the next one's fill, so this is pessimistic, but weights set a harder limit. The TPU paper says the matrix unit "holds one 64 KiB tile of weights plus one for double-buffering (to hide the 256 cycles it takes to shift a tile in)." A tile is only worth loading if it then meets at least a few hundred input rows. In batch-1 decode, each weight tile meets one row, and the array waits on HBM, just as [Part 4](/blog/gpu-profiling-roofline/)'s GPU waited on DRAM at 1 FLOP per byte. A 256 × 256 array does four times the work of a 128 × 128 one per cycle but needs bigger matrices, which is why Google's [performance guide](https://docs.cloud.google.com/tpu/docs/performance-guide) asks for "dimensions of matrix multiplication ... larger than the MXU size."

### XLA and static shapes

A TPU program is compiled whole by **XLA** (Accelerated Linear Algebra), the compiler behind JAX and PyTorch/XLA, and it needs every shape. JAX's [docs](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) say it "re-runs the Python function when the type or shape of the argument changes," which means a new compilation. The performance guide adds that data should be tiled "into 128 x 8 chunks. When the data for a matrix computation does not fill an entire 128 x 8 chunk, the XLA compiler pads tensors." TPU serving engines therefore bucket batch sizes and sequence lengths, as vLLM does for CUDA graphs ([Part 1](/blog/gpu-from-python-to-silicon/)). Custom kernels are written in Pallas, which JAX's docs call "an extension to JAX for writing custom kernels for GPU and TPU."

### The chips and the torus

| Chip (Google docs) | Dense BF16 | FP8 | HBM | HBM bandwidth | ICI per chip, bidirectional | Pod |
|---|---|---|---|---|---|---|
| [v6e (Trillium)](https://docs.cloud.google.com/tpu/docs/v6e) | 918 TFLOPS | not faster than BF16 | 32 GB | 1,638 GB/s | 800 GB/s | 256 chips, 2D torus |
| [TPU7x (Ironwood)](https://docs.cloud.google.com/tpu/docs/tpu7x) | 2,307 TFLOPS | 4,614 TFLOPS | 192 GiB | 7.37 TB/s | 1,200 GB/s | 9,216 chips, 3D torus |

Chips connect through **ICI** (inter-chip interconnect) links straight to their neighbors, with no switch inside a slice. On Ironwood, "each of the six ICI links ... provides 200 GBps of bidirectional bandwidth," so each chip has neighbors in ±x, ±y and ±z of a **3D torus**, a grid whose edges wrap around. [Part 13](/blog/nccl-collectives/)'s ring all-reduce maps onto each dimension's ring. Larger slices join 4 × 4 × 4 cubes through "reconfigurable optical circuit switches," which the [TPU v4 paper](https://arxiv.org/abs/2304.01433) says are "<5% of system cost and <3% of system power." **SparseCores** on the same chips accelerate "sparse operations," mainly embedding lookups.

The next generation, [TPU 8t and 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive), announced in April 2026 with only FP4 rates (12.6 and 10.1 PFLOPS), is still "Coming soon" on Google's TPU page as of October 2026.

### AWS Trainium: the same idea with four engines

AWS's Trainium chips hold eight **NeuronCores** each. The [NeuronCore-v3 docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html) list "4 main engines: Tensor, Vector, Scalar, and GPSIMD, with on-chip software-managed SRAM," and the Tensor Engine "is built around a systolic array with 128 rows and 128 columns of processing elements, which streams input data from SBUF and writes output to PSUM" ([NKI architecture guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html)). SBUF (state buffer) is a 28 MiB scratchpad per NeuronCore-v3; PSUM, 2 MiB, accumulates results.

| Chip (Neuron docs) | Dense BF16 | FP8 | HBM | Bandwidth | Chip-to-chip |
|---|---|---|---|---|---|
| [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) | 667 TFLOPS | 1,299 TFLOPS | 96 GiB | 2.9 TB/s | NeuronLink-v3, 1.28 TB/s; 4 × 4 2D torus in a 16-chip instance |
| [Trainium3](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html) | 671 TFLOPS | 2,517 (MXFP8) | 144 GiB | 4.9 TB/s | NeuronLink-v4, 2.56 TB/s; all-to-all switched, up to 144 chips |

Trainium3's gains are in MXFP8 and bandwidth, not BF16. Kernels are written in **NKI** (Neuron Kernel Interface), which the [NKI docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html) describe as having "syntax and tile-level semantics that are similar to Triton and NumPy." PyTorch's [Neuron backend](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/compile/about-torch-compile.html) "lowers each graph through StableHLO" and compiles "on the first call with a given input signature": XLA's static-shape tax again.

## Step 5: Groq and Cerebras, SRAM instead of HBM

### Groq: the compiler is the scheduler

Groq's chip, which it calls a tensor streaming processor and later an LPU (language processing unit), was described in [Think Fast](https://web.archive.org/web/20241014113400id_/https://groq.com/wp-content/uploads/2024/02/2020-Isca.pdf) (Abts et al., ISCA 2020). It has "220 MiBytes of globally shared SRAM" and no HBM, and it guarantees "determinism by eliminating all reactive elements in the hardware (e.g. arbiters, and caches)." Every instruction's timing is fixed at compile time, so there's no latency for a scheduler to hide. Groq's [product brief](https://web.archive.org/web/20231221072647id_/https://groq.com/wp-content/uploads/2022/10/GroqChip%E2%84%A2-Processor-Product-Brief-v1.5.pdf) lists "Up to 750 TOPs, 188 TFLOPs (INT8, FP16 @900 MHz)," "230 MB SRAM per chip" and "Up to 80 TB/s on-die memory bandwidth." A [2022 paper](https://web.archive.org/web/20241203015545id_/https://groq.com/wp-content/uploads/2024/02/GroqISCAPaper2022_ASoftwareDefinedTensorStreamingMultiprocessorForLargeScaleMachineLearning-1.pdf) extends the determinism to a "software-scheduled" network of up to 10,440 chips.

With 188 TFLOPS on 80 TB/s, the ridge is about 2.4 FLOPs per byte, so decode at 1 FLOP per byte is nearly balanced instead of a hundred times short. The cost is capacity. Llama 3.1 70B in BF16 is 141.1 GB of weights:

```text
141.1 GB ÷ 0.230 GB per chip = 613.5   → 614 chips, for the weights alone
```

The KV cache needs SRAM too. Groq's [newsroom](https://groq.com/newsroom/groq-lpu-inference-engine-leads-in-first-independent-llm-benchmark) reported its internal benchmarks of Llama 2 70B "reaching 300 tokens per second consistently" without giving the chip count. In December 2025 Groq [licensed its technology](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale) to NVIDIA, whose [Groq 3 LPX](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) (March 2026) has "500 MB of compiler-managed SRAM" per chip and 128 GB per 256-chip rack. It runs the "latency-sensitive FFN and MoE execution" while GPUs keep attention.

### Cerebras: one chip per wafer

Cerebras doesn't cut the wafer into chips. Its [WSE-3](https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine) (wafer-scale engine, 2024) has "4 trillion transistors 900,000 AI cores 125 petaflops of peak AI performance 44GB on-chip SRAM," and the [CS-3 datasheet](https://cdn.sanity.io/files/e4qjo92p/production/864133620041e5af418074842e42a669c137fa23.pdf) adds "21 PB/s on‑chip SRAM bandwidth." The datasheet marks the 125 PFLOPS FP16 with "FLOPS shown as sparse," and Cerebras doesn't publish a dense figure, so it gets no ridge point in the table below.

The capacity arithmetic again: 141.1 GB ÷ 44 GB = 3.2, so four wafers hold a 70B model in 16 bits. Cerebras's [inference launch post](https://www.cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed) agrees: "70B models fit on as few as four systems," split "at layer boundaries," serving Llama 3.1 70B at "450 tokens per second" with "16-bit model weights." For training, Cerebras streams weights from an external store: "We store all of the model weights externally in a device called MemoryX," its whitepaper says, streaming them in "one layer at a time."

### Others, briefly

Intel's [Gaudi 3](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf) pairs 1,678 TFLOPS of BF16 with "24x 200 GbE RoCE for scaleup and scale-out," Ethernet built into the chip instead of NVLink; its inference successor, the [Intel GPU D-series](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-accelerates-open-infrastructure-agentic-ai-era.html), is sampling with up to 480 GB of LPDDR5X. Tenstorrent's [Blackhole p150](https://docs.tenstorrent.com/aibs/blackhole/index.html) has 120 Tensix cores, 32 GB of GDDR6 at 512 GB/s and Ethernet ports, programmed through the open-source TT-Metalium stack.

## Step 6: the same roofline for every chip

This table applies Part 4's arithmetic to every chip. Specs are per chip, dense BF16, from the vendor pages linked above; the RTX 4060 row is measured in [Part 4](/blog/gpu-profiling-roofline/), and the H100 and B200 rows come from [Inference Part 3](/blog/gpus-for-llms/). The last three columns are computed here: the ridge is dense BF16 FLOPs ÷ bandwidth; "chips to hold 70B" is 141.1 GB ÷ capacity, rounded up, ignoring the KV cache; and the decode ceiling assumes those chips split every layer evenly and read their shares in parallel, so it's `chips × bandwidth ÷ 141.1 GB`, an upper bound that ignores communication.

| Chip | Dense BF16 | Memory | Bandwidth | Ridge (FLOPs per byte, computed) | Chips to hold 70B BF16 | Batch-1 decode ceiling, 70B BF16 |
|---|---|---|---|---|---|---|
| RTX 4060 Laptop (measured) | 25.3 TFLOPS | 8 GB GDDR6 | 248.7 GB/s | 102 | 18 | n/a |
| NVIDIA H100 SXM | 989 TFLOPS | 80 GB | 3.35 TB/s | 295 | 2 | 48 tokens/s |
| NVIDIA B200 | 2,250 TFLOPS | 180 GB | 8 TB/s | 281 | 1 | 57 tokens/s |
| AMD MI355X | 2,500 TFLOPS | 288 GB | 8 TB/s | 313 | 1 | 57 tokens/s |
| AMD MI455X | 5,000 TFLOPS | 432 GB | 23.3 TB/s | 215 | 1 | 165 tokens/s |
| Google TPU v6e | 918 TFLOPS | 32 GB | 1.64 TB/s | 560 | 5 | 58 tokens/s |
| Google TPU7x (Ironwood) | 2,307 TFLOPS | 192 GiB (206 GB) | 7.37 TB/s | 313 | 1 | 52 tokens/s |
| AWS Trainium2 | 667 TFLOPS | 96 GiB (103 GB) | 2.9 TB/s | 230 | 2 | 41 tokens/s |
| AWS Trainium3 | 671 TFLOPS | 144 GiB (155 GB) | 4.9 TB/s | 137 | 1 | 35 tokens/s |
| Intel Gaudi 3 | 1,678 TFLOPS | 128 GB | 3.7 TB/s | 454 | 2 | 52 tokens/s |
| Apple M5 Ultra | Not published | 512 GB unified | 1.2 TB/s | n/a | 1 | 8.5 tokens/s |
| GroqChip (2020) | 188 TFLOPS (FP16) | 230 MB SRAM | 80 TB/s | 2.4 | 614 | Not bandwidth-bound |
| Cerebras WSE-3 | Not published (125 PFLOPS sparse) | 44 GB SRAM | 21 PB/s | n/a | 4 | Not bandwidth-bound |

Every HBM chip's ridge sits between about 140 and 560 FLOPs per byte, so batch-1 decode at 1 FLOP per byte is memory-bound on all of them; batching is how you use any of them. Their 70B decode ceilings cluster between 35 and 58 tokens per second, with the MI455X's HBM4 the outlier at 165. For the SRAM machines the formula gives hundreds of thousands of tokens per second, which is meaningless: there, per-chip compute and the latency of passing activations through hundreds of chips set the pace. Groq's 300 and Cerebras's 450 tokens per second for a 70B model are several times the single-user ceilings of the HBM chips, bought with hundreds of chips or four wafers.

## In the real world

Who runs what, from the companies' own statements:

- Google trained Gemini 1.0 "using TPUv5e and TPUv4" ([Gemini report](https://arxiv.org/abs/2312.11805)) and says Gemini 2.5 is "the first to be trained on TPUv5p architecture," across "multiple 8960-chip pods" ([Gemini 2.5 report](https://arxiv.org/abs/2507.06261)).
- Anthropic named "three chip platforms" (TPUs, Trainium and NVIDIA GPUs) when it announced access to "up to one million TPUs" in [October 2025](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services). In [April 2026](https://www.anthropic.com/news/anthropic-amazon-compute) it said it uses "over one million Trainium2 chips to train and serve Claude." In July 2026 AMD and Anthropic [announced](https://ir.amd.com/news-events/press-releases/detail/1292) a plan to "deploy up to 2 gigawatts of AMD Instinct MI450 Series GPUs," building on "Anthropic's existing use of AMD Instinct MI355X GPUs."
- OpenAI agreed in [October 2025](https://ir.amd.com/news-events/press-releases/detail/1260/amd-and-openai-announce-strategic-partnership-to-deploy-6-gigawatts-of-amd-gpus) "to deploy 6 gigawatts of AMD GPUs," starting with MI450s in the second half of 2026, and in [January 2026](https://www.cerebras.ai/blog/openai-partners-with-cerebras-to-bring-high-speed-inference-to-the-mainstream) to deploy "750 megawatts of Cerebras wafer-scale systems."
- Microsoft [optimized GPT-4 Turbo](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/introducing-the-new-azure-ai-infrastructure-vm-series-nd-mi300x-v5/4145152) for its MI300X VMs in 2024, and AMD [reported](https://ir.amd.com/news-events/press-releases/detail/1218/amd-unveils-leadership-ai-solutions-at-advancing-ai-2024) Meta using "MI300X serving all live traffic on Llama 405B."

Portable software above the chips makes this possible. Triton compiles for NVIDIA and AMD; TorchInductor "leverages OpenAI Triton" on NVIDIA, AMD and Intel GPUs ([Part 7](/blog/triton-kernels/)); Trainium and Apple have PyTorch backends; JAX and XLA target TPUs and GPUs, and AWS's compiler takes StableHLO, XLA's input format. NVIDIA's remaining lead sits below that layer, in libraries and tuned kernels, which is why AMD ships a counterpart to each and every vendor documents its own kernel language.

## Common misconceptions

- **"TFLOPS decides how fast an LLM runs."** For decode, bandwidth does. Apple's M5 brought Neural Accelerators that made prefill up to 4 times faster than on the M4, but generation only 19% to 27% faster, matching the 28% bandwidth gain.
- **"Only NVIDIA hardware can run frontier models."** Gemini trains on TPUs, Claude trains and serves on over a million Trainium2 chips, and Meta served Llama 405B on MI300X, per AMD.
- **"HIPIFY turns CUDA code into AMD code."** It renames the API. Code that assumes 32-thread warps, as Part 8's shuffle reductions do, still breaks on 64-wide wavefronts.
- **"A bigger matrix unit is always faster."** A 256 × 256 systolic array does four times the work per cycle of a 128 × 128 one, but a batch of 128 input rows kept it 20% busy against 34%. Small batches waste big arrays.
- **"All-SRAM chips make memory problems disappear."** They swap bandwidth for capacity: 614 GroqChips or four Cerebras wafers just to hold a 70B model in 16 bits.

## Key takeaways

- Every accelerator has the same parts: lanes in lockstep or a systolic grid, fast on-chip memory, a matrix unit, big off-chip memory and links. The series' arithmetic applies to all.
- AMD's Instinct GPUs map almost one to one onto NVIDIA's: compute units, 64-wide wavefronts, LDS, MFMA, HIP, rocBLAS and RCCL. The same Triton source compiled here for both, with different matrix instructions.
- TPUs and Trainium use weight-stationary systolic arrays, scheduled by a compiler that needs static shapes. A tile of weights only pays off when it meets hundreds of input rows, so decode is bandwidth-bound there too.
- Groq and Cerebras keep weights in SRAM, which lifts the bandwidth ceiling but needs hundreds of chips or several wafers for a 70B model.
- On every HBM chip, as of October 2026, the ridge sits at 140 to 560 FLOPs per byte, and batch-1 decode at 1. Batching is the answer everywhere.

## What's next

This part looked outward. The final part, [Part 15](/blog/token-kernel-trace/), looks all the way in: one decode step of Qwen2.5-1.5B on this laptop, traced kernel by kernel, with every kernel's time and its place on the roofline, tying together everything the series has measured.

## Further reading

- [In-Datacenter Performance Analysis of a Tensor Processing Unit (Jouppi et al., ISCA 2017)](https://arxiv.org/abs/1704.04760): the first TPU, its 256 × 256 systolic array and its weight-stationary dataflow.
- [TPU v4: An Optically Reconfigurable Supercomputer (Jouppi et al., ISCA 2023)](https://arxiv.org/abs/2304.01433): the 3D torus, optical circuit switches and SparseCores.
- [Cloud TPU system architecture (Google)](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) and the [TPU7x page](https://docs.cloud.google.com/tpu/docs/tpu7x): MXUs, TensorCores, ICI and per-chip specs.
- [AMD CDNA 4 architecture whitepaper (AMD)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf): compute units, LDS, matrix cores and Infinity Fabric on the MI350 series.
- [HIP documentation (AMD)](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html) and [AMD matrix cores on CDNA 3 and 4 (AMD)](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html): the CUDA-like programming model and the MFMA instructions.
- [AWS Neuron architecture docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) and the [NKI docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html): NeuronCores, the systolic Tensor Engine and the kernel language.
- [Think Fast: A Tensor Streaming Processor (Abts et al., ISCA 2020)](https://web.archive.org/web/20241014113400id_/https://groq.com/wp-content/uploads/2024/02/2020-Isca.pdf): Groq's deterministic, compiler-scheduled chip.
- [Inside NVIDIA Groq 3 LPX (NVIDIA)](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/): the LPU design inside NVIDIA's platform.
- [Introducing Cerebras Inference (Cerebras)](https://www.cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed): how a 70B model is split across four wafers.
- [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU (Apple)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5): a vendor's own measurement of compute-bound prefill against bandwidth-bound decode.
- [MLX documentation (Apple)](https://ml-explore.github.io/mlx/build/html/index.html): lazy evaluation and unified memory in practice.
- [Triton (GitHub)](https://github.com/triton-lang/triton): the compiler whose NVIDIA and AMD backends Step 2 used.
