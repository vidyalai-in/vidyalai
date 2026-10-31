---
title: "Low-Precision Kernels: INT4, INT8 and FP8 Matmuls, Measured"
description: "How 4-bit weight kernels dequantize in registers, when INT4, INT8 and FP8 make inference faster on a laptop GPU, and how FP8, MX and NVFP4 scales work."
slug: "low-precision-kernels"
category: "gpu-kernels"
tags: ["gpu", "cuda", "latency", "quantization", "triton"]
series: "llm-gpu"
seriesOrder: 11
pubDate: 2026-06-03
---

## TL;DR

A 4-bit weight is half a byte, so decode, which reads every weight once per token, can run almost four times faster than in BF16. No instruction multiplies 4-bit integers by BF16 activations, so the kernel unpacks the weights with shifts and masks in registers, applies a per-group scale and feeds BF16 tensor cores. This article writes that kernel in Triton and times it against cuBLAS, PyTorch's 4-bit kernel, INT8 and FP8 on a laptop RTX 4060, from 1 to 4,096 rows of activations. A 4-bit Qwen2.5-1.5B decoded a token in 5.9 ms against 15.1 ms in BF16, but the 4-bit matmul stopped helping at about 128 rows, where an INT8 matmul with 8-bit activations ran 2 to 3.3 times faster than BF16. Then come FP8 scaling factors and the MX and NVFP4 formats.

## Why it matters

A 4-bit copy of Qwen2.5-1.5B produced a token in 5.9 milliseconds on this laptop. The BF16 original took 15.1 ms, on the same GPU with the same surrounding kernels. Give a 4-bit matrix multiply of Llama-3-8B's size a batch of 1,024 prompt tokens, and it takes 17% longer than BF16; PyTorch's built-in 4-bit kernel takes 4.5 times as long.

[Inference Part 8](/blog/quantization/) explained the formats and why they help decode, and [Math Part 14](/blog/floating-point-math/) worked out their error. This part goes into the kernels. A quantized format only saves time if a kernel can turn its bytes into tensor-core math without spending the time it saved, and that depends on how many rows of activations share each weight.

## The intuition

Think of a furniture shop that delivers by truck. Flat-packed, four wardrobes fit where one assembled wardrobe did. When the truck is the bottleneck, one customer and one wardrobe at a time, flat-packing makes deliveries four times faster, even though someone must assemble each wardrobe at the door. For an order of 500 wardrobes for one hotel, the bottleneck is the assembly crew, and flat-pack only adds work. Decode at batch size 1 is all truck; a big batch is all assembly.

The analogy breaks in one useful place: some crews can use the flat-pack parts directly. This GPU's tensor cores multiply 8-bit integers and floats natively, at two to four times their BF16 rate, so 8-bit weights with 8-bit activations speed up the assembly too.

## Step 1: why fewer bytes speed up decode

[Part 4](/blog/gpu-profiling-roofline/) placed decode on the roofline: a matrix-vector product (GEMV) does 2 floating-point operations per 2-byte BF16 weight, an arithmetic intensity of about 1 FLOP per byte, against this GPU's ridge point of about 102 (25.3 BF16 TFLOPS over 248.7 GB/s, both measured in Part 4). Its time is bytes divided by bandwidth.

Quantization changes only the bytes. For the up projection of a Llama-3-8B MLP (multi-layer perceptron) block, 4,096 inputs by 14,336 outputs:

| Weight format | Bits per weight | Weight bytes | Time at 248.7 GB/s | Ideal speedup |
|---|---|---|---|---|
| BF16 | 16 | 117.4 MB | 472 µs | 1.0× |
| INT8 or FP8, one scale per row | 8.008 | 58.8 MB | 236 µs | 2.0× |
| INT4, group 128, BF16 scale and min | 4.25 | 31.2 MB | 126 µs | 3.76× |

In the INT4 row, each group of 128 weights also stores a BF16 scale and a BF16 minimum, so 4.25 bits per weight. Marlin, the best-known 4-bit kernel, quotes the same kind of ceiling: "the maximum possible 3.87× speedup" for its 4.125-bit format.

Now add rows. A batch of M tokens multiplies M rows of activations by the same weights, so the FLOPs grow with M while the weight bytes don't. The intensity of a 4-bit matmul is about 3.76 × M FLOPs per byte, so it crosses the ridge at M ≈ 28; BF16 crosses at about M = 100. Between the two, the 4-bit advantage shrinks from 3.76× to nothing. Beyond that both are compute-bound on the same BF16 tensor cores, and weight-only quantization can only lose, because it still has to unpack every weight.

## Step 2: a W4A16 kernel in Triton

W4A16 means 4-bit weights and 16-bit activations. Offline, each group of 128 consecutive weights along the input dimension K gets `scale = (max − min) / 15`, and each weight becomes an integer `q = round((w − min) / scale)` from 0 to 15. Eight of these 4-bit "nibbles" are packed into one 32-bit integer along K, so an N × K weight becomes a K/8 × N packed matrix plus K/128 × N scales and minimums.

The kernel is [Part 7](/blog/triton-kernels/)'s Triton matmul with one change: before `tl.dot`, the B tile is built from packed integers instead of loaded as BF16.

```python
@triton.jit
def w4a16_kernel(x_ptr, qw_ptr, s_ptr, m_ptr, y_ptr, M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr):
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rp = tl.arange(0, BLOCK_K // 8)                                    # packed rows: 8 values of k per int32
    x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]
    q_ptrs = qw_ptr + rp[:, None] * N + rn[None, :]
    shifts = tl.arange(0, 8) * 4
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=rm[:, None] < M, other=0.0)           # BF16 activations
        p = tl.load(q_ptrs)                                            # [BLOCK_K / 8, BLOCK_N] int32
        q = (p[:, None, :] >> shifts[None, :, None]) & 0xF             # unpack: 8 nibbles per word
        q = tl.reshape(q, (BLOCK_K, BLOCK_N))                          # row 8r + j = nibble j of word r
        s = tl.load(s_ptr + (k0 // GROUP) * N + rn).to(tl.float32)     # this group's scale and min
        m = tl.load(m_ptr + (k0 // GROUP) * N + rn).to(tl.float32)
        w = (q.to(tl.float32) * s[None, :] + m[None, :]).to(tl.bfloat16)   # dequantize in registers
        acc = tl.dot(x, w, acc)                                        # BF16 tensor cores, FP32 sums
        x_ptrs += BLOCK_K
        q_ptrs += (BLOCK_K // 8) * N
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=rm[:, None] < M)
```

With 128 × 64 weight tiles, each step loads 4 KB of packed words where a BF16 kernel would load 16 KB. The shift-and-mask line turns each word into 8 integers, the multiply-add turns them into BF16, and `tl.dot` never sees the integers. BF16 weights are never written back to memory.

![A left-to-right flowchart. In GPU memory (DRAM) sit three inputs: BF16 activations of size M by K, packed INT4 weights with 8 per int32, and a scale and min per 128 weights. Inside one SM, for each K slice, the packed weights and scales are loaded into registers, unpacked with shift and mask, and dequantized as q times scale plus min. The resulting BF16 tile and the activations go into tl.dot on the BF16 tensor cores, which adds into an FP32 accumulator. After the last K slice, the accumulator is written out as the BF16 output of size M by N.](/diagrams/g11-dequant-datapath.svg "The dequantize-in-kernel data path: 4-bit weights cross the memory bus, BF16 weights exist only in registers.")

Against an FP64 product with the same 4-bit weights, the output's relative error was 2.3 × 10⁻³, the size of BF16 rounding. Against the original weights it was 10%, from rounding random weights to 16 levels; Math Part 14 measured 10.7% on Qwen2.5-1.5B's real weights in a near-identical format. At M = 16 rows, on the Llama-3-8B shape, the kernel ran in 221 to 250 µs over three runs (the memory clock switched between 7,000 and 8,000 MHz), against 584 µs for BF16 in cuBLAS.

### How the measurements were taken

Same machine as [Part 1](/blog/gpu-from-python-to-silicon/): NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, compute capability 8.9) under WSL2, driver 592.82, PyTorch 2.11.0 with CUDA 12.8, Triton 3.6.0. Each time is the median from `triton.testing.do_bench` (25 ms warm-up, 400 ms of calls, L2 cleared before each call), then the median of three rounds. Clearing L2 matters: the 31.2 MB 4-bit matrix fits in the 32 MB L2, and unflushed, one kernel timed at 112 µs, faster than DRAM allows. During the sweep the SM clock ran at 1,725 to 2,175 MHz (median 2,085), the memory clock at 8,000 MHz for three quarters of samples and 7,000 for most of the rest, and power reached 50.5 W against a 50 W limit. Part 4's read kernel, rerun today, reached 244.3 GB/s; the roof below uses Part 4's 248.7 GB/s.

Triton's autotuner picked tiles per M; for M = 1 a separate GEMV kernel sums `s × Σ x·q + min × Σ x` per group in FP32 without tensor cores.

### Where the time goes at M = 1

At one row, the 4-bit kernels moved 169 GB/s, 68% of the roof, while cuBLAS's BF16 GEMV moved 228 GB/s, 92%. Nsight Compute (which locks clocks at base, so its times are only for comparison) shows why:

| Kernel at M = 1 | DRAM throughput | Instructions executed | ALU pipe busy (integer and logic) | Issue slots used |
|---|---|---|---|---|
| BF16 GEMV (cuBLAS) | 96.6% | 15.1 million | 15.9% | 25.6% |
| W4A16 GEMV (this article's Triton) | 73.2% | 15.3 million | 64.5% | 69.4% |
| W4A16, PyTorch tinygemm | 83.8% | 11.1 million | 33.5% | 61.5% |

The 4-bit kernel executes as many instructions as the BF16 one while moving a quarter of the bytes. Unpacking and scaling kept the ALU pipe 64.5% busy, and with only 33% of warp slots occupied, the SMs couldn't keep enough memory requests in flight. At 4 bits, even a GEMV is partly an instruction problem, which is why production kernels put so much work into dequantization. Marlin turns nibbles into FP16 with "a single `lop3` instruction": OR the 4 bits into the mantissa of a number with a fixed exponent, then subtract a constant. That trick in this Triton kernel (`q | 0x4300`, read as BF16, is exactly 128 + q) made no kernel faster, so the conversion itself wasn't the limit here.

## Step 3: measured from 1 row to 4,096

The same Llama-3-8B up projection, every path PyTorch 2.11 offers on this GPU. Time in µs, with speedup over BF16 in parentheses:

| M (rows) | BF16, cuBLAS | W4A16, Triton | W4A16, tinygemm | W8A16, Triton | FP8, per-tensor | FP8, per-row | INT8 W8A8 |
|---|---|---|---|---|---|---|---|
| 1 | 514 | 184 (2.79×) | 183 (2.80×) | 343 (1.50×) | 342 (1.50×) | 344 (1.49×) | needs M > 16 |
| 4 | 581 | 218 (2.66×) | 184 (3.15×) | 336 (1.73×) | 342 (1.70×) | 344 (1.69×) | needs M > 16 |
| 16 | 584 | 222 (2.63×) | 310 (1.88×) | 337 (1.73×) | 343 (1.70×) | 344 (1.70×) | needs M > 16 |
| 32 | 591 | 353 (1.67×) | 651 (0.91×) | 345 (1.71×) | 344 (1.72×) | 347 (1.70×) | 336 (1.76×) |
| 64 | 646 | 414 (1.56×) | 1,307 (0.49×) | 383 (1.69×) | 347 (1.86×) | 356 (1.81×) | 338 (1.91×) |
| 128 | 739 | 711 (1.04×) | 2,642 (0.28×) | 798 (0.93×) | 366 (2.02×) | 866 (0.85×) | 368 (2.01×) |
| 256 | 1,421 | 1,450 (0.98×) | 5,347 (0.27×) | 1,543 (0.92×) | 666 (2.14×) | 1,693 (0.84×) | 432 (3.29×) |
| 1,024 | 4,718 | 5,498 (0.86×) | 21,180 (0.22×) | 5,878 (0.80×) | 2,541 (1.86×) | 6,145 (0.77×) | 1,417 (3.33×) |
| 4,096 | 18,582 | 23,028 (0.81×) | 82,701 (0.22×) | 23,938 (0.78×) | 10,354 (1.79×) | 24,015 (0.77×) | 6,114 (3.04×) |

FP8 and INT8 columns time only the matmul, on already-quantized activations. tinygemm is PyTorch's built-in INT4 kernel, `torch.ops.aten._weight_int4pack_mm`, which the commit that added it credits to "the tinnygemm project which developed by Jeff Johnson." It got the same integers, scales and minimums as the Triton kernel, and both matched BF16 on the dequantized weights to BF16 rounding.

Weight-only 4-bit pays off only for small M. The Triton kernel held 2.6× up to M = 16, fell to 1.6× at 64, broke even at 128, and from 1,024 up took 17% to 24% longer than BF16. That's Step 1's roofline: past M ≈ 100 both run on the same tensor cores (20.9 TFLOPS against cuBLAS's 25.9 at M = 4,096), and dequantization is pure overhead, repeated by every program that needs the same weight tile.

tinygemm is built for decode. It was fastest from M = 2 to 8, 3.15× at M = 4, then its time doubled with every doubling of M, flat at 5.7 TFLOPS, until it took 4.5 times as long as BF16. torchao, PyTorch's quantization library, can pack 4-bit weight-only models in tinygemm's format, and prefill run through this kernel pays that price.

8-bit formats reached 1.5× to 1.9× at small M, short of 2×: at M = 1, cuBLAS ran FP8 through a GEMM kernel with 64 × 64 tiles at 172 GB/s, against 228 GB/s for its dedicated BF16 GEMV.

The scale granularity changed which kernel ran. Per-tensor FP8 ran cuBLAS's `sm89_xmma_gemm_e4m3...` kernel and reached 46.5 TFLOPS at M = 4,096, 1.8 times BF16. Per-row FP8 ran a CUTLASS kernel inside PyTorch, `cutlass::Kernel2<at::cuda::detail::enable_2x_kernel_for_sm89<...>>`, peaked at 20 TFLOPS and lost to BF16 from M = 128. The [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/) list per-tensor ("tensorwide") FP8 scaling for compute capability 8.9 and up, but per-row ("outer vector") only for 9.0, Hopper.

### INT8 on a GeForce chip

`torch._int_mm` (cuBLAS, 8-bit integer inputs, 32-bit integer output) reached 84.9 TOPS (trillions of operations per second) at M = 1,024, 3.3 times BF16. The [Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)'s RTX 4090 rates, scaled per SM as in Part 1, explain it: per SM per clock, 512 BF16 operations with FP32 accumulation, 1,024 FP8 with FP32 accumulation, and 2,048 INT8. On this GeForce chip INT8 is the fastest tensor-core path, twice FP8, and `_int_mm` reached 89% to 92% of its peak for M ≥ 1,024. PyTorch requires `self.size(0) > 16`, so batch-1 decode can't use it.

### The cost of quantizing activations

W8A8 needs 8-bit activations, and those have to be computed for every batch. Done as separate PyTorch operations (absolute value, maximum per row, divide, round, convert, then a rescale of the output), it costs a lot:

| M | FP8 per-row, matmul only | Same, with activation quantization in eager PyTorch | INT8, matmul only | Same, with quantization and rescale |
|---|---|---|---|---|
| 1 | 344 µs | 500 µs | n/a | n/a |
| 256 | 1,693 µs | 1,895 µs | 432 µs | 675 µs |
| 4,096 | 24,015 µs | 29,678 µs | 6,114 µs | 15,525 µs |

At M = 4,096 INT8's speedup fell from 3.0× to 1.2×: the eager version writes a 235 MB INT32 result, then reads it back to rescale. Production kernels fuse the quantization into the previous operation and apply both scales in the matmul's epilogue, the kernel's last step before it writes, [Part 9](/blog/kernel-fusion-cuda-graphs/)'s fusion argument again.

## Step 4: one decode step in 4 bits

Back to [Part 9](/blog/kernel-fusion-cuda-graphs/)'s workload: Qwen2.5-1.5B-Instruct, batch size 1, a 43-token prompt, greedy decoding with a static KV cache. All 196 linear layers in the 28 blocks were quantized to the INT4 format above (round to nearest, no calibration) and run through tinygemm; optionally the output layer too, which Qwen ties to its 233-million-parameter embedding table. All versions ran in one process, as CUDA graphs and, separately, with `torch.compile(mode="reduce-overhead")`; each time is the median of 40 synchronized steps over three or four interleaved rounds. The SM clock ran at 1,845 to 2,175 MHz, memory at 8,000 MHz, and power reached 53.9 W under a limit drifting from 46 to 50 W.

| Version | Weight bytes per step | Bandwidth floor | CUDA graph | Graph + `torch.compile` | Share of floor (compiled) |
|---|---|---|---|---|---|
| BF16 | 3.087 GB | 12.4 ms | 16.1 ms | 15.1 ms | 82% |
| INT4 blocks, BF16 output layer | 1.163 GB | 4.7 ms | 8.15 ms | 7.0 ms | 67% |
| INT4 everywhere | 0.820 GB | 3.3 ms | 6.74 ms | 5.86 ms | 56% |

The fully 4-bit model ran 2.58 times faster than BF16, short of the 3.76× the bytes allow. In the graph version, tinygemm took 3.68 ms for all 197 matmuls, but attention took 0.85 ms and 1,233 small elementwise kernels 2.1 ms. Those didn't shrink with the weights, so once the matmuls got 3.5 times faster they were 44% of the step; compiling fused most of them and saved 0.9 ms. (This session's compiled BF16 step was about 1 ms slower than Part 9's 13.8 ms; every comparison here comes from one process.)

Quality moved too. On 2,048 tokens of *The Wonderful Wizard of Oz*, perplexity (the exponential of the average negative log-probability of each true next token) went from 7.17 in BF16 to 8.05 with 4-bit blocks and 8.52 with a 4-bit output layer as well. The greedy answer matched BF16 up to "designed to handle graphics" with 4-bit blocks, and differed from the first token with a 4-bit output layer. Round-to-nearest is the crudest method; GPTQ and AWQ ([Inference Part 8](/blog/quantization/)) recover much of the loss with the same kernels, since they only change which integers get stored.

## Step 5: FP8 scaling factors

FP8 E4M3 (4 exponent bits, 3 mantissa bits) covers ±448. A scale maps each tensor into that range: `x_fp8 = x / scale`, and `scale = amax / 448`, where **amax** is the largest absolute value. Two choices define a recipe: how many values share a scale, and when amax is measured.

### How many values share a scale

From coarse to fine: one scale per tensor, one per weight row and activation token, or blocks. DeepSeek-V3's [technical report](https://arxiv.org/abs/2412.19437) scales activations "on a 1x128 tile basis (i.e., per token per 128 channels)" and weights "on a 128x128 block basis." Block scales differ along K, so they can't be applied after the matmul; the kernel scales partial sums as it goes, which DeepSeek folds into promoting them to FP32 every 128 values of K, the fix for the 14-bit FP8 accumulator that [Part 6](/blog/tensor-cores/) measured.

What each choice costs, on block 15 of Qwen2.5-1.5B with Math Part 14's captured inputs (1,705 tokens), rounding each matrix and its inputs to FP8 and back, and comparing the layer output Y with an FP32 product:

| Weights / activations | Query projection, error in Y | MLP down projection, error in Y |
|---|---|---|
| Per-tensor / per-tensor | 3.09% | 3.36% |
| Per-row / per-token | 3.00% | 3.30% |
| 128 × 128 blocks / 1 × 128 tiles (DeepSeek-V3) | 2.82% | 3.09% |
| 1 × 32 blocks with power-of-two scales (MXFP8 style) | 3.90% | 4.18% |
| Per-tensor, activation scale 4 times too small | 16.0% | 6.6% |

Granularity barely mattered: about 3% everywhere. FP8's rounding error is a fixed percentage of each value, whatever the scale, and one scale per tensor flushed only 0.11% of the down projection's inputs to zero, though its largest input was 109 times their root-mean-square size. Clipping is what hurts: with a scale four times too small, everything above a quarter of amax saturates at 448, and the error rose two- to fivefold. For FP8, when amax is measured matters more than how finely values are grouped, the reverse of INT4.

### When amax is measured

**Current scaling** computes amax from the tensor being quantized: exact, but an extra pass over the data. **Delayed scaling**, in NVIDIA's [Transformer Engine docs](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html), "chooses the scaling factor based on the maximums of absolute values seen in some number of previous iterations" (its `DelayedScaling` recipe keeps 1,024 by default). It saves the pass but clips when a value larger than the history arrives. DeepSeek-V3 chose the first: "we calculate the maximum absolute value online for each 1x128 activation tile or 128x128 weight block." Inference engines often use **static scaling**, amax measured once on calibration data. Here an activation amax taken from a different chapter gave the same error as the current one: this model's outliers sit in the same channels at similar sizes.

On this GPU, PyTorch 2.11's `torch._scaled_mm` ran per-tensor and per-row FP8 but rejected the 1 × 128 and 128 × 128 modes ("only supported for CUDA 12.9 and above"), which cuBLAS lists for compute capability 9.0.

## Step 6: microscaling, MX and NVFP4

**Microscaling** puts a scale on every few dozen values and applies it in the tensor core. The [OCP Microscaling Formats (MX) v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), from Microsoft, AMD, Arm, Intel, Meta, NVIDIA and Qualcomm, defines blocks of 32 elements sharing one 8-bit E8M0 scale, "an unsigned representation of a conventional biased Float32 exponent": a pure power of two. Elements are MXFP8 (E5M2 or E4M3), MXFP6 (E3M2 or E2M3), MXFP4 (E2M1) or MXINT8, so MXFP4 costs 4 + 8/32 = 4.25 bits per value. The scale is "the largest power-of-two less than or equal to max(|Vi|), divided by the largest power-of-two representable in the element data type," and larger values are "clamped to the max normal." For E4M3 that lands each block's maximum between 256 and 512, so values above 448 clip, which is why the MXFP8-style row in Step 5 lost to per-tensor scaling.

NVFP4, NVIDIA's format, uses "one FP8 scale per 16 values (4.5 bits per value)," in E4M3, plus "a second-level FP32 scalar applied per tensor" ([NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)). Smaller blocks and non-power-of-two scales fit each block more tightly: Math Part 14 measured 11.8% weight error for MXFP4 and 9.4% for NVFP4 on Qwen's block 15, against 12.8% for INT4 in groups of 128.

These formats are fast only where hardware applies the scales. The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) defines `tcgen05.mma` kinds (`mxf8f6f4`, `mxf4`, `mxf4nvf4`) that compute `(A * scale_A) * (B * scale_B) + D` with scales in tensor memory, on data center Blackwell (`sm_100a`); block-scaled `mma.sync` "requires sm_120a," GeForce Blackwell. FP8 `mma.sync` "requires sm_89 or higher," so this laptop has FP8 but no FP6, FP4 or block-scaled instructions, and `torch._scaled_mm` with E8M0 block scales failed here with `CUBLAS_STATUS_NOT_SUPPORTED`. cuBLAS lists the 32-element E8M0 and 16-element E4M3 modes for compute capability 10.0. None of this was measured here.

## In the real world

- Marlin ([Frantar et al., 2024](https://arxiv.org/abs/2408.11743)) is the reference 4-bit kernel for Ampere-class GPUs. On an NVIDIA A10 it reports "speedups of approximately 3.9× relative to FP16 ... for batch sizes of up to 16-32," falling "towards 1.5× at batch size 128," and "up to a 2.8× speedup" end to end in vLLM. Beyond Step 2's kernel, it reshuffles weights offline into contiguous 16 × 64 tiles with nibbles interleaved for the `lop3` trick, loads with `cp.async` and an "evict first" cache hint, pipelines 4 stages through shared memory, and splits work across SMs in stream-K-like stripes.
- Machete is "the spiritual successor to the Marlin kernels" for Hopper, built on CUTLASS and `wgmma`, because "Marlin ... struggles on Hopper GPUs" ([Red Hat, 2024](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)). It reports performance "competitive with FP16" at batch 128 and above, where Step 3's kernels lost.
- vLLM walks a list of mixed-precision kernels, `CutlassW4A8LinearKernel, MacheteLinearKernel, MarlinLinearKernel, ...`, and takes the first that fits; in its source, Machete requires compute capability 9.0 exactly and Marlin 7.5 or newer. Its [quantization docs](https://docs.vllm.ai/en/latest/features/quantization/) list Marlin from Turing through Hopper and FP8 W8A8 on Ada and Hopper.
- llama.cpp stores weights in blocks ([`ggml-common.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)): `block_q4_0` is 32 weights with one FP16 scale, 4.5 bits per weight; `block_q4_K` is a 256-weight super-block of 8 sub-blocks with 6-bit scales and minimums, "effectively 4.5 bits per weight." Its own CUDA kernels skip FP16: activations become `q8_1` blocks and dot products run on integers, through `__dp4a` on older GPUs and INT8 `mma.sync ... s8.s8.s32` in the MMQ kernels, "enabled by default on GPUs with int8 tensor core support" ([build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)). On GeForce cards that uses the fastest tensor-core path Step 3 found.
- bitsandbytes added "4-bit inference kernels for batch size=1" in version 0.40.0. AutoAWQ, now deprecated in favor of vLLM's llm-compressor, shipped a GEMV kernel, "20% faster than GEMM, only batch size 1," and a GEMM "much faster than FP16 at batch sizes below 8." Both pick the kernel by batch size.

## Common misconceptions

- **"4-bit weights make inference 4 times faster."** Only for small batches, and not quite 4 times: the scales cap it at 3.76× for group 128, and this laptop's kernels reached 2.8× at one row. By M = 128 the gain was gone; at M = 4,096 the 4-bit kernels took 24% longer than BF16 (Triton) or 4.5 times as long (tinygemm).
- **"Dequantization is free because decode is memory-bound."** At 4 bits it isn't. The 4-bit GEMV executed as many instructions as the BF16 one while moving a quarter of the bytes, and reached 68% of the bandwidth roof against BF16's 92%.
- **"Finer scales always help."** For INT4 they do most of the work. For FP8, block scales cut the error from 3.36% to 3.09%, while a scale four times too small doubled it: avoiding clipping matters more.
- **"FP8 is twice as fast as BF16 wherever it's supported."** It depends on the kernel the scales select. Per-tensor FP8 ran 1.8 times BF16 at large M here; per-row FP8 took 29% longer than BF16.

## Key takeaways

- Weight-only quantization moves decode's memory-bound limit with the bytes: a fully 4-bit Qwen2.5-1.5B decoded in 5.9 ms against 15.1 ms in BF16.
- A W4A16 kernel loads packed integers, unpacks them with shifts and masks, applies a scale per group of 128, and hands BF16 tiles to the tensor cores. BF16 weights only ever exist in registers.
- The gain fades as rows grow: a 4-bit matmul is compute-bound from about 28 rows, and on this GPU it was no faster than BF16 at 128 rows and slower beyond.
- Big batches need quantized activations too. INT8 W8A8 ran 3.3 times faster than BF16 on this GeForce chip, but only if the quantization is fused.
- FP8 cares about clipping more than granularity. MX and NVFP4 put a scale on every 32 or 16 values, and only Blackwell applies them in hardware.

## What's next

Mixture-of-experts models make the small-batch problem worse: each token visits only a few experts, so even a large batch turns into many small matrix multiplies, one per expert, often with 4-bit or FP8 weights. [Part 12](/blog/moe-kernels/) covers mixture-of-experts kernels: routing and permuting tokens, grouped GEMMs, fused MoE kernels in vLLM and DeepGEMM, and load imbalance on the GPU.

## Further reading

- [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models (Frantar et al., 2024)](https://arxiv.org/abs/2408.11743): the 4-bit kernel design, with near-ideal speedups to batch 16-32 and its vLLM results.
- [OCP Microscaling Formats (MX) Specification v1.0 (Open Compute Project, 2023)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): MXFP8, MXFP6, MXFP4 and MXINT8, E8M0 scales and the scale rule.
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference (NVIDIA, 2025)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/): 16-value blocks with E4M3 scales and a per-tensor FP32 scale.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): 1 × 128 and 128 × 128 FP8 scaling, online amax and promotion to FP32.
- [Transformer Engine: Using FP8 and FP4 (NVIDIA)](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html): delayed and current scaling, and the MXFP8 and NVFP4 recipes.
- [cuBLAS documentation (NVIDIA)](https://docs.nvidia.com/cuda/cublas/): FP8 and FP4 scaling modes and the compute capability each needs.
- [PTX ISA (NVIDIA)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): `mma.sync` FP8 on sm_89, block-scaled `tcgen05.mma` on Blackwell.
- [NVIDIA Ada GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf): INT8, FP8 and BF16 tensor-core rates on Ada.
- [Introducing Machete, a Mixed-Input GEMM Kernel Optimized for NVIDIA Hopper GPUs (Red Hat, 2024)](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel): Marlin's successor on `wgmma`.
- [vLLM docs: quantization](https://docs.vllm.ai/en/latest/features/quantization/): which formats and kernels run on which GPUs.
- [llama.cpp `ggml-common.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h) and [build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md): GGUF block layouts and the MMQ integer kernels.
- [AutoAWQ (Casper Hansen)](https://github.com/casper-hansen/AutoAWQ): AWQ's GEMV and GEMM kernels and when each wins.
