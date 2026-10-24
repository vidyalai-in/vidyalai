---
title: "Kernel Fusion, torch.compile and CUDA Graphs, Measured"
description: "Fuse SwiGLU and RMSNorm by hand and with torch.compile, read the Triton it writes, replay decode as a CUDA graph, and see how PyTorch's allocator fragments."
slug: "kernel-fusion-cuda-graphs"
category: "gpu-kernels"
tags: ["gpu", "cuda", "latency", "triton", "memory"]
series: "llm-gpu"
seriesOrder: 9
pubDate: 2026-10-24
---

## TL;DR

Elementwise operations such as adds, activations and normalizations do about one floating-point operation per byte they move, so memory bandwidth alone sets their speed. Run as separate kernels, each one writes its result to memory and the next reads it back. On a laptop RTX 4060, the residual add and RMSNorm from Hugging Face's Qwen2 code launched 9 kernels and moved 42 bytes per element; one fused Triton kernel moved 8 and ran 5.2 times faster, and `torch.compile` wrote the same kernel by itself. In decode, launching costs more than computing: CUDA graphs replay a whole step with one launch, and with fusion they took a Qwen2.5-1.5B decode step from 22.2 ms to 13.6 ms, 91% of the bandwidth limit. You'll also read the Triton that Inductor generates and see PyTorch's caching allocator run out of memory with 896 MiB free.

## Why it matters

Here is RMSNorm, the normalization in every Llama-style layer, written exactly as Hugging Face's Qwen2 model writes it, after a residual add. On an 8,192 × 4,096 BF16 tensor it launches 9 GPU kernels and takes 6.2 ms. A 20-line Triton kernel computes the same result in 1.19 ms. The arithmetic is identical; the slow version moves five times as many bytes.

The second cost doesn't show up in a kernel benchmark. [Part 4](/blog/gpu-profiling-roofline/) traced one eager decode step of Qwen2.5-1.5B: 1,142 kernel launches, with the GPU idle 31% of the time while Python queued work. [Part 2](/blog/cuda-programming-model/) said memory-bound kernels gain most from "not running them at all, by fusing them into neighbors." This part does that, then removes the idle time.

## The intuition

Picture a workshop with a warehouse at the end of a long corridor. Each elementwise kernel is a worker who carries a crate from the warehouse to the bench, does one small job (add, square, scale) and carries it back. Nine kernels are nine round trips with the same crate; fusion is one trip with all nine jobs done at the bench.

The foreman (the CPU) also hands out jobs one slip at a time, and some slips take longer to write than the jobs take to do. A CUDA graph is the whole shift's list, printed once and handed over in one go.

The analogy breaks in two places. Fusion only works for jobs that fit on one bench: a matrix multiply needs crates from all over the warehouse, so compilers usually hand it to a specialist library. And the printed list is rigid: same jobs, same crates, same shelf positions, every shift.

## Step 1: why elementwise kernels are memory-bound

[Part 4](/blog/gpu-profiling-roofline/) measured this GPU's roofline: 25.3 BF16 TFLOPS against 248.7 GB/s of DRAM bandwidth (DRAM is the GPU's main memory), so a kernel needs about 102 FLOPs (floating-point operations) per byte before compute limits it. SwiGLU's activation, `silu(g) * u`, computes `g / (1 + exp(-g)) * u`: about 5 FLOPs per element, counting the exponential as one. In BF16 it reads two 2-byte inputs and writes one 2-byte output. That's 0.8 FLOPs per byte, more than 100 times below the ridge.

So an elementwise kernel's time is its bytes divided by bandwidth, which for mixed reads and writes on this laptop Part 2 found "levels off near 226 GB/s." To go faster, move fewer bytes: drop the intermediates one kernel writes and the next reads back.

Count them in the Qwen2 code. `Qwen2RMSNorm.forward` in Transformers 5.20.2 casts to FP32, squares, averages, adds epsilon, takes the reciprocal square root, multiplies, casts back to BF16 and multiplies by the weight. With the residual add in front, the profiler counted 9 kernels. Per element of an 8,192 × 4,096 tensor:

| Kernel | Reads | Writes | Bytes per element |
|---|---|---|---|
| `h = res + x` (BF16) | 4 | 2 | 6 |
| `h.to(float32)` | 2 | 4 | 6 |
| `.pow(2)` | 4 | 4 | 8 |
| `.mean(-1)` | 4 | 0 | 4 |
| `+ eps`, `rsqrt` (one value per row) | 0 | 0 | 0 |
| `x * rsqrt(...)` | 4 | 4 | 8 |
| `.to(bfloat16)` | 4 | 2 | 6 |
| `weight * x` | 2 | 2 | 4 |
| Total, 9 kernels | 24 | 18 | 42 |
| One fused kernel: read `x` and `res`, write `h` and the output | 4 | 4 | 8 |

At 226 GB/s, 42 bytes per element over 33.5 million elements takes 6.2 ms, and 8 bytes takes 1.19 ms. SwiGLU is the same story, smaller: eager PyTorch runs `silu` and the multiply as two kernels, 10 bytes per element instead of 6, the 40% Part 4 found missing.

## Step 2: fusing them by hand in Triton

[Part 7](/blog/triton-kernels/) introduced Triton, where you write the work of one program (one thread block) as operations on whole tensors. A fused residual add and RMSNorm takes one program per row, because the mean runs along the row:

```python
@triton.jit
def add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, h_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)                               # one program per token (row)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    h = (x + r).to(tl.bfloat16)                          # residual stream stays BF16, as in eager
    tl.store(h_ptr + row * n_cols + offs, h, mask=mask)
    hf = h.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(hf * hf, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = (hf * rstd).to(tl.bfloat16) * w                  # round to BF16, then scale, as in eager
    tl.store(y_ptr + row * n_cols + offs, y, mask=mask)

def add_rmsnorm_triton(x, res, w):
    h = torch.empty_like(x); y = torch.empty_like(x)
    n_rows, n_cols = x.shape
    add_rmsnorm_kernel[(n_rows,)](x, res, w, h, y, n_cols, 1e-6,
                                  BLOCK=triton.next_power_of_2(n_cols), num_warps=8)
    return h, y
```

It writes the new residual `h` and the normalized `y`; the FP32 intermediates never leave the chip, and `tl.sum` hides the reduction tree that [Part 8](/blog/gpu-reductions/) builds by hand. `h` matched eager PyTorch bit for bit, and `y` differed by at most one BF16 rounding step (0.031), since the sum is added up in a different order. The SwiGLU kernel is Part 7's vector add with `g * tl.sigmoid(g) * u` in the middle.

### Measured

Every case uses 8,192 × 4,096 BF16 tensors, Part 4's shape, on [Part 1](/blog/gpu-from-python-to-silicon/)'s machine: an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, 24 SMs) under WSL2, driver 592.82, PyTorch 2.11.0 with CUDA 12.8, Triton 3.6.0 and Transformers 5.20.2. Times are `triton.testing.do_bench` medians over one second after warm-up, in three rounds; kernel counts come from `torch.profiler`. The memory clock held 8,000 MHz (a 256 GB/s roof), the SM clock ran 1,410 to 1,815 MHz, and power sat at a 40 W limit. "Effective GB/s" divides the fused kernel's bytes (8 or 6 per element) by the time, so it measures the job, not the traffic:

| Version | Kernels | Time | Bytes actually moved | Effective GB/s |
|---|---|---|---|---|
| Add + RMSNorm, eager Qwen2 code | 9 | 6,177 µs | 1,409 MB | 43.5 |
| Add, then PyTorch's `F.rms_norm` | 2 | 1,473 µs | 336 MB | 182.2 |
| Add + RMSNorm, hand-written Triton | 1 | 1,194 µs | 268 MB | 224.8 |
| Add + RMSNorm, `torch.compile` | 1 | 1,196 µs | 268 MB | 224.4 |
| `silu(g) * u`, eager | 2 | 1,484 µs | 336 MB | 135.7 |
| `silu(g) * u`, hand-written Triton | 1 | 897 µs | 201 MB | 224.4 |
| `silu(g) * u`, `torch.compile` | 1 | 896 µs | 201 MB | 224.6 |

Divide the bytes each version actually moved by its time and every row lands between 224 and 228 GB/s, Part 2's read-write ceiling: the eager versions ran at full bandwidth too, with five or two times as much to move. Fusion made add plus RMSNorm 5.2 times faster and cut SwiGLU's time by 40%, as Part 4 predicted. PyTorch's `F.rms_norm` is already one fused kernel; the Qwen2 code pays for spelling RMSNorm out in Python. The compiled versions tied the hand-written ones within 0.2%.

## Step 3: what torch.compile generates

`torch.compile` produced those kernels from the unchanged eager functions. The PyTorch 2 paper ([Ansel et al., ASPLOS 2024](https://doi.org/10.1145/3620665.3640366)) describes its parts. **TorchDynamo** "hooks into the Python frame evaluation API in CPython to dynamically modify Python bytecode right before it is executed," pulling the PyTorch operations out into an **FX graph**, a list of operations and their inputs. It also records **guards**, checks on the inputs (shapes, data types, devices) that must hold for the graph to be reused. **AOTAutograd** traces through PyTorch's autograd engine to get a graph of ATen operations (ATen is PyTorch's C++ operator library), plus a backward graph when training. **TorchInductor** "translates PyTorch programs into OpenAI's Triton for GPUs."

![A top-to-bottom flowchart. A Python function first meets a check: do the guards match a cached compile? If yes, the cached Python wrapper launches the kernels. If no, TorchDynamo records an FX graph and guards; unsupported code such as .item() causes a graph break and runs in Python. The graph goes to AOTAutograd, which produces an ATen-level graph, then into TorchInductor, which fuses, schedules and reuses buffers and emits Triton kernels and cuBLAS calls through extern_kernels.mm. Both feed a Python wrapper that launches the kernels; with mode reduce-overhead, that wrapper is replayed as one CUDA graph.](/diagrams/g09-torch-compile.svg "torch.compile from Python to kernels. The guard check runs on every call.")

Every call pays for the guard check; only a failed check pays for compiling again.

### Reading the generated code

`torch._inductor.utils.run_and_get_code(compiled_fn, *args)` returns the Python module Inductor wrote (`TORCH_LOGS=output_code` prints it). For add plus RMSNorm it held one Triton kernel, named after every operation it absorbed. Trimmed, with some temporaries folded together:

```python
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2,
                                                        xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 8192                                    # rows: the shape is baked in
    r0_numel = 4096                                  # columns, reduced over
    ...
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):        # pass 1: add, store h, sum of squares
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        _tmp6 = tl.where(r0_mask, _tmp6 + tmp2 * tmp2, _tmp6)
        tl.store(out_ptr0 + (r0_1 + 4096*x0), tmp2, r0_mask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):        # pass 2: reload h, scale, multiply by weight
        tmp8 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(out_ptr0 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = libdevice.rsqrt(tmp6 / 4096.0 + 1e-06)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp8 * (tmp9 * tmp15), r0_mask)
```

Unlike the hand-written kernel, it walks each row in chunks and re-reads `h` in a second pass (the 8 KB row was just written, so this mostly hits the cache), adds cache hints (`evict_first` for data used once, `evict_last` for the shared weight vector), and squares the FP32 sum before rounding it to BF16, skipping one rounding eager PyTorch does. The shape is a constant: this kernel only runs on 8,192 × 4,096.

### Where it stops: the matmuls

Now the MLP half of a Qwen2.5-1.5B layer (hidden size 1,536, MLP width 8,960, random weights): residual add, RMSNorm, gate and up projections, `silu * mul`, down projection, second residual add. Inductor's `call()` function for one token, trimmed:

```python
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg2_1, buf1, buf5, 1, 1536, stream=stream0)
extern_kernels.mm(buf1, reinterpret_tensor(arg3_1, (1536, 8960), (1, 1536), 0), out=buf2)   # gate_proj
extern_kernels.mm(buf1, reinterpret_tensor(arg4_1, (1536, 8960), (1, 1536), 0), out=buf3)   # up_proj
buf4 = buf2; del buf2  # reuse
triton_poi_fused_mul_silu_1.run(buf4, buf3, 8960, stream=stream0)
extern_kernels.addmm(buf5, buf4, reinterpret_tensor(arg5_1, (8960, 1536), (1, 8960), 0), alpha=1, beta=1, out=buf6)
```

The elementwise chains became two Triton kernels (`red` for a reduction, `poi` for pointwise), and the three matrix multiplies went to cuBLAS through `extern_kernels`. The last residual add vanished into `addmm`, cuBLAS's "multiply and add a matrix" call, at the cost of a 1.3 µs copy of the residual into the output. `buf4 = buf2` reuses the gate output's memory for the SwiGLU result.

Per-kernel GPU time from the profiler, median of 20 calls:

| Tokens | Eager kernels | Compiled kernels | Matmul time | Other kernels, eager | Other kernels, compiled |
|---|---|---|---|---|---|
| 1 (decode) | 15 | 6 | 354 to 357 µs | 15.6 to 20.7 µs | 4.7 µs |
| 2,048 (prefill) | 16 | 7 | 7.83 to 7.89 ms | 958 to 992 µs | 543 µs |

For one token, fusion saved 11 to 16 µs of GPU time, but the 9 kernels it removed ran 1 to 2 µs each and cost the CPU about 8 µs each to launch (8.1 µs, measured as in Part 1). For 2,048 tokens it saved 0.43 ms of 8.4, 5%, because cuBLAS's tensor-core matmuls ([Part 6](/blog/tensor-cores/)) dominate prefill.

### Compile time, recompiles and graph breaks

The first call compiles: 2.5 s for add plus RMSNorm on a cold cache, and 31 to 63 s for the whole decode step, depending on the mode. Inductor caches its output on disk, and with a warm cache a new process compiled the decode step in 3 to 5 s.

A shape change fails a guard. Calling the compiled MLP half with 1 token and then 2, `TORCH_LOGS=recompiles` printed:

```text
Recompiling function mlp_half in f2_layer.py:20
    triggered by the following guard failure(s):
    - 0/0: tensor 'x' size mismatch at index 0. expected 1, actual 2
```

That recompile took 1.28 s, and calls with 3 and 64 tokens then compiled nothing: by default, say the [`torch.compile` docs](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html), "we automatically detect if dynamism has occurred and compile a more dynamic kernel upon recompile," one where the token count is a variable.

A **graph break** happens where Dynamo meets code it can't capture. Add `if h.float().abs().max().item() > 100: print(...)` to the norm function, and `torch._dynamo.explain` reports 2 graphs and 1 break: "Unsupported Tensor.item() call with capture_scalar_outputs=False." `.item()` copies a value to the CPU so Python can pick a branch, so the code on either side compiles separately, with a CPU round trip and no fusion across the gap. The Qwen2 decode step with a static cache had no breaks: one graph of 1,089 operations.

### Modes: default, reduce-overhead, max-autotune

Per the docs, `"reduce-overhead"` "reduces the overhead of python with CUDA graphs," and `"max-autotune"` "leverages Triton or template based matrix multiplications on supported devices" and "enables CUDA graphs by default." PyTorch 2.11's source adds two details that matter here:

- Inductor's `is_big_gpu` check wants 68 SMs (its comment names the RTX 3080) before it tries Triton matmul templates. This GPU has 24, so every compile logged "Not enough SMs to use max_autotune_gemm mode."
- `max-autotune` turns on `coordinate_descent_tuning`, and with it `torch/_inductor/decomposition.py` rewrites any matrix multiply with a single row as a multiply and a sum, a reduction Inductor can fuse with its neighbors. Its comment: "Our matrix vector multiplies only achieve peak bandwidth with coordinate descent tuning."

So under `max-autotune`, decode's matrix-vector products became Triton kernels. The down projection became `triton_red_fused__unsafe_view_mm_mul_silu_t_view_9`, which computes `silu(g) * u` on the fly while reading the weights. Only the 84 projections with a bias (query, key and value in 28 layers) stayed with cuBLAS.

## Step 4: CUDA graphs and the decode step

A **CUDA graph** records a sequence of kernel launches once and replays them with one call; [Part 1](/blog/gpu-from-python-to-silicon/) introduced PyTorch's `torch.cuda.CUDAGraph` and `torch.cuda.graph`. The price of a launch on its own: 1,000 back-to-back `x.add_(1)` kernels on a one-element tensor, timed with CUDA events, median of 21:

| Launched as | Time per 1,000 kernels | Per kernel |
|---|---|---|
| Eager PyTorch, one launch each | 10.4 to 11.4 ms | 10.4 to 11.4 µs |
| One CUDA graph replay | 1.15 to 1.37 ms | 1.15 to 1.37 µs |

From the graph, tiny kernels run 8 to 9 times faster, paying only the GPU's own scheduling. In exchange the graph is rigid. PyTorch's [CUDA semantics notes](https://docs.pytorch.org/docs/2.11/notes/cuda.html) say "Ops that synchronize the CPU with the GPU (e.g., `.item()` calls) are prohibited" and "Dynamic shapes are prohibited," as is control flow that depends on data. Some mistakes fail silently: "Every replay reads from and writes to the same (virtual) memory addresses," so new inputs must be copied into the captured input tensors. A `.item()` inside the capture above failed with a CUDA error.

That rules out Hugging Face's default cache, which grows by concatenation. Transformers 5.20.2's `StaticCache` preallocates keys and values at their maximum length (512 tokens here) and writes each new token at the index `cache_position`, so every step has the same shapes and addresses.

### The decode step, six ways

The workload is Part 4's: Qwen2.5-1.5B-Instruct in BF16, batch size 1, a 43-token chat prompt, greedy decoding with PyTorch's scaled dot-product attention. All six versions ran in one process, sharing one `StaticCache` and its input buffers, in four interleaved rounds so clock drift hit them alike; each round timed 40 steps with `synchronize()` around each (median), then profiled 5 more. All six produced the same 20 greedy tokens. The memory clock held 8,000 MHz (one round dipped to 7,500), the SM clock ran 1,845 to 2,175 MHz, and power reached 52.6 W under a limit of 47 to 50 W. Medians of the four rounds:

| Version | Kernels per step | CPU time until the call returns | Step time | Tokens/s | GPU busy (profiler) | GPU idle |
|---|---|---|---|---|---|---|
| Eager, default `DynamicCache` (Part 4's setup) | 1,145 | 20.3 ms | 22.2 ms | 45.1 | 13.2 ms | 40% |
| Eager, `StaticCache` | 1,374 | 22.9 ms | 24.9 ms | 40.1 | 14.6 ms | 42% |
| `StaticCache` step captured as a CUDA graph | 1,374 | 0.67 ms | 14.9 ms | 67.3 | 14.7 ms | 1% |
| `torch.compile`, default mode | 509 | 8.4 ms | 15.6 ms | 64.2 | 13.2 ms | 15% |
| `torch.compile`, `mode="reduce-overhead"` | 510 | 0.72 ms | 13.8 ms | 72.6 | 13.2 ms | 4% |
| `torch.compile`, `mode="max-autotune"` | 454 | 0.76 ms | 13.6 ms | 73.4 | 13.0 ms | 4% |

The busy column sums kernel durations from the profiler, which on this WSL2 setup read about 9% shorter than CUDA events on the same kernel (1.708 against 1.867 ms for a 467 MB matrix-vector product), so the idle column is an upper bound.

Read the table in pairs:

- A `StaticCache` makes eager decode slower: 229 more small kernels per step, for index copies and the mask over 512 slots. It pays off only once the step is captured.
- Capturing that step took 0.2 s and cut it from 24.9 to 14.9 ms with the same 1,374 kernels. The CPU's part shrank to a 0.67 ms replay call, and the GPU, fed continuously, set the pace.
- `torch.compile` in default mode fused 1,149 small kernels (2.0 ms of GPU time) into 282 Triton kernels (0.5 ms), but still launched all 509 from Python, so the GPU waited 15% of the time.
- `reduce-overhead` did both, for 13.8 ms; the 1.1 ms it gained on the plain graph is mostly the 1.5 ms of GPU work fusion removed. `max-autotune` took another 0.14 ms off with its Triton matrix-vector kernels, after a 63 s cold compile.

The eager rows depend on the CPU. An earlier session, with other work loading the laptop's CPU, measured eager decode at 37.6 ms but the static-cache graph at 16.3 ms, only about 1 ms slower (partly from a lower memory clock).

### How close to the limit?

Each step must read all 3.087 GB of weights once. At Part 4's measured 248.7 GB/s that takes 12.4 ms, a ceiling of 80.6 tokens per second. The `reduce-overhead` step reached 90% of it and `max-autotune` 91%, against 56% for eager decode. What remains is mostly the matrix-vector kernels themselves, about 12 ms of each step, plus attention over the 512-slot cache.

## Step 5: the caching allocator

CUDA graphs can trust fixed addresses because PyTorch rarely gives memory back to CUDA. Every GPU tensor comes from its **caching allocator**, which the [CUDA semantics notes](https://docs.pytorch.org/docs/2.11/notes/cuda.html) say exists "to speed up memory allocations" and lets memory be freed without synchronizing the device. Setting `PYTORCH_NO_CUDA_MEMORY_CACHING=1` makes PyTorch call `cudaMalloc` and `cudaFree` directly, which tests both claims:

| Operation (median of 200) | Caching allocator | Plain `cudaMalloc`/`cudaFree` |
|---|---|---|
| Allocate and free 4 KiB | 2.1 µs | 463 µs |
| Allocate and free 1 MiB | 1.4 µs | 479 µs |
| Allocate and free 64 MiB | 2.3 µs | 2,670 µs |
| `del` a 64 MiB tensor with 44 ms of matmuls queued | 0.007 ms | 41.5 ms |

In the last row `cudaFree` waited for the queued work to finish, stalling a CPU that [Part 1](/blog/gpu-from-python-to-silicon/) showed should run ahead. With caching, a freed block returns to PyTorch's pool, tagged with its stream, and the next allocation on that stream reuses it at once, since a stream runs its work in order.

### Blocks and segments

The rules, from the comments and constants in PyTorch 2.11's `CUDACachingAllocator.cpp`: sizes round up to multiples of 512 bytes; requests up to 1 MiB are "small" and get packed into 2 MiB **segments** (one `cudaMalloc` each); requests between 1 and 10 MiB "will allocate and split a 20MB block"; larger ones get their own segment, rounded to 2 MiB. A cached free block bigger than a request can be split. `torch.cuda.memory_allocated()` counts bytes in live tensors; `torch.cuda.memory_reserved()` counts everything held from CUDA:

| After | `memory_allocated` | `memory_reserved` |
|---|---|---|
| A 1,000-byte tensor | 0.001 MiB | 2 MiB |
| Plus a 5 MiB tensor | 5.001 MiB | 22 MiB |
| Plus a 50 MiB + 1 byte tensor | 55.001 MiB | 74 MiB |
| `del` all three | 0 | 74 MiB |
| `torch.cuda.empty_cache()` | 0 | 0 |

`nvidia-smi` reports the reserved figure. `empty_cache()` releases only segments with nothing alive in them.

### Fragmentation: out of memory with 896 MiB free

Splitting fragments memory. Under a 2 GiB cap (`torch.cuda.set_per_process_memory_fraction`, to keep the test small), allocate sixteen 120 MiB tensors and free them, then allocate sixteen 64 MiB tensors. Each takes a cached 120 MiB block and splits it, leaving a 56 MiB hole in every segment. Then ask for 200 MiB:

```text
16 x 120 MiB                               allocated   1920 MiB  reserved   1922 MiB
freed them (cached, not returned)          allocated      0 MiB  reserved   1922 MiB
16 x 64 MiB (each splits a cached 120)     allocated   1024 MiB  reserved   1922 MiB
200 MiB request: OutOfMemoryError
  Of the allocated memory 1024.00 MiB is allocated by PyTorch, and 896.00 MiB is reserved
  by PyTorch but unallocated. If reserved but unallocated memory is large try setting
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

896 MiB were free, in sixteen 56 MiB pieces. When `cudaMalloc` fails, the allocator frees cached blocks and retries, but only blocks "that are not split," and every segment here held a live tensor. This is **external fragmentation**: enough free memory in total, none in one piece. Serving hits it when batch sizes and sequence lengths drift.

With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the same script succeeded: 1,224 MiB allocated in the same 1,922 MiB reserved. The docs say the option "allows the allocator to create a segment initially and then expand its size later when more memory is needed." Per the source's design note, CUDA's virtual memory API separates reserving addresses from mapping physical memory, so one segment per stream grows in 2 MiB or 20 MiB pages, and when memory runs out "we can unmap all the memory in our segment corresponding to empty physical pages" and map it where it's needed. The costs: slower first allocations, and extra steps to share the memory between processes.

In a real program, `torch.cuda.memory_summary()` prints the allocator's counters, including the "inactive split" bytes (896 MiB above). For a timeline, `torch.cuda.memory._record_memory_history()` then `torch.cuda.memory._dump_snapshot("snap.pickle")` writes a file that pytorch.org/memory_viz draws, per the [memory snapshot docs](https://docs.pytorch.org/docs/2.11/torch_cuda_memory.html).

CUDA graphs rely on the allocator too: capture uses "a separate private pool for each capture," per PyTorch's notes, held until the graph and its tensors are gone. That keeps a replayed step's addresses valid, and it's why each captured batch size costs memory.

## In the real world

vLLM uses both tools by default. Its [torch.compile design doc](https://docs.vllm.ai/en/latest/design/torch_compile.html) says that "in vLLM's V1 architecture, `torch.compile` is enabled by default," for a symbolic batch size, split at attention. Its [CUDA graphs doc](https://docs.vllm.ai/en/latest/design/cuda_graphs.html) defaults to `FULL_AND_PIECEWISE`: "full CUDA Graph for uniform decode, piecewise CUDA Graphs for others," where piecewise runs attention eagerly between captured pieces. A graph "captured for a specific size can only be used for the same size," per the [engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html), so vLLM captures batch sizes `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))`, up to 512 by default (1,024 on data center Blackwell GPUs). `--enforce-eager` turns both off.

SGLang's [server arguments](https://docs.sglang.io/advanced_features/server_arguments.html) set separate maximum batch sizes for decode and prefill CUDA graphs, and `--enable-torch-compile` "accelerates small models on small batch sizes," up to `--torch-compile-max-bs` (default 32). llama.cpp's CUDA backend captures each token's work as a graph at batch size 1; NVIDIA's [write-up](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/) reports up to 1.2 times faster for "the smallest Llama 7B model on the fastest NVIDIA H100 GPUs," and patches changed kernel parameters into the graph as the KV cache grows instead of recapturing.

Engines capture graphs mainly for decode. In Step 3's 2,048-token run, the MLP half's 7 to 16 kernels averaged 0.5 to 1.2 ms each, so an 8 µs launch hides behind them, and prefill lengths vary with every request, which would need a graph per length. Fusion still helps prefill, by the few percent Step 3 measured.

Hand-written fused kernels live on beside compilers. vLLM ships `fused_add_rms_norm` and `SiluAndMul`, described in [Inference Part 7](/blog/flashattention/), and the Liger-Kernel and Unsloth libraries from [Part 7](/blog/triton-kernels/) keep Triton RMSNorm and SwiGLU kernels for training.

## Common misconceptions

- **"Fused kernels are faster because they compute less."** The arithmetic is the same. Add plus RMSNorm ran 5.2 times faster because it moved 8 bytes per element instead of 42; every version, eager or fused, moved its bytes at 224 to 228 GB/s.
- **"torch.compile makes decode fast by fusing."** Fusion cut the step from 1,374 kernels to 509, but default mode still spent 8.4 ms of CPU time launching them and the step took 15.6 ms. CUDA graphs made the big difference, with or without the compiler.
- **"max-autotune tunes the matmuls."** On this 24-SM laptop Inductor skips its Triton matmul templates (it wants 68 SMs). Its decode gain, 1% over `reduce-overhead`, came from rewriting matrix-vector products as fusable Triton reductions.
- **"If memory is free, an allocation will fit."** With 896 MiB reserved and unused, a 200 MiB request failed because the free space was sixteen 56 MiB holes. `expandable_segments:True` let it succeed.
- **"A CUDA graph records what your code did."** It records addresses. Every replay reads and writes the same memory, so inputs must be copied into the captured buffers, and anything that needs the CPU mid-step, like `.item()`, can't be captured.

## Key takeaways

- Elementwise kernels do about one FLOP per byte, a hundred times below this GPU's ridge, so their time is bytes divided by bandwidth. Fusion removes the intermediate writes and reads, and nothing else.
- Hugging Face's Qwen2 add and RMSNorm moved 42 bytes per element in 9 kernels. One fused Triton kernel, hand-written or from `torch.compile`, moved 8 and ran 5.2 times faster.
- TorchInductor fuses elementwise chains and reductions into Triton, hands matmuls to cuBLAS, bakes in shapes and recompiles when a guard fails. `TORCH_LOGS=output_code` shows what it wrote.
- For batch-1 decode, launches cost more than the small kernels. A CUDA graph cut a Qwen2.5-1.5B step from 24.9 to 14.9 ms, and `torch.compile` with graphs reached 13.6 ms, 91% of the bandwidth limit.
- The caching allocator turns a synchronizing `cudaFree` into a microsecond pool operation, but split blocks can fragment memory. Compare `memory_reserved` with `memory_allocated`, and try `expandable_segments:True`.

## What's next

Fusion stopped at attention in this part: Inductor left it to PyTorch's scaled dot-product kernel, and vLLM splits its graphs around it. [Part 10](/blog/attention-kernel/), Writing an attention kernel, builds one in Triton: FlashAttention-style tiling with the online softmax, measured against a naive version and PyTorch's SDPA, then decode attention with split-KV and paged-KV indexing.

## Further reading

- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation (Ansel et al., ASPLOS 2024)](https://doi.org/10.1145/3620665.3640366): TorchDynamo and TorchInductor, from the people who built them.
- [torch.compile API (PyTorch 2.11 docs)](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html): modes, `dynamic` and the other options measured here.
- [CUDA semantics (PyTorch 2.11 docs)](https://docs.pytorch.org/docs/2.11/notes/cuda.html): the caching allocator, `PYTORCH_CUDA_ALLOC_CONF` options and CUDA graph constraints.
- [Understanding CUDA memory usage (PyTorch 2.11 docs)](https://docs.pytorch.org/docs/2.11/torch_cuda_memory.html): memory snapshots and the pytorch.org/memory_viz viewer.
- [CUDACachingAllocator.cpp (PyTorch source, v2.11.0)](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp): the allocator's rules and the expandable segments design note, in its comments.
- [CUDA graphs chapter, CUDA Programming Guide (NVIDIA)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html): how graphs are built, instantiated and launched.
- [vLLM design docs: torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile.html) and [CUDA graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html): how a production engine uses both.
- [Optimizing llama.cpp AI Inference with CUDA Graphs (NVIDIA)](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/): graphs in a C++ engine, updated as the KV cache grows.
- [Making Deep Learning Go Brrrr From First Principles (Horace He)](https://horace.io/brrr_intro.html): bandwidth, overhead and why fusion is the main tool against both.
