---
title: "Many GPUs: NVLink, NCCL and Collectives"
description: "How GPUs add up tensors together: ring and tree all-reduce, the latency-bandwidth cost model, NVLink vs PCIe vs InfiniBand, and hiding the traffic."
slug: "nccl-collectives"
category: "gpu-kernels"
tags: ["gpu", "cuda", "latency", "serving"]
series: "llm-gpu"
seriesOrder: 13
pubDate: 2026-10-24
---

## TL;DR

GPUs that share a model exchange data through a few collective operations: all-reduce, all-gather, reduce-scatter, broadcast and all-to-all. Each costs a fixed latency per step plus bytes divided by link bandwidth. A ring all-reduce has each GPU send just under twice the message, however many GPUs there are, but takes `2(p − 1)` steps, so small messages are latency-bound and trees or in-switch reductions win there. This laptop has one GPU, so no multi-GPU collective is measured. Instead you'll fit the cost model to its real PCIe link (8.25 µs plus 13.45 GB/s), simulate a ring in NumPy, hide a copy behind compute with two CUDA streams, and price tensor parallelism of a 70B model on NVLink, PCIe and InfiniBand.

## Why it matters

Run a 4,096-token prompt through Llama 3.1 70B split across eight H100s. The math takes about 73 ms. The 160 all-reduces that tensor parallelism adds take about 44 ms over NVLink, and about 300 ms over PCIe.

[Inference Part 10](/blog/model-parallelism/) explained why tensor parallelism needs those all-reduces. This part goes one level down: how an all-reduce moves its bytes, when a tree beats a ring, what changes when the switch does the adding, and how to hide the traffic behind compute.

The laptop has one GPU, and NCCL, NVIDIA's collective library, refuses to put two ranks on it: PyTorch's bundled NCCL 2.28.9 stops with `NCCL WARN Duplicate GPU detected : rank 1 and rank 0 both on CUDA device`. So every multi-GPU number in this article comes from NVIDIA's documentation or from arithmetic, not from a measurement. What the laptop can measure is one real link, its CPU-to-GPU PCIe connection, and the copy-and-compute overlap that multi-GPU systems use to hide their traffic.

## The intuition

Four teachers have each graded one section of the same exam for 400 students, and every teacher needs every student's total.

The obvious plan sends every list to a head teacher, who adds them and sends the totals back: three lists in, three out. With 64 teachers, 63 each way.

The ring plan splits the students into four groups of 100. Each teacher passes one group's partial totals to the left, where the neighbor adds their own scores and passes the sum on. After three passes, each teacher holds the final totals for one group; three more passes share them around. Each teacher handles 1.5 lists in total, and under 2 even with 64 teachers.

The analogy breaks in two ways. Every pass has a fixed setup time however short the list, so 126 passes around 64 teachers are slow for a ten-name list. And on a GPU the adding runs on the same streaming multiprocessors (SMs) that do the model's math.

## Step 1: the latency-bandwidth model, on a real link

The standard cost model for one message has two numbers. Thakur, Rabenseifner and Gropp's [MPICH collectives paper](https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf) models the time to send a message as "α + nβ, where α is the latency (or startup time) per message, independent of message size, β is the transfer time per byte". With bandwidth `B = 1/β`:

```text
t(n) = α + n / B
```

Small messages cost about α, large ones about n/B. The two terms are equal at `n½ = α × B`, the size at which you get half the link's bandwidth.

### Measuring α and B on this laptop's PCIe link

The laptop's GPU sits on PCI Express (PCIe), the standard expansion bus. `nvidia-smi -q` reports the link it negotiated:

```text
        GPU Link Info
            PCIe Generation
                Max                                    : 4
                Current                                : 4
            ...
            Link Width
                Max                                    : 8x
                Current                                : 8x
```

PCIe Gen4 runs 16 GT/s (billion transfers per second) per lane, which [Wikipedia's PCIe table](https://en.wikipedia.org/wiki/PCI_Express) turns into 15.754 GB/s for eight lanes, "in each direction". That is the ceiling.

The benchmark copies a buffer between host (CPU) and device (GPU) memory with `cudaMemcpyAsync`, timed with CUDA events: 3 warm-up copies, then 15 trials of up to 200 back-to-back copies (7 trials of 2 at 256 MiB), reporting the median per copy. Ordinary **pageable** host memory from `malloc` can be moved by the operating system; **pinned** (page-locked) memory from `cudaMallocHost` stays put, so the GPU's copy engine can read it directly:

```cpp
CK(cudaEventRecord(a));
for (int i = 0; i < reps; i++) CK(cudaMemcpyAsync(dst, src, n, kind));
CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms * 1000.f / reps);
```

Results on the RTX 4060 Laptop GPU (driver 592.82, CUDA 12.8, WSL2), with the SM clock at 2,190 MHz, the memory clock at 7,000 MHz and the GPU drawing 9 to 16 W (copies barely load it):

| Bytes | Host to GPU, pinned | GPU to host, pinned | Host to GPU, pageable | GPU to host, pageable | Model α + n/B | Achieved, pinned H2D |
|---|---|---|---|---|---|---|
| 4 | 8.7 µs | 6.7 µs | 10.4 µs | 24.0 µs | 8.3 µs | 0.0005 GB/s |
| 4,096 | 9.1 µs | 9.4 µs | 11.1 µs | 26.4 µs | 8.6 µs | 0.45 GB/s |
| 65,536 | 11.1 µs | 12.4 µs | 14.1 µs | 32.8 µs | 13.1 µs | 5.9 GB/s |
| 262,144 | 23.7 µs | 22.6 µs | 25.8 µs | 56.4 µs | 27.8 µs | 11.0 GB/s |
| 1,048,576 | 81 µs | 81 µs | 81 µs | 162 µs | 86 µs | 12.9 GB/s |
| 16,777,216 | 1.25 ms | 1.27 ms | 1.45 ms | 1.42 ms | 1.26 ms | 13.4 GB/s |
| 268,435,456 | 20.1 ms | 20.4 ms | 22.4 ms | 22.0 ms | 20.0 ms | 13.3 GB/s |

A fit to the pinned host-to-GPU column (least squares on the log of the times) gives `α = 8.25 µs` and `B = 13.45 GB/s`, so `n½ ≈ 111 KB`. Below about 100 KB this link is a latency device; above a few megabytes, a bandwidth device. A 4-byte copy and a 4,096-byte copy cost the same 9 µs.

The 13.4 GB/s plateau is 85% of the line rate. PCIe wraps data in packets with headers, and Wikipedia puts packet efficiency "for a 128 byte payload" at 86%; this laptop's payload size isn't visible from WSL2, so the match is only suggestive.

Three more findings:

- α depends on what you count. The 8.25 µs is the gap between back-to-back copies on the GPU's timeline; seen from the CPU, one synchronous 4-byte `cudaMemcpy` took 21 to 24 µs round trip (median of 2,000), including the driver, WSL2's virtualization layer and the wait.
- Pinned memory was only about 11% faster at large sizes (13.3 versus 12.0 GB/s). Its real benefit shows up in Step 5.
- This GPU has one copy engine for host transfers: `asyncEngineCount = 1`. NVIDIA's [CUDA 12.8 programming guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html) says copies in both directions overlap only where that count "is equal to 2". Sure enough, 128 MiB each way at once took 20.2 ms, a combined 13.3 GB/s, no more than one direction alone.

## Step 2: the ring all-reduce, derived

An **all-reduce** takes one tensor from each of `p` GPUs (ranks, in NCCL's terms), adds them element by element, and leaves the total on every GPU.

The ring version cuts every GPU's N-byte message into `p` chunks and runs two phases of `p − 1` steps. In the **reduce-scatter** phase, each GPU sends one chunk per step to its right-hand neighbor, which adds it to its own copy; at the end, each GPU holds the full sum of one chunk. In the **all-gather** phase, finished chunks travel around the ring until every GPU has all of them. This NumPy simulation counts the bytes each rank sends:

```python
def ring_allreduce(xs):
    p = len(xs)
    chunks = [np.array_split(x.copy(), p) for x in xs]       # each rank's vector, cut into p chunks
    sent = np.zeros(p, dtype=np.int64)                        # bytes sent by each rank
    for phase in ("reduce-scatter", "all-gather"):
        for step in range(p - 1):
            msgs = []                                         # all ranks send at the same time
            for r in range(p):
                c = (r - step) % p if phase == "reduce-scatter" else (r + 1 - step) % p
                msgs.append((r, (r + 1) % p, c, chunks[r][c].copy()))
            for src, dst, c, data in msgs:
                if phase == "reduce-scatter":
                    chunks[dst][c] += data                    # add the incoming partial sum
                else:
                    chunks[dst][c] = data                     # overwrite with the finished chunk
                sent[src] += data.nbytes
    return [np.concatenate(c) for c in chunks], sent
```

With four ranks of one million FP32 numbers each (4 MB), the full script, which also prints every step, gives:

```text
reduce-scatter step 1: r0->r1 chunk 0, r1->r2 chunk 1, r2->r3 chunk 2, r3->r0 chunk 3
reduce-scatter step 2: r0->r1 chunk 3, r1->r2 chunk 0, r2->r3 chunk 1, r3->r0 chunk 2
reduce-scatter step 3: r0->r1 chunk 2, r1->r2 chunk 3, r2->r3 chunk 0, r3->r0 chunk 1
all-gather     step 1: r0->r1 chunk 1, r1->r2 chunk 2, r2->r3 chunk 3, r3->r0 chunk 0
all-gather     step 2: r0->r1 chunk 0, r1->r2 chunk 1, r2->r3 chunk 2, r3->r0 chunk 3
all-gather     step 3: r0->r1 chunk 3, r1->r2 chunk 0, r2->r3 chunk 1, r3->r0 chunk 2
p=4: message 4.0 MB, each rank sent 6.00 MB = 1.500 x N, formula 2(p-1)/p = 1.500
```

The script checks that every rank ends with exactly the sum of all four inputs.

![A flowchart that follows chunk 0 through a ring all-reduce on four GPUs. In the reduce-scatter phase, GPU 0 sends its chunk 0 to GPU 1 in step 1, GPU 1 adds its own chunk 0 and sends the sum of two GPUs to GPU 2 in step 2, and GPU 2 adds its chunk 0 and sends the sum of three GPUs to GPU 3 in step 3, where GPU 3 adds its chunk 0 to get the full sum. In the all-gather phase, GPU 3 sends the full sum to GPU 0 in step 4, GPU 0 stores it and forwards it to GPU 1 in step 5, and GPU 1 forwards it to GPU 2 in step 6. A note says chunks 1, 2 and 3 move at the same time, each one GPU further on.](/diagrams/g13-ring-allreduce.svg "One chunk's trip through a four-GPU ring all-reduce. The other three chunks travel the same path at the same time, shifted by one GPU.")

Every link carries one chunk at every step, so no GPU is a bottleneck.

### The cost formula

Each step sends one chunk of `N/p` bytes and pays one latency α, so:

```text
t_ring = 2(p − 1) × α  +  2(p − 1)/p × N / B
```

The simulation's byte counts for growing `p`, against the head-teacher plan where one root receives everything, adds it and sends it back:

| GPUs p | Ring: bytes each GPU sends | One root: bytes the root sends |
|---|---|---|
| 2 | 1.000 × N | 1 × N |
| 4 | 1.500 × N | 3 × N |
| 8 | 1.750 × N | 7 × N |
| 16 | 1.875 × N | 15 × N |
| 64 | 1.969 × N | 63 × N |

The ring's bandwidth term approaches `2N/B` and stops there, which is optimal: for the ring all-gather, Thakur and colleagues note that "the bandwidth term cannot be reduced further because each process must receive" the data it doesn't have, and an all-reduce is a reduce-scatter plus an all-gather. The weakness is the latency term, which grows linearly with `p`: 14 steps for 8 GPUs, 126 for 64.

### algbw and busbw

NVIDIA's benchmark suite, nccl-tests, reports two bandwidths. Its [performance notes](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) define the **algorithm bandwidth** as "size (S) / time (t)" and the best all-reduce time as `t = (S/B) × (2(n − 1)/n)`. Multiplying algbw by that factor gives the **bus bandwidth**, so that "we can compare it with the hardware peak bandwidth, independently of the number of ranks used."

For an ideal eight-GPU ring at 450 GB/s per direction, algbw is `450 ÷ 1.75 ≈ 257 GB/s` and busbw 450 GB/s. busbw tells you how well the hardware is used; algbw tells you how long your tensor takes.

## Step 3: trees, switches and the other collectives

### Trees cut the latency

A tree all-reduce sums up a binary tree to a root, then broadcasts the result back down: about `2 log₂ p` steps instead of `2(p − 1)`, 6 instead of 14 for 8 GPUs. But in a single tree the leaves only send and the root only receives, so links sit idle.

NCCL 2.4 fixed that with **double binary trees**: per Sylvain Jeaugey's [NVIDIA post](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) (2019), each rank is "at most a node in one tree and a leaf in the other", each tree carries half the data, and together they "offer full bandwidth and a logarithmic latency." Rings, by contrast, have latency that "scales linearly with the number of GPUs"; on up to 24,576 GPUs of the Summit supercomputer, trees cut latency by "up to 180x".

In a simplified model, each GPU in the double tree sends about N going up and N coming down:

```text
t_tree ≈ 2 log₂(p) × α  +  2N / B        (ignoring pipeline fill)
t_ring =  2(p − 1) × α  +  2(p − 1)/p × N / B
```

The tree saves steps and pays slightly more bandwidth, so it wins on small messages and the ring on large ones. With `p = 8` and an illustrative `α = 1 µs` per step, they cross at `N = 32 × α × B`: about 14 MB on a 450 GB/s link, 1.6 MB on a 50 GB/s network. NCCL's [environment docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) (version 2.32) list `Ring`, `Tree`, `CollnetDirect`, `CollnetChain`, `NVLS`, `NVLSTree` and `PAT` for `NCCL_ALGO`, and by default NCCL "chooses algorithms automatically based on node topology and architecture."

### Switches that add

The `NVLS` entries are a third option: let the switch do the arithmetic. NVIDIA's [Hopper architecture post](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) says the third-generation NVSwitch "provides hardware acceleration for collective operations with multicast and NVIDIA SHARP in-network reductions" (SHARP: Scalable Hierarchical Aggregation and Reduction Protocol), with "up to 2x throughput gain", and the NCCL docs say "NVLS and NVLSTree enable NVLink SHARP offload." Each GPU sends its data into the switch about once and gets the sum back. It frees GPU cores too: NVIDIA's [NCCL 2.27 post](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27) says "traditional ring-based implementations can consume 16 or more SMs," while with NVLink and InfiniBand SHARP "this demand is reduced to 6 SMs or fewer."

### The other collectives, and who uses them

NCCL's [collectives page](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) defines the rest. A **broadcast** "copies an N-element buffer from the root rank to all the ranks". An **all-gather** "gathers N values from k ranks into an output buffer of size k*N" on every rank. An **all-to-all** sends a distinct chunk from every rank to every other. Four of them in PyTorch's `torch.distributed`, on four CPU processes with the gloo backend:

```python
x = torch.arange(4.0) + 10 * rank          # rank r holds [10r, 10r+1, 10r+2, 10r+3]
s = x.clone(); dist.all_reduce(s)                                    # sum to everyone
g = [torch.empty(4) for _ in range(world)]; dist.all_gather(g, x)    # everything to everyone
rs = torch.empty(1); dist.reduce_scatter(rs, list(x.chunk(world)))  # rank r: sum of piece r
a = torch.empty(4); dist.all_to_all_single(a, x)                    # piece j of rank r -> rank j
```

```text
rank 0: all_reduce [60.0, 64.0, 68.0, 72.0]  reduce_scatter [60.0]  all_to_all [0.0, 10.0, 20.0, 30.0]  all_gather [0.0, 1.0, 2.0, 3.0, 10.0, 11.0]...
rank 1: all_reduce [60.0, 64.0, 68.0, 72.0]  reduce_scatter [64.0]  all_to_all [1.0, 11.0, 21.0, 31.0]  all_gather [0.0, 1.0, 2.0, 3.0, 10.0, 11.0]...
rank 2: all_reduce [60.0, 64.0, 68.0, 72.0]  reduce_scatter [68.0]  all_to_all [2.0, 12.0, 22.0, 32.0]  all_gather [0.0, 1.0, 2.0, 3.0, 10.0, 11.0]...
rank 3: all_reduce [60.0, 64.0, 68.0, 72.0]  reduce_scatter [72.0]  all_to_all [3.0, 13.0, 23.0, 33.0]  all_gather [0.0, 1.0, 2.0, 3.0, 10.0, 11.0]...
```

The reduce-scatter outputs are the all-reduce's four elements, one per rank: the first half of a ring all-reduce. The all-to-all is a transpose. With the `"nccl"` backend and one GPU per process, the same code runs on GPUs.

Each kind of parallelism leans on a different collective (bus factors from nccl-tests):

| Parallelism | Collective | When | Bytes each GPU sends, per N-byte tensor |
|---|---|---|---|
| Tensor parallel (TP) | all-reduce (or reduce-scatter plus all-gather) | twice per Transformer block | `2(p − 1)/p × N` |
| Fully sharded data parallel (FSDP, ZeRO) | all-gather weights, reduce-scatter gradients | every layer, every training step | `(p − 1)/p × N` each |
| Plain data parallel, training | all-reduce of gradients | once per step | `2(p − 1)/p × N` |
| Expert parallel (EP) | all-to-all, dispatch and combine | every MoE layer | up to `(p − 1)/p × N` |
| Pipeline parallel (PP) | point-to-point send and receive | at each stage boundary | N to the next stage |

For scale: FSDP training of a 70B model in BF16 on 8 GPUs all-gathers `7/8 × 141 GB ≈ 124 GB` per GPU per forward pass, and about 371 GB per step counting the backward pass and the gradient reduce-scatter: 0.82 s at 450 GB/s, hidden by prefetching (Step 5). [Part 12](/blog/moe-kernels/) covers the kernels around expert parallelism's all-to-all.

## Step 4: the links, and what tensor parallelism costs on each

### NVLink, NVSwitch, PCIe and InfiniBand

Inside a server, NVIDIA GPUs connect with **NVLink**, a direct GPU-to-GPU link, through **NVSwitch** chips that join every GPU to every other. The H100 has "18 fourth-generation NVLink links to provide 900 GB/sec total bandwidth", each with "25 GB/sec effective bandwidth in each direction", per the Hopper post: 450 GB/s each way. Between servers, traffic goes through a network card. **InfiniBand** is a low-latency network built for clusters, and **RoCE** (RDMA over Converged Ethernet) brings the same remote memory access to Ethernet. **GPUDirect RDMA** (remote direct memory access) "enables a direct path for data exchange" between the GPU and a peer device such as a network card, per NVIDIA's [docs](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html), so data skips CPU memory; the two devices "must share the same upstream PCI Express root complex."

The numbers, per direction, which is what an all-reduce's B means:

| Link | Total, as NVIDIA lists it | Per direction | Source |
|---|---|---|---|
| NVLink 5 (B200) | 1,800 GB/s per GPU | 900 GB/s | [NVLink page](https://www.nvidia.com/en-us/data-center/nvlink/) |
| NVLink 4 (H100) | 900 GB/s per GPU | 450 GB/s | [Hopper post](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| PCIe Gen5 x16 | 128 GB/s ([H100 page](https://www.nvidia.com/en-us/data-center/h100/)) | 63 GB/s | Wikipedia's PCIe table |
| InfiniBand NDR, one ConnectX-7 port | 400 Gb/s | 50 GB/s | [NVIDIA adapters page](https://www.nvidia.com/en-us/networking/infiniband-adapters/) |
| This laptop's PCIe Gen4 x8 | not listed | 15.75 GB/s line rate, 13.45 measured | Step 1 |

Watch the units: network ports are quoted in gigabits per second, GPU links as the sum of both directions. A [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) has 8 GPUs, "14.4 TB/s aggregate" NVLink bandwidth and "8x single-port NVIDIA ConnectX-7 VPI" cards at up to 400 Gb/s: each GPU's NVLink moves 18 times what its network port does. That cliff is why the **NVLink domain**, the set of GPUs on one NVLink fabric, bounds tensor parallelism. NVIDIA's NVLink page lists NVLink 5 domains of 8 or 72 GPUs, 130 TB/s across a 72-GPU NVL72 rack.

### Worked example: Llama 3.1 70B at TP = 8

Llama 3.1 70B has a hidden size of 8,192 and 80 layers, so tensor parallelism runs 160 all-reduces per forward pass, each 16 KiB per token in BF16. Ring and tree times for one all-reduce, with `p = 8`, the per-direction bandwidths above and an illustrative `α = 1 µs` per step (the last row uses this laptop's measured α and B, as if eight such links formed a ring):

| Link | 1 token (16 KiB): ring / tree | 64 tokens (1 MiB): ring / tree | 4,096 tokens (64 MiB): ring / tree |
|---|---|---|---|
| NVLink 5, 900 GB/s | 14.0 / 6.0 µs | 16.0 / 8.3 µs | 145 / 155 µs |
| NVLink 4, 450 GB/s | 14.1 / 6.1 µs | 18.1 / 10.7 µs | 275 / 304 µs |
| PCIe Gen5 x16, 63 GB/s | 14.5 / 6.5 µs | 43 / 39 µs | 1,878 / 2,136 µs |
| InfiniBand NDR, 50 GB/s | 14.6 / 6.7 µs | 51 / 48 µs | 2,363 / 2,690 µs |
| Laptop PCIe (α 8.25 µs, B 13.45 GB/s, measured) | 118 / 52 µs | 252 / 205 µs | 8,847 / 10,029 µs |

Read the table by column:

- With one token (single-user decode), every link gives about 14 µs for the ring: 14 steps of 1 µs dwarf the 0.06 µs it takes to push 16 KiB over NVLink. Times 160, that's 2.25 ms per token for the ring and 0.97 ms for a tree, next to the roughly 5.3 ms each H100 needs to read its share of the weights ([Inference Part 10](/blog/model-parallelism/)). Cutting steps would help; extra bandwidth wouldn't.
- With 64 tokens (batched decode), NVLink is still latency-bound, while PCIe and InfiniBand have become bandwidth-bound.
- With 4,096 tokens (prefill), it's pure bandwidth and the ring beats the tree. The totals are 44 ms over NVLink 4, 300 ms over PCIe Gen5 and 378 ms over InfiniBand, against 73 ms of math on eight H100s at their 989 TFLOPS BF16 peak. A B200 needs about 32 ms of math (2.25 dense BF16 PFLOPS per GPU, from [Part 4](/blog/gpu-profiling-roofline/)) and 23 ms of NVLink 5 all-reduce: both roughly halved, so talking still takes 60% to 70% as long as computing.

The useful number for any link is where the ring's two terms meet, `N = p × α × B`: 3.6 MB on NVLink 4 and 0.4 MB on InfiniBand with α = 1 µs, 0.9 MB on the laptop's link. Below that, a message is a latency problem, and small batches get only a fraction of the rated bandwidth.

## Step 5: hiding communication behind compute

A GPU has separate hardware for moving data (copy engines, NVLink and network traffic) and for math (the SMs). If a transfer doesn't depend on the current computation, both can run at once, on separate CUDA streams, the ordered work queues from [Part 1](/blog/gpu-from-python-to-silicon/).

### Measured: one copy behind one compute

The test puts a 128 MiB host-to-GPU copy on a side stream while the default stream runs two 4096 × 4096 BF16 matmuls (275 GFLOP, billion floating-point operations):

```python
def overlapped(host):
    side.wait_stream(main)                       # start after earlier work on main
    with torch.cuda.stream(side):
        dst.copy_(host, non_blocking=True)       # copy engine
    compute()                                    # SMs, at the same time
    main.wait_stream(side)                       # main's later work waits for the copy
```

Timed with CUDA events after 3 warm-up runs, median of 11, SM clock 1,620 to 2,175 MHz, memory at 7,000 or 8,000 MHz, up to 44.6 W under a 50 W power limit:

| Case | Time |
|---|---|
| Copy alone (pinned, 13.4 GB/s) | 10.02 ms |
| Compute alone (22.2 TFLOPS) | 12.38 ms |
| Copy then compute, one stream | 20.96 ms |
| Copy and compute on two streams, pinned memory | 12.27 ms |
| Same, pageable memory | 21.52 ms |
| Pageable, but compute queued before the copy | 12.01 ms |

With pinned memory the copy disappeared: two streams took no longer than the compute alone. With pageable memory the overlap vanished, and the reason is on the CPU. A `non_blocking` copy from pageable memory kept the CPU busy for a median of 10.74 ms, so the matmuls weren't queued until the copy was nearly done. NVIDIA's [programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) states the rule: "In order for memory copies involving CPU memory to be carried out asynchronously, the host buffers must be pinned and page-locked." Queuing the compute first restored the overlap, which puts the blame on the blocking call and clears the link.

### Measured: prefetching the next layer

Frameworks use this to fetch the next layer's weights while the current layer computes. The test keeps eight layers of 4096 × 4096 BF16 weights (32 MiB each) in pinned CPU memory and multiplies a 2048 × 4096 activation through them, alternating between two GPU buffers, with events so a buffer is never overwritten while in use or read before it's filled:

| Case | Time (median of 9) |
|---|---|
| One layer's copy | 2.50 ms |
| One layer's matmul (24.8 TFLOPS) | 2.77 ms |
| 8 layers, copy then compute on one stream | 42.76 ms |
| 8 layers, next layer prefetched on a second stream | 27.77 ms |
| Ideal overlap: first copy plus 8 matmuls | 24.66 ms |

Prefetching saved 35%, within 3.1 ms of perfect overlap, with bit-identical output. FSDP does the same with all-gathers. PyTorch's [FSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) says "All-gathers are queued into its own cuda stream while layer i computation happens in the default stream," so "all-gather i+1 can overlap with computation for layer i."

### Overlap inside tensor parallelism

Tensor parallelism is harder, because each all-reduce depends on the matmul just before it. PyTorch's [async tensor parallelism](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) (Wang et al., 2024) splits an all-gather followed by a matmul, and a matmul followed by a reduce-scatter, into chunks: "by decomposing dependent communication and computation operators, we can create overlapping opportunities." Its transfers use SymmetricMemory, "a buffer symmetrically allocated across a group of devices", and are "handled by copy engines", so they don't take SMs from the matmuls. On 64 H100s it reported up to about 20% faster forward passes and 8% faster end-to-end training for Llama 3 70B.

Communication kernels that run on SMs slow the math beside them. The [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) found "only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink," built its DualPipe schedule "to overlap the computation and communication," and asked for "hardware that offloads these communication tasks."

### GPU-initiated communication: NVSHMEM and its successors

So far the CPU has launched every transfer. Fine-grained patterns such as expert dispatch would rather have GPU threads send data the moment it's ready. [NVSHMEM](https://developer.nvidia.com/nvshmem) is NVIDIA's library for that: "a parallel programming interface based on OpenSHMEM" with an "in-kernel communication API for use by GPU threads" and "GPU-initiated data transfers, eliminating synchronization overheads between the CPU and the GPU." Its key idea is **symmetric memory**: `nvshmem_malloc` is a collective call whose buffer "is allocated from the symmetric heap" ([API docs](https://docs.nvidia.com/nvshmem/archives/nvshmem-290/api/gen/api/memory.html)), so every GPU holds a matching buffer that the others can address directly.

The idea has moved into NCCL itself. "Starting with version 2.28, NCCL provides a device-side communication API" ([docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)) for "communication primitives directly from user CUDA kernels", over NVLink, NVLink SHARP multicast, or the network through GIN (GPU-Initiated Networking). DeepSeek's [DeepEP](https://github.com/deepseek-ai/DeepEP), the expert-parallel all-to-all library from Part 12, followed: V2 added "the lightweight NCCL Gin backend," and V2.5 removed the old "NVSHMEM backend" entirely.

## In the real world

- An H100 server joins eight GPUs through NVSwitch at "7x the bandwidth of PCIe Gen 5" (Hopper post), and the NVLink Switch System "supports up to 256 GPUs."
- In a GB200 NVL72 rack, 72 GPUs share one NVLink 5 domain at 130 TB/s, so expert-parallel all-to-all can stay on NVLink instead of 50 GB/s InfiniBand.
- DeepSeek-V3's report gives the H800's NVLink as "160 GB/s, roughly 3.2 times that of IB (50 GB/s)", and limits "each token to be dispatched to at most 4 nodes, thereby reducing IB traffic", a routing rule set by the network.
- NVIDIA's [NCCL 2.27 post](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27) reports "up to 9x reduction in latency for small message sizes" from symmetric memory kernels within one NVLink domain: the latency-bound decode all-reduces of Step 4.
- Disaggregated serving ([Inference Part 12](/blog/disaggregated-serving-llm-d/)) moves KV caches between machines over RDMA. That's point-to-point traffic, priced by the same latency-bandwidth arithmetic.

## Common misconceptions

- **"An all-reduce on more GPUs sends more data per GPU."** Each GPU sends `2(p − 1)/p × N`: 1.5 N at 4 GPUs, only 1.97 N at 64. What grows with `p` is the number of steps.
- **"A faster link makes decode all-reduces faster."** A 16 KiB all-reduce over NVLink spends 0.06 µs on bandwidth and about 14 fixed-latency steps. Fewer steps help; more GB/s barely moves it.
- **"NVLink's 900 GB/s means a 900 MB tensor crosses in a millisecond."** 900 GB/s is the total of both directions over 18 links; one direction is 450 GB/s, and an eight-GPU ring all-reduce of that tensor would need `1.75 × 900 MB ÷ 450 GB/s = 3.5 ms` at best.
- **"`non_blocking=True` makes a copy asynchronous."** Only from pinned memory. From pageable memory the CPU stayed blocked for 10.7 ms of an 11 ms copy, and a copy that should have been hidden added 9 ms.

## Key takeaways

- Every transfer costs a fixed latency plus bytes over bandwidth. On this laptop's PCIe link that was 8.25 µs plus 13.45 GB/s, so anything under about 100 KB is latency-bound.
- A ring all-reduce sends `2(p − 1)/p` times the message per GPU, the minimum possible, but takes `2(p − 1)` steps. Trees take about `2 log₂ p`, and NVLink SHARP lets the switch do the adding.
- Tensor parallelism's all-reduces are latency-bound in decode and bandwidth-bound in prefill. A 70B prefill on eight H100s spends about 44 ms on NVLink all-reduces against 73 ms of math, and four to five times the math over PCIe or InfiniBand.
- NVLink moves 450 to 900 GB/s each way per GPU; a 400 Gb/s network port moves 50 GB/s. That's why tensor parallelism stays inside an NVLink domain.
- With pinned memory and two streams, a 10 ms copy vanished behind 12 ms of matmuls, and prefetching cut eight layers from 42.8 to 27.8 ms. FSDP and async tensor parallelism use the same overlap for their collectives.

## What's next

Everything in this series so far has run on NVIDIA hardware and software. Part 14 looks at the alternatives: AMD's ROCm and MI-series GPUs, Apple's Metal and MLX, Google's TPUs and XLA, AWS Trainium, Groq, Cerebras and others, and how the same ideas, from memory bandwidth to collectives, map onto each.

## Further reading

- [NCCL User Guide (NVIDIA)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html): the [collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html), [environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) such as `NCCL_ALGO`, and the [device-initiated API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html).
- [nccl-tests performance notes (NVIDIA)](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md): algbw, busbw and the ideal time of each collective.
- [Optimization of Collective Communication Operations in MPICH (Thakur, Rabenseifner and Gropp, 2005)](https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf): the α + nβ cost model and ring, tree and recursive algorithms, with their costs.
- [Massively Scale Your Deep Learning Training with NCCL 2.4 (Sylvain Jeaugey, NVIDIA, 2019)](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/): double binary trees and why rings stop scaling.
- [NVIDIA Hopper Architecture In-Depth (NVIDIA, 2022)](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/): NVLink 4, NVSwitch, SHARP in-network reductions and the NVLink Switch System.
- [NVLink and NVLink Switch (NVIDIA)](https://www.nvidia.com/en-us/data-center/nvlink/): bandwidth per GPU and per domain for each generation.
- [GPUDirect RDMA (NVIDIA)](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html): how network cards read and write GPU memory directly.
- [Enabling Fast Inference and Resilient Training with NCCL 2.27 (NVIDIA, 2025)](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27): symmetric memory kernels, small-message latency and SM usage.
- [NVSHMEM (NVIDIA)](https://developer.nvidia.com/nvshmem): GPU-initiated communication over a symmetric heap.
- [DeepEP (DeepSeek)](https://github.com/deepseek-ai/DeepEP): expert-parallel all-to-all kernels, now built on NCCL's GPU-initiated networking.
- [Introducing Async Tensor Parallelism in PyTorch (Wang et al., 2024)](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) and [PyTorch's FSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html): overlapping collectives with compute in practice.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): DualPipe, node-limited routing and dedicating 20 SMs to communication.
- [CUDA C++ Programming Guide 12.8 (NVIDIA)](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html): page-locked memory, copy engines and overlapping transfers with kernels.
