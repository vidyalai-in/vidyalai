---
title: "Build an LLM From Scratch: Fitting Bigger Models in Memory"
description: "Where 8 GB goes when training the 88M model, predicted and measured; activation checkpointing, 8-bit AdamW and offload timed; ZeRO and FSDP explained."
slug: "training-memory"
category: "llm-training"
tags: ["training", "gpu", "fine-tuning", "lora", "cost"]
series: "llm-from-scratch"
seriesOrder: 14
pubDate: 2026-10-29
---

## TL;DR

Training the 88M model with Part 6's settings peaks at 6.75 GiB, and only 1.49 GiB of it is weights, gradients, AdamW state and autocast's BF16 weight copies. The rest is activations, predicted here line by line from `model.py` with an adapted version of Korthikanti et al.'s `sbh(34 + 5as/h)` formula and checked against PyTorch's memory snapshots to within 0.6%. Checkpointing every block cut them by 72% for 22% more time per token, less than the 33% rule of thumb. With 8-bit AdamW and BF16 weights on top, this 8 GB laptop GPU fits a 1-billion-parameter model; what limits it is arithmetic, at 2,347 tokens per second. ZeRO and FSDP split the same state across GPUs, shown here with two CPU processes.

## Why it matters

The 88M model's weights take 336 MiB. Training it with Part 6's settings peaks at 6.75 GiB, twenty times the weights, and the same 179 lines without `torch.compile` would need 10 GiB for the same batch, more than the card holds. On one GPU, memory is usually the first wall a bigger model hits.

[Part 1](/blog/build-llm-from-scratch/) counted 16 bytes per parameter for weights, gradients and AdamW, and concluded that a 1-billion-parameter model "would fill two of these GPUs before a single activation." This part checks the whole bill against PyTorch's allocator and measures the tools that shrink it: checkpointing, 8-bit AdamW, BF16 weights, CPU offload and LoRA on one GPU, then ZeRO and FSDP across many. [Math Part 8](/blog/backpropagation-math/) derived checkpointing on paper and [Fundamentals Part 12](/blog/pretraining-scaling-laws/) tabled ZeRO; this part measures both on the laptop.

## The intuition

Picture a hiker who must later retrace the same trail turn by turn. The return trip is the backward pass, and the notes about each turn are the activations. Activation checkpointing keeps notes only at each mile post; on the way back, the hiker re-walks each mile from its marker to rebuild them: a lighter pack, some miles walked twice. ZeRO has four hikers each carry a quarter of the shared maps, copying pages from the others as needed.

The analogy breaks in two places. A pack's weight is a sum, but GPU memory is handed out in blocks, and free space in the wrong place can't hold the next request. And only the blocks get re-walked: the output layer, a quarter of this model's arithmetic, never is.

## Step 1: the memory bill, item by item

Training memory has two parts. The model's state, weights, gradients and optimizer state, is fixed by the parameter count. Activations, the intermediate results the forward pass keeps for the backward pass, grow with every token in the micro-batch, the group of sequences that goes through one forward and backward pass together.

### Weights, gradients and AdamW: 16 bytes per parameter, plus 2

[Part 5](/blog/llm-training-loop/)'s `train.py` keeps an FP32 weight, an FP32 gradient and AdamW's two FP32 running averages (its moments) per parameter. Under BF16 autocast there's a fifth item during the step: each matrix multiply casts its weight to BF16 and keeps that copy for the backward pass. For the 88,096,512-parameter model:

| Item | Bytes per parameter | 88M model |
|---|---|---|
| FP32 weights | 4 | 336.1 MiB |
| FP32 gradients | 4 | 336.1 MiB |
| AdamW moments, FP32 | 8 | 672.1 MiB |
| BF16 copies of the weights, during forward and backward | 2 | 168.0 MiB |
| Total during a step | 18 | 1.48 GiB |

The ZeRO paper (Step 5) splits the same 16 bytes differently: 16-bit weights and gradients, plus 12 bytes of FP32 master weights and moments.

### Activations: Korthikanti's formula, and what our model changes

Korthikanti et al. (2022) count the activations of one GPT-3-style layer in bytes, assuming 16-bit activations and 1-byte dropout masks ([Math Part 8](/blog/backpropagation-math/) walks through it):

```text
Activations memory per layer = sbh(34 + 5as/h)
```

Here `s` is the sequence length, `b` the micro-batch, `h` the width and `a` the attention heads; `5as/h` is attention's stored `s × s` scores and the 34 everything else. For our `s = 1,024`, `h = 768` and `a = 12`, `5as/h = 80`, so the formula predicts 114 × 768 = 87,552 bytes per token per layer.

Our block changes five terms. FlashAttention, run by `scaled_dot_product_attention`, never stores the scores: the 80 becomes one FP32 log-sum-exp per head. There's no dropout. Grouped-query attention makes K and V a third of `h`. SwiGLU keeps three or four tensors of width 2,048 instead of GeLU's two of 3,072. And the residual stream is FP32, because the embedding table is, so RMSNorm's inputs and outputs take 4 bytes. Counting what each line of `model.py` keeps, in bytes per token per layer divided by `h`:

| Saved for the backward pass | GPT-3 layer (Korthikanti) | Ours, eager | Ours, compiled |
|---|---|---|---|
| Norm inputs and outputs | 4 | 16 (FP32 in and out) | 8 (FP32 inputs) |
| Inputs to the Q/K/V and MLP projections | 4 | 10 (a BF16 copy per linear layer) | 4 |
| Q, K and V | 6 | 3.33 | 3.33 |
| Attention scores, softmax, dropout | 80 | 0.06 (log-sum-exp) | 0.06 |
| Attention output and its projection's input | 2 | 4 | 4 |
| Dropout masks | 2 | 0 | 0 |
| MLP hidden activations | 16 | 21.33 | 16 |
| Total, per unit of width | 114 | 54.7 | 35.4 |
| Bytes per token per layer | 87,552 | 42,032 | 27,184 |

"Eager" is PyTorch's default, each operation run as it comes. A hook on autograd's saved tensors (`torch.autograd.graph.saved_tensors_hooks`) counted 42,050 bytes per token per layer in an eager forward pass, 18 bytes off. The compiled column is what `torch.compile`'s graph kept, read from Step 2's snapshot: one BF16 copy of each norm's output, no SiLU output, norms recomputed in the backward pass. FlashAttention alone removes 70% of the formula's total, and after that our compiled layer keeps about what the formula's 34 predicts.

The output layer matters more here than in the paper. Its term, `4sbh(1 + v/h)` with `v` the vocabulary, is under 0.01% for the paper's 22B model; our `v/h` is 42.7. The eager loss keeps an FP32 log-softmax of 32,768 numbers per token, 128 KiB, and the compiled graph the BF16 logits, 64 KiB: in measured memory, as much as two and a half to three blocks. At 16 sequences the compiled activations come to 5.22 GiB and the eager ones to 8.53 GiB, so eager training would need 10.0 GiB, the arithmetic behind [Part 4](/blog/write-gpt-model/)'s finding that eager mode fits only 4 sequences.

## Step 2: checking it against the allocator

PyTorch can record every GPU allocation with the Python stack that made it. The benchmark script records one optimizer step of four micro-batches:

```python
# bench.py (excerpt)
torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
step()                                                    # one optimizer step, 4 micro-batches
pickle.dump(torch.cuda.memory._snapshot(), open("step.pickle", "wb"))
```

A short script replays the allocations in order and labels each live block by the line that made it; compiled code still carries its `model.py` line and each buffer's shape and dtype. Gradients are the blocks whose addresses match `p.grad`. Settings are Part 6's: BF16 autocast, fused AdamW, 1,024-token sequences. All runs in this part are on the NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) under WSL2, driver 592.82, PyTorch 2.11.0 with CUDA 12.8, Triton 3.6.0, seed 1337; median SM clocks ran 1,695 to 2,175 MHz, the memory clock 7,000 MHz with occasional 5,870 and 8,000, and the power limit drifted between 40.1 and 50 W.

| Component | Eager, 4 sequences: predicted | Measured | Compiled, 16 sequences: predicted | Measured |
|---|---|---|---|---|
| FP32 weights | 336 MiB | 336 MiB | 336 MiB | 336 MiB |
| AdamW state | 672 MiB | 693 MiB | 672 MiB | 688 MiB |
| FP32 gradients | 336 MiB | 336 MiB | 336 MiB | 336 MiB |
| BF16 weight copies | 168 MiB | 168 MiB | 168 MiB | 168 MiB |
| Activations, 10 blocks | 1,642 MiB | 1,666 MiB | 4,248 MiB | 4,225 MiB |
| Output layer and loss | 542 MiB | 518 MiB | 1,096 MiB | 1,096 MiB |
| Total before the 2nd backward pass | 3.61 GiB | 3.63 GiB | 6.70 GiB | 6.69 GiB |
| Backward temporaries at the peak | | 1.03 GiB | | 0.06 GiB |
| Peak allocated (`max_memory_allocated`) | | 4.66 GiB | | 6.75 GiB |
| Peak reserved (`max_memory_reserved`) | | 5.01 GiB | | 6.91 GiB |

Just before the second micro-batch's backward pass, with gradients present and activations complete, prediction and measurement agree within 0.6% eager and 0.1% compiled; the 16 to 21 MiB extra under AdamW are other buffers that outlive a step, which the replay can't name.

![Horizontal stacked bars of GPU memory in GiB at a training step's peak for the 88M model, split into FP32 weights, AdamW state, FP32 gradients, BF16 weight copies, activations of the 10 blocks, output layer and loss, backward temporaries and allocator cache, with dashed lines at 7.56 GiB usable and 8.0 GiB on the card. Eager with 4 sequences: predicted 3.61, measured 5.01 including 1.03 of backward temporaries and 0.35 of allocator cache. Compiled with 16 sequences: predicted 6.69, measured 6.91, with tiny temporaries and 0.16 of cache. Eager with 16 sequences, predicted: 10.01, past both lines, with 6.4 GiB of block activations and 2.1 of output layer.](/diagrams/t14-memory-breakdown.svg "Predicted and measured memory for one training step of the 88M model. Measured bars end at the peak reserved memory.")

The predictions miss only the backward pass's temporaries. In eager mode the loss's backward briefly holds two FP32 tensors of 134 million logits, 1.03 GiB; the compiled loss kernel computes its gradient in one fused pass. State and weight copies are 1.49 GiB of the compiled 6.75; activations are three quarters of the bill.

### What 8 GB means here

The card has 8,188 MiB. A probe that allocated 256 MiB blocks and timed a fill of each ran at about 150 GB/s up to 7,680 MiB, then 14.9 and 7 GB/s: WSL2's driver had started placing memory in the computer's main memory, across PCIe, without an error. `nvidia-smi` charged 107 MiB to the CUDA context, the driver's per-process state, and topped out at 7,851 MiB. So about 7.56 GiB is usable, and every benchmark here caps PyTorch's allocator at 7.45 GiB with `torch.cuda.set_per_process_memory_fraction`, so a configuration that doesn't fit fails instead of running ten times slower.

### Reserved versus allocated

PyTorch's caching allocator keeps freed blocks for reuse; in its documentation's words, "the unused memory managed by the allocator will still show as if used in nvidia-smi." `max_memory_reserved` counts what it holds, `max_memory_allocated` what tensors use, and the gap is cache plus fragmentation: free pieces too small or in the wrong place for the next request. Compiled at 16 sequences it was 157 MiB. With every block checkpointed at 40 sequences it was 2.17 GiB, 5.28 GiB of tensors in the full 7.45 GiB.

## Step 3: activation checkpointing, measured

Activation checkpointing keeps only each chosen block's input during the forward pass and reruns the block's forward during the backward pass to rebuild what the gradients need. PyTorch's [`torch.utils.checkpoint`](https://docs.pytorch.org/docs/2.11/checkpoint.html) calls it "a technique that trades compute for memory." The new module `memory.py` wraps the blocks of an unchanged `model.py`:

```python
# memory.py (excerpt)
_KEEP = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
         torch.ops.aten._scaled_dot_product_flash_attention.default,
         torch.ops.aten._scaled_dot_product_efficient_attention.default,
         torch.ops.aten._scaled_dot_product_cudnn_attention.default}


def _keep_matmuls(ctx, op, *args, **kwargs):
    """Selective policy: store what the matmuls and attention produce, recompute everything in between."""
    return CheckpointPolicy.MUST_SAVE if op in _KEEP else CheckpointPolicy.PREFER_RECOMPUTE


def _recompute(forward, context_fn, *args):
    if context_fn is None:
        return checkpoint(forward, *args, use_reentrant=False)
    return checkpoint(forward, *args, use_reentrant=False, context_fn=context_fn)


def checkpoint_blocks(model, every=1, selective=False):
    """Make every `every`-th block keep only its input and rerun its forward during the backward pass.
    selective=True keeps the matmul and attention outputs too and reruns only the cheap ops between them."""
    context_fn = functools.partial(create_selective_checkpoint_contexts, _keep_matmuls) if selective else None
    for i, block in enumerate(model.blocks):
        if i % every == 0:
            block.forward = functools.partial(_recompute, block.forward, context_fn)
    return model
```

`checkpoint_blocks(model)` before `torch.compile(model)` checkpoints every block, `every=2` every other one. The selective policy keeps the matmul and attention outputs and recomputes only the norms, RoPE and SwiGLU's elementwise math, cheap operations in the spirit of Korthikanti et al.'s "selective activation recomputation."

Each row is the 88M model compiled, 16 sequences per micro-batch, 4 per optimizer step, median of 6 timed steps. The clock drifts by more than some of these effects ([Part 6](/blog/fast-llm-training/)), so the cost column compares tokens per second per GHz of median SM clock:

| Checkpointing | Peak allocated | Tokens/s | SM clock, median | Extra time per token | Largest micro-batch that fits |
|---|---|---|---|---|---|
| None | 6.75 GiB | 31,276 | 1,950 MHz | | 17 |
| Selective: keep matmul and attention outputs | 5.11 GiB | 30,489 | 1,980 MHz | 4.2% | 22 |
| Every other block | 4.83 GiB | 27,861 | 1,995 MHz | 14.9% | 24 |
| Every block | 2.97 GiB | 26,455 | 2,010 MHz | 21.9% | 40 (48 with expandable segments) |

Checkpointing every block cut everything above the 1.49 GiB of state and weight copies from 5.26 to 1.48 GiB, 72% less. Most of what's left is the output layer's 1 GiB of BF16 logits, which no block checkpoint touches.

### The cost, against the 33% rule

The usual rule of thumb, in the ZeRO paper's words, is "33% re-computation overhead": one extra forward pass on a step that costs three. Here the arithmetic predicts less: rerunning all 10 blocks adds `2 × 62.9M` FLOPs per token for their weights plus 31.5M for attention's score products, 157M on Part 6's 622.95M, 25.3%, because the output layer is never rerun. Measured, every block cost 21.9% and every other block 14.9%, against 25.3% and 12.6% predicted. The selective policy reruns almost no FLOPs yet cost 4.2%: elementwise kernels are limited by memory traffic.

Their largest micro-batches ran between 3% slower and 5% faster per GHz than 16 sequences did, so checkpointing buys room for a bigger model and no speed.

### Where checkpointing stopped: one 3 GiB tensor

With every block checkpointed, 48 sequences failed while allocating 3.00 GiB, exactly the BF16 logits of 48 × 1,024 tokens times 32,768. The allocator held 2.34 GiB in tensors and 3.30 GiB reserved but free, in pieces none of which could take 3 GiB. With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which lets the allocator "create a segment initially and then expand its size later when more memory is needed," the same run fit at 6.04 GiB, so the limit at 48 was fragmentation.

## Step 4: the other levers on one GPU

The other levers shrink the state. The setup is Step 3's:

| Lever | GPU bytes per parameter | Peak allocated | Saved | Tokens/s | Speed per GHz vs baseline |
|---|---|---|---|---|---|
| Baseline: FP32 weights and AdamW, BF16 autocast | 16, plus 2 for BF16 copies | 6.75 GiB | | 31,276 | |
| 8-bit AdamW (bitsandbytes 0.50.2) | 10, plus 2 | 6.28 GiB | 0.47 GiB | 30,899 | −0.8% |
| Pure BF16: weights, gradients and AdamW moments | 8 | 5.52 GiB | 1.23 GiB | 32,480 | +6.3% |
| AdamW state offloaded to the CPU, FP32 weights on the GPU | 8, plus 2 | 6.10 GiB | 0.65 GiB | 27,439 | −11.6% |
| Same, BF16 weights on the GPU, FP32 master copy on the CPU | 4 | 5.19 GiB | 1.56 GiB | 27,417 | −10.3% |
| LoRA, rank 8 on every linear layer, FP32 frozen weights | 4 for the frozen model | 5.73 GiB | 1.02 GiB | 32,894 | −2.0% |
| LoRA, rank 8, BF16 frozen weights | 2 for the frozen model | 4.93 GiB | 1.82 GiB | 33,968 | +4.6% |

![Scatter plot of speed per GHz relative to the baseline, from 80% to 110%, against peak memory allocated, 3 to 7 GiB, for the 88M model at 16 sequences per micro-batch. The baseline sits at 6.75 GiB and 100%. Checkpointing points in orange: selective at 5.11 GiB and 96%, every other block at 4.83 and 87%, every block at 2.97 and 82%. State levers in blue: 8-bit AdamW at 6.28 and 99%, pure BF16 at 5.52 and 106%, CPU offload of AdamW at 6.10 and 88%, CPU offload with BF16 weights on the GPU at 5.19 and 90%. LoRA in purple: FP32 base at 5.73 and 98%, BF16 base at 4.93 and 105%.](/diagrams/t14-techniques.svg "Memory against clock-adjusted speed for each lever on the 88M model. Up and to the left is better.")

Only checkpointing gets below 4 GiB, and only pure BF16 and LoRA on BF16 weights also ran faster than the baseline. The five full-training runs tracked the baseline's losses within 0.01. 8-bit AdamW (bitsandbytes' `AdamW8bit`) stores both moments in 8 bits with block-wise quantization, one FP32 scale per block of 256 values in version 0.50.2 (the paper used 2,048), so an outlier only coarsens its own block: 2.03 bytes per parameter instead of 8, the 0.47 GiB saved.

### BF16 weights, or an FP32 master copy

Dropping the FP32 copy halves the state and makes the residual stream BF16, which halves its saved activations: 1.23 GiB saved and a faster step. The catch, from [Math Part 14](/blog/floating-point-math/): a BF16 weight can't absorb an update smaller than about 1/256 of itself. The 22M model on the same 19.7 million tokens (300 steps of 65,536, Part 5's schedule, peak rate 1.5 × 10⁻³):

| Precision | Validation loss at step 100 | Step 200 | Step 300 | Peak memory |
|---|---|---|---|---|
| FP32 weights and AdamW (mixed precision), run 1 | 6.507 | 5.982 | 5.796 | 2.69 GiB |
| Same, run 2 | 6.561 | 6.030 | 5.852 | 2.69 GiB |
| FP32 weights, 8-bit AdamW | 6.572 | 6.045 | 5.856 | 2.56 GiB |
| Everything BF16, run 1 | 6.621 | 6.085 | 5.924 | 2.35 GiB |
| Same, run 2 | 6.583 | 6.103 | 5.936 | 2.35 GiB |

Identical mixed-precision runs ended 0.057 apart, from nondeterministic GPU sums (Part 5). Both BF16 runs ended 0.07 to 0.14 worse than both mixed runs and were behind at every evaluation from step 100 on. 8-bit AdamW landed at the edge of the mixed runs' spread. For pretraining, the master copy is worth its 4 bytes; 8-bit moments look nearly free.

### Gradient accumulation

Accumulating 4 or 32 micro-batches peaked at the same 6.75 GiB; one per step peaked at 6.42, since no gradients exist during its forward pass. Under whole-model compilation, accumulation has a hidden cost: the compiled backward returns all new gradients together before they're added to the stored ones. In a 692M-parameter BF16 run with every block checkpointed, the second backward pass peaked 1.86 GiB above its end state against 0.56 for the first, a difference equal to the model's 1.30 GiB of gradients, and ran out of memory. Compiling each block separately (`block.compile()`) limits that to one block.

### Offloading the optimizer to the CPU

`memory.CPUAdamW` is ZeRO-Offload's split done by hand: the GPU keeps the weights it computes with and their gradients, while FP32 master weights and moments live in CPU memory and the update runs on the 16 CPU cores.

```python
    # memory.py, class CPUAdamW (excerpt)
    @torch.no_grad()
    def step(self):
        for buf, p in zip(self.bufs, self.gpu):
            buf.copy_(p.grad, non_blocking=True)                 # gradient: GPU -> pinned CPU buffer
        torch.cuda.synchronize()
        for c, buf in zip(self.cpu, self.bufs):
            c.grad = buf.float()                                 # FP32 gradient for the FP32 update
        self.opt.step()
        for buf, c, p in zip(self.bufs, self.cpu, self.gpu):
            buf.copy_(c)                                         # new FP32 weights -> GPU dtype, same buffer
            p.copy_(buf, non_blocking=True)                      # pinned CPU -> GPU
```

With FP32 weights on the GPU, its losses matched the baseline's to 0.0005. Each update took 0.27 to 0.44 s against 18 ms on the GPU: two PCIe copies of 336 MiB at the 13.45 GB/s [GPU Part 13](/blog/nccl-collectives/) measured, about 52 ms, plus CPU arithmetic. That's 10% to 12% of a 4-micro-batch step but would be under 2.5% of [Part 7](/blog/pretraining-run/)'s 17.5 s step: offload pays when updates are rare. Host memory is the limit: CPU state and buffers took the process from 1.22 to 3.27 GiB of RAM, on a 15 GB machine.

### LoRA for fine-tuning

LoRA (low-rank adaptation) freezes the model and trains a pair of small matrices beside each linear layer, here 1,085,440 parameters, 1.2% of the model. Frozen weights need no gradients or AdamW state: 1.02 GiB saved, 1.82 GiB with them (and so the residual stream) in BF16. Activations stay, since gradients still flow back through every layer to the adapters; the [LoRA fine-tuning guide](/blog/lora-fine-tuning-5k-examples/) works the budget for an 8B model.

### The biggest model that fits

Scaling width and depth up from the 88M model's family (heads of 64, three query heads per key-value head, `d_ff = 8d/3`), each setting was pushed until the next size ran out of memory, at 4 sequences per micro-batch and 4 micro-batches per step. From 554M up, the runs also compiled each block separately and used `memory.ChunkedLossGPT`, which computes the loss 2,048 tokens at a time inside a checkpoint so the full logits never exist; the two largest used 512-token chunks and expandable segments. (Compiled as a whole, the 88M model used more memory with the chunked loss: 4.54 against 2.97 GiB.)

| Setting | Largest that fit | Peak allocated | Tokens/s | MFU | Next size tried |
|---|---|---|---|---|---|
| FP32 AdamW, no checkpointing | 208M (width 1,152, 12 layers) | 5.74 GiB | 13,487 | 75% | 264M: out of memory |
| + checkpoint every block | 352M (1,536, 12) | 6.87 GiB | 7,197 | 67% | 403M: out of memory |
| + 8-bit AdamW | 554M (1,536, 20) | 6.70 GiB | 3,705 | 54% | 654M: out of memory |
| Pure BF16, checkpoint every block | 849M (1,920, 20) | 7.10 GiB | 2,983 | 66% | 1,007M: out of memory, tried with 2,048-token chunks |
| BF16 weights, 8-bit AdamW, checkpoint every block | 1,007M (1,920, 24) | 6.49 GiB | 2,347 | 61% | 1,208M: out of memory, tried with 2,048-token chunks |
| AdamW offloaded to the CPU, BF16 on the GPU | under 352M | | | | 352M and 453M: killed at the 8 GiB host-memory cap |
| LoRA rank 8, BF16 frozen weights, checkpointing (fine-tuning) | 1,435M (2,304, 24), not pushed further | 4.05 GiB | 1,984 | | |

MFU (model FLOPs utilization) counts only the model's own arithmetic, Part 6's `6N + 12·L·H·Q·T` per token, against the measured 25.3 TFLOPS. Memory lets this laptop train a 1-billion-parameter model; arithmetic stops it. Part 1's 20 tokens per parameter would take 3.6 days at 208M, 11 at 352M and 99 at 1B, against 17 hours for Part 7's 88M run, so a few hundred million parameters is the practical pretraining limit. For fine-tuning, the 1B model handles 10 million tokens in 71 minutes. The ladder is also an order of preference: checkpointing costs only time, 8-bit moments almost nothing, dropping the master copy costs loss, and offload needs host memory this machine lacks.

## Step 5: ZeRO and FSDP, sharding across GPUs

Plain data parallelism, where each GPU trains a full copy on different data and averages gradients with an all-reduce ([Inference Part 10](/blog/model-parallelism/)), keeps all 16 bytes per parameter on every GPU. The ZeRO paper (Rajbhandari et al., 2020) counts 16-bit weights and gradients, "2Ψ and 2Ψ bytes respectively," for Ψ parameters, plus FP32 weights, momentum and variance with K the "memory multiplier of the optimizer states": "Mixed-precision Adam has K = 12. In total, this results in 2Ψ + 2Ψ + KΨ = 16Ψ bytes of memory requirement." ZeRO (Zero Redundancy Optimizer) removes the copies in three stages, each sharding one more item across the `N` data-parallel GPUs:

![A top-to-bottom flow of four boxes. Plain data parallelism: 2 psi plus 2 psi plus 12 psi equals 16 psi bytes per GPU; a 7.5B model on 64 GPUs needs 120 GB each. An arrow labeled shard Adam state, same traffic, leads to ZeRO stage 1: 4 psi plus 12 psi over N bytes, 31.4 GB. An arrow labeled shard gradients, reduce-scatter them, leads to ZeRO stage 2: 2 psi plus 14 psi over N bytes, 16.6 GB. An arrow labeled shard weights, all-gather per layer, 1.5 times the traffic, leads to ZeRO stage 3 or FSDP: 16 psi over N bytes, 1.9 GB.](/diagrams/t14-zero-stages.svg "ZeRO's three stages, with the paper's 7.5B-parameter example on 64 GPUs.")

The example is the paper's: a 7.5B model "requires 31.4GB of memory" with optimizer state sharded 64 ways, "while requiring 120 GB with standard DP," then "only 16.6 GB" and "1.9 GB"; `memory.zero_bytes()` reproduces all four. In [GPU Part 13](/blog/nccl-collectives/)'s terms, stages 1 and 2 swap the gradient all-reduce for a reduce-scatter (each GPU gets the summed gradients of its slice) plus an all-gather of updated weights, "exactly the same as the baseline DP." Stage 3 also all-gathers each layer's weights before its forward and backward passes, "1.5x compared to the baseline."

### FSDP2 in PyTorch

PyTorch's stage 3 is FSDP (Fully Sharded Data Parallel); its current version, FSDP2, is [`fully_shard`](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html), which "converts model.parameters() from plain torch.Tensor to DTensor in-place," sharding each tensor along its first dimension, with hooks that all-gather and free them around each forward and backward. It "should be applied to each layer before applying it to the root model":

```python
# fsdp_cpu.py (excerpt), run with: torchrun --nproc_per_node 2 fsdp_cpu.py
mesh = init_device_mesh("cpu", (world,))
for block in model.blocks:
    fully_shard(block, mesh=mesh)          # one all-gather group per block, bottom-up
fully_shard(model, mesh=mesh)              # the rest: tied embedding table, final norm
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
```

One GPU can't measure sharding, but two CPU processes with PyTorch's gloo backend show correctness and memory per rank, though not speed. The 22M model, three AdamW steps, two ranks of 2 sequences against one process on all 4:

| Quantity | Two ranks with `fully_shard` | One process, no sharding |
|---|---|---|
| Parameters stored per rank | 11,012,544 (half) | 22,025,088 |
| AdamW state per rank | 88.1 MB | 176.2 MB |
| Loss after steps 1, 2, 3 | 10.43542, 9.74469, 9.07031 | 10.43542, 9.74469, 9.07031 |
| Largest weight difference after 3 steps | 0.000026 | |
| Bytes each rank sent in step 1 | all-gather 62.9 MB, reduce-scatter 44.1 MB | |

The all-gathers are 44.1 MB for the forward pass plus 18.9 MB to bring the 10 blocks back for the backward pass; the tied embedding table, 57% of this model, stays gathered, because by default the root module isn't resharded after the forward pass.

### Our model, 7B and 70B

Model state per GPU, in GB:

| Model | GPUs | Plain DP | Stage 1 | Stage 2 | Stage 3 / FSDP |
|---|---|---|---|---|---|
| 88M (ours) | 8 | 1.41 | 0.48 | 0.33 | 0.18 |
| 7B | 8 | 112 | 38.5 | 26.3 | 14.0 |
| 7B | 64 | 112 | 29.3 | 15.5 | 1.75 |
| 70B | 8 | 1,120 | 385 | 263 | 140 |
| 70B | 64 | 1,120 | 293 | 155 | 17.5 |
| 70B | 512 | 1,120 | 282 | 142 | 2.19 |

On 80 GB GPUs, a 7B model on eight needs at least stage 1, leaving room for activations, while a 70B model needs stage 3 and dozens of GPUs. The price is traffic: a 7B model on 64 GPUs sends about 27.6 GB of BF16 data per GPU per step at stages 1 and 2 and 41.3 GB at stage 3, 0.09 s over NVLink 4 and 0.83 s over 50 GB/s InfiniBand, which frameworks hide behind compute.

ZeRO never shards activations. For a 70B model (`h = 8,192`, 80 layers) the formula's 34 alone is 91 GB for one 4,096-token sequence, and full checkpointing, `2sbh` per layer, still leaves 5.4 GB. So frontier runs add tensor, pipeline and context parallelism, which split each layer's matrices, the layers and the sequence, and divide activations too. [Fundamentals Part 12](/blog/pretraining-scaling-laws/) tables Llama 3's "4D parallelism." DeepSeek-V3 "applies 16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)," and optimized "the memory footprint during training, thereby enabling us to train DeepSeek-V3 without using costly Tensor Parallelism (TP)."

## In the real world

- After tuning pipeline stages and freeing buffers early, Meta "could pre-train Llama 3 on sequences of 8K tokens without activation checkpointing," letting 4D parallelism divide the memory instead ([Llama 3 paper](https://arxiv.org/abs/2407.21783)).
- DeepSeek-V3 recomputes selectively: "We recompute all RMSNorm operations and MLA up-projections during back-propagation," and its MoE layers "cache the inputs of the SwiGLU operator and recompute its output in the backward pass," in FP8. Its moving average of the weights is "stored in CPU memory" ([DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)).
- Korthikanti et al. measured full recomputation of a 22B model's layer at 39% extra time, "as opposed to the expected 33%," and selective recomputation at 7%; for GPT-3, selective recomputation saves 70% of activation memory for 2.7% more FLOPs ([paper](https://arxiv.org/abs/2205.05198)).
- ZeRO-Offload trained "over 13 billion parameters on a single GPU" at 40 TFLOPS on a V100, against 30 TFLOPS for the 1.4B model plain PyTorch could fit. It offloads only work that scales with model size, "O(M)," since forward and backward passes cost "O(MB)" with B the batch ([Ren et al., 2021](https://arxiv.org/abs/2101.06840)).
- LoRA cut GPT-3 175B's training "VRAM consumption during training from 1.2TB to 350GB" ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)).
- Optimizer states "use 33-75% of the total memory footprint during training," and 8-bit Adam matched 32-bit results up to 1.5B-parameter language models "without changes to the original optimizer hyperparameters" ([Dettmers et al., 2022](https://arxiv.org/abs/2110.02861)).

## Common misconceptions

- **"The parameter count decides how much memory training needs."** At 16 sequences the 88M model's state and weight copies are 1.49 GiB of a 6.75 GiB peak; activations scale with tokens, and the output layer alone keeps as much as two and a half to three blocks.
- **"Activation checkpointing costs 33%."** The output layer isn't recomputed, so the arithmetic predicts 25%; it measured 22%, and selective checkpointing 4%.
- **"An 8 GB card holds 8 GB of tensors."** About 7.56 GiB ran at full speed; past it, WSL2 silently moved allocations to host memory, and fills slowed from 150 to 7 GB/s.
- **"Out of memory means out of memory."** At 48 sequences the allocator had 3.30 GiB free in pieces and couldn't place one 3 GiB tensor; expandable segments made the run fit.
- **"ZeRO shards everything."** It shards weights, gradients and optimizer state; each GPU still keeps its own micro-batch's activations, which is why large runs also split matrices, layers and sequences.

## Key takeaways

- Training memory is 16 bytes per parameter of state, plus 2 for autocast's BF16 copies, plus activations that grow with every token. For the 88M model at 16 sequences, activations are three quarters of the 6.75 GiB peak.
- Korthikanti's formula, adapted to FlashAttention, SwiGLU, grouped-query attention and an FP32 residual stream, matched PyTorch's measured memory within 0.6%. FlashAttention alone removes 70% of it.
- Checkpointing every block cut activations by 72% for 22% more time per token, under the 33% rule because the output layer isn't recomputed. The selective version saved nearly a third for 4%.
- Checkpointing, 8-bit AdamW and BF16 weights fit a 1-billion-parameter model on this 8 GB GPU. At 2,347 tokens per second it would need 99 days of pretraining, so compute sets the practical limit.
- ZeRO shards optimizer state, then gradients, then weights: a 7.5B model needs 120 GB per GPU with plain data parallelism and 1.9 GB at stage 3 on 64 GPUs, for 1.5 times the traffic.

## What's next

Part 15 is the finale. It sets the recipes behind Llama 3, DeepSeek-V3 and OLMo next to ours, from data and parallelism to post-training, then exports our model, quantizes it and runs it in llama.cpp on a laptop CPU, and looks back over the whole series.

## Further reading

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054): the 2Ψ + 2Ψ + KΨ accounting, the three sharding stages and their communication volume, plus residual memory and fragmentation.
- [Reducing Activation Recomputation in Large Transformer Models (Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198): the `sbh(34 + 5as/h)` formula, sequence parallelism and selective recomputation, measured.
- [Training Deep Nets with Sublinear Memory Cost (Chen et al., 2016)](https://arxiv.org/abs/1604.06174): activation checkpointing, with O(√n) memory "with only the computational cost of an extra forward pass per mini-batch."
- [8-bit Optimizers via Block-wise Quantization (Dettmers et al., 2022)](https://arxiv.org/abs/2110.02861): the method behind bitsandbytes' `AdamW8bit`.
- [ZeRO-Offload: Democratizing Billion-Scale Model Training (Ren et al., 2021)](https://arxiv.org/abs/2101.06840): what to move to the CPU, and why the update can go but the forward and backward passes can't.
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (Zhao et al., 2023)](https://arxiv.org/abs/2304.11277) and the [`fully_shard` documentation](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html): FSDP's design, and the FSDP2 API used in Step 5, including `reshard_after_forward` and `CPUOffloadPolicy`.
- [PyTorch `torch.utils.checkpoint` documentation](https://docs.pytorch.org/docs/2.11/checkpoint.html): non-reentrant checkpointing and the selective-checkpoint policies used in Step 3.
- [Understanding CUDA Memory Usage (PyTorch)](https://docs.pytorch.org/docs/2.11/torch_cuda_memory.html) and [CUDA semantics: memory management (PyTorch)](https://docs.pytorch.org/docs/2.11/notes/cuda.html): memory snapshots, the caching allocator and `expandable_segments`.
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685): low-rank adapters and their memory savings on GPT-3.
- [The Llama 3 Herd of Models (Llama Team, 2024)](https://arxiv.org/abs/2407.21783) and [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): two frontier runs' parallelism layouts and memory tricks.
- [The Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook): data, tensor, pipeline and context parallelism and ZeRO, explained with measurements on many GPUs.
