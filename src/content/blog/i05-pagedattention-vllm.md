---
title: "PagedAttention and vLLM: Paging the KV Cache Like an OS"
description: "Why early LLM servers wasted most of their KV cache memory, how PagedAttention's blocks and block tables fixed it, and how vLLM uses them today."
slug: "pagedattention-vllm"
category: "inference-optimization"
tags: ["serving", "caching", "gpu", "latency"]
series: "llm-inference"
seriesOrder: 5
pubDate: 2026-10-23
---

## TL;DR

Before 2023, LLM servers stored each request's KV cache in one contiguous slab of GPU memory sized for the longest possible answer. When the vLLM team measured those systems, only 20.4% to 38.2% of that memory held actual tokens. PagedAttention borrows the operating system's fix for the same problem: split the cache into small fixed-size blocks (16 tokens by default in vLLM), hand them out only as tokens arrive, and keep a per-request block table that maps the request's logical blocks to wherever they physically sit. Waste drops to less than one block per sequence, requests can share blocks through reference counts and copy-on-write, and throughput rose 2 to 4 times over the best systems of the day. This article shows where the memory went, how paging gets it back, what it costs, and how vLLM works today.

## Why it matters

In 2023, the fastest LLM servers threw away most of the GPU memory they set aside for the KV cache. The [vLLM paper](https://arxiv.org/abs/2309.06180) profiled them and found that only 20.4% to 38.2% of the KV cache memory "is used to store the actual token states." The rest was reserved for tokens that hadn't been generated yet, reserved for tokens that never would be, or stranded in gaps between allocations.

The same paper's system used 96.3%. That one change let a GPU hold several times more requests at once, and [Part 1](/blog/how-llm-inference-works/) showed why that matters: decode is memory-bound, so throughput comes from batching many users onto each read of the weights. When LMSYS moved its Chatbot Arena traffic onto vLLM in 2023, it [cut the number of GPUs](https://vllm.ai/blog/2023-06-20-vllm) serving that traffic by 50%.

[Part 2](/blog/kv-cache/) worked out how big the KV cache is. This part covers how a server lays it out when it grows one token at a time, for hundreds of requests whose final lengths nobody knows.

## The intuition

Picture a classroom where each student is writing an answer of unknown length. The old approach hands every student a bound notebook with 4,096 pages on day one, because some answer might be that long. Most students write 300 pages. The rest stay blank, and no one else can use them, because they're bound into someone else's notebook.

The new approach uses ring binders and a shared stack of loose sheets. A student takes a new sheet only when the current one is full. An index card in each binder lists its sheets in order, so they can come from anywhere in the stack. Two students answering the same question can share the printed question sheets. A student who wants to write on a shared sheet photocopies it first and writes on the copy. In PagedAttention, sheets are blocks, the index card is the block table, and photocopy-before-writing is copy-on-write.

The analogy breaks in one place: the GPU never gathers the sheets into the binder. Blocks stay scattered across memory, and the attention code jumps from block to block using the table. Step 5 measures what that costs.

## Step 1: where the memory went

Deep learning frameworks mostly expect a tensor to sit in one contiguous range of memory, so early servers stored each request's KV cache that way. The problem is that a request's final length isn't known when it arrives: the model decides when to stop. So the server reserved a chunk big enough for the maximum possible length and kept it for the request's whole lifetime.

The vLLM paper names three kinds of waste that follow:

- **Reservation**: slots set aside for tokens the request will generate later. They'll be used eventually, but until then no other request can use them.
- **Internal fragmentation**: slots inside a request's chunk that are never used, because the answer finished well before the maximum length.
- **External fragmentation**: free memory stranded between chunks. Chunks of different sizes come and go, and the leftover gaps are too small or too scattered for the next request.

### A worked example

Take Llama 3.1 8B, which needs 128 KiB of KV cache per token ([Part 2](/blog/kv-cache/)). Suppose the server reserves room for 4,096 tokens per request: 512 MiB each. Here's a snapshot of four requests partway through generation. The lengths are illustrative:

| Request | Prompt | Generated so far | Final length (unknown to the server) | Slots holding tokens now | Reserved for later tokens | Never used |
|---|---|---|---|---|---|---|
| A | 200 | 100 | 450 | 300 | 150 | 3,646 |
| B | 1,200 | 300 | 1,800 | 1,500 | 300 | 2,296 |
| C | 50 | 20 | 300 | 70 | 230 | 3,796 |
| D | 800 | 600 | 1,500 | 1,400 | 100 | 2,596 |

```python
MAX_LEN, KIB = 4096, 1024
per_token = 128 * KIB                        # Llama 3.1 8B, 16-bit
reqs = [(200, 100, 450), (1200, 300, 1800),  # (prompt, generated, final)
        (50, 20, 300), (800, 600, 1500)]

reserved = len(reqs) * MAX_LEN                # 16,384 slots
used = sum(p + g for p, g, f in reqs)        # 3,270 slots
later = sum(f - (p + g) for p, g, f in reqs) # 780 slots
never = sum(MAX_LEN - f for p, g, f in reqs) # 12,334 slots

print(used / reserved)                       # 0.1996
print(reserved * per_token / 2**20, used * per_token / 2**20)  # 2048.0 MiB, 408.75 MiB
```

The four requests tie up 2,048 MiB, and 408.75 MiB of it holds tokens: 20.0% utilization. Another 4.8% is reserved for tokens still to come, and 75.3% will never be touched.

It gets worse with long context windows. Llama 3.1 8B accepts 128K tokens, and reserving that much costs 16 GiB per request. An 80 GB H100 with about 63.9 GB left after the weights then fits 3 requests, even if each one only ever writes 300 tokens.

### External fragmentation

Smarter servers reserved less. One option rounds each request's expected length up to a power of two, so chunks come in different sizes. Suppose a region of 4,096 slots holds chunks of 1,024, 512 and 2,048 slots, plus 512 free. The 512-slot request finishes. Now 1,024 slots are free in total, but as two separate 512-slot holes, and a new request that needs 1,024 contiguous slots doesn't fit. That stranded memory is external fragmentation. The numbers here are illustrative, but the pattern is what the paper's Figure 3 shows.

### What the paper measured

The vLLM team built three versions of Orca, a 2022 research system that introduced iteration-level scheduling (the idea [Part 4](/blog/continuous-batching/) builds continuous batching on), each reserving memory differently:

| Version | What it reserves for each request's output | Share of KV memory holding tokens |
|---|---|---|
| Orca (Max) | Up to the model's maximum sequence length | 20.4% |
| Orca (Pow2) | Up to 2 times the true length, rounded to a power of two | 26.8% |
| Orca (Oracle) | Exactly the true length, known in advance (impossible in practice) | 38.2% |
| vLLM | Nothing ahead of time: blocks on demand | 96.3% |

Even the Oracle version, which magically knew every output length, used only 38.2%, because it still held the whole chunk from the first token to the last.

## Step 2: borrowing from the operating system

Operating systems solved this problem long ago with **virtual memory**. Each program sees one long, contiguous range of addresses. Behind the scenes, the operating system splits physical memory into fixed-size **pages** and keeps a **page table** for each program that maps its virtual pages to whichever physical pages hold them. Consecutive virtual pages can sit anywhere in physical memory, and physical pages are handed out only when the program actually needs them. The [Linux kernel docs](https://docs.kernel.org/admin-guide/mm/concepts.html) put it simply: "The virtual memory abstracts the details of physical memory from the application software."

The vLLM paper maps the ideas across directly: "one can think of blocks as pages, tokens as bytes, and requests as processes."

| Operating system | PagedAttention in vLLM |
|---|---|
| Process | Request (one sequence of tokens) |
| Byte | Token |
| Page | KV block (a fixed number of tokens' keys and values) |
| Page table | Block table |
| Physical page frame | Physical KV block in GPU memory |
| Copy-on-write after a process forks | Copy-on-write when sequences that share a block diverge |
| Swapping pages to disk | Swapping blocks to CPU memory, or recomputing them |

Because every block is the same size, any free block fits any request, so external fragmentation disappears. Because blocks are small and allocated on demand, internal fragmentation and reservation shrink to the unfilled part of one block.

The analogy isn't exact. Four differences matter:

- **Translation happens in software.** A CPU translates addresses in hardware and caches recent translations in a TLB (translation lookaside buffer). On the GPU, vLLM's own attention kernel reads the block table and computes addresses itself.
- **No page faults.** An operating system allocates a page when a program first touches it. vLLM's scheduler allocates the blocks each request will need before every step runs.
- **All or nothing.** An operating system evicts individual pages. A request needs every one of its blocks to produce its next token, so vLLM evicts all of a sequence's blocks or none of them.
- **Recomputation.** An operating system can't regenerate lost data, but vLLM can rebuild an evicted cache by running the tokens through the model again.

## Step 3: blocks and block tables

A **KV block** holds the keys and values for a fixed number of consecutive tokens. Each request has a list of **logical blocks**, filled left to right as tokens arrive: logical block 0 holds tokens 1 to 16, block 1 holds tokens 17 to 32, and so on. The **block table** records, for each logical block, which **physical block** in GPU memory holds it and how many of its slots are filled.

![Two requests, each with logical blocks that map through a block table to physical blocks in GPU memory. Request A has three logical blocks holding tokens 1-16, 17-32 and 33-40; its block table maps them to physical blocks 7, 1 and 3. Request B has two logical blocks holding tokens 1-16 and 17-20; its block table maps them to physical blocks 5 and 2. Physical block 4 is free.](/diagrams/i05-block-table.svg "Each request sees its blocks in order; the block table says where each one actually sits in GPU memory.")

Consecutive logical blocks land wherever a free physical block happened to be, and neither request reserves anything past its last block. Here's how request A got there:

1. **Prefill.** A arrives with a 37-token prompt. The scheduler takes three free blocks from the pool (7, 1 and 3) and records them in A's block table. Prefill writes tokens 1 to 16 into block 7, 17 to 32 into block 1, and 33 to 37 into block 3. Block 3 has 11 empty slots.
2. **Decode.** Each new token goes into the next empty slot of block 3, and the table's filled count goes up by one. After 3 decode steps, A holds 40 tokens and block 3 still has 8 free slots.
3. **A new block.** When block 3 fills at token 48, the next token needs a fresh block. The scheduler pops one from the free pool before that step runs and adds it to A's table.
4. **Finish.** When A ends, its blocks go back to the pool.

### How much can still be wasted

Only the last block of a sequence can be partly empty. With 16-token blocks, the worst case is 15 empty slots:

```text
max waste per sequence = (16 − 1) slots × 128 KiB = 1,920 KiB ≈ 1.9 MiB
compare: one 4,096-token reservation = 512 MiB
```

Rerun the four requests from Step 1 with 16-token blocks and nothing reserved ahead:

```python
import math
BLOCK = 16
blocks = sum(math.ceil((p + g) / BLOCK) for p, g, f in reqs)  # 206 blocks
print(used / (blocks * BLOCK))                                  # 0.9921
print(blocks * BLOCK * per_token / 2**20)                       # 412.0 MiB
```

The same four requests now take 412 MiB instead of 2,048 MiB, and 99.2% of the allocated slots hold tokens. The other 1,636 MiB is free for more requests.

Each block of Llama 3.1 8B's cache is `16 tokens × 128 KiB = 2 MiB`, spread over the model's 32 layers at 64 KiB per layer. In current vLLM, layers in the same group "share the same block ids," according to its [design docs](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html), so one block ID names the same slot range in every layer's cache. The 63.9 GB free on an H100 becomes a pool of about 30,470 blocks, or 487,500 tokens.

### Why 16 tokens

Small blocks waste less and share more easily, but each block is a separate chunk of work for the GPU, and tiny chunks leave its cores underused. Large blocks keep the GPU busy but waste more per sequence. The paper tested this on two datasets: on ShareGPT (real ChatGPT conversations, long) block sizes from 16 to 128 did best; on Alpaca (short instructions) 16 and 32 did well and larger blocks hurt. "Accordingly, vLLM sets its default block size as 16."

That default still holds: vLLM's cache configuration sets [`DEFAULT_BLOCK_SIZE = 16`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py), and the `--block-size` flag overrides it. Some attention backends ask for larger blocks when their kernels need them.

## Step 4: sharing blocks with reference counts and copy-on-write

Once a block table sits between a sequence and its memory, two sequences can point at the same physical block.

### Parallel sampling

**Parallel sampling** asks for several different answers to one prompt, for example 4 code suggestions to pick from. All 4 sequences start from the same prompt, so its keys and values are identical. vLLM stores them once and points all 4 block tables at the same physical blocks. Each physical block carries a **reference count**: how many sequences currently use it.

The tricky block is the last prompt block, which is only partly full. When sample 1 wants to write its first new token there, vLLM sees a reference count above 1, allocates a new block, copies the existing tokens into it, points sample 1's table at the copy, and lowers the original's count. This is **copy-on-write**: shared until someone writes, then copied. Samples that later find the count down to 1 write in place.

With a 1,000-token prompt, 4 samples and 200 generated tokens each:

```text
without sharing: 4 × ceil(1,200 / 16)         = 4 × 75 = 300 blocks = 600 MiB
with sharing:    62 shared full prompt blocks
                 + 4 × (75 − 62) private blocks = 114 blocks = 228 MiB  (62% less)
```

The 62 full prompt blocks are shared; the partial 63rd block is copied once per sample and then filled with that sample's own tokens.

### Beam search

**Beam search** keeps the k most likely partial answers (the beam width) and extends each by one token per step. Candidates branch from each other and get dropped, so the sharing pattern changes constantly. With block tables, a new candidate that branches from an old one just copies the old one's block table and bumps the reference counts. When a candidate drops out of the top k, its blocks' counts go down, and any block that reaches zero returns to the free pool. Older systems copied large parts of the cache between candidates; vLLM copies only a shared block that's still being written.

### How much it saved

The paper measured the memory saved by sharing on OPT-13B (a 13 billion-parameter model):

| Workload | Parallel sampling | Beam search |
|---|---|---|
| Alpaca trace | 6.1% to 9.8% | 37.6% to 55.2% |
| ShareGPT trace | 16.2% to 30.5% | 44.3% to 66.3% |

Beam search saves more because candidates also share much of their generated text. On Alpaca, vLLM's throughput advantage over Orca (Oracle) grew from 1.3 times with ordinary sampling to 2.3 times with beam search at width 6.

The same mechanism lets different requests share blocks for a common opening, such as a long system prompt. Keeping those blocks after a request finishes, so the next one can reuse them, is prefix caching, and Part 6 covers it.

Today's vLLM [splits a request for n samples into n child requests](https://github.com/vllm-project/vllm/issues/10980) that reuse the prompt's blocks through the prefix cache. That cache [stores only full blocks](https://docs.vllm.ai/en/latest/design/prefix_caching.html), so each child fills its own last block instead of copying a shared one.

## Step 5: attention over scattered blocks

Ordinary attention kernels assume each request's keys and values sit in one contiguous array. PagedAttention rewrites the computation block by block. For a new token's query, the kernel walks the request's block table; for each logical block it finds the physical block, multiplies the query by the 16 keys stored there to get 16 attention scores, and later multiplies the attention weights by the 16 values. The result is identical to ordinary attention; only the memory addresses differ.

The paper's kernels do three jobs:

- **Write**: split each step's new keys and values into blocks and store them where the block table says, fused into one GPU kernel to avoid launch overhead.
- **Read and attend**: an attention kernel adapted from NVIDIA's FasterTransformer library that reads the cache through the block table, with one warp (a group of 32 GPU threads, from [Part 3](/blog/gpus-for-llms/)) assigned to each block so memory reads stay coalesced.
- **Copy**: copy-on-write can trigger many small block copies in one step, so one kernel batches them instead of issuing a separate memory copy per block.

Table lookups, extra branches and variable lengths made vLLM's attention kernel **20% to 26% slower** than FasterTransformer's highly optimized one. The authors call the overhead small because it touches only attention, not the matrix multiplications that make up most of the model, and a batch several times bigger more than repays it.

General-purpose kernels now accept paged layouts. FlashAttention's `flash_attn_with_kvcache` function, for example, takes a [`block_table`](https://github.com/Dao-AILab/flash-attention) argument for "paged KV cache." Part 7 covers FlashAttention itself.

## In the real world

### The paper's results

Kwon et al. presented PagedAttention at SOSP 2023 (the Symposium on Operating Systems Principles), testing OPT models from 13B to 175B parameters and LLaMA-13B on NVIDIA A100 GPUs. vLLM improved throughput "by 2-4× with the same level of latency" over FasterTransformer and Orca. On ShareGPT, it sustained 1.7 to 2.7 times higher request rates than Orca (Oracle) and up to 22 times higher than FasterTransformer. The reason shows in the batch: for OPT-13B, vLLM kept 2.2 times more requests in flight than Orca (Oracle).

### The launch

The [vLLM launch post](https://vllm.ai/blog/2023-06-20-vllm) (June 20, 2023) reported "up to 24x higher throughput compared to HF and up to 3.5x higher throughput than TGI," comparing against Hugging Face Transformers and Hugging Face's Text Generation Inference server. It put PagedAttention's waste at "under 4%," against 60% to 80% for existing systems. LMSYS had used it since mid-April 2023 to serve Vicuna and other models in Chatbot Arena, averaging 30,000 requests a day with a peak of 60,000.

### vLLM today

vLLM has since been rebuilt. The [V1 engine](https://vllm.ai/blog/2025-01-27-v1-alpha-release), announced in January 2025, reported up to 1.7 times the throughput of the original design. [Release v0.11.0](https://github.com/vllm-project/vllm/releases/tag/v0.11.0) completed the removal of the old engine: "V1 is the only engine in the codebase now." The [architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview.html) describes three kinds of processes:

- **API server**: handles HTTP requests, such as the OpenAI-compatible API.
- **Engine core**: "runs the scheduler, manages KV cache, and coordinates model execution across GPU workers," one per data-parallel copy of the model.
- **GPU workers**: one per GPU, each with a model runner that loads and runs the model.

Each step, the scheduler decides how many tokens each request processes, written as `{request_id: num_tokens}`, treating prompt and generated tokens alike. Part 4 covers that scheduling.

The KV cache manager follows the paper's design. At startup it turns the memory left after the weights and a profiling run into a block pool, with vLLM allowed 92% of GPU memory by default (`--gpu-memory-utilization 0.92`). Free blocks wait in a queue, and each block keeps a reference count of "the number of requests using this block now." Prefix caching is on by default in V1, and the V1 announcement reported it costs "less than 1% decrease in throughput even when the cache hit rate is 0%."

### When blocks run out

Blocks are allocated as requests grow, so a batch that fit at the start of a step can outgrow the pool later. vLLM then **preempts** requests: it frees all of their blocks and runs them again later. V1 recomputes by default, and the [tuning guide](https://docs.vllm.ai/en/latest/configuration/optimization.html) suggests raising `gpu_memory_utilization`, lowering `max_num_seqs` or `max_num_batched_tokens`, or adding GPUs with tensor parallelism if preemption is frequent. [Part 2](/blog/kv-cache/) covered the choice between swapping and recomputing.

### Other engines

Paged caches are now the norm:

- **SGLang** stores its cache in "a non-contiguous, paged layout, where the size of each page is equivalent to one token," according to its [paper](https://arxiv.org/abs/2312.07104). Its `--page-size` flag still [defaults to 1](https://docs.sglang.io/advanced_features/server_arguments.html). One-token pages suit its RadixAttention prefix cache, which Part 6 covers.
- **TensorRT-LLM**, NVIDIA's engine and the [successor to FasterTransformer](https://github.com/NVIDIA/FasterTransformer), has a [paged KV cache](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html) that "decomposes the KV cache into blocks that are distributed to the different requests by a cache manager." Its docs say the older contiguous cache "uses a lot more memory than needed when the sequences are shorter than the maximum sequence length."
- **Text Generation Inference**, Hugging Face's server, uses [the custom CUDA kernels developed by the vLLM project](https://huggingface.co/docs/text-generation-inference/conceptual/paged_attention) for its PagedAttention.

## Common misconceptions

- **"PagedAttention makes attention faster."** It makes attention slightly slower: 20% to 26% in the paper's kernel benchmark. The win comes from fitting several times more requests in memory, which makes every read of the weights serve more users.
- **"Paging means the KV cache moves to CPU memory."** Paging here is about how the cache is laid out inside GPU memory. Moving blocks to CPU memory is swapping or offloading, a separate step that Part 6 covers.
- **"Paged memory has zero waste."** Each sequence can still waste up to one block minus one token: 15 slots, or about 1.9 MiB for Llama 3.1 8B. The paper measured 96.3% utilization, not 100%.
- **"Smaller blocks are always better."** They waste less per sequence but keep the GPU less busy. The paper found 16 a good balance, and vLLM still defaults to it.
- **"vLLM and PagedAttention are the same thing."** PagedAttention is the memory technique; vLLM is a full serving engine built around it, with a scheduler, workers and an API server. SGLang, TensorRT-LLM and Text Generation Inference use paged caches too.

## Key takeaways

- Early LLM servers reserved one contiguous slab per request, sized for the maximum length, and only 20.4% to 38.2% of their KV cache memory held real tokens.
- PagedAttention splits the cache into fixed-size blocks, 16 tokens by default, and uses a block table per request to map logical blocks to physical ones anywhere in GPU memory, just as an operating system's page table does.
- Blocks are allocated only when needed, so each sequence wastes at most part of its last block: about 1.9 MiB for Llama 3.1 8B instead of hundreds of megabytes.
- Reference counts and copy-on-write let parallel samples and beam search candidates share blocks, saving up to 66.3% of memory for beam search in the paper's tests.
- The attention kernel runs 20% to 26% slower, but bigger batches made vLLM 2 to 4 times faster than the best systems of 2023, and paged caches are now standard in vLLM, SGLang and TensorRT-LLM.

## What's next

PagedAttention frees a request's blocks the moment it finishes. But the next request often starts with the same system prompt, or the same conversation history plus one new message, and recomputing that is wasted work. [Part 6](/blog/prefix-caching/) covers prefix caching and KV cache offloading: keeping caches between requests, finding them again, and spilling them to CPU memory and SSD when the GPU is full.

## Further reading

- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180): the SOSP 2023 paper behind vLLM, with the waste measurements, the block design, sharing and the kernel overhead.
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention (vLLM team, 2023)](https://vllm.ai/blog/2023-06-20-vllm): the launch post, with the throughput comparisons and the LMSYS deployment.
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture (vLLM team, 2025)](https://vllm.ai/blog/2025-01-27-v1-alpha-release): the current engine's design, scheduler and default prefix caching.
- [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html) and [Automatic Prefix Caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html): processes, the block pool, the free queue and reference counts in today's code.
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization.html): preemption and the settings that control KV cache memory.
- [SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2023)](https://arxiv.org/abs/2312.07104): one-token pages and the RadixAttention prefix cache.
- [TensorRT-LLM: Multi-Head, Multi-Query and Group-Query Attention](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html) and [KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html): NVIDIA's paged KV cache and block pool.
- [Linux kernel: Concepts overview (memory management)](https://docs.kernel.org/admin-guide/mm/concepts.html): virtual memory, pages and page tables, the operating-system ideas PagedAttention borrows.
