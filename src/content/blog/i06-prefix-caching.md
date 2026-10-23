---
title: "Prefix Caching and KV Cache Offloading: Never Prefill Twice"
description: "How servers keep KV caches between requests, find them again by hashing or a radix tree, and spill them to CPU memory and SSD when the GPU fills up."
slug: "prefix-caching"
category: "inference-optimization"
tags: ["caching", "latency", "cost", "serving"]
series: "llm-inference"
seriesOrder: 6
pubDate: 2026-10-23
---

## TL;DR

Many prompts start with text the server has already processed: the same system prompt, the same tool definitions, the same conversation history plus one new message. Prefix caching keeps the KV cache (the stored keys and values from [Part 2](/blog/kv-cache/)) after a request finishes, so the next request that begins with exactly the same tokens skips that part of prefill. vLLM finds cached prefixes by hashing each 16-token block together with everything before it; SGLang keeps them in a radix tree. When GPU memory runs out, caches can spill to CPU memory and SSD, which is usually faster to load back than to recompute. Every major API provider now sells the same idea as prompt caching, with cached input billed at 2% to 10% of the normal price.

## Why it matters

Take a 20-turn chat with a 4,000-token system prompt. If the server forgets everything between turns, it prefills 158,000 tokens over the conversation. With prefix caching it prefills 11,700. More than 92% of the work was repeated.

DeepSeek reported the same effect at a larger scale when it [launched disk caching](https://api-docs.deepseek.com/news/news0802) in 2024: for "a 128K prompt with high reference, the first token latency is cut from 13s to just 500ms." Anthropic's [prompt caching announcement](https://claude.com/blog/prompt-caching) measured a 100,000-token prompt going from 11.5 seconds to 2.4 seconds before the first token, at 90% lower cost.

[Part 5](/blog/pagedattention-vllm/) showed how vLLM stores the KV cache in small blocks and frees them the moment a request ends. This part is about not freeing them: keeping caches between requests, finding them again, and moving them somewhere cheaper when the GPU is full.

## The intuition

Imagine reading a long mystery novel. Someone asks you a question about chapter 12. You can answer quickly because you already read chapters 1 to 11 and remember them. Ask a second question about chapter 12 and you reuse the same memory. That's prefix caching: the chapters you read are the cached prefix, and only the new question needs fresh reading.

Now someone hands you a different novel that happens to contain the exact same chapter 12, word for word. Your memory of chapter 12 is no help, because what it meant depended on the eleven chapters before it. The same is true for a model: a cached prefix is only reusable when everything before it matches too.

The analogy breaks in one way. A person can often make sense of a passage out of context. A model's cached keys and values are exact numbers computed from one specific history. Change one earlier token and every later number changes.

## Step 1: where repeated prefixes come from

Repeated openings show up in almost every kind of traffic:

- **System prompts and tool definitions.** Every request to a product carries the same instructions and the same tool schemas, often thousands of tokens.
- **Few-shot examples.** Prompts that show the model 5 worked examples send the same examples every time. The [SGLang paper](https://arxiv.org/abs/2312.07104) reuses "the KV cache of the 5-shot examples" on the MMLU benchmark.
- **Multi-turn chat.** Each turn re-sends the whole conversation so far, plus one new message.
- **Documents.** In RAG (retrieval-augmented generation) and document question answering, many questions arrive about the same contract or report.
- **Agent loops.** An agent calls a tool, appends the result, and calls the model again with the whole history, often dozens of times per task.
- **Parallel samples.** Asking for several answers to one prompt shares the entire prompt.

### A worked example: one chat, twenty turns

Take a chat with a 4,000-token system prompt, user messages of 100 tokens and replies of 300 tokens. Without caching, turn k prefills the system prompt, every earlier exchange and the new message. With caching, it prefills only what's new since the last request: the previous reply and the new question. That counts the reply as new input, which is the conservative case.

```python
S, u, a, T = 4000, 100, 300, 20                      # system, user, reply, turns
no_cache = [S + (k - 1) * (u + a) + u for k in range(1, T + 1)]
cached   = [S + u] + [a + u] * (T - 1)
print(sum(no_cache), sum(cached))                    # 158,000 vs 11,700 tokens
```

Prefill time comes from [Part 1](/blog/how-llm-inference-works/): about `2 × 8.03 billion = 16.06 billion` FLOPs (floating-point operations) per token for Llama 3.1 8B, on an H100 that peaks at 989 trillion FLOPs per second. That's 16.2 microseconds per token at peak.

| Turn | Tokens prefilled, no cache | TTFT at peak | Tokens prefilled, cached | TTFT at peak |
|---|---|---|---|---|
| 1 | 4,100 | 66.6 ms | 4,100 | 66.6 ms |
| 2 | 4,500 | 73.1 ms | 400 | 6.5 ms |
| 10 | 7,700 | 125.0 ms | 400 | 6.5 ms |
| 20 | 11,700 | 190.0 ms | 400 | 6.5 ms |
| All 20 turns | 158,000 | 2.57 s of GPU time | 11,700 | 0.19 s of GPU time |

Without caching, TTFT (time to first token) grows every turn. With caching it stays flat after the first. Attention adds work that grows with context length: counting it, turn 20 costs about 226 ms uncached against about 9 ms cached. Real servers run below peak, so both columns stretch, but the ratio holds. Across the whole conversation the server does 13.5 times less prefill, and the GPU time it frees goes to other users' requests in the same batch ([Part 4](/blog/continuous-batching/)).

Sharing also saves memory. Llama 3.1 8B needs 128 KiB of cache per token, so one copy of the 4,000-token system prompt takes 500 MiB. A hundred conversations holding private copies would need about 52 GB, most of the 64 GB an H100 has left after the weights. With shared blocks and reference counts from Part 5, they need 500 MiB.

## Step 2: why only exact prefixes count

Part 2 explained that each token's key and value are computed from that token's hidden state inside each layer. In the first layer, that state comes from the token itself plus its position, which RoPE (rotary position embedding) mixes into the keys. From the second layer on, it's the output of attention over every earlier token. So beyond the first layer, the key and value of token 500 depend on tokens 1 to 499.

That gives a strict rule: a cached KV entry is valid only if every earlier token is identical and in the same position. Three consequences follow:

- **A shared opening can be reused.** If two prompts agree on their first 4,000 tokens, those 4,000 tokens' keys and values are identical.
- **The same paragraph later in a prompt can't.** If a document appears at position 0 in one request and position 3,000 in another, its keys sit at different positions and saw different context. None of the cached numbers match.
- **One changed token ends the match.** A timestamp in line 1 of the system prompt changes the first block, and everything after it is a miss.

Matching happens on token IDs, not text. Text that looks the same can tokenize differently where it meets the next piece, for example when a template adds or drops a space.

Research systems do try to reuse non-prefix caches. CacheBlend ([Yao et al., 2024](https://arxiv.org/abs/2405.16444)), now part of LMCache, reuses a document's cache in a new position and "selectively recomputes the KV values of a small subset of tokens" to repair the missing cross-attention. It reports 2.2 to 3.3 times lower TTFT than full recompute. That's an approximation with a quality check, while prefix caching gives exactly the same numbers as a full prefill.

## Step 3: finding a cached prefix

The server needs a fast way to answer one question for every new request: how many of its first tokens are already in the cache?

### vLLM: a hash per block, chained

vLLM's [automatic prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html) gives every full KV block a hash of "the tokens in the block and the tokens in the prefix before the block." It doesn't rehash the whole prefix each time. Each block's hash takes its parent block's hash plus its own 16 tokens, so the hash of the 10th block stands for all 160 tokens up to its end. A toy version:

```python
import hashlib

BLOCK = 16

def block_hashes(tokens):
    hashes, parent = [], b""
    full = len(tokens) - len(tokens) % BLOCK          # only full blocks get a hash
    for i in range(0, full, BLOCK):
        block = str(tokens[i:i + BLOCK]).encode()
        parent = hashlib.sha256(parent + block).digest()  # chain in the prefix's hash
        hashes.append(parent.hex()[:8])
    return hashes

system = list(range(1000, 1032))      # a 32-token system prompt (2 blocks)
q1 = list(range(5000, 5020))          # a 20-token question
q2 = list(range(7000, 7020))          # a different 20-token question

print(block_hashes(system + q1))      # ['9266d336', '8ebb5710', '958c2299']
print(block_hashes(system + q2))      # ['9266d336', '8ebb5710', 'fbc40512']
print(block_hashes(q1[:16] + system)) # ['ccdfc352', 'c93afb47', '2c43ec71']
```

The two requests that share the system prompt agree on their first two hashes. The last line puts the same 32 system-prompt tokens after a different opening, and no hash matches: Step 2's rule, in code. A new request computes its block hashes, looks each one up in order, and stops at the first miss. Every hit is a block it can point its block table at instead of computing.

The rest of the design builds on Part 5's block pool:

- **Full blocks only.** The docs say: "We only cache full blocks." A 4,010-token prefix reuses 250 blocks (4,000 tokens) and recomputes the last 10.
- **A map from hash to block.** Alongside the pool, vLLM keeps a mapping "from hash key to block IDs."
- **Reference counts.** A block's count is "the number of requests using this block now." When a request finishes, its blocks' counts drop, and a block that reaches zero joins the free queue with its contents and hash intact, so a later request can still hit it.
- **LRU eviction.** When a new block is needed, vLLM takes the head of the free queue, the least recently used block, and removes its hash from the map. A finished request's blocks join the queue in reverse order, because "the last block of a request must hash more tokens and is less likely to be reused," so it goes first.
- **Extra keys.** The hash also covers LoRA adapter IDs, image hashes for multimodal prompts and an optional cache salt "to isolate caches in multi-tenant environments," so different tenants never share blocks by accident.

Since v0.11, the default hash is SHA-256, chosen to address collision risks; faster non-cryptographic options exist behind `--prefix-caching-hash-algo`. The [V1 announcement](https://vllm.ai/blog/2025-01-27-v1-alpha-release) turned prefix caching on by default and measured "less than 1% decrease in throughput even when the cache hit rate is 0%." vLLM exports `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` [counters](https://docs.vllm.ai/en/latest/design/metrics.html), so the hit rate is one division away.

### SGLang: a radix tree

SGLang's RadixAttention, from [Zheng et al.](https://arxiv.org/abs/2312.07104), stores cached sequences in a **radix tree**, a compact prefix tree where each edge holds a run of tokens instead of a single one. Its pages are one token long, so a match can end at any token instead of on a 16-token boundary.

![A tree of cached token sequences. The root is a shared 4,000-token system prompt. Three branches leave it: chat A's first turn, which continues into chat A's second turn; chat B's first question; and an 8,000-token contract, which branches into two different questions about the contract.](/diagrams/i06-prefix-tree.svg "Requests that share a prefix share the path from the root; each branch point is where two requests first differ.")

Every request is a path from the root. Shared openings are stored once, and a new request walks the tree as far as its tokens match. Each node keeps a reference counter of how many running requests use it, and eviction removes "the least recently used leaf first," so shared trunks like the system prompt outlive the branches hanging off them.

The tree also helps the scheduler. The paper orders waiting requests by **longest shared prefix first**, so requests that share a cached prefix run while it's still there. The paper proves that visiting the tree in depth-first order gives the best possible hit rate offline, and reports that its online scheduler reaches 96% of that optimum on average. Across its benchmarks, hit rates ran from 50% to 99%, and SGLang reached "up to 6.4x higher throughput" than other systems of the time. Managing the tree cost 0.2 seconds of a 74.3-second run when there was nothing to reuse.

| Feature | vLLM automatic prefix caching | SGLang RadixAttention |
|---|---|---|
| Lookup | Chained hash per block, hash map | Walk a radix tree |
| Match granularity | 16-token blocks by default | One token |
| Eviction | LRU over blocks with reference count 0 | LRU over leaves with reference count 0 |
| Scheduling | First come, first served by default | Longest prefix match (`--schedule-policy lpm`) available; first come, first served by default |
| On by default | Yes, since V1 | Yes |

## Step 4: when the GPU is full, spill to slower memory

A GPU's free memory fills fast. Part 1 found room for about 488,000 tokens of Llama 3.1 8B cache on one H100, about 120 conversations of 4,096 tokens. Every token beyond that evicts a cached prefix. KV cache offloading keeps evicted blocks in bigger, slower tiers. Here's what one [NVIDIA DGX H100](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html) server offers, in Llama 3.1 8B tokens:

| Tier | Capacity in a DGX H100 | Tokens at 128 KiB each | Bandwidth to the GPU |
|---|---|---|---|
| GPU HBM (high-bandwidth memory), after weights | 8 × about 64 GB (one 8B model per GPU) | about 3.9 million | 3.35 TB/s |
| CPU DRAM | 2 TB | about 15 million | 64 GB/s each way (PCIe Gen5) |
| Local NVMe SSD | 8 × 3.84 TB | about 234 million | about 14 GB/s per Gen5 drive |
| Remote storage or another server | Cluster-sized | Cluster-sized | 50 GB/s per 400 Gb/s port |

The DRAM and SSD figures are the DGX H100's specs, the HBM and network figures come from [Part 3](/blog/gpus-for-llms/), and the SSD speed is the rated sequential read of one current PCIe Gen5 drive, the [Micron 9550](https://www.micron.com/products/storage/ssd/data-center-ssd/9550-ssd). Each step down holds about 4 to 15 times more tokens than the one above, and moves them more slowly.

### Loading versus recomputing

Loading only helps if it beats recomputing. Both grow with the number of tokens, so compare per token:

```text
recompute, Llama 3.1 8B: 16.06 billion FLOPs ÷ 989 trillion FLOPs/s = 16.2 µs per token
load from CPU over PCIe: 131,072 bytes ÷ 64 GB/s                  =  2.0 µs per token
load from a Gen5 SSD:    131,072 bytes ÷ 14 GB/s                  =  9.4 µs per token
break-even bandwidth:    131,072 bytes ÷ 16.2 µs                  =  8.1 GB/s
```

Any tier that delivers more than about 8 GB/s beats recomputing at peak. For a 10,000-token prefix (1.31 GB of cache):

| Source | Time for 10,000 tokens |
|---|---|
| Already in HBM | 0.4 ms |
| CPU memory, PCIe Gen5 at 64 GB/s | 20 ms |
| Another server, 400 Gb/s network | 26 ms |
| PCIe Gen5 SSD at 14 GB/s | 94 ms |
| Recompute at peak, weights only | 162 ms |
| Recompute at peak, with attention | 189 ms |

Three things push the balance further toward loading. Real prefill runs below peak. Attention grows with context: for Llama 3.1 8B it adds about 13% to the prefill of an 8,192-token prompt and 54% at 32,768 tokens. And bigger models compute more per byte of cache. Llama 3.1 70B stores 320 KiB per token but spends about 143 microseconds recomputing each one, so its break-even bandwidth is only 2.3 GB/s.

Two things pull the other way. Small transfers pay fixed overheads, so very short prefixes are cheaper to recompute. And a tier that serves many GPUs at once splits its bandwidth among them.

### What ships today

- **vLLM** has an [`OffloadingConnector`](https://docs.vllm.ai/en/latest/features/kv_offloading_usage.html) that "extends the prefix cache by offloading completed KV blocks to slower but larger tiers," with CPU memory as the first tier and optional filesystem, object-store and peer tiers behind it. "Hits in the offload tiers are promoted back to GPU on demand." The [engine flags](https://docs.vllm.ai/en/latest/configuration/engine_args.html) `--kv-offloading-size` (GiB of buffer) and `--kv-offloading-backend` (`native` or `lmcache`) turn it on. The vLLM team's [January 2026 post](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) measured transfers of "up to about 50GB/s" in one direction on an H100 and TTFT 2 to 22 times lower than recomputing, depending on prompt size.
- **LMCache** is an open-source cache layer that moves KV caches "out of GPU memory into a tiered storage hierarchy": CPU memory, local disk, Redis and S3-compatible stores among others, per its [GitHub page](https://github.com/LMCache/LMCache). Its [paper](https://arxiv.org/abs/2510.09665) reports up to 15 times higher throughput with vLLM on multi-round question answering and document analysis, and shares caches across vLLM and SGLang instances.
- **Mooncake**, the serving platform behind Moonshot AI's Kimi, is built around the cache. Its [paper](https://arxiv.org/abs/2407.00079) uses "the underutilized CPU, DRAM, and SSD resources of the GPU cluster" as a shared KV cache pool, and its scheduler sends each request to the prefill machine with the shortest estimated TTFT given which prefixes it already holds. Mooncake let Kimi "handle 75% more requests" on real traffic. Its traces also show the limits: raising the cache from 1,000 to 50,000 blocks lifted the hit rate from 34% to 55%, with little gain beyond.
- **DeepSeek** serves its API's cache [from disk](https://api-docs.deepseek.com/guides/kv_cache), on by default for every user. The 2024 launch post credits multi-head latent attention (MLA, from Part 2) for making each token's cache small enough to keep on low-cost disks.

## Step 5: finding the right replica

A large deployment runs many copies of the model, called replicas, and each has its own GPU cache. Send a conversation's next turn to a different replica and the cache is in the wrong place. **Cache-aware routing** sends each request to the replica most likely to hold its prefix.

SGLang's paper sketches it with a router that keeps a "meta-tree" of which worker holds which prefixes. [llm-d](https://llm-d.ai/) offers "precise prefix-cache aware routing": each vLLM or SGLang server publishes events as blocks enter and leave its cache, and the router keeps an index of which blocks live where. Providers do the same internally. OpenAI's [prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching) routes on "a hash of the initial tokens" and warns that traffic above about 15 requests per minute for one prefix can overflow to other machines.

The catch is load balance: sending every request for a popular prefix to one replica overloads it. llm-d falls back to the least-loaded servers when the cache-warm ones are saturated. Part 12 covers llm-d's scheduler and moving caches between machines in depth.

## In the real world

### Provider prompt caching, as of October 2026

Every major API sells prefix caching, under the name prompt caching or context caching. Terms change often; these are from each provider's docs as of October 2026:

| Provider | How it's enabled | Minimum prefix | Lifetime | Cached input price | Extra charge to write |
|---|---|---|---|---|---|
| [Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) | One top-level `cache_control` field (automatic) or up to 4 breakpoints | 512 tokens on current models; up to 4,096 on some older ones | 5 minutes, refreshed free on each hit; 1 hour optional | 0.1x input (0.05x Opus 5.5 and Sonnet 5.5; 0.025x Fable 5.1 and Mythos 5.1) | 1.25x for 5 minutes, 2x for 1 hour |
| [OpenAI](https://developers.openai.com/api/docs/guides/prompt-caching) | On by default | 1,024 tokens (GPT-5.6 and later) | 30 minutes default; 12 hours optional | 0.1x on GPT-5.6 and later (0.05x on GPT-6.1 Sol) | 1.25x for 30 minutes, 2.5x for 12 hours; none on earlier models |
| [Google Gemini](https://ai.google.dev/gemini-api/docs/generate-content/caching) | Implicit caching on by default for 2.5 and newer; explicit caches on Gemini 3 and 2.5 | 2,048 to 4,096 tokens by model | Explicit caches: 1 hour unless set | 0.05x to 0.1x on the [pricing page](https://ai.google.dev/gemini-api/docs/pricing) | Explicit caches pay storage per hour, e.g. $4.50 per million tokens per hour on Gemini 3.1 Pro Preview |
| [DeepSeek](https://api-docs.deepseek.com/guides/kv_cache) | On by default, on disk | Best effort | Cleared "usually within a few hours to a few days" when unused | About 0.02x to 0.03x on the [pricing page](https://api-docs.deepseek.com/quick_start/pricing) | None listed |

Anthropic's docs spell out Step 2's exact-match rule. Hits "require 100 percent identical prompt segments," built in the order tools, then system, then messages, so editing a tool definition "invalidates the entire cache." The docs also isolate tenants: "Different organizations never share caches, even if they use identical prompts."

Apply Anthropic's standard prices to the 20-turn chat from Step 1, assuming each turn arrives within 5 minutes of the last. Each request writes its new tokens at 1.25x and reads the rest at 0.1x: 11,700 written and 146,300 read. That costs `1.25 × 11,700 + 0.1 × 146,300 = 29,255` token-equivalents, against 158,000 uncached: 81% off the input bill. At a 0.05x read price it's 86% off.

### Designing prompts for the cache

The cache only matches from the first token, so prompt layout decides the hit rate:

- **Stable content first, changing content last.** System prompt and tool definitions, then few-shot examples, then documents, then conversation history, then the new message.
- **No timestamps or request IDs at the top.** A timestamp in the first line makes every request a miss. Put it near the end, or use just the date if that's all the model needs.
- **Deterministic ordering.** Tools, retrieved documents and examples should come out in the same order every time; a set iterated in random order breaks the prefix.
- **Append, don't rewrite.** Summarizing or trimming old turns changes the prefix, so everything after the first changed token is prefilled again. The LMCache paper found that truncating context can halve the prefix cache hit rate.
- **Watch the hit rate.** Anthropic reports `cache_read_input_tokens` in every response, DeepSeek reports `prompt_cache_hit_tokens`, and vLLM exports hit counters. A sudden drop usually means someone changed the top of the prompt.

[How to Cut LLM Cost by 70%](/blog/cut-llm-cost-70-percent/) works through the cost side of these rules, and [How to Reduce Time to First Token](/blog/reduce-time-to-first-token/) the latency side.

## Common misconceptions

- **"Prefix caching is the same as semantic caching."** A [semantic cache](/blog/semantic-cache-hit-rate-tracking/) returns a stored answer when a new question means roughly the same as an old one, so it can be wrong. Prefix caching reuses intermediate numbers for an identical opening and still runs the model on the rest, and Anthropic says the response "is identical to what you would get if prompt caching were not used."
- **"A cached document can be reused anywhere in the prompt."** Only at the same position after the same tokens. Moving it, or putting anything new before it, means recomputing it unless the server uses an approximate method such as CacheBlend.
- **"Prefix caching makes generation faster."** It cuts prefill: TTFT and input cost. As vLLM's docs put it, it "does not reduce the time of generating new tokens."
- **"Offloading to CPU memory is too slow to help."** For Llama 3.1 8B, PCIe Gen5 moves a token's cache about 8 times faster than an H100 can recompute it at peak, and even a fast SSD comes out ahead.
- **"A bigger cache always means more hits."** Hit rate is capped by how much traffic actually repeats. In Kimi's traces, at most 55% of the cache could be reused even with unlimited storage.

## Key takeaways

- Many prompts start with text the server has already seen: system prompts, tools, few-shot examples, documents and chat history. In a 20-turn chat with a 4,000-token system prompt, caching cuts prefill from 158,000 tokens to 11,700.
- A token's keys and values depend on every token before it, so only an exact prefix can be reused. One changed token near the top turns the whole prompt into a miss.
- vLLM finds cached prefixes with a chained hash per 16-token block and evicts least recently used blocks; SGLang uses a radix tree, and its paper schedules requests that share a prefix together.
- When the GPU fills up, caches can move to CPU memory and SSD. For Llama 3.1 8B, any tier faster than about 8 GB/s beats recomputing, and CPU memory over PCIe is about 8 times faster.
- Providers sell the same mechanism as prompt caching, with cached input at 2% to 10% of the normal price. Put stable content first and keep it byte-for-byte identical.

## What's next

Prefix caching skips work that was already done. The work that's left still has to be fast, and attention is the part that grows with every token of context. [Part 7](/blog/flashattention/) covers FlashAttention and fused kernels: how to compute attention without ever writing the full table of token-to-token scores to GPU memory.

## Further reading

- [SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2023)](https://arxiv.org/abs/2312.07104): RadixAttention, LRU leaf eviction, cache-aware scheduling and its optimality proof.
- [vLLM: Automatic Prefix Caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html): chained block hashes, the free queue and eviction order in today's code.
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture (vLLM team, 2025)](https://vllm.ai/blog/2025-01-27-v1-alpha-release): prefix caching on by default and its near-zero overhead.
- [vLLM: KV Offloading Usage Guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage.html) and [Inside vLLM's New KV Offloading Connector (vLLM team, 2026)](https://vllm.ai/blog/2026-01-08-kv-offloading-connector): CPU and storage tiers, with measured transfer speeds and TTFT gains.
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (Qin et al., 2024)](https://arxiv.org/abs/2407.00079): a production system built around a cluster-wide cache, with real hit-rate data from Kimi.
- [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference (Liu et al., 2025)](https://arxiv.org/abs/2510.09665) and [LMCache on GitHub](https://github.com/LMCache/LMCache): tiered KV storage shared across engines.
- [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion (Yao et al., 2024)](https://arxiv.org/abs/2405.16444): reusing caches outside the prefix with selective recomputation.
- [Anthropic: Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [OpenAI: Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [Google: Context caching](https://ai.google.dev/gemini-api/docs/generate-content/caching) and [DeepSeek: Context caching](https://api-docs.deepseek.com/guides/kv_cache): current provider rules, lifetimes and prices.
- [DeepSeek API introduces Context Caching on Disk (DeepSeek, 2024)](https://api-docs.deepseek.com/news/news0802): why a small MLA cache makes disk storage practical.
- [llm-d: Precise prefix-cache aware routing](https://llm-d.ai/docs/well-lit-paths/foundations/precise-prefix-cache-routing): routing requests to the replica that already holds their cache.
