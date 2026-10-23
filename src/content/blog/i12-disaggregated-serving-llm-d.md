---
title: "Disaggregated Serving and llm-d: Splitting Prefill from Decode"
description: "Why large LLM deployments run prefill and decode on separate GPUs, what moving the KV cache between them costs, and how llm-d runs it on Kubernetes."
slug: "disaggregated-serving-llm-d"
category: "inference-optimization"
tags: ["serving", "latency", "gpu", "deployment", "caching"]
series: "llm-inference"
seriesOrder: 12
pubDate: 2026-10-23
---

## TL;DR

Prefill reads a whole prompt in one burst of arithmetic, and decode then writes the answer one token at a time, limited by memory bandwidth. Disaggregated serving runs the two phases on separate groups of GPUs and ships each request's KV cache from the first group to the second. That ends the interference between them, lets each group use its own parallelism and hardware, and lets you scale time to first token and time per output token independently. The price is moving the cache: 1.34 GB for a 4,096-token prompt on Llama 3.1 70B, which takes 27 ms over one 400 Gb/s link but hides almost entirely behind prefill when it streams layer by layer. By the end you'll know when splitting pays off, how libraries such as NIXL move the cache, and how llm-d schedules prefill and decode pods on Kubernetes.

## Why it matters

DeepSeek's [published statistics](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) for one day of V3 and R1 traffic list 608 billion input tokens, 168 billion output tokens, and per-node speeds of about 73,700 input tokens per second in prefill (counting tokens served from cache) and 14,800 output tokens per second in decode. Divide each daily total by its speed and you get about 95 nodes reading prompts and 131 writing answers: 227 nodes, almost exactly the 226.75 that DeepSeek reported running on average that day.

Those per-node speeds come from machines that each do one job. Prefill runs on deployment units of 4 nodes and decode on units of 18, each with its own parallelism. The [DistServe paper](https://arxiv.org/abs/2401.09670), which made the case for this split in 2024, served "7.4x more requests or 12.6x tighter SLO" than the systems it compared against.

[Part 4](/blog/continuous-batching/) left a question open. Chunked prefill makes the two phases share a GPU fairly, but they still compete for it. This part covers what happens when you stop sharing: running prefill and decode on different machines, moving the KV cache between them, and the open-source stack, [llm-d](https://github.com/llm-d/llm-d), that does it on Kubernetes.

## The intuition

Picture a restaurant where every cook does everything. A cook halfway through plating twenty dishes gets handed a banquet order and has to stop and prep it, and twenty tables wait. That's prefill and decode on one GPU.

The fix is a prep kitchen and a service line. Prep cooks handle the big, intense jobs: chopping, marinating, building each order's tray of ingredients. Line cooks work through many plates at a steady pace. Each team can be sized for its own workload and given the right equipment. But every tray has to be carried down the hallway, and if the hallway is slow, the line waits.

The analogy breaks in one useful way. A real tray is finished before it's carried. A KV cache is built one layer at a time, so the server can send layer 1 down the hallway while layer 2 is still being prepared. By the time prep finishes, almost everything has already arrived.

## Step 1: why split the two phases

[Part 1](/blog/how-llm-inference-works/) showed that the two phases stress the GPU in opposite ways. Here they are side by side:

| Property | Prefill | Decode |
|---|---|---|
| Work per request | Whole prompt in one forward pass | One token per forward pass |
| Limited by | Arithmetic (compute-bound) | Memory bandwidth (memory-bound) |
| Latency it sets | TTFT (time to first token) | TPOT (time per output token) |
| Batch it wants | A few long prompts fill the GPU | Hundreds of users per step |
| Best parallelism | Less: smaller tensor-parallel groups, more replicas | More: wider groups that pool bandwidth and memory for the KV cache |

Putting both on the same GPU causes four problems, and disaggregation solves each one.

### 1. Interference

Part 4 worked out that a 32,768-token prompt freezes 64 other streams of an 8-billion-parameter model on an H100 for about 0.8 seconds. Chunked prefill caps the damage per step, but every step still carries some prompt work. On separate GPUs, a decode step never contains prefill. vLLM's docs give this as the reason for the feature: it controls "tail ITL" (inter-token latency, another name for TPOT), and while chunked prefill "also can achieve the same goal," the right chunk size is "hard to figure out" in practice.

### 2. Different parallelism

Each phase has its own best way to spread across GPUs ([Part 10](/blog/model-parallelism/) explains tensor and pipeline parallelism). For OPT-175B, DistServe's search picked a prefill instance with 3 pipeline stages of 3 GPUs and a decode instance with 3 stages of 4 GPUs. [llm-d's design notes](https://github.com/llm-d/llm-d/blob/main/docs/architecture/disaggregation/pd-disaggregation.md) give the same rule of thumb, "a larger TP for the memory-bound decoding phase while a smaller TP for the computation-bound prefill phase," and an example: 8 prefill pods of 1 GPU each and 2 decode pods of 4 GPUs each for gpt-oss-120b with 5,000-token prompts and 250-token answers.

### 3. Different hardware

[Splitwise](https://arxiv.org/abs/2311.18677), from Microsoft and the University of Washington (Patel et al., 2023), observed that token generation "does not require the compute capability of the latest GPUs." One of its designs runs prompts on DGX-H100 machines and token generation on cheaper DGX-A100s; another caps the decode machines' power, since decode showed "no performance impact with 50% lower power cap per GPU." Overall, Splitwise reported "1.4x higher throughput at 20% lower cost than current designs," or "2.35x more throughput with the same cost and power budgets."

### 4. Independent scaling

TTFT depends on how many GPUs read prompts, and TPOT on how many write answers. With two pools, a burst of long prompts adds prefill replicas and a burst of long answers adds decode replicas, without buying capacity the other target doesn't need. DeepSeek's numbers above put about 42% of the nodes on prefill and 58% on decode.

### What the papers measured

DistServe measured **goodput**, the highest request rate a system sustains per GPU while at least 90% of requests meet both their TTFT and TPOT targets. Against vLLM on a chatbot workload, it sustained 2.0 to 4.6 times the request rate; against DeepSpeed-MII, which uses chunked prefill, 1.6 to 7.4 times. The headline 12.6 times came from a summarization workload with long prompts, where DistServe met latency targets 12.6 times stricter than vLLM could.

[Mooncake](https://arxiv.org/abs/2407.00079), Moonshot AI's serving platform for Kimi, separates prefill and decode clusters too and builds its scheduler around a shared KV cache pool. Under real workloads it let Kimi "handle 75% more requests." [Part 6](/blog/prefix-caching/) covers its cache design.

## Step 2: the price is moving the KV cache

Splitting the phases means the KV cache, the stored keys and values from [Part 2](/blog/kv-cache/), is built on one machine and needed on another. Every request has to cross the network once before its second token.

### How big is the package?

Take Llama 3.1 70B in 16-bit precision. Part 2 sized its cache at 320 KiB per token:

```text
KV per token  = 2 (K and V) × 80 layers × 8 KV heads × 128 numbers × 2 bytes = 327,680 bytes
4,096 tokens  = 327,680 × 4,096 = 1.34 GB
```

Here's how long 1.34 GB takes to move over the links from [Part 3](/blog/gpus-for-llms/), next to the work around it. The prefill and decode figures assume the model runs on 8 H100s with tensor parallelism and use peak speeds, so real ones are longer:

| What | Time |
|---|---|
| Prefill of the 4,096-token prompt (600 trillion FLOPs on 8 H100s at peak) | 76 ms |
| One decode step (each GPU reads 17.6 GB of weights) | 5.3 ms |
| Transfer over NVLink 4 (450 GB/s) | 3.0 ms |
| Transfer over one PCIe Gen5 link (64 GB/s) | 21 ms |
| Transfer over one 400 Gb/s RDMA link (50 GB/s) | 27 ms |
| Transfer with each of the 8 GPUs sending its own share over its own 400 Gb/s link | 3.4 ms |

RDMA (remote direct memory access) lets one machine's network card read or write another machine's memory without involving either CPU. With GPUDirect RDMA, the card reads straight out of GPU memory.

Sent in one piece over a single link, the cache costs 27 ms, about five decode steps, before token 2. Two things shrink it. First, the cache is already split: at tensor-parallel size 8, each GPU holds one of the 8 KV heads, 168 MB of this prompt's cache. If every GPU has its own network card, as on a typical 8-GPU server, the eight shares travel in parallel in 3.4 ms. Second, the cache doesn't have to wait for prefill to finish.

### Streaming layer by layer

Prefill computes the model one layer at a time, and each layer's keys and values are final as soon as that layer is done. Splitwise exploits this: "At the end of each layer, we trigger an asynchronous transfer of the KV-cache for that layer while the prompt computation continues to the next layer."

Work it per layer for our example:

```text
KV per layer      = 1.34 GB ÷ 80 layers = 16.8 MB
send one layer    = 16.8 MB ÷ 50 GB/s   = 0.34 ms   (even over a single link)
compute one layer = 76 ms ÷ 80 layers   = 0.95 ms   (at peak; real prefill is slower)
```

Each layer's cache is on the wire long before the next layer finishes computing, so only the last layer's 0.34 ms is left when prefill ends. Splitwise measured it on real hardware: sending the cache in one piece added 64% to the second token's latency, sending it per layer added 16.5%, and the cost to end-to-end latency was 0.8%. For short prompts it sends in one piece, since per-layer synchronization isn't worth it.

DistServe took a different route. It placed each prefill stage and its matching decode stage on the same machine, so the cache moves over NVLink. Even for OPT-175B on a cluster with only 25 Gb/s between machines, the transfer was "less than 0.1% of the total latency," and over 95% of requests waited less than 30 ms for it.

### How much bandwidth does it take?

NVIDIA's study [Beyond the Buzz](https://arxiv.org/abs/2506.05508) gives the rule: the network must carry each prompt's cache in the time it takes to prefill it. For our example, that's `1.34 GB ÷ 76 ms = 17.7 GB/s` for the whole 8-GPU prefill instance, about a third of one 400 Gb/s link. The paper points out two trends. Prefill time grows faster than linearly with prompt length while the cache grows linearly, so the bandwidth needed falls as prompts get longer. And prefill time grows with a model's active parameters while its cache grows more slowly, so a large model with compact attention can need less bandwidth than a smaller one.

Cache-efficient attention helps most. DeepSeek-V3's multi-head latent attention (MLA, from Part 2) stores one 576-number latent per token per layer: 68.6 KiB per token, 4.7 times less than Llama 3.1 70B's despite being a much larger model. A 4,096-token prompt's cache is 288 MB, 5.8 ms over a single 400 Gb/s link. That small cache helps make DeepSeek's cross-node split cheap.

Inside Kubernetes, llm-d's docs note, NVLink transfers "cannot cross pod boundaries," so prefill and decode pods on the same node still talk over RDMA. And without RDMA at all, the transfer falls back to TCP, which they describe as "not efficient and should only be used for testing and development."

## Step 3: the libraries that move the cache

- [NIXL](https://github.com/ai-dynamo/nixl), the NVIDIA Inference Xfer Library, is "targeted for accelerating point to point communications in AI inference frameworks," with plug-ins for GPU and CPU memory and for storage. Its network backends (UCX, UCCL, libfabric) reach InfiniBand, RoCE (RDMA over Converged Ethernet) and AWS's EFA. vLLM and SGLang register their KV cache memory with NIXL, so blocks move straight from one engine's cache to the other's.
- [vLLM's KV connector API](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) is still marked "experimental and subject to change." Its connectors include `NixlConnector`, `MooncakeConnector`, `LMCacheConnectorV1` (which can use NIXL underneath), `OffloadingConnector` for CPU tiers, and `MultiConnector` to chain several. One flag turns it on: `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`.
- [SGLang's PD disaggregation](https://docs.sglang.io/advanced_features/pd_disaggregation.html) supports Mooncake and NIXL as transfer engines. Its docs add a reason specific to MoE models: with data-parallel attention, one worker prefilling while another decodes slows the decode.
- [Mooncake's Transfer Engine](https://github.com/kvcache-ai/Mooncake), the core of Kimi's platform, spreads each transfer over several network cards. Moving 40 GB (128K tokens of Llama 3 70B cache), it reports up to 87 GB/s over 4×200 Gb/s RoCE and 190 GB/s over 8×400 Gb/s.

On the request side, vLLM's protocol has two calls. The prefill instance gets the prompt with `max_tokens=1` and `"do_remote_decode": true`, runs prefill, and returns `kv_transfer_params` that say where the blocks are. The decode instance gets the same request with those parameters and `"do_remote_prefill": true`, pulls the blocks, and generates. Choosing both instances and passing the parameters between them is llm-d's job.

## Step 4: how llm-d runs it on Kubernetes

[llm-d](https://github.com/llm-d/llm-d) describes itself as "a high-performance distributed inference serving stack optimized for production deployments on Kubernetes." Red Hat [launched it in May 2025](https://www.redhat.com/en/about/press-releases/red-hat-launches-llm-d-community-powering-distributed-gen-ai-inference-scale) with CoreWeave, Google Cloud, IBM Research and NVIDIA. It joined the Cloud Native Computing Foundation (CNCF) as a [sandbox project](https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/) in March 2026, and [v1.0.0](https://github.com/llm-d/llm-d/releases/tag/v1.0.0) shipped on October 20, 2026. It runs vLLM or SGLang as model servers and adds routing, cache management and orchestration above them.

Kubernetes runs containers in units called pods, and a **custom resource** is a new object type that extends its API. The [Gateway API](https://gateway-api.sigs.k8s.io/) is Kubernetes' standard for configuring HTTP traffic into a cluster. llm-d builds on its [Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension), which turns an ordinary gateway into an "inference gateway."

### The parts

- An `InferencePool` is a custom resource that names a group of model server pods (by label) and the service that picks among them. One pool usually serves one model, and prefill and decode pods live in the same pool, told apart by a label: `llm-d.ai/role=prefill` or `llm-d.ai/role=decode`.
- The proxy can be any gateway that supports the extension, such as Istio, agentgateway or Google Cloud's load balancer. It "parks" each request and asks the Endpoint Picker where to send it, over Envoy's external-processing protocol.
- The Endpoint Picker (EPP) makes the routing decisions. It filters candidate pods, scores them and picks one. The llm-d Router is the proxy plus the EPP; the project used to call this component the inference scheduler.
- A routing sidecar, a small proxy, runs next to each decode engine and carries out the two-step prefill-then-decode protocol.
- NIXL moves the cache over RDMA.

### One request, start to finish

![A sequence diagram with six participants: client, gateway proxy, endpoint picker, decode sidecar, prefill pod and decode vLLM. The client sends a chat request to the gateway. The gateway asks the endpoint picker which pods to use, and it answers with a decode pod and a prefill pod. The gateway forwards the request with the prefill address to the decode sidecar, which sends the same prompt to the prefill pod with max_tokens set to 1. The prefill pod writes the KV cache and returns KV transfer parameters to the sidecar. The sidecar sends the request and parameters to the decode vLLM, which pulls the KV blocks from the prefill pod over NIXL and RDMA. The prefill pod frees its copy, and the decode engine streams tokens back to the client through the sidecar and gateway.](/diagrams/i12-pd-request-flow.svg "A disaggregated request in llm-d: the router picks both pods, the decode sidecar runs prefill remotely, and the decode engine pulls the cache over NIXL.")

The decode side drives the exchange: the request lands on the decode pod first, and the decode engine pulls the cache. DistServe also chose "pull", to survive bursts of traffic: decode instances fetch caches when they have room, so a burst doesn't overload their memory, and prefill GPUs act as the queue.

### How the picker scores pods

The EPP runs a **filter, score, pick** pipeline. For a disaggregated request, its `disagg-profile-handler` runs it twice: a decode profile that keeps only decode pods, and a prefill profile that keeps only prefill pods. Each scorer returns a value from 0 to 1, and the EPP multiplies each by a weight and adds them up. The [recommended configuration](https://github.com/llm-d/llm-d/blob/main/docs/architecture/router/configuration.md) weights the prefix-cache scorer 3 and the KV-cache utilization and queue scorers 2 each. With three pods and illustrative scores:

| Pod | Prefix-cache match (×3) | KV cache free (×2) | Short queue (×2) | Total |
|---|---|---|---|---|
| A: holds most of the prompt, but busy | 0.9 | 0.3 | 0.2 | 3.7 |
| B: cold cache, idle | 0.0 | 0.9 | 1.0 | 3.8 |
| C: holds part of the prompt, moderately loaded | 0.6 | 0.6 | 0.6 | 4.2 |

Pod C wins: a partial cache hit on a pod with room beats a big hit on a swamped one, the balance between cache affinity and load from Part 6. Other scorers cover token load, LoRA adapters, session affinity and predicted latency.

The EPP doesn't have to disaggregate every request. After it picks a decode pod, a **decider** checks how much of the prompt that pod already holds in its prefix cache. If only a short uncached suffix is left, the decode pod prefills it locally. If a large part is uncached, the EPP also picks a prefill pod. The `prefix-based-pd-decider` makes this choice per request; the reference P/D guide uses `always-disagg-pd-decider`, and SGLang deployments require it, because SGLang decode workers can't prefill locally.

### Scaling and running it

llm-d scales on signals from the EPP rather than GPU utilization, because, as its [autoscaling docs](https://github.com/llm-d/llm-d/blob/main/docs/architecture/autoscaling/README.md) put it, "GPU utilization can sit near 100% during active batching whether the server is lightly loaded or saturated." The recommended setup uses KEDA, a Kubernetes event-driven autoscaler, to scale each role on queue depth, token backlog or predicted latency. The earlier Workload Variant Autoscaler is now deprecated.

Two pools also mean two things to upgrade. v1.0's `DisaggregatedSet` resource manages prefill and decode replicas "as one logical unit," with lockstep upgrades. And for large mixture-of-experts models, the split combines with wide expert parallelism: llm-d's [wide-EP guide](https://github.com/llm-d/llm-d/tree/main/guides/wide-ep), covered in Part 10, runs DeepSeek-R1 as prefill and decode groups on 32 H200s linked by NIXL.

## Step 5: when splitting pays off, and when it doesn't

vLLM's documentation states: "Disaggregated prefill DOES NOT improve throughput." Splitting removes interference and adds freedom. It doesn't make the GPUs do more arithmetic, and it adds a network transfer, a second pool to balance and new ways to fail.

### Where it helps

Beyond the Buzz evaluated "hundreds of thousands of design points" and found that disaggregation "is most effective for prefill-heavy traffic patterns and larger models." Larger models are spread over more GPUs, which gives more room to choose different layouts for each phase. llm-d's guidance lines up: try it for medium to large models, "Longer input sequence lengths (e.g 10k ISL | 1k OSL, not 200 ISL | 200 OSL)," where ISL and OSL are input and output sequence lengths, and sparse MoE models that use wide expert parallelism.

| Workload | Split? | Why |
|---|---|---|
| Long prompts ([Part 11](/blog/long-context-inference/)), strict TPOT target | Yes | Prefill dominates and would stall decode |
| Large MoE model on many nodes | Yes | Each phase needs its own expert and data parallelism |
| Short prompts and short answers | Usually no | Little interference to remove; the transfer is pure overhead |
| Most of each prompt already cached | Per request | Prefill is small; llm-d's decider can skip the split |
| Small model on one GPU, light load | No | Chunked prefill is simpler and nearly as good |
| No RDMA between machines | No | TCP transfers are too slow for production |

### Rate matching

The ratio of prefill to decode instances, written xPyD (x prefill and y decode instances), has to match the traffic. Too few prefill instances and requests queue before their first token; too few decode instances and the decode pool fills up with caches waiting for a slot. Beyond the Buzz calls this **rate matching** and stresses that it must adapt as traffic changes.

The DeepSeek calculation from the start is a rate-matching example:

```python
input_per_s  = 608e9 / 86_400       # 7.04 million input tokens per second, on average
output_per_s = 168e9 / 86_400       # 1.94 million output tokens per second
prefill_nodes = input_per_s / 73_700   # about 95 nodes
decode_nodes  = output_per_s / 14_800  # about 131 nodes
print(prefill_nodes / (prefill_nodes + decode_nodes))   # 0.42
```

Input outnumbered output 3.6 to 1, but a node reads prompts 5 times faster than it writes answers, so decode still needs more machines. Change the traffic, say to reasoning models that write 10,000-token answers, and the ratio swings toward decode. A shared pool absorbs that swing automatically; two pools have to be rebalanced, which is why llm-d scales each role separately.

## In the real world

### DeepSeek

DeepSeek's [inference overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) states, "we have adopted prefill-decode disaggregation architecture." Prefill units span 4 nodes (expert parallelism over 32 GPUs) and decode units 18 nodes (over 144 GPUs), all H800s. Part 14 covers the whole stack.

### llm-d in practice

The llm-d [v0.5 release post](https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale) reports about 50,000 output tokens per second from 16 prefill and 16 decode B200 GPUs with wide expert parallelism, about 3,100 output tokens per second per decode GPU, on 1,000-token prompts and 1,000-token answers. Its reference P/D guide ships without benchmark numbers, because gains depend on the model and workload.

### Other stacks

- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) calls itself "the open-source, datacenter-scale inference stack." Like llm-d, it sits above vLLM, SGLang or TensorRT-LLM, with disaggregated prefill and decode, KV-aware routing, a KV Block Manager that offloads caches to CPU, SSD and remote storage, and a Planner that autoscales against latency targets. NIXL comes from the same project.
- [AIBrix](https://github.com/vllm-project/aibrix), from the vLLM project, is "a Kubernetes control plane for production LLM inference" whose gateway routes on prefix cache, load and prefill/decode placement.
- The System Design post [Serving Open LLMs at Scale](/blog/serving-open-llms-at-scale/) shows where disaggregation fits in a full fleet, next to quantization and speculative decoding.

## Common misconceptions

- **"Disaggregation makes the cluster faster."** It removes interference and lets each phase be tuned, which raises goodput under tight latency targets. vLLM's docs say it does not improve raw throughput.
- **"Shipping the KV cache costs as much as recomputing it."** A 4,096-token cache for Llama 3.1 70B takes 27 ms over one 400 Gb/s link, against 76 ms of prefill on 8 H100s at peak, and streaming it layer by layer hides almost all of that. Splitwise measured 0.8% added end-to-end latency.
- **"Every request should be disaggregated."** Short prompts, and prompts mostly cached on the decode pod, are cheaper to prefill where they'll decode. llm-d's decider makes that choice per request.
- **"Chunked prefill makes disaggregation unnecessary."** Every chunked step still carries some prompt work, and both phases share one parallelism layout. At small scale chunking is often enough; for large models and long prompts the split wins.
- **"Any network will do."** Without RDMA the transfer falls back to TCP, which llm-d's docs reserve for testing and development.

## Key takeaways

- Prefill is compute-bound and decode is memory-bound. Running them on separate GPUs ends prefill stalls in decode, lets each phase use its own parallelism and hardware, and lets TTFT and TPOT capacity scale separately.
- DistServe served up to 7.4 times more requests within latency targets, and Splitwise reported 1.4 times the throughput at 20% lower cost.
- The cost is moving the KV cache: 1.34 GB for a 4,096-token prompt on Llama 3.1 70B. Streaming it layer by layer over RDMA hides most of the transfer behind prefill, and MLA models like DeepSeek-V3 send 4.7 times less per token.
- llm-d runs the split on Kubernetes: a gateway asks the Endpoint Picker for a decode pod and a prefill pod, a sidecar runs the prefill remotely, and NIXL moves the cache. It reached v1.0 in October 2026.
- Splitting pays off for large models and long prompts. For short prompts, small models or slow networks, chunked prefill on one pool is simpler and usually as good.

## What's next

This part was about the largest deployments: hundreds of GPUs, RDMA networks and Kubernetes. [Part 13](/blog/local-llm-inference/) goes to the other end of the scale: running LLMs on your own machine with llama.cpp, the GGUF file format and Ollama, why Apple silicon's unified memory changes what fits, and how to work out which models your laptop can run.

## Further reading

- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (Zhong et al., OSDI 2024)](https://arxiv.org/abs/2401.09670): the goodput case for splitting, placement search and pull-based KV transfer.
- [Splitwise: Efficient generative LLM inference using phase splitting (Patel et al., 2023)](https://arxiv.org/abs/2311.18677): different GPUs and power caps per phase, and layer-wise KV transfer.
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (Qin et al., 2024)](https://arxiv.org/abs/2407.00079): disaggregation built around a cluster-wide KV cache, in production at Kimi.
- [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation (Mitra et al., 2025)](https://arxiv.org/abs/2506.05508): when disaggregation helps, rate matching and KV bandwidth requirements.
- [DeepSeek-V3/R1 Inference System Overview (DeepSeek, 2025)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md): production prefill and decode units with daily traffic statistics.
- [vLLM: Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) and [NixlConnector Usage Guide](https://docs.vllm.ai/en/latest/features/nixl_connector_usage.html): the KV connector API and its transfer backends.
- [SGLang: PD Disaggregation](https://docs.sglang.io/advanced_features/pd_disaggregation.html): prefill and decode separation with Mooncake and NIXL.
- [NIXL (NVIDIA)](https://github.com/ai-dynamo/nixl) and [Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake): the libraries that move KV caches between machines.
- [llm-d on GitHub](https://github.com/llm-d/llm-d), with its [disaggregated serving design](https://github.com/llm-d/llm-d/blob/main/docs/architecture/disaggregation/pd-disaggregation.md), [router](https://github.com/llm-d/llm-d/blob/main/docs/architecture/router/README.md) and [request scheduler](https://github.com/llm-d/llm-d/blob/main/docs/architecture/router/scheduling.md) docs, and the [v1.0.0 release notes](https://github.com/llm-d/llm-d/releases/tag/v1.0.0).
- [Gateway API Inference Extension (Kubernetes SIGs)](https://github.com/kubernetes-sigs/gateway-api-inference-extension): InferencePool and the inference gateway model llm-d builds on.
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) and [AIBrix](https://github.com/vllm-project/aibrix): two other open-source stacks for disaggregated serving.
- [Red Hat Launches the llm-d Community (Red Hat, 2025)](https://www.redhat.com/en/about/press-releases/red-hat-launches-llm-d-community-powering-distributed-gen-ai-inference-scale): the launch announcement and founding contributors.
