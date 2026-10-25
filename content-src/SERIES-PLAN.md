# Series plan: the explainer series

The agreed outline for the two explainer series. Each part follows the explainer template in
`AUTHORING.md`. File names: `src/content/blog/f<NN>-<slug>.md` and `i<NN>-<slug>.md`. Slugs below
are proposals until the article exists. Topics that the LLM System Design series already covers
(RAG and embedding search, evals and LLM-as-judge, hallucination, structured output, fine-tuning and
LoRA, routing, semantic caching) are linked, not re-explained.

## LLM Fundamentals (16 parts)

| # | Title (working) | Slug | Scope | Status |
|---|---|---|---|---|
| 1 | How a Large Language Model Works: The Big Picture | `how-llms-work` | Next-token prediction, the generation loop, tokens → vectors → blocks → logits → softmax → sampling, parameters, training in brief, recent models and MoE in brief, why the 2017 paper mattered | Published |
| 2 | Tokenization | `tokenization` | Why not characters or words; BPE step by step; byte-level BPE; vocab sizes; special tokens; why "strawberry" and arithmetic are hard; token counts differ by language; cost and context limits | Published |
| 3 | Neural networks: weights, gradient descent and backpropagation | `neural-networks-basics` | A neuron as weighted sum + nonlinearity; layers; loss; gradient descent with a tiny worked example; backpropagation intuition; learning rate; why GPUs (matrix multiplication) | Published |
| 4 | Embeddings and logits | `embeddings-and-logits` | Embedding matrix as lookup; vector similarity (dot product, cosine); meaning as direction; unembedding / LM head; weight tying; logits → softmax recap | Published |
| 5 | Attention: queries, keys and values | `attention-explained` | QKV with a small numeric example; scaled dot-product; softmax weights; causal mask; why √d; cost grows with sequence length squared | Published |
| 6 | Multi-head attention and the Transformer block | `transformer-block` | Multiple heads; the MLP; residual stream; layer norm; parameter counts per block; stacking | Published |
| 7 | Position: how Transformers know word order | `positional-encoding` | Attention is order-blind; sinusoidal; learned; RoPE; ALiBi; extending context (position interpolation, YaRN) | Published |
| 8 | Reading "Attention Is All You Need" | `attention-is-all-you-need` | Guided walk through the paper: encoder-decoder, cross-attention, training setup, results, what changed since | Published |
| 9 | Encoders, decoders and GPT, and how sampling works | `encoders-decoders-gpt` | Encoder-only (BERT, embedding models), decoder-only (GPT), encoder-decoder (T5); greedy, temperature, top-k, top-p, min-p, beam search | Published |
| 10 | The modern LLM block: what changed since 2017 | `modern-llm-architecture` | Pre-norm, RMSNorm, SwiGLU, RoPE, MQA/GQA, MLA, no biases; a 2017 block next to a 2026 block | Published |
| 11 | Mixture of experts | `mixture-of-experts` | Router, experts, top-k; total vs active parameters; load balancing, shared experts; DeepSeek, Kimi, Qwen examples | Published |
| 12 | Pretraining: data, scaling laws and training at scale | `pretraining-scaling-laws` | Data collection and filtering, dedup, mixtures; 6ND; Kaplan vs Chinchilla; AdamW, learning-rate schedules, mixed precision; thousands of GPUs | Published |
| 13 | From base model to assistant: SFT, RLHF and DPO | `base-model-to-assistant` | Instruction tuning; reward models; RLHF with PPO; DPO; chat templates; safety tuning | Published |
| 14 | Reasoning models and test-time compute | `reasoning-models` | Chain of thought; RL with verifiable rewards (o1, DeepSeek-R1, GRPO); thinking tokens; test-time compute scaling | Published |
| 15 | Multimodal models: how LLMs see and hear | `multimodal-llms` | Image patches and vision encoders (ViT), projecting into the token stream, audio tokens, image generation in brief | Published |
| 16 | Inside the black box: interpretability | `llm-interpretability` | Features, superposition, sparse autoencoders, circuits, what we can and can't explain | Published |

## LLM Inference & Optimization (14 parts)

| # | Title (working) | Slug | Scope | Status |
|---|---|---|---|---|
| 1 | How LLM Inference Works: Prefill, Decode and the Memory Wall | `how-llm-inference-works` | Prefill vs decode, TTFT/TPOT/goodput, roofline, batching table, KV cache intro, map of the series, engines | Published |
| 2 | The KV cache | `kv-cache` | What is stored and why; size math; MQA, GQA, MLA; cache precision; one long user vs many short | Published |
| 3 | How GPUs work for LLMs | `gpus-for-llms` | SMs, tensor cores, HBM vs SRAM, memory hierarchy, number formats (FP32 → FP4), NVLink, H100/H200/B200, other accelerators in brief | Published |
| 4 | Continuous batching and chunked prefill | `continuous-batching` | Static vs dynamic vs continuous (Orca); scheduling; chunked prefill (Sarathi); latency/throughput tradeoff | Published |
| 5 | PagedAttention and vLLM | `pagedattention-vllm` | Fragmentation; blocks and block tables; copy-on-write; vLLM architecture (scheduler, engine) | Published |
| 6 | Prefix caching and KV cache offloading | `prefix-caching` | Automatic prefix caching; RadixAttention; cache-aware routing; LMCache and CPU/SSD tiers; provider prompt caching | Published |
| 7 | FlashAttention and fused kernels | `flashattention` | Why naive attention is memory-bound; tiling and online softmax; FA2/FA3; kernel fusion, CUDA graphs, torch.compile | Published |
| 8 | Quantization | `quantization` | Number formats; weight-only (GPTQ, AWQ), activations (SmoothQuant), FP8/FP4, KV cache quantization; quality checks | Published |
| 9 | Speculative decoding | `speculative-decoding` | Draft and verify; acceptance-rate math; Medusa, EAGLE, MTP; when it helps | Published |
| 10 | Tensor, pipeline and expert parallelism | `model-parallelism` | Splitting models across GPUs; communication costs; NVLink vs network; MoE expert parallelism | Published |
| 11 | Long-context inference | `long-context-inference` | Quadratic attention; sliding windows; sparse attention (DeepSeek sparse attention, NSA); KV eviction and compression | Published |
| 12 | Disaggregated serving and llm-d | `disaggregated-serving-llm-d` | Splitting prefill and decode; KV transfer; llm-d architecture, inference scheduler, autoscaling | Published |
| 13 | Running LLMs on your own machine | `local-llm-inference` | llama.cpp, GGUF, Ollama, Apple silicon unified memory, sizing what fits, quantization in practice | Published |
| 14 | Case study: DeepSeek's inference stack and DeepInfer | `deepseek-inference-deepinfer` | V3/R1 serving overview, MLA, MoE EP with DeepEP and EPLB, FlashMLA, DeepGEMM, MTP, DeepInfer | Published |

## The Math Behind LLMs (16 parts)

Assumes high-school algebra; builds up vectors, derivatives, probability and logs from scratch. Each
part derives what the "Builds on" parts explain in words (f = Fundamentals, i = Inference) and links
them instead of repeating them. Files: `src/content/blog/m<NN>-<slug>.md`. Formulas use KaTeX (see
AUTHORING.md, "Math notation").

| # | Title (working) | Slug | Scope | Builds on | Status |
|---|---|---|---|---|---|
| 1 | The math you need for LLMs, and the shapes inside one | `math-for-llms` | Roadmap of the series; scalars, vectors, matrices, tensors; the shape of every tensor in one forward pass of a real model; matrix multiply as many dot products; reading einsum notation | f01, f06 | Published |
| 2 | Vectors and dot products | `vectors-dot-products` | Length, angle, cosine, projection; the dot product as "how much one vector points along another"; high-dimensional geometry (random vectors are nearly perpendicular); embedding arithmetic | f04, f16 | Published |
| 3 | Matrices as transformations | `matrices-linear-maps` | Matrix × vector as a linear map; composition; rank; why a stack of linear layers collapses into one; nonlinearity; batching as matmul; counting FLOPs of a matmul (2mnk) | f03, f06 | Published |
| 4 | Probability and the chain rule of language | `probability-language-models` | Distributions; conditional probability; p(text) = a product of next-token probabilities; logs and log-probabilities; sampling; temperature, top-k and top-p as operations on the distribution | f01, f09 | Published |
| 5 | Softmax and log-sum-exp | `softmax-math` | Deriving softmax; invariance to shifts; overflow and the max trick; log-softmax; temperature limits; softmax as the smooth version of argmax; online softmax (the core of FlashAttention) | f04, i07 | Published |
| 6 | Entropy, cross-entropy and perplexity | `cross-entropy-perplexity` | Surprise, entropy, cross-entropy as the training loss, nats vs bits, perplexity, bits per byte, KL divergence, the link between language modeling and compression | f03, f12 | Published |
| 7 | Derivatives, gradients and the chain rule | `derivatives-gradients` | Slope; partial derivatives; the gradient as the direction of steepest climb; the chain rule; Jacobians in plain terms; numerical vs analytic derivatives | f03 | Published |
| 8 | Backpropagation by hand | `backpropagation-math` | Computational graphs; reverse-mode autodiff; deriving the backward pass of matmul, ReLU and GELU, softmax + cross-entropy (p − y); why backward costs about 2× forward; activation memory; checking gradients numerically; a tiny autograd in Python | f03, f12 | Published |
| 9 | Optimizers: from gradient descent to AdamW | `optimizers-adam` | SGD, mini-batch noise, momentum, RMSProp, Adam's bias correction, AdamW's decoupled weight decay, warm-up and cosine schedules, gradient clipping math, newer optimizers (e.g. Muon, if verifiable) | f03, f12 | Published |
| 10 | Keeping signals stable: initialization, normalization and residuals | `initialization-normalization` | Variance through layers; Xavier and He initialization; vanishing and exploding gradients; LayerNorm vs RMSNorm math; why residual connections keep gradients flowing; pre-norm vs post-norm; μP in brief | f06, f10 | Published |
| 11 | The math of attention | `attention-math` | Scaled dot-product attention derived; the variance argument for √d; the causal mask as −∞; the softmax Jacobian; O(n²d) cost; multi-head attention as low-rank pieces; GQA and MLA as factorizations; linear attention and state-space recurrences (Mamba) as the way around the n² cost | f05, f06, i02 | Published |
| 12 | Rotations, complex numbers and RoPE | `rope-math` | 2D rotation matrices; complex numbers as rotations; deriving RoPE's relative-position property; frequency bases; context-extension math (position interpolation, NTK, YaRN); ALiBi | f07, i11 | Published |
| 13 | Low rank and the SVD | `low-rank-svd` | Rank, SVD and truncated SVD; Eckart-Young; why LoRA's update is low rank; low-rank structure in MLA and in weight compression; worked SVD of a real weight matrix | f10, i02 | Published |
| 14 | Floating point and quantization math | `floating-point-math` | Sign, exponent and mantissa; FP32, BF16, FP16, FP8 and FP4 ranges and precision; rounding error and accumulation; mixed precision and loss scaling; quantization scales and zero points; group size vs error; the error statistics behind GPTQ/AWQ in brief | i03, i08 | Published |
| 15 | Counting: parameters, FLOPs, memory and scaling laws | `counting-flops-scaling-laws` | Exact parameter counts per layer; deriving 6ND; KV cache formula; arithmetic intensity; power laws and log-log fits; deriving the compute-optimal N and D (Chinchilla); mixture of experts: total vs active parameters, the router's top-k softmax and the load-balancing loss | f12, i01 | Published |
| 16 | The math of alignment: policy gradients, PPO, DPO and GRPO | `alignment-math` | Expected reward; the REINFORCE gradient; baselines and advantages; PPO's clipped objective; Bradley-Terry; the KL-regularized optimum; deriving DPO from it; GRPO's group baseline; the speculative-decoding correctness proof as a bonus example of rejection sampling. Closes the series. | f13, f14, i09 | Published |

## LLMs on the GPU: Kernels and Hardware (15 parts)

One level below Inference Parts 3 (GPUs) and 7 (FlashAttention): real CUDA C++ and Triton code, timed
on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB). Data center features (TMA, wgmma, tcgen05, NVLink)
come from NVIDIA's docs. Files: `src/content/blog/g<NN>-<slug>.md`. Only one GPU-measuring article is
written at a time.

| # | Title (working) | Slug | Scope | Builds on | Status |
|---|---|---|---|---|---|
| 1 | From torch.matmul to silicon | `gpu-from-python-to-silicon` | What happens when PyTorch runs a layer: the dispatcher, kernel launch, CUDA streams and asynchronous execution, launch overhead; roadmap of the series; a first measured trace of one decode step | i03, i07 | Published |
| 2 | The CUDA programming model | `cuda-programming-model` | Threads, warps, blocks, grids; SIMT and warp divergence; SMs and occupancy; writing, compiling and timing a first kernel (vector add) and seeing it hit the bandwidth limit; catching bugs with compute-sanitizer | i03 | Published |
| 3 | The memory hierarchy | `gpu-memory-hierarchy` | Registers, shared memory, L1, L2, DRAM; coalesced vs strided access measured; shared-memory bank conflicts; a copy-kernel bandwidth test vs the spec; caches and access patterns | i03 | Published |
| 4 | Profiling and the roofline | `gpu-profiling-roofline` | Nsight Systems and Nsight Compute; the PyTorch profiler; achieved bandwidth and FLOPs; plotting our kernels on the 4060's roofline; finding the bottleneck; clocks, power limits and throttling on a laptop GPU | i01, i03 | Published |
| 5 | Writing a fast matmul, step by step | `cuda-matmul-optimization` | Naive → coalesced → shared-memory tiling → register blocking → vectorized loads, measured in GFLOPS at each step vs cuBLAS | m03 | Published |
| 6 | Tensor cores | `tensor-cores` | What an MMA instruction does; WMMA/mma.sync; data layouts and swizzling; FP16, BF16 and FP8 on Ada; Hopper wgmma and TMA and Blackwell tcgen05 from docs; CUTLASS and CuTe in brief; reading PTX and SASS; measured tensor-core vs CUDA-core matmul | i03, g05 | Published |
| 7 | Triton: GPU kernels in Python | `triton-kernels` | Block-level programming; vector add, fused softmax and matmul in Triton; autotuning; how it compiles (to PTX); measured against PyTorch | g05 | Published |
| 8 | Reductions and normalization kernels | `gpu-reductions` | Parallel reduction trees; warp shuffles; block-level reductions; softmax, RMSNorm and LayerNorm kernels; numerical stability on the GPU; sampling kernels (argmax, top-k, top-p) | m05, m10 | Published |
| 9 | Kernel fusion, torch.compile and CUDA graphs | `kernel-fusion-cuda-graphs` | Why elementwise ops are memory-bound; fusing SwiGLU, residual add and norm; torch.compile and TorchInductor's generated Triton; CUDA graphs removing launch overhead, measured in decode; the PyTorch caching allocator and memory fragmentation | i07 | Published |
| 10 | Writing an attention kernel | `attention-kernel` | FlashAttention-style tiling with online softmax in Triton; measured vs naive and vs PyTorch SDPA; decode attention, split-KV (Flash-Decoding) and paged-KV indexing | i05, i07, m11 | Published |
| 11 | Low-precision kernels | `low-precision-kernels` | Dequantize-in-kernel INT4/INT8 weight-only GEMM (e.g. Marlin, llama.cpp kernels); FP8 scaling factors; microscaling (MX/NVFP4) formats; when quantization actually speeds things up, measured | i08, m14 | Published |
| 12 | Mixture-of-experts kernels | `moe-kernels` | Token routing and permutation; grouped GEMM; fused MoE kernels (vLLM, DeepGEMM); load imbalance on the GPU | f11, i10 | Published |
| 13 | Many GPUs: NVLink, NCCL and collectives | `nccl-collectives` | Ring and tree all-reduce algorithms and their cost model; all-to-all; NVLink/NVSwitch vs PCIe vs InfiniBand; overlapping communication with compute; NVSHMEM in brief | i10, i12 | Published |
| 14 | Beyond NVIDIA: AMD, Apple, TPUs and other accelerators | `ai-accelerators` | AMD ROCm/HIP and MI-series; Apple Metal and MLX; Google TPU systolic arrays and XLA; AWS Trainium; Groq, Cerebras and others; how the same ideas map onto each | i03, i13 | Published |
| 15 | One token, kernel by kernel | `token-kernel-trace` | A full Nsight trace of Llama-class decode on the 4060: every kernel, its time, and its place on the roofline, tying the series together. Closes the series. | all | Published |

## Writing order and reading path (series 6-10)

Writing order: phase 1 Agentic AI + From Scratch (alternating batches of four; API vs GPU, no contention), phase 2 Interpretability + Measuring LLMs, phase 3 Beyond Text. Reading path in `SERIES`: Fundamentals, Math, From Scratch, Interpretability, Inference, GPU, Beyond Text, Measuring LLMs, System Design, Agentic AI. Each series is added to `src/data/series.mjs` (and its category to `categories.ts`) when its Part 1 is published.

## Agentic AI: How AI Agents Work (16 parts)

Series slug `agentic-ai`, short name "Agents", file prefix `a`, category `agents-tool-use`. Each part is a standalone deep dive; every experiment runs on one shared lab harness (Anthropic Python SDK, manual tool loop, prompt caching, usage ledger) introduced in Part 1. Models: `claude-haiku-5-5` for sweeps and repeats, `claude-sonnet-5-5` for small headline comparisons. The whole project has a hard API budget of $90: $75 for this series (per-part caps below), $15 for the Evals series. Builds on System Design 10 (tool-using agents), 11 (agent loops), 12 (long-term memory), 17 (prompt injection) and 23 (multi-agent tracing), written SD10 etc. below: link, do not repeat.

| # | Title (working) | Slug | Scope | Builds on | API budget | Status |
|---|---|---|---|---|---|---|
| 1 | What an agent is: the loop, measured | `what-is-an-ai-agent` | LLM + tools + loop; workflows vs agents; the 150-line lab harness; tokens, cost and latency per step | SD10, f14 | $1 | Published |
| 2 | How tool calling works inside the model | `tool-calling-internals` | Tool schemas in the prompt, tool-use blocks, strict mode, parallel calls; malformed-call rate measured with strict on and off | SD13, SD10 | $2 | Published |
| 3 | Reasoning and acting: ReAct, planning and thinking | `react-planning` | ReAct, plan-and-execute, adaptive thinking and effort levels; accuracy vs tokens on a fixed task set | f14 | $5 | Published |
| 4 | Reflection and self-correction | `agent-self-correction` | Reflexion, critique loops, verifier tools (tests); when self-critique helps and when it hurts, measured | a03 | $4 | Published |
| 5 | Context engineering | `context-engineering` | What fills the window; sizing tool results; compaction vs context editing vs memory files; prompt-caching economics on long tasks | SD12, i06 | $5 | Published |
| 6 | MCP: connecting agents to tools | `model-context-protocol` | The protocol, transports, servers; tool search and deferred loading; tool-choice accuracy as tool count grows | a02 | $3 | Published |
| 7 | Multi-agent systems and agent-to-agent protocols | `multi-agent-systems` | Orchestrator-workers, sub-agents, A2A; single agent vs orchestrator on the same tasks (tokens ×, accuracy) | SD23 | $6 | Published |
| 8 | Coding agents | `coding-agents` | How SWE-bench-style tasks are solved: repo search, edit tools, test loops; a 20–30-task mini-benchmark built from real repos | a03, a04 | $10 | Published |
| 9 | Computer-use and browser agents | `computer-use-agents` | Screenshots, actions, the computer toolset; image-token cost; a small sandboxed browser task set | a01 | $8 | Published |
| 10 | Research agents: search, read, cite | `research-agents` | Web search and fetch tools, deep-research loops, citation accuracy checked by hand | a05 | $5 | Published |
| 11 | Evaluating agents | `agent-evals` | SWE-bench, τ-bench, OSWorld, GAIA; pass@k vs pass^k; run-to-run variance and cost-normalized scores, measured | SD15, SD16 | $8 | Published |
| 12 | Long-horizon reliability | `agent-reliability` | Compounding error (p^n), retries, idempotent tools, checkpoints, task budgets, stopping rules | SD11 | $3 | Published |
| 13 | Sandboxing and permissions | `agent-sandboxing` | Least privilege, containers, network and secret isolation, human approval gates (prompt injection itself stays in SD17) | SD17 | $2 | Published |
| 14 | Training models to be agents | `training-agentic-models` | RL in tool environments, verifiable rewards, synthetic environments, trajectory data; papers only | m16, f14 | $0 | Published |
| 15 | Cost and latency of agents | `agent-cost-latency` | Cost per completed task; caching, effort, Haiku workers under a Sonnet lead, batch; measured | SD06, SD09 | $5 |  |
| 16 | An agent end to end | `agents-in-production` | Finale: one useful agent built and evaluated with everything above (e.g. a docs-maintenance agent), its limits, series review | all | $5 |  |

## Build an LLM From Scratch (15 parts)

Series slug `llm-from-scratch`, short name "From Scratch", file prefix `t`, category `llm-training`. A GPT/Llama-style model of about 50-125M parameters, trained on the RTX 4060 Laptop GPU (8 GB) on a FineWeb-Edu sample (TinyStories for first quick runs). Code is PyTorch; each part extends one code folder but explains its own piece in full.

| # | Title (working) | Slug | Scope | Builds on | Status |
|---|---|---|---|---|---|
| 1 | What a laptop can train | `build-llm-from-scratch` | Roadmap; compute budget with 6ND; choosing model size, tokens and time | m15, f12 | Published |
| 2 | Training a BPE tokenizer | `train-bpe-tokenizer` | BPE from scratch, vocab size trade-offs, compression measured | f02 | Published |
| 3 | The pretraining data pipeline | `pretraining-data-pipeline` | Download, quality filters, MinHash dedup, shuffling, packing into token shards | f12 | Published |
| 4 | Writing the model | `write-gpt-model` | RMSNorm, RoPE, GQA, SwiGLU in about 200 lines; parameter count checked against the formula | f06, f10, m15 | Published |
| 5 | The training loop | `llm-training-loop` | AdamW, warm-up and cosine schedule, clipping, BF16, gradient accumulation; first loss curve | m09, m10 | Published |
| 6 | Making training fast on one GPU | `fast-llm-training` | torch.compile, SDPA/FlashAttention, fused AdamW, MFU measured | g09, g10 | Published |
| 7 | The pretraining run | `pretraining-run` | The multi-hour run; reading loss curves, spikes, held-out loss, samples over time | t05 |  |
| 8 | Scaling laws on a laptop | `scaling-laws-experiment` | Five model sizes, a fitted power law, comparison with Chinchilla | m15 |  |
| 9 | Evaluating a base model | `evaluate-base-model` | Perplexity, bits per byte, few-shot multiple choice | m06 |  |
| 10 | Supervised fine-tuning | `sft-from-scratch` | Chat template, loss masking, instruction data | f13 |  |
| 11 | Preference tuning with DPO | `dpo-from-scratch` | Preference pairs, the DPO loss, before/after behaviour | m16 |  |
| 12 | RL with verifiable rewards (GRPO) | `grpo-from-scratch` | GRPO on arithmetic or format tasks; reward curves | m16, f14 |  |
| 13 | Synthetic data and distillation | `distillation-synthetic-data` | Teacher-generated data (Haiku via batch, out of the Evals budget, or an open model); logit vs sequence distillation | t10 |  |
| 14 | Fitting bigger models in memory | `training-memory` | Memory math, activation checkpointing (measured), ZeRO/FSDP stages explained | i10, g13 |  |
| 15 | From our model to frontier models | `frontier-training-recipes` | Finale: published recipes (Llama 3, DeepSeek-V3, OLMo) vs ours; export, quantize, run in llama.cpp; series review | all |  |

## Inside the Model: Interpretability (12 parts)

Series slug `llm-interpretability`, short name "Interpretability", file prefix `p`, category `interpretability`. Models run locally: GPT-2 small, Qwen2.5-0.5B/1.5B, Gemma 2 2B with Gemma Scope SAEs, and the From Scratch checkpoints.

| # | Title (working) | Slug | Scope | Builds on | Status |
|---|---|---|---|---|---|
| 1 | Opening the black box | `what-is-interpretability` | Goals, methods, what counts as an explanation | f16 |  |
| 2 | The residual stream and the logit lens | `logit-lens` | Reading predictions layer by layer; tuned lens | f06 |  |
| 3 | Linear probes | `linear-probes` | What activations encode; probes for truth and syntax; probe pitfalls | m02 |  |
| 4 | Attention heads and induction heads | `induction-heads` | Head roles, induction heads and in-context learning, measured | m11 |  |
| 5 | Activation patching and causal tracing | `activation-patching` | IOI-style patching, locating factual recall | p02 |  |
| 6 | Superposition | `superposition` | Toy models reproduced; more features than neurons | m13 |  |
| 7 | Sparse autoencoders | `sparse-autoencoders` | Train an SAE on one layer; explore Gemma Scope features | p06 |  |
| 8 | Circuits and attribution graphs | `circuit-tracing` | Transcoders and attribution graphs with open tools, if runnable on 8 GB | p07 |  |
| 9 | Steering with activations | `activation-steering` | Steering vectors, the refusal direction, side effects measured | p03 |  |
| 10 | What training builds inside a model | `training-dynamics-interpretability` | When induction heads and features form, using From Scratch checkpoints | t07 |  |
| 11 | Interpretability for safety | `interpretability-for-safety` | Deception and sycophancy probes, model diffing, audits; research-based | p09 |  |
| 12 | Limits and open problems | `interpretability-open-problems` | Finale: what works, what doesn't, series review | all |  |

## Beyond Text: Images, Speech and Video (13 parts)

Series slug `multimodal-ai`, short name "Multimodal", file prefix `v`, category `multimodal`. Local models that fit 8 GB (CLIP, a small VLM, SD 1.5 or SDXL-Turbo, Whisper, Kokoro with stock voices); Claude vision via Haiku from the Evals budget.

| # | Title (working) | Slug | Scope | Builds on | Status |
|---|---|---|---|---|---|
| 1 | How AI sees, hears and draws | `multimodal-ai-overview` | Roadmap; everything as tokens or latents | f15 |  |
| 2 | Vision transformers | `vision-transformers` | Patches, position, ViT measured | f06 |  |
| 3 | CLIP and contrastive learning | `clip-contrastive-learning` | The contrastive loss, zero-shot classification run locally | m06 |  |
| 4 | Vision-language models | `vision-language-models` | Connectors (LLaVA style), image-token counts and cost | v02, v03 |  |
| 5 | Diffusion from scratch | `diffusion-math` | DDPM math; train a tiny diffusion model on the laptop | m04, m07 |  |
| 6 | Latent diffusion | `latent-diffusion` | VAE, U-Net, classifier-free guidance; Stable Diffusion run and timed | v05 |  |
| 7 | Diffusion transformers and flow matching | `flow-matching-dit` | DiT, rectified flow (SD3/FLUX family) | v06 |  |
| 8 | Autoregressive and unified image generation | `autoregressive-image-generation` | VQ tokens, native image generation in LLMs; research-based | v04 |  |
| 9 | Speech recognition with Whisper | `speech-recognition-whisper` | Log-mel spectrograms, encoder-decoder; WER measured | f15 |  |
| 10 | Text-to-speech | `text-to-speech` | Kokoro and neural codecs; real-time factor measured | v09 |  |
| 11 | Audio tokens and voice agents | `voice-agents` | EnCodec/Mimi codecs, speech-to-speech models, latency budget | v10, a01 |  |
| 12 | Video generation | `video-generation` | Spacetime patches, video diffusion; mostly research | v07 |  |
| 13 | Serving multimodal models | `multimodal-inference` | Finale: image and audio token costs, encoder caching, series review | i01 |  |

## Measuring LLMs: Evals and Benchmarks (11 parts)

Series slug `llm-evals`, short name "Evals", file prefix `e`, category `evaluation-reliability`. API runs use `claude-haiku-5-5` through the Batch API; budget $15 (shared with v04 and t13).

| # | Title (working) | Slug | Scope | Builds on | Status |
|---|---|---|---|---|---|
| 1 | Why measuring LLMs is hard | `why-llm-evals-are-hard` | Validity, saturation, Goodhart | SD15 |  |
| 2 | How benchmarks are built | `how-benchmarks-are-built` | MMLU to GPQA to Humanity's Last Exam; item writing and validation | e01 |  |
| 3 | The statistics of evals | `eval-statistics` | Confidence intervals, sample size, paired tests; measured on a benchmark subset | m04 |  |
| 4 | Prompt and harness sensitivity | `eval-harness-sensitivity` | Format, few-shot choice and answer extraction move scores; measured | e03 |  |
| 5 | Contamination and memorization | `benchmark-contamination` | Detection methods, tested on open models | t09 |  |
| 6 | Grading open-ended outputs | `grading-generative-outputs` | Reference metrics vs rubric graders vs LLM judges | SD16 |  |
| 7 | Arenas and Elo rankings | `chatbot-arena-elo` | Bradley-Terry, style control, arena biases | m16 |  |
| 8 | Code and math benchmarks | `code-math-benchmarks` | pass@k, verifiers, HumanEval to SWE-bench and AIME | a11 |  |
| 9 | Long-context evals | `long-context-evals` | Needle tests vs RULER-style tasks | i11 |  |
| 10 | Safety and capability evals | `safety-evals` | Dangerous-capability evals, refusal benchmarks, red-teaming; research-based | e01 |  |
| 11 | Reading a model card, building your own eval | `reading-model-cards` | Finale: interpreting reported numbers; links SD15 | all |  |
