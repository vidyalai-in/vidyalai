# Series plan: LLM Fundamentals and LLM Inference & Optimization

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
