# Code behind the articles

The code the articles run, kept here so it isn't lost. The site build ignores this folder.

## llm-from-scratch/ (Build an LLM From Scratch)

| File | Part | What it is |
|---|---|---|
| `train_tokenizer.py`, `tokenizer.py`, `tokenizer-32k.json` | 2 | BPE training, the tokenizer, and the trained 32K vocabulary |
| `data.py` | 3 | Download, MinHash dedup, tokenizing, packing into shards, `ShardLoader` |
| `model.py` | 4 | `GPT`, `GPTConfig`, `MAIN` (88M) and `SMALL` (22M) |
| `train.py` | 5 | The training loop (`--compile --fused` for speed, see Part 6; `--snapshot-at` added in Part 7) |
| `snapshots.py` | 7 | Load saved weights, full validation loss, text completion from a snapshot |
| `sizes.py` | 8 | The scaling sweep's model sizes and total/non-embedding parameter counts |
| `evals.py` | 9 | Bits per byte over text, log-likelihood multiple-choice scoring (HellaSwag, ARC, PIQA), few-shot prompts, wrappers for our models and Hugging Face models |
| `sft.py` | 10 | Chat template on the reserved tokens, smol-smoltalk preparation, length-sorted padding, masked loss normalized per step, SFT training and chat generation |
| `dpo.py` | 11 | UltraFeedback pairs in the chat template, cached reference log-probs, sequence log-probs over reply tokens, the DPO loss and training loop, held-out pair evaluation |
| `sampling.py` | 11 | Batched sampling with a KV cache (greedy, temperature, top-k), stopping at `<\|end\|>` |
| `grpo.py` | 12 | The addition task and its checkers, the warm-start (optionally mixed with chat data), group advantages (GRPO and Dr. GRPO), the clipped GRPO loss with a k3 KL penalty, the training loop, pass@k |
| `distill.py` | 13 | The KD loss (temperature and hard-label mix), top-k teacher mass and storage math, a KD training entry point on `train.py`'s flags, batched teacher generation with folded grouped-query attention, SFT on teacher-written data |
| `memory.py` | 14 | Training-memory accounting (state, activations eager and compiled, Korthikanti's formula, ZeRO stages), block checkpointing (all, every k, selective), a chunked loss, LoRA layers, a CPU-offloaded AdamW |
| `export.py` | 15 | Export to a Hugging Face `LlamaForCausalLM` folder (RoPE row permutation, norm epsilon, tied embeddings, tokenizer and chat template) with a logit check; `experiments/t15` has the llama.cpp GGUF conversion wrapper, perplexity and benchmark scripts |

`experiments/t01` to `t15` hold each part's benchmark, ablation and chart-data scripts and small result files.

Not included (rebuildable or too large): the token shards in `data/` (about 2.5B tokens; rebuild with `data.py`), checkpoints, run logs, and downloaded papers and third-party source.

## llm-interpretability/ (Inside the Model: Interpretability)

| File | Part | What it is |
|---|---|---|
| `hooks.py` | 1 | Model loading with stated precision and weight processing, batched run-with-cache, direct logit attribution (LayerNorm or RMSNorm, folded or not), zero and mean ablation hooks, bootstrap CIs |
| `lens.py` | 2 | Residual-stream points, the logit lens, a tuned lens (Muon, SGD or Adam training), KL and top-1 evaluation |
| `probes.py` | 3 | Activations saved to disk, linear and MLP probes with held-out evaluation, control-task labels, mass-mean directions, bootstrap and paired-bootstrap CIs |
| `heads.py` | 4 | Per-head previous-token, duplicate-token and induction scores, mean and zero head ablation, K-composition scores with a random baseline, key-path ablation |
| `patching.py` | 5 | Logit-difference and probability metrics, normalized patching effects with bootstrap CIs, patch hooks and sweeps, path patching, Gaussian-noise embedding corruption for causal tracing |
| `toy.py` | 6 | Batched linear, ReLU-output and hidden-ReLU toy models of superposition, training, feature dimensionality, feature classification, per-neuron feature counts |
| `sae.py` | 7 | ReLU-L1, TopK (with AuxK) and JumpReLU SAEs, a checkpointed training loop, splice evaluation on GPT-2, top-activation tracking, a Gemma Scope loader and a bf16 Gemma runner |
| `attribution.py` | 8 | Helpers for circuit-tracer on Gemma 2 2B in 8 GB (GPU-side model build, lazy bf16 transcoders), transcoder fidelity, attribution patching |

`experiments/p01` to `p08` hold each part's scripts and small result files. They run on the TransformerLens 3.9.0 / SAELens environment described in the articles; Part 8's circuit-tracer scripts need a separate environment (circuit-tracer 0.5.0 pins `transformers<=4.57.3`). Not included: saved activations (Part 3 saved 5.9 GB; `collect.py` rebuilds them), tuned-lens and SAE weights, attribution graphs, Gemma 2 weights and Gemma Scope downloads, downloaded datasets (UD English EWT, GeoNames, FineWeb-Edu and WikiText samples; the `prep_*.py` scripts rebuild them) and Part 3's generated city statements (`prep_truth.py` rebuilds them from GeoNames, CC BY 4.0).

## agents-lab/ (Agentic AI)

| File | Part | What it is |
|---|---|---|
| `ledger.py` | 1 | API client, pricing, per-part and total budget caps, JSONL cost ledger |
| `harness.py` | 1 | The manual tool loop with prompt caching |
| `react.py` | 3 | `run_loop` with thinking, effort and timeout control; ReAct and plan-and-execute |
| `context.py` | 5 | Context strategies: sizing, context editing, compaction, memory file |
| `mcp_lab.py`, `mcp_demo.py`, `tickets_server.py` | 6 | MCP servers as harness tools, and the demo server |
| `multiagent.py` | 7 | A lead agent with a `delegate` tool that runs workers in fresh contexts |
| `coding.py` | 8 | Coding-agent tools (grep, read, string-replace edit, sandboxed test runner), repo map, arms |
| `computer.py` | 9 | Playwright executors for the computer and browser toolsets, localhost only |
| `research.py` | 10 | Web search and fetch server-tool loop with `pause_turn` handling |
| `evals.py` | 11 | pass@k and pass^k estimators, Wilson intervals, task bootstrap, paired power by simulation |
| `reliability.py` | 12 | Durable loop with checkpoints and write-ahead resume, fault injection, retries with jitter, idempotent payments backend |
| `sandbox.py` | 13 | bubblewrap and hardened Docker runners, fake canary token, allow/ask/deny policy hook |
| `costlat.py` | 15 | Loop with caching on, off or 1-hour; Sonnet-lead team; agent loop over Message Batches; latency percentiles |
| `finale.py` | 16 | The docs-maintenance agent: seven strict tools, sandboxed checks, permission policy, durable loop, patch output |

`experiments/a01` to `a16` hold each part's task sets, runners and analysis scripts (Part 14 only analyses Part 8's saved runs; it made no API calls).

The API key is read from `~/.config/vidyalai/anthropic.env` and is never stored here. Not included: the cost ledger (`ledger.jsonl`), run transcripts, fetched docs, and third-party source (Part 16 reviews a copy of the python-semver 3.1.0 docs; `build_corpus.py` rebuilds it from a checkout). Scripts use absolute paths from the machine they ran on (`/home/amit/.cache/vidyalai-scratch/...`); adjust them to rerun elsewhere.
