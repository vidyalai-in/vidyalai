---
title: "Fine-Tuning a Small Model with LoRA on 5,000 Examples"
description: "A practical LoRA recipe for an 8B model on 5K examples: data hygiene, hyperparameters, a real GPU memory budget, and eval-driven checkpoint selection."
slug: "lora-fine-tuning-5k-examples"
category: "fine-tuning"
tags: ["fine-tuning", "lora", "evals", "gpu", "serving"]
seriesOrder: 20
pubDate: 2026-02-06
updatedDate: 2026-10-16
sourceDoc: "1peTYAI7ps58PRKJWERCzNUnsehYjGOwhNRHViG8g-bs"
---

## TL;DR

With 5,000 examples, data quality and the eval affect the outcome more than any hyperparameter does, so spend most of the effort there. For a 7-9B instruct model, apply LoRA (low-rank adaptation) to all linear layers (attention and MLP) with rank 16-32, alpha at 1-2 times the rank, and a learning rate around 1e-4 to 2e-4, roughly 10 times the learning rate you would use for full fine-tuning. Train for 2-3 epochs with cosine decay, 3-5% warmup, an effective batch size of 32-64 and loss on completion tokens only. In bf16 the frozen 8B base weights alone take about 16 GB, so with activations a comfortable budget is 40-80 GB of GPU memory; QLoRA (a 4-bit base, about 6 GB of weights) fits on a 24 GB card. Each run takes minutes on one H100. Select the checkpoint on a held-out task metric, not on validation loss, and check for regressions in general capability before shipping.

## Requirements to pin down first

**What is the task, and what does success look like?** Classification or extraction (checkable), structured generation, or open-ended style? This determines the eval, and whether the job is SFT (supervised fine-tuning) only or SFT followed by DPO (direct preference optimization).

**Is there a prompted baseline?** If a frontier model with a good prompt scores 92% and you need 95% at a tenth of the cost, this is a distillation problem and the target is clear. If you haven't yet decided that fine-tuning is the right tool, start with [Fine-Tuning vs RAG vs Prompting](/blog/fine-tuning-vs-rag-vs-prompting/).

**Where did the 5,000 examples come from?** Human-written, synthetic or mined from logs? Are they deduplicated and checked for label noise? Do they overlap with the eval?

**What serving constraints apply?** Will you serve many adapters on one shared base (multi-LoRA serving, as in vLLM or S-LoRA), merge the adapter into the weights, or run on-device with a quantized model? This choice affects quantization and deployment, so make it early.

## Choosing the base model

Choose the base model by its prompted performance on your task. A stronger base with LoRA usually beats a weaker base with full fine-tuning.

This post uses Qwen3-8B (dense, text-only, Apache 2.0, 8.19B parameters) as the worked example, with Llama 3.1 8B numbers alongside for comparison. Newer small open models exist as of October 2026, such as Qwen3.5-9B and the Gemma 4 family, and they are worth evaluating as bases. Some of them use hybrid linear-attention layers and multimodal wrappers, so their LoRA target-module names and parameter counts differ. The method below carries over unchanged; recompute the arithmetic from the model's `config.json`.

## The data: where the outcome is decided

![Flowchart of the LoRA training and evaluation pipeline. In the data stage, 5,000 examples are deduplicated and 200 are reviewed by hand, then split into 4,500 train, 250 validation and 250 test, stratified by category. In the training stage, LoRA runs with rank 16 on all linear layers sweep the learning rate over 2 to 3 epochs and save checkpoints. In the evaluation stage, the checkpoint is picked by task metric on the validation split, then compared with baselines on the test split with bootstrap confidence intervals, then checked against a regression suite. If the regression suite fails, the loop returns to fixing the data or learning rate. If it passes, the adapter is merged or served as multi-LoRA, quantized and re-evaluated, and shipped as a versioned artifact.](/diagrams/q20-lora-pipeline.svg "The LoRA pipeline: data hygiene first, then a learning-rate sweep, then eval-driven selection and a regression gate.")

Most of the pipeline is data preparation and evaluation; the training step itself is the shortest stage.

- Split the data into about 4,500 train / 250 validation / 250 test, stratified by category. If the training data is synthetic, keep a separate, human-verified test set. Deduplicate with near-duplicate hashing across the splits so the test set doesn't leak into training.
- Do a quality pass: review 200 examples by hand. With 5,000 samples, 5% label noise is 250 mislabeled examples. Filter with an LLM judge or a rules check, and fix systematic errors at the source. As LIMA showed, 1,000 excellent examples often beat 10,000 mediocre ones.
- Use the base model's exact chat template, including its special tokens. Mask the loss on system and user tokens and train only on assistant tokens. Packing several short sequences into one training sequence, with correct attention masking so packed examples can't attend to each other, raises throughput by 2-3x.
- Check coverage by comparing the distribution against production traffic. Add hard negatives and "abstain" examples if the model has to refuse or say "not found".

## The training recipe

### Target modules and rank

Apply LoRA to all linear projections: `q`, `k`, `v`, `o`, `gate`, `up` and `down`. Attention-only LoRA underperforms noticeably. Thinking Machines' "LoRA Without Regret" (2025) found that the MLP adapters do most of the work: attention-only LoRA underperformed MLP-only LoRA even at matched parameter counts.

For rank, use `r=16` as the default, and `r=32-64` if the task is far from pretraining (a new language or a domain-specific language). Higher rank rarely helps with 5,000 examples and makes overfitting easier.

The trainable parameter count is `r × Σ (d_in + d_out)` over every adapted matrix. For Qwen3-8B (hidden size 4,096, MLP size 12,288, 8 key-value heads of dimension 128, 36 layers):

| Projection | d_in + d_out |
|---|---|
| q (4,096 to 4,096) | 8,192 |
| k, v (4,096 to 1,024 each) | 5,120 each |
| o (4,096 to 4,096) | 8,192 |
| gate, up (4,096 to 12,288 each) | 16,384 each |
| down (12,288 to 4,096) | 16,384 |
| **Per layer** | **75,776** |

At `r=16` that is `16 × 75,776 × 36 ≈ 43.6M` trainable parameters, 0.53% of 8.19B. For Llama 3.1 8B (MLP size 14,336, 32 layers) the same formula gives about 41.9M, 0.52% of 8.03B.

### Alpha and scaling

Set alpha to 16-32, so that alpha/r is 1-2. If you sweep rank, use rsLoRA scaling (`alpha / sqrt(r)` instead of `alpha / r`) so the effective learning rate doesn't change with rank.

### Learning rate

Use 1e-4 to 2e-4 with AdamW, weight decay 0-0.01, a cosine schedule and 3-5% warmup. The learning rate is the most sensitive knob, so sweep it: {5e-5, 1e-4, 2e-4, 4e-4}. "LoRA Without Regret" found the optimal LoRA learning rate to be consistently about 10x the full fine-tuning rate (a fitted multiplier of 9.8 across 14 Llama and Qwen models), and suggests going somewhat higher for very short runs.

### Batch size and epochs

Use an effective batch of 32-64 examples for 2-3 epochs. With 4,500 training examples at batch 32, that is about 141 steps per epoch, or roughly 280-420 steps in total (about 140-210 at batch 64). Don't push the batch much higher: LoRA is less tolerant of large batches than full fine-tuning, and raising the rank doesn't fix that. More than 3 epochs on 5,000 examples usually overfits, and the model starts memorizing phrasing.

### Dropout

Use 0.05-0.1 dropout on the LoRA layers; it helps on small datasets.

In Hugging Face PEFT, the defaults above look like this:

```python
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
```

## The GPU memory budget

A common shortcut says "an 8B model in bf16 with LoRA needs about 16 GB". That is the frozen base weights alone (8.19B parameters × 2 bytes ≈ 16.4 GB). The full budget has four parts:

1. Base weights: 2 bytes per parameter in bf16, or about 0.5 bytes in 4-bit NF4 (QLoRA).
2. LoRA state: trainable weights, gradients and the two AdamW moments, all in fp32, so 16 bytes per trainable parameter. For 43.6M parameters that is about 0.7 GB.
3. Activations saved for the backward pass. LoRA freezes the base weights, but gradients still have to flow back through every layer to reach the adapters, so the activations are still needed. As a rough estimate for an 8B dense model with FlashAttention, each token stores on the order of 170 KB per layer in bf16, about 6 MB per token across 36 layers, so about 25 GB for a 4,096-token micro-batch. LoRA dropout adds more, because each adapted projection keeps its own dropped-out copy of the input. Gradient checkpointing stores only each layer's input (36 × 4,096 tokens × 4,096 × 2 bytes ≈ 1.2 GB) and recomputes the rest, at the cost of one extra forward pass.
4. Logits and loss: Qwen3's vocabulary has about 152,000 tokens, so 4,096 tokens produce 622M logits. Holding them in bf16, upcasting them to fp32 for the loss and storing the fp32 gradient takes roughly 4-6 GB at peak. A fused or chunked cross-entropy kernel (for example Liger Kernel or Cut Cross-Entropy) brings this under 1 GB.

Putting it together for Qwen3-8B with a 4,096-token micro-batch (rough estimates; measure with `torch.cuda.max_memory_allocated()` on your stack):

| Component | bf16 LoRA, no checkpointing | bf16 LoRA, checkpointing | QLoRA (NF4), checkpointing |
|---|---|---|---|
| Base weights | 16.4 GB | 16.4 GB | about 6 GB |
| LoRA weights, gradients, AdamW | 0.7 GB | 0.7 GB | 0.7 GB |
| Activations | about 25 GB | about 2 GB | about 2 GB |
| Logits and loss | 4-6 GB | 4-6 GB | 4-6 GB |
| **Total** | **about 46-48 GB** | **about 23-25 GB** | **about 13-15 GB** |
| With fused cross-entropy | about 43 GB | about 20 GB | about 9-10 GB |

The QLoRA weight figure is about 6 GB rather than 4 GB because only the transformer's linear layers are quantized: the input embeddings and the output head (622M parameters each in Qwen3-8B) stay in bf16, about 2.5 GB together. For Llama 3.1 8B the equivalent is about 5.7 GB.

What this means for hardware:

- 80 GB (A100 or H100): bf16 LoRA at 4,096 tokens without checkpointing, with headroom.
- 40-48 GB: bf16 LoRA with gradient checkpointing, comfortably.
- 24 GB: bf16 LoRA only with shorter micro-batches (2,048 tokens or less), checkpointing and a fused loss. QLoRA fits comfortably.
- 70B models: QLoRA needs about 40 GB for NF4 weights plus about 3.3 GB of LoRA state at `r=16` (207M trainable parameters), so one 80 GB GPU works, and 48 GB only with short sequences. The QLoRA paper fine-tuned a 65B model on a single 48 GB GPU.

QLoRA is slower than bf16 LoRA, because the 4-bit weights are dequantized on every forward and backward pass, and it costs a small amount of quality. A 10-30% throughput penalty is a reasonable planning figure; measure it on your stack.

## Wall-clock time

At about 300 tokens per example, 4,500 training examples are about 1.35M tokens per epoch. A forward pass costs about `2 × N` FLOPs per token for an `N`-parameter model. LoRA training costs about `4 × N` per token (forward, plus a backward pass through the activations; weight gradients for the frozen base are skipped), or about `6 × N` with gradient checkpointing because of the recomputed forward pass. For Qwen3-8B that is `4 × 8.2B × 1.35M ≈ 4.4 × 10^16` FLOPs per epoch, or about 6.6 × 10^16 with checkpointing. An H100's dense bf16 peak is about 989 TFLOP/s; at a realistic 30-40% utilization that is 2-4 minutes per epoch, so a full 3-epoch run with evaluation takes roughly 10-20 minutes. A 10-20 run sweep fits in an afternoon on one GPU, so run the sweep.

## How to measure it

### Task metric and checkpoint selection

Measure the task metric on held-out data: exact match or F1 for extraction, pass rate for schemas, and a calibrated judge's win rate against the baseline for open-ended generation (see [Making LLM-as-a-Judge Trustworthy](/blog/trustworthy-llm-as-a-judge/)). Pick the checkpoint by the task metric on the validation split, not by the lowest validation loss. Loss keeps falling after quality peaks, especially on generation tasks.

### Baselines and error bars

Compare against three baselines: the prompted base model, the prompted frontier model, and the fine-tuned model. Report the deltas with bootstrap confidence intervals.

On 250 test items those intervals are wide. The 95% half-width of a single accuracy is `1.96 × sqrt(p(1 − p) / 250)`: about ±3.7 points at 90% accuracy, ±5.7 at 70%, and ±6.2 at 50%. When comparing two models on the same 250 items, analyze per-item differences; the interval then depends only on the items where the models disagree and is usually tighter.

### Regression suite

Run a general-capability regression suite: instruction following (an IFEval-style suite), a knowledge benchmark slice (MMLU-Pro or similar; the original MMLU is saturated for current models), safety refusals, and other output formats. LoRA forgets less than full fine-tuning, but it still forgets, especially at high learning rates.

### Overfitting signals

- training loss approaching 0
- outputs copying training phrasing verbatim
- a drop on paraphrased versions of the test prompts

### Ablate before claiming success

Would 2,500 examples give the same result? If so, buy better data rather than more data. Would better prompting close the gap? If so, fine-tuning wasn't needed.

## Shipping it

- Merge the adapter into the base weights for single-tenant, low-latency serving. Keep it separate for multi-tenant multi-LoRA serving, where many adapters share one base.
- Quantize after merging, then re-evaluate. Quantization can cost 0.5-2 points, so the shipped artifact needs its own eval run.
- Version everything together. Training data, config, base model hash and eval results form one release artifact.
- Plan for base model upgrades. Re-run the pipeline when the base model changes; it should be a scheduled job, not a project.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| LoRA (bf16 base) | Cheap, fast, forgets less, swappable adapters | Slightly below full fine-tuning on hard or far-from-pretraining tasks | Default for 1K-50K examples |
| QLoRA (4-bit base) | Fits big models on one GPU | Slower, small quality hit, merge/quantization mismatch | Memory-bound, or you want a 30-70B base |
| Full fine-tuning | Highest capacity, best for large domain shift | Large memory (about 16 bytes/param with mixed-precision AdamW, so about 130 GB for an 8B model before activations), more forgetting, overfits on 5K | Large data, continued pretraining |
| DoRA / rsLoRA variants | Often small gains at the same rank, stable across ranks | Extra complexity, marginal on easy tasks | When sweeping rank or when results plateau |
| SFT then DPO | Learns preferences beyond imitation | Needs pairwise data, more tuning | A clear good-vs-bad output signal exists |

- Spend the next dollar on data quality, not on rank or epochs.
- Choose the base model for its prompted performance on your task.
- Decide early between merged weights and multi-LoRA serving, because it affects quantization and deployment.

## Common mistakes

- **Using the wrong chat template or computing loss on prompt tokens.** Results degrade silently, with no error to point at the cause.
- **Using the full fine-tuning learning rate (around 1e-5) for LoRA.** The adapter barely trains. Going the other way to 1e-3 diverges and causes forgetting.
- **Picking the checkpoint by validation loss, or evaluating on data that leaked from training.** Both overstate quality.
- **Applying LoRA only to the q and v projections (an old default) and concluding that LoRA doesn't work.** Most of the benefit comes from the MLP layers.
- **Skipping the general-capability regression check.** The model ships with broken instruction following or safety behavior.
- **Changing precision without re-evaluating.** Merging into a 4-bit-quantized base, or quantizing after merging, shifts the weights; without a fresh eval you don't know what you shipped.

## Thinking at scale

- Treat the pipeline as the product. One fine-tune is a project; a cheap, automated pipeline (data versioning, sweep, eval, regression gate, packaging) turns every base-model upgrade and data refresh into a scheduled job. That is what keeps a fine-tune from being overtaken by the next frontier model with a prompt.
- Own the data, not just the weights. The 5,000 examples, their provenance and the hand-review notes are the durable asset. Base models and adapters are rebuilt from them.
- Standardize serving early. If several teams will fine-tune, a shared multi-LoRA serving tier with one blessed base model avoids a fleet of merged, separately quantized models that each need their own eval and capacity planning.

## Further reading

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685): the original method.
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314): NF4, double quantization and paged optimizers; also shows that small, high-quality data wins.
- [LoRA Learns Less and Forgets Less (Biderman et al., 2024)](https://arxiv.org/abs/2405.09673): LoRA vs full fine-tuning on capacity and forgetting.
- [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (Kalajdzievski, 2023)](https://arxiv.org/abs/2312.03732): rsLoRA, and why `alpha / sqrt(r)` scaling is better.
- [DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)](https://arxiv.org/abs/2402.09353): a magnitude/direction decomposition variant.
- [LIMA: Less Is More for Alignment (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206): 1,000 curated examples, where quality beats quantity.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index): the reference implementation and LoRA config options.
- [LoRA Without Regret (Schulman and Thinking Machines Lab, 2025)](https://thinkingmachines.ai/blog/lora/): LoRA should target all layers including the MLP, its optimal learning rate is about 10x full fine-tuning, and it is less tolerant of large batches.
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B): architecture and license of the worked-example base model.
