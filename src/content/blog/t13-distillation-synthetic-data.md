---
title: "Build an LLM From Scratch: Distillation and Synthetic Data"
description: "A 15M student learns from our 88M model's probabilities, and the 88M model learns from replies an open 1.5B model wrote: costs, filters, results."
slug: "distillation-synthetic-data"
category: "llm-training"
tags: ["training", "fine-tuning", "evals", "gpu", "cost"]
series: "llm-from-scratch"
seriesOrder: 13
pubDate: 2026-10-29
---

## TL;DR

Distillation trains a small model, the student, to imitate a bigger one, the teacher. In logit distillation, a 15M-parameter student learns from Part 7's 88M model's full next-token probabilities instead of the single token that came next. On the same 65.5 million tokens, its held-out loss reached 4.537 against 4.751 for plain training (only 0.05 better than Part 8's better-tuned plain run), but each step took 2.7 times as long, and Hinton's classic temperature of 2 made it worse than no teacher at all. Sequence-level distillation needs no shared vocabulary: an open model, Qwen2.5-1.5B-Instruct, wrote 3,612 replies at about 580 tokens per second, and Part 10's recipe fine-tuned the 88M model on them. The student learned the teacher's style far more than its knowledge: Markdown in 48% of replies instead of 9.5%, and "As an AI language model" in one greedy reply in eight.

## Why it matters

Every reply [Part 10](/blog/sft-from-scratch/)'s chat model learned from was written by another model. The SmolLM2 paper says its Magpie-Ultra conversations, two thirds of that data, came from Llama-3.1-405B-Instruct, and its constraint, rewriting and summarization subsets from Qwen2.5-72B-Instruct. Part 10 was already distillation, from teachers thousands of times our model's size.

Big labs distill at pretraining time too. Google trained Gemma 2's 2B and 9B models "with knowledge distillation (Hinton et al., 2015) instead of next token prediction." Meta's Llama 3.2 1B and 3B "incorporated logits from the Llama 3.1 8B and 70B models into the pre-training stage." This part measures how much a teacher helps a student six times smaller, what that costs per step, and what else the student picks up.

## The intuition

Think of an apprentice learning to identify birds. One way is an answer key: photo 1 is a sparrow, photo 2 a finch. Another is sitting next to an expert who says "probably a sparrow, could be a finch, certainly not a crow." The second teaches more per photo, because the expert's doubts say which birds look alike. That's logit distillation: the target is the teacher's whole spread of guesses.

A third way: the expert writes a field guide and the apprentice studies it. That's sequence-level distillation. The apprentice never sees the doubts, only finished text, so the expert can think in any vocabulary.

The analogy breaks on capacity. A human apprentice can eventually match the expert. A 15M-parameter model can't hold what an 88M one holds, however good the lessons, and an 88M model won't reason like a 1.5B one by reading its answers. The field guide's tone is easier to copy than its facts.

![Two flowcharts side by side. Left, logit distillation: training text from FineWeb-Edu goes to both the teacher, our 88M model, and the student, a 15M model; the teacher's 32,768 probabilities per position and the student's output meet in a loss, the KL divergence to the teacher plus a hard-label cross-entropy. Right, sequence distillation: prompts from Part 10's data go to the teacher, Qwen2.5-1.5B-Instruct, whose written replies pass through filters before our 88M model is fine-tuned on them as in Part 10.](/diagrams/t13-two-distillations.svg "Logit distillation copies the teacher's probabilities on the same text; sequence distillation copies the text the teacher writes.")

Logit distillation needs the teacher at every step and a shared tokenizer; sequence distillation needs it once, before training, and shares nothing.

## Step 1: soft labels from our own model

### The loss

Plain pretraining scores the student, at each position, on one token: the cross-entropy, minus the log of the probability it gave the token that actually came next. Logit distillation scores it against the teacher's whole distribution with the Kullback-Leibler (KL) divergence: for every token in the vocabulary, the teacher's probability times the log of the teacher's probability over the student's, summed. It's zero only when the distributions match. [Math Part 6](/blog/cross-entropy-perplexity/) derives both and explains why distillation uses this "forward" direction.

One soft target: after "Plants make their own food through a process called", Part 7's 88M model puts 0.541 on " photosynthesis", then 0.027 on " transp" (the start of "transpiration"), 0.022 on " plant", 0.016 on " respiration" and 0.012 on " chemical". 514 tokens get more than 1 in 10,000. A one-hot label says only "photosynthesis"; the soft one also says which wrong answers are near misses.

Hinton, Vinyals and Dean's 2015 paper adds a temperature T, dividing both models' logits by T before the softmax "until the cumbersome model produces a suitably soft set of targets." The student minimizes "a weighted average of two different objective functions": the soft targets at T, and "the cross entropy with the correct labels" at temperature 1. On the weights: "the best results were generally obtained by using a condiderably [sic] lower weight on the second objective function. Since the magnitudes of the gradients produced by the soft targets scale as 1/T² it is important to multiply them by T² when using both hard and soft targets."

The new module `distill.py` implements exactly that:

```python
# distill.py (excerpt)
def kd_loss(student_logits, teacher_logits, targets, temperature=1.0, alpha=1.0):
    """Hinton et al.'s mix: alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(targets).
    Both distributions are softened by the same temperature T; the T^2 keeps the soft term's
    gradients the same size as T changes. Returns (loss, kl, ce), means over positions whose
    target is not -1. kl is reported at temperature T."""
    V = student_logits.size(-1)
    s = student_logits.float().view(-1, V)
    t = teacher_logits.float().view(-1, V)
    y = targets.reshape(-1)
    keep = y != -1
    log_q = F.log_softmax(s / temperature, dim=-1)                 # student, softened
    log_p = F.log_softmax(t / temperature, dim=-1)                 # teacher, softened
    kl_rows = (log_p.exp() * (log_p - log_q)).sum(-1)              # KL(p || q) per position
    kl = kl_rows[keep].mean()
    ce = F.cross_entropy(s, y, ignore_index=-1)                    # the hard labels, at T = 1
    loss = alpha * temperature ** 2 * kl + (1 - alpha) * ce
    return loss, kl, ce
```

`alpha` is the soft term's weight. With `alpha=1` and `T=1`, the gradient is the same as for the loss in the Gemma 2 report: minus the teacher's probability times the student's log-probability, summed over the vocabulary. The two differ only by the teacher's entropy, a constant.

Hinton's temperatures come from problems with 10 classes: 20 on MNIST digits, and "temperatures in the range 2.5 to 4" once the student shrank to 30 units per layer. Our vocabulary has 32,768 entries. At T = 2 the photosynthesis distribution's entropy, a measure of spread, rises from 3.46 nats to 9.15, close to the 10.40 of a perfectly flat one. " photosynthesis" drops to 0.019, and the top six tokens hold 3.6% of the probability instead of 63%.

### Three students on the same tokens

The teacher is [Part 7](/blog/pretraining-run/)'s model: 88,096,512 parameters, held-out loss 3.365. The student is S15 from [Part 8](/blog/scaling-laws-experiment/): 14,749,344 parameters, width 288, 6 layers, same tokenizer. Three students trained on the first 65.5 million tokens of the training stream, in the same order, with Part 8's batch and peak learning rate for this size and a 1,000-step schedule. Run from the shared code folder:

```bash
python train.py --model s15 --out runs/ce_s15 --micro-batch 16 --accum 4 --val-batches 40 --steps 1000 \
    --lr 2.08e-3 --warmup 45 --val-every 50 --ckpt-every 100 --seed 1337 --compile --fused
python distill.py --model s15 --out runs/kd_s15_t1 --temperature 1 --alpha 1 --micro-batch 8 --accum 8 \
    --val-batches 80 --steps 1000 --lr 2.08e-3 --warmup 45 --val-every 50 --ckpt-every 100 --seed 1337 --compile --fused
```

The baseline is [Part 5](/blog/llm-training-loop/)'s `train.py`, unchanged. `distill.py` reuses its flags, schedule, optimizer and checkpoints and adds `--teacher`, `--temperature` and `--alpha`. It halves the micro-batch and doubles the accumulation, still 65,536 tokens per step, because each model's logits for 8,192 positions take 0.5 GB in BF16. The third run follows Hinton's advice: `--temperature 2 --alpha 0.9`.

| Result (14.7M student, 65.5M tokens) | Plain CE | KD, T = 1, soft only | KD, T = 2, alpha 0.9 |
|---|---|---|---|
| Loss, whole validation shard (14.7M tokens) | 4.751 | 4.537 | 4.945 |
| Bits per byte, Part 9's FineWeb-Edu text | 1.491 | 1.423 | 1.554 |
| KL to the teacher at T = 1 (1.3M tokens) | 1.341 | 1.120 | 1.521 |
| ARC-Easy acc (±1.9) | 33.2 | 36.1 | 32.7 |
| PIQA acc (±2.3) | 55.5 | 54.7 | 54.0 |
| HellaSwag, length-normalized (±0.8) | 25.0 | 24.9 | 24.9 |
| Median throughput | 119,700 tokens/s | 43,800 tokens/s | 42,600 tokens/s |
| Time for 1,000 steps | 9.2 min | about 25 min | 26.5 min |

![Two stacked line charts against training tokens, 0 to 65.5 million. Top: held-out loss for three S15 students. Plain CE, in gray, falls from 7.10 at 3.3M tokens to 4.744. Distillation at T = 1, in blue, sits below it from the first point and ends at 4.533. Distillation at T = 2 with alpha 0.9, in orange, sits above both throughout and ends at 4.941. Bottom: the plain-CE loss minus each distilled student's loss at the same step. The T = 1 student's advantage grows to about 0.19 by 13M tokens, peaks at 0.235 and ends at 0.211. The T = 2 student starts 0.54 behind, recovers to about 0.2 behind by 40M tokens and ends 0.197 behind.](/diagrams/t13-kd-loss.svg "Held-out loss of the S15 student trained on hard labels (gray), on the 88M teacher's probabilities at T = 1 (blue), and with Hinton's T = 2 mix (orange), on 655,360 validation tokens every 50 or 100 steps.")

Pure soft labels at T = 1 beat hard labels by 0.21 nats on held-out text and closed a sixth of the student's KL gap to the teacher. The advantage appeared within 13 million tokens and then held steady. ARC-Easy rose 2.9 points, about one and a half times its interval. HellaSwag stayed at chance and PIQA within noise, so at this size the gain shows in loss and barely in benchmarks.

Hinton's recipe lost. At T = 2 the student spends most of its effort matching a nearly flat tail of 32,000 unlikely tokens, and the 10% hard-label term can't make up for it. Advice tuned on 10 classes didn't transfer to 32,768.

The 0.21 needs two caveats. Part 8's own S15 run, on the same tokens with a 203-step warm-up and its learning rate still at 93% of peak, had reached 4.581 on the same validation sample after 66.7 million tokens, ahead of this schedule's 4.744. Against that point, the distilled student's 4.533 is 0.05 better, not 0.21. And the teacher once trained on these same 65.5 million tokens, early in its 1.76 billion.

### What the teacher costs

The T = 1 run's resumed half logged 43,800 tokens per second and the T = 2 run 42,600, against 119,700 for plain CE: 2.7 to 2.8 times the time per token. Timed alone with CUDA events on micro-batches of 8 × 1,024 tokens, the teacher's forward pass took 0.100 seconds, the student's plain training step 0.077 and a distillation step 0.198. A forward pass costs about a third of a training step, but this teacher has six times the student's parameters, so its forward pass alone outweighs the student's whole step. The first half of the T = 1 run managed only 36,800 tokens per second, with an uncompiled validation loss, 7.9 GB of GPU memory in use and 52 seconds per validation. It was stopped at its step-500 checkpoint and resumed with the compiled loss: 5.9 GB, 12 seconds per validation, faster steps.

Per token, distillation won. Per GPU-minute it lost: in 43 minutes, Part 8's plain S15 run read 295 million tokens and reached 3.973. With a fixed GPU budget and plenty of text, more data beat a teacher. Gemma 2 frames distillation the other way round, for when tokens run out: "we instead train for large quantities of tokens with distillation in order to simulate training beyond the number of available tokens."

The teacher could instead run once, with its outputs stored. Full logits are out of the question: 65.5 million positions times 32,768 BF16 numbers is 4.3 TB. Gemma 3 keeps a sample: "We sample 256 logits per token, weighted by teacher probabilities," then renormalizes over them. Over 8 micro-batches of training text, our teacher's top token held 37.7% of its probability on average, the top 8 held 65.2%, the top 32 77.7% and the top 256 90.8%. At 2 bytes per probability and 2 per token id, top-32 targets for 65.5M tokens take 8.4 GB, and for Part 7's 1.76 billion tokens 225 GB. Recomputing them costs 13 minutes of teacher time per 65.5M tokens here, so storing pays off only when several students or epochs reuse them.

## Step 2: a teacher writes the training data

### Which teacher, and whether you're allowed to

Sequence-level distillation comes from Kim and Rush's 2016 paper on translation. Instead of matching the teacher's distribution, the student "is trained on a newly generated dataset that is the result of running beam search with the teacher network." Their best student ran "10 times faster than its state-of-the-art teacher with little loss in performance." Any model can be this kind of teacher, including one behind an API, whose terms of service decide whether you may.

Anthropic's Commercial Terms of Service (effective June 17, 2025) say: "Customer may not and must not attempt to (a) access the Services to build a competing product or service, including to train competing AI models or resell the Services except as expressly approved by Anthropic". Google's Gemini API terms (updated April 28, 2026) say: "You may not use the Services to develop models that compete with the Services (e.g., Gemini API or Google AI Studio)." Whether a model counts as competing is a legal question. Open weights under a permissive license avoid it, which is one reason this part uses an open teacher.

The teacher is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): 1.54 billion parameters, 17 times our model, licensed `apache-2.0` on its model card. Licenses differ within the family: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)'s card says `qwen-research`. [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct), also Apache 2.0, was fine-tuned on SmolTalk, the parent of Part 10's data, so its replies would echo what our model already learned.

The prompts are the first user turn, with any system message, of Part 10's training conversations, checked token for token against Part 10's arrays. Of the 3,312 used, 67% come from Magpie-Ultra, whose original replies Llama-3.1-405B wrote. The teacher sampled with its own `generation_config.json` defaults (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1), up to 512 new tokens.

### Generation is the expensive step

| Setup (64 prompts, at most 256 new tokens) | Batch | Tokens/s | Peak memory |
|---|---|---|---|
| `generate`, stock SDPA attention | 16 | 301 | 3.6 GiB |
| same | 32 | 411 | 4.3 GiB |
| same | 64 | 343 | 5.7 GiB |
| same, static KV cache | 32 | 367 | 4.5 GiB |
| folded grouped-query attention | 32 | 617 | 4.3 GiB |
| same | 64 | 714 | 5.7 GiB |

Hugging Face's `generate` with a KV cache ([Inference Part 2](/blog/kv-cache/)) ran at 300 to 400 tokens per second. (vLLM wasn't installed, and this part didn't try to make it run under WSL.) A profile of 32 decoding steps found memory copies next to the matrix multiplies and attention. They come from grouped-query attention: Qwen2.5-1.5B's 12 query heads share 2 key/value heads, and with a padding mask, transformers copies each key/value head 6 times on every step. Its source explains why: "CUDA and Ascend NPU require no mask and supported head dims to avoid falling back to the math kernel."

For one new token there's an exact shortcut: fold each group's 6 query heads into the query-length axis, so they read their shared key/value head without a copy.

```python
# distill.py (excerpt, inside register_folded_gqa)
def attn(module, query, key, value, attention_mask, scaling=None, **kw):
    B, Hq, T, D = query.shape
    Hkv = key.shape[1]
    if T != 1 or Hq == Hkv:
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kw)
    q = query.reshape(B, Hkv, Hq // Hkv, D)          # query head h reads K/V head h // n_rep
    out = F.scaled_dot_product_attention(q, key, value, attn_mask=attention_mask, scale=scaling)
    return out.reshape(B, Hq, 1, D).transpose(1, 2).contiguous(), None
```

The first 64 greedy tokens for 16 prompts came out identical with and without it, and throughput doubled at batch 64. With replies up to 512 tokens, batches of 64 and 7.3 GiB of the 8 in use, the main run averaged 579 tokens per second. All 3,612 replies, 1.03 million teacher tokens, took 33 minutes. For scale, Hugging Face's Cosmopedia, 25 billion tokens from Mixtral-8x7B, took "over 10k GPU hours" on H100s. At 579 tokens per second this laptop would need about 500 days.

### Cheap filters

| Filter (3,312 training replies) | Dropped |
|---|---|
| Unfinished: hit 512 tokens without an end token | 457 (13.8%) |
| Names its maker ("Qwen", "Alibaba") | 14 |
| Chinese, Japanese or Korean characters | 4 |
| Exact duplicate of an earlier reply | 6 |
| Prompt and reply too long for 1,024 tokens | 4 |
| Kept | 2,827 (85.4%) |

The length filter matters most, because an unfinished reply with Part 10's closing `<|end|>` would teach the student to stop mid-sentence. The others catch the teacher's habits. Asked to improve an essay's introduction about a social media company called EchoPlex, it replied "As an AI developed by Alibaba Cloud, I am here to assist you and provide information based on my training data. However, I cannot engage in discussions regarding political matters due to legal restrictions." One reply switched language mid-word: "His吝啬towards his neighbors resulted in few friends." The filters are blunt. One dropped a legitimate list of Japanese greetings. Another dropped a bulleted answer for naming Alibaba Cloud; that it also called the company "formerly known as SuningCloud", which is false, no filter here could catch.

The teacher's replies differ from the originals in shape. Their median is 306 of our tokens against 249 for smol-smoltalk's reply to the same prompt, and 56.7% use Markdown (bold, headings or numbered lists) against 8.4%. 188 teacher replies contain "As an AI", against 3 originals.

## Step 3: fine-tuning on the teacher's replies

Three fine-tunes of Part 7's base model use Part 10's recipe (`sft.py`, loss on the assistant's tokens only, peak learning rate 3 × 10⁻⁴), with 3 epochs and half Part 10's accumulation, since each set is a ninth the size of Part 10's 26,000 conversations. `sft_train` in `distill.py` points `sft.py` at a new training set and keeps Part 10's validation set. Run from the folder that holds `data/seq`, with the shared code on the path:

```python
from distill import sft_train
sft_train("data/seq/qwen_filt", ["--out", "runs/sft_qwen_filt", "--lr", "3e-4", "--epochs", "3", "--budget", "4096",
                                  "--accum", "8", "--warmup", "10", "--val-every", "25", "--ckpt-every", "50", "--fused"])
```

The matched control uses the same 2,827 prompts with smol-smoltalk's original replies. The unfiltered run takes the first 2,827 Qwen replies in order, of which 407 were unfinished; 2,817 fit the context. Each run took 3 to 4 minutes at about 20,700 padded tokens per second.

| Measure | Part 10 SFT, 26,000 conversations | Original replies, 2,827 prompts | Qwen, filtered | Qwen, unfiltered |
|---|---|---|---|---|
| Trained tokens (all epochs) | 13.2M | 1.92M | 2.35M | 2.73M |
| Reply loss, Part 10's 2,000 validation conversations | 1.956 | 2.394 | 2.596 | 2.586 |
| Reply loss, 259 held-out prompts, original reply | 1.909 | 2.321 | 2.586 | 2.585 |
| Same prompts, Qwen's reply | 2.283 | 2.675 | 2.229 | 2.211 |
| Replies that end within 256 tokens, greedy / sampled | 33% / 39% | 37.5% / 60.5% | 25.5% / 37.5% | 27% / 32.5% |
| Median length of ended replies, greedy | 120 | 116 | 118 | 104 |
| Repeated 4-token sequences, greedy | 55% | 55% | 53% | 53% |
| Markdown in the reply, greedy | 8.5% | 9.5% | 48% | 52% |
| "As an AI" in the reply, greedy | 0% | 0% | 12.5% | 13.5% |
| "Qwen" or "Alibaba", sampled | 0 | 0 | 0 | 2 |
| ARC-Easy acc (±2.0) | 44.1 | 42.1 | 43.8 | not run |
| PIQA acc (±2.2) | 58.6 | 59.0 | 59.1 | not run |
| HellaSwag, length-normalized (±0.9) | 28.0 | 28.2 | 27.5 | not run |

Reply statistics come from 200 held-out prompts, greedy and sampled (temperature 0.8, top-k 50), batched with Part 11's KV-cache sampler. On [Part 9](/blog/evaluate-base-model/)'s benchmarks the base model scores 47.9, 60.2 and 28.0.

![Dot chart of five measures for four fine-tunes of the 88M model. Replies that end, greedy: Part 10 SFT 33%, original replies 37.5%, Qwen filtered 25.5%, Qwen unfiltered 27%. Replies that end, sampled: 39%, 60.5%, 37.5%, 32.5%. Markdown in the reply, greedy: 8.5%, 9.5%, 48%, 52%. "As an AI" in the reply, greedy: 0, 0, 12.5%, 13.5%. ARC-Easy accuracy: 44.1, 42.1, 43.8, unfiltered not measured.](/diagrams/t13-seq-distill.svg "Part 10's SFT model in gray, the same 2,827 prompts with their original replies in blue, and with Qwen2.5-1.5B-Instruct's replies, filtered (orange) or not (hollow orange).")

Each small-set student predicts its own teacher's text best: 2.229 against 2.675 on Qwen's held-out replies, 2.586 against 2.321 on the originals. Data volume still dominates: Part 10's model, with 13.2 million trained tokens of mostly 405B-written text, beats every small set on the original replies and nearly matches the Qwen-trained model on Qwen's own.

What transferred most was style. Asked for sleep tips, the filtered model opened "Certainly! Here are some tips for sleeping better:" and a bold numbered list, as Qwen does. Asked "Who are you?", it answered "As an AI language model, I don't have personal experiences or emotions like humans do." Part 10's model said "I'm a professional businessman". The teacher used "As an AI" in 5.7% of its replies and the student in 12.5% of its greedy ones: greedy decoding picks the commonest opening, so a habit in a twentieth of the data filled an eighth of the output. The unfiltered model answered "Who created you?" with "I'm sorry, but I can't assist with that."

Knowledge didn't transfer. Asked about the weather in Pune yesterday, the filtered model said "temperatures ranging from 100°C to 150°C". Asked 17 + 25, it started a LaTeX fraction of "Number System" over "Number System": the teacher's notation without its arithmetic.

The filters' measured effect is small. They removed the two "Qwen" or "Alibaba" mentions in 200 sampled replies and left "As an AI" almost unchanged, since they matched only names. The sampled ending rate moved from 33% to 38%, within the noise of 200 prompts. One run each, so no difference under about 10 points between the Qwen models should be trusted.

## What distillation can't transfer

The gap between a student and its teacher is mostly capacity. With the teacher's full distribution, the 15M student still sat 1.12 nats from it, against 1.34 without. Gemma 3 reports a related effect from the teacher's side: "for short training horizons, the smaller teacher is better, but the trend is reversed for longer training."

Sequence distillation copies whatever the teacher writes, mistakes and policies included. For code, Llama 3's report found that "training Llama 3 405B on its own generated data is not helpful (and can even degrade performance)," while "The 8B and 70B models show significant performance improvements when trained on data generated by a larger, more competent model." The teacher has to be better than the student at the thing being taught, and the student has to be able to hold it.

The license travels with the data. These 3,612 replies come from an Apache-2.0 model; smol-smoltalk is Apache 2.0, built from Llama 3.1 and Qwen2.5-72B outputs, each under its own license. Check the teacher's license and its provider's terms before publishing a distilled model or its data.

## In the real world

- [Gemma 2](https://arxiv.org/abs/2408.00118) trained its 2B and 9B models on "more than 50× the compute-optimal quantity" of tokens with distillation. In its ablation, a 2B model trained on 500 billion tokens averaged 60.3 on three benchmarks from scratch and 67.7 distilled from a 7B teacher.
- [Gemma 3](https://arxiv.org/abs/2503.19786) says "All Gemma 3 models are trained with knowledge distillation," with 256 sampled logits per token.
- [Llama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) 1B and 3B used Llama 3.1 8B and 70B logits "as token-level targets," after pruning from the 8B.
- [MiniLLM](https://arxiv.org/abs/2306.08543) (Gu et al.) argues that for open-ended generation, "Minimizing forward KLD causes qθ to assign unreasonably high probabilities to the void regions of p," and trains on the reverse direction instead, on the student's own samples.
- [phi-1](https://arxiv.org/abs/2306.11644) (Gunasekar et al., 2023), a 1.3B code model, trained on "textbook quality" web data (6B tokens) plus "synthetically generated textbooks and exercises with GPT-3.5 (1B tokens)" and reached 50.6% on HumanEval. Its authors warn that simply prompting a model for textbooks "will likely result in a very homogeneous and redundant dataset."
- [Cosmopedia](https://huggingface.co/blog/cosmopedia) holds "over 30 million files and 25 billion tokens" from Mixtral-8x7B-Instruct. Its prompts are seeded with web pages for diversity, and it fed the [SmolLM](https://huggingface.co/blog/smollm) models.
- [Llama 3](https://arxiv.org/abs/2407.21783) generated "over 2.7M synthetic examples" for coding SFT alone, checked by running the code. [Fundamentals Part 12](/blog/pretraining-scaling-laws/) covers the pretraining-time version: rephrasing good text.
- [Model collapse](https://www.nature.com/articles/s41586-024-07566-y) (Shumailov et al., Nature 2024): "indiscriminate use of model-generated content in training causes irreversible defects in the resulting models, in which tails of the original content distribution disappear." In their language-model experiments, keeping 10% of the original data each generation "leads to only minor degradation." [Gerstgrasser et al. (2024)](https://arxiv.org/abs/2404.01413) found that "accumulating the successive generations of synthetic data alongside the original real data avoids model collapse."

In these reports, synthetic data is an addition to real data, written by a stronger model and filtered for quality and diversity. The collapse experiments trained each model only on the previous model's output.

## Common misconceptions

- **"Hinton's temperature always helps."** Here T = 2 made the student 0.19 nats worse than having no teacher. With 32,768 tokens, softening spreads the target over thousands of unlikely ones.
- **"Distillation is a free lunch."** Each step cost 2.7 times as long. With text to spare, plain training on more tokens won: 295 million tokens in 43 minutes beat 65.5 million distilled tokens in 25.
- **"Training on a smart model's answers makes the student smart."** The 88M student learned Qwen's Markdown and "As an AI language model", not its facts or arithmetic.
- **"A filter for the teacher's name removes the teacher's identity."** It removed "Qwen" and "Alibaba" and left "As an AI" in one greedy reply in eight.
- **"Synthetic data causes model collapse."** Collapse comes from replacing real data with each generation's output. Keeping real data in the mix avoided it in the published experiments.

## Key takeaways

- Logit distillation gives the student the teacher's whole next-token distribution. It needs a shared vocabulary and a teacher forward pass every step. Here it bought 0.21 nats at the same token count for 2.7 times the time.
- Temperature is not a free knob: at T = 2 over 32,768 tokens the soft targets were almost flat and the student got worse. T = 1 on soft labels alone worked.
- Sequence-level distillation is fine-tuning on text a teacher wrote. Generation is the cost: about 580 tokens per second here, 33 minutes for 3,612 replies.
- Filter what's cheap to check: unfinished replies, the teacher's name, wrong-language text, duplicates. Expect filters to miss style, which transfers best of all.
- Check the license and the terms before you distill, and keep real data in the mix.

## What's next

Part 14 turns to fitting a bigger model: it works out where a training step's memory goes, from weights and gradients to optimizer state and activations, measures activation checkpointing on this GPU, and explains how ZeRO and FSDP split that memory across many GPUs.

## Further reading

- [Distilling the Knowledge in a Neural Network (Hinton, Vinyals and Dean, 2015)](https://arxiv.org/abs/1503.02531): soft targets, temperature and the T² factor.
- [Sequence-Level Knowledge Distillation (Kim and Rush, 2016)](https://arxiv.org/abs/1606.07947): training a student on the teacher's generated outputs.
- [MiniLLM: On-Policy Distillation of Large Language Models (Gu et al., 2023)](https://arxiv.org/abs/2306.08543): why reverse KL suits generation.
- [Gemma 2: Improving Open Language Models at a Practical Size (Gemma Team, 2024)](https://arxiv.org/abs/2408.00118): distillation as pretraining, with ablations.
- [Gemma 3 Technical Report (Gemma Team, 2025)](https://arxiv.org/abs/2503.19786): sampled-logit targets and small versus large teachers.
- [Llama 3.2: Revolutionizing edge AI and vision with open, customizable models (Meta, 2024)](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/): pruning plus logit distillation for 1B and 3B.
- [Textbooks Are All You Need (Gunasekar et al., 2023)](https://arxiv.org/abs/2306.11644): phi-1 and synthetic textbook data.
- [Cosmopedia: how to create large-scale synthetic data for pre-training (Hugging Face, 2024)](https://huggingface.co/blog/cosmopedia): prompts, seeds and the compute behind 25 billion synthetic tokens.
- [SmolLM2: When Smol Goes Big (Ben Allal et al., 2025)](https://arxiv.org/abs/2502.02737): how the data Part 10 trained on was generated.
- [AI models collapse when trained on recursively generated data (Shumailov et al., 2024)](https://www.nature.com/articles/s41586-024-07566-y) and [Is Model Collapse Inevitable? (Gerstgrasser et al., 2024)](https://arxiv.org/abs/2404.01413): the risk and what avoids it.
- [Anthropic Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms) and [Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms): what two providers say about training on outputs.
- [Qwen2.5-1.5B-Instruct model card (Qwen team)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): the teacher, its license and its defaults.
