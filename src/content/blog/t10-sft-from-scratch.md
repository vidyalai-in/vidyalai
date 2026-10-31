---
title: "Build an LLM From Scratch: Supervised Fine-Tuning"
description: "Turning the 88M base model into a chat model in 15 minutes: a chat template on reserved tokens, masked loss, 26,000 conversations, and what it cost."
slug: "sft-from-scratch"
category: "llm-training"
tags: ["fine-tuning", "training", "evals", "gpu"]
series: "llm-from-scratch"
seriesOrder: 10
pubDate: 2026-07-29
---

## TL;DR

This part fine-tunes Part 7's 88M base model into a chat model on the laptop GPU. A chat template built from the special tokens Part 2 reserved turns each conversation into token ids. The loss is masked to the assistant's replies and divided by each step's real count of trained tokens, not averaged per micro-batch. The training data is 26,000 conversations from smol-smoltalk, an openly licensed set built for small models. In 15 minutes the held-out loss on assistant tokens fell from 2.82 to 1.96, and the loss on the end-of-turn token fell from 19.0 to 0.73. The model now answers in the assistant's voice and sometimes stops on its own, where the base model never did. It also invents facts, loops under greedy decoding and lost 3.8 points on ARC-Easy. Training on every token instead of only the replies gave the same reply loss, lost 8.6 points on ARC-Easy, and ended more of its replies.

## Why it matters

Give the base model a chat-formatted question, "What is the capital of France?", and it writes "The capital of France is the capital of France. Its capital is the capital of France." until the 200-token limit cuts it off. It can't stop. The base model gives the end-of-turn token an average probability of about 1 in 180 million.

After 291 optimizer steps of supervised fine-tuning, the same weights answer "The capital of France is Paris." The answer is right, but by step 291 the model has learned only part of the job. The greedy reply runs on about "rich history, culture, and culture" for another 190 tokens.

[Fundamentals Part 13](/blog/base-model-to-assistant/) explained supervised fine-tuning (SFT): next-token training on example conversations, with the loss on the assistant's side only. This part runs it on a model small enough to fine-tune in a quarter of an hour and measures the details: the bytes of the template, which tokens count, how a batch of uneven conversations is averaged, and what fine-tuning costs in knowledge the model already had.

## The intuition

Think of a widely read person starting work at a help desk. What they have to learn is the job's format: tickets arrive with a header, the reply goes in a box, and you click "send" when the answer is done. Training is a stack of past tickets with the replies filled in, and only the replies are graded.

The analogy breaks in two places. A new employee learns the format in an afternoon and keeps everything they knew; a model has one set of weights, so learning the format overwrites a little of everything else. And our employee has read far less than a real assistant has: 1.76 billion tokens of educational web pages. The format can't supply knowledge the base model never learned.

## Step 1: the chat template

### Five reserved tokens

[Part 2](/blog/train-bpe-tokenizer/) set aside ids for chat before the vocabulary was trained: `<|endoftext|>` (0), `<|system|>` (1), `<|user|>` (2), `<|assistant|>` (3), `<|end|>` (4), and three spares. A chat template is the rule that lays a conversation out in those tokens. The series uses the simplest one that works:

```text
<|endoftext|><|system|>You are a helpful assistant.<|end|><|user|>What is the capital of France?<|end|><|assistant|>The capital of France is Paris.<|end|>
```

Each message is its role token, its text, and `<|end|>`. Every conversation starts with `<|endoftext|>`, because every pretraining document did (Part 3 put token 0 before each one), so the model begins a chat in the state it learned for "a new document starts here". No newline follows a role token: the special tokens already separate the fields, and content starting right after one is tokenized like the start of a document. At inference time, the prompt ends with `<|assistant|>` and generation stops when the model emits `<|end|>`.

`render` in the new `sft.py` builds the ids and the training targets together:

```python
# sft.py (excerpt)
EOT, SYSTEM, USER, ASSISTANT, END = 0, 1, 2, 3, 4          # Part 2's reserved ids
ROLE = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}

def render(messages, tok, train_on="assistant", bodies=None):
    """One conversation -> (ids, targets). ids start with <|endoftext|>, like every pretraining
    document; each message is <|role|> content <|end|>. targets[i] is the token after ids[i] if
    that token is trained, else -1: the assistant's content and its <|end|>, or everything."""
    ids, learn = [EOT], [False]
    for k, m in enumerate(messages):
        body = bodies[k] if bodies else tok.encode(m["content"].strip()).ids
        ids += [ROLE[m["role"]]] + body + [END]
        trained = train_on == "all" or m["role"] == "assistant"
        learn += [train_on == "all"] + [trained] * (len(body) + 1)
    targets = [t if ok else -1 for t, ok in zip(ids[1:], learn[1:])] + [-1]
    return ids, targets
```

Run from the shared code folder:

```python
from sft import chat_tokenizer, render
tok = chat_tokenizer()
ids, targets = render([{"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": "What is the capital of France?"},
                       {"role": "assistant", "content": "The capital of France is Paris."}], tok)
print(ids)
print(targets)
```

```text
[0, 1, 2370, 357, 265, 5111, 10324, 21, 4, 2, 1553, 319, 270, 3274, 289, 4185, 38, 4, 3, 469, 3274, 289, 4185, 319, 5690, 21, 4]
[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 469, 3274, 289, 4185, 319, 5690, 21, 4, -1]
```

Of 27 positions, 8 are trained: the seven tokens of the reply and its closing `<|end|>` (id 4). The target at position 18, where the input is `<|assistant|>`, is the reply's first word, `The` (469). The role token itself is never a target, since the template always supplies it. The `-1` is the value [Part 4](/blog/write-gpt-model/) set as the loss's `ignore_index` for exactly this.

`chat_tokenizer()` sets `encode_special_tokens = True`, so a user who types the literal text `<|end|>` gets six ordinary tokens (`[35, 99, 508, 99, 37, 17699]`), not id 4. OpenAI's original ChatML document warned that a format like this "inherently allows injections from user input containing special-token syntax, similar to SQL injections."

### Next to published templates

| Part of the format | This series | SmolLM2-Instruct (ChatML) | Llama 3 |
|---|---|---|---|
| Start of the text | `<\|endoftext\|>` | nothing added | `<\|begin_of_text\|>` |
| A turn begins | `<\|user\|>` (one token) | `<\|im_start\|>user` and a newline | `<\|start_header_id\|>user<\|end_header_id\|>` and a blank line |
| A turn ends | `<\|end\|>` | `<\|im_end\|>` and a newline | `<\|eot_id\|>` |
| Default system message | none | "You are a helpful AI assistant named SmolLM, trained by Hugging Face" | none |

SmolLM2-135M-Instruct's [`tokenizer_config.json`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/main/tokenizer_config.json) writes each message as `'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'`, so the role is ordinary text between two special tokens. Meta's [Llama 3 prompt format](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/) does the same with header tokens, and the model "signals the end of the {{assistant_message}} by generating the <|eot_id|>." Spelling roles as text lets a template add new roles without new tokens. Our vocabulary was fixed in Part 2 with a token per role, so that's what the template uses.

### The rows those tokens start from

[Part 7](/blog/pretraining-run/) found that none of these ids occurred in 1.76 billion pretraining tokens. Their embedding rows grew to a length of about 1.30, against 1.00 for a typical row, and the four chat tokens point almost the same way: pairwise cosine similarity 0.89 to 0.90. Their nearest neighbors in the table, measured by cosine similarity (how closely two vectors point the same way), are control bytes and invalid UTF-8 bytes, most of which never occurred in pretraining either. As inputs they're barely distinguishable, and as outputs they're pushed far down, which is why `<|end|>` costs the base model 19.0 nats, a probability of about 1 in 180 million. SFT has to fix both sides.

## Step 2: the instruction data

### Choosing a dataset

The data must be openly licensed, conversational, short enough for a 1,024-token context, and pitched at a model that has read only educational web pages. [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) is all four. Its card describes it as "a subset of SmolTalk dataset adapted for smol models with less than 1B parameters", used to build SmolLM2-135M-Instruct and SmolLM2-360M-Instruct, with "no function calling and less rewriting and summarization examples" and no "advanced math datasets". It's released under Apache 2.0, with 460,341 training and 24,229 test conversations.

It's synthetic: model-written conversations. Two thirds of what this part keeps comes from its Smol-Magpie-Ultra subset, and the [SmolLM2 paper](https://arxiv.org/abs/2502.02737) says Magpie-Ultra was generated with Llama-3.1-405B-Instruct using the two-step method of [Magpie](https://arxiv.org/abs/2406.08464). Human-written alternatives such as databricks-dolly-15k are far smaller, and none was filtered for small models the way smol-smoltalk was, for models of 135M and 360M parameters.

### Filtering

Two of its sources don't fit this model. `self-oss-instruct` is Python code, which the tokenizer handles badly and the base model barely saw (FineWeb's filters drop every page containing `{`, so that byte never occurred in pretraining). `longalign` holds long documents that can't fit in 1,024 tokens. The rest was filtered with nothing fancy:

| Filter | Training split | Test split |
|---|---|---|
| Conversations | 460,341 | 24,229 |
| Dropped: code and long-document sources | 51,631 | 2,764 |
| Dropped: exact duplicate conversation | 116 | 0 |
| Dropped: first user turn identical to a validation conversation's | 11,987 | not applied |
| Cut to fewer whole exchanges to fit 1,024 tokens | 182,614 | 9,757 |
| Dropped: first exchange alone over 1,024 tokens | 5,872 | 313 |
| Kept | 390,735 | first 2,000 used for validation |

Only 53% of the kept conversations fit whole. The rest were cut after the last assistant reply that fits, so every training example still ends on a complete answer.

The overlap filter turned up repeated prompts. The snail-in-a-well riddle ("A snail is at the bottom of a 20-foot well...") opens 6,350 training conversations, and the bat-and-ball puzzle 2,551, worded identically. Both also appear in the validation set, so without this filter the held-out loss would partly measure memorization.

### Token counts

| Count | Training pool | Validation | This part's run |
|---|---|---|---|
| Conversations | 390,735 | 2,000 | 26,000 |
| Tokens with the template | 245.2M | 1.25M | 16.3M |
| Assistant tokens (trained) | 198.3M (80.9%) | 1.02M (81.2%) | 13.2M (80.7%) |
| Everything else: system, user, template | 46.9M | 0.24M | 3.1M |
| Median length | 716 tokens | 721 tokens | |
| With more than one assistant turn | 64.8% | 65.4% | |

Replies dominate. In [Llama 3](https://arxiv.org/abs/2407.21783)'s SFT data, by comparison, an average example has 535.7 tokens of context, all the turns before the last, and 310.4 in the final response. One pass over the training pool would take about 3.6 hours on this GPU, so the run uses the first 26,000 conversations, whose mix of sources matches the whole pool to within a point.

## Step 3: loss masking and the batch

### Padding, not packing

Pretraining packed documents end to end into 1,024-token rows. That's harmless there: a document reading the end of the previous one is just noise. In a chat batch it's worse, because `model.py` has only a causal mask (`is_causal=True` in PyTorch's attention call), so a packed conversation would attend to a stranger's conversation, and its rotary positions would start wherever the previous one ended. Packing correctly needs a block-diagonal mask and per-conversation positions, and our model takes neither.

So each conversation gets its own row, padded on the right with token 0 and targets of `-1`. Under a causal mask, padding after a conversation can't change anything before it, and it's never trained on. Padding wastes compute, so `plan` shuffles the conversations, sorts windows of 1,000 by length, and cuts them into micro-batches of at most 4,096 padded tokens, so each micro-batch holds conversations of similar length. On the run's 26,000 conversations that wastes 5.2% of the tokens, against 30.8% for random micro-batches.

### Part 5's trap, fixed

Each micro-batch's forward pass returns the mean loss over its own trained tokens. [Part 5](/blog/llm-training-loop/) left a warning: with masked targets, micro-batches hold different numbers of trained tokens, so averaging their means is not the step's true mean. The fix is to scale each micro-batch's mean by its share of the step's trained tokens:

```python
# sft.py (excerpt): one optimizer step
micro = [collate(data, grp, T, a.train_on) for grp, T in steps[step]]
n_step = sum(int((y != -1).sum()) for _, y in micro)      # real targets in this whole step
loss_sum = torch.zeros((), device="cuda")
for x, y in micro:
    n = int((y != -1).sum())
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y) * (n / n_step)    # this micro-batch's share of the step's mean
    loss.backward()
    loss_sum += loss.detach()
```

`n / n_step` turns each mean back into a sum and divides by the step's total, so every trained token in the step counts once. `train.py`'s `loss / a.accum` would weight each micro-batch equally instead.

Measured once, on three real steps of four micro-batches from the base weights, the micro-batches held between 1,210 and 3,358 trained tokens. Against the correct gradient, the mean of means was off by 2.5%, 26.4% and 7.3% in length (relative norm of the difference). In the 26% step, each token of the 1,210-token micro-batch weighed 2.5 times as much as each token of a 3,025-token one. Whole steps varied less: 38,398 to 49,675 trained tokens, apart from a short final step.

The [Tülu 3 report](https://arxiv.org/abs/2411.15124) hit the same issue at 8B scale and traced it to "Averaging the loss across padding tokens without taking into account gradient accumulation or distributed training setups." Its fix was a sum instead of a mean, which "requires an adjustment to learning rates." Dividing by the step's count is the same weighting with the learning rate left alone.

## Step 4: the run

### Speed: eager mode

Every micro-batch now has a different shape. Under `torch.compile(dynamic=True)`, the first of 40 timed micro-batches took 48.6 seconds and a later one 37.3, recompiling for new shapes, and after the first ten they ran anywhere from 4,900 to 33,000 tokens per second. Uncompiled ("eager") PyTorch, which runs each operation as its own GPU kernel, ran a steady 20,100 at micro-batches of 4,096 tokens, using 4.3 GiB. The runs use eager mode, well below the 29,399 tokens per second of Part 7's compiled run, whose every micro-batch had the same shape.

### Choosing the learning rate

Four pilots ran on the first 4,000 conversations: 45 steps each, 3 warm-up steps, cosine decay to zero, held-out loss on the assistant tokens of all 2,000 validation conversations (the base model scores 2.818):

| Peak learning rate | Held-out loss after 45 steps |
|---|---|
| 3 × 10⁻⁵ | 2.405 |
| 1 × 10⁻⁴ | 2.304 |
| 3 × 10⁻⁴ | 2.266 |
| 1 × 10⁻³ | 2.746 |

At 1 × 10⁻³ the training loss climbed from 2.80 to 7.91 by step 9, with a gradient norm of 101, before recovering. [Part 5](/blog/llm-training-loop/) showed that short pilots reward fast starters, so this ranking is only a rough guide. The winner, 3 × 10⁻⁴, half the pretraining peak, was used as is. That's far above Tülu 3's 5 × 10⁻⁶ for 8B, but small models take larger rates: SmolLM2-1.7B's SFT used 3 × 10⁻⁴.

### The run

```bash
python sft.py train --out runs/sft88 --lr 3e-4 --limit 26000 --epochs 1 --budget 4096 --accum 16 \
    --warmup 10 --val-every 50 --ckpt-every 50 --fused
```

| Setting or result | Value |
|---|---|
| Model | 88,096,512 parameters, starting from Part 7's final weights |
| Data | 26,000 conversations, 1 epoch: 16,313,297 tokens, 13,170,480 of them trained |
| Batch and steps | 291 steps of 16 micro-batches of at most 4,096 padded tokens: about 59,100 padded tokens or 89 conversations per step |
| Learning rate | 10 warm-up steps to 3 × 10⁻⁴, cosine to 0 at step 291 |
| Optimizer | Part 5's AdamW groups, betas (0.9, 0.95), epsilon 10⁻⁸, no weight decay, clipping at 1.0 (9 steps clipped: warm-up and the short last step) |
| Precision and mode | BF16 autocast over FP32 weights, eager, fused AdamW, seed 1337 |
| Wall time | 15.3 minutes, including 7 validation passes and 6 checkpoints; not resumed |
| Throughput | 18,823 real tokens/s (19,857 including padding); MFU about 39% |
| Peak GPU memory | 4.65 GiB |
| GPU | SM clock median 1,935 MHz (90% between 1,815 and 2,040), memory 7,000 MHz, 47.0 W average, 80 °C median; `nvidia-smi` reported no power limit |

MFU (model FLOPs utilization) here counts 6 FLOPs per parameter per real token, leaving out attention, against the 25.3 TFLOPS [GPU Part 4](/blog/gpu-profiling-roofline/) measured. Hardware and software are the series' usual: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), WSL2, driver 592.82, PyTorch 2.11.0 with CUDA 12.8. The final weights, a weights-only `state_dict` of 352 MB, are what Part 11 starts from.

### Masking, measured: the all-token ablation

The same command with `--train-on all` trains on every token: system prompts, user turns and template tokens too. Same data, steps, schedule and seed.

![Two stacked line charts against optimizer step, 0 to 291. Top: held-out loss on the assistant tokens of 500 validation conversations. Both runs start at the base model's 2.85, fall to about 2.26 at step 50, 2.16 at 100, 2.08 at 150, 2.02 at 200, and end at 1.986 for the assistant-only run and 1.991 for the all-token run; the two lines overlap everywhere, never more than 0.006 apart. Bottom: training loss per step, with a 9-step average. The assistant-only run falls from about 2.7 to about 1.95; the all-token run, which also averages over prompt tokens, sits about 0.05 to 0.2 higher throughout, falling from about 2.9 to about 2.05.](/diagrams/t10-sft-loss.svg "Held-out reply loss (top) and training loss (bottom) for SFT on assistant tokens only, in blue, and on every token, in orange.")

On held-out replies the two runs never differ by more than 0.006. The bottom curves differ only because they average over different tokens. With 81% of the tokens already replies, masking the other 19% changes little about how well replies are predicted. Where prompts dominate, it matters a bit more: on the summarization subset, where only 21% of tokens are replies, the masked model's reply loss is 0.024 lower.

On the prompt text, the all-token model reaches 2.44 nats against 3.21. It spent capacity learning to write user turns and system prompts, down to 0.92 nats on the formulaic prompts of the constraints subset. Step 5 suggests some of that capacity came out of what the base model knew.

## Step 5: before and after

### What the model writes

Both models got the same prompts in the chat template. Greedy decoding takes the most likely token every time; sampling uses temperature 0.8, top-k 50 and seed 0 ([Fundamentals Part 9](/blog/encoders-decoders-gpt/) explains both). The limit is 200 new tokens. Replies are quoted verbatim and trimmed:

| Prompt | Base model, greedy | SFT model, greedy |
|---|---|---|
| What is the capital of France? | "The capital of France is the capital of France. Its capital is the capital of France." (no end) | "The capital of France is Paris. It is a city known for its rich history, culture, and culture." (no end) |
| Explain photosynthesis to a ten-year-old in two sentences. | "iterateiterateiterateiterate..." (no end) | "Photosynthesis is a process that occurs in the chloroplasts of plants, algae, and certain bacteria, where sunlight is absorbed by the plant. This process is crucial for the growth and development of plants, as it helps to maintain the balance of the sunlight and water in the soil." (ends, 58 tokens) |
| Give me three tips for sleeping better. | "- 1.1.1.1.1.1.1.1..." (no end) | "Sleeping can be a great way to improve your sleep quality, reduce stress, and improve your overall well-being. Here are some tips to help you sleep better: 1. \*\*Sleeping\*\*: Make sure your body is in good shape, and your body is in good shape." (no end) |
| What is 17 + 25? | "percece? 17 + 25?¶¶¶¶¶¶..." (no end) | "To find 17 + 25, we can use the concept of a sequence of numbers. We start by finding the first number, 17, which is 17." (no end) |
| What was the weather like in Pune yesterday? | "The weather like in Pune today is a combination of the weather like in Pune today." (no end) | "In Pune yesterday, the weather was quite mild, with a slight chill. The air was thick with the smell of freshly baked bread" (no end) |
| What is my name? | "perperperper...man, the first time you have a child" (no end) | "My name is Jack, and I'm a professional sports coach. I've been coaching for over 20 years now" (no end) |

The base model produces junk or loops. The SFT model answers in the assistant's register: it restates the question, writes "Here are some tips", and uses Markdown bold. Sampled, its capital answer was "The capital of France is Paris." followed by an invented founder, "Edmund", and it ended after 120 tokens. Asked 17 + 25 with sampling, it ended after 124 tokens with "Therefore, 17 + 25 is 31."

In a two-turn exchange, the base model repeated "I'm planning a trip to Japan in spring." for both turns. The SFT model opened its first reply with "A trip to Japan can be a thrilling experience" and then looped on "Japan has a relatively short summer season". Asked "Which of those would be best for young kids?", it answered with its own first reply's phrases: "While Japan has a relatively short summer season, it's essential to consider the potential risks and benefits". It followed the turn structure but never used the question.

### What can be measured without a judge

| Measure | Base | SFT, assistant tokens | SFT, every token |
|---|---|---|---|
| Held-out loss on assistant tokens (2,000 conversations) | 2.818 | 1.956 | 1.962 |
| The `<\|end\|>` targets alone | 19.01 | 0.73 | 0.70 |
| Loss on the prompt text | 3.56 | 3.21 | 2.44 |
| Replies that end within 256 tokens, greedy (100 held-out prompts) | 0% | 36% | 38% |
| Same, sampled | not run | 40% | 60% |
| Median length of replies that ended, greedy | none ended | 148 tokens | 166 tokens |
| Repeated 4-token sequences in a reply, greedy | 89% | 53% | 54% |
| Same, sampled | not run | 14% | 11% |
| HellaSwag acc_norm | 28.0 ± 0.9 | 28.0 ± 0.9 | 28.2 ± 0.9 |
| ARC-Easy acc | 47.9 ± 2.0 | 44.1 ± 2.0 | 39.3 ± 2.0 |
| PIQA acc | 60.2 ± 2.2 | 58.6 ± 2.3 | 58.3 ± 2.3 |

The held-out prompts are the first user turns (with any system message) of 100 validation conversations. The repeat score is the share of a reply's 4-token sequences that already occurred earlier in it. Sampling wasn't run for the base model because, at 19 nats per `<|end|>`, it wouldn't end anyway. The benchmarks are [Part 9](/blog/evaluate-base-model/)'s, run unchanged through `evals.py` (`OurLM`, `load_task`, `score_task`, `summarize`), zero-shot, in their plain-text formats.

The model learned to stop, but only partly. The `<|end|>` target fell from 19 nats to 0.73, yet 64% of greedy replies still hit the limit. Greedy replies repeat themselves heavily, and a loop, where the most likely next phrase is the one just written, never reaches the end of an answer. Sampling breaks most loops (repeats fell from 53% to 14%) but ended only 40% of replies.

The all-token model ended 60% of sampled replies against 40%. That's 20 points on 100 prompts, about three standard errors. One plausible reason is that it also trained on the `<|end|>` after every user turn, so it saw more end tokens. This run can't confirm that.

### The alignment tax

![Dot chart of zero-shot accuracy with 95% interval bars for three models on three benchmarks, with a dashed tick at chance. HellaSwag acc_norm: base model 28.0, SFT on assistant tokens 28.0, SFT on every token 28.2, all just above the 25% chance line. ARC-Easy acc: base 47.9, assistant-only SFT 44.1, all-token SFT 39.3, each interval about plus or minus 2 points, so the all-token model sits clearly below the base. PIQA acc: base 60.2, assistant-only SFT 58.6, all-token SFT 58.3, against a chance line at 50.](/diagrams/t10-before-after.svg "Part 9's benchmarks before and after SFT: the base model in gray, SFT on assistant tokens in blue, SFT on every token in orange.")

The alignment tax is the drop in a model's existing abilities caused by post-training. Here it's uneven. HellaSwag didn't move. PIQA fell 1.6 points, within its interval. ARC-Easy fell 3.8 points, nearly twice its ±2.0 interval, and 8.6 points after the all-token run. The masked model kept more of the base model's grade-school science. These benchmarks score plain-text prompts by log-likelihood, a format the fine-tuning trained away from, so some of the drop may be format rather than lost knowledge. Each number comes from one run.

### What happened to the weights

Relative to the base weights, SFT moved the whole model by 13% (L2 norm of the change over the norm of the weights): the embedding table 16%, attention and MLP matrices 12% each, the RMSNorm gains 0.6%. The special tokens' rows moved no more than ordinary rows did: `<|end|>` by 0.19 and `<|user|>` by 0.09, against a median row's 0.12. Their lengths stayed at 1.29 to 1.31, the four chat tokens still have pairwise cosines of 0.86 to 0.88, and their nearest neighbors are still control and invalid bytes.

So Part 2's promise is only half kept. The rows were trained, but 291 steps barely separated them. The model presumably tells `<|user|>` from `<|assistant|>` by those small differences plus where each sits in the conversation. A longer run, or a larger learning rate for those eight rows alone, might separate them further. That wasn't tested here.

## In the real world

Published SFT stages, next to this one:

| Recipe | SFT data | Learning rate | Epochs or steps | Batch | Context |
|---|---|---|---|---|---|
| [InstructGPT](https://arxiv.org/abs/2203.02155) (2022) | "about 13k training prompts" | 9.65 × 10⁻⁶ (1.3B, 6B), 5.03 × 10⁻⁶ (175B) | 16 epochs | 32, or 8 at 175B | not given |
| [Llama 3](https://arxiv.org/abs/2407.21783) (2024) | total not given; 846.1 tokens per example on average | 10⁻⁵ | 8.5K to 9K steps | not given | not given |
| [Tülu 3](https://arxiv.org/abs/2411.15124) 8B (2024) | 939,344 prompts | 5 × 10⁻⁶, sum loss | 2 epochs | 128 | 4,096 |
| [SmolLM2](https://arxiv.org/abs/2502.02737) 1.7B (2025) | SmolTalk, 1.1M pairs | 3 × 10⁻⁴ | 2 epochs | 128 | 8,192 |
| This part, 88M | 26,000 conversations, 13.2M trained tokens | 3 × 10⁻⁴ | 1 epoch, 291 steps | about 89 conversations | 1,024 |

- Llama 3 describes the same mask as `render`: "a standard cross entropy loss on the target tokens (while masking loss on prompt tokens)." Its largest models "are finetuned with a learning rate of 10⁻⁵ over the course of 8.5K to 9K steps."
- InstructGPT saw the overfitting that one epoch here can't show: "our SFT models overfit on validation loss after 1 epoch; however, we find that training for more epochs helps both the RM score and human preference ratings, despite this overfitting."
- Tülu 3 found that "training for longer did not yield further improvements, and so used 2 epochs for training." [OLMo 2](https://arxiv.org/abs/2501.00656)'s instruct models "follow the Tülu 3 recipe," with an SFT mix of 939,104 prompts for the 7B and 13B models.

At 88M parameters the ceiling is the base model. It got 47.9% on ARC-Easy and invented "Charles Columbus" in Part 7, and no template fixes that. SFT changed how the model handles a question: it answers in the assistant's voice, at an assistant's length, and sometimes stops. Its knowledge is the base model's. It makes facts up confidently ("My name is Jack"), loses track of what was said in the first turn, and loops under greedy decoding. SmolLM2-135M-Instruct, by comparison, had 2 trillion pretraining tokens behind it, 1,100 times ours.

## Common misconceptions

- **"Masking the prompt is what makes SFT work."** Here, training on every token reached the same held-out reply loss, within 0.006. Masking mattered elsewhere: the unmasked model learned to imitate prompts and lost 4.8 more points on ARC-Easy.
- **"Averaging micro-batch losses is fine with gradient accumulation."** Only when every micro-batch has the same number of trained tokens. With masked chat data, the mean of means was off by up to 26% on a real step; dividing by the step's trained-token count fixes it.
- **"Once the model learns the end token, it stops."** The `<|end|>` loss fell from 19 nats to 0.73, yet 64% of greedy replies ran to the limit, mostly because a loop never reaches the end of an answer.
- **"Fine-tuning teaches the model facts from the data."** 13 million trained tokens taught a format. The model still says 17 + 25 is 31 and invents a name for the user.
- **"New special tokens get their own meaning during SFT."** After 291 steps the chat tokens' rows had moved no more than ordinary rows and still sat together among the never-seen bytes.

## Key takeaways

- A chat template is a rule for laying out a conversation in tokens. This one is `<|role|>`, the text and `<|end|>` for each message, after an `<|endoftext|>`. Generation starts after `<|assistant|>` and stops at `<|end|>`.
- Loss masking means targets of -1 everywhere except the assistant's replies and their end token. With masked targets, divide by the step's total count of trained tokens, not by the number of micro-batches.
- Pad conversations into length-sorted micro-batches rather than packing them, unless the model supports per-conversation masks and positions. Sorting cut the padding from 31% to 5%.
- Fifteen minutes on 26,000 conversations took the held-out reply loss from 2.82 to 1.96 and taught the 88M model an assistant's format. Greedy replies still loop, and only 36% end on their own.
- The alignment tax showed up as 3.8 points on ARC-Easy, and 8.6 when the loss also covered the prompts.

## What's next

SFT taught the model to imitate good replies, but it never saw a bad one. Part 11 starts from these SFT weights and adds preference tuning with DPO: pairs of a better and a worse answer to the same prompt, the DPO loss that raises one and lowers the other against a frozen reference, and a before-and-after look at whether a model this small can learn what people prefer, and whether it loops less.

## Further reading

- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124): a fully open SFT recipe, including the loss-aggregation problem and the sum-loss fix.
- [The Llama 3 Herd of Models (Llama Team, Meta, 2024)](https://arxiv.org/abs/2407.21783): Section 4, the chat format, masked SFT and SFT data statistics.
- [SmolLM2: When Smol Goes Big (Ben Allal et al., 2025)](https://arxiv.org/abs/2502.02737): how SmolTalk and smol-smoltalk were built, and SFT settings for small models.
- [smol-smoltalk dataset card (Hugging Face)](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk): the data used here, its sources and its license.
- [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing (Xu et al., 2024)](https://arxiv.org/abs/2406.08464): the method behind most of the conversations.
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155): InstructGPT's SFT stage, and why validation loss isn't the target.
- [2 OLMo 2 Furious (OLMo team, 2024)](https://arxiv.org/abs/2501.00656): Section 5, Tülu 3's recipe applied to another open model family.
- [Meta Llama 3 prompt format (Meta)](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/) and [ChatML (OpenAI)](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md): two published chat templates and their special tokens.
- [SmolLM2-135M-Instruct tokenizer configuration (Hugging Face)](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/main/tokenizer_config.json): a ChatML template as a model actually ships it.
