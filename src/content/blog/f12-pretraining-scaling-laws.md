---
title: "Pretraining: Data, Scaling Laws and Training at Scale"
description: "Where pretraining text comes from and how it's filtered, what scaling laws say about model size versus data, and how a run spreads over 16,000 GPUs."
slug: "pretraining-scaling-laws"
category: "llm-fundamentals"
tags: ["training", "scaling-laws", "data", "cost", "gpu"]
series: "llm-fundamentals"
seriesOrder: 12
pubDate: 2026-04-01
updatedDate: 2026-10-23
---

## TL;DR

Pretraining is the long first phase of training, where a model learns next-token prediction on trillions of tokens of text. This article follows that text from a raw web crawl to a filtered training mix, explains where the `6 × N × D` compute rule comes from, and shows what scaling laws say about trading model size against data. Then it covers the recipe that keeps a months-long run stable (AdamW, cosine decay, mixed precision, gradient clipping) and how Meta spread Llama 3.1 405B over 16,384 GPUs, where the job stopped unexpectedly every few hours.

## Why it matters

In 2022, DeepMind trained a model a quarter the size of its own Gopher on about the same compute, and the smaller model won. Chinchilla, with 70 billion parameters, beat the 280-billion-parameter Gopher "uniformly and significantly" across the authors' evaluations and scored 67.5% on the MMLU knowledge benchmark against Gopher's 60.0% ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). Chinchilla had read 1.4 trillion tokens; Gopher had read 300 billion.

Getting that split wrong is expensive. [Part 1](/blog/how-llms-work/) showed that Llama 3.1 405B took 30.84 million GPU-hours. At an illustrative rental price of $2 per GPU-hour, that's about $62 million for one run. The model's size, its data and its recipe all have to be fixed before it starts.

## The intuition

Think of studying for an exam with a fixed number of hours. You can spend them building a thicker set of notes or reading more books. Each extra page of notes makes every book slower to work through, because there's more to update as you read. Compute is the hours, the notebook is the parameter count `N`, and the books are the training tokens `D`. Scaling laws are the measured answer to "how big a notebook for how many hours?"

The analogy breaks in three places. Some books aren't worth reading, so filtering matters as much as volume. Rereading a good book helps almost as much as a new one, up to a point. And after the exam you carry the notebook around for years: a model runs billions of times after training, which is why labs often pick a smaller model than the training budget alone would justify.

## Step 1: from a web crawl to a training set

Most pretraining text starts as web pages. [Common Crawl](https://commoncrawl.org/), a non-profit founded in 2007, has archived "over 300 billion pages" and adds 3 to 5 billion each month. Its [August 2026 crawl](https://commoncrawl.org/blog/august-2026-crawl-archive-now-available) alone holds 2.14 billion pages, 360 TiB uncompressed, gathered in two weeks. The web, as FineWeb's authors put it, has "many aggregators, mirrors, templated pages or just otherwise repeated content."

![Three stages left to right. Clean the crawl: Common Crawl raw web pages, extract text and drop menus and ads, language ID, URL, safety and PII filters, heuristic filters for repetition and punctuation. Select: deduplicate by URL, MinHash and lines, a quality classifier trained on LLM labels, then a mix by domain covering web, code, math and languages, which also takes in other sources such as code, books and papers. Train: tokenize, pretrain on trillions of tokens, anneal on top-quality data.](/diagrams/f12-data-pipeline.svg "A typical pretraining data pipeline, following FineWeb and Llama 3.")

Every stage throws text away, and the expensive model-based scoring runs late, on what the cheap rules left behind.

### The pipeline, stage by stage

The [Llama 3 paper](https://arxiv.org/abs/2407.21783) and Hugging Face's [FineWeb paper](https://arxiv.org/abs/2406.17557) describe the same steps:

1. Extract the text. Crawls come as raw HTML (WARC files) or as pre-extracted text (WET files). FineWeb found the WET text "retained too much boilerplate and menu text" and got better models by extracting its own with the trafilatura library. Meta wrote a custom parser that keeps math and code intact, and strips Markdown after finding that it hurt.
2. Identify the language. A small fastText classifier labels each page. FineWeb keeps pages scored as English with probability at least 0.65; Llama 3 sorts documents into 176 languages.
3. Filter for safety and privacy. URL blocklists remove adult sites, and both pipelines drop or mask personally identifiable information (PII). FineWeb anonymizes email and public IP addresses.
4. Apply heuristics. Simple rules catch repeated lines, pages where few lines end in punctuation, and "dirty word" counts. Llama 3 also deletes any line that appears more than 6 times in a bucket of 30 million documents, which strips cookie banners and navigation menus along with some frequent good text.
5. Deduplicate. Exact matching catches copies of a URL or document; MinHash catches near-copies.
6. Score quality with a model. Meta trained small DistilRoBERTa classifiers on quality labels produced by Llama 2, plus separate ones for code and math pages.

### MinHash, by the numbers

Comparing every page with every other page is impossible at billions of pages, so MinHash compresses each document into a short signature that collides with the signatures of similar documents. FineWeb takes each document's 5-word phrases, computes 112 hash values, and groups them into 14 buckets of 8. Two documents count as duplicates if all 8 values in any one bucket match.

If two documents' phrase sets overlap by a fraction `s`, one bucket matches with probability `s⁸`, and at least one of the 14 matches with probability `1 − (1 − s⁸)¹⁴`:

| Overlap `s` | One bucket matches | Flagged as duplicate |
|---|---|---|
| 0.50 | 0.4% | 5.3% |
| 0.70 | 5.8% | 56.5% |
| 0.75 | 10.0% | 77.2% |
| 0.80 | 16.8% | 92.4% |
| 0.90 | 43.1% | 99.96% |

The curve turns sharply around 75% overlap, the threshold FineWeb's settings target. Pages that share half their phrases almost always survive, and near-copies almost never do.

### The FineWeb funnel

FineWeb processed 96 Common Crawl snapshots and checked each step by training small models (1.71 billion parameters) on the result:

| Stage | Tokens left (GPT-2 tokenizer) |
|---|---|
| Text from 96 snapshots, after base filtering | about 36 trillion |
| MinHash deduplication within each snapshot | 20 trillion |
| C4's filters, three custom filters, PII masking: FineWeb | 15 trillion |
| Educational-quality score of 3 or more: FineWeb-Edu | 1.3 trillion |

Deduplicating across all 96 snapshots at once went wrong. It left only 4 trillion tokens and barely improved the model. In one older snapshot, from 2013, the 10% that survived was worse than the 90% removed, with more "ads, incoherent lists of keywords and generally badly formatted text." Deduplicating each snapshot on its own gave better models.

The quality classifier paid off most. Hugging Face had Llama-3-70B-Instruct rate 460,000 pages for educational value from 0 to 5, trained a small classifier on those ratings, and ran it over all 15 trillion tokens in 6,000 H100 GPU-hours. Training only on pages scoring 3 or higher raised MMLU from 33% to 37% and the ARC reasoning benchmark from 46% to 57%. FineWeb-Edu reached another dataset's final MMLU score with almost 10 times fewer tokens.

### Mixing, annealing and repetition

Meta's final Llama 3 mix was "roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens." Meta chose it by training small models on candidate mixes and extrapolating with scaling laws, and it downsampled topics the web over-represents, "for example, arts and entertainment."

The best data comes last. Over the final 40 million tokens, Meta lowered the learning rate to zero while upsampling very high-quality sources, a step called annealing. Annealing raised Llama 3 8B's scores on the GSM8K and MATH benchmarks by 24.0% and 6.4%; on 405B the gains were "negligible."

Good text eventually runs out. [Muennighoff et al. (2023)](https://arxiv.org/abs/2305.16264) found that "up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data." An 8.7-billion-parameter model that saw 44 billion unique tokens four times ended with only 0.5% higher loss than one that saw 178 billion tokens once. Beyond that the value drops quickly, and by 40 epochs repetition adds nothing. Labs also have an LLM rewrite good text in new words. On the SimpleQA fact-recall benchmark, an early [Kimi K2](https://arxiv.org/abs/2507.20534) checkpoint scored 28.94 after reading 10 rephrasings once, against 23.76 after reading the original 10 times.

## Step 2: how big, and how long

### Where the 6 in `6 × N × D` comes from

[Part 3](/blog/neural-networks-basics/) showed that a forward pass costs about 2 FLOPs (floating-point operations) per parameter per token, and backpropagation about twice that. One weight matrix shows why. Take a 4,096 × 4,096 layer with 16.8 million weights. Training runs three matrix multiplications through it for each token:

| Multiplication | What it computes | FLOPs per token |
|---|---|---|
| Forward | The layer's output | 33.6 million (2 per weight) |
| Backward, for inputs | The slope to pass down to the layer below | 33.6 million |
| Backward, for weights | The slope for each weight | 33.6 million |

That's 6 FLOPs per weight per token. The [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) names the same three: "Fprop (forward pass), Dgrad (activation backward pass), and Wgrad (weight backward pass)." Multiply by every parameter `N` and every training token `D` and you get `C ≈ 6 × N × D`.

For Llama 3.1 405B, `6 × 405 billion × 15.6 trillion = 3.79 × 10²⁵` FLOPs, matching the paper's 3.8 × 10²⁵. For a mixture-of-experts model, `N` is the active count ([Part 11](/blog/mixture-of-experts/)). The rule leaves out attention's token-to-token comparisons, which [Kaplan et al.](https://arxiv.org/abs/2001.08361) estimate at `2 × layers × context × width` FLOPs per token going forward. For 405B with 8,192 tokens of context, that adds about 4%. At 128,000 tokens it would add about 67%, one reason Meta trained on long sequences only near the end, for about 800 billion tokens.

### From FLOPs to GPU-days

MFU (model FLOPs utilization) is the share of a chip's peak speed that goes into the model's own arithmetic. An H100's dense BF16 peak is about 989 TFLOPS ([How GPUs Work for LLMs](/blog/gpus-for-llms/)). Meta reports 380 to 430 TFLOPS per GPU, an MFU of 38% to 43%; communication, memory traffic and waiting take the rest.

```text
GPU-seconds = 3.79 × 10²⁵ FLOPs / 400 × 10¹² FLOPs per second = 9.5 × 10¹⁰
            = 26.3 million GPU-hours
on 16,384 GPUs: 26.3 million / 16,384 = 1,606 hours ≈ 67 days
```

Meta's model card lists 30.84 million GPU-hours, 17% more than this ideal, or about 342 TFLOPS per GPU averaged over the run. The paper gives no start and end dates. The gap covers restarts, checkpoints, slower long-context stages and stragglers.

### Kaplan's laws: grow the model first

[Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) found that test loss falls as a power law in `N`, `D` and compute `C`, "with some trends spanning more than seven orders of magnitude." On a log-log plot a power law is a straight line: each 10 times more compute cuts the loss by the same fraction.

Their advice was to spend most of a bigger budget on model size. Optimal size grew as `N ∝ C^0.73`, so 10 times more compute should buy a 5.4 times bigger model and only 1.9 times more data, "stopping significantly before convergence." The large models that followed, GPT-3 and Gopher among them, trained on about 300 billion tokens each, under 2 tokens per parameter.

### Chinchilla's correction: 20 tokens per parameter

Hoffmann et al. trained over 400 models, from 70 million to 16 billion parameters, and matched each run's learning-rate schedule to its length, where Kaplan's runs had shared one schedule and token count. All three of their fitting methods put the exponents for `N` and `D` near 0.5: "for every doubling of model size the number of training tokens should also be doubled." Their Table 3 comes to about 20 tokens per parameter: 20.2 billion tokens for a 1-billion-parameter model, 205.1 billion for 10 billion.

With `D = 20N`, the budget is `C = 6 × N × 20N = 120N²`, so `N = √(C / 120)`:

| Compute budget | Compute-optimal parameters | Training tokens |
|---|---|---|
| 10²² FLOPs | 9.1 billion | 183 billion |
| 10²³ FLOPs | 28.9 billion | 577 billion |
| 10²⁴ FLOPs | 91 billion | 1.83 trillion |
| 10²⁵ FLOPs | 289 billion | 5.77 trillion |
| 3.8 × 10²⁵ (Llama 3.1 405B) | 563 billion | 11.3 trillion |

Each 10 times more compute buys about 3.2 times more parameters and 3.2 times more tokens. Chinchilla, at 70 billion parameters and 1.4 trillion tokens, sits exactly at 20 to 1; Gopher had about 1.

The ratio depends on the data. Meta fitted its own law on Llama 3 data, with runs from 6 × 10¹⁸ to 10²² FLOPs. Extrapolated to 3.8 × 10²⁵, it suggested "a 402B parameter model on 16.55T tokens," about 41 tokens per parameter, close to the 405 billion on 15.6 trillion that Meta chose. Meta also used the small runs to forecast benchmark accuracy, and its forecast for the ARC Challenge, extrapolated over four orders of magnitude, "only slightly underestimates" the final model.

### Why small models get trained far past Chinchilla

Compute-optimal means the lowest loss for a training budget, and it ignores the cost of running the model afterward. Llama 3 8B trained on 15 trillion tokens, about 1,875 tokens per parameter, 94 times Chinchilla's ratio. The same 7.2 × 10²³ FLOPs could have trained a compute-optimal model of about 77 billion parameters on 1.55 trillion tokens.

Chinchilla's fitted loss formula, `L = 1.69 + 406.4 / N^0.34 + 410.7 / D^0.28`, gives a rough comparison (it was fitted on DeepMind's data, so treat the values as illustrative). The 77B model reaches 1.929 and the over-trained 8B model 1.949, about 1% higher. The 8B model then costs about 9.7 times less for every token it generates. Meta trained its smaller models "much longer than is compute-optimal" and reports that they "perform better than compute-optimal models at the same inference budget."

### Smooth loss, sudden abilities?

Some benchmark scores don't improve smoothly. [Wei et al. (2022)](https://arxiv.org/abs/2206.07682) called an ability emergent "if it is not present in smaller models but is present in larger models." On a BIG-Bench arithmetic test, GPT-3 scored near zero until about 2 × 10²² training FLOPs (13 billion parameters), then jumped. [Schaeffer et al. (2023)](https://arxiv.org/abs/2304.15004) traced much of this to scoring. Exact-match accuracy needs every token right, so steady per-token gains look like a leap. More than 92% of the emergent abilities they examined on BIG-Bench appeared under such all-or-nothing metrics, and with smoother metrics the jumps often "evaporate."

## Step 3: a recipe that survives months of training

### AdamW, warm-up and cosine decay

Most large models train with AdamW, the Adam optimizer from Part 3 with decoupled weight decay. Weight decay shrinks every weight slightly each step, pulling weights the loss doesn't need toward zero. [Loshchilov and Hutter](https://arxiv.org/abs/1711.05101) showed that the usual way of adding it to Adam, through the gradient, isn't equivalent to real weight decay, and that applying it directly to the weights works better. DeepSeek-V3 uses `β1 = 0.9`, `β2 = 0.95` and weight decay 0.1.

The learning rate ramps up linearly (the warm-up from [Part 8](/blog/attention-is-all-you-need/)) and then usually follows a cosine curve down. A sketch of Llama 3.1 405B's schedule:

```python
import math

def lr(step, peak=8e-5, final=8e-7, warmup=8_000, total=1_200_000):
    if step < warmup:
        return peak * step / warmup          # linear warm-up
    progress = (step - warmup) / (total - warmup)
    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * progress))

# step 2,000: 2.0e-05 | 8,000: 8.0e-05 | 600,000: 4.1e-05 | 1,200,000: 8.0e-07
```

The rate falls slowly at first, fastest in the middle and slowly again near the end, finishing 100 times below the peak. Chinchilla's authors found the cosine length has to match the run: overshoot the real number of steps by more than 25% and "performance is noticeably degraded." Since that makes a run hard to extend, some labs hold the rate flat and decay late. DeepSeek-V3 stayed at 2.2 × 10⁻⁴ until 10 trillion tokens, then followed a cosine down to 2.2 × 10⁻⁵ over 4.3 trillion more.

Batches grow during the run. Llama 3.1 405B went from 4 million to 16 million tokens per batch (Part 3 lists the steps), and DeepSeek-V3 grew from 3,072 to 15,360 sequences over its first 469 billion tokens. Meta used the smaller early batches for stability and the larger later ones for efficiency. Dropout, from Part 8, is usually left out of pretraining; [PaLM](https://arxiv.org/abs/2204.02311) trained without it.

### Mixed precision and FP8

Matrix multiplications run in 16-bit BF16, half the memory and traffic of 32-bit, but small updates vanish in 16-bit rounding. So training keeps a 32-bit master copy of the weights. Chinchilla computed "the forward and backward pass" in bfloat16 and stored "a float32 copy of the weights" with the optimizer. Meta also accumulates gradients in FP32 and sums them across GPUs in FP32, to fix numerical problems that hurt stability. Those extra copies are why Adam costs [16 bytes per parameter](/blog/neural-networks-basics/).

DeepSeek-V3 went lower. All three multiplications per layer ran in 8-bit FP8 ([Inference Part 8](/blog/quantization/) explains the format), with fine-grained scaling to work around FP8's narrow range. Adam's running averages were kept in BF16, and master weights and gradients in FP32. Tested on smaller models over about 1 trillion tokens, the loss stayed within 0.25% of a BF16 run.

### Loss spikes, failures and checkpoints

Gradient clipping caps each update: if the gradient's overall length exceeds a limit, usually 1.0, it's scaled down to that limit. Runs can still diverge. Meta's [OPT-175B](https://arxiv.org/abs/2205.01068) team recovered by "lowering the learning rate and restarting from an earlier checkpoint," tightened clipping from 1.0 to 0.3, and published its logbook. Hardware failures there caused at least 35 manual restarts and an estimated 70 or more automatic ones.

At 16,000 GPUs, hardware fails constantly. In a 54-day window of Llama 3 405B pretraining, the job stopped 466 times: 47 planned, 419 unexpected, or one unexpected stop about every 3 hours. About 78% of the unexpected ones were confirmed or suspected hardware faults, led by faulty GPUs (148) and GPU memory (72). Training is synchronous, so "a single GPU failure may require a restart of the entire job." Meta's answer was frequent checkpoints, saving each GPU's share of the model state (1 MB to 4 GB per GPU), and fast automated restarts. Effective training time stayed above 90%, and only three failures needed significant manual intervention. Kimi K2 later trained on 15.5 trillion tokens "without a single loss spike."

## Step 4: spreading the run over 16,000 GPUs

### Data parallelism and sharding

In data parallelism, every GPU holds a full copy of the model, works on different batches, and averages gradients with the others through an all-reduce after each step ([Inference Part 10](/blog/model-parallelism/) covers the communication). Training's memory bill gets in the way. At 16 bytes per parameter, a 70-billion-parameter model needs 1,120 GB for weights, gradients and Adam's state, 14 times an 80 GB GPU.

[ZeRO](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer) removes the duplicate copies in stages. Each of `N_d` data-parallel GPUs keeps only its slice of the optimizer state, then of the gradients too, then of the weights, fetching other slices when it needs them. For 70B across 64 GPUs, using the paper's formulas:

| What's sharded | Formula (bytes, Ψ = parameters) | Per GPU |
|---|---|---|
| Nothing (plain data parallelism) | 16Ψ | 1,120 GB |
| Optimizer state (stage 1) | 4Ψ + 12Ψ/64 | 293 GB |
| Plus gradients (stage 2) | 2Ψ + 14Ψ/64 | 155 GB |
| Plus weights (stage 3) | 16Ψ/64 | 17.5 GB |

PyTorch's version of stage 3 is [FSDP](https://arxiv.org/abs/2304.11277) (Fully Sharded Data Parallel). Before each layer runs, the GPUs gather its full weights, compute, then free the borrowed pieces; after the backward pass, a reduce-scatter leaves each GPU with only its slice of the gradients.

### Llama 3's 4D parallelism

For 405B, Meta combined four kinds of parallelism, ordered from the fastest links to the slowest. Tensor parallelism (TP) splits each weight matrix across the 8 GPUs of one NVLink server. Context parallelism (CP) splits very long sequences across GPUs. Pipeline parallelism (PP) gives groups of GPUs consecutive layers. FSDP-style data parallelism (DP) sits outermost because it tolerates the most network latency.

| GPUs | TP | CP | PP | DP | Sequence length | Tokens per batch | TFLOPS per GPU | MFU |
|---|---|---|---|---|---|---|---|---|
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 16M | 430 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 16M | 400 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 16M | 380 | 38% |

The four columns multiply to the GPU count: `8 × 1 × 16 × 128 = 16,384`. Doubling the GPUs at a fixed 16 million tokens per batch halved each data-parallel group's share, and MFU fell 2 points. The long-context stage traded data parallelism for context parallelism to fit 131,072-token sequences. The GPUs were linked by 400 Gbps network connections.

## In the real world

Recent runs, with an illustrative $2 per GPU-hour applied to each lab's reported GPU-hours:

| Model | Parameters (active) | Training tokens | Tokens per parameter | Compute (6ND) | GPU-hours | At $2 per GPU-hour |
|---|---|---|---|---|---|---|
| Chinchilla (2022) | 70B | 1.4T | 20 | 5.9 × 10²³ | not reported (TPUs) | n/a |
| Llama 3 8B (2024) | 8B | 15T+ | about 1,875 | 7.2 × 10²³ | 1.3M H100 | $2.6M |
| Llama 3 70B (2024) | 70B | 15T+ | about 214 | 6.3 × 10²⁴ | 6.4M H100 | $12.8M |
| Llama 3.1 405B (2024) | 405B | 15.6T | about 39 | 3.8 × 10²⁵ | 30.84M H100 | $61.7M |
| DeepSeek-V3 (2024) | 671B (37B) | 14.8T | about 400 per active | 3.3 × 10²⁴ | 2.788M H800 | $5.576M |

Sources: [Chinchilla](https://arxiv.org/abs/2203.15556), [Llama 3 model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md), [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md), [DeepSeek-V3](https://arxiv.org/abs/2412.19437).

- From first principles, a 70B model on 15 trillion tokens needs `6 × 70 billion × 15 trillion = 6.3 × 10²⁴` FLOPs. At 400 TFLOPS per GPU that's 4.4 million GPU-hours, or $8.8 million at $2. Meta reported 6.4 million for Llama 3 70B and 7.0 million for Llama 3.1 70B, 45% to 60% above the ideal.
- DeepSeek-V3's $5.576 million is its own figure, "assuming the rental price of the H800 GPU is $2 per GPU hour." It covers "only the official training of DeepSeek-V3, excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data." The total splits into 2,664K GPU-hours of pretraining, 119K for context extension and 5K for post-training, on 2,048 H800s: 3.7 days per trillion tokens and under two months in all.

## Common misconceptions

- **"Models are trained on the whole internet."** They're trained on a heavily filtered sliver. FineWeb kept 15 trillion tokens from 96 crawls, and its educational subset kept 1.3 trillion. Llama 3 also downsampled whole topics.
- **"More data always beats cleaner data."** FineWeb-Edu matched another dataset's MMLU score with almost 10 times fewer tokens, and FineWeb's aggressive global deduplication kept the worst pages of an older crawl.
- **"Chinchilla says to train 20 tokens per parameter."** Twenty is the cheapest way to reach a given loss in training. Models that will serve many users are trained far longer: Llama 3 8B went to about 1,875.
- **"Scaling laws predict what a model can do."** They predict loss well. Benchmark scores need an extra modeling step, which Meta built for Llama 3, and some apparent jumps in ability come from how they're scored.
- **"Training compute is the cost of a model."** Reported GPU-hours cover the final run. Experiments, failed runs, data processing and staff come on top, as DeepSeek's report says of its own figure.

## Key takeaways

- Pretraining data starts as raw web crawls and goes through extraction, language identification, safety filters, heuristics, deduplication and quality classifiers. Most of the text is thrown away, and what's kept matters as much as how much.
- Training costs about 6 FLOPs per parameter per token: one multiplication forward and two backward. Llama 3.1 405B needed 3.8 × 10²⁵ of them, about 31 million H100-hours at roughly 40% utilization.
- Kaplan's scaling laws said to grow models fastest. Chinchilla showed parameters and tokens should grow together, at about 20 tokens per parameter, and its 70B model beat the 280B Gopher on the same budget.
- Models that will be used heavily are deliberately over-trained, because a smaller model costs less every time it runs.
- Large runs rely on AdamW, warm-up and cosine decay, BF16 or FP8 math with FP32 master weights, gradient clipping, frequent checkpoints, and four kinds of parallelism to keep 16,000 GPUs working through hundreds of failures.

## What's next

Pretraining produces a base model that continues text but doesn't follow instructions. [Part 13](/blog/base-model-to-assistant/) covers how it becomes an assistant: supervised fine-tuning on example conversations, reinforcement learning from human feedback, and DPO, a simpler method many labs now use in its place.

## Further reading

- [The Llama 3 Herd of Models (Llama Team, Meta, 2024)](https://arxiv.org/abs/2407.21783): the most detailed public account of a large pretraining run, from the data pipeline and scaling laws to 4D parallelism and failure statistics.
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (Penedo et al., 2024)](https://arxiv.org/abs/2406.17557): every filtering and deduplication choice, tested with ablation models, plus FineWeb-Edu.
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361): the original power laws and the `6N` FLOPs-per-token estimate.
- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556): the Chinchilla paper and its three methods for splitting a budget between parameters and tokens.
- [Scaling Data-Constrained Language Models (Muennighoff et al., 2023)](https://arxiv.org/abs/2305.16264): how much repeated data is worth.
- [Emergent Abilities of Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2206.07682) and [Are Emergent Abilities of Large Language Models a Mirage? (Schaeffer et al., 2023)](https://arxiv.org/abs/2304.15004): both sides of the emergence debate.
- [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101): the AdamW paper.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054) and [PyTorch FSDP (Zhao et al., 2023)](https://arxiv.org/abs/2304.11277): sharding training state across data-parallel GPUs.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): FP8 training at scale and a transparent cost breakdown.
- [OPT: Open Pre-trained Transformer Language Models (Zhang et al., 2022)](https://arxiv.org/abs/2205.01068) and its [logbook](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles): what goes wrong in a 175B run, day by day.
- [Common Crawl](https://commoncrawl.org/): the open web archive behind most pretraining datasets.
- [FineWeb: decanting the web for the finest text data at scale (Hugging Face, 2024)](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1): the readable companion to the FineWeb paper, with interactive plots.
- [The Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook): data, tensor, pipeline and context parallelism and ZeRO, explained with measurements.
- [How To Scale Your Model (Google DeepMind)](https://jax-ml.github.io/scaling-book/): a free book on the arithmetic of training and serving on thousands of chips.
