---
title: "Mixture of Experts: How Giant Models Use a Fraction of Their Weights"
description: "How MoE models work: a router picks a few expert MLPs per token, why total and active parameters differ, and how training keeps experts balanced."
slug: "mixture-of-experts"
category: "llm-fundamentals"
tags: ["moe", "architecture", "transformers", "cost"]
series: "llm-fundamentals"
seriesOrder: 11
pubDate: 2026-10-23
---

## TL;DR

A mixture-of-experts (MoE) model replaces the single MLP in each Transformer block with many smaller MLPs, called experts, plus a router that picks a few of them for each token. Only the chosen experts run, so a model can store hundreds of billions of parameters while each token uses a few percent of them. This article works through a router by hand, shows why compute per token follows the active parameters while memory follows the total, and covers the tricks that keep experts from collapsing during training: load-balancing losses, capacity limits, the router z-loss, DeepSeek's bias-based balancing, and fine-grained and shared experts. It ends with the real configurations of Mixtral, DeepSeek, Qwen, Llama 4, gpt-oss and Kimi.

## Why it matters

DeepSeek-V3 has 14,848 small expert networks, 256 in each of its 58 mixture-of-experts layers. A token passing through the model uses 464 of them, about 3%. The rest sit idle for that token, and the next token picks a different 464.

That is how DeepSeek-V3 holds [671 billion parameters](https://arxiv.org/abs/2412.19437) but spends about as much arithmetic per token as a 37-billion-parameter model, roughly 18 times less than a dense model of the same size. [Part 1](/blog/how-llms-work/) showed that most of the largest open models since late 2024 work this way. This part explains how.

The idea is older than the Transformer. It dates to 1991, and a 2017 version of it appears in [Part 8's](/blog/attention-is-all-you-need/) results table as one of the models the Transformer beat. Using it at today's scale took a string of training tricks, from 2017 to 2024, that keep thousands of experts evenly busy.

## The intuition

Think of a large hospital. A patient arrives and a triage nurse sends them to two of the hospital's eight specialists. A general practitioner also sees every patient. The hospital employs a lot of doctors, but each patient takes up only three doctors' time, so it can treat far more patients than if every doctor saw everyone.

The specialists are experts, the triage nurse is the router, and the general practitioner is a shared expert. The hospital's payroll is the total parameter count; the doctors one patient sees are the active parameters.

The analogy breaks in three places. Routing happens per token and per layer, not once per question, so one sentence may visit dozens of different experts on its way through the model. The nurse is a single small matrix, learned along with everything else. And the specialists don't end up with tidy specialties like cardiology or neurology. Step 4 shows what they do specialize in.

## Step 1: from one MLP to many experts

[Part 6](/blog/transformer-block/) showed that each Transformer block has two halves: attention, which moves information between tokens, and an MLP, which works on each token alone and holds about two-thirds of the block's parameters. MoE leaves attention alone and swaps the MLP for a set of experts. Each expert is an ordinary MLP, often a smaller one.

The layer's output for a token `x` is:

```text
y = Σ g_i × E_i(x)    summed over the k experts the router picked
```

`E_i(x)` is expert `i`'s output for the token, a vector the same size as `x`, and `g_i` is the router's weight for that expert. The weights of the chosen experts add up to 1, and every other expert gets weight 0 and isn't run at all. The savings come from those zeros: an expert that isn't run costs no arithmetic.

### Thirty years of experts

| Year | Work | What it added |
|---|---|---|
| 1991 | [Jacobs, Jordan, Nowlan and Hinton](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) | Several "expert" networks plus a gating network that "decides which of the experts should be used for each training case"; tested on telling vowels apart |
| 2017 | [Shazeer et al.](https://arxiv.org/abs/1701.06538) | Sparse gating: keep only the top-k experts per example. Up to 131,072 experts and 137 billion parameters between LSTM layers |
| 2020 | [GShard](https://arxiv.org/abs/2006.16668) | MoE inside a Transformer, in every other MLP, with top-2 routing. A 600-billion-parameter translator trained on 2,048 TPU v3 cores in 4 days |
| 2021 | [Switch Transformer](https://arxiv.org/abs/2101.03961) | Top-1 routing, the simplest version. Models up to 1.6 trillion parameters with 2,048 experts |
| 2023 | [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) | An open-weight MoE (8 experts, top-2) that "outperforms or matches Llama 2 70B" on the benchmarks its authors ran |

Shazeer's 2017 paper (an LSTM model, one of the recurrent networks Part 1 described) claimed "greater than 1000x improvements in model capacity with only minor losses in computational efficiency." GShard's numbers show why people cared. Its 600-billion-parameter MoE cost 22 TPU v3 core-years to train and gave the best translations. Its best dense model, at 2.3 billion parameters, took 235.5 core-years and translated worse.

## Step 2: the router, by hand

The router is a single linear layer: one row of learned weights per expert. It takes the token's vector, computes a dot product with each row, and gets one score, or logit, per expert. Here is a toy layer with 8 experts and 4-number vectors. The token is `x = [1, 0, 2, −1]`.

| Expert | Router weights | Logit (dot product with x) | Softmax | Chosen? | Final weight |
|---|---|---|---|---|---|
| 1 | [0.2, 0.1, −0.3, 0.0] | −0.4 | 3.0% | | 0 |
| 2 | [0.5, −0.2, 0.4, 0.1] | 1.2 | 15.1% | | 0 |
| 3 | [−0.1, 0.3, 0.1, 0.6] | −0.5 | 2.8% | | 0 |
| 4 | [0.6, 0.0, 0.7, −0.2] | 2.2 | 41.1% | yes | 0.599 |
| 5 | [0.0, 0.4, −0.1, 0.3] | −0.5 | 2.8% | | 0 |
| 6 | [0.3, 0.2, 0.1, 0.5] | 0.0 | 4.6% | | 0 |
| 7 | [0.4, −0.3, 0.5, −0.4] | 1.8 | 27.6% | yes | 0.401 |
| 8 | [−0.2, 0.1, 0.0, 0.2] | −0.4 | 3.0% | | 0 |

Follow expert 4: `0.6×1 + 0.0×0 + 0.7×2 + (−0.2)×(−1) = 0.6 + 1.4 + 0.2 = 2.2`. Then four steps:

1. **Softmax.** Turn the 8 logits into probabilities, exactly as [Part 1](/blog/how-llms-work/) did for next-token logits. Expert 4 gets 41.1%, expert 7 gets 27.6%.
2. **Top-k.** Keep the 2 highest. It's the same move as top-k sampling in [Part 9](/blog/encoders-decoders-gpt/), but it picks experts instead of tokens, and there's no dice roll: the top 2 always win.
3. **Renormalize.** The two survivors hold 68.7% of the probability between them. Divide each by 0.687 so they add up to 1: `0.411 / 0.687 = 0.599` and `0.276 / 0.687 = 0.401`.
4. **Run and mix.** Send the token to experts 4 and 7 only. Say expert 4's MLP returns `[0.5, −1.0, 2.0, 0.0]` and expert 7's returns `[1.5, 0.5, 1.0, −0.5]` (made-up outputs; each is a full MLP like Part 6's). The layer's output is their weighted sum:

```text
0.599 × [0.5, −1.0, 2.0, 0.0] + 0.401 × [1.5, 0.5, 1.0, −0.5]
  = [0.901, −0.398, 1.599, −0.201]
```

That vector is added to the residual stream like any MLP output, and six of the eight experts never ran.

Steps 1 to 3 have a shortcut. Taking the softmax of just the two winning logits, 2.2 and 1.8, gives the same 0.599 and 0.401. That's how [Mixtral's paper](https://arxiv.org/abs/2401.04088) writes its router, `Softmax(Top2(x · Wg))`, and [gpt-oss](https://arxiv.org/abs/2508.10925) does the same over its top 4. The Switch Transformer keeps only the top 1 and doesn't renormalize: the token goes to expert 4 alone, and the output is expert 4's vector times 0.411. The router's weight has to scale the output somehow, because that's how the router gets gradients, the training signal from [Part 3](/blog/neural-networks-basics/), to learn from.

![A left-to-right flow. A token vector from attention goes to a router that gives one score per expert. A softmax, top-2 and renormalize step sends the token to expert 4 with weight 0.60 and expert 7 with weight 0.40, inside a group of 8 routed experts; a dotted line marks the other 6 experts as not chosen and skipped. Separately, the token vector also goes to a shared expert that runs for every token. The outputs of expert 4, expert 7 and the shared expert meet at a plus node, and the result is added to the residual stream.](/diagrams/f11-moe-layer.svg "One MoE layer for one token: the router picks 2 of 8 experts and weights them; a shared expert, as in DeepSeek's design, runs for every token.")

The router only decides which experts run and how much each one counts; the experts do the actual work.

## Step 3: total parameters versus active parameters

Mixtral 8x7B shows the accounting clearly. From its [published config](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json), each of its 32 layers has attention with about 41.9 million parameters and 8 experts with about 176.2 million each. Attention and the embedding tables are shared; only the experts multiply. Adding everything up:

| Count | Parameters | Mistral's figure |
|---|---|---|
| Total, all 8 experts in every layer | 46.7 billion | 46.7B |
| Active per token, 2 experts per layer | 12.9 billion | 12.9B |
| Experts' share of the total | 96.6% | |

Both totals match what Mistral [announced](https://mistral.ai/news/mixtral-of-experts): "46.7B total parameters but only uses 12.9B parameters per token."

### Compute follows the active count

A forward pass costs about 2 floating-point operations (FLOPs) per parameter that touches the token: one multiply and one add. So the arithmetic per token is roughly `2 × active parameters`, leaving out attention over the context, which grows with prompt length.

| Model | Active | FLOPs per token | A dense model of the same total size |
|---|---|---|---|
| Mixtral 8x7B | 12.9B | 25.8 billion | 93.4 billion, 3.6 times more |
| DeepSeek-V3 | 37B | 74 billion | 1.34 trillion, 18 times more |

Training follows the same rule. Part 1's `6 × N × D` estimate uses the active count for `N`. For DeepSeek-V3's 14.8 trillion training tokens that gives about 3.3 × 10²⁴ FLOPs, against 6.0 × 10²⁵ if all 671 billion parameters were active. Its full training took [2.788 million H800 GPU-hours](https://arxiv.org/abs/2412.19437).

### Memory follows the total

The router can pick any expert for the next token, so every expert has to sit in GPU memory, ready. DeepSeek-V3 ships its weights in FP8, 1 byte per parameter ([Inference Part 8](/blog/quantization/) covers these formats), so its 671 billion parameters need about 671 GB. Eight 80 GB H100s, a standard server, hold 640 GB. The weights alone don't fit, before any [KV cache](/blog/kv-cache/). Mixtral in 16-bit takes 93.4 GB, more than one H100.

The Mixtral paper says it plainly: active parameters are "directly proportional to the inference compute cost," while "the memory costs for serving Mixtral are proportional to its sparse parameter count, 47B."

### What the extra parameters buy

The usual summary is that an MoE gets the knowledge of a big model for the compute of a small one. The evidence:

- Mixtral "largely outperforms Llama 2 70B on all benchmarks, except on reading comprehension benchmarks while using 5x lower active parameters", in its authors' words.
- The Switch Transformer's 64-expert model, with the same FLOPs per token as the dense T5-Base, reached the same quality "in one-seventh the time".
- [DeepSeekMoE 16B](https://arxiv.org/abs/2401.06066) matched Llama 2 7B "with only about 40% of computations".

The same DeepSeekMoE paper also marks the ceiling. Its 2-billion-parameter MoE "nearly approaches" a dense model with the same total parameters, which the authors treat as "the upper bound of MoE models." So an MoE lands between a dense model of its active size and one of its total size, and DeepSeekMoE's small model came close to the top of that range.

## Step 4: keeping the experts busy

A router trained naively breaks. Shazeer et al. saw it in 2017: the router "tends to converge to a state where it always produces large weights for the same few experts. This imbalance is self-reinforcing, as the favored experts are trained more rapidly and thus are selected even more." This is routing collapse. A few experts do all the work, the rest waste memory, and when experts live on different GPUs, the busy ones become a traffic jam.

### A load-balancing loss

The Switch Transformer adds a small extra term, an auxiliary loss, to the training loss in each MoE layer:

```text
loss_balance = α × N × Σ f_i × P_i
```

`N` is the number of experts. `f_i` is the fraction of the batch's tokens sent to expert `i`, and `P_i` is the average router probability given to expert `i`. With perfectly even routing, every `f_i` and `P_i` is `1/N` and the sum times `N` comes to exactly 1. Take 4 experts where every token goes to expert 1, with average probabilities of 0.7, 0.1, 0.1 and 0.1: the value is `4 × (1 × 0.7) = 2.8`, nearly 3 times the balanced value. Training pushes it back down. The paper uses both terms because `f_i` is a count, which gradients can't flow through, while `P_i` is smooth. It found `α = 0.01` big enough to balance the load "while small enough to not to overwhelm" the main next-token loss.

### Capacity and dropped tokens

Hardware wants fixed-size work, so GShard and Switch give each expert a fixed number of slots per batch:

```text
expert capacity = (tokens in batch / number of experts) × capacity factor
```

With 1,024 tokens and 8 experts, a capacity factor of 1.0 gives each expert 128 slots, and 1.25 gives 160. If 200 tokens pick the same expert, 40 overflow. Those dropped tokens skip the expert entirely and pass to the next layer through the residual connection. Switch saw typically under 1% of tokens dropped, and found it worked best with low capacity factors of 1.0 to 1.25, because spare slots cost memory and compute.

### The router z-loss

Large MoE models are also prone to unstable training. The router adds extra exponentials, which magnify rounding errors. The 16-bit bfloat16 format has up to 65,536 times larger rounding errors than 32-bit floats, and the errors grow with the size of the number. [ST-MoE](https://arxiv.org/abs/2202.08906) added the router z-loss, which penalizes large router logits:

```text
z_loss = average over tokens of (log Σ_j e^(logit_j))²
```

For the toy token, `log Σ e^logit = 3.09`, so the penalty is 9.5. Multiply every logit by 10 and the ranking stays the same, but the penalty jumps to 485 and the router turns almost certain: 98.2% on expert 4 and 1.8% on expert 7. ST-MoE weights this loss by 0.001. In its tests, 4 of 6 baseline runs trained stably and all 3 runs with the z-loss did, with slightly better quality. The authors went on to train a 269-billion-parameter model, ST-MoE-32B.

### DeepSeek's balancing without a loss

Every auxiliary loss pulls against the main objective, and DeepSeek-V3 notes that "too large an auxiliary loss will impair the model performance." Its fix adds a bias `b_i` to each expert's score, used only to choose the experts. After each training step, an overloaded expert's bias drops by `γ = 0.001` and an underloaded one's rises by the same amount.

DeepSeek-V3 scores experts with a sigmoid, `1 / (1 + e^(−logit))`, instead of a softmax. For the toy token that gives 0.900 for expert 4, 0.858 for expert 7 and 0.769 for expert 2. If expert 4 has been overloaded for 150 steps, its bias is −0.15 and its adjusted score of 0.750 falls below expert 2's. The token goes to experts 7 and 2, weighted 0.528 and 0.472 from their original, unbiased scores.

The model's outputs never see the bias; only the choice of experts does. DeepSeek-V3 kept a tiny sequence-level balance loss (`α = 0.0001`) as a backstop, and it dropped no tokens in training or serving. In its ablations, the loss-free method beat a sequence-level balancing loss by a small margin: validation losses of 2.253 against 2.258 for a 1-billion-parameter MoE, and 2.080 against 2.085 at 3 billion.

### Fine-grained and shared experts

[DeepSeekMoE](https://arxiv.org/abs/2401.06066) changed the shape of the experts. With only 8 or 16 large experts, it argued, each one has to learn "vastly different types of knowledge", and several end up learning the same common knowledge. Its two fixes:

1. **Fine-grained experts.** Split each expert into several smaller ones and activate proportionally more, keeping compute the same. With 16 experts and top-2 there are 120 possible pairs. Split each into 4 and pick 8 of 64, and there are 4,426,165,368 combinations.
2. **Shared experts.** Set aside one or more experts that every token uses, to hold common knowledge so the routed experts don't each relearn it. When the authors switched off the shared expert and activated one more routed expert instead, at the same compute, the loss on the Pile test set rose from 1.808 to 2.414.

DeepSeek-V3 follows both ideas: 256 small routed experts of 44 million parameters each, 8 chosen per token, plus 1 shared expert. Its routed experts hold 97.4% of its parameters.

### Do experts specialize?

Less than the name suggests. Mixtral's authors measured which experts handled text from different sources and found that, "surprisingly, we do not observe obvious patterns in the assignment of experts based on the topic." Expert choices looked about the same for arXiv papers, biology abstracts and philosophy papers. What they did see was syntax. `self` in Python often goes through the same expert, indentation tokens in code "are always assigned to the same experts," and consecutive tokens often share an expert. At layer 15, a token's first-choice expert matched the previous token's 23.6% to 28.4% of the time, against 12.5% by chance.

ST-MoE found the same split by part of the model. Encoder experts specialized in "punctuation, verbs, proper names, counting," while decoder experts showed little "meaningful specialization (semantics or syntax)." DeepSeek's evidence is indirect. DeepSeekMoE's routed experts were less redundant, because switching off the top ones hurt more than in a GShard-style model. And DeepSeek-V3 reports that its loss-free balancing produced "greater expert specialization patterns" across domains than a loss-based model.

## In the real world

Counts come from each model's official config file or model card; the last column is active divided by total.

| Model | Year | Total parameters | Active | Routed experts | Chosen per token | Shared experts | Active share |
|---|---|---|---|---|---|---|---|
| Mixtral 8x7B | 2023 | 46.7B | 12.9B | 8 | 2 | 0 | 27.6% |
| DeepSeek-V3 | 2024 | 671B | 37B | 256 | 8 | 1 | 5.5% |
| Qwen3-235B-A22B | 2025 | 235B | 22B | 128 | 8 | 0 | 9.4% |
| Llama 4 Maverick | 2025 | 400B | 17B | 128 | 1 | 1 | 4.3% |
| gpt-oss-120b | 2025 | 116.8B | 5.1B | 128 | 4 | 0 | 4.4% |
| Kimi K2 | 2025 | 1T | 32B | 384 | 8 | 1 | 3.2% |
| DeepSeek-V4-Pro | 2026 | 1.6T | 49B | 384 | 6 | 1 | 3.1% |
| Kimi K3 | 2026 | 2.8T | 104B | 896 | 16 | 2 | 3.7% |

Sources: [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json), [Qwen3](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json), [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), [gpt-oss](https://arxiv.org/abs/2508.10925), [Kimi K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json), [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json), [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3). Recomputing from the configs gives 671.0B and 37.6B for DeepSeek-V3, 235.1B and 22.2B for Qwen3, and 1.03T and 32.9B for Kimi K2.

- The trend is toward more, smaller experts. Experts per layer went from 8 in Mixtral to 896 in Kimi K3, while the active share settled near 3% to 5%. Kimi K2's report found that at a fixed compute budget, raising the ratio of total to active experts kept lowering the loss. Activating 8 of 384 needed 1.69 times fewer FLOPs to reach the same loss as a ratio of 8. The report also notes that this "comes with increased infrastructure complexity."
- Llama 4 alternates dense and MoE layers, and each token visits the shared expert plus just 1 of 128 routed ones. Qwen3 dropped the shared experts its previous MoE had. DeepSeek-V3 keeps its first 3 layers dense. In Mixtral and gpt-oss, each expert is a SwiGLU MLP, the gated design covered in [Part 10](/blog/modern-llm-architecture/).
- Memory is the main serving cost. gpt-oss-120b fits on one 80 GB GPU only because its expert weights, "90+% of the total parameter count", are stored in a 4-bit format. Mixtral's authors note MoE layers suit "batched workloads where one can reach a good degree of arithmetic intensity."
- Small batches waste the sparsity. One user's token touches 8 of DeepSeek-V3's 256 routed experts per layer. If routing were uniform, a batch of 32 tokens would touch about 163 of them and a batch of 128 about 252, so the GPUs read nearly the whole model from memory for a handful of tokens per expert. Shazeer's 2017 paper called this the "shrinking batch problem."
- At scale, experts live on different GPUs, and every MoE layer sends each token to its experts' GPUs and back, an exchange called all-to-all communication. DeepSeek-V3's decoding setup spreads one expert per GPU over 320 GPUs. Inference Part 10 covers expert parallelism, and Inference Part 14 covers DeepSeek's serving stack, including its DeepEP communication library and EPLB load balancer.

## Common misconceptions

- **"Mixtral 8x7B has 56 billion parameters."** It has 46.7 billion. Only the MLPs are copied eight times; attention and embeddings are shared, so the total is well under 8 times a 7B model.
- **"Each expert is an expert in a subject."** Mixtral's authors found no clear topic pattern; routing tracked syntax and token position more than subject.
- **"An MoE is several models voting."** It's one model. The experts are MLPs inside each layer, attention is shared, and the choice is made again for every token at every layer.
- **"Only the active parameters need to be loaded."** All of them must be in memory, because the next token may need any expert. Active parameters set compute; total parameters set memory.
- **"A 671B MoE is as good as a 671B dense model."** DeepSeekMoE treats a dense model with the same total as the upper bound. An MoE trades a little quality per stored parameter for much less compute per token.
- **"MoE is always cheaper to run."** It's cheaper per token in arithmetic. It needs more memory, more GPUs and large batches to keep every expert busy, which suits a big serving cluster better than a laptop.

## Key takeaways

- An MoE layer swaps the block's single MLP for many expert MLPs and a router, a linear layer that scores every expert and sends each token to the top few.
- The output is a weighted sum of the chosen experts' outputs. Experts that aren't chosen cost nothing for that token.
- Compute per token follows the active parameters, about 2 FLOPs each, and memory follows the total. DeepSeek-V3 uses 37 billion of its 671 billion parameters per token, but all 671 billion must sit in memory.
- Left alone, routers collapse onto a few experts. Load-balancing losses, capacity limits, the z-loss and DeepSeek's bias adjustments keep the load even and training stable.
- Modern MoEs use hundreds of small experts, sometimes with a shared expert that every token uses, and activate about 3% to 5% of their weights per token.

## What's next

Every model in this part learned its weights from trillions of tokens of text. Part 12 covers pretraining: where that text comes from and how it's filtered, the scaling laws that say how big a model should be for a given budget, and how training runs spread across thousands of GPUs.

## Further reading

- [Adaptive Mixtures of Local Experts (Jacobs, Jordan, Nowlan & Hinton, 1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf): the original mixture of experts, with a gating network that assigns each case to an expert.
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538): noisy top-k gating, the shrinking batch problem and the first balancing losses.
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668): top-2 MoE inside a Transformer, expert capacity and a 600B translation model.
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus, Zoph & Shazeer, 2021)](https://arxiv.org/abs/2101.03961): top-1 routing, the load-balancing loss and capacity factor.
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models (Zoph et al., 2022)](https://arxiv.org/abs/2202.08906): the router z-loss and an analysis of what experts specialize in.
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088): an open 8-expert model and its routing analysis.
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (Dai et al., 2024)](https://arxiv.org/abs/2401.06066): fine-grained and shared experts.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): auxiliary-loss-free balancing, node-limited routing and training a 671B MoE.
- [Kimi K2: Open Agentic Intelligence (Kimi Team, 2025)](https://arxiv.org/abs/2507.20534): the sparsity scaling law behind 384 experts.
- [gpt-oss-120b & gpt-oss-20b Model Card (OpenAI, 2025)](https://arxiv.org/abs/2508.10925): a parameter breakdown by component and 4-bit expert weights.
- [The Llama 4 herd (Meta, 2025)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/): alternating dense and MoE layers with a shared expert.
- [Mixture of Experts Explained (Hugging Face, 2023)](https://huggingface.co/blog/moe): a readable tour of the history, training tricks and serving issues.
- [A Visual Guide to Mixture of Experts (Maarten Grootendorst, 2024)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts): routers, load balancing and capacity, drawn step by step.
