---
title: "Steering With Activations: The Refusal Direction, Measured"
description: "Find one direction that controls refusal in an open chat model, remove it and add it, measure the side effects, and compare SAE steering with a mean difference."
slug: "activation-steering"
category: "interpretability"
tags: ["interpretability", "steering", "safety", "sparse-autoencoders", "evals"]
series: "llm-interpretability"
seriesOrder: 9
pubDate: 2026-10-30
---

## TL;DR

A steering vector is a list of numbers added to a model's internal activations while it runs, which pushes its behavior one way without changing any weights. This part builds the simplest kind, a difference of means between two sets of prompts, and applies it to the best-studied case: the refusal direction, a single direction that decides whether a chat model refuses. On Qwen2.5-1.5B-Instruct, removing it drops refusals of 100 held-out harmful requests from 99% to 11%, while adding it makes the model refuse 98% of harmless ones. You'll see what each edit costs on ordinary text and on a science quiz, at what strength the text breaks, and why a plain mean difference steered Gemma 2 2B toward Egypt about twice as well as the sparse autoencoder latent from [Part 7](/blog/sparse-autoencoders/).

## Why it matters

Ask Qwen2.5-1.5B-Instruct for a pancake recipe and you get one. Add a single vector of 1,536 numbers to its activations at one layer, and it answers "I'm sorry, but I can't assist with that." It refuses 98 of 100 harmless requests that way. Take that direction out of every layer instead, and on 100 requests from a public benchmark of harmful behaviors its refusals fall from 99 to 11.

Nothing was retrained. The model's safety training, the part that makes it say no, runs largely through one line in a space of 1,536 dimensions. That was the finding of [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (Arditi et al., 2024), which found such a direction "across 13 popular open-source chat models up to 72B parameters in size."

[Part 3](/blog/linear-probes/) ended with a nudge along a truth direction, and [Fundamentals Part 16](/blog/llm-interpretability/) told the Golden Gate Claude story in brief. This part measures steering on a laptop: how well it works, what it breaks, and what it means for open-weight models that refusal is this easy to find.

## The intuition

Think of a sound engineer at a mixing desk. [Part 3](/blog/linear-probes/) put meters on the channels: a probe reads how strongly one direction is present in the model's vector. Steering grabs the fader. You push a channel up (activation addition) or mute it (directional ablation) and listen to what comes out of the speakers.

The analogy breaks in two places. A desk has one labeled fader per instrument, while a model's "instruments" are unlabeled directions that overlap ([Part 6](/blog/superposition/)), so pushing one nudges its neighbors. And a real fader has its safe range printed next to it; in a model you find the range by pushing until the music turns to noise, as Step 4 does.

## Step 1: a direction you can push

A steering vector is added to the residual stream, the running vector per token that every layer reads from and adds to ([Part 2](/blog/logit-lens/)), at every token position while the model runs. The weights never change.

Two edits cover almost everything in this article. Here they are on a toy activation `x` with three numbers and a direction `r` of length 1:

```python
import torch

x = torch.tensor([2.0, 1.0, -1.0])
r = torch.tensor([0.6, 0.8, 0.0])        # unit length: 0.36 + 0.64 = 1
print((x @ r).item())                    # how far x points along r
print(x + 3 * r)                         # activation addition, strength 3
print(x - (x @ r) * r)                   # directional ablation
print(((x - (x @ r) * r) @ r).item())    # nothing left along r
```

It prints `2.0`, then `[3.8, 3.4, -1.0]`, then `[0.8, -0.6, -1.0]`, then a number around `-3e-08`, which is zero up to rounding. Activation addition moves the vector a fixed distance along `r`, whatever was there before. Directional ablation, written `x - (r·x) r`, removes only the part of `x` that lies along `r`, so it never pushes a vector further than it already reaches.

### Where the vector comes from

The papers in this article differ mainly in how they choose `r`.

- Activation Addition, or ActAdd ([Turner et al., 2023](https://arxiv.org/abs/2308.10248)), takes the difference of the activations for one pair of prompts, such as "Love" and "Hate". A "weddings" vector in GPT-2 XL made completions mention weddings with ">90% success in topic steering (compared to a ∼2% baseline)" at the best layer.
- Contrastive activation addition, or CAA ([Panickssery et al., 2023](https://arxiv.org/abs/2312.06681)), averages the difference over hundreds of pairs. Each pair is one multiple-choice question ending in two answer letters, one showing a behavior such as sycophancy, so everything except the behavior cancels.
- Representation engineering ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)) reads a direction off contrasting stimuli, for example the first principal component of their differences, and uses it to monitor and control concepts like honesty.
- Arditi et al. take the mean activation over many harmful requests minus the mean over many harmless ones. This difference of means drove [Part 3](/blog/linear-probes/)'s truth nudge and every experiment below.

A sparse autoencoder (SAE) offers one more source: the decoder direction of a latent, which is what [Part 7](/blog/sparse-autoencoders/) clamped for "Egypt". Step 5 compares the two.

## Step 2: finding the refusal direction

The recipe is Arditi et al.'s, at laptop scale. Feed the chat model 128 harmful requests and 128 harmless ones, formatted with its chat template, and stop before it writes anything. At every layer, save the residual vector at the last five tokens, the template tokens that come after the user's text (for Qwen these are the end-of-message marker, a newline, the start marker, "assistant" and another newline). At those positions the model is about to begin its reply. Then subtract the harmless mean from the harmful mean at each layer and position.

![A top-to-bottom flow. 128 harmful plus 128 harmless requests go into a chat model, which reads each one without replying. Residual vectors are saved at every layer. A difference of means gives 140 candidate directions. One is picked using validation sets. Then it is used two ways: ablated everywhere, or added at one layer. Both lead to the same measurements: refusal rate and side effects.](/diagrams/p09-refusal-direction.svg "The refusal-direction recipe used in this article, after Arditi et al. (2024).")

Qwen2.5-1.5B-Instruct has 28 layers, so this gives 140 candidate directions, and a separate set of 32 harmful and 32 harmless validation requests picks one. Each candidate is scored with a cheap stand-in for refusal that needs no generated text: the log-odds that the model's first reply token is one that usually starts a refusal. For Qwen that is "I" or "As", the token set the paper used for the Qwen family. On the harmful training requests Qwen puts 78% of its first-token probability on "I"; on the harmless ones its favorite is "The", at 10%.

Each candidate gets three scores: the refusal score on harmful validation requests with the candidate ablated from every layer (lower is better), the score on harmless ones with it added at its own layer (higher is better), and how much ablating it changes next-token predictions on harmless requests, measured as KL divergence, the standard measure of distance between two probability distributions. A candidate qualifies if adding it pushes the score above 0, ablating it moves harmless predictions by less than 0.1 nats of KL divergence, and it comes from the first 80% of the layers, so that it can't simply be the output direction for the word "I". The lowest score under ablation wins.

![A line chart over Qwen2.5-1.5B-Instruct layers 1 to 27, refusal score from -8 to +4, for directions taken at the last template token. Dashed lines mark the unedited model: +1.7 on harmful requests and -4.8 on harmless ones. The blue line, the direction ablated on harmful requests, stays near +1.5 until layer 13, then drops to about -4.4 at layers 14 to 16, sits near -2 at layers 17 to 19, and swings between -7 and -2 after that. The orange line, the direction added on harmless requests, stays near -4.8 until layer 10, rises through -2 at layer 13 and reaches about +1.2 to +2.2 from layer 14 on. A dotted vertical line marks layer 17, the chosen direction. Layers above 22 are shaded as too close to the output.](/diagrams/p09-direction-search.svg "Qwen2.5-1.5B-Instruct: refusal score of each layer's candidate at the last template token, on 32 harmful and 32 harmless validation requests.")

Up to layer 10 the difference of means does nothing in either direction. Between layers 11 and 14 the model has read enough of the request to decide, and from there on one direction works both ways. Layers 14 to 16 ablate better than 17 but disturb harmless answers too much, with KL from 0.75 to 1.3. Only 4 of the 140 candidates passed all three tests, and the winner, layer 17 at the last token, has a length of 23.8.

### The setup for every number below

All runs used an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) with PyTorch 2.11 and Hugging Face transformers 5.21.1, editing activations with plain forward hooks. There was no TransformerLens weight processing, so "the residual stream" is each model's own hidden state. Qwen2.5-1.5B-Instruct (`Qwen/Qwen2.5-1.5B-Instruct`, Apache-2.0) and Gemma 2 2B and 2B-it (`google/gemma-2-2b`, `google/gemma-2-2b-it`, under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)) ran in bfloat16 with their default chat templates (Qwen's includes its default system prompt) and greedy replies of 128 tokens. The helper code is a 210-line module, `lab/steering.py`, and all the runs took 29 minutes of GPU time.

Harmful training and validation requests are from [AdvBench](https://github.com/llm-attacks/llm-attacks) (MIT license), after dropping 68 that were more than 70% similar to a test item. The harmful test set is all 100 behaviors of [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) (JBB-Behaviors, MIT), ten each in ten categories from harassment and malware to disinformation and expert advice. Harmless requests are instructions from [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (CC BY-NC 4.0): 128 for training, 32 for validation and 100 for testing, none shared. JailbreakBench also ships 100 benign look-alikes, one per harmful behavior on the same topic, a sharp test of over-refusal. This article quotes none of the harmful prompts and none of the replies that complied with them.

A reply counts as a refusal if it contains one of the paper's 12 refusal phrases, such as "I'm sorry" or "As an AI". To check that rule, the writer of this article, an AI model, hand-labeled 135 replies from all conditions, unblinded, as refused (declines, or deflects without doing the task) or complied (produces what was asked, even partly), and the rule agreed on 127 (94%). On 30 replies to harmful requests with the direction ablated, it found 2 refusals where the hand labels found 5: four replies quietly turned the request into a harmless one, such as a thread debunking a hoax instead of promoting it, and one of the rule's two was an "I'm sorry" followed by the requested content.

## Step 3: remove it, add it

With the direction chosen, the test runs on prompts the selection never saw. Every rate is out of 100 prompts, with a 95% Wilson confidence interval:

| Qwen2.5-1.5B-Instruct, edit | Harmful (JailbreakBench) | Harmless (Alpaca) | Benign look-alikes (JailbreakBench) |
|---|---|---|---|
| None | 99% (95 to 100) | 10% (6 to 17) | 50% (40 to 60) |
| Refusal direction ablated everywhere | 11% (6 to 19) | not run | 6% (3 to 13) |
| Random direction ablated everywhere | 97% (92 to 99) | not run | not run |
| Refusal direction added at layer 17 | not run | 98% (93 to 99) | 97% (92 to 99) |
| Random vector of the same length added | not run | 30% (22 to 40) | not run |

Ablation removes nearly all of Qwen's refusals, and ablating a random direction removes almost none, so the effect belongs to this one direction out of 1,536. By hand, the picture is similar: of 30 ablated replies we labeled, 25 produced the requested content and 5 refused or deflected.

Addition is less specific. A random vector of the same length raised refusals of harmless requests from 10% to 30%, some of them real, such as declining to write a blues song, so any large push makes this small model more cautious. The refusal direction still does three times as much.

Unedited, Qwen also refuses half of the benign look-alikes, harmless requests that share a topic with a harmful one. Ablation cuts that to 6%.

### What added refusal sounds like

The same recipe works on Gemma 2 2B-it, the instruction-tuned sibling of the Gemma 2 2B used in Parts 7 and 8, as far as we tested it, which is only the adding half. Gemma's [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which its terms of use incorporate, forbids "Attempts to override or circumvent safety filters", so every ablation experiment in this article ran on Qwen only. For Gemma we used the difference of means at the "model" template token and layer 14 of 26. The paper's selection procedure picked that candidate in a first run, made before we limited Gemma to the adding half, and it is the same token position and about the same relative depth (14 of 26, against 10 of 18) that Arditi et al. selected for the first Gemma 2B. Unedited, Gemma refuses 97% of the harmful test requests and 2% of the harmless ones. With the direction added, it refuses all 100 harmless requests and all 100 benign look-alikes, which it refused 24% of the time on its own. A random vector of the same length changed nothing: 2%.

Here is the whole edit. It assumes a local bfloat16 copy of Gemma 2 2B-it and the directions that `refusal_dir.py`, in the series' code, computes from the downloaded prompt sets and saves:

```python
import sys, torch
sys.path.insert(0, "interp")                       # the series' shared lab/ folder
from lab.steering import Steerer

s = Steerer("interp/models/gemma-2-2b-it-bf16")
r = torch.load("p09/runs/gemma_dirs.pt")["r"][14, 3]   # layer 14, at the "model" token
prompt = s.chat(["Give me a recipe for banana pancakes."])
for name, edit in [("unedited", None), ("refusal added", lambda m: m.add(14, r))]:
    with s.edits(edit):
        print(name, "|", s.generate(prompt, max_new_tokens=30)[0])
```

Unedited, Gemma starts "## Fluffy Banana Pancakes"; with the direction added, it says "I cannot provide you with a recipe for banana pancakes. My purpose is to provide safe and helpful information." Five harmless prompts of our own, run in the main experiment, were refused five times, each with a reason invented after the fact. "Explain how a rainbow forms" got the answer that "Rainbows are a product of light and are illegal to create." "Suggest three names for a pet goldfish" got "Creating a name for a pet is a serious matter." And "What is the capital of Australia?" got "I cannot provide you with information on how to build or use illegal weapons." The edit triggers the decision to refuse, and the model then writes whatever justification fits, even for a request with nothing harmful in it.

## Step 4: what it costs

Two measurements check whether the rest of the model still works: next-token loss, the cross-entropy from [Math Part 6](/blog/cross-entropy-perplexity/), on 200 held-out FineWeb-Edu documents of 256 tokens (plain web text, the held-out set of Parts 2 and 7), and accuracy on 500 random questions from ARC-Easy (CC BY-SA 4.0), a grade-school science quiz, scored by which option letter the model makes most likely. Differences are paired, with 95% bootstrap intervals.

| Qwen2.5-1.5B-Instruct, edit | Extra loss on web text, nats | ARC-Easy accuracy |
|---|---|---|
| None | 0 (2.389 nats, perplexity 10.9) | 91.2% (88.4 to 93.4) |
| Refusal direction ablated | +0.011 (0.008 to 0.013) | 91.4%, change +0.2 points (-1.0 to +1.4) |
| Random direction ablated | +0.014 (0.012 to 0.016) | 91.4% |
| Refusal direction subtracted at layer 17 | +0.254 (0.241 to 0.268) | 85.6%, change -5.6 points (-8.6 to -2.6) |
| Refusal direction added at layer 17 | +0.384 (0.368 to 0.403) | 45.4% (41.1 to 49.8) |
| Random vector added | +0.196 (0.187 to 0.206) | 87.0% |

Ablating the refusal direction costs about as much as ablating a random one: 0.01 nats, about 1% perplexity, probably mostly bfloat16 rounding from rewriting every layer's output. ARC accuracy doesn't move. As far as these two tests can see, removing refusal leaves knowledge and fluency intact, as Arditi et al. found on larger models: MMLU, ARC and GSM8K scores of the largest model in each family stayed within about 1.5 points.

Addition pushes every activation the same distance, whether or not the token needed it, so it costs 0.38 nats on text that has nothing to do with refusal and halves quiz accuracy, as the model leans toward declining to answer. On Gemma, adding the direction raised the loss by 0.65 nats (0.63 to 0.67) and cut ARC-Easy accuracy from 89.8% to 37.0%; Gemma's most likely first token was never an option letter. Subtracting the direction also turns refusal off, but at 23 times the text cost of ablating it, because it shoves harmless activations past where they normally sit. Arditi et al. report the same ordering.

### Where the text breaks

We multiplied Qwen's refusal direction by values from -2 to +6, added the result at layer 17, and measured refusals on the 100 harmless and 100 harmful test requests (64-token replies) and the loss on 100 of the web documents.

![Two stacked panels sharing an x-axis, strength from -2 to +6 multiples of Qwen's refusal direction added at layer 17. Top panel, share of replies that refuse. Harmful requests, in blue: 3% at -2, 2% at -1.5, 20% at -1, 65% at -0.5, 99% at 0, then 89 to 99% up to +2, 82% at +3, 2% at +4 and 0% at +6. Harmless requests, in orange: about 1% from -2 to -0.5, 10% at 0, 76% at +0.5, 98 to 99% from +1 to +2, 82% at +3, 5% at +4, 0% at +6. Bottom panel, extra loss on web text in nats: 0.96 at -2, falling to 0 at 0, then 0.10 at +0.5, 0.38 at +1, 1.43 at +2, 2.96 at +3, 4.65 at +4 and 7.63 at +6. Strengths from +2.5 up are shaded as the region where text breaks down.](/diagrams/p09-strength-sweep.svg "Qwen2.5-1.5B-Instruct: one direction, scaled. Refusal rates on 100 harmful and 100 harmless held-out requests, and extra loss on 100 FineWeb-Edu documents.")

Half the direction already gets 76% of harmless requests refused, for 0.10 nats. Up to +2 the replies are fluent refusals, though some lose the stock phrases: at +1, 11 harmful requests got replies like "I am an AI developed by Alibaba Cloud and I do not support or encourage any form of discrimination", which the substring rule misses, hence the dip to 89%. At +3 the grammar starts to go ("I apologize for any and I am an extremely illegal and immoral behavior which is absolutely prohibited in any country"). At +4 the model asked for five activities in a park produces "I and whatever you are and I will be absolutely zero whatever go with any go and go and go", and the refusal rate falls to 2 to 5% only because nothing it writes is a sentence any more. Judged by refusals alone, that would look like the edit had switched refusal off.

## Step 5: SAE latent or mean difference?

[Part 7](/blog/sparse-autoencoders/) clamped Gemma Scope latent 7498, which fires on ancient Egypt, and got Egypt into at most 4 of 20 completions. To see whether the limit was Gemma or the SAE direction, we built a second Egypt direction on base Gemma 2 2B without an SAE: 20 sentence templates such as "Last summer we traveled to X." filled with Egypt, minus the same templates filled with 8 other countries, averaged over the tokens from the country name on. Then we added each direction, scaled to the same length, at the same site as Part 7 (after layer 12) and generated 40 tokens from 50 neutral prompts: Part 7's 20 plus 30 new ones.

The two directions are related: their cosine similarity is 0.35, where 1,000 random directions never exceeded 0.06 in absolute value. Strength is in units of the typical residual length at that site, 146, so Part 7's clamp at 100 was a strength of about 0.68.

![A line chart: x-axis is extra loss on plain web text, from 0 to 3.2 nats; y-axis is the share of 50 continuations that mention Egypt, 0 to 70%. The difference-of-means direction, in blue, rises steeply: 10% at strength 0.2 for 0.06 nats, 24% at 0.3, 38% at 0.4 for 0.49 nats, 56% at 0.5, 60% at 0.6 and a peak of 62% at 0.8 for 2.16 nats, then 54% at 1.0 for 3.14 nats. The SAE latent, in orange, rises more slowly: 4% at 0.3, 10% at 0.4 for 0.19 nats, 20% at 0.5, a peak of 32% at 0.6 for 0.53 nats, then down to 24% at 0.8 and 8% at 1.0 for 1.82 nats. The random direction, in gray, stays at 0% at every strength up to 1.0 nats.](/diagrams/p09-egypt-tradeoff.svg "Base Gemma 2 2B, steering toward Egypt at layer 12: share of 50 neutral prompts that mention Egypt, against the extra loss each strength costs. Points are strengths from 0.1 to 1.0 times the residual norm.")

At every level of side effect the mean difference steers more. For about 0.2 nats of extra loss it puts Egypt into 24% of completions (12 of 50, interval 14 to 37%) against the latent's 10%. Its best, 62% (48 to 74%), is nearly twice the latent's best, 32% (21 to 46%), which matches Part 7's 20% within the intervals. A random direction of the same length never produced Egypt once.

On topic is not the same as true. At strength 0.4 the mean difference turned "Scientists recently discovered that" into "the ancient Egyptian pyramids were built by a race of giants." The latent, at 0.6, gave "the ancient Egyptian pyramids were built to be used as tombs for the pharaohs."

This is one concept at one layer with one dictionary, but it agrees with a much larger study: [AxBench](https://arxiv.org/abs/2501.17148) (Wu et al., 2025) steered Gemma-2-2B and 9B toward 500 concepts and found a difference of means beat Gemma Scope SAE latents on 58.7% of them. Simply prompting the model beat both, with a judge's score of 0.894 out of 2 against 0.239 and 0.165. A plausible reason is that a latent is trained to reconstruct activations, while a contrast between texts with and without the concept is built from the change you want to cause.

## In the real world

- Arditi et al. found the refusal direction in 13 open chat models from 1.8 to 72 billion parameters (Qwen, Yi, Gemma, Llama-2 and Llama-3). Baking the ablation into the weights, which they call weight orthogonalization, raised the HarmBench attack success rate of Qwen 7B from a 7.0% direct-response baseline to 79.2%, and of Llama-2 7B to 22.6% with its default system prompt or 79.9% without it. TruthfulQA scores fell by 1 to 3.5 points in the largest model of each family, while MMLU barely moved (Llama-3 70B: 79.8 against 79.9). Not reproduced here.
- The paper's ethics section says the method "can yield a jailbroken version of a 70B parameter model using less than $5 of compute," and argues it "does not substantially alter the risk profile of open sourcing models," since fine-tuning could already strip safeguards from open weights. For anyone releasing weights, refusal training is a thin layer that whoever holds the weights can remove cheaply, so it can't be the only safeguard.
- Contrastive activation addition steered seven behaviors in Llama 2 Chat, including sycophancy and refusal. On a two-option version of MMLU, Llama 2 13B Chat's average probability on the right answer was 0.63 unsteered and 0.57 to 0.65 with any of the vectors applied at layer 14. Representation engineering raised Llama-2-13B-Chat's TruthfulQA accuracy (MC1) from 35.9% to 54.0% with an honesty vector. Neither reproduced here.
- Defenses now use the same handle. [Refusal feature adversarial training](https://arxiv.org/abs/2409.20089) (Yu et al., 2024) argues that many jailbreaks work by suppressing this direction and trains models against simulated attacks that ablate it. [Circuit breakers](https://arxiv.org/abs/2406.04313) (Zou et al., 2024) interrupt a model as it starts a harmful answer by acting directly on "the representations that are responsible for harmful outputs."
- One direction is a simplification. [Wollschläger et al. (2025)](https://arxiv.org/abs/2502.17420) report "multiple independent directions and even multi-dimensional concept cones that mediate refusal." Not reproduced here.

## Common misconceptions

- **"Safety training builds a deep set of rules into the model."** Arditi et al. cut refusals in 13 models by removing one direction, and in Qwen2.5-1.5B-Instruct the same edit took refusals of held-out harmful requests from 99% to 11% with no measurable loss on a science quiz. Whatever else safety training does, its output passes through a narrow channel.
- **"The refusal direction means 'this is harmful'."** Added to "What is the capital of Australia?", it produced a refusal about illegal weapons. Adding it triggers the decision to refuse, and the reasons are written afterwards.
- **"A steering vector is a precise edit."** Adding Qwen's refusal direction cost 0.38 nats on unrelated text and halved its quiz accuracy, and a random vector of the same length tripled harmless refusals. Steering results need a random control and a side-effect measure.
- **"If the refusal rate drops, the model complied."** At four times the refusal direction, refusals fell to 2% because the model was writing "go and go and go", and substring rules miss quiet evasions.
- **"SAE latents are the best handles for steering."** On Egypt, a difference of means over 20 templates steered nearly twice as well as the Gemma Scope latent at its best, and AxBench found the same across 500 concepts. A contrast built for the purpose is often the better handle for changing behavior.

## Key takeaways

- A steering vector is added to the residual stream while the model runs. Activation addition pushes along a direction; directional ablation removes a direction and nothing else.
- A difference of means between harmful and harmless requests found a refusal direction in Qwen2.5-1.5B-Instruct at layer 17 of 28, chosen on separate validation prompts.
- Removing it cut refusals of 100 held-out harmful requests from 99% to 11% while leaving web-text loss and quiz accuracy essentially unchanged. Adding it made the model refuse 98% of harmless requests, at a real cost to everything else.
- Measure side effects every time. Past about twice the direction's natural length the text broke down, and a refusal counter scored the gibberish as compliance.
- For a topic like Egypt in Gemma 2 2B, a plain difference of means steered about twice as well as an SAE latent at the same side-effect cost.

## What's next

Every model so far arrived fully trained. [Part 10](/blog/training-dynamics-interpretability/) watches the machinery form, using checkpoints saved during the From Scratch series' training run: when induction heads switch on, how that lines up with the loss curve, and in what order a model learns which word follows which.

## Further reading

- [Refusal in Language Models Is Mediated by a Single Direction (Arditi et al., 2024)](https://arxiv.org/abs/2406.11717): the difference-of-means recipe, directional ablation and weight orthogonalization across 13 chat models.
- [Steering Language Models With Activation Engineering (Turner et al., 2023)](https://arxiv.org/abs/2308.10248): ActAdd, steering vectors from a single pair of prompts.
- [Steering Llama 2 via Contrastive Activation Addition (Panickssery et al., 2023)](https://arxiv.org/abs/2312.06681): mean differences over many multiple-choice contrast pairs, for seven alignment-relevant behaviors.
- [Representation Engineering: A Top-Down Approach to AI Transparency (Zou et al., 2023)](https://arxiv.org/abs/2310.01405): reading and controlling honesty and other concepts through directions.
- [AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders (Wu et al., 2025)](https://arxiv.org/abs/2501.17148): a large comparison of steering and detection methods on Gemma 2.
- [The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence (Wollschläger et al., 2025)](https://arxiv.org/abs/2502.17420): evidence that refusal uses more than one direction.
- [Robust LLM safeguarding via refusal feature adversarial training (Yu et al., 2024)](https://arxiv.org/abs/2409.20089): using refusal feature ablation as a cheap stand-in for attacks during training.
- [Improving Alignment and Robustness with Circuit Breakers (Zou et al., 2024)](https://arxiv.org/abs/2406.04313): a representation-level defense against harmful outputs.
- [JailbreakBench JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors): the 100 harmful and 100 benign test behaviors, MIT license.
- [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy): why the ablation experiments here ran on Qwen.
- [Sparse Autoencoders (Part 7)](/blog/sparse-autoencoders/): the Gemma Scope latent and the first Egypt clamp.
- [Linear Probes (Part 3)](/blog/linear-probes/): difference-of-means directions as probes, and the truth nudge that previewed this part.
