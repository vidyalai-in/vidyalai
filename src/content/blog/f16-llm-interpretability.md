---
title: "Inside the Black Box: Features, Superposition and Circuits"
description: "How researchers read an LLM's internals: the logit lens, superposition, sparse autoencoders, circuits and steering, with real numbers and honest limits."
slug: "llm-interpretability"
category: "llm-fundamentals"
tags: ["interpretability", "alignment", "safety", "transformers", "embeddings"]
series: "llm-fundamentals"
seriesOrder: 16
pubDate: 2026-10-23
---

## TL;DR

Interpretability research reads a trained model's internal activations to work out what it computes; its most detailed branch, mechanistic interpretability, tries to reverse-engineer that computation step by step. The simplest tool, the logit lens, decodes the model's half-finished vector at each layer into words. Going deeper runs into superposition: a model stores far more concepts than it has dimensions, as directions that almost never fire at the same time, so single neurons look like a jumble. Sparse autoencoders untangle those directions into millions of readable features, and circuit tracing links features into step-by-step explanations. This article works through each tool with real numbers, then shows where labs use them and where they fall short.

## Why it matters

In May 2024, for 24 hours, anyone could chat with a version of Claude that thought it was the Golden Gate Bridge. Asked how to spend $10, it suggested driving across the bridge and paying the toll. Anthropic hadn't changed the prompt or retrained the model. Its researchers had found one direction inside Claude 3 Sonnet that lights up on [mentions and pictures of the bridge](https://www.anthropic.com/news/golden-gate-claude), and [clamped it to 10 times its maximum value](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).

So a model with billions of weights can hold a concept somewhere a person can find it, and changing it changes behavior predictably. Finding it was the hard part. The bridge was one entry in a dictionary of 34 million candidate directions that the team trained for the purpose.

[Part 1](/blog/how-llms-work/) called the model a stack of blocks that turns tokens into a next-token guess. Fifteen parts later you know every matrix in that stack. This part asks what those matrices compute, and how anyone could find out.

## The intuition

Picture a recording of a crowded party made with only a few microphones. Dozens of conversations are mixed on each track, so listening to one microphone gives you fragments of everything. Untangling it works only because most people are silent at any given moment.

The model's vector is the recording, each number in it is a microphone, and the concepts are the voices. Interpretability is the work of separating the voices, then figuring out who is talking to whom.

The analogy breaks in two places. At a party you know the voices are people. In a model nobody knows what the true units are, or how many there are, so every method makes a guess about that. And hearing a voice isn't proof that it mattered. Interpretability has to change the voice and watch whether the output changes too.

## Step 1: reading the residual stream with the logit lens

[Part 6](/blog/transformer-block/) introduced the residual stream: one vector per token that every attention layer and MLP reads from and adds to. [Part 4](/blog/embeddings-and-logits/) showed how the vector at the top becomes logits, a score per vocabulary token, through the unembedding. Since every block writes into the same space, there's an obvious experiment: apply the unembedding early, to the vector after block 3 or block 9, and see what the model would say if it stopped there.

That's the logit lens, named in a [2020 post by nostalgebraist](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) that tried it on the largest GPT-2. The post found that the decoded guesses "gradually converge to the final distribution over the layers of the network."

We ran it on GPT-2 small (12 blocks, 768 numbers per vector) on CPU, with the prompt "The capital of France is". At each depth we applied GPT-2's final layer norm and its unembedding to the last token's vector:

| Read after | Top guess (probability) | Rank of " Paris" | P(" Paris") |
|---|---|---|---|
| Embedding only | " destro" (53%) | 21,416 | ~0% |
| Block 3 | " now" (37%) | 13,624 | ~0% |
| Block 5 | " now" (63%) | 3,106 | ~0% |
| Block 7 | " now" (57%) | 200 | 0.01% |
| Block 9 | " now" (42%) | 52 | 0.06% |
| Block 10 | " France" (62%) | 2 | 18.2% |
| Block 11 | " France" (24%) | 2 | 13.9% |
| Block 12 (the real output) | " the" (8.5%) | 5 | 3.2% |

For nine blocks the model's best guess is grammatical filler: "The capital of France is now...". The fact shows up late and suddenly, with " Paris" jumping from rank 52 to rank 2 in block 10. Then the last block pushes it back down to the 3.2% that [Part 4's table](/blog/embeddings-and-logits/) reported, as " the" takes over. On "The Eiffel Tower is located in the city of", the same run puts " Paris" at rank 13 after block 9 and at rank 1 from block 11 on.

The logit lens is brittle on some models, because early vectors aren't meant to be read by the last layer. The [tuned lens](https://arxiv.org/abs/2303.08112) (Belrose et al., 2023) fixes that by training a small affine translator for each block, and found it "more predictive, reliable and unbiased than the logit lens" on models up to 20 billion parameters.

### Probes: asking a yes-or-no question of a vector

The logit lens can only read out vocabulary words. To look for anything else, train a probe: a small classifier, usually linear, that reads a layer's vectors and predicts a property such as "this sentence is true" or "this city is in Europe". Since linear probes were [proposed in 2016](https://arxiv.org/abs/1610.01644), they've shown that Llama 2 encodes [a place's coordinates and an event's date](https://arxiv.org/abs/2310.02207) along directions, and that models represent [whether a statement is true or false](https://arxiv.org/abs/2310.06824) the same way.

A probe shows that information can be decoded, not that the model uses it ([Belinkov's review](https://arxiv.org/abs/2102.12452) covers this at length), and it only finds what you thought to look for. To find concepts nobody thought to ask about, you have to take the vector apart.

## Step 2: superposition, or more concepts than dimensions

The obvious place to start is the neurons, the individual numbers inside each MLP, but they rarely mean one thing. In a small language model Anthropic studied, a single neuron [responds to "a mixture of academic citations, English dialogue, HTTP requests, and Korean text"](https://transformer-circuits.pub/2023/monosemantic-features/index.html). Such neurons are called polysemantic, and they're the norm.

[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Elhage et al., 2022) explained why. A feature here is any property of the input the model represents, like "this text is in French". The authors trained tiny networks to squeeze 5 features through 2 dimensions and rebuild them. When every feature was active most of the time, the network kept the two most important ones and dropped the other three. When the five were equally important and each was active only 5% of the time, it stored all five, as the corners of a regular pentagon.

That's superposition: the model represents more features than it has dimensions, each as its own direction, and accepts a little crosstalk. The paper notes that while an n-dimensional space has only n exactly perpendicular directions, it has room for exponentially many almost-perpendicular ones.

### How much room is there?

Cosine similarity, from Part 4, measures the angle between two directions: 0 means perpendicular, 1 means identical. We generated 2,000 random directions in spaces of different sizes and measured every pair:

| Dimensions | Typical size of the cosine | Worst pair of 1,999,000 |
|---|---|---|
| 3 | 0.50 | 1.00 |
| 100 | 0.080 | 0.47 |
| 768 (GPT-2 small) | 0.029 | 0.19 |
| 4,096 (Llama 3 8B) | 0.013 | 0.080 |

The typical cosine shrinks like `1/√d`, where `d` is the number of dimensions: `1/√768 = 0.036` is the spread, and the typical size is a bit smaller. Then we pushed further: 10,000 random directions in 768 dimensions, 13 times more directions than dimensions. Across all 49,995,000 pairs, the worst was 0.196, an angle of about 79 degrees. In 4,096 dimensions the worst of the same 10,000 was 0.093, or 85 degrees. A trained model chooses its directions, so it can do better than random.

### Why it only works when features are rare

Crosstalk adds up. Say `k` features are active at strength 1, and the model reads one of the inactive ones by taking a dot product with its direction. Each active feature leaks about `1/√d` into the reading, so the noise grows like `√(k/d)`. We simulated it with the 10,000 directions in 768 dimensions. The noise is an average over 300 random draws; the last two columns come from one draw:

| Features active at once | Noise on an inactive feature | Worst false reading | Weakest true reading |
|---|---|---|---|
| 3 | 0.062 | 0.23 | 1.02 |
| 10 | 0.114 | 0.46 | 0.81 |
| 30 | 0.197 | 0.76 | 0.67 |
| 100 | 0.361 | 1.50 | -0.01 |

With 3 features on, every true reading is far above every false one. By 30 they overlap, and at 100 the model can no longer tell what's on. Real text keeps the count low, because most concepts, from "Golden Gate Bridge" to "base64", are absent from almost every token.

## Step 3: sparse autoencoders pull the features out

If features are directions that are rarely active, finding them is a well-posed problem: find a large set of directions such that every activation vector is the sum of a few of them. A sparse autoencoder (SAE) is trained to find them. It's a two-layer network trained on activations collected from the model:

```text
f = ReLU(W_enc x + b_enc)        encoder: one number per feature, most of them 0
x̂ = W_dec f + b_dec              decoder: rebuild the activation from the features
loss = ‖x − x̂‖² + λ × Σ f_i      reconstruction error + sparsity penalty
```

`x` is one residual-stream vector. The encoder is much wider than `x`, with thousands to millions of features. Each column of `W_dec` is a feature's direction. The first term of the loss asks for an accurate rebuild; the second, an L1 penalty, charges `λ` for every unit of feature activation, which pushes most features to exactly 0.

![A left-to-right flow. A residual stream activation x of, for example, 4,096 numbers goes into an encoder (linear plus ReLU). Its output is a layer of feature activations, for example 1 million wide, where a few are active, for instance Golden Gate Bridge at 2.0 and fog at 0.5, and all the rest are exactly 0. The active features go to a linear decoder that produces a reconstruction x-hat of the same 4,096 numbers. The loss adds the squared reconstruction error to lambda times the sum of the feature activations, shown as a dotted sparsity-penalty arrow from the active features.](/diagrams/f16-sparse-autoencoder.svg "A sparse autoencoder rewrites one activation as a few active features out of a very large dictionary. The feature values are illustrative.")

The decoder has to rebuild the original vector from the handful of features that the penalty lets through.

### The penalty, worked by hand

Suppose the true activation is `x = 2.0 × bridge + 0.5 × fog`, where `bridge` and `fog` are perpendicular unit directions. If the SAE uses just those two features, with activations `a` and `b`, the loss is `(2.0 − a)² + (0.5 − b)² + λ × (a + b)`. Minimizing it gives `a = 2.0 − λ/2` and `b = 0.5 − λ/2`, never below 0:

| λ | Best bridge | Best fog | Reconstruction error | λ × (a + b) | Loss |
|---|---|---|---|---|---|
| 0 | 2.00 | 0.50 | 0 | 0 | 0 |
| 0.1 | 1.95 | 0.45 | 0.005 | 0.24 | 0.245 |
| 1.0 | 1.50 | 0 (off) | 0.50 | 1.50 | 2.00 |

The penalty shrinks every active feature by `λ/2`, so SAEs slightly underestimate how strongly features fire. It also switches off any feature weaker than `λ/2`: at `λ = 1`, "fog" drops out of the explanation even though it was there. With `λ = 0` nothing pushes toward sparsity, and a dictionary with a million features could smear `x` across thousands of them. A larger `λ` buys shorter explanations that leave out more.

### From a toy model to Claude

[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Bricken et al., 2023) tried this on a one-layer Transformer with a 512-neuron MLP. Training on 8 billion activation samples, with dictionaries from 512 to 131,072 features, it found features that no single neuron showed. One fires on Arabic script: Arabic is 0.13% of the training tokens but 81% of the tokens where the feature is active. Others fire on DNA sequences, base64 strings and Hebrew. Turning the base64 feature on made the model write base64.

Eight months later, [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Templeton et al., 2024) did the same on the middle layer of Claude 3 Sonnet, a production model:

| Dictionary | Features | Dead features (never fired in 10 million tokens) |
|---|---|---|
| 1M | 1,048,576 | about 2% |
| 4M | 4,194,304 | about 35% |
| 34M | 33,554,432 | about 65% |

In all three, fewer than 300 features were active on an average token, and the SAE explained at least 65% of the variance of the activations. Many features were multilingual and multimodal, firing on a concept in any language and in images. One feature, `1M/1013764`, fires on a misspelled variable named "rihgt" in Python, on similar bugs in C and Scheme, and on divide-by-zero errors, but not on typos in English prose. Clamped high on correct code, it made Claude invent an error message. The team also found features for unsafe code, bias, sycophancy, deception and power-seeking.

The paper adds a warning: "there's a difference (for example) between knowing about lies, being capable of lying, and actually lying in the real world."

Other labs have scaled the method up. OpenAI trained a [16 million-feature SAE on GPT-4](https://arxiv.org/abs/2406.04093) on 40 billion tokens and released SAEs for open-source models. Google DeepMind's [Gemma Scope](https://arxiv.org/abs/2408.05147) released more than 400 SAEs with more than 30 million features in total, covering every layer of Gemma 2 2B and 9B. Training them took over 20% of GPT-3's training compute and about 20 pebibytes of saved activations. Neuronpedia hosts an interactive browser for them, so you can try this yourself.

## Step 4: circuits, or how features connect

A list of active features doesn't yet explain a behavior. For that you need to know which components pass what to which. A circuit is a small subgraph of the model that carries out one behavior, and finding one takes interventions as well as observation.

### Activation patching: swap one piece and watch

The core experiment is activation patching. Run the model on a clean prompt and on a broken one, copy one internal activation from the clean run into the broken run, and measure how much of the right answer comes back. A piece that brings the answer back is carrying it.

[Meng et al. (2022)](https://arxiv.org/abs/2202.05262) called their version causal tracing and used it to ask where GPT-2 XL (1.5 billion parameters) stores facts like "The Space Needle is in downtown → Seattle". They corrupted the subject's token embeddings with Gaussian noise three times the embeddings' usual spread. Across 1,000 facts that dropped the right answer's average probability from 27.0% to 8.47%. Restoring one hidden state at a time, they found a strong "early site" at the last token of the subject, carried by the MLPs in middle layers. That fits [Part 6](/blog/transformer-block/)'s picture of the MLP as a key-value memory. The authors then edited single facts by changing one MLP's weights, a method called ROME (Rank-One Model Editing).

### A circuit in GPT-2 small

[Part 5](/blog/attention-explained/) met induction heads, attention heads that complete `[A][B] ... [A] → [B]`, which [Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) tied to learning from the prompt. [Wang et al. (2022)](https://arxiv.org/abs/2211.00593) traced a much larger circuit, in what they called the largest end-to-end attempt yet to reverse-engineer a natural behavior. They studied how GPT-2 small completes "When Mary and John went to the store, John gave a drink to" with "Mary". They found 26 attention heads in 7 classes, built around a three-step algorithm:

1. Duplicate token heads notice that "John" appeared twice.
2. S-inhibition heads use that to stop later heads from looking at "John".
3. Name mover heads copy the remaining name, "Mary", to the output.

Run on its own, the circuit achieves 87% of the full model's performance on the task. The paper also found backup heads that take over when the main name movers are knocked out, redundancy that a single tidy explanation would miss.

### Attribution graphs: tracing a frontier model

In 2025, Anthropic automated much of this tracing. [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Ameisen et al.) replaces the model's MLPs with a cross-layer transcoder, an SAE-like layer whose features read from the residual stream at one layer and feed the MLP outputs of every later layer, then draws an attribution graph of which features pushed which on a single prompt. Whatever the replacement fails to reproduce appears as "error nodes" that can't be interpreted.

The companion paper, [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Lindsey et al.), used a transcoder with 30 million features on Claude 3.5 Haiku. Two of its findings stand out.

The first is two-hop reasoning. For "Fact: the capital of the state containing Dallas is", the graph shows Dallas features activating Texas features, which activate "say Austin". When the researchers suppressed Texas and inserted California features taken from an Oakland prompt, the model answered "Sacramento".

The second is planning. Asked for a rhyming couplet that starts "He saw a carrot and had to grab it,", Claude wrote "His hunger was like a starving rabbit". Features for "rabbit" and "habit" were already active on the newline token, before the second line began. Suppressing "rabbit" made the model rewrite the line to end in "habit". Inserting a planned word into 25 poems made the line end with that word 70% of the time.

Anthropic [open-sourced the method](https://www.anthropic.com/research/open-source-circuit-tracing) in May 2025 as the `circuit-tracer` library, which has been used on Gemma-2-2b and Llama-3.2-1b, with graphs you can explore on Neuronpedia.

## In the real world

You don't always need an SAE to steer a model. [Activation Addition](https://arxiv.org/abs/2308.10248) (Turner et al., 2023) runs the model on two prompts, such as "Love" and "Hate", subtracts the two activation vectors at one layer, and adds the difference while the model generates. With it, a completion of "I hate you because" turned from "...you are the most disgusting thing I have ever seen" into "...you are so beautiful and I want to be with you forever." [Representation engineering](https://arxiv.org/abs/2310.01405) (Zou et al., 2023) uses the same kind of directions to monitor and control honesty, harmlessness and power-seeking.

Interpretability has also uncovered a hidden goal. Anthropic [trained a model with a hidden objective](https://arxiv.org/abs/2503.10965): it was taught about 52 made-up flaws in reward models (the scoring models from [Part 13](/blog/base-model-to-assistant/)) and trained to exploit 47 of them, and it learned to exploit the other 5 as well. Four blind teams then investigated. The three with access to weights and training data found the objective. One researcher with SAE access found a reward-model-bias feature in 90 minutes. The team with API access only failed after more than 70 researcher-hours.

It now feeds into testing before release. For Claude Sonnet 4.5 (September 2025), Anthropic ran what its system card calls ["the first pre-deployment white-box interpretability assessment"](https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf). An SAE showed features like "Fake or suspicious content" growing stronger over training on test scenarios, a sign that the model recognized evaluations. Suppressing those features raised misaligned behavior in some scenarios, but even in the worst case the model was no more misaligned than Claude Opus 4.1 or Claude Sonnet 4.

For monitoring, a probe is often enough. Google DeepMind found that [linear probes](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/) detected harmful intent in prompts almost perfectly, with an AUROC (a 0-to-1 score where 1 means every case is ranked correctly) of 0.999 even on data unlike their training set.

### How much is still dark

- Claude 3 Sonnet's SAEs explained at least 65% of the activations' variance, so up to a third was left out. When OpenAI spliced its 16 million-feature SAE into GPT-4 in place of the real activations, GPT-4 predicted text only as well as a model trained with about 10% of its compute.
- Features depend on the dictionary: a "San Francisco" feature in the 1M dictionary splits into 2 features in the 4M one and 11 in the 34M one, and none of them is the one true feature.
- Dictionaries miss things. Claude 3 Sonnet can list all the London boroughs, but the 34M SAE had features for only about 60% of them.
- A feature's name is a guess, made by people or another model reading the text that activates it. Steering tests the label, but only for the behaviors someone thought to test.
- Anthropic says its attribution graphs give "satisfying insight for about a quarter of the prompts we've tried", and that even the successes capture "a small fraction of the mechanisms of the model". The rest sits in the error nodes, which the authors call "dark matter".
- In March 2025, DeepMind's interpretability team reported that SAEs [underperformed linear probes](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/) at that harmful-intent detection, and said it was "deprioritising fundamental SAE research for the moment" while keeping SAEs "in a practitioner's toolkit".

## Common misconceptions

- **"Each neuron detects one concept."** Most neurons are polysemantic, like the one that fires on citations, dialogue, HTTP requests and Korean. Concepts live in directions that cut across many neurons, which is why SAEs exist.
- **"A feature for deception means the model is deceiving you."** It means the model represents the concept, as it represents every topic it read about. Whether it acts on it needs a behavioral test or an intervention.
- **"Steering a feature is a precise edit."** Golden Gate Claude brought up the bridge in love stories and budgets alike, and Anthropic warned that it might behave in unexpected, even jarring, ways. Clamping a direction changes everything that reads from it.
- **"SAE features are the model's true building blocks."** They're one useful decomposition of the model's own vectors, and they change with dictionary size and the sparsity penalty.
- **"If you can't find it with interpretability, it isn't there."** SAEs miss rare concepts, up to a third of the variance goes unexplained, and circuit tracing gives a satisfying answer for about a quarter of prompts.

## Key takeaways

- The residual stream is readable: decoding it layer by layer with the logit lens showed GPT-2 small giving " Paris" an 18% chance at block 10, before the final block lowered it to 3%.
- Models store more features than dimensions by giving each a nearly perpendicular direction. In 768 dimensions, no two of 10,000 random directions came closer than about 79 degrees, and the trick works because only a few features are active at once.
- A sparse autoencoder untangles those directions by rebuilding each activation from a few features out of millions, with a loss of reconstruction error plus `λ` times the total activation.
- Circuits explain behavior when you intervene: activation patching located facts in middle-layer MLPs, and attribution graphs showed Claude 3.5 Haiku going from Dallas to Texas to Austin, and planning a rhyme before writing the line.
- The tools already steer models and helped audit them before release, but they explain only part of any model, so treat their findings as evidence, not proof.

## What's next

That ends LLM Fundamentals: from a single next-token guess, through tokens, attention, training and alignment, to looking inside the trained model. The [LLM Inference & Optimization series](/series/llm-inference/) picks up from here, starting with [How LLM Inference Works](/blog/how-llm-inference-works/), and shows how these models are served quickly and cheaply. For building products on top of them, the [LLM System Design series](/series/llm-system-design/) works through real designs, and the [full Fundamentals reading order](/series/llm-fundamentals/) is there whenever you want to revisit a part.

## Further reading

- [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet (Templeton et al., 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html): SAEs with up to 34 million features on a production model, the Golden Gate Bridge feature and safety-relevant features.
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (Bricken et al., 2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html): the first convincing SAE results, on a one-layer Transformer, with polysemantic neurons and feature splitting.
- [Toy Models of Superposition (Elhage et al., 2022)](https://transformer-circuits.pub/2022/toy_model/index.html): why sparse features get packed into fewer dimensions, with the pentagon and other geometries.
- [A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)](https://transformer-circuits.pub/2021/framework/index.html): the residual stream as a channel that layers read from and write to.
- [Circuit Tracing: Revealing Computational Graphs in Language Models (Ameisen et al., 2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html): cross-layer transcoders and attribution graphs.
- [On the Biology of a Large Language Model (Lindsey et al., 2025)](https://transformer-circuits.pub/2025/attribution-graphs/biology.html): two-hop reasoning, rhyme planning and other case studies in Claude 3.5 Haiku.
- [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (Wang et al., 2022)](https://arxiv.org/abs/2211.00593): the 26-head IOI circuit and how to test a circuit's faithfulness.
- [Locating and Editing Factual Associations in GPT (Meng et al., 2022)](https://arxiv.org/abs/2202.05262): causal tracing and the ROME editing method.
- [In-context Learning and Induction Heads (Olsson et al., 2022)](https://arxiv.org/abs/2209.11895): the pattern-copying heads and their link to learning from the prompt.
- [Scaling and evaluating sparse autoencoders (Gao et al., 2024)](https://arxiv.org/abs/2406.04093): OpenAI's TopK SAEs, scaling laws and the 16 million-feature SAE on GPT-4.
- [Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 (Lieberum et al., 2024)](https://arxiv.org/abs/2408.05147): open SAEs for every layer of Gemma 2 2B and 9B.
- [Eliciting Latent Predictions from Transformers with the Tuned Lens (Belrose et al., 2023)](https://arxiv.org/abs/2303.08112): a trained, more reliable version of the logit lens.
- [Steering Language Models With Activation Engineering (Turner et al., 2023)](https://arxiv.org/abs/2308.10248): Activation Addition and steering vectors.
- [Representation Engineering: A Top-Down Approach to AI Transparency (Zou et al., 2023)](https://arxiv.org/abs/2310.01405): reading and controlling honesty and other concepts through directions.
- [Auditing language models for hidden objectives (Marks et al., 2025)](https://arxiv.org/abs/2503.10965): a blind auditing game in which SAEs helped uncover a hidden goal.
- [Open Problems in Mechanistic Interpretability (Sharkey et al., 2025)](https://arxiv.org/abs/2501.16496): a survey of what the field can't do yet.
- [Probing Classifiers: Promises, Shortcomings, and Advances (Belinkov, 2021)](https://arxiv.org/abs/2102.12452): what probes can and can't tell you.
- [interpreting GPT: the logit lens (nostalgebraist, 2020)](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens): the original logit-lens post.
- [Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research (Google DeepMind, 2025)](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/): where SAEs lost to linear probes.
- [Golden Gate Claude (Anthropic, 2024)](https://www.anthropic.com/news/golden-gate-claude): the 24-hour public demo of feature steering.
