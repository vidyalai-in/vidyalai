---
title: "How a Large Language Model Works: The Big Picture"
description: "What really happens when ChatGPT or Llama answers you: tokens, next-token prediction, the Transformer, and why one 2017 paper changed everything."
slug: "how-llms-work"
category: "llm-fundamentals"
tags: ["transformers", "attention", "tokens", "training"]
series: "llm-fundamentals"
seriesOrder: 1
pubDate: 2026-10-23
---

## TL;DR

A large language model (LLM) does one thing: given some text, it predicts which small piece of text, called a token, is most likely to come next. It does that with a neural network called a Transformer, a stack of identical layers that turns each token into a list of numbers, lets the tokens share information through a mechanism called attention, and produces a probability for every token in its vocabulary. To write a whole answer, the model picks one token, adds it to the text, and runs again, one token at a time. Everything that looks like knowledge or reasoning comes from training that prediction on trillions of tokens of text. This article walks through that loop end to end; the rest of the series opens up each part.

## Why it matters

Type "The capital of France is" into a chatbot and it answers "Paris" in a fraction of a second. It didn't look the answer up in a database, a search engine or a list of capitals. A file of numbers produced it, one word-piece at a time, and the same file can also write Python, explain photosynthesis and translate Hindi into German.

How can "predict the next word" produce all of that? Nobody fully understands why it works as well as it does. *How* it works, the actual machinery, is well understood, and easier to follow than you might expect. Once you see it, a lot of LLM behavior stops being mysterious: why models are confidently wrong, why long prompts cost more, why the same question can get different answers, and why the GPUs that run them are so expensive.

Most of that machinery comes from one paper. In 2017, a team of eight researchers, most of them at Google, published ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). It introduced the Transformer to translate between languages. Every major LLM since, from GPT to Llama to Claude to Gemini, is built on it.

## The intuition

Think of the world's most well-read autocomplete. Your phone's keyboard suggests the next word from the last word or two you typed. An LLM does the same job, but it looks at everything you've written so far, sometimes hundreds of pages, and it has read a large share of the public internet, digitized books and code to learn which continuations are likely.

To predict the next word well, you have to understand a lot. To continue "The patient was given penicillin because the infection was", you need to know something about medicine. To continue a half-written Python function, you need to know Python. Training a model to predict text well forces it to pick up grammar, facts and patterns of reasoning along the way, because they all make the prediction better.

The analogy breaks in one place. Autocomplete stores phrases. An LLM doesn't store the text it read. It stores billions of numbers, called parameters, that were nudged during training until the predictions got good. What it "knows" is spread across those numbers, which is why it can't tell you where it learned something, and why it sometimes fills gaps with confident inventions.

## Step 1: text becomes tokens

A model can only work with numbers, so the first step is a tokenizer, a fixed program that splits text into pieces from a vocabulary and replaces each piece with its ID number.

Tokens are often whole common words, but rare words get split into smaller pieces. Spaces and punctuation are part of the pieces too. Our example might come out like this (real tokenizers differ in the details):

| Text piece | `The` | ` capital` | ` of` | ` France` | ` is` |
|---|---|---|---|---|---|
| Token ID (illustrative) | 791 | 6864 | 315 | 9822 | 374 |

For English, [OpenAI's rule of thumb](https://developers.openai.com/api/docs/concepts) is that one token is about 4 characters, or about three-quarters of a word. So 1,000 words is roughly 1,300 tokens.

Vocabularies are big. GPT-2 used [50,257 tokens](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf); Llama 3 uses [128,000](https://arxiv.org/abs/2407.21783). Tokens are also how you're billed and how context limits are measured, which is why "tokens per second" and "price per million tokens" show up everywhere. Part 2 of this series covers how tokenizers are built and why they make models bad at counting the letters in "strawberry".

## Step 2: tokens become vectors

A token ID like 9822 is just a label. The number itself carries no meaning: token 9822 isn't "bigger" than token 315 in any useful sense. So the next step swaps each ID for a vector, a list of numbers, taken from a big lookup table called the embedding matrix.

In GPT-3 each vector has 12,288 numbers. You can think of each one as a point in a space with 12,288 directions. During training, tokens with related meanings end up near each other, so "France", "Paris" and "French" sit closer together than "France" and "carburetor". The model learns this layout; nobody designs it.

Now our five tokens are five vectors. Everything from here to the end is arithmetic on vectors.

## Step 3: the Transformer stack mixes in context

A single token's vector only knows about that token. The word "bank" gets the same starting vector in "river bank" and "bank account". The model's main job is to update each token's vector with information from the tokens around it, so that by the end, the last position knows enough about the whole sentence to predict what comes next.

That work happens in a stack of identical layers called Transformer blocks. GPT-3 has 96 of them. Llama 3.1 405B has 126. Each block does two things:

1. **Attention.** Every token looks back at the tokens before it and decides which ones matter for it right now. In "The capital of France is", the position for "is" will weight "capital" and "France" heavily. It then pulls in a weighted mix of their information. This is the "attention" in "Attention Is All You Need", and it's how "bank" can learn which bank it is.
2. **A feed-forward network** (also called the MLP, for multi-layer perceptron). Each token's vector is processed on its own by a small neural network. This is where much of the model's stored knowledge seems to live. [Researchers have found](https://arxiv.org/abs/2012.14913) that these layers behave like key-value memories: patterns in the input act as keys that pull out stored associations, roughly how "France + capital" can turn into something close to "Paris".

Each block adds what it computed to the vector instead of replacing it. A good mental picture is a shared notepad per token that each of the 96 layers reads and writes a few notes on. After the last block, the vector at the final position holds the model's best summary of "what should come next".

Attention is the subject of Parts 4 and 5. For now, remember that it connects every token to every earlier token directly, in one step, no matter how far apart they are.

## Step 4: scores become probabilities

The last vector is turned into one score for every token in the vocabulary. These raw scores are called logits. With a 50,257-token vocabulary, that's 50,257 scores, each saying how strongly the model favors that token as the next one.

Scores aren't probabilities yet. A function called softmax converts them: it raises the constant `e` (about 2.718) to the power of each score, then divides by the total so everything adds up to 1. Bigger scores get much bigger shares.

Suppose the model's top five candidates have these logits (made-up numbers, and we'll pretend the rest of the vocabulary doesn't exist):

| Candidate | Logit | `e^logit` | Probability |
|---|---|---|---|
| Paris | 6.0 | 403.4 | **85.2%** |
| a | 3.5 | 33.1 | 7.0% |
| the | 3.0 | 20.1 | 4.2% |
| known | 2.5 | 12.2 | 2.6% |
| Lyon | 1.5 | 4.5 | 0.9% |
| **Total** | | 473.3 | 100% |

The probability of "Paris" is `403.4 / 473.3 = 0.852`. In formula form, `p(token) = e^logit / Σ e^logit`: the exponential of this token's score divided by the sum of everyone's exponentials. Notice that a lead of 2.5 points in logits became a 12-to-1 lead in probability.

## Step 5: pick a token, append it, repeat

Now the model has a probability for every possible next token. Something has to choose one. That step is called sampling, and you control it:

- Greedy decoding always takes the top token. It's predictable but tends to sound flat and can get stuck repeating itself.
- Sampling rolls weighted dice: "Paris" 85.2% of the time, "a" 7% of the time, and so on. This is why the same prompt can give different answers.
- Temperature reshapes the odds before the roll by dividing every logit by a number `T`. Below 1 makes the leader stronger; above 1 flattens the field.

Same logits, three temperatures:

| Candidate | T = 0.5 | T = 1 | T = 2 |
|---|---|---|---|
| Paris | 99.0% | 85.2% | 55.9% |
| a | 0.7% | 7.0% | 16.0% |
| the | 0.2% | 4.2% | 12.5% |
| known | 0.1% | 2.6% | 9.7% |
| Lyon | 0.0% | 0.9% | 5.9% |

At `T = 2`, "Lyon" goes from almost never to about one draw in seventeen. That's the "creativity" knob, and also a source of mistakes. Other settings, like top-p, cut off the long tail of unlikely tokens before sampling. Part 8 covers them.

Once a token is chosen, it's appended to the text, and the whole model runs again on "The capital of France is Paris" to predict the token after that. The loop keeps going until the model produces a special end token or hits a length limit.

![A left-to-right flow: text so far goes into the tokenizer, then embeddings, then a stack of N Transformer blocks with attention and MLP, then logits, softmax, and picking one token, Paris, which is appended before the whole model runs again.](/diagrams/f01-generation-loop.svg "One trip through the model produces one token. Generating an answer repeats the trip once per token.")

So a 500-token answer means 500 trips through all 96 layers. That fact explains a lot about LLM speed and cost, and it's the starting point of the [LLM Inference & Optimization series](/series/llm-inference/).

## Where the numbers come from: parameters

Every step above uses learned numbers: the embedding matrix, the attention weights and the MLP weights in every block, and the final layer that produces logits. These are the model's parameters, and "175B" in "GPT-3 175B" counts them.

You can roughly rebuild that number yourself. In a GPT-style block with vector size `d`, attention holds about `4 × d²` parameters and the MLP about `8 × d²`, so a block holds about `12 × d²`. For GPT-3, `d = 12,288` and there are 96 blocks:

```text
per block:   12 × 12,288²          ≈ 1.81 billion
96 blocks:   96 × 1.81 billion      ≈ 173.9 billion
embeddings:  50,257 tokens × 12,288 ≈ 0.6 billion
total                               ≈ 174.6 billion  (GPT-3 reports 175B)
```

The arithmetic matches the [GPT-3 paper's](https://arxiv.org/abs/2005.14165) 175 billion to within half a percent. Almost all of the parameters live in the repeated blocks; the vocabulary table is about 0.35% of the total.

Each parameter is usually stored in 2 bytes, so GPT-3's weights alone take about 350 GB, more than four 80 GB GPUs can hold. Every generated token touches all of them.

## Step 0: how the model learned all this

None of these numbers is set by hand. They start random, and training adjusts them. The training task is the same next-token prediction you just saw:

1. Take a chunk of real text from the training data.
2. At every position, ask the model to predict the next token.
3. Measure how much probability it gave to the token that actually came next. The standard measure is cross-entropy loss: low when the model put high probability on the right answer, high when it was surprised.
4. Nudge every parameter slightly in the direction that would have made the right answer more likely. The algorithm that works out those directions is called backpropagation.
5. Repeat, trillions of times.

No one labels the data. The text is its own answer key, which is why models can learn from so much of it. Some numbers for scale:

- GPT-3 trained on [300 billion tokens](https://arxiv.org/abs/2005.14165).
- Llama 3.1 405B trained on [15.6 trillion tokens](https://arxiv.org/abs/2407.21783), 52 times more. At about 0.75 words per token, that's roughly 11.7 trillion words.
- It used [3.8 × 10²⁵](https://arxiv.org/abs/2407.21783) floating-point operations (FLOPs), simple multiplications and additions, on up to 16,000 H100 GPUs. The model card lists [30.84 million GPU-hours](https://huggingface.co/meta-llama/Llama-3.1-405B), which on a single GPU would take about 3,500 years.

A handy rule from the [scaling-laws paper](https://arxiv.org/abs/2001.08361) predicts that compute: training costs about `6 × N × D` FLOPs, where `N` is the number of parameters and `D` is the number of training tokens. Check it on GPT-3: `6 × 175 billion × 300 billion = 3.15 × 10²³`, and the paper reports 3.14 × 10²³. Part 9 explains where the 6 comes from and what scaling laws say about choosing `N` and `D`.

### From base model to assistant

A model trained only to predict internet text is called a base model. Ask it a question and it might continue with three more questions, because that's what a list of questions on a web page looks like. As the [InstructGPT paper](https://arxiv.org/abs/2203.02155) put it, predicting the next token on a web page is a different objective from following instructions helpfully and safely.

So chat models get a second round of training on examples of good conversations, then learn from human preferences between pairs of answers. The effect is large: in that paper, people preferred answers from a 1.3 billion-parameter instruction-tuned model over the raw 175 billion-parameter GPT-3, a model more than 100 times bigger. Part 10 covers this post-training.

## Why the 2017 paper was the turning point

Before the Transformer, the best language models were recurrent neural networks (RNNs). An RNN reads text the way you read a sentence aloud: one word at a time, carrying a running summary forward. To process word 100, it has to finish words 1 through 99 first.

That design had two problems. Information from early words faded as the summary was overwritten again and again. And because each step waited on the previous one, the work couldn't be spread across the thousands of parallel arithmetic units in a GPU. The Transformer paper names it directly: this "inherently sequential nature precludes parallelization within training examples."

The Transformer dropped recurrence entirely. Attention connects every position to every other position in one step, so during training all positions of a sentence are processed at the same time. The results in the paper:

- 28.4 BLEU on English-to-German translation, a new best score. BLEU measures overlap with human translations.
- 41.8 BLEU on English-to-French, at a fraction of the training cost of previous best models.
- Trained on one machine with 8 GPUs: 12 hours for the base model, 3.5 days for the big one.
- The base model had 65 million parameters, the big one 213 million.

A parallel architecture meant more data and more compute could be turned straight into better models. A year later, OpenAI's GPT-1 kept only the Transformer's decoder half, the part that generates text, and trained it on 7,000 books. After that, the models mostly got bigger:

| Model | Year | Parameters | Training tokens | Context window |
|---|---|---|---|---|
| Transformer (big) | 2017 | 213M | (translation pairs) | n/a |
| GPT-1 | 2018 | 117M | (BooksCorpus) | 512 |
| GPT-2 | 2019 | 1.5B | (40 GB of web text) | 1,024 |
| GPT-3 | 2020 | 175B | 300B | 2,048 |
| Llama 3.1 405B | 2024 | 405B | 15.6T | 128K |

From GPT-1 to Llama 3.1 405B, parameter counts grew about 3,500 times in six years. The block design underneath barely changed.

## In the real world

- Open-weight models like Llama, Mistral, Qwen and DeepSeek publish their parameters, so you can download them and run the loop above on your own hardware. An 8 billion-parameter model needs about 16 GB just for its weights.
- Closed models like GPT, Claude and Gemini run the same kind of loop on their providers' servers. Their sizes aren't published.
- Context windows have grown from 512 tokens in GPT-1 to 128,000 in Llama 3.1, 256 times more. Every token in the window is something attention can look back at, and every one costs memory and compute.
- Speed is usually quoted in tokens per second, because generation is one trip through the model per token. Even an 8 billion-parameter model on a top data-center GPU tops out around 200 tokens per second for a single user; [Part 1 of the inference series](/blog/how-llm-inference-works/) shows why.

## Common misconceptions

- **"The model looks up answers in a database."** There's no database inside. Facts are spread across billions of parameters as statistical patterns, which is why models can be wrong with full confidence and can't cite where they learned something. Systems that need sources add retrieval on top; see [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/).
- **"It plans the whole answer, then types it out."** The model commits to one token at a time. Any planning has to happen inside the vectors at each step. That's one reason asking a model to reason step by step in writing often helps: the written steps become context for the tokens that follow.
- **"Temperature 0 makes it deterministic."** It makes the choice greedy, but on a shared server your request is batched with other people's, and the [arithmetic can come out very slightly different](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) depending on the batch size. That's enough to flip a close call between two tokens, so outputs can still vary between runs.
- **"Bigger is always better."** The InstructGPT result shows training matters as much as size: a 100 times smaller model won when it was trained to follow instructions.
- **"It reads words."** It reads tokens. That's why tasks like counting letters or reversing a word can trip up a model that writes fluent essays.

## Key takeaways

- An LLM predicts one token at a time. Generating an answer is a loop: predict, pick, append, repeat.
- Each trip turns tokens into vectors, mixes context through a stack of Transformer blocks (attention plus an MLP), then turns the final vector into probabilities with softmax.
- Sampling settings like temperature decide how adventurous the pick is, which is why the same prompt can get different answers.
- All of the model's skill lives in its parameters, learned by predicting the next token across trillions of tokens of text: 175 billion parameters for GPT-3, 405 billion for Llama 3.1.
- "Attention Is All You Need" made this possible by replacing step-by-step RNNs with attention, which processes every position at once and scales with compute.

## What's next

Part 2 is about the first thing that happens to your text: tokenization. It shows how a tokenizer builds its vocabulary from raw text, why "strawberry" is hard for a model to spell, and why the same sentence can cost more tokens in one language than in another.

The rest of the series, in order: tokenization, embeddings and logits, attention, multi-head attention and the Transformer block, positional encoding, a guided reading of "Attention Is All You Need", from the Transformer to GPT, pretraining and scaling laws, and turning a base model into an assistant.

## Further reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): the paper that introduced the Transformer and the anchor for this series.
- [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165): GPT-3, with its architecture table and training compute.
- [The Llama 3 Herd of Models (Llama Team, Meta, 2024)](https://arxiv.org/abs/2407.21783): a detailed, current account of building and training an open LLM.
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155): why a chat assistant differs from a raw next-token predictor.
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361): the source of the `6 × N × D` compute rule.
- [The Illustrated Transformer (Jay Alammar, 2018)](https://jalammar.github.io/illustrated-transformer/): the standard visual walkthrough of the architecture.
- [Transformers, the tech behind LLMs (3Blue1Brown, 2024)](https://www.youtube.com/watch?v=wjZofJX0v4M): an animated look at embeddings, next-token prediction and softmax.
- [Intro to Large Language Models (Andrej Karpathy, 2023)](https://www.youtube.com/watch?v=zjkBMFhNj_g): a one-hour big-picture talk on pretraining, fine-tuning and what LLMs are.
- [Let's build GPT: from scratch, in code, spelled out (Andrej Karpathy, 2023)](https://www.youtube.com/watch?v=kCc8FmEb1nY): build a small GPT line by line, for readers who want to see the code.
