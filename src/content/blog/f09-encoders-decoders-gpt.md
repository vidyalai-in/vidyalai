---
title: "Encoders, Decoders and GPT, and How Sampling Works"
description: "BERT, GPT and T5 share one Transformer and differ in what each token may see. Why decoder-only won, and how top-k, top-p and min-p pick tokens."
slug: "encoders-decoders-gpt"
category: "llm-fundamentals"
tags: ["transformers", "attention", "embeddings", "sampling"]
series: "llm-fundamentals"
seriesOrder: 9
pubDate: 2026-03-18
---

## TL;DR

The 2017 Transformer had two halves, an encoder that reads and a decoder that writes, and the models that followed split it three ways. Encoder-only models like BERT let every token see every other token and are trained to fill in blanks, which makes them good at understanding text and turning it into vectors for search. Decoder-only models like GPT let each token see only the past and are trained to predict the next token, which makes them good at writing, and that shape now runs almost every chatbot. Encoder-decoder models like T5 and Whisper keep both halves for tasks that turn one sequence into another. The second half covers the last step of generation: how greedy decoding, beam search, temperature, top-k, top-p, min-p and repetition penalties turn probabilities into one token, worked on real GPT-2 numbers.

## Why it matters

Ask the smallest GPT-2 to continue "The capital of France is" and always take its single most likely token. You get "the capital of the French Republic, and the capital of the French Republic is the capital of the French Republic", and it never stops. Over 200 tokens it uses just 11 different ones. The loop comes from the rule that picks tokens, and a fix from 2019 is simple: before you roll the dice, throw away the unlikely tokens that together hold the last 10% of the probability. For that same prompt, GPT-2 keeps 1,503 candidates. For "The United States of", it keeps exactly one.

Before any of that, a model has to be built to write at all. In 2018, Google's BERT and OpenAI's GPT were built from the same Transformer layers. They differed in what they were trained to predict and in one mask, the grid of which tokens may look at which, and those choices decided which one could hold a conversation.

## The intuition

Think of three people working with text. An editor gets a printed page with a few words blacked out and fills them in, reading the whole page, before and after each gap. A storyteller makes up a tale aloud, one word at a time, and can only build on what has already been said. An interpreter listens to a whole sentence in one language, then speaks it in another, word by word, glancing back at what they heard.

The editor is an encoder-only model, the storyteller is a decoder-only model, and the interpreter is an encoder-decoder.

The analogy breaks because people can't easily switch jobs, and a big enough storyteller can. Give it a page with a blank, or French to put into English, and it does the job. Step 3 explains why that happened.

## Step 1: three masks, three families

[Part 5](/blog/attention-explained/) built attention as a grid: every token's query scored against every token's key, then softmax across each row. A mask decides which cells of that grid are allowed. Take Part 5's sentence, "The river bank flooded". Each row is a token doing the looking; each column is a token it may look at.

**Encoder-only (BERT):** no mask. Every token sees every token.

| Query ↓ / key → | The | river | bank | flooded |
|---|---|---|---|---|
| The | ✓ | ✓ | ✓ | ✓ |
| river | ✓ | ✓ | ✓ | ✓ |
| bank | ✓ | ✓ | ✓ | ✓ |
| flooded | ✓ | ✓ | ✓ | ✓ |

**Decoder-only (GPT):** the causal mask. Each token sees itself and the tokens before it.

| Query ↓ / key → | The | river | bank | flooded |
|---|---|---|---|---|
| The | ✓ | ✗ | ✗ | ✗ |
| river | ✓ | ✓ | ✗ | ✗ |
| bank | ✓ | ✓ | ✓ | ✗ |
| flooded | ✓ | ✓ | ✓ | ✓ |

**Encoder-decoder (T5):** two stacks and three kinds of attention. Using the T5 paper's example of translating "That is good" into "Das ist gut", the encoder reads the English with the full grid above. The decoder writes the German with a causal grid over its own tokens. A third kind, cross-attention, lets every German token look at every English token:

| German query ↓ / English key → | That | is | good |
|---|---|---|---|
| Das | ✓ | ✓ | ✓ |
| ist | ✓ | ✓ | ✓ |
| gut | ✓ | ✓ | ✓ |

[Part 8](/blog/attention-is-all-you-need/) walks through cross-attention in the original paper.

The BERT paper names the two halves after this exact difference. A Transformer with bidirectional attention "is often referred to as a 'Transformer encoder' while the left-context-only version is referred to as a 'Transformer decoder' since it can be used for text generation" ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)).

Side by side:

| Family | Attention mask | Training objective | Typical use | Example models |
|---|---|---|---|---|
| Encoder-only | Full: every token sees every token | Fill in masked tokens | Classification, named-entity recognition, embeddings for search | BERT, RoBERTa, Sentence-BERT, ModernBERT |
| Decoder-only | Causal: each token sees only the past | Predict the next token | Chat, writing, code, almost every LLM | GPT-1 to GPT-3, Llama, Qwen, DeepSeek |
| Encoder-decoder | Full in the encoder, causal in the decoder, cross-attention between them | Rebuild corrupted text, or map input to output | Translation, speech recognition, summarization | Original Transformer, T5, BART, Whisper |

## Step 2: encoder-only models read in both directions

The authors of [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers, Google, 2018) argued that standard language models were too limited. "The major limitation is that standard language models are unidirectional," they wrote, and in OpenAI GPT "every token can only attend to previous tokens." Answering a question about a passage, for example, needs context from both sides.

Removing the mask creates a problem, though. If every token can see every token, the word to be predicted would "see itself", in the paper's words, and the task becomes trivial.

### Masked language modeling

BERT's fix is to hide some words and ask the model to recover them. The paper calls this a "masked LM" (MLM), a version of the old "Cloze" fill-in-the-blank test. In each sequence, 15% of the tokens are picked for prediction. Each picked token is then:

- replaced with a special `[MASK]` token 80% of the time ("my dog is hairy" becomes "my dog is [MASK]"),
- replaced with a random token 10% of the time ("my dog is apple"),
- left unchanged 10% of the time.

The random and unchanged cases exist because `[MASK]` never appears in real text later. The model can't tell which visible words might be wrong, so it has to keep a useful representation of every token. Random swaps hit only 1.5% of all tokens (10% of 15%), which the authors found does no harm.

BERT also had a second task, next sentence prediction: given two sentences, guess whether the second really came next (50% of the time it did) or was a random sentence. A year later, [RoBERTa](https://arxiv.org/abs/1907.11692) found that "removing the NSP loss matches or slightly improves downstream task performance", and most later encoders dropped it.

### Sizes, and what both directions buy you

BERT-base has 12 layers, a vector size of 768 and 110 million parameters, chosen to match GPT-1 for comparison. BERT-large has 24 layers, a vector size of 1,024 and 340 million. Both trained on BooksCorpus (800 million words) and English Wikipedia (2,500 million words). On release, BERT set records on 11 language-understanding tasks and raised the GLUE benchmark score to 80.5%, 7.7 points above the previous best.

The public `bert-base-uncased` checkpoint shows what reading both ways buys:

| Input to BERT-base | Top prediction for [MASK] |
|---|---|
| The capital of France is [MASK]. | paris (41.7%), lille (7.1%), lyon (6.3%) |
| The [MASK] of France is Paris. | capital (99.98%) |
| I went to the [MASK] to deposit my paycheck. | bank (90.1%) |
| I went to the [MASK] to catch some fish. | lake (33.3%), pond (20.9%), river (14.9%) |

In the second and third rows, the clue comes after the blank. A left-to-right model filling the second slot of "The ___ of France is Paris" has seen only "The", so it has no way to use "Paris". BERT reads the whole sentence at once.

### What encoders are used for

An encoder doesn't generate text. It turns text into one vector per token, and a small layer on top does the task: a label for the whole text (spam or not), a label per token (named-entity recognition, which tags people, places and companies), or the start and end of an answer span.

The largest use today is embeddings for search, which [Part 4](/blog/embeddings-and-logits/) introduced: one vector per passage, with similar meanings close together. Plain BERT is bad at this. Comparing two sentences means feeding both into the network together, and the [Sentence-BERT paper](https://arxiv.org/abs/1908.10084) worked out that finding the most similar pair among 10,000 sentences takes about 50 million passes, roughly 65 hours on a V100 GPU. Sentence-BERT fine-tunes BERT so each sentence gets its own vector, which cuts that to about 5 seconds. The popular [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is a 6-layer BERT-style encoder of this kind, mapping text to 384 numbers. [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/) and [BM25 vs Dense vs Hybrid Search](/blog/bm25-dense-hybrid-search/) show these vectors at work.

## Step 3: decoder-only and encoder-decoder models

### The GPT line

[GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (OpenAI, 2018) was "a 12-layer decoder-only transformer with masked self-attention heads" with 768-number vectors, trained on over 7,000 unpublished books to predict the next token. That's the causal objective from [Part 1](/blog/how-llms-work/). It was then fine-tuned separately for each task and beat the previous best on 9 of 12 tasks.

[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019) kept the design and scaled it to 1.5 billion parameters and 40 GB of web text. It showed that tasks could be stated in the text itself, with no fine-tuning: add "TL;DR:" after an article and the model writes a summary.

[GPT-3](https://arxiv.org/abs/2005.14165) (2020) scaled to 175 billion parameters and named the effect in-context learning: the model picks up a task from instructions or examples in its prompt, "without any gradient updates or fine-tuning". On the TriviaQA quiz benchmark, it scored 64.3% with no examples, 68.0% with one and 71.2% with a few, which matched the best fine-tuned systems in the same closed-book setting.

### Encoder-decoders: T5, BART and Whisper

[T5](https://arxiv.org/abs/1910.10683) (Text-to-Text Transfer Transformer, Google, 2019) recast every task as text in, text out. Translation is "translate English to German: That is good." and the answer is "Das ist gut." It trained with span corruption, a variant of masked language modeling: drop 15% of the tokens, replace each run of dropped tokens with one placeholder, and have the decoder write out only the missing pieces. "Thank you for inviting me to your party last week." becomes "Thank you `<X>` me to your party `<Y>` week." and the target is "`<X>` for inviting `<Y>` last `<Z>`".

T5 also ran a controlled comparison of the families, built from BERT-base-sized stacks and matched on compute per example (the encoder-decoder had twice the parameters but the same compute as the decoder-only model). Its conclusion: "For all tasks, the encoder-decoder architecture with the denoising objective performed best." The gap was large:

| Architecture (T5 paper, Table 2) | Objective | GLUE | SQuAD |
|---|---|---|---|
| Encoder-decoder | Fill in corrupted spans | 83.28 | 80.88 |
| Prefix LM (one stack, full attention over the input) | Fill in corrupted spans | 81.82 | 78.94 |
| Decoder-only | Fill in corrupted spans | 74.70 | 61.14 |
| Decoder-only | Predict the next token | 73.78 | 53.81 |

[BART](https://arxiv.org/abs/1910.13461) (Facebook AI, 2019) joins a BERT-like encoder to a GPT-like decoder and trains them to rebuild text that was shuffled and had spans blanked out. It was "particularly effective when fine tuned for text generation".

Encoder-decoders still fit best when input and output are clearly separate. OpenAI's [Whisper](https://arxiv.org/abs/2212.04356) speech recognizer chose "an encoder-decoder Transformer" because it "has been well validated to scale reliably": the encoder reads 30 seconds of audio, and the decoder writes the transcript. It was trained on 680,000 hours of labeled audio, and its largest version has 1,550 million parameters. Meta's [NLLB-200](https://arxiv.org/abs/2207.04672) translation models, covering 200 languages, are encoder-decoders too.

### Why decoder-only won anyway

If T5 found the encoder-decoder best, why is nearly every chatbot a decoder? Four reasons stand out.

1. T5 scored models after fine-tuning each one on each task. A 2022 study by [Wang et al.](https://arxiv.org/abs/2204.05832), with models of over 5 billion parameters, found that causal decoders trained on next-token prediction "exhibit the strongest zero-shot generalization after purely unsupervised pretraining." Masked encoder-style models led only after extra multitask fine-tuning.
2. A decoder can phrase any task as "continue this text", so the same weights answer questions, translate, summarize and write code. GPT-3's authors said they chose this design because it is "straightforward to both sample and compute likelihoods with". They also admitted the cost: likely "worse performance on tasks which empirically benefit from bidirectionality", such as fill-in-the-blank.
3. Next-token prediction gives a training signal at every position. BERT's MLM only predicts the 15% of tokens it masked, and its paper notes this means "more pre-training steps may be required".
4. Generation is cheap to serve. Because of the causal mask, nothing a token computed changes when later tokens arrive, so a decoder can store each token's keys and values and reuse them. That stored set is the [KV cache](/blog/kv-cache/).

## Step 4: from probabilities to a token

Every generating model ends each step with a logit per vocabulary token, turned into probabilities by softmax. [Part 1](/blog/how-llms-work/) introduced the two basic ways to choose: greedy decoding (always take the top token) and sampling (roll weighted dice), with temperature dividing the logits by `T` to sharpen or flatten the odds first.

### Greedy and beam search get stuck

The loop from the opening isn't a quirk of one prompt. In ["The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751), Holtzman et al. (2019) generated 5,000 passages with GPT-2 Large. They counted a passage as stuck when a phrase repeated at least three times at its end:

| Decoding method (GPT-2 Large) | Stuck in a repetition loop | Perplexity of the text |
|---|---|---|
| Human-written text | 0.28% | 12.38 |
| Greedy | 73.66% | 1.50 |
| Beam search, 16 beams | 28.94% | 1.48 |
| Pure sampling | 0.22% | 22.73 |
| Top-k, k = 40 | 0.78% | 6.88 |
| Nucleus (top-p), p = 0.95 | 0.36% | 13.13 |

Perplexity measures how surprised the model is by the text; lower means more predictable. Greedy and beam text is far more predictable than human writing, and the paper explains why: "Natural language rarely remains in a high probability zone for multiple consecutive time steps." People avoid saying the obvious. Worse, each repetition makes the next one more likely, a "positive feedback loop".

Beam search is the more careful version of greedy. It keeps the `b` most probable partial outputs, called beams, extends each one, and keeps the best `b` again. For translation, where the right output is tightly set by the input, that works well. The original Transformer used 4 beams (Part 8). For open-ended writing it fails the same way greedy does: the paper's Figure 1 shows 32 beams repeating "Universidad Nacional Autónoma de México" over and over.

Sampling from the full distribution avoids loops but fails differently. Each of GPT-2's 50,257 tokens gets a little probability, so every so often the draw lands on one of thousands of bad options, and later tokens follow it into odd territory. That's why pure sampling's perplexity, 22.73, is higher than human text's 12.38.

## Step 5: cutting the tail with top-k, top-p and min-p

The fix is to sample, but only from the believable part of the distribution. Three popular rules decide where to cut. Here they are on the real GPT-2 small distribution for "The capital of France is", the same one [Part 4](/blog/embeddings-and-logits/) showed, with " the" at 8.46% and " Paris" fifth at 3.22%.

**Top-k** keeps the `k` most likely tokens and renormalizes, meaning it divides each survivor's probability by their total so they add up to 1 again. [Fan et al. (2018)](https://arxiv.org/abs/1805.04833) used `k = 10` for story writing; GPT-2's paper used `k = 2` for its "TL;DR:" summaries.

**Top-p**, or nucleus sampling, is from the Holtzman paper. It keeps "the smallest set" of top tokens whose probabilities add up to at least `p`.

**Min-p**, from [Nguyen et al. (2024)](https://arxiv.org/abs/2407.01082), keeps every token whose probability is at least `min_p` times the top token's.

With top-k = 5, five tokens survive, holding 24.33% of the probability between them:

| Token | Original probability | After top-k = 5 |
|---|---|---|
| " the" | 8.46% | 34.77% |
| " now" | 4.79% | 19.71% |
| " a" | 4.62% | 18.97% |
| " France" | 3.24% | 13.31% |
| " Paris" | 3.22% | 13.25% |

Each survivor's share is its probability divided by 0.2433: for " the", `8.46 / 24.33 = 34.77%`. Here is the same distribution with all three rules, and with temperature 0.7 applied first:

| Setting (GPT-2 small, "The capital of France is") | Tokens kept | " the" after renormalizing | " Paris" after renormalizing |
|---|---|---|---|
| No filter | 50,257 | 8.46% | 3.22% |
| Top-k = 5 | 5 | 34.77% | 13.25% |
| Top-p = 0.9 | 1,503 | 9.40% | 3.58% |
| Min-p = 0.1 (cut-off 0.85%) | 16 | 19.67% | 7.50% |
| Temperature 0.7, then top-p = 0.9 | 62 | 25.26% | 6.37% |
| Temperature 0.7, then min-p = 0.1 | 9 | 32.5% | 8.2% |

We checked the counts with Hugging Face Transformers' own filter classes. Top-p keeps 1,503 tokens because GPT-2 small is unsure: the last token in, " equal", has a probability of 0.0054%. Min-p's cut-off is `0.1 × 8.46% = 0.85%`, which lets in 16 tokens down to " at" (0.87%) and stops before " an" (0.80%).

The real difference between the rules shows up when the model is confident. Same three settings, four prompts:

| Prompt (GPT-2 small) | Top token | Top-k = 5 keeps | Top-p = 0.9 keeps | Min-p = 0.1 keeps |
|---|---|---|---|---|
| "The United States of" | " America", 96.6% | 5 | 1 | 1 |
| "Once upon a" | " time", 17.7% | 5 | 1,956 | 4 |
| "The capital of France is" | " the", 8.5% | 5 | 1,503 | 16 |
| "My favorite food is" | " the", 8.4% | 5 | 1,839 | 10 |

Top-k always keeps 5, so after "The United States of" it keeps four bad options and gives them more weight. As Holtzman et al. put it, in some contexts a small `k` risks "generating bland or generic text", and in others a large `k` lets in "inappropriate candidates". Top-p and min-p both adjust to the model's confidence. Top-p measures the total weight of the tail, so a flat distribution lets in thousands of tokens; min-p ties the cut-off to the top token. Its authors report better quality and diversity, especially at high temperatures. A 2025 re-analysis by [Schaeffer et al.](https://arxiv.org/abs/2506.13681) disputes those results, so treat min-p as one reasonable option rather than a proven upgrade.

Here is what sampling produces with GPT-2 small at top-p = 0.9, 30 tokens, three random seeds:

- Seed 0: "The capital of France is a gritty bastion of resistance against the "New Right." ..."
- Seed 1: "The capital of France is upset with Sir Bennington, London's top novelist, ..."
- Seed 2: "The capital of France is Madrid, which is between Madrid and Barcelona ..."

No loops, but seed 2 shows the cost: sampling sometimes picks a wrong fact the model gave some probability to, and a 124-million-parameter model gives wrong facts plenty.

### Repetition penalties

A second family of fixes changes the logits of tokens that already appeared. OpenAI's [API documentation](https://developers.openai.com/api/docs/guides/advanced-usage) defines its two penalties exactly:

```text
logit[j] = logit[j] - count[j] × frequency_penalty - (1 if count[j] > 0 else 0) × presence_penalty
```

Here `count[j]` is how many times token `j` has been sampled so far. The presence penalty is a one-time hit for any token that has appeared, nudging the model toward new topics. The frequency penalty grows with each use and targets word-for-word loops. The docs suggest values "around 0.1 to 1" for mild effect and up to 2 for strong suppression. For example, a token used 3 times, with both penalties set to 0.5, loses `3 × 0.5 + 0.5 = 2` from its logit. That divides its odds by `e² ≈ 7.4`.

Hugging Face's `repetition_penalty` comes from the [CTRL paper](https://arxiv.org/abs/1909.05858) and works differently. It divides a positive logit by the penalty and multiplies a negative one by it, and by default it counts the prompt's tokens too. Because it scales logits rather than shifting them, its effect depends on their size. GPT-2's logits sit near -100 (Part 4), so a penalty of 1.2 turns -100.25 into -120.3, a drop of 20 that bans the token. Greedy decoding with that setting escapes the loop at once: "The capital of France is the city, and its inhabitants are not only rich but also poor."

### Putting it together

Libraries apply these steps in a fixed order. In [Hugging Face Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py), the repetition penalty and other logit processors run first, then temperature, then top-k, then top-p, then min-p. [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py) applies penalties, then temperature, then min-p, then top-k and top-p. The order matters: temperature changes the probabilities that top-p and min-p measure. At temperature 0.7, top-p = 0.9 keeps 62 tokens instead of 1,503.

![A left-to-right flow. Logits, one score per token, pass through penalties for tokens already used. From there, greedy decoding takes the top token straight to the output. Otherwise temperature divides the logits by T, then a group labeled cut the tail applies top-k, top-p and min-p in turn. Softmax renormalizes over the surviving tokens, and a random draw from a seeded generator picks the next token.](/diagrams/f09-sampling-pipeline.svg "The path from logits to one token, in the order Hugging Face Transformers applies the steps.")

Every setting either reshapes the logits or deletes candidates before a single weighted draw.

The "random" draw comes from a pseudo-random number generator. Fix its seed and the same logits give the same token: running seed 0 again above reproduced its text exactly. OpenAI's API promises only "(mostly) deterministic outputs" with a fixed `seed`, in [its docs' words](https://developers.openai.com/api/docs/guides/advanced-usage), because batching on a shared server can change the logits [very slightly](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) between runs.

Constrained decoding for structured output works like every filter above: set the logits of forbidden tokens to minus infinity before the draw. If the grammar for a JSON schema says the next character must be `"` or `}`, every other token is masked out. [Guaranteed Valid JSON Output](/blog/guaranteed-valid-json-output/) covers how libraries compile schemas into those masks.

## In the real world

- Open-weight chat model families such as Llama, Qwen and DeepSeek generate with a causal mask, and the sampling settings above show up as parameters in their serving engines and in hosted APIs.
- Embedding models for retrieval are often BERT-family encoders: all-MiniLM-L6-v2 has 6 layers and 384-number vectors. Encoders are still being built. [ModernBERT](https://arxiv.org/abs/2412.13663) (2024) trained on 2 trillion tokens with an 8,192-token context. Decoders have moved in here too: [Qwen3 Embedding](https://arxiv.org/abs/2506.05176) builds embedding models of 0.6 to 8 billion parameters on top of the Qwen3 LLMs.
- Whisper's decoding recipe sums up Step 4. It uses beam search with 5 beams "to reduce repetition looping which happens more frequently in greedy decoding". It starts at temperature 0 and raises it by 0.2, up to 1.0, whenever the output looks unreliable: an average log-probability below -1, or text so repetitive it compresses by more than 2.4 times with gzip.

## Common misconceptions

- **"BERT and GPT are different architectures."** They use nearly the same Transformer block. BERT-base was sized to match GPT-1 on purpose. The main differences are the attention mask and the training objective.
- **"Encoder-decoders lost because they're worse."** In T5's controlled, fine-tuned comparison they won clearly. Decoder-only won for general models because it's simpler, learns from every token, picks up tasks from the prompt and is cheap to serve.
- **"Greedy decoding gives the best answer."** It gives the most likely next token, one step at a time. For open-ended text, 73.66% of greedy GPT-2 Large passages ended in a loop, against 0.28% of human ones.
- **"Top-k and top-p do the same thing."** Top-k always keeps `k` tokens. Top-p kept 1 token after "The United States of" and 1,956 after "Once upon a".
- **"Temperature 0 with a fixed seed is fully reproducible on an API."** It's reproducible on your own machine. On a shared server, batching can change the arithmetic enough to flip a close call between two tokens.

## Key takeaways

- One Transformer, three masks: encoders see everything, decoders see only the past, and encoder-decoders do both, joined by cross-attention.
- BERT learns by filling in blanks, so it uses context on both sides. That makes encoders strong at classification and at producing embeddings for search.
- GPT learns by predicting the next token. That one objective, scaled up, gave in-context learning, and it's why almost every modern LLM is decoder-only.
- Greedy decoding and beam search fall into loops on open-ended text, and pure sampling wanders off, so good generation samples from a trimmed distribution.
- Top-k keeps a fixed number of tokens, top-p keeps a fixed share of the probability, and min-p keeps tokens close to the leader. Penalties, temperature and the order of these steps all change the result.

## What's next

GPT-2 small, the model behind every number in this article, is a 2019 design. [Part 10](/blog/modern-llm-architecture/) puts its block next to the one inside a 2026 model and goes through what changed: RMSNorm instead of layer norm, SwiGLU in the MLP, rotary positions, and grouped-query and multi-head latent attention to shrink the KV cache.

## Further reading

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805): masked language modeling, next sentence prediction, and the encoder/decoder naming.
- [Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): GPT-1, the first decoder-only pretrain-then-fine-tune model.
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2 and tasks specified in plain text.
- [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165): GPT-3 and in-context learning.
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683): T5, span corruption and the controlled architecture comparison.
- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization? (Wang et al., 2022)](https://arxiv.org/abs/2204.05832): why causal decoders win when there is no task-specific fine-tuning.
- [BART: Denoising Sequence-to-Sequence Pre-training (Lewis et al., 2019)](https://arxiv.org/abs/1910.13461): a BERT-like encoder joined to a GPT-like decoder.
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084): turning an encoder into an embedding model for search.
- [Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022)](https://arxiv.org/abs/2212.04356): Whisper, an encoder-decoder with beam search and temperature fallback.
- [The Curious Case of Neural Text Degeneration (Holtzman et al., 2019)](https://arxiv.org/abs/1904.09751): why greedy and beam search loop, and nucleus (top-p) sampling.
- [Hierarchical Neural Story Generation (Fan et al., 2018)](https://arxiv.org/abs/1805.04833): top-k sampling for story writing.
- [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs (Nguyen et al., 2024)](https://arxiv.org/abs/2407.01082): the min-p rule; read with [Min-p, Max Exaggeration (Schaeffer et al., 2025)](https://arxiv.org/abs/2506.13681), which disputes its gains.
- [CTRL: A Conditional Transformer Language Model for Controllable Generation (Keskar et al., 2019)](https://arxiv.org/abs/1909.05858): the source of the multiplicative repetition penalty.
- [Advanced usage (OpenAI API docs)](https://developers.openai.com/api/docs/guides/advanced-usage): exact definitions of frequency and presence penalties, and seeds.
- [Generation strategies (Hugging Face docs)](https://huggingface.co/docs/transformers/generation_strategies): greedy, beam search and sampling options in Transformers.
- [How to generate text (Patrick von Platen, Hugging Face, 2020)](https://huggingface.co/blog/how-to-generate): a visual walk through greedy, beam, top-k and top-p with GPT-2.
- [The Illustrated BERT, ELMo, and co. (Jay Alammar, 2018)](https://jalammar.github.io/illustrated-bert/): a visual guide to how BERT is pretrained and used.
