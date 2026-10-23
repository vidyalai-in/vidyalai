---
title: "Tokenization: How LLMs Turn Text into Tokens"
description: "How byte-pair encoding builds a vocabulary, why models stumble on spelling and arithmetic, and why the same sentence costs more tokens in Hindi."
slug: "tokenization"
category: "llm-fundamentals"
tags: ["tokens", "tokenization", "transformers", "cost", "multilingual"]
series: "llm-fundamentals"
seriesOrder: 2
pubDate: 2026-10-23
---

## TL;DR

Before a language model sees your text, a tokenizer cuts it into pieces called tokens and swaps each piece for an ID number. Most modern tokenizers learn their pieces with byte-pair encoding (BPE): start from single bytes, then repeatedly glue together the pair of neighbors that appears most often in a big pile of text, until the vocabulary reaches a target size such as 50,257 or 200,000 tokens. The result is efficient for common English words and clumsy for everything else. That one design choice explains why models struggle to count the letters in "strawberry", why they find long arithmetic harder than it looks, and why the same sentence can cost several times more tokens in Hindi or Tamil than in English.

## Why it matters

Ask GPT-4o how many r's are in "strawberry". The model never sees the letters s-t-r-a-w-b-e-r-r-y. In OpenAI's current tokenizer, " strawberry" (with its leading space) is a single token, ID 101830. Counting r's in it is like counting the letters in a word you've only ever heard as one sound.

Now take one sentence from a standard translation benchmark: "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool..." In English, the tokenizer behind GPT-4 turns it into 49 tokens. The same sentence in Hindi takes 304 tokens with that tokenizer, more than six times as many. Since you pay per token and context windows are measured in tokens, a Hindi speaker using that tokenizer paid about six times more and could fit about a sixth as much text.

Both effects come from the tokenizer, a small, fixed program that runs before the neural network does. [Part 1](/blog/how-llms-work/) treated it as a black box; this article explains how it is built and what it does to the model's input.

## The intuition

Picture a typesetter with a box of rubber stamps. Some stamps are single letters. Others are whole chunks that come up all the time: "the", "ing", " capital", " France". To print a line, the typesetter uses the biggest stamps that fit, falling back to single letters only when nothing bigger matches. Common words take one stamp. A rare word like "Pangasinan" takes three or four.

Now suppose the typesetter is only allowed a fixed number of stamps, say 50,000. Which ones should they make? The obvious answer is to look at a huge pile of past print jobs and make stamps for whatever pieces show up most often. That is roughly what a tokenizer's training does.

The analogy breaks in two places. First, the stamps are chosen purely by frequency, not by meaning. "Strawberry" can be split as `st` + `raw` + `berry`, which has nothing to do with straw. Second, the model downstream never sees the ink. It only sees which stamp was used, as a number. It has to learn from context alone that stamp 101830 is spelled with three r's.

## Step 1: why not letters, or whole words?

A tokenizer has to balance two costs. Small pieces make sequences long. Big pieces make the vocabulary huge and still miss rare words.

Here are three ways to cut the 2,009 English sentences of the FLORES-200 benchmark (a set of Wikipedia sentences translated by people into 200 languages), counted with Python:

| Unit | Count | Vocabulary needed | Problem |
|---|---|---|---|
| Characters | 257,160 | a few hundred for this text, over 130,000 for all of Unicode | 4.9 times longer sequences than tokens |
| Words (split on spaces) | 42,855 | one entry per distinct word, never complete | any new word, name or typo is unknown |
| Tokens (GPT-4's `cl100k_base`) | 52,983 | about 100,000 | the tradeoffs this article is about |

Characters are tempting because nothing is ever unknown. But every token is one trip through the whole model (Part 1 covered that loop), and attention cost grows with sequence length. Making every sequence about five times longer makes every request much more expensive.

Whole words go wrong the other way. A word list that covers English, plus names, misspellings, code and every other language, never ends. Any word missing from the list becomes an "unknown" placeholder, and the model can't read or write it.

Subword tokens sit in the middle. Frequent words get their own token, and rare words are built from a few pieces. In that measurement, GPT-4's tokenizer needed 1.24 tokens per English word, or 4.85 characters per token. That matches [OpenAI's rule of thumb](https://developers.openai.com/api/docs/concepts) of about 4 characters, or three-quarters of a word, per token.

## Step 2: building a vocabulary with byte-pair encoding

Byte-pair encoding started as a data compression trick from 1994. [Sennrich, Haddow and Birch](https://arxiv.org/abs/1508.07909) adapted it for neural translation in 2015, and GPT-2, GPT-4 and the Llama models all use a version of it. The training loop is short:

1. Split every word in the training text into single characters.
2. Count every pair of neighboring symbols, weighted by how often each word appears.
3. Merge the most frequent pair into one new symbol and record that merge.
4. Repeat until the vocabulary is as big as you want.

The paper's own toy corpus shows it well. It has four words with these counts: "low" 5 times, "lower" 2, "newest" 6 and "widest" 3. An underscore `_` marks the end of each word, so the tokenizer can tell "est" at the end of a word from "est" in the middle. That gives 11 starting symbols (`l o w e r n s t i d _`) and 95 symbols of text in total.

Running the paper's Python code for ten merges:

| Merge | Pair merged | Times seen | New symbol | Symbols left in corpus |
|---|---|---|---|---|
| start | | | | 95 |
| 1 | `e` + `s` | 9 (6 newest + 3 widest) | `es` | 86 |
| 2 | `es` + `t` | 9 | `est` | 77 |
| 3 | `est` + `_` | 9 | `est_` | 68 |
| 4 | `l` + `o` | 7 (5 low + 2 lower) | `lo` | 61 |
| 5 | `lo` + `w` | 7 | `low` | 54 |
| 6 | `n` + `e` | 6 | `ne` | 48 |
| 7 | `ne` + `w` | 6 | `new` | 42 |
| 8 | `new` + `est_` | 6 | `newest_` | 36 |
| 9 | `low` + `_` | 5 | `low_` | 31 |
| 10 | `w` + `i` | 3 | `wi` | 28 |

Each merge saves exactly as many symbols as the pair's count: merging `e` + `s`, seen 9 times, takes the corpus from 95 symbols to 86. After ten merges the same text is 28 symbols long, 3.4 times shorter, and the vocabulary has grown from 11 entries to 21. Several pairs tie at the top along the way (`e s`, `s t` and `t _` all start at 9); this implementation simply takes the first one it counted.

In general, `vocabulary size = starting symbols + number of merges`. You choose the vocabulary size, and that decides how many merges to learn.

Encoding new text replays the merges in the order they were learned. Take "lowest", which never appeared in the corpus. It starts as `l o w e s t _`. Merges 1 to 3 turn `e s t _` into `est_`, merges 4 and 5 turn `l o w` into `low`, and the result is two tokens: `low` + `est_`. The tokenizer has never seen the word, but it can still write it from known parts.

![Two stacked stages. Training, done once on a large corpus: raw text is split into bytes giving 256 starting tokens, neighboring token pairs are counted, the top pair is merged into a new token, repeating until the vocabulary is full, producing a vocabulary and merge list (GPT-2: 50,257 tokens). Encoding, done on every request: your text is pre-split into words, numbers and punctuation, the merges are applied in learned order, and token IDs are sent to the model.](/diagrams/f02-bpe-pipeline.svg "A tokenizer is trained once. Every request after that replays its merge list.")

Training the tokenizer happens once and is separate from training the model. After that, the merge list is frozen, and every piece of text the model ever reads goes through the same fixed rules.

## Step 3: the tokenizers real models use

### Byte-level BPE: GPT-2 and tiktoken

Starting from characters has a catch: Unicode has a huge number of them. The [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) notes that a character-level BPE would need "a base vocabulary of over 130,000 before any multi-symbol tokens are added." Its fix was to run BPE on bytes instead. Every piece of text is stored as UTF-8 bytes, and there are only 256 possible byte values. So the base vocabulary is 256, and no input can ever be unknown.

GPT-2's vocabulary size follows from this: 256 byte tokens, plus 50,000 learned merges, plus one special `<|endoftext|>` token makes 50,257.

The GPT-2 team found one more problem. Plain BPE wasted vocabulary slots on variants like "dog.", "dog!" and "dog?". So before merging, the text is pre-split with a pattern that keeps letters, digits and punctuation apart, and attaches a leading space to the word that follows it. That is why most English words are tokens that start with a space, and why "strawberry" and " strawberry" are different tokens.

OpenAI's [tiktoken](https://github.com/openai/tiktoken) library is a fast implementation of these byte-level tokenizers. It ships the encodings used by OpenAI's models:

| Encoding | Used by | Regular tokens |
|---|---|---|
| `gpt2` | GPT-2 | 50,256 + 1 special |
| `cl100k_base` | GPT-3.5 Turbo, GPT-4 | 100,256 |
| `o200k_base` | GPT-4o, o1, GPT-5 | 199,998 |

Model mappings are from tiktoken's [model table](https://github.com/openai/tiktoken/blob/main/tiktoken/model.py); token counts are from the library itself (version 0.14.0). As a check, the token IDs shown for "The capital of France is" in Part 1 (791, 6864, 315, 9822, 374) are exactly what `cl100k_base` produces.

Bytes guarantee coverage, but they can be expensive. The Hindi greeting "नमस्ते" (namaste) is 6 Unicode characters and 18 UTF-8 bytes, because each Devanagari character takes 3 bytes. GPT-2's tokenizer splits it into 12 tokens, many of them meaningless fragments of a single character. `cl100k_base` needs 6, and `o200k_base` needs 4.

### SentencePiece and the unigram model

Google's [SentencePiece](https://arxiv.org/abs/1808.06226) (Kudo and Richardson, 2018) takes a different starting point. It treats the input as a raw stream of characters, spaces included, and marks each space with a visible symbol `▁`, so "Hello world." can become `▁He ll o ▁world .` and decode back exactly. It doesn't need a language-specific word splitter, which matters for languages like Japanese and Chinese that don't put spaces between words.

SentencePiece can train BPE, or a second algorithm called the [unigram language model](https://arxiv.org/abs/1804.10959). Unigram runs BPE in reverse. It starts from a big vocabulary of candidate pieces, estimates how likely each one is, and repeatedly drops the pieces whose removal hurts the overall fit least (the paper's example keeps the top 80% each round), always keeping single characters so nothing becomes unknown. Because it is a probability model, it can also produce several possible splits of the same word with their probabilities.

Real models mix and match:

- Llama 2 used SentencePiece BPE with a [32,000-token vocabulary](https://arxiv.org/abs/2307.09288), split every number into single digits, and fell back to bytes for unknown characters.
- Llama 3 switched to a [128,000-token vocabulary](https://arxiv.org/abs/2407.21783): 100,000 tokens from tiktoken plus 28,000 extra tokens for non-English languages. On English text it fits 3.94 characters per token, up from 3.17 with Llama 2's tokenizer, so the same text takes about 20% fewer tokens.
- Gemma 3 uses Gemini 2.0's SentencePiece tokenizer, with split digits and byte fallback, and a vocabulary of [262,000 entries](https://arxiv.org/abs/2503.19786) that its report describes as "more balanced for non-English languages".

### Special tokens and chat templates

A vocabulary also holds a few special tokens that never come from ordinary text. They act as control signals. GPT-2 has just one, `<|endoftext|>` (ID 50256), which separates documents in training and tells the model when to stop.

Chat models need more structure: who is speaking, where each message ends, where the assistant's reply begins. Llama 3 [formats a conversation](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/meta-llama-3/) like this before tokenizing it:

```text
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Each `<|...|>` marker is a special token. `<|eot_id|>` closes each message, and the prompt ends right after the assistant header, so the model's next tokens become its answer. This layout is called a chat template, and each model family defines its own. Part 13 covers how models are trained on these conversations.

## Step 4: what tokens do to spelling, math and languages

### Spelling and counting letters

Here is how three OpenAI tokenizers split the word, computed with tiktoken:

| Text | `gpt2` | `cl100k_base` | `o200k_base` |
|---|---|---|---|
| `strawberry` (start of text) | `st` `raw` `berry` | `str` `aw` `berry` | `st` `raw` `berry` |
| ` strawberry` (after a space) | ` strawberry` | ` strawberry` | ` strawberry` |
| `Strawberry` | `St` `raw` `berry` | `Str` `aw` `berry` | `Str` `aw` `berry` |

In a normal question like "How many r's are in strawberry?", every one of these tokenizers turns the word into one token. The letters are nowhere in the input. The model can only answer if it has learned, from patterns in training text, which letters that token contains, and then counted them correctly.

Tokenization is not the whole story, though. The [CUTE benchmark](https://arxiv.org/abs/2409.15452) found that most models "seem to know the spelling of their tokens, yet fail to use this information effectively to manipulate text". And a [study of letter counting](https://arxiv.org/abs/2412.18626) across many models found that errors tracked how hard the counting was, especially letters that appear more than twice, more than how common the word or its tokens were. So tokens hide the letters, and counting them is a separate weakness on top of that.

### Arithmetic

Numbers get chopped up too. `cl100k_base` and `o200k_base` split digit runs into chunks of up to three, starting from the left:

```text
8302080 + 3529456 = 11831536
[830][208][0]  +  [352][945][6]  =  [118][315][36]
```

The chunks don't line up with place value. The last token of the first number is the units digit `0`, while the last token of the answer is `36`, the tens and units together. To add these numbers the model has to realign digits that the tokenizer scattered across tokens of different sizes.

[Singh and Strouse](https://arxiv.org/abs/2402.14903) tested exactly this. Writing the numbers with commas (`8,302,080`) makes the tokenizer split them into aligned groups of three from the right. On addition problems with 7- to 9-digit numbers and 8 worked examples in the prompt, that raised accuracy from 75.6% to 97.8% for GPT-3.5 and from 84.4% to 98.9% for GPT-4. The models and the numbers were the same; only the tokens changed. Llama 2 and Gemma 3 sidestep part of the problem by splitting every number into single digits.

The decimals in "9.11 > 9.9" show a related quirk. All three tokenizers split the two numbers into `9` `.` `11` and `9` `.` `9`. To the model, "11" is one token and "9" is another, so the shorter-looking number has the bigger-looking token after the decimal point.

### Languages

Tokenizers learn their merges from their training data. When that data is mostly English, as GPT-2's web text was, other languages get fewer merges, and their text breaks into more, smaller pieces.

[Petrov et al.](https://arxiv.org/abs/2305.15425) measured this on FLORES-200 and called the ratio the tokenization premium: how many tokens a language needs for the same content, relative to English. They found differences "up to 15 times in some cases". We reran their measurement on the same 2,009 sentences, adding OpenAI's newer `o200k_base`:

| Language | `gpt2` | `cl100k_base` (GPT-4) | `o200k_base` (GPT-4o and later) |
|---|---|---|---|
| English | 1.00 | 1.00 | 1.00 |
| Portuguese | 1.93 | 1.48 | 1.23 |
| German | 2.14 | 1.58 | 1.31 |
| Chinese (Simplified) | 3.17 | 1.88 | 1.26 |
| Russian | 5.72 | 2.48 | 1.43 |
| Arabic | 4.36 | 3.03 | 1.38 |
| Hindi | 7.43 | 4.78 | 1.58 |
| Tamil | 15.48 | 7.63 | 1.98 |
| Burmese | 16.82 | 11.64 | 3.17 |
| Shan | 18.68 | 14.99 | 7.92 |

Our `cl100k_base` column matches the paper's published figures to within about 1% (Hindi 4.79, Arabic 3.04, Shan 15.05 in the paper). The `o200k_base` column shows how much a bigger, more multilingual vocabulary helps: Hindi drops from 4.78 times English to 1.58. Shan, a language of Myanmar's Shan State, still needs almost eight times as many tokens.

## In the real world

Token counts decide what you pay, how much fits in the context window and how long a reply takes:

- Suppose a provider charges $2.50 per million input tokens (an illustrative price). One million words of English content in `cl100k_base` is about 1.24 million tokens, or $3.09. The same content in Hindi is 4.78 times as many tokens, about $14.76. With `o200k_base`, the Hindi bill falls to about $4.82. Petrov et al. found that with GPT-4's tokenizer, German or Italian cost about 50% more than English, and Dzongkha, Odia, Santali or Shan more than 12 times more.
- A 128,000-token window holds about 103,500 English words with `cl100k_base`. Filled with Hindi, it holds the content of only about 21,700 English words. Long documents, retrieved passages and chat history all run out sooner.
- Generation runs one token at a time, so a reply that needs 4.78 times more tokens takes roughly 4.78 times as many steps to write.
- Model builders treat token efficiency as a design goal. Llama 3's tokenizer change alone made English text about 20% shorter (3.17 to 3.94 characters per token), which, as Meta's report puts it, lets the model "read" more text for the same training compute.
- Vocabulary sizes range from 32,000 (Llama 2) and 50,257 (GPT-2) up to 128,000 (Llama 3), about 200,000 (GPT-4o) and 262,000 (Gemma 3). Bigger vocabularies mean shorter sequences, but every token needs its own row in the model's embedding table and its own score at the output. Part 4 looks at those two tables.
- Since prices and limits are in tokens, applications count them before sending a request. For OpenAI models, `tiktoken` gives the exact count locally: `len(tiktoken.get_encoding("o200k_base").encode(text))`.

## Common misconceptions

- **"One token is one word."** Common English words often are one token, with the leading space attached. Rare words, numbers, code and most other languages take several. In our FLORES measurement, English averaged 1.24 tokens per word with `cl100k_base`; Hindi took 4.78 times as many tokens for the same content.
- **"Token counts are the same across models."** Every model family has its own tokenizer. The same Hindi sentence was 475 tokens with GPT-2's tokenizer, 304 with GPT-4's and 94 with GPT-4o's. Compare prices and context limits per word of your actual text, not per token.
- **"Models can't spell because of tokenization, full stop."** Tokens hide letters, but research such as CUTE shows models often do know how their tokens are spelled. They fail more at using that knowledge, and at counting in general. Written with spaces, as "s t r a w b e r r y", the word becomes 10 tokens, one per letter, and the letters are finally visible to the model.
- **"The tokenizer learns along with the model."** It is trained separately, before the model, and then frozen. The model's embedding table has one row per token ID, so swapping in a different tokenizer would scramble everything the model learned. Llama 3's new tokenizer came with a newly trained model.
- **"Unknown words break the model."** With byte-level BPE nothing is unknown. Any string, including emoji, typos and scripts the tokenizer barely saw, falls back to bytes. It may cost many tokens, but it always encodes, and it decodes back exactly.

## Key takeaways

- A tokenizer turns text into token IDs before the model sees it. It is trained once, separately, and then frozen.
- Byte-pair encoding builds the vocabulary by repeatedly merging the most frequent neighboring pair. Starting from 256 bytes means no text is ever unknown.
- Vocabulary sizes have grown from 50,257 in GPT-2 to about 200,000 in GPT-4o and 262,000 in Gemma 3, partly to handle more languages efficiently.
- Models see token IDs, not letters or digits. That makes spelling, letter counting and multi-digit arithmetic harder than they look, and formatting can change accuracy: commas in numbers lifted GPT-4's addition accuracy from 84.4% to 98.9%.
- The same sentence can take many times more tokens in one language than another: 4.78 times more for Hindi than English with GPT-4's tokenizer, 1.58 times with GPT-4o's. Tokens are what you pay for and what fills the context window.

## What's next

Tokens are now numbers, and the model is a huge pile of other numbers that turns them into predictions. [Part 3](/blog/neural-networks-basics/) explains how that pile learns: what a neuron and a weight are, how gradient descent nudges millions of weights toward better answers, and how backpropagation works out which way to nudge each one.

## Further reading

- [Neural Machine Translation of Rare Words with Subword Units (Sennrich, Haddow and Birch, 2016)](https://arxiv.org/abs/1508.07909): the paper that brought byte-pair encoding to neural language models, with the toy example worked above.
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2, which introduced byte-level BPE and the 50,257-token vocabulary.
- [SentencePiece (Kudo and Richardson, 2018)](https://arxiv.org/abs/1808.06226): a language-independent tokenizer that treats spaces as ordinary symbols.
- [Subword Regularization (Kudo, 2018)](https://arxiv.org/abs/1804.10959): the unigram language model tokenizer.
- [Language Model Tokenizers Introduce Unfairness Between Languages (Petrov et al., 2023)](https://arxiv.org/abs/2305.15425): the tokenization premium measured across 200 languages.
- [Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs (Singh and Strouse, 2024)](https://arxiv.org/abs/2402.14903): how the way numbers are split changes GPT-3.5 and GPT-4's addition accuracy.
- [CUTE: Measuring LLMs' Understanding of Their Tokens (Edman, Schmid and Fraser, 2024)](https://arxiv.org/abs/2409.15452): what models know about the letters inside their tokens.
- [The Llama 3 Herd of Models (Llama Team, Meta, 2024)](https://arxiv.org/abs/2407.21783): the move to a 128K vocabulary and its effect on compression.
- [tiktoken (OpenAI)](https://github.com/openai/tiktoken): the tokenizer library for OpenAI models; useful for counting tokens yourself.
- [Byte-Pair Encoding tokenization (Hugging Face LLM Course)](https://huggingface.co/learn/llm-course/chapter6/5): a step-by-step walkthrough with code.
- [Let's build the GPT Tokenizer (Andrej Karpathy, 2024)](https://www.youtube.com/watch?v=zduSFxRajkE): a two-hour build of a GPT-style tokenizer from scratch, with its many quirks.
