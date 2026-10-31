---
title: "Training a BPE Tokenizer: 40 Lines of Python to 32K Tokens"
description: "Build byte-level BPE from scratch, train the series' 32,768-token tokenizer on FineWeb-Edu, and measure what vocabulary size and training data change."
slug: "train-bpe-tokenizer"
category: "llm-training"
tags: ["tokenization", "tokens", "pretraining", "training"]
series: "llm-from-scratch"
seriesOrder: 2
pubDate: 2026-06-18
---

## TL;DR

This part builds the tokenizer the rest of the series uses. You'll write byte-level byte-pair encoding (BPE) in about 40 lines of Python, watch it learn its first merges on one sentence, then train the real thing: a 32,768-token vocabulary learned from 406 MB of FineWeb-Edu in 42 seconds with Hugging Face's `tokenizers` library, a job the pure-Python trainer would need an estimated 14 hours for. On held-out text it averages 4.61 bytes per token, against 4.06 for Llama 2's tokenizer of almost the same size, because ours saw only the kind of text the model will train on. Measurements of five vocabulary sizes, eight real tokenizers and the tokens the model will barely ever see explain why 32K is a reasonable choice for a ~100M-parameter model and what it gives up.

## Why it matters

Llama 2's tokenizer has 32,000 tokens. The one built in this article has 32,768. On the same 26 MB of held-out educational web text, Llama 2's needs 6.42 million tokens and ours needs 5.66 million, 12% fewer.

The difference is the training text. Llama 2's tokenizer had to cover many languages and code; ours saw only English educational pages, the same kind of text the model will train on. Every token saved is training compute saved, and more text fits in each 1,024-token context window.

The tokenizer is also the one part you can't fix later. The model's embedding table has one row per token ID, so a new tokenizer means a new model. [Fundamentals Part 2](/blog/tokenization/) explained in words how BPE builds a vocabulary and why tokens trip models up on spelling, arithmetic and other languages. This part writes the code, trains the tokenizer and measures the choices.

## The intuition

Picture a court stenographer inventing shorthand. Each evening they look over the day's transcripts, find the pair of symbols they wrote side by side most often, and invent one new symbol for that pair. After a month of medical trials they have one-stroke symbols for "patient", "diagnosis" and "the". Send them to a patent hearing about software and they're back to spelling words out. BPE works the same way: frequent neighbors become one symbol, and what counts as frequent depends on the text it was built from.

The analogy breaks in two places. A stenographer would drop symbols they stopped needing; BPE keeps every intermediate symbol, even after a longer one has swallowed it, and some of those leftovers become tokens the model almost never sees. And BPE's fallback is lower than the alphabet: raw bytes, so it can write anything, including emoji and Hindi, at a cost of several tokens per character.

## Step 1: BPE on bytes, by hand

A byte-level tokenizer starts from UTF-8 bytes, the standard encoding that turns every character into one to four numbers from 0 to 255. English letters take one byte each; the Hindi letter न takes three. Starting from bytes makes the base vocabulary exactly 256 tokens, and no input can ever be unknown.

Take the sentence `the cat sat on the mat. the cat ate the rat.`: 44 bytes, 12 distinct values. Training repeats one step: count every pair of neighboring tokens and replace the most frequent pair with a new token. The first six merges, computed with the trainer in Step 2:

| Merge | Pair | Times seen | New token | Tokens left |
|---|---|---|---|---|
| start | | | | 44 |
| 1 | `a` + `t` | 6 | `at` (id 256) | 38 |
| 2 | `t` + `h` | 4 | `th` (257) | 34 |
| 3 | `th` + `e` | 4 | `the` (258) | 30 |
| 4 | ` ` + `the` | 3 | ` the` (259) | 27 |
| 5 | ` ` + `c` | 2 | ` c` (260) | 25 |
| 6 | ` c` + `at` | 2 | ` cat` (261) | 23 |

Merge 1 finds `at` six times: cat (twice), sat, mat, ate and rat. Each merge shortens the text by exactly the pair's count. Merge 2 was a tie, since `t h` and `h e` both appear 4 times; the trainer takes whichever it counted first. Ties like this are why two BPE implementations trained on the same text can disagree on a few merges.

### Why split the text before merging

No merge above joins the end of one word to the start of the next, and BPE alone wouldn't prevent that. Before counting, the trainer cut the sentence into pieces with a regular expression (`the`, ` cat`, ` sat`, ` on`, ...) and counted pairs only inside a piece.

Run the same six merges on the raw sentence and merge 2 becomes `e` + space (5 times), merge 4 `the ` and merge 6 `the c`, a token that only makes sense in "the cat". The [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) saw the same waste at scale ("dog." "dog!" "dog?" as separate tokens) and prevented BPE "from merging across character categories for any byte sequence", with an exception for spaces.

That rule is the pre-tokenizer. GPT-2's, in OpenAI's [encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py), is:

```python
r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

Contractions like `'s` and `'ll` are pieces of their own; then come an optional space plus a run of letters (`\p{L}` is any Unicode letter), an optional space plus a run of digits (`\p{N}`), an optional space plus punctuation, and whitespace. The optional leading space is why English words become tokens like ` cat`, and why `cat` at the start of a line is a different token. A comment above the line admits a bug: "Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions."

GPT-4's `cl100k_base`, in [tiktoken's source](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py), made contractions case-insensitive, let one non-letter such as an opening quote join the next word, and capped digit runs at three. Llama 3 uses the same pattern. This series uses it with single digits:

```python
PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```

With `\p{N}` instead of `\p{N}{1,3}`, "1969" is four tokens, 1, 9, 6 and 9, and each digit is the same token in every number. Llama 2 split digits the same way, and this exact pattern is the one in Qwen2.5's `tokenizer.json`.

## Step 2: the trainer in 40 lines

Here is the core of `tokenizer.py`, the series' readable reference implementation. `train_bpe` runs the regex once, counts how often each distinct piece occurs, and merges on the counted pieces. ` the` occurs about 3.4 million times in the training sample, but its pairs are counted once and weighted.

```python
from collections import Counter
import regex as re
from tokenizer import PATTERN    # the regex above

def pair_counts(chunks):
    """Count adjacent id pairs; each chunk is weighted by how often it occurs."""
    counts = Counter()
    for ids, n in chunks.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += n
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def train_bpe(texts, n_merges, verbose=False):
    """Learn `n_merges` byte-pair merges. Returns merges {(id, id): new_id} and vocab {id: bytes}."""
    # 1. Pre-tokenize, and count identical pieces once ("the" may occur a million times).
    chunks = Counter()
    for text in texts:
        for piece in re.findall(PATTERN, text):
            chunks[tuple(piece.encode("utf-8"))] += 1
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    # 2. Merge the most frequent pair, n_merges times.
    for new_id in range(256, 256 + n_merges):
        counts = pair_counts(chunks)
        if not counts:
            break
        pair = max(counts, key=counts.get)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        chunks = {merge(ids, pair, new_id): n for ids, n in chunks.items()}
        if verbose:
            print(new_id, vocab[new_id], counts[pair])
    return merges, vocab
```

The rest of the file wraps this in a `BPETokenizer` class with special tokens, `encode` and `decode`. Encoding replays history: for each piece, apply whichever neighboring pair was merged earliest in training, and repeat until no learned pair remains. Decoding concatenates each token's bytes.

```python
from tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(["the cat sat on the mat. the cat ate the rat."], vocab_size=256 + 6 + 8)
ids = tok.encode("the cat ate the hat.<|endoftext|>")
print(ids)                           # [258, 261, 32, 256, 101, 259, 32, 104, 256, 46, 262]
print([tok.decode([i]) for i in ids])
# ['the', ' cat', ' ', 'at', 'e', ' the', ' ', 'h', 'at', '.', '<|endoftext|>']
```

The vocabulary is 256 bytes, 6 merges and 8 special tokens. "hat" never appeared in training, so it comes out as `h` + `at`: nothing is unknown, just more expensive. Byte-level BPE gets this for free because every byte is a token. SentencePiece tokenizers such as Llama 2's start from characters and add byte fallback: a character missing from the vocabulary is spelled out as its UTF-8 bytes.

### How slow is readable?

Each merge here recounts every pair in every distinct piece and rebuilds every piece. Hugging Face's `tokenizers` runs the same algorithm in Rust with two tricks visible in its [trainer source](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/trainer.rs): a priority queue of pair counts, and an index from each pair to the words that contain it (`where_to_update`), so a merge touches only the words it changes. It also counts on all CPU threads.

Same 10 MB slice of FineWeb-Edu (1,926 documents), same 4,096-token target, on this laptop's CPU (16 logical cores; other jobs were running):

| Trainer | Threads | Training time | Speed-up |
|---|---|---|---|
| `tokenizer.py`, pure Python | 1 | 636.6 s | 1× |
| Hugging Face `tokenizers` | 1 | 2.4 s | 269× |
| Hugging Face `tokenizers` | 16 | 4.0 s | 160× |

The Python time is one run and the Rust times are medians of three; on a job this small, 16 threads were slower than one. The two trainers learned 3,827 of the same 3,832 tokens, though their order drifts apart from merge 311 on, where ties are broken differently.

Training time grows with the number of distinct pieces: 92,664 in the 10 MB slice, 826,732 in the full 406 MB, whose distinct pieces hold 9.3 times as many bytes. Scaling the 0.17 seconds per merge by that factor and by 32,504 merges gives roughly 14 hours for the full job. That run wasn't attempted, which is why the series trains its real tokenizer with `tokenizers` and keeps `tokenizer.py` as the version you read.

Encoding is much closer. Loaded with the production tokenizer's 32,504 merges (`python_from_production()` in `tokenizer.py`), the Python encoder produced exactly the same tokens as `tokenizers` on all 5,500 held-out documents, and on one thread it was faster: 4.9 MB/s against 3.2 MB/s. It caches every piece it encodes, most pieces repeat, and the regex runs in C inside the `regex` module. `tokenizers` pulls ahead with threads, at 14.5 MB/s on 16 of them, and OpenAI's `tiktoken` encodes GPT-4's `cl100k_base` at 12.5 MB/s on one thread and 45 MB/s on 16.

## Step 3: training the series tokenizer

### The data

FineWeb-Edu is Hugging Face's 1.3-trillion-token set of educational web pages. Per its [dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), Llama3-70B-Instruct scored 500,000 pages for educational value from 0 to 5, a classifier trained on those scores rated the whole FineWeb crawl, and pages under 3 were dropped, 92% of the total. It's released under the Open Data Commons Attribution License (ODC-By) v1.0, subject also to Common Crawl's terms of use.

A tokenizer only needs a sample. `train_tokenizer.py` streams FineWeb-Edu with `datasets` (`streaming=True`) and takes the first 800 documents of each of its 110 Common Crawl snapshots, 2013 to 2025: 750 for training and 50 held out. That gives 82,500 training documents (406 MB) and 5,500 held-out documents (26.1 MB). The download took 5 minutes 49 seconds.

### Special tokens

Special tokens are IDs reserved for control signals; ordinary text never produces them. Eight are reserved before training:

| ID | Token | Purpose |
|---|---|---|
| 0 | `<\|endoftext\|>` | Document boundary in pretraining data (Part 3) |
| 1 | `<\|system\|>` | Chat: start of a system message (Part 10) |
| 2 | `<\|user\|>` | Chat: start of a user message |
| 3 | `<\|assistant\|>` | Chat: start of the model's reply |
| 4 | `<\|end\|>` | Chat: end of a message |
| 5-7 | `<\|reserved_0\|>` to `<\|reserved_2\|>` | Spares, so a later part can add one without resizing the model |

The chat tokens are reserved now because the vocabulary size is fixed into the embedding table in Part 4. Their embeddings stay untrained through pretraining and are learned in Part 10.

### The code

```python
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizer import PATTERN, SPECIAL_TOKENS
from train_tokenizer import read_docs       # streams the 406 MB sample, one document at a time

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=32_768,
    special_tokens=SPECIAL_TOKENS,                         # ids 0..7
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, ids 8..263
)
tok.train_from_iterator(read_docs("train"), trainer)       # 32,504 merges, ids 264..32767
tok.save("tokenizer-32k.json")
```

`ByteLevel` maps each byte to a printable character, GPT-2's trick for keeping spaces visible in the saved vocabulary, which is why ` cat` prints as `Ġcat`. Training took 41.8 seconds and peaked at 1.13 GB of memory. The first merges were `Ġ`+`t`, `Ġ`+`a`, `h`+`e`, `i`+`n`, `r`+`e`, `o`+`n` and `Ġt`+`he`. The last ten are whole words such as ` abbreviations`, ` homeschooling` and ` summoned`, each occurring 208 times in the encoded training sample: roughly how common a pair had to be to win a slot at the end.

![A top-to-bottom flowchart of encoding with the series tokenizer. The text 'The cat sat.' followed by the end-of-text token goes through five steps: special tokens are split out (end-of-text becomes id 0); the regex pre-tokenizes the rest into 'The', space-cat, space-sat and a period; each piece becomes UTF-8 bytes (space-cat is 32 99 97 116); learned merges from a table of 32,504 are applied earliest first (a plus t, then space plus c, then space-c plus at); each token is looked up, giving ids 469 2289 2472 21 0 for the model.](/diagrams/t02-encode-pipeline.svg "Encoding one sentence with the 32K tokenizer. The merge table is the only thing training produced.")

In this tokenizer, ` cat` takes three merges: `a`+`t` is merge 9, `Ġ`+`c` merge 13 and `Ġc`+`at` merge 2,026.

### Vocabulary size: compression

More tokens means each covers more text, so sequences get shorter. But every token also needs a row in the embedding table. The same setup, trained at five sizes on the same 406 MB and measured on the 5,500 held-out documents:

| Vocabulary | Training time | Bytes per token | Tokens per word | Vocabulary used on held-out text |
|---|---|---|---|---|
| 4,096 | 20.3 s | 3.48 | 1.78 | 98.2% |
| 8,192 | 24.1 s | 3.93 | 1.57 | 98.9% |
| 16,384 | 32.1 s | 4.32 | 1.43 | 99.1% |
| 32,768 | 41.8 s | 4.61 | 1.34 | 98.6% |
| 65,536 | 64.2 s | 4.80 | 1.29 | 94.2% |

Bytes per token is UTF-8 bytes divided by tokens; tokens per word divides by whitespace-separated words. Each doubling buys less: 4K to 8K shortens the text by 11.5%, 16K to 32K by 6.3%, 32K to 64K by 4.0%.

![Line chart of bytes per token on held-out FineWeb-Edu against vocabulary size on a log scale. The tokenizers trained here rise from 3.48 at 4K to 3.93 at 8K, 4.32 at 16K, 4.61 at 32K and 4.80 at 64K, flattening with each doubling. Published tokenizers on the same text are hollow points: Llama 2 at 32K scores 4.06, well below ours at the same size; SmolLM2 at 49K scores 4.61 and GPT-2 at 50K 4.66; GPT-4's cl100k and Llama 3 about 4.79; Qwen2.5 at 152K 4.70; GPT-4o's o200k at 200K 4.85.](/diagrams/t02-vocab-compression.svg "Compression on held-out FineWeb-Edu. Filled points: tokenizers trained here. Hollow points: published tokenizers on the same text.")

On this text, our 32K tokenizer matches SmolLM2's 49K, comes within 1% of GPT-2's 50K and within 5% of GPT-4o's 200K.

### Vocabulary size: parameters and compute

Part 4 writes the model. The numbers below assume a model width (`d_model`, the length of each token's vector) of 768 and tied embeddings, one table serving as both the input embedding and the output layer, as in Part 1's prototypes. The table holds `vocab × d_model` numbers. Two bodies bracket Part 1's candidate sizes: 10 layers (62.9M parameters outside the embedding) and 16 layers (100.7M).

| Vocabulary | Embedding parameters | Share of 10-layer model | Share of 16-layer model | Training FLOPs per byte, 10 layers |
|---|---|---|---|---|
| 4,096 | 3.1M | 4.8% | 3.0% | 114M |
| 8,192 | 6.3M | 9.1% | 5.9% | 106M |
| 16,384 | 12.6M | 16.7% | 11.1% | 105M |
| 32,768 | 25.2M | 28.6% | 20.0% | 115M |
| 65,536 | 50.3M | 44.4% | 33.3% | 142M |

The last column counts training FLOPs (floating-point operations) per byte of text. It uses the 6N rule from [Math Part 15](/blog/counting-flops-scaling-laws/), about 6 FLOPs per parameter per training token, so `6 × (body + vocab × d_model)` per token with the output layer counted, divided by bytes per token. It leaves out attention's score computation, which doesn't depend on the vocabulary. A bigger vocabulary makes each token costlier and the text shorter, and between 8K and 32K the two nearly cancel.

### Why 32K anyway

On this arithmetic, 16K is cheapest: 32K costs 9% more compute per byte for the 10-layer model and 4% more for the 16-layer one. Research agrees that small models want small vocabularies. [Tao et al. (2024)](https://arxiv.org/abs/2407.13623) found the compute-optimal vocabulary grows more slowly than the rest of the model; for 302M non-embedding parameters their experiments put it near 16K, rising to 24K when training on more data than compute-optimal. At the other end, they estimate Llama 2 70B should have had at least 216K instead of 32K.

The series takes 32,768 anyway. The FLOP count misses four things:

- At 4.61 bytes per token, a 1,024-token context window holds about 4,700 bytes, against 4,400 at 16K: 7% more document per window.
- Part 1's training budget is counted in tokens, and a fixed number of tokens covers 7% more text at 32K.
- 32,768 is 2¹⁵, so the output layer's matrix multiply has a dimension divisible by 64. NVIDIA's [performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html) gives an example vocabulary of 33,708 that can't use tensor cores (the matrix units from [GPU Part 6](/blog/tensor-cores/)) until four padding tokens make it 33,712, a multiple of 8.
- It follows precedent: Llama 2 used 32,000 tokens for models from 7B to 70B parameters, and GPT-2 used 50,257 for every size, down to its smallest.

None of these is decisive, and the compute gap is under 10% either way. Part 9 measures bits per byte, a score that doesn't depend on the tokenizer, so models with different vocabularies can still be compared fairly.

## Step 4: quirks that matter for training and evals

### Leading spaces and case

`the` is token 1073, ` the` 270, `The` 469 and ` The` 370: four unrelated embedding rows for one word. ` Paris` is one token, but `Paris` at the start of a line is `Par` + `is`. The model must learn these are related, and evals must match the training text: scoring the answer `"Paris"` instead of `" Paris"` compares different tokens with different probabilities.

### The space before a number

The regex attaches a space to letters and punctuation but not to digits, so in `In 1969` the space becomes a token of its own. "In 1969, 600,000,000 people watched." is 22 tokens: 2 lone spaces, 13 digits, 4 punctuation marks and 3 words. GPT-4's `cl100k_base` also leaves that lone space. It's harmless, but it's one more token to predict before every number.

### Tokens the model will barely see

Encoding the whole 406 MB training sample gives 88.5 million tokens. How often each of the 32,760 regular tokens appears:

| Times seen in 88.5M tokens | Number of tokens |
|---|---|
| never | 78 |
| fewer than 10 | 211 |
| fewer than 100 | 1,341 |
| fewer than 1,000 | 23,895 |

52 of the never-seen tokens are single bytes absent from all 406 MB. Thirteen can't occur in valid UTF-8 (0xC0, 0xC1 and 0xF5 to 0xFF), 31 are ASCII control characters, 7 are lead bytes of rarely used Unicode ranges such as private-use areas, and one is the curly bracket `{`. FineWeb kept C4's rule of dropping every document that contains a `{`, as the [FineWeb paper](https://arxiv.org/abs/2406.17557) describes, so the model in this series will never see one in pretraining.

The other 26 are intermediate fragments such as ` ingred`, ` neurotransmit` and ` cryptocur`. Each was frequent enough to be merged, then a longer token (` ingredients`) swallowed nearly every occurrence, so replaying the merges never leaves the fragment on its own.

These are the raw material of glitch tokens. [Rumbelow and Watkins (2023)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) found tokens that broke GPT-2 and GPT-3 models: asked to repeat ` SolidGoldMagikarp` (GPT-2 token 43453), a GPT-3 model answered "distribute". Their partial explanation was that the tokenizer was built from scraped web content while the model's training text was "more heavily curated". [Land and Bartolo (2024)](https://arxiv.org/abs/2405.05417) built detectors for such under-trained tokens and found them across many model families, including unreachable bytes and "intermediate 'junk' tokens" left by BPE merges, the same two kinds counted here. Our risk is smaller because tokenizer and model train on the same dataset, but the rarest rows of the embedding table will stay close to their random start. Part 7's model can be tested on them.

### Special tokens inside ordinary text

By default, `tokenizers` matches special tokens anywhere in the input, so a web page containing the literal text `<|endoftext|>` would get token 0 in the middle, a fake document boundary. The sample contains no such string, but pages about language models can. With `tok.encode_special_tokens = True` the string is encoded as plain text (`<`, `|`, `end`, `of`, `text`, `|`, `>`), and the Part 3 pipeline can insert token 0 itself between documents.

### Round trips

Byte-level BPE should be lossless: `decode(encode(text)) == text`. On all 5,500 held-out documents, Python's standard library and edge cases (emoji with skin-tone modifiers, Hindi, `\r\n`, a NUL byte, a zero-width space, both Unicode forms of é), the 32K tokenizer had zero failures. One trap remains: न is two tokens, and decoding either alone gives `�`, because neither holds a whole UTF-8 character. Code that streams generated text token by token has to buffer bytes until a character completes.

## In the real world

A tokenizer is a compressed summary of its training text. Here are eight tokenizers on four kinds of text. Prose is the FineWeb-Edu held-out set. Code is the 167 top-level `.py` files of Python 3.12's standard library (4.7 MB). Numbers are every digit run in the held-out documents, space-separated. Hindi is the 2,009 FLORES-200 sentences, human translations of the same English sentences; the last column compares Hindi and English token counts.

| Tokenizer | Vocabulary | Prose | Code | Numbers | Hindi | Hindi tokens vs English |
|---|---|---|---|---|---|---|
| Ours | 32,768 | 4.61 | 2.08 | 1.00 | 1.81 | 6.61× |
| Llama 2 | 32,000 | 4.06 | 3.35 | 0.99 | 2.37 | 4.60× |
| SmolLM2 | 49,152 | 4.61 | 3.80 | 1.00 | 2.36 | 5.16× |
| GPT-2 | 50,257 | 4.66 | 2.21 | 2.82 | 1.68 | 7.43× |
| GPT-4 (`cl100k_base`) | 100,277 | 4.79 | 4.28 | 1.49 | 2.61 | 4.78× |
| Llama 3 | 128,256 | 4.79 | 4.28 | 1.49 | 4.92 | 2.53× |
| Qwen2.5 | 151,665 | 4.70 | 4.22 | 1.00 | 2.76 | 4.43× |
| GPT-4o (`o200k_base`) | 200,019 | 4.85 | 4.24 | 1.49 | 7.99 | 1.58× |

Columns 3 to 6 are bytes per token, higher is better. Vocabulary sizes include special tokens and come from each tokenizer's published files (Hugging Face `tokenizer.json`, or `tiktoken` 0.14.0 for OpenAI's).

- Ours is the worst at Python code, because FineWeb-Edu is prose: it never learned merges for runs of spaces, so indentation comes out one space token at a time. SmolLM2's tokenizer is the nearest relative, built for a model family from 135M parameters up and, per the [SmolLM2 paper](https://arxiv.org/abs/2502.02737), trained on 70% FineWeb-Edu plus 13% code and 8% math. It ties ours on prose and is 83% better on code.
- On numbers, tokenizers that split digits (ours, Llama 2, SmolLM2, Qwen2.5) score about one byte per token. GPT-2 merges whole digit runs, so ` 1969` is one token and ` 600` another, depending on what was frequent. GPT-4 and Llama 3 cap runs at three digits.
- For Hindi, ours needs 6.6 times as many tokens as for the English translations. नमस्ते (namaste, 18 bytes) is 10 tokens, mostly fragments of characters. GPT-4o's needs 1.58 times as many. [Fundamentals Part 2](/blog/tokenization/) covers this across 200 languages.
- Llama 2's [32,000 tokens](https://arxiv.org/abs/2307.09288) split digits and use "bytes to decompose unknown UTF-8 characters", but on English prose it's the weakest tokenizer here. Llama 3's [128K](https://arxiv.org/abs/2407.21783) (100K from tiktoken plus 28K for other languages) raised English compression from 3.17 to 3.94 characters per token on Meta's sample; on FineWeb-Edu the gain is 4.06 to 4.79 bytes per token.
- Bigger models want bigger vocabularies: [Tao et al.](https://arxiv.org/abs/2407.13623) raised a 3B-class model's vocabulary from 32K to 43K at the same compute (2.3 × 10²¹ FLOPs) and ARC-Challenge accuracy went from 29.1 to 32.0.

## Common misconceptions

- **"A bigger vocabulary is always better."** Each doubling buys less (6.3% fewer tokens from 16K to 32K, 4.0% from 32K to 64K), while the embedding table doubles. At 64K it would be 44% of a 10-layer, 768-wide model.
- **"Same vocabulary size, same compression."** Ours and Llama 2's differ by 2.4% in size and 12% on English prose, and Llama 2's is 61% better on code. Training data decides what the slots are spent on.
- **"BPE training is slow."** The readable Python is. Compiled, it took 42 seconds for 406 MB on a laptop CPU.
- **"Every token in the vocabulary gets trained."** 78 of ours never appear in the training sample, and 1,341 appear fewer than 100 times. Their embedding rows learn little or nothing.

## Key takeaways

- BPE is a short loop: count neighboring pairs, merge the most frequent, repeat. Starting from 256 bytes means no text is ever unknown.
- The pre-tokenizer regex decides which merges are allowed. It keeps words, numbers and punctuation apart, attaches leading spaces to words and, here, splits numbers into single digits.
- The series tokenizer has 32,768 tokens, trained on 406 MB of FineWeb-Edu in 42 seconds. It compresses held-out English to 4.61 bytes per token, 12% fewer tokens than Llama 2's tokenizer of the same size.
- Vocabulary size trades shorter sequences for a bigger embedding table. For a ~100M-parameter model, 32K takes 20 to 29% of the parameters and costs under 10% more compute per byte than the cheapest size.
- A tokenizer is good at whatever it was trained on. Ours is strong on English prose and weak on code and Hindi, which suits the data this series trains on.

## What's next

The tokenizer is ready, and FineWeb-Edu has 1.3 trillion tokens waiting. Part 3 builds the pretraining data pipeline: downloading a few billion tokens' worth, filtering and deduplicating it, shuffling it, and packing it with this tokenizer into token shards the training loop can read at full speed.

## Further reading

- [Neural Machine Translation of Rare Words with Subword Units (Sennrich, Haddow and Birch, 2016)](https://arxiv.org/abs/1508.07909): the paper that brought BPE to neural language models.
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2, byte-level BPE and the case for pre-tokenization.
- [GPT-2 `encoder.py` (OpenAI)](https://github.com/openai/gpt-2/blob/master/src/encoder.py) and [tiktoken encodings (OpenAI)](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py): the regexes for GPT-2, GPT-4 and GPT-4o.
- [Hugging Face `tokenizers` BPE trainer (Rust source)](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/trainer.rs): the fast algorithm, with incremental pair counts and a priority queue.
- [FineWeb-Edu dataset card (Hugging Face)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [The FineWeb Datasets (Penedo et al., 2024)](https://arxiv.org/abs/2406.17557): how the data was filtered, and its license.
- [Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies (Tao et al., 2024)](https://arxiv.org/abs/2407.13623): the compute-optimal vocabulary size as a function of model size.
- [Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models (Land and Bartolo, 2024)](https://arxiv.org/abs/2405.05417): where glitch tokens come from and how to find them.
- [SolidGoldMagikarp (plus, prompt generation) (Rumbelow and Watkins, 2023)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation): the post that made glitch tokens famous.
- [The Llama 3 Herd of Models (Llama Team, Meta, 2024)](https://arxiv.org/abs/2407.21783) and [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288): two tokenizer generations and their trade-offs.
- [SmolLM2: When Smol Goes Big (Ben Allal et al., 2025)](https://arxiv.org/abs/2502.02737): a small-model family whose tokenizer was trained mostly on FineWeb-Edu.
- [Let's build the GPT Tokenizer (Andrej Karpathy, 2024)](https://www.youtube.com/watch?v=zduSFxRajkE): a two-hour build of a GPT-style tokenizer, with its quirks.
