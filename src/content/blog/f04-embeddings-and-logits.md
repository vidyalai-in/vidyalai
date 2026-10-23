---
title: "Embeddings and Logits: How LLMs Turn Tokens into Vectors and Back"
description: "The table that turns each token into a vector, the dot products that turn a vector back into scores, and why related words point the same way."
slug: "embeddings-and-logits"
category: "llm-fundamentals"
tags: ["embeddings", "transformers", "tokens", "vector-search"]
series: "llm-fundamentals"
seriesOrder: 4
pubDate: 2026-10-23
---

## TL;DR

A language model can't do arithmetic on a token ID, so its first step is to look up each ID in a big table and swap it for a vector, a list of a few hundred to a few thousand numbers. That table is the embedding matrix, and training arranges it so that tokens used in similar ways end up pointing in similar directions. At the other end, the model compares its final vector with one vector per vocabulary token using a dot product, and each comparison gives a score called a logit. Many models use the same table for both ends, a trick called weight tying. By the end you'll be able to compute a similarity, read a row of logits, and see where the famous "king minus man plus woman" example holds up and where it doesn't.

## Why it matters

Nobody ever told GPT-2 that Monday and Tuesday are related. Yet in its embedding table, the vectors for " Monday" and " Tuesday" score 0.89 on a similarity scale where 1.0 means "pointing in exactly the same direction", while " Monday" and " banana" score 0.28. The model worked that out by itself, from nothing but next-token prediction.

That table is also big. In the smallest GPT-2, 38.6 million of its 124 million parameters, 31%, sit in this one lookup table. Every token you send goes in through it, and every token the model writes comes out through a matching set of dot products at the other end.

[Part 3](/blog/neural-networks-basics/) treated the network's inputs and outputs as plain lists of numbers. This part looks at the two ends where words become numbers and numbers become words again.

## The intuition

Think of a map. Every city has coordinates, and cities that are close on the map tend to have things in common. Better still, directions carry meaning: the arrow from France to Paris points roughly the same way as the arrow from Germany to Berlin, because both mean "go to the capital".

An embedding table is a map of the vocabulary. Each token gets coordinates, and training moves tokens around until tokens used in similar contexts sit near each other.

The analogy breaks in three places. A paper map has two dimensions; GPT-2's map has 768 and Llama 3 8B's has 4,096. The axes of a paper map are named (north, east), while the axes of an embedding have no names, and a single one rarely means anything you could label. And nobody draws this map: it's whatever arrangement made next-token predictions best.

## Step 1: the embedding table is a lookup

The embedding matrix has one row per token in the vocabulary and one column per dimension of the model's vectors. Turning a token into a vector means reading its row.

Here is a toy table with a 4-token vocabulary and 3 numbers per vector. The numbers are made up for illustration:

| Token ID | Token | Dim 1 | Dim 2 | Dim 3 |
|---|---|---|---|---|
| 0 | cat | 2 | 3 | 0 |
| 1 | dog | 3 | 2 | 0 |
| 2 | car | 0 | 1 | 3 |
| 3 | truck | 1 | 0 | 3 |

To embed "dog", token 1, the model reads row 1, `[3, 2, 0]`, and does nothing else.

Textbooks often describe it as a matrix multiplication instead. Write the token as a one-hot vector, a list with a 1 in the token's position and 0 everywhere else, so "dog" is `[0, 1, 0, 0]`. Multiply it by the table:

```text
[0, 1, 0, 0] × table
  = 0 × [2, 3, 0] + 1 × [3, 2, 0] + 0 × [0, 1, 3] + 0 × [1, 0, 3]
  = [3, 2, 0]
```

Every row is multiplied by zero except the one you want. Both descriptions give the same answer, and the multiplication view explains how training works: the table's numbers are ordinary parameters, so backpropagation (from Part 3) can nudge them like any other weight. In practice, code skips the multiplication and just copies the row, because multiplying by 50,000 zeros is a waste.

### Real tables are large

A real table's size is `vocabulary size × vector size`. A few examples:

| Model | Vocabulary | Vector size | One table holds | Share of all parameters |
|---|---|---|---|---|
| GPT-2 (smallest) | 50,257 | 768 | 38.6 million | 31% of 124 million |
| GPT-3 175B | 50,257 | 12,288 | 617.6 million | 0.35% |
| Llama 3 8B | 128,256 | 4,096 | 525.3 million | 6.5% (13.1% for its two tables) |
| Llama 3.1 405B | 128,256 | 16,384 | 2.1 billion | 0.5% (1.0% for its two tables) |

Sources: GPT-2's [published weights](https://huggingface.co/openai-community/gpt2) (we counted the parameters ourselves), [GPT-3 paper](https://arxiv.org/abs/2005.14165), and Meta's reference code for [Llama 3 sizes](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py). The "two tables" note is explained in Step 4.

The table grows with the vocabulary times the width, while the rest of the model grows with the width squared times the number of layers (Part 1 showed about `12 × d²` per block). So in small models the table is a big slice, and in giant models it's a rounding error. The 128,256 in Llama 3 is its 128,000-token vocabulary from [Part 2](/blog/tokenization/) plus [256 slots for special tokens](https://github.com/meta-llama/llama-models/blob/main/models/llama3/tokenizer.py).

## Step 2: measuring similarity with dot products

Once tokens are vectors, you can ask how similar two of them are. The basic tool is the dot product: multiply the two vectors position by position and add up the results.

From the toy table:

```text
cat · dog   = 2×3 + 3×2 + 0×0 = 12
cat · car   = 2×0 + 3×1 + 0×3 =  3
car · truck = 0×1 + 1×0 + 3×3 =  9
```

A big positive dot product means the vectors point the same way. Zero means they're at right angles, unrelated. Negative means they point in opposite directions.

The dot product has a catch: it also grows with the vectors' lengths. Make a vector `[4, 6, 0]` that points exactly where "cat" does but is twice as long, and its dot product with "dog" doubles to 24, even though its direction hasn't changed. To compare directions only, divide by both lengths. That's cosine similarity:

```text
cosine(a, b) = (a · b) / (length(a) × length(b))
length(a)    = √(a₁² + a₂² + a₃²)
```

For cat and dog, both lengths are `√(4 + 9 + 0) = √13 ≈ 3.606`, so the cosine is `12 / 13 ≈ 0.923`. For cat and car, it's `3 / (3.606 × 3.162) ≈ 0.263`. Cosine always lands between -1 and 1, and it's the cosine of the angle between the two arrows: 1 means the same direction, 0 a right angle, -1 opposite. Our doubled "cat" still scores 0.923 with "dog".

### Real numbers from GPT-2

The toy table was made up. GPT-2's isn't: it's 50,257 rows of 768 learned numbers, and its weights are public. We loaded the table and computed cosine similarities between rows (tokens that start with a space are whole words in the middle of a sentence, as Part 2 explained):

| Token pair | Cosine similarity in GPT-2's embedding table |
|---|---|
| " Monday" and " Tuesday" | 0.89 |
| " seven" and " six" | 0.87 |
| " France" and " French" | 0.71 |
| " France" and " Germany" | 0.70 |
| " king" and " queen" | 0.66 |
| " France" and " Paris" | 0.63 |
| " cat" and " dog" | 0.55 |
| " cat" and " car" | 0.38 |
| " Monday" and " banana" | 0.28 |

The nearest neighbors show the same structure. The tokens closest to " seven" are " six", " eight", " five", " nine" and " four". The closest to " king" are " kings", " King", " queen", " prince" and " monarch". Nothing in training labeled these as numbers or royalty. Roughly, tokens that show up in similar contexts get similar slopes during training, so gradient descent pulls their rows in similar directions.

One caution about the scale. In GPT-2's table, two tokens picked at random average a cosine of about 0.27, not 0. The vectors all lean somewhat in a shared direction, so "0.38 for cat and car" means less than it sounds. Compare scores against each other, not against zero.

## Step 3: meaning as direction

The map analogy said directions can carry meaning. The word2vec paper made that idea famous.

In 2013, Tomas Mikolov and colleagues at Google published ["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781), which introduced word2vec: small, fast models whose only job was to learn one vector per word. Training on 1.6 billion words took less than a day. Citing [earlier work by Mikolov's team](https://aclanthology.org/N13-1090/), the paper notes that `vector("King") - vector("Man") + vector("Woman")` "results in a vector that is closest to the vector representation of the word Queen."

The idea: the arrow from "man" to "king" captures something like "royal", and adding it to "woman" should land near "queen". The paper built a test set of 19,544 such questions, from "Paris is to France as Berlin is to ?" to "walking is to walked as swimming is to ?". Its best model, 300-dimensional skip-gram vectors trained on 783 million words, answered 53.3% of them exactly right.

### Trying it on GPT-2

We ran the same arithmetic on GPT-2's embedding table, then listed the closest tokens by cosine similarity:

| Query | Closest token | Second closest |
|---|---|---|
| " king" - " man" + " woman" | " king" (0.78) | " queen" (0.71) |
| " Paris" - " France" + " Germany" | " Paris" (0.73) | " Berlin" (0.67) |
| " walking" - " walk" + " swim" | " swim" (0.84) | " swimming" (0.73) |

The right answer comes second every time. First place goes to one of the words you typed in.

A 2020 paper by [Nissim, van Noord and van der Goot](https://aclanthology.org/2020.cl-2.7/) explains the pattern. The original word2vec code, they point out, never allows the answer to be one of the input words. The method from the paper above says so too: "we discard the input question words during this search." With that rule removed, accuracy on the word2vec test set fell from 74% to 21%, mostly because the answer to "man is to king as woman is to ?" became "king".

So the directions are real, but weaker than the famous example suggests. Adding the man-to-woman arrow moves "king" toward "queen", just not far enough to pass "king" itself. Nissim and colleagues cared because analogy searches were being used to reveal social bias in embeddings, and the exclusion rule could force a biased-looking answer. When "man is to doctor as woman is to ?" isn't allowed to answer "doctor", it has to answer something else.

### The same token, one vector, every context

Every vector in this section came straight from the embedding table, and that table has exactly one row per token. The " bank" in "river bank" and the " bank" in "bank account" get the identical starting vector. These are called static embeddings.

The Transformer blocks are what make them contextual. As the vector for " bank" passes through the layers, attention mixes in information from the surrounding words (Parts 5 and 6 show how). We can watch it happen. We ran GPT-2 on six sentences, three about rivers ("I sat on the river bank") and three about money ("The bank approved my loan"), and compared the " bank" vectors at each layer:

| Where in GPT-2 | Same meaning (average cosine) | Different meaning (average cosine) |
|---|---|---|
| Input, before any block | 0.94 | 0.95 |
| After block 3 of 12 | 0.95 | 0.89 |
| After block 6 of 12 | 0.91 | 0.80 |

At the input, river banks and money banks look the same; the small differences come only from GPT-2 adding a learned vector for each position, which Part 7 covers. By the middle of the model, the two senses have separated: the table supplied a starting point, and the blocks turned it into a vector for this word in this sentence.

## Step 4: back out, from a vector to logits

After the last block, the model has one final vector per position. To predict the next token, it needs a score for each of the 50,257 tokens in GPT-2's vocabulary (or 128,256 in Llama 3). This last step is called the unembedding, or the LM head (language-model head).

It's a second table with the same shape as the first: one row per token, one column per dimension. The score for each token is the dot product of the final vector with that token's row. The scores are the logits.

Back to the toy table. Suppose the final vector is `h = [1, 2, 0]`. Use the toy table as the output table too:

| Token | Row | Logit `h · row` | `e^logit` | Probability |
|---|---|---|---|---|
| cat | [2, 3, 0] | 1×2 + 2×3 + 0×0 = 8 | 2,981.0 | 72.9% |
| dog | [3, 2, 0] | 1×3 + 2×2 + 0×0 = 7 | 1,096.6 | 26.8% |
| car | [0, 1, 3] | 1×0 + 2×1 + 0×3 = 2 | 7.4 | 0.18% |
| truck | [1, 0, 3] | 1×1 + 2×0 + 0×3 = 1 | 2.7 | 0.07% |

The model's final vector points toward the "animal" corner of the space, so animals get high logits. Then softmax, which Part 1 introduced, turns logits into probabilities: raise `e` (about 2.718) to each logit and divide by the total, here 4,087.7.

In matrix form, all the logits come out of one multiplication, `logits = output_table × h`. For Llama 3 8B, that's 128,256 rows times 4,096 numbers, about 525 million multiply-adds for every token generated, roughly 7% of the per-token weight arithmetic.

### Real logits from GPT-2

We ran the full GPT-2 model on "The capital of France is" and read off the logits for the next token:

| Rank | Next token | Logit | Probability |
|---|---|---|---|
| 1 | " the" | -100.25 | 8.5% |
| 2 | " now" | -100.82 | 4.8% |
| 3 | " a" | -100.86 | 4.6% |
| 4 | " France" | -101.21 | 3.2% |
| 5 | " Paris" | -101.21 | 3.2% |
| 49 | " London" | -103.94 | 0.21% |
| 112 | " Lyon" | -104.72 | 0.10% |
| 25,190 | " carrot" | -114.33 | 0.0000065% |

The smallest GPT-2 isn't sure the answer is "Paris". It rates " the" higher, as if heading for something like "the capital of France is the largest city". Part 1's 85% for "Paris" was a made-up illustration; a real 124-million-parameter model spreads its bets much wider.

Every logit is also around -100, which does no harm, because softmax only cares about differences between logits: add or subtract the same number from all of them and the probabilities don't change. " the" leads " Paris" by 0.96, so it's `e^0.96 ≈ 2.6` times as likely, whether the scores are -100 and -101 or 5 and 4. On its own, the value of a logit tells you nothing.

Temperature, which Part 1 showed, divides every logit by a number before softmax, sharpening or flattening these gaps. Part 9 covers how a model then picks a token from the probabilities.

![A left-to-right flow for GPT-2: token ID 4881, France, selects row 4881 of the 50,257 by 768 embedding table, giving an input vector of 768 numbers. It passes through the Transformer blocks (Parts 5 and 6) to a final vector of 768 numbers. That vector is dot-multiplied with every row of the output table to give 50,257 logits, one per token, and softmax turns them into probabilities. A dotted arrow from the embedding table to the dot-product step marks that GPT-2 uses the same table for both.](/diagrams/f04-embed-unembed.svg "Words enter as rows of one table and leave as dot products with the rows of another, or, when the weights are tied, the same one.")

Both ends of the model work against a vocabulary-sized table, and only the blocks in the middle read the context.

### Weight tying: one table for both ends

The two tables have the same shape, and they want similar things. Press and Wolf, who [proposed tying them in 2016](https://arxiv.org/abs/1608.05859), describe it like this: in the input table, the network should react similarly to synonyms, and in the output table, words that are interchangeable should get similar scores. So why not use one table for both?

They reported "a significant reduction in perplexity", meaning better next-token predictions, across several language models, and showed that tying could shrink translation models "to less than half of their original size" without hurting their quality. The original Transformer paper adopted it, sharing one matrix between its embedding layers and "the pre-softmax linear transformation" ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

Real models split both ways:

| Model | Tied? | How you can tell |
|---|---|---|
| GPT-2 | Yes | its weight file has an input table and no separate output table |
| Qwen3-0.6B | Yes | `"tie_word_embeddings": true` in its [config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) |
| Qwen3-8B | No | `"tie_word_embeddings": false` in its [config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) |
| Llama 3 8B | No | Meta's [reference code](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py) builds a separate `output` layer next to `tok_embeddings` |

The Qwen3 model cards show the cost. Qwen3-0.6B lists [0.6 billion parameters, 0.44 billion of them non-embedding](https://huggingface.co/Qwen/Qwen3-0.6B): its single tied table of 151,936 × 1,024 accounts for the 0.16 billion difference, about a quarter of the model. Qwen3-8B lists [8.2 billion total and 6.95 billion non-embedding](https://huggingface.co/Qwen/Qwen3-8B), a gap of 1.25 billion that matches two untied tables of 151,936 × 4,096. The pattern follows the table in Step 1: in a small model, a second table would be a big extra cost, so tying is attractive; in a big one, the extra table is cheap and the two ends are free to learn different things.

## In the real world

- Each extra vocabulary token adds one row to the input table and, if untied, one to the output table. Llama 3.1 405B spends about 2.1 billion parameters on each of its two tables. That's why vocabulary size is a design decision weighed against the shorter sequences a bigger vocabulary buys (Part 2).
- Every generated token pays for the output table. The input lookup is just a copy, but the logits need a full multiplication: 525 million multiply-adds per token for Llama 3 8B, about 7% of its weight arithmetic. In GPT-2's smallest model, whose table is almost a third of all parameters, the share is far bigger.
- Embedding models for search are a close cousin of the input table. A text embedding model turns a whole sentence or document into one vector, so that searching for meaning becomes finding the nearest vectors. OpenAI's [text-embedding-3-small and -large](https://developers.openai.com/api/docs/guides/embeddings) produce 1,536 and 3,072 numbers. The docs recommend cosine similarity and note the vectors are "normalized to length 1", so a plain dot product gives the same answer faster. Unlike the static table in this article, these vectors come from running a whole Transformer over the text. [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/) and [BM25 vs Dense vs Hybrid Search](/blog/bm25-dense-hybrid-search/) show how they're used in retrieval.
- Open-weight models publish both tables, so you can check all of this yourself. Loading GPT-2's 50,257 × 768 table and computing the similarities in this article takes a few lines of Python and no GPU.

## Common misconceptions

- **"Each dimension of an embedding means something, like 'royalty' or 'gender'."** Meaning lives in directions, which are usually mixtures of many dimensions. Training has no reason to line concepts up with the axes. Part 16 looks at how interpretability research hunts for meaningful directions.
- **"king - man + woman = queen, exactly."** The arithmetic gets close to "queen", but in GPT-2's table "king" itself is still closer, and in word2vec the trick only works because the search throws out the input words. Without that rule, accuracy on the standard test fell from 74% to 21%.
- **"A word has one embedding."** It has one row in the input table. The vector the model actually reasons with changes at every layer and depends on the whole sentence: GPT-2's " bank" vectors for river and money sentences start out the same and drift apart through the blocks.
- **"Logits are probabilities, or at least positive."** They're raw scores of any sign and size. GPT-2's are all near -100. Only the gaps between them matter, and softmax turns those gaps into probabilities.
- **"Embeddings are designed by people."** Every number in the table starts random and is learned by gradient descent, just like the other weights. The structure, days of the week together and numbers together, emerges from predicting the next token.

## Key takeaways

- The embedding table has one row per vocabulary token. Embedding a token means reading its row, which is the same as multiplying a one-hot vector by the table.
- Similarity between vectors is a dot product. Cosine similarity divides out the lengths so it compares direction alone; GPT-2 rates " Monday" and " Tuesday" at 0.89.
- Directions carry some meaning, so "king minus man plus woman" lands near "queen", but the famous result depends on excluding the input words. Without that, "king" usually wins.
- The input table gives every token the same vector in every sentence. The Transformer blocks are what make it depend on context.
- The output layer scores every token with a dot product against its row: those scores are the logits, and only their differences matter. GPT-2 uses one table for both ends; Llama 3 8B uses two.

## What's next

We now know how tokens become vectors and how vectors become scores. Everything in between, the part that turns " bank" into a river bank or a money bank, starts with attention. [Part 5](/blog/attention-explained/) shows how each token asks a question with a query vector, how other tokens answer with keys and values, and how a few dot products, the same operation you just learned, decide what each word pays attention to.

## Further reading

- [Efficient Estimation of Word Representations in Vector Space (Mikolov, Chen, Corrado & Dean, 2013)](https://arxiv.org/abs/1301.3781): the word2vec paper, with the analogy test set and the king-to-queen example.
- [Linguistic Regularities in Continuous Space Word Representations (Mikolov, Yih & Zweig, 2013)](https://aclanthology.org/N13-1090/): the earlier paper that first showed vector offsets like King - Man + Woman.
- [Fair Is Better than Sensational: Man Is to Doctor as Woman Is to Doctor (Nissim, van Noord & van der Goot, 2020)](https://aclanthology.org/2020.cl-2.7/): how excluding the input words shaped what analogy results seemed to show.
- [Using the Output Embedding to Improve Language Models (Press & Wolf, 2016)](https://arxiv.org/abs/1608.05859): the case for tying the input and output tables.
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): section 3.4 describes the shared embedding and pre-softmax weights.
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2, whose public weights produced the numbers in this article.
- [Vector embeddings (OpenAI API docs)](https://developers.openai.com/api/docs/guides/embeddings): how text embedding models are used for search, with dimensions and distance functions.
- [The Illustrated Word2vec (Jay Alammar, 2019)](https://jalammar.github.io/illustrated-word2vec/): a visual walk through word vectors and how word2vec trains them.
- [Transformers, the tech behind LLMs (3Blue1Brown, 2024)](https://www.youtube.com/watch?v=wjZofJX0v4M): an animated look at embeddings, next-token prediction and softmax.
