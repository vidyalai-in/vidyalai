---
title: "Multimodal Models: How LLMs See and Hear"
description: "How images, video and audio become vectors an LLM can read: ViT patches, CLIP, LLaVA-style projectors, Whisper and audio codec tokens, with real numbers."
slug: "multimodal-llms"
category: "llm-fundamentals"
tags: ["multimodal", "transformers", "embeddings", "tokens", "cost"]
series: "llm-fundamentals"
seriesOrder: 15
pubDate: 2026-04-10
---

## TL;DR

A language model only reads vectors, so seeing and hearing come down to turning pixels and sound into vectors it can read. Images are cut into small square patches, a vision encoder turns each patch into a vector, and a small projector maps those vectors into the same space as word embeddings, so an image arrives as a few hundred to a few thousand "visual tokens" in the same sequence as your text. Audio goes in either as spectrogram frames through an encoder like Whisper's, or as discrete tokens from a neural audio codec. Along the way you'll compute patch counts, a contrastive loss and the token cost of a photo, a second of video and a second of speech, and see why those numbers decide what multimodal models are good and bad at.

## Why it matters

Send a 1000 × 1000-pixel photo to Claude and it costs [1,296 tokens](https://platform.claude.com/docs/en/build-with-claude/vision). At [Part 1](/blog/how-llms-work/)'s rule of thumb of 0.75 words per token, that's what about 1,000 words of English would cost. A 12-megapixel phone photo sent to the open Qwen2-VL model at its default limits becomes 15,552 tokens. And an hour of video on Gemini, at high resolution, comes to about [1.08 million](https://ai.google.dev/gemini-api/docs/video-understanding) tokens, enough to fill a 1-million-token context window on its own.

No step in the pipeline writes a caption like "a dog on a beach" for the model to read. The image becomes a grid of vectors in the same sequence as your prompt, and the model attends to them the way it attends to words.

So what is a visual token, and how does a model trained on text learn to read one? Why can a model describe a painting's mood but miscount the apples in a bowl? And how does a voice assistant answer in a fifth of a second?

## The intuition

Picture a photo cut into a grid of small square tiles, each tile numbered by its row and column. A specialist looks at every tile, together with all the others, and writes a short note about each one in the reader's own private shorthand. The reader never sees the photo. It reads the notes, in order, next to your question.

In a real model, the tiles are patches, the specialist is the vision encoder, and the shorthand is the embedding space of the language model from [Part 4](/blog/embeddings-and-logits/).

The analogy breaks in two places. The notes aren't words: each is a vector of a few thousand numbers that may match no token in the vocabulary and can carry color, texture and position that no word would. And the reader doesn't go through them in order; at every layer, attention mixes tile notes with your words.

## Step 1: cutting an image into patches

### Why not one token per pixel?

To a computer, a color image is a grid of numbers: height × width × 3 color channels (red, green, blue). A 224 × 224 image holds 150,528 numbers.

The obvious design would make each pixel a token. That gives 50,176 tokens, and attention compares every token with every other one ([Part 5](/blog/attention-explained/)), so one attention map would have 50,176² ≈ 2.5 billion entries per head per layer, for one small picture.

### The Vision Transformer

In 2020, Google's [Vision Transformer](https://arxiv.org/abs/2010.11929) (ViT) paper, titled "An Image is Worth 16x16 Words", showed a cheaper way. Cut the image into 16 × 16-pixel patches, flatten each patch into one list of numbers, and multiply it by a learned matrix to get a vector of the model's width. The paper calls these patch embeddings. Then add a learned position embedding to each one, as in [Part 7](/blog/positional-encoding/), and feed the sequence to an ordinary Transformer encoder.

A 224 × 224 image becomes a 14 × 14 grid, so 196 patches. Each patch holds 16 × 16 × 3 = 768 numbers. Here is the whole input step in NumPy, with random numbers standing in for the learned weights:

```python
import numpy as np

image = np.random.rand(224, 224, 3)            # height, width, RGB
P = 16                                         # patch size in pixels
patches = (image.reshape(224 // P, P, 224 // P, P, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(-1, P * P * 3))
print(patches.shape)                           # (196, 768)

W = np.random.randn(768, 768) * 0.02           # learned projection, D = 768 (ViT-Base)
pos = np.random.randn(196, 768) * 0.02         # learned position embeddings
tokens = patches @ W + pos                     # 196 vectors for the encoder
```

Compare this with the embedding table from Part 4. Text has a fixed vocabulary, so each token ID picks a row. Patches have no vocabulary: any of the 768 numbers can take any value, so the model uses a matrix multiplication instead of a lookup, and the result is continuous.

The paper tried 2D-aware position embeddings and found no significant gain over plain 1D ones. Three sizes became standard:

| Model | Layers | Width D | Heads | Parameters |
|---|---|---|---|---|
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1,024 | 16 | 307M |
| ViT-Huge | 32 | 1,280 | 16 | 632M |

Names like ViT-L/14 give the size and the patch width, here Large with 14-pixel patches. The patch width matters a lot, because the paper notes that sequence length is "inversely proportional to the square of the patch size": halve the patch and you get four times as many tokens.

ViT lacks the built-in image knowledge of a convolutional network, which is wired to look at neighboring pixels, so it needs more data. Trained on ImageNet's 1.3 million images alone, it trailed. Pretrained on Google's internal set of 303 million images, ViT-H/14 reached 88.55% on ImageNet, ahead of the best convolutional networks of the time.

### Resolution is expensive

Patches make resolution a direct cost. A 1024 × 1024 image at 16-pixel patches gives 64 × 64 = 4,096 patches, and its attention map is 4,096² / 196² ≈ 437 times larger than the 224-pixel version's. Much of what follows, from merging neighboring tokens to sampling only a frame or two per second of video, exists to keep that number down.

## Step 2: teaching images and text to share a space

A ViT trained to label ImageNet learns 1,000 fixed categories. A language model needs vectors that line up with what text means. OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-training, 2021) built that link with contrastive learning, a training method that pulls matching pairs together and pushes mismatched pairs apart.

CLIP has an image encoder and a text encoder, trained together on 400 million (image, text) pairs collected from the internet. For a batch of N pairs, both encoders produce vectors of the same length, and CLIP computes the cosine similarity (Part 4) of every image with every caption, an N × N grid. The loss pushes the N matching pairs on the diagonal up and the N² − N mismatches down.

### A contrastive loss by hand

Take a batch of three pairs, with cosine similarities made up for illustration:

| Image | "a dog" | "a cat" | "a red car" |
|---|---|---|---|
| Dog photo | **0.30** | 0.22 | 0.05 |
| Cat photo | 0.20 | **0.28** | 0.04 |
| Car photo | 0.03 | 0.02 | **0.31** |

Cosine similarities are small numbers, so CLIP multiplies them by a learned scale before the softmax. It starts at 1/0.07 ≈ 14.3, and the paper caps it at 100. Applying softmax to each row at 14.3 turns the first row into probabilities of 0.742, 0.237 and 0.021: the model's guess at which caption belongs to the dog photo.

The loss for that row is cross-entropy on the correct caption, `−ln(0.742) = 0.298`. The cat row gives 0.301 and the car row only 0.034, because the car is easy to tell apart. CLIP also runs the softmax down each column, asking which image belongs to each caption, and averages the two directions:

```text
loss = ( mean of row losses + mean of column losses ) / 2  =  0.213
```

The biggest losses sit where dog and cat get confused, so that's where the gradient pushes hardest. In real training N is 32,768, so each image is scored against 32,767 wrong captions at every step.

### What CLIP bought

The payoff is zero-shot classification, sorting images into categories never seen in training. To classify an image, write one caption per class ("A photo of a {label}." worked well as a default) and pick the caption with the highest similarity. With no ImageNet training at all, the best CLIP model reached 76.2% on ImageNet, matching the original ResNet-50, which had trained on all 1.28 million labeled examples.

It had blind spots. CLIP did poorly at counting objects in synthetic scenes, and scored only 88% on handwritten digits, below logistic regression on raw pixels, because its web data held almost no images like them.

CLIP's image encoder, usually ViT-L/14, became the default "eyes" for open multimodal models, and its successor SigLIP plays the same role in newer ones.

## Step 3: plugging vision into a language model

![Flowchart with three input paths feeding one sequence. Image path: an image is cut into 14 by 14-pixel patches, encoded by a vision encoder (ViT), then 2 by 2 neighbors are merged and an MLP projector maps them to the LLM's width. Audio path: audio becomes a log-Mel spectrogram, then goes through a Whisper-style audio encoder. Text path: prompt text goes through the tokenizer and embedding table. All three join one sequence of vectors (text, image, audio, text), which a decoder-only LLM reads to predict the next text token.](/diagrams/f15-vision-language-pipeline.svg "The common design: encoders turn each modality into vectors of the LLM's width, and the LLM reads one mixed sequence.")

Each modality gets its own encoder, but the language model reads one sequence.

### LLaVA: project and insert

[LLaVA](https://arxiv.org/abs/2304.08485) (Large Language and Vision Assistant, 2023) showed how little glue this takes. It took a frozen CLIP ViT-L/14 and multiplied each patch vector by one trainable matrix, giving it the width of the language model's word embeddings. The projected vectors go into the input sequence next to the text, and the language model, Vicuna, treats them like any other embedding.

LLaVA-1.5 ([Liu et al., 2023](https://arxiv.org/abs/2310.03744)) swapped the matrix for a two-layer MLP (multilayer perceptron, Part 3's small neural network) and raised the input to 336 pixels. Follow the shapes:

```text
336 × 336 image  ->  24 × 24 = 576 patches of 14 × 14 pixels
CLIP ViT-L/14    ->  576 vectors × 1,024 numbers
MLP projector    ->  576 vectors × 4,096 numbers   (Vicuna-7B's width)
LLM input        ->  [system text] [576 image vectors] [your question]
```

Every image costs the LLM 576 positions of context, whatever is in the picture.

Training ran in two stages. First, both big models stayed frozen and only the projector learned, on 595,000 image-caption pairs, which the paper describes as training "a compatible visual tokenizer for the frozen LLM". Then the projector and the LLM were fine-tuned together on 158,000 conversations about images that a text-only GPT-4 wrote from captions and object boxes. That's the SFT from [Part 13](/blog/base-model-to-assistant/), with the loss on the answer tokens only. The two stages took about 4 and 10 hours on eight A100 GPUs.

### Flamingo: cross-attention into a frozen model

DeepMind's [Flamingo](https://arxiv.org/abs/2204.14198) (2022) took the other route. It left the 70B Chinchilla language model frozen and inserted new cross-attention layers between its blocks. Cross-attention, from [Part 8](/blog/attention-is-all-you-need/), lets text positions look up information in a separate set of vectors, here 64 visual tokens per image or video, squeezed out by a module called the Perceiver Resampler.

Each new layer's output is multiplied by `tanh(α)`, with `α` starting at 0. Since tanh(0) = 0, the model starts out exactly as the frozen language model and learns how much to use the images. Trained on interleaved images and text from about 43 million web pages, Flamingo-80B could pick up new visual tasks from a few examples in the prompt.

Meta used the same idea for the [Llama 3](https://arxiv.org/abs/2407.21783) vision models: cross-attention layers after every fourth self-attention layer, about 100 billion new parameters for the 405B model, with the text weights frozen to keep text performance unchanged.

| Design | Where image vectors enter | Visual tokens per image | Examples |
|---|---|---|---|
| Projector into the input | Input sequence, next to the text | One per (merged) patch: 576 for LLaVA-1.5 | LLaVA, Qwen-VL family |
| Cross-attention adapter | New layers inside a frozen LM | Fixed: 64 for Flamingo | Flamingo, Llama 3 vision |
| Early fusion | Input sequence, trained together from the start | Varies | Chameleon, Llama 4 |

### Any resolution, with position in 2D

A fixed 336 × 336 input squashes a long receipt and a wide screenshot into the same small square. Alibaba's [Qwen2-VL](https://arxiv.org/abs/2409.12191) (2024) accepts images at close to their native size instead. Its 675M-parameter ViT uses 14-pixel patches and 2D RoPE, then a small MLP merges each 2 × 2 block of neighboring patches into one token, so each token covers 28 × 28 pixels. Special `<|vision_start|>` and `<|vision_end|>` tokens wrap the result. The paper's example: a 224 × 224 image becomes 16 × 16 = 256 patches, 64 merged tokens, plus the two markers, 66 in all.

Bigger images are first rounded to multiples of 28 pixels by the [preprocessor](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py), with a default ceiling of 16,384 merged tokens:

| Image | Resized to | Patches | Tokens to the LLM |
|---|---|---|---|
| 224 × 224 | 224 × 224 | 256 | 64 + 2 |
| 1024 × 1024 | 1036 × 1036 | 5,476 | 1,369 + 2 |
| 1920 × 1080 | 1932 × 1092 | 10,764 | 2,691 + 2 |
| 4032 × 3024 (12 MP) | not resized | 62,208 | 15,552 + 2 |

The model also needs to know where each visual token sits. Plain RoPE gives every token one position number, which flattens an image into a long line. Qwen2-VL's M-RoPE (multimodal RoPE) gives each token three numbers: time, height and width. Text tokens get the same value in all three, which makes it ordinary RoPE. Image tokens share one time value and get height and width from their grid cell. Take three text tokens, then an image of 2 × 3 merged tokens, then more text:

```text
text   "What is"     (0,0,0) (1,1,1) (2,2,2)
image  row 0         (3,3,3) (3,3,4) (3,3,5)
image  row 1         (3,4,3) (3,4,4) (3,4,5)
text   "this?"       (6,6,6) ...
```

The next text token starts one past the largest ID used, at 6, not at 9 as plain positions would give. Videos add a time step per frame. The paper credits these smaller position numbers with helping the model handle videos longer than any it trained on.

### Native multimodal models: early fusion

Every design so far bolts a separately trained vision encoder onto an LLM. Meta's [Chameleon](https://arxiv.org/abs/2405.09818) (2024) went further. It turns each 512 × 512 image into 1,024 discrete tokens, each one an ID from a codebook of 8,192 entries, so each token stands for a 16 × 16-pixel area. Those IDs sit in the same 65,536-entry vocabulary as text tokens, and one Transformer was trained from scratch on about 10 trillion tokens of mixed text and images. The paper calls this early fusion.

The compression is steep. Each ID takes 13 bits (2^13 = 8,192), so an image is 1,024 × 13 = 13,312 bits, about 1.7 KB, against 786 KB of raw pixels. The paper names the cost: the tokenizer struggles to reconstruct images with a lot of text, which caps the model on OCR-heavy tasks (optical character recognition, reading text in images).

Early fusion doesn't always mean discrete tokens. Meta says [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) uses early fusion to pretrain on text, images and video together, but still with a vision encoder, based on MetaCLIP. OpenAI's [GPT-4o system card](https://arxiv.org/abs/2410.21276) describes "an autoregressive omni model" whose inputs and outputs, text, vision and audio, "are processed by the same neural network."

## Step 4: video and sound

### Video is images over time

A video is a stack of frames, so models sample a few per second and encode each like an image. Flamingo sampled 1 frame per second. Qwen2-VL samples 2, then a 3D convolution with a depth of two folds each pair of frames into one set of patch "tubes", so a second of video costs about what one image does. At 448 × 448 that's 256 tokens per second. Training capped each video at 16,384 tokens, which at that size is only 64 seconds, so longer videos get lower resolution per frame.

Gemini's [documented](https://ai.google.dev/gemini-api/docs/video-understanding) default is 1 frame per second at 258 tokens per frame (66 at low resolution), plus 32 tokens per second of audio:

| Gemini video setting | Frame tokens per second | Audio tokens per second | Total, with metadata | One hour |
|---|---|---|---|---|
| High resolution | 258 | 32 | about 300 | about 1.08M |
| Low resolution | 66 | 32 | about 100 | about 360K |

That's why the docs say a 1-million-token model fits about 1 hour of video at high resolution or 3 hours at low. Long video is a long-context problem, with all the attention and KV cache costs from [Long-Context Inference](/blog/long-context-inference/). The docs also warn that 1 frame per second can miss fast action.

### Hearing: spectrograms and Whisper

Sound arrives as a waveform, air pressure measured thousands of times a second. OpenAI's [Whisper](https://arxiv.org/abs/2212.04356) speech recognizer, the encoder-decoder from [Part 9](/blog/encoders-decoders-gpt/), shows the standard way in. It resamples audio to 16,000 samples per second and computes a log-Mel spectrogram: for each 25-millisecond window, stepping every 10 milliseconds, it measures how much energy falls in each of 80 frequency bands spaced the way human hearing is. A spectrogram is effectively a picture of sound, with time across and pitch up.

Whisper reads 30 seconds at a time. The arithmetic, which matches the constants in its [code](https://github.com/openai/whisper/blob/main/whisper/audio.py):

```text
30 s × 16,000 samples/s          = 480,000 samples
480,000 / 160-sample hop (10 ms) = 3,000 spectrogram frames × 80 bands
two convolutions, the second with stride 2  ->  1,500 encoder positions
                                              = one vector per 20 ms
```

Audio LLMs often reuse this encoder. [Qwen2-Audio](https://arxiv.org/abs/2407.10759) starts from Whisper-large-v3's and adds pooling, so each vector covers about 40 milliseconds, 25 per second, and they enter the language model much like LLaVA's image vectors. Gemini counts 32 tokens per second of audio.

### Audio as discrete tokens

To *speak*, a model has to output sound, and next-token prediction needs a vocabulary. A neural audio codec provides one: a small network that compresses audio into a stream of integer codes and decodes them back into sound.

Meta's [EnCodec](https://arxiv.org/abs/2210.13438) (2022) turns 24 kHz audio into 75 frames per second. Each frame's vector is then quantized with residual vector quantization (RVQ), a method from Google's [SoundStream](https://arxiv.org/abs/2107.03312) codec. Here's RVQ on a toy 2-number vector `[0.83, −0.41]`, with two codebooks of four entries each:

| Stage | Codebook entries | Nearest entry (its ID) | What's left over | Error |
|---|---|---|---|---|
| 1 | `[1,0] [0,1] [−1,0] [0,−1]` | `[1, 0]` (0) | `[−0.17, −0.41]` | 0.44 |
| 2 | `[0.4,0] [0,0.4] [−0.4,0] [0,−0.4]` | `[0, −0.4]` (3) | `[−0.17, −0.01]` | 0.17 |

The frame becomes two IDs, (0, 3), and the decoder adds `[1, 0] + [0, −0.4]`. Each extra codebook quantizes what the previous ones missed, so more codebooks mean better sound and more tokens. EnCodec uses up to 32 codebooks of 1,024 entries, 10 bits each:

| Codebooks | Bitrate | Tokens per second |
|---|---|---|
| 2 | 1.5 kbps | 150 |
| 8 | 6 kbps | 600 |
| 32 | 24 kbps | 2,400 |

At 6 kbps that's 64 times smaller than raw 16-bit audio at 24 kHz (384 kbps), and 600 tokens for every second of sound.

### Speech in, speech out

Older voice assistants chained three models: speech recognition, a text LLM, then text-to-speech. Kyutai's [Moshi](https://arxiv.org/abs/2410.00037) paper puts the typical latency of such pipelines at several seconds, and notes that converting to text throws away emotion, accent and background sounds.

Moshi replaces the chain with one model. Its codec, Mimi, runs at 12.5 frames per second with 8 codebooks of 2,048 entries: `12.5 × 8 × 11 bits = 1.1 kbps`, or 100 tokens per second. A 7B text model, Helium, sits at the core. Moshi models the user's audio and its own as two parallel streams, so it can listen while it talks, and it predicts its own words as text tokens before the matching audio tokens, which the paper calls "Inner Monologue". Its theoretical latency is 160 ms, 200 ms in practice, below the 230 ms average gap in natural conversation the paper cites. GPT-4o's system card reports audio responses in as little as 232 ms, 320 ms on average.

### Making pictures, in brief

Generating images runs the other way, with two main families. Diffusion models start from random noise and remove it step by step; the [DDPM](https://arxiv.org/abs/2006.11239) paper used 1,000 noise levels. [Latent diffusion](https://arxiv.org/abs/2112.10752) runs that process on a compressed image, with each side shrunk 4 to 16 times, which makes it much cheaper.

Autoregressive models predict image tokens one at a time, like text. Chameleon generates its 1,024 codebook IDs and a decoder turns them back into pixels. GPT-4o's system card says it generates text, audio and image outputs from one autoregressive network.

## In the real world

Open models, and how images get in:

| Model | Vision encoder | How images enter | Image tokens |
|---|---|---|---|
| [LLaVA-1.5](https://arxiv.org/abs/2310.03744) | CLIP ViT-L/14, 336 px | Two-layer MLP projector | 576 |
| [Flamingo](https://arxiv.org/abs/2204.14198) | NFNet-F6 (contrastively trained) | Resampler plus gated cross-attention | 64 |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191) | 675M ViT, 14 px patches | 2 × 2 merge, MLP, M-RoPE | 66 at 224 px, up to 16,384 |
| [Qwen3-VL](https://arxiv.org/abs/2511.21631) | SigLIP-2 | 2 × 2 merge; also feeds ViT features into several LLM layers | Native resolution, 256K-token context |
| [Llama 3 vision](https://arxiv.org/abs/2407.21783) | ViT-H/14 | Cross-attention every fourth layer | None in the text sequence |
| [Chameleon](https://arxiv.org/abs/2405.09818) | None (codebook tokenizer) | Discrete tokens, early fusion | 1,024 per 512 × 512 |

Hosted APIs publish how they count:

| API | Rule | Example |
|---|---|---|
| [Anthropic](https://platform.claude.com/docs/en/build-with-claude/vision) | `⌈width / 28⌉ × ⌈height / 28⌉`, one token per 28 × 28-pixel patch; large images downscaled to at most 1,568 tokens (4,784 on the high-resolution tier) | 1920 × 1080 screenshot: 1,560 tokens, or 2,691 on the high-resolution tier |
| [OpenAI](https://developers.openai.com/api/docs/guides/images-vision) | Newer models: 32 × 32-pixel patches times a per-model multiplier. GPT-4o: 85 tokens plus 170 per 512-pixel tile | 1024 × 1024 image: 1,024 patches before the multiplier |
| [Gemini](https://ai.google.dev/gemini-api/docs/tokens) | 258 tokens up to 384 × 384 pixels; larger images tiled at 768 × 768, 258 tokens per tile | Video: about 300 tokens per second at high resolution |

Those tokens behave like prompt tokens. The vision encoder runs first, then they go through the prefill phase from [How LLM Inference Works](/blog/how-llm-inference-works/), so a few large images can outweigh the text prompt in tokens and in time to first token. vLLM's [prefix cache](/blog/prefix-caching/) includes image hashes in its keys, so a repeated image can reuse cached work.

The patch grid also explains the known failures. Anthropic's docs warn that counts can be off, "especially with large numbers of small objects", and that downscaling can make text less legible: a word that fits inside one 28-pixel patch shares a vector with everything else there. For document pipelines that parse and OCR pages instead, see [Designing a RAG System for 10 Million Documents](/blog/rag-system-10m-documents/).

## Common misconceptions

- **"The model sees the image the way we do."** It sees a grid of vectors, often after downscaling: about 1,300 for a 1-megapixel photo on Claude, each summarizing a 28 × 28-pixel block. Details smaller than a patch, like tiny text or dozens of small objects, are easy to lose.
- **"Images are turned into a text description first."** In LLaVA-style models the image becomes continuous vectors projected straight into the embedding space, and they need not match any word. That's how a model can answer about colors, layout or handwriting that no caption would mention.
- **"Higher resolution always gives better answers."** It always costs more tokens, but the Qwen2-VL authors found that "merely increasing the image size does not always lead to improved performance".
- **"Multimodal models were trained from scratch on images and text."** Most join a pretrained vision encoder to a pretrained LLM; LLaVA's first stage trained only the projector. Fully native models like Chameleon are the exception.
- **"Voice assistants transcribe you, then read a text reply aloud."** Older ones did. GPT-4o and Moshi handle audio in and out with one network, which is how they answer in about 200 to 320 ms.

## Key takeaways

- An LLM only reads vectors. Multimodal models turn images and sound into vectors of the LLM's width and put them in the same sequence as the text.
- Images are cut into patches, 14 or 16 pixels wide, and a Vision Transformer encodes them. Token count grows with the square of resolution, so a photo can cost from about 60 tokens to more than 15,000.
- CLIP's contrastive training on 400 million image-text pairs gave vision encoders a shared space with language. A small projector, trained first on its own, is enough to connect one to an LLM.
- Video is frames sampled at one or two per second, about 300 tokens per second on Gemini at high resolution, which makes long video a long-context problem.
- Audio enters as spectrogram frames through an encoder like Whisper's, or as codec tokens built with residual vector quantization (100 per second for Mimi, 600 for EnCodec at 6 kbps), which is what lets one model listen and speak.

## What's next

We've now built the whole model, from tokens and attention to training, reasoning and, here, eyes and ears. But we still can't say much about what happens inside it. [Part 16](/blog/llm-interpretability/) opens the black box: how interpretability researchers find features in a model's activations, why superposition packs more concepts than there are neurons, how sparse autoencoders pull those features apart, and what circuits reveal, and still miss, about how a model reaches its answers.

## Further reading

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929): the Vision Transformer, patch embeddings and the role of pretraining data size.
- [Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020): CLIP, its contrastive loss and zero-shot transfer.
- [Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2304.08485): LLaVA's projector and two-stage training.
- [Improved Baselines with Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2310.03744): LLaVA-1.5, the MLP connector and 336-pixel input.
- [Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198): the Perceiver Resampler and gated cross-attention into a frozen LM.
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (Wang et al., 2024)](https://arxiv.org/abs/2409.12191): dynamic resolution, 2 × 2 token merging and M-RoPE.
- [Qwen3-VL Technical Report (Bai et al., 2025)](https://arxiv.org/abs/2511.21631): SigLIP-2 encoder, interleaved MRoPE, DeepStack and video timestamps.
- [Chameleon: Mixed-Modal Early-Fusion Foundation Models (Chameleon Team, 2024)](https://arxiv.org/abs/2405.09818): images as discrete tokens in one shared vocabulary.
- [The Llama 3 Herd of Models (Llama Team, Meta, 2024)](https://arxiv.org/abs/2407.21783): Section 7 adds vision with cross-attention adapters.
- [Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022)](https://arxiv.org/abs/2212.04356): Whisper's log-Mel input and encoder-decoder.
- [Qwen2-Audio Technical Report (Chu et al., 2024)](https://arxiv.org/abs/2407.10759): a Whisper-initialized audio encoder feeding an LLM.
- [High Fidelity Neural Audio Compression (Défossez et al., 2022)](https://arxiv.org/abs/2210.13438): EnCodec, residual vector quantization and bitrates.
- [SoundStream: An End-to-End Neural Audio Codec (Zeghidour et al., 2021)](https://arxiv.org/abs/2107.03312): where residual vector quantization for neural codecs comes from.
- [Moshi: a speech-text foundation model for real-time dialogue (Défossez et al., 2024)](https://arxiv.org/abs/2410.00037): full-duplex speech-to-speech with the Mimi codec.
- [GPT-4o System Card (OpenAI, 2024)](https://arxiv.org/abs/2410.21276): one network for text, vision and audio, and its response latency.
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) and [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2021)](https://arxiv.org/abs/2112.10752): the two papers behind modern diffusion image generators.
- [Vision (Anthropic docs)](https://platform.claude.com/docs/en/build-with-claude/vision), [Images and vision (OpenAI docs)](https://developers.openai.com/api/docs/guides/images-vision) and [Video understanding (Gemini API docs)](https://ai.google.dev/gemini-api/docs/video-understanding): how hosted models count image and video tokens.
