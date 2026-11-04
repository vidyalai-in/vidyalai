---
title: "Serving Multimodal Models: Encoders, Image Tokens and Caching"
description: "Where the time and money go when a model reads images or audio: encoder cost, image-token prices, prompt and encoder caching, measured, plus a series review."
slug: "multimodal-inference"
category: "multimodal"
tags: ["multimodal", "serving", "caching", "latency", "cost"]
series: "multimodal-ai"
seriesOrder: 13
pubDate: 2026-10-29
updatedDate: 2026-11-05
---

## TL;DR

A request that carries an image or a sound clip has a stage a text request doesn't: an encoder turns pixels or samples into vectors before the language model reads anything, and those vectors arrive as hundreds or thousands of input tokens. This finale measures where the time and money go: per-stage timings of two open vision-language models on a laptop GPU, what an image costs on hosted APIs, and two kinds of caching. Prompt caching of an image on Claude Haiku 5.5 cut the cost of each later question by up to 10 times, and reusing image features locally cut a follow-up turn from 1.22 s to 0.34 s with identical outputs. It closes with a review of all 13 parts.

## Why it matters

Ask a 2-billion-parameter vision-language model on a laptop one question about a 1344 × 896 report page, and it spends 884 ms looking at the page before it reads a word of the question. That is 77% of the time to the first word of its answer. Ask a second question about the same page and a naive server looks at it all over again.

Hosted APIs charge for the same work. On Claude Haiku 5.5 one 1200 × 800 page is 1,250 input tokens, and with the question and answer that makes $0.13 per thousand questions. Mark the image for caching and every later question costs $0.018 per thousand, about 7 times less, from the second question on.

[Inference Part 1](/blog/how-llm-inference-works/) split a text request into prefill and decode. A multimodal request adds a third stage in front, and much of serving these models comes down to running that stage as rarely as possible.

## The intuition

Think of a simultaneous interpreter handed a thick printed report before a press conference. Before they can answer any question about it, they have to read it, and reading takes longer than answering. A sensible interpreter reads it once and keeps notes. A careless one rereads the whole report every time a journalist raises a hand.

The vision encoder is the reading, and the notes are the encoder's output, or further along, the language model's KV cache (the stored keys and values of every earlier token, from [Inference Part 2](/blog/kv-cache/)). Caching either one means the report is read once.

The analogy breaks in two places. A person's notes stay useful if the report gets a new cover page; a model's cached prefix becomes useless after any change, even a few pixels, because every later token depends on every earlier one. And a model can read many reports at once in a batch, which no interpreter can.

## Step 1: what's different about a multimodal request

Three things change compared with a text request.

1. An encoder stage runs before prefill. A vision transformer (ViT) for images, as in [Part 2](/blog/vision-transformers/), or an audio encoder such as Whisper's, as in [Part 9](/blog/speech-recognition-whisper/). Its output passes through a connector into the language model's input ([Part 4](/blog/vision-language-models/)).
2. Inputs are long. One picture becomes as many tokens as a few pages of text.
3. Outputs may not be text. A generator runs tens of denoising steps (Parts [5](/blog/diffusion-math/) to [7](/blog/flow-matching-dit/)) or writes audio codec tokens that a decoder turns into sound ([Part 11](/blog/voice-agents/)).

Earlier parts measured how many positions each input becomes:

| Input | Representation | Positions | Measured in |
|---|---|---|---|
| 1200 × 800 page | Claude visual tokens | 1,250 | this part |
| 1344 × 896 page | Qwen2-VL-2B image tokens | 1,536 | this part |
| 30-second audio window | Whisper encoder positions | 1,500 | [Part 1](/blog/multimodal-ai-overview/) |
| 1 second of speech | Mimi codec tokens, 8 codebooks | 100 | [Part 11](/blog/voice-agents/) |
| 5 seconds of 720p video | spacetime patches in a video DiT | 75,600 | [Part 12](/blog/video-generation/) |

![Flowchart of one multimodal request with its two caches. The request, an image plus a question, first has its image hashed. If the image is new, it is preprocessed (resized, tiled and normalized) and run through the vision encoder and connector, and the result is stored in an encoder cache of image embeddings. If the image was seen before, the stored embeddings are used directly. Prefill then reads the image and text tokens, skipping any prefix whose keys and values are already in the prefix cache from earlier turns. Decode writes one token per step and produces the text reply.](/diagrams/v13-request-path.svg "A vision-language request on a server with an encoder cache and a prefix cache. Each cache removes a stage for repeated inputs.")

## Step 2: where the time goes in a local VLM request

The test input is a report page drawn in code: a bar chart and a small table of made-up depot figures, no people, logos or brands. Every question has an exact answer on the page.

![A synthetic report page titled Depot operations report, page 1. On the left, a bar chart of trucks unloaded per day from January to June, with values 74, 32, 70, 65, 28 and 79 printed above the bars. On the right, a table of weekly depot totals for five depots, with columns for pallets in, pallets out and late trucks.](/diagrams/v13-report-page.png "Page 1 of the test report at 1200 × 800 pixels, rendered by lab/serving.py. Questions such as \"what is the 'Pallets out' value for South depot?\" (327) have exact answers.")

Two open models ran it on the laptop's NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) in bf16, with PyTorch 2.11 and transformers 5.21.1: [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) (a 675M-parameter ViT and a 1.5B language model) and [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), both Apache-2.0 and both from Part 4. Each request was timed in four stages: the processor (resizing, tiling and normalizing on the CPU, plus the copy to the GPU), the vision encoder with its connector, the language model's prefill given the image features, and decode. Each number is the median of 7 runs after 2 warm-ups, with `torch.cuda.synchronize()` around each. Decode per token is the difference between generating 33 tokens and 1, divided by 32.

The transformers version used here accepts precomputed image features, which is what makes the stages separable:

```python
with torch.inference_mode():
    feats = model.model.get_image_features(pixel_values=x["pixel_values"],
                                           image_grid_thw=x["image_grid_thw"], return_dict=True)
    text_only = {k: v for k, v in x.items() if k != "pixel_values"}
    out = model(**text_only, mm_encoder_outputs={"image": feats}, logits_to_keep=1)
```

![Horizontal stacked bar chart of one request, from 0 to 2.5 seconds, split into processor, vision encoder, language-model prefill and 32 decoded tokens, with a tick at the first token. Qwen2-VL-2B at 336 by 224, 96 image tokens: first token at 90 ms. At 672 by 448, 384 tokens: 238 ms. At 1344 by 896, 1,536 tokens: the encoder alone is 884 ms and the first token comes at 1,143 ms. SmolVLM-256M at its default setting, 832 tokens: the CPU processor takes 301 ms, the encoder 173 ms, first token at 499 ms. SmolVLM-256M with longest edge 512, 64 tokens: first token at 77 ms. The 32 decoded tokens take about 0.8 seconds in every row.](/diagrams/v13-vlm-stages.svg "Where the time goes in one local VLM request, RTX 4060 Laptop GPU, bf16. Medians of 7 runs.")

| Model, image | Image tokens | Processor | Vision encoder | LM prefill | First token | Decode per token |
|---|---|---|---|---|---|---|
| Qwen2-VL-2B, 336 × 224 | 96 | 2.5 ms | 46 ms | 41 ms | 90 ms | 24.4 ms |
| Qwen2-VL-2B, 672 × 448 | 384 | 4.2 ms | 156 ms | 77 ms | 238 ms | 25.3 ms |
| Qwen2-VL-2B, 1344 × 896 | 1,536 | 12.7 ms | 884 ms | 246 ms | 1,143 ms | 27.1 ms |
| SmolVLM-256M, default | 832 | 301 ms | 173 ms | 25 ms | 499 ms | 25.1 ms |
| SmolVLM-256M, `longest_edge` 512 | 64 | 22 ms | 25 ms | 30 ms | 77 ms | 25.5 ms |

The GPU stages of the two larger Qwen2-VL inputs stayed within 4% of their median; stages under 60 ms and the CPU processor varied more, up to 43% from the median. nvidia-smi read 1,545 to 2,160 MHz and 12 to 45 W, and peak GPU memory was 4.2 to 4.5 GB for Qwen2-VL and under 0.9 GB for SmolVLM.

Qwen2-VL's encoder grows faster than its token count: 16 times the tokens cost 19 times the encoder time, because its ViT attends over every 14-pixel patch, 6,144 of them at 1344 × 896, before merging them 4 to 1. The encoder is half the time to first token for the small image and 77% for the large one, which matches Part 4's 50% to 81% on square images.

SmolVLM's slowest stage isn't on the GPU at all. Its default processor enlarges the page to 2,048 pixels and cuts 12 tiles plus a global view, which took 301 ms of CPU time, 60% of its time to first token. That is why vLLM caches processor output too; its design notes say "some HF processors, such as the one for Qwen2-VL, are very slow."

Decode costs 24 to 27 ms per token whatever the image, on both models. Once the image is in the KV cache it's just more context, and each step is dominated by per-step overhead and weight reads, as [Inference Part 1](/blog/how-llm-inference-works/) explained.

A first attempt took 7.9 s for the large encoder and 15.7 GB of peak memory, twice the card: the features had been computed outside `torch.inference_mode()`, so PyTorch kept every activation for a backward pass that never came. Serving code needs gradients off.

## Step 3: what an image costs on a hosted API

### Claude's rule, extended

Anthropic's [vision docs](https://platform.claude.com/docs/en/build-with-claude/vision) give the rule: "Each patch is a 28×28-pixel block of the image, referred to as a visual token. An image, therefore, costs `⌈width / 28⌉ × ⌈height / 28⌉` visual tokens." Claude 4.7 and later models, Haiku 5.5 among them, accept up to 2,576 pixels on the long edge and 4,784 visual tokens; larger images are shrunk to fit. Parts 1 and 4 checked that against the free token-counting endpoint at 16 sizes and found the formula plus 3 every time.

Three more checks, on `claude-haiku-5-5`, subtracting the count for the question alone:

| What was sent | Bytes | Input tokens for the content |
|---|---|---|
| 1 page, 1200 × 800 PNG | 105 KB | 1,250 |
| 2, 4 and 8 different pages | PNG | 2,500, 5,000 and 10,000 |
| Page 1 as JPEG, quality 85 / quality 30 | 74 KB / 40 KB | 1,250 / 1,250 |
| Page 1 as WebP, quality 80 | 28 KB | 1,250 |
| Page 1 as a PDF with a text layer | 27 KB | 1,782 |
| Page 1 as an image-only PDF, 1200 × 800 or 864 × 576 | not recorded | 1,617 either way |

The 3 extra tokens come with every image, not once per request: 8 pages cost 8 × (1,247 + 3). Compression changes the upload size fourfold and the bill not at all, since Claude counts patches, not bytes. A PDF costs more than its page as a picture. The [PDF docs](https://platform.claude.com/docs/en/build-with-claude/pdf-support) explain: "The system converts each page of the document into an image," and "the text from each page is extracted and provided alongside each page's image." The image-only PDF cost the same 1,617 tokens at two page sizes, so the API renders PDF pages at its own resolution, and the text layer added 165 tokens, about what the page's text costs as plain text (166).

### Other providers

Each provider tokenizes media differently, read from their own docs on November 5, 2026, and not measured here:

| Provider | Images | Audio |
|---|---|---|
| Anthropic (Claude) | "⌈width / 28⌉ × ⌈height / 28⌉ visual tokens", up to 4,784 per image on the high-resolution tier | No audio price on the pricing page; Part 11's agent sent Claude a Whisper transcript instead |
| OpenAI ([images and vision](https://developers.openai.com/api/docs/guides/images-vision)) | Newer models "tokenize images by covering them with 32px x 32px patches", then multiply by a per-model factor, 1.2 for gpt-5.5 and gpt-6-astra: a 1024 × 1024 image is "ceil(1024 × 1.2) = 1229 tokens". Older models: a base plus tiles, 85 + 170 per 512-pixel tile on gpt-4o | Per-second audio token rate not found in the current docs (the cost guide's address returned 404) |
| Google ([Gemini tokens](https://aistudio.google.com/docs/tokens)) | "Images ≤384 pixels in both dimensions count as 258 tokens. Larger images are tiled into 768x768 pixel tiles, each counting as 258 tokens." | "32 tokens per second" |

The same 1024 × 1024 image is 1,369 tokens on Claude, 1,229 on gpt-6-astra at high detail and, by Google's rule, 4 tiles of 258 = 1,032 on Gemini, if each tile is a full 768 × 768 square. Token counts aren't prices: multiply each by that model's own input price.

## Step 4: prompt caching with images

Prompt caching, covered for text in [Inference Part 6](/blog/prefix-caching/), keeps the server's KV cache for a marked prefix so a later request that starts with the same bytes skips that part of prefill. Claude's [prompt-caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) list "Images and documents" in user turns among what can be cached, as long as the prompt is "100 percent identical ... including all text and images up to and including the block marked with cache control." Haiku 5.5's minimum is 512 tokens, so one 1200 × 800 image clears it alone.

The [pricing page](https://platform.claude.com/docs/en/about-claude/pricing) sets the trade for Haiku 5.5 (prompts up to 100,000 tokens): $0.10 per million input tokens, $0.125 per million for 5-minute cache writes (1.25 times), and $0.01 for cache hits (0.1 times).

### The experiment

Three arms ran interleaved in rounds, in shuffled order, with a new question each round shared by all three. The image went first and the question after it, as the vision docs recommend:

- plain: the image with no `cache_control`.
- write: `cache_control` on the image, but the page footer carries a unique reference string, so every call is a cache miss that writes a new entry.
- hit: `cache_control` on the identical image, written once by a priming call before the first round.

Every call used `claude-haiku-5-5` with thinking disabled, `max_tokens` 40 and a one-line system prompt asking for the number only, streamed through the Anthropic SDK; time to first token (TTFT) is the first text delta. Three sizes: S, page 1 at 1200 × 800; L, page 1 at 2352 × 1568; XL, all eight pages at 2352 × 1568. That made 40 rounds per size and 360 calls, on November 5, 2026, from India over the public API.

```python
blocks = [image_block(page) for page in pages]          # base64 PNGs
blocks[-1]["cache_control"] = {"type": "ephemeral"}      # cache everything up to the last image
with client.messages.stream(model="claude-haiku-5-5", max_tokens=40, system=SYSTEM,
                            thinking={"type": "disabled"},
                            messages=[{"role": "user", "content": blocks + [{"type": "text", "text": q}]}]) as s:
    ...
print(s.get_final_message().usage)   # input_tokens, cache_creation_input_tokens, cache_read_input_tokens
```

| Prompt, arm | Input / cache write / cache read tokens | Cost per question | Time to first token, median (95% CI) | p90 |
|---|---|---|---|---|
| S, plain | 1,301 / 0 / 0 | $0.000133 | 1.24 s (1.16-1.31) | 1.55 s |
| S, write | 27 / 1,274 / 0 | $0.000165 | 1.26 s (1.15-1.35) | 1.66 s |
| S, hit | 27 / 0 / 1,274 | $0.000018 | 1.13 s (1.08-1.19) | 1.37 s |
| L, plain | 4,758 / 0 / 0 | $0.000478 | 1.38 s (1.22-1.52) | 1.70 s |
| L, write | 27 / 4,731 / 0 | $0.000597 | 1.47 s (1.38-1.60) | 2.52 s |
| L, hit | 27 / 0 / 4,731 | $0.000053 | 1.47 s (1.36-1.58) | 1.81 s |
| XL, plain | 37,707 / 0 / 0 | $0.00377 | 2.68 s (1.88-3.04) | 4.18 s |
| XL, write | 27 / 37,680 / 0 | $0.00471 | 2.76 s (2.50-2.98) | 3.77 s |
| XL, hit | 27 / 0 / 37,680 | $0.00038 | 1.93 s (1.85-2.06) | 3.10 s |

Tokens are the median `usage` per call, costs the mean from the usage ledger, intervals bootstrapped over the 40 calls; p90 is the time 1 call in 10 exceeded.

Every call reported the split it should have: plain never read the cache, and each write stored the system prompt plus the images, leaving only the question's 27 tokens as ordinary input. XL's priming call read 4,731 tokens from the cache, the system prompt and page 1 that the L run had left there minutes before, because they were an exact prefix of the XL prompt. The last number in each reply matched the page in 38 to 40 of the 40 calls in every arm, no reply hit the 40-token limit, and all three arms gave the same answer in 38, 38 and 40 of the 40 rounds.

### When caching pays

Write T for the cached prefix in tokens and n for the number of questions. Without caching each question pays T at the base price. With caching the first pays 1.25 T and each later one 0.1 T:

```text
no cache:  n × T × $0.10 per million
cache:     (1.25 + 0.1 × (n - 1)) × T × $0.10 per million
n = 1:  1.00 T  vs 1.25 T   caching costs 25% more
n = 2:  2.00 T  vs 1.35 T   caching is already cheaper
n = 10: 10.0 T  vs 2.15 T   4.7 times cheaper
```

With the measured usage, question and output tokens included, the S page cost $0.000133 per question uncached, $0.000165 for the write and $0.000018 per hit: cheaper from the second question, and ten questions for $0.00032 instead of $0.0013. For the eight-page XL prompt, a question cost $0.00377 uncached, $0.00471 for the write and $0.00038 per hit, so ten questions came to $0.0081 instead of $0.038. The one condition is time: a 5-minute entry expires unless a hit refreshes it, so questions spaced more than 5 minutes apart need the 1-hour write, at 2 times the base price, which the docs say "pays off after two cache reads."

![Dot chart of time to first token in seconds for three prompt sizes and three arms, 40 calls each. One page at 1200 by 800: no cache 1.24 s, cache write 1.26 s, cache hit 1.13 s. One page at 2352 by 1568: 1.38, 1.47 and 1.47 s. Eight pages at 2352 by 1568: no cache 2.68 s with a wide interval from 1.88 to 3.04, cache write 2.76 s, cache hit 1.93 s. The 90th percentiles sit 0.2 to 1.5 s to the right of the medians, furthest for the eight-page prompt.](/diagrams/v13-cache-ttft.svg "Time to first token on Claude Haiku 5.5 with and without a cached image prefix. Dots: medians with 95% bootstrap intervals; diamonds: 90th percentiles.")

Speed moved much less than cost. Paired within each round, a hit beat the uncached call by a median 0.08 s on S (95% CI 0.02 to 0.17 s; faster in 31 of 40 rounds), by nothing on L (0.08 s slower, interval −0.04 to +0.16), and by 0.25 s on XL (0.13 to 0.54 s; 30 of 40). A cache write added a median 0.12 to 0.23 s over plain, though only for L did the 95% interval exclude zero. Every arm still uploads its images, 2.5 MB of base64 for XL, and a text-only Haiku call from the same place took 0.78 s to its first token in Part 11, so most of the wait is network and queueing that no cache removes. [Agentic Part 15](/blog/agent-cost-latency/) saw the same on agents: caching cut cost by 37% and left speed unchanged. Here the time saved only showed at 37,680 cached tokens.

## Step 5: caching and batching in a local server

### Reusing image features across turns

On the laptop, a 6-turn conversation about the same 1344 × 896 page (1,536 image tokens) with Qwen2-VL-2B, greedy decoding, ran three ways, 1 warm-up conversation and 5 timed:

- A, stateless: every turn runs the processor and encoder again and prefills the whole conversation.
- B, encoder cache: turn 1 stores the image features; later turns only tokenize the text and pass the stored features (`mm_encoder_outputs`).
- C, encoder and prefix cache: B plus the KV cache of earlier turns, trimmed to the longest common prefix, so each turn prefills only its new tokens.

| Variant | Turn 1 | Later turns, median | 6-turn conversation | Tokens prefilled on turn 6 | Replies identical to A |
|---|---|---|---|---|---|
| A, stateless | 1,231 ms | 1,224 ms | 7.37 s | 1,764 | the reference |
| B, image features cached | 1,196 ms | 337 ms | 2.85 s | 1,764 | 30 of 30 |
| C, features and KV cached | 1,202 ms | 217 ms | 2.28 s | 35 | 30 of 30 |

Every turn of every timed run fell within 6% of its median except one C turn at 307 ms; nvidia-smi read a median 2,010 MHz and 44 W. All 6 answers were right in every variant.

Caching the image features cut a follow-up turn 3.6 times, because the 884 ms encoder ran once instead of six times. Reusing the KV cache cut it to 217 ms, about 100 ms of it the 3 or 4 decoded tokens at 25 ms each. B's first-token logits matched A's exactly (largest difference 0.0); C's differed by at most 0.31 on logits of size 27, the rounding of bf16 kernels run on different shapes, and no reply changed.

### What the serving systems do

[vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang), the open servers from [Inference Parts 5](/blog/pagedattention-vllm/) and [6](/blog/prefix-caching/), build these steps in. Described from their documentation, not run here:

- Processor cache. vLLM's multimodal processor cache "is automatically enabled to avoid repeatedly processing the same multi-modal inputs", sized by `mm_processor_cache_gb` (default 4 GiB).
- Images in the prefix cache. vLLM hashes each 16-token KV block with its tokens, but image placeholder tokens look the same whatever the picture, so "we encode the image hash generated by the frontend image processor" into each block's hash. Two requests share cached blocks only if they carry the same image.
- Encoder disaggregation. vLLM's disaggregated encoder "runs the vision-encoder stage of a multimodal LLM in a process that is separate from the pre-fill / decoder stage", which also allows "cross-process reuse and caching of encoder outputs." SGLang's EPD (encoder, prefill, decode) disaggregation launches `--encoder-only` and `--language-only` servers, with an optional global cache of ViT embeddings shared across instances. The EPD paper ([Singh et al., 2025](https://arxiv.org/abs/2501.05460)) reports "up to 15x lower peak memory utilization", "up to 22x larger" batches and "up to 71%" lower TTFT against systems that don't split the stages (reported, not reproduced here).
- Encoder parallelism. Vision encoders are small, so splitting their weights across GPUs buys little. vLLM's `mm_encoder_tp_mode="data"` splits the batch of images instead, which its docs say improved "throughput and TTFT by around 10% for `tensor_parallel_size=8`"; SGLang's `--mm-enable-dp-encoder` does the same.

### Batching the encoder doesn't help here

The same Qwen2-VL-2B served batches of distinct 672 × 448 pages (384 image tokens each), median of 5 runs after warm-up:

| Batch | Encoder images per second | Whole requests per second (16 output tokens) | Output tokens per second | Peak GPU memory |
|---|---|---|---|---|
| 1 | 6.4 | 1.6 | 25 | 4.3 GB |
| 4 | 6.0 | 3.0 | 48 | 4.5 GB |
| 16 | 6.1 | 3.8 | 62 | 5.4 GB |
| 32 | 4.0 | 3.0 | 48 | 6.6 GB |

The encoder gives 6 images per second at any batch size: one 1,536-patch image already keeps the GPU's arithmetic busy, and this transformers version runs attention one image at a time. Decode is what batching speeds up, since all 16 requests share each read of the weights, so whole requests rose 2.4 times while the encoder stayed flat. At batch 32 peak memory passed what the 8 GB card had free and throughput fell. That asymmetry, a compute-bound encoder next to a memory-bound decoder, is the case for running them on separate workers, as [disaggregated prefill and decode](/blog/disaggregated-serving-llm-d/) does for text.

## Step 6: serving generators and audio

A generator's cost is roughly steps × cost per step; the cost per step grows with resolution, and classifier-free guidance doubles the work. Earlier parts measured each knob:

- Steps. Stable Diffusion 1.5 took 1.09 s per 512-pixel image at 8 steps, 3.5 s at 30 and 11.6 s at 100, in batches of 4, and 8 steps already matched the 100-step image's CLIP score ([Part 6](/blog/latent-diffusion/)).
- Guidance. Running the U-Net with and without the prompt cost 128.7 ms per step against 80.2 ms for one pass, 1.6 times ([Part 6](/blog/latent-diffusion/)).
- Resolution. A 5-second 720p clip is 75,600 tokens, and one full attention call over them took 491 ms on the laptop against 32 ms factorized ([Part 12](/blog/video-generation/)).

Few-step distillation attacks the first knob. A student model is trained to jump where the teacher needs many steps: Latent Consistency Models ([Luo et al., 2023](https://arxiv.org/abs/2310.04378)) report "2~4-step" sampling trained in "32 A100 GPU hours", and Adversarial Diffusion Distillation ([Sauer et al., 2023](https://arxiv.org/abs/2311.17042)) "in just 1-4 steps" (both reported, not reproduced here). [Part 7](/blog/flow-matching-dit/) showed the same idea at toy scale: one round of reflow took a one-step sample's error from 724 to 46.5.

For a server, generators are the opposite of VLMs: no long prompt to cache, and a full network pass at every step, so batching requests per step and cutting steps matter most.

For audio, streaming changes the budget more than any model choice. [Part 11](/blog/voice-agents/)'s cascaded voice agent took 2.41 s from the end of speech to its first sound, and 2.02 s when the first finished sentence went straight to synthesis; [Part 10](/blog/text-to-speech/) found sentence-by-sentence synthesis brought Kokoro's first audio from 325 ms to 72 ms. Chunk sizes set the floor: 32 ms detector chunks, a 500 ms silence rule that fired 605 ms after the last word, Mimi's 80 ms codec frame.

Whisper pads every clip to a 30-second window ([Part 9](/blog/speech-recognition-whisper/)), so it doesn't stream by design. Whisper-Streaming ([Macháček et al., 2023](https://arxiv.org/abs/2307.14743)) re-transcribes a growing buffer, commits only words that consecutive passes agree on, and reports "3.3 seconds latency" on long-form speech (reported, not reproduced here). Speech-to-speech models like Moshi skip the chain, at "200ms in practice."

## The series in one table

Each part's headline measurement, all on the same RTX 4060 Laptop GPU or Claude's API:

| Part | Topic | Headline measured number |
|---|---|---|
| [1](/blog/multimodal-ai-overview/) | Everything becomes tokens | One 512-pixel picture: 361 tokens on Claude, 324 on Qwen2-VL, 1,088 on SmolVLM; Haiku counted small dots right 25% of the time at 27 image tokens and 100% at 1,250 |
| [2](/blog/vision-transformers/) | Vision transformers | Removing ViT-B/16's position embeddings: 89.4% to 21.8% accuracy; a 1024-pixel image runs 32 times slower than 224 |
| [3](/blog/clip-contrastive-learning/) | CLIP | 89.8% zero-shot on CIFAR-10, worth 16 labelled images per class; a written label fooled it 94% of the time |
| [4](/blog/vision-language-models/) | Vision-language models | Haiku read a 5-character code 100 of 100 times at 896 pixels and 43 at 224; told not to list dots, its dense counts fell from 100% to 21% |
| [5](/blog/diffusion-math/) | Diffusion from scratch | DDIM at 20 steps (FD 8.9) beat DDPM at 50 (10.4) with the same 1.6M-parameter U-Net |
| [6](/blog/latent-diffusion/) | Latent diffusion | 2.77 s per Stable Diffusion image, 93% in the U-Net; guidance 1 to 7.5 lifted judged adherence from 13% to 66% |
| [7](/blog/flow-matching-dit/) | Flow matching and DiT | One round of reflow cut the one-step error from 724 to 46.5 |
| [8](/blog/autoregressive-image-generation/) | Autoregressive images | 49 raster passes: 139 ms and FD 15.5; 8 masked passes: 26 ms and 37.3 |
| [9](/blog/speech-recognition-whisper/) | Whisper | large-v3-turbo: 2.2% WER on LibriSpeech test-clean at 50 times real time; it still wrote text for 89 of 151 clips with nobody speaking |
| [10](/blog/text-to-speech/) | Text-to-speech | Kokoro-82M at 51 times real time; first audio after 72 ms sentence by sentence, 325 ms in one piece |
| [11](/blog/voice-agents/) | Voice agents | A cascaded agent answered in 2.41 s, 2.02 s streaming the first sentence; Mimi at 1.1 kbps kept WER at 0.45% |
| [12](/blog/video-generation/) | Video generation | Full attention over a 5-second 720p clip's 75,600 tokens: 491 ms per call, 32 ms factorized |
| 13 | Serving | Encoder 77% of Qwen2-VL's first-token time at 1,536 image tokens; cached image features cut a follow-up turn from 1.22 s to 0.34 s |

What this series didn't cover:

- Fine-tuning vision-language and speech models, for example with LoRA on your own documents or accents.
- Image editing, inpainting and control signals such as ControlNet, and music generation.
- 3D and scene representations (NeRFs, Gaussian splatting) and vision-language-action models for robots.
- Long video understanding, where frame sampling and token budgets decide what a model can see.
- Provenance and misuse: watermarking, content credentials and detecting generated media.
- Large-scale multimodal benchmarks; the experiments here used synthetic, checkable task sets.

## In the real world

- vLLM's disaggregated-encoder example runs `Qwen/Qwen2.5-VL-3B-Instruct` as one encoder instance feeding one prefill-decode instance, and its proxy passes audio for models such as Qwen2-Audio and Qwen3-Omni as "metadata-only references", so the second instance never repeats audio preprocessing.
- SGLang's EPD example puts two encoder servers in front of one language server for Qwen3-VL-8B, for workloads where the encoder is the slow stage.
- Claude's vision docs say high resolution "can use up to roughly three times more visual tokens" and advise downsampling when the detail isn't needed.
- At Gemini's 32 tokens per second, an hour of audio is 115,200 tokens before any answer is written.

## Common misconceptions

- **"A smaller image file is a cheaper image."** Claude counts 28-pixel patches. PNG, JPEG and WebP versions of the same page, from 105 KB down to 28 KB, all cost 1,250 tokens.
- **"Images make every token slower."** They make the first token slower. Qwen2-VL decoded at 24 to 27 ms per token whether the image was 96 or 1,536 tokens.
- **"The GPU is always the bottleneck."** SmolVLM's default processor spent 301 ms on the CPU, more than its encoder and prefill together.
- **"Caching changes what the model answers."** Reused image features gave bit-identical logits, a reused KV cache moved them by bf16 rounding, and no reply changed in 30 turns.
- **"Caching only pays for long conversations."** On Claude a cached image is cheaper from the second question, because a hit costs a tenth of the base price.

## Key takeaways

- A multimodal request adds an encoder stage before prefill. On a laptop it was half of Qwen2-VL's time to first token for a small image and 77% for a large one; decode didn't notice the image.
- Image tokens follow each provider's own rule: 28-pixel patches on Claude, 32-pixel patches times a model multiplier on OpenAI's newer models, 768-pixel tiles of 258 tokens on Gemini. File size doesn't change the count.
- Prompt caching an image on Claude Haiku 5.5 costs 25% more on the first question and 90% less on every later one, so it pays from question two.
- Locally, keep the image features and the KV cache: a follow-up turn fell from 1.22 s to 0.34 s and then 0.22 s, with the same answers.
- Batching helps decode, not a busy encoder; that's why servers can now run the encoder on its own workers. Generators and voice agents are served by cutting steps and streaming chunks.

## What's next

This was the last part of Beyond Text. To go deeper on the serving machinery used here, the [Inference series](/blog/how-llm-inference-works/) covers batching, paging, caching and disaggregation for text models. [Measuring LLMs](/blog/why-llm-evals-are-hard/) shows how to test models properly, and [Agentic AI](/blog/what-is-an-ai-agent/) puts models like these to work in loops with tools, screenshots and voices.

## Further reading

- [Vision (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/vision): the visual-token formula, resolution tiers, limits and image placement.
- [Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): what can be cached, minimum lengths, breakpoints and invalidation.
- [Pricing (Claude API docs)](https://platform.claude.com/docs/en/about-claude/pricing): base, cache write, cache hit and batch prices for Haiku 5.5.
- [PDF support (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/pdf-support): how pages are converted into an image plus extracted text.
- [Efficiently Serving Large Multimodal Models Using EPD Disaggregation (Singh et al., 2025)](https://arxiv.org/abs/2501.05460): separating encode, prefill and decode onto different resources.
- [vLLM: disaggregated encoder](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_encoder.md), [optimization and multimodal caching](https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md) and [prefix caching design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md): processor caching, image hashes in KV blocks and encoder data parallelism.
- [SGLang: EPD disaggregation](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/epd_disaggregation.mdx) and [DP for the multimodal encoder](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dp_for_multi_modal_encoder.mdx).
- [Latent Consistency Models (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) and [Adversarial Diffusion Distillation (Sauer et al., 2023)](https://arxiv.org/abs/2311.17042): few-step image generation by distillation.
- [Turning Whisper into Real-Time Transcription System (Macháček et al., 2023)](https://arxiv.org/abs/2307.14743): streaming recognition on top of a 30-second-window model.
- [Images and vision (OpenAI docs)](https://developers.openai.com/api/docs/guides/images-vision) and [Understand and count tokens (Gemini API docs)](https://aistudio.google.com/docs/tokens): other providers' image and audio token rules.
- [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) and [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) (Hugging Face model cards): the two local models, both Apache-2.0.
- [Prefix Caching and KV Cache Offloading (Inference Part 6)](/blog/prefix-caching/) and [Cost and Latency of Agents (Agentic Part 15)](/blog/agent-cost-latency/): the text-only versions of the caching measured here.
