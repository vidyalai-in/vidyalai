---
title: "Build an LLM From Scratch: From Our Model to Frontier Models"
description: "Our 88M run next to Llama 3, DeepSeek-V3, OLMo 2 and SmolLM, the model exported to Hugging Face and llama.cpp, and the whole series reviewed."
slug: "frontier-training-recipes"
category: "llm-training"
tags: ["training", "pretraining", "quantization", "cost", "local"]
series: "llm-from-scratch"
seriesOrder: 15
pubDate: 2026-08-07
updatedDate: 2026-10-29
---

## TL;DR

This finale puts the series' 88M-parameter model next to the frontier. Llama 3 405B used 41 million times our training compute, DeepSeek-V3 3.5 million times and SmolLM2-135M, the closest small recipe, about 1,700 times, with the same kind of Transformer block. Then it ships our chat model: `export.py` writes a Hugging Face Llama checkpoint whose logits match ours within 2.3 × 10⁻⁵, and llama.cpp converts it to GGUF once told which pre-tokenizer it uses. At Q4_K_M the file is 59.6 MB, its perplexity is 0.9% above F32's, and it generates about 600 tokens per second on the laptop's CPU. The series review sums up what each part measured and what an 88M model can and can't do.

## Why it matters

Llama 3 405B was trained with 3.8 × 10²⁵ floating-point operations. The model this series trained in [Part 7](/blog/pretraining-run/) used 9.3 × 10¹⁷. One frontier run is 40.8 million of ours, and it cost about 31 million H100 GPU-hours where ours took 16.6 hours on one laptop GPU.

Yet both are decoder-only Transformers with RMSNorm, rotary positions, grouped-query attention and a SwiGLU feed-forward layer, trained with AdamW on next-token prediction and then tuned to chat. So the finale asks two questions. What does the other factor of 40 million buy? And can the tools that run everyone else's models load ours, quantize it and chat with it?

This part puts our run next to the published recipes of Llama 3, DeepSeek-V3, OLMo 2 and SmolLM, exports our chat model to Hugging Face and llama.cpp formats, measures it there, and reviews the series.

## The intuition

Think of a school science fair rocket next to a launch vehicle. The physics is identical: thrust, mass, drag, the same rocket equation. A student who has flown the small one reads the big one's specifications knowing what each number costs.

The analogy breaks in two places. A rocket scales by building more of the same parts, but a training recipe changes as it grows: the frontier runs use mixtures of experts, 8-bit arithmetic, thousands of GPUs in four kinds of parallelism and stages of training our model never needed. And a bigger rocket only flies higher, while a bigger language model gains abilities, such as arithmetic and multi-step answers, that our 88M model never shows at all, so its curves can't tell you when they appear.

## Step 1: our run next to published recipes

### Six runs in one table

Every number below is quoted from the run's own paper, report or model card, except where the table says it was computed. Compute uses the 6ND rule from [Part 1](/blog/build-llm-from-scratch/) (6 FLOPs per parameter per training token) where the source doesn't state its own total; for DeepSeek-V3, a mixture-of-experts model, N is the 37B parameters each token actually uses.

| Run | Parameters | Training tokens | Tokens per parameter | Compute (FLOPs) | GPU time | Data | Context | Vocabulary | Post-training |
|---|---|---|---|---|---|---|---|---|---|
| Ours, 88M (Part 7, Parts 10 to 12) | 88.1M | 1.76B | 20 | 9.3 × 10¹⁷ (6ND) | 16.6 h on one RTX 4060 Laptop GPU | FineWeb-Edu sample | 1,024 | 32,768 | SFT, then DPO or GRPO |
| [SmolLM2-135M](https://arxiv.org/abs/2502.02737) | 135M | 2T | 14,800 | 1.6 × 10²¹ (6ND) | "64 H100" (hours not given) | FineWeb-Edu-filtered DCLM, Stack-Edu, FineMath, Cosmopedia | 8,192 (config) | 49,152 | SFT on filtered SmolTalk, DPO on UltraFeedback |
| [SmolLM3-3B](https://huggingface.co/blog/smollm3) | 3B | 11.2T | 3,700 | 2.0 × 10²³ (6ND) | "384 H100 GPUs for 24 days": 221,000 GPU-hours (computed) | Web, code and math; "training configs with exact data weights" published | 4,096, extended to 64K | Llama 3.2's tokenizer | SFT, APO preference tuning, model merging |
| [OLMo 2 7B](https://arxiv.org/abs/2501.00656) | 7B | 4.05T | 580 | 1.8 × 10²³ (paper) | Not given in hours; 131 MWh of GPU power | OLMo-Mix-1124 and Dolmino-Mix-1124, both released | 4,096 | About 100K (cl100k) | Tülu 3: SFT, DPO, RLVR |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B total, 37B active | 14.8T | 400 per active parameter | 3.3 × 10²⁴ (6ND, active) | 2.788M H800 GPU-hours on 2,048 H800s; "$5.576M" at $2 per hour | Not released | 4K, extended to 32K, then 128K | 128K | SFT on 1.5M instances, RL with GRPO, reasoning distilled from DeepSeek-R1 |
| [Llama 3 405B](https://arxiv.org/abs/2407.21783) | 405B | 15.6T | 39 | 3.8 × 10²⁵ (paper) | 30.84M H100 GPU-hours (Llama 3.1 model card), up to 16K H100s | Not released | 8K, extended to 128K | 128K | Six rounds of SFT, rejection sampling and DPO |

SFT is supervised fine-tuning on example conversations, DPO (direct preference optimization) and APO (anchored preference optimization) learn from pairs of better and worse replies, GRPO (group relative policy optimization) and RLVR (reinforcement learning with verifiable rewards) learn from scored samples, and rejection sampling keeps the best of several sampled replies for further SFT. The GPU-hours for Llama 3.1 405B come from its [model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md), which lists "a cumulative of 39.3M GPU hours" for the three sizes and 30.84 million for the largest. OLMo 3's 7B model, released in late 2025, trained on 5.93 trillion tokens ([model card](https://huggingface.co/allenai/Olmo-3-1025-7B)); the table keeps OLMo 2, whose paper states its compute.

![Horizontal bar chart of training compute on a log scale from 10 to the 16 to 10 to the 26 FLOPs. In orange, Part 8's 11M run at 1.4 times 10 to the 16, 0.015 times ours, and our 88M model at 9.3 times 10 to the 17. In blue: SmolLM2-135M on 2T tokens, 1,733 times ours; OLMo 2 7B on 4.05T tokens, 193,310 times; SmolLM3-3B on 11.2T tokens, 216,507 times; OLMo 2 32B on 6.6T tokens, 1.4 million times; DeepSeek-V3 on 14.8T tokens, 3.5 million times; Llama 3 405B on 15.6T tokens, 40.8 million times.](/diagrams/t15-compute.svg "Training compute of our runs and six published ones, as multiples of the 88M model's 9.3 × 10¹⁷ FLOPs.")

### What each factor of a thousand buys

Our run did 15.5 trillion FLOPs per second of 6ND work, averaged over its 16.6 hours. Divide Llama 3 405B's 3.8 × 10²⁵ FLOPs by its 30.84 million GPU-hours and each H100 did 342 trillion per second; DeepSeek-V3's pretraining, 3.3 × 10²⁴ FLOPs in 2.664 million H800-hours, comes to the same 342. A data-center GPU-hour is worth about 22 laptop GPU-hours, and these runs bought millions of them.

Three steps of about a thousand each separate the rows:

- From 10¹⁸ to 10²¹, SmolLM2-135M kept the model small and spent the compute on data, 14,800 tokens per parameter. On the same text [Part 9](/blog/evaluate-base-model/) measured it at 0.853 bits per byte against our 1.045, and 64.4% on ARC-Easy against our 47.9%. This laptop would need about 3.3 years for that run.
- At 10²³, Chinchilla's rule would build a 39B model; OLMo 2 chose a 7B model, cheaper to run, on 580 tokens per parameter. In its own evaluation suite it scores 83.8 on HellaSwag, where ours scores 28.0 against chance at 25, and 67.5% on GSM8K, grade-school math word problems ours can't attempt.
- At 10²⁵, Llama 3 405B is close to compute-optimal by its own scaling law, which "suggests training a 402B parameter model on 16.55T tokens." That law came from IsoFLOP curves, the method [Part 8](/blog/scaling-laws-experiment/) ran at 2.6 × 10¹⁶ FLOPs.

### What the frontier recipes add

Some differences are pure scale: Llama 3 405B's batch grew from 4 million to 16 million tokens, against our 524,288, over 1.2 million steps against our 3,360. Others change the model or the arithmetic:

- Mixture of experts (MoE): DeepSeek-V3 stores 671B parameters but routes each token through only 37B of them, so compute per token follows the active count ([Fundamentals Part 11](/blog/mixture-of-experts/)).
- Multi-head latent attention (MLA) compresses each token's stored keys and values; multi-token prediction (MTP) also predicts one extra token ahead ([Inference Part 14](/blog/deepseek-inference-deepinfer/)).
- FP8 training: in DeepSeek-V3 "most compute-density operations are conducted in FP8," 8-bit floating point, and its report says the approach is "for the first time" validated "on an extremely large-scale model" ([GPU Part 6](/blog/tensor-cores/) measures FP8 tensor cores).
- Long-context extension: pretrain on short sequences, then continue on long documents, from 8K to 128K for Llama 3 and from 4K to 32K to 128K for DeepSeek-V3 with YaRN, a RoPE rescaling ([Inference Part 11](/blog/long-context-inference/)).
- Four kinds of parallelism (tensor, context, pipeline and data) spread Llama 3 405B over up to 16,384 GPUs at 38 to 43% model FLOPs utilization ([Inference Part 10](/blog/model-parallelism/), [Part 14](/blog/training-memory/)).
- Mid-training, a last stage on upsampled high-quality and synthetic data: OLMo 2's takes 5 to 10% of its compute. SmolLM2 and SmolLM3 use a warmup-stable-decay schedule, which holds the peak learning rate and decays only at the end.
- Post-training in rounds: Llama 3 repeats SFT, rejection sampling and DPO "in six rounds," each with new data sampled "from the latest models." Our Parts 10 to 12 ran each stage once.

### What the recipes leave out

DeepSeek-V3's $5.576M covers "only the official training," "excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data." Neither Meta nor DeepSeek released its pretraining data, so the step [Part 3](/blog/pretraining-data-pipeline/) built by hand, filtering and deduplicating the web, can't be rerun from their papers. OLMo 2 is the exception, with data, code, logs and checkpoints public, and SmolLM2's authors "release both SmolLM2 as well as all of the datasets we prepared."

Our own numbers have limits too. Part 8's power law was fitted between 10¹⁶ and 10¹⁸ FLOPs, and its floor moved by 0.7 nats when one point was left out; carrying it seven orders of magnitude further is a guess. Our losses are in our tokenizer's units, so they can't be set next to anyone else's. And the abilities that matter most at the frontier, multi-step instructions, arithmetic, reasoning, never appeared at 88M, so our curves can't say when they arrive.

## Step 2: exporting to Hugging Face

### Which chat model

On Part 10's 100 chat prompts, [Part 11](/blog/dpo-from-scratch/)'s on-policy DPO model ended 73% of its greedy replies, against 33% for [Part 10](/blog/sft-from-scratch/)'s SFT model and 24% for [Part 12](/blog/grpo-from-scratch/)'s GRPO model trained from a chat-mixed warm start. All three scored alike on the benchmarks. GRPO taught the third to add, but a chat model that stops when it's done is what a demo needs, so the export uses the on-policy DPO weights.

### Same weights, different names

[Part 4](/blog/write-gpt-model/) already ran our weights through `LlamaForCausalLM` (transformers 5.20.2) and found the four settings that matter. The new module `export.py` turns that check into a checkpoint anyone can load: a `config.json`, the weights in `model.safetensors`, the tokenizer and its chat template.

| In `model.py` | In `LlamaForCausalLM` | What has to match |
|---|---|---|
| `attn.wq`, `attn.wk` | `q_proj`, `k_proj` | Rows reordered per head: our RoPE rotates pairs (2i, 2i+1), Hugging Face's rotates (i, i + 32) |
| `attn.wv`, `attn.wo`, `mlp.gate`, `mlp.up`, `mlp.down` | `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Copied as they are; no biases |
| `attn_norm`, `mlp_norm`, `norm` | `input_layernorm`, `post_attention_layernorm`, `model.norm` | `rms_norm_eps=1e-5` (the library's default is 1e-6) |
| `tok_emb` (also the output layer) | `embed_tokens`, `lm_head` | `tie_word_embeddings=True`, so the 25.2M-parameter table is stored once |
| `rope_base`, `max_seq_len` | `rope_theta`, `max_position_embeddings` | 10,000 and 1,024 |

The reordering is one line:

```python
def permute(w, n_heads):
    """Our RoPE pairs (2i, 2i+1) -> Hugging Face's pairs (i, i + head_dim/2), per head (rows of wq/wk)."""
    out, d = w.shape
    return w.view(n_heads, out // n_heads // 2, 2, d).transpose(1, 2).reshape(out, d)
```

The tokenizer needs no conversion: Part 2 trained it with Hugging Face's `tokenizers`, so `tokenizer.json` is copied unchanged. What it lacked were the settings around it: `<|endoftext|>` as the start token, `<|end|>` as the end token for the chat model, and Part 10's template written in Jinja, the template language both transformers and llama.cpp read:

```python
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' + (message['content'] | trim) + '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)
```

The template writes `<|endoftext|>` itself because transformers' generic tokenizer class ignores the `add_bos_token` setting; llama.cpp honors that setting and removes the template's copy so the token isn't doubled.

### Checking the export

Run from the shared code folder, with the GPU:

```bash
python export.py ../t11/runs/onpolicy/model_final.pt ../t15/export/chat88 --chat --check
```

```text
wrote ../t15/export/chat88 ['config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']
  796 tokens  max |diff| 2.10e-05  argmax agree 100.00%
  543 tokens  max |diff| 1.91e-05  argmax agree 100.00%
  434 tokens  max |diff| 1.76e-05  argmax agree 100.00%
  987 tokens  max |diff| 2.29e-05  argmax agree 100.00%
worst max |diff|: 2.288818359375e-05
```

`--check` reloads the folder with `AutoModelForCausalLM`, as any user would, and compares FP32 logits with `model.py`'s on four held-out documents: at most 2.3 × 10⁻⁵ apart, with the same top token everywhere (4.4 × 10⁻⁵ for the base model). On 310 held-out documents the exported tokenizer produced Part 2's ids exactly, and `apply_chat_template` matched Part 10's `render` on one-turn and four-turn conversations.

## Step 3: GGUF and llama.cpp

### Building llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) is the C and C++ engine under most local LLM tools, and GGUF is its single-file format holding weights, tokenizer, chat template and settings ([Inference Part 13](/blog/local-llm-inference/) covers both). The build is pinned to git tag b11982 (commit `8a199ad`, 29 October 2026). This WSL2 install has no CUDA toolkit, so it's a CPU build, with `cmake` from PyPI:

```bash
git clone --depth 1 --branch b11982 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build -j 8 --target llama-cli llama-server llama-bench llama-perplexity llama-quantize llama-tokenize
```

`GGML_NATIVE` targets this CPU, an AMD Ryzen 9 8945HS with 8 cores and 16 threads, including its AVX-512 instructions. The build took about a minute and a half.

### Converting: one missing package, one unknown tokenizer

`convert_hf_to_gguf.py` reads Step 2's folder. It stopped twice. First, its Llama converter needs the `sentencepiece` package just to find out the model has no SentencePiece tokenizer; installing it fixed that. Second, it failed with "BPE pre-tokenizer was not recognized." The pre-tokenizer is the regular expression that splits text into words, numbers and punctuation before BPE merges run ([Part 2](/blog/train-bpe-tokenizer/)). llama.cpp doesn't read that regex from `tokenizer.json`; it tokenizes a fixed test string, hashes the resulting ids, and looks the hash up in a list of known models. Ours, `f12d392d…`, isn't on it.

Part 2's regex is GPT-4's pattern with numbers split into single digits, and that's exactly the pattern llama.cpp implements as `qwen2`. So instead of editing llama.cpp's source, a small wrapper answers "qwen2" when the folder's regex matches, and runs the converter unchanged:

```python
# convert_gguf.py (excerpt): run llama.cpp's converter, telling it which pre-tokenizer we use
QWEN2_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

sys.path[:0] = [str(LLAMA), str(LLAMA / "gguf-py")]
from conversion import base

stock = base.TextModel.get_vocab_base_pre

def get_vocab_base_pre(self, tokenizer):
    pre = json.load(open(self.dir_model / "tokenizer.json"))["pre_tokenizer"]
    if pre["pretokenizers"][0]["pattern"]["Regex"] == QWEN2_REGEX:   # Part 2's PATTERN
        return "qwen2"
    return stock(self, tokenizer)

base.TextModel.get_vocab_base_pre = get_vocab_base_pre
runpy.run_path(str(LLAMA / "convert_hf_to_gguf.py"), run_name="__main__")
```

A wrong answer would still convert, and the model would then tokenize some text differently with no error. So the check is direct: `llama-tokenize` on the 1.43 MB held-out slice used below produced 310,414 ids, every one equal to Part 2's tokenizer's. The converter also read the start and end tokens and the chat template from Step 2's files, and undid the query and key reordering, since llama.cpp's RoPE pairs neighbours as ours does.

### Quantizing, and what it costs

```bash
python convert_gguf.py export/chat88 --outtype f32 --outfile gguf/chat88-f32.gguf
python convert_gguf.py export/chat88 --outtype f16 --outfile gguf/chat88-f16.gguf
llama-quantize gguf/chat88-f32.gguf gguf/chat88-q8_0.gguf Q8_0
llama-quantize gguf/chat88-f32.gguf gguf/chat88-q4_k_m.gguf Q4_K_M
```

Each step took under a second. Q8_0 stores blocks of 32 weights as 8-bit integers with one scale each; Q4_K_M is the k-quant mix from [Inference Part 8](/blog/quantization/), 4-bit for most matrices and 6-bit for the ones it treats as sensitive. Q4_K_M kept 11 of the 71 matrices at 6 bits: the tied embedding and the value and down projections in half the layers.

The held-out text is the first 319 documents of Part 3's validation shard, 1.43 MB. `llama-perplexity` cuts it into 303 chunks of 1,024 tokens, starts each with `<|endoftext|>` and scores each chunk's second half, so every scored token has at least 512 tokens of context; `ppl_reference.py` scores the same 154,833 tokens the same way with PyTorch in FP32 on the GPU. Speeds are 8 CPU threads, the range over two `llama-bench` runs of 5 repetitions each:

| Chat model as | File size | Bits per weight | Perplexity | Change vs F32 | KL divergence vs F32 | Same top token as F32 | Prompt, tokens/s | Generation, tokens/s |
|---|---|---|---|---|---|---|---|---|
| PyTorch, FP32 | 352 MB (weights) | 32 | 47.911 | reference | none | none | not measured | not measured |
| GGUF F32 | 353.6 MB | 32 | 47.911 ± 0.40 | 0.000% | none | none | 3,088 to 3,120 | 133 to 136 |
| GGUF F16 | 177.4 MB | 16 | 47.911 | 0.00% | 0.000001 | 99.94% | 3,767 to 3,992 | 213 to 252 |
| GGUF Q8_0 | 94.8 MB | 8.50 | 47.895 | −0.03% | 0.0003 | 98.93% | 3,131 to 3,308 | 418 to 456 |
| GGUF Q4_K_M | 59.6 MB | 5.30 | 48.349 | +0.91% | 0.014 | 92.71% | 4,531 to 4,889 | 583 to 615 |

KL divergence, the expected extra surprise from using the quantized model's next-token distribution in place of the F32 one, was measured on the first 40 chunks; "same top token" is how often both pick the same most likely next token there. The base model behaves the same way: its F32 file scores 28.3675, equal to PyTorch's 28.3675, and Q4_K_M costs it 1.02%.

![Two stacked line charts against GGUF file size on a log scale from 50 to 400 MB. Top: perplexity change relative to F32 for the chat model in blue and the base model in orange. F32 at 354 MB and F16 at 177 MB are at zero; Q8_0 at 95 MB is at minus 0.03% for the chat model and plus 0.03% for the base model; Q4_K_M at 60 MB rises to plus 0.91% and plus 1.02%. Bottom: generation speed on 8 CPU threads, rising from 134 tokens per second for F32 to 232 for F16, 437 for Q8_0 and 599 for Q4_K_M.](/diagrams/t15-quant.svg "Quantizing the 88M model: what each GGUF format costs in perplexity and buys in generation speed on the laptop CPU.")

The conversion is exact: F32 in llama.cpp reproduces PyTorch's perplexity to four decimals, which is a stronger check than the logits in Step 2 because it runs through llama.cpp's own tokenizer, RoPE and attention. Q8_0 is free, and Q4_K_M costs 0.91%, the same as the [k-quants pull request](https://github.com/ggml-org/llama.cpp/pull/1684) measured on LLaMA 7B, but at 5.30 bits per weight against the 4.89 it averages on Llama 3.1 8B, because Q4_K_M keeps the tied embedding, 29% of this model, at 6 bits. And generation speed follows file size, since each new token reads every weight once: 134 tokens per second at F32 is 47 GB/s of weights, about half of the 89.6 GB/s the laptop's two DDR5-5600 sticks can deliver ([Inference Part 13](/blog/local-llm-inference/) works through that formula). Prompt processing, which batches 512 tokens per weight read, gained only 1.5 times.

At 16 threads, one per hardware thread, every format was slower and far noisier. Mind the units: `llama-perplexity` reports perplexity per token, e to the mean nats per token. Converted to bits per byte over the same scored text, the chat model scores 1.212 and the base model 1.048, close to Part 9's 1.045 for the base model with a different windowing. So Parts 10 and 11 cost the model 0.16 bits per byte on raw web text, a cost no chat benchmark showed.

### Talking to it

`llama-server` serves the Q4_K_M file with an OpenAI-compatible API, applying the chat template stored in the GGUF:

```bash
llama-server -m gguf/chat88-q4_k_m.gguf --jinja -c 1024 -t 8 --port 8089
curl -s localhost:8089/v1/chat/completions -d '{"messages": [{"role": "user",
  "content": "What is the capital of France?"}], "temperature": 0, "max_tokens": 256}'
```

The rendered prompt is `<|user|>What is the capital of France?<|end|><|assistant|>`, with `<|endoftext|>` added in front as the start token. Greedy replies, verbatim:

> User: What is the capital of France?
>
> Assistant: The capital of France is Paris.
>
> User: What is it famous for?
>
> Assistant: The capital of France is Paris.

> User: Explain photosynthesis to a ten-year-old in two sentences.
>
> Assistant: Photosynthesis is a vital process in the life of plants, algae, and other organisms. It is a process that involves the conversion of sunlight into chemical energy, which is then used to convert carbon dioxide and water into glucose.

> User: What is 17 + 25?
>
> Assistant: To find 17 + 25, we'll use the formula for the sum of the numbers in the given range. First, let's find the sum of the numbers. 17 + 25 = 17 17 + 25 = 17 17 + 25 = 17 …

The first and second replies stopped on their own after 8 and 46 tokens; the sum ran to the 256-token limit. Each reply streamed at 480 to 575 tokens per second on the CPU. Served from the F16 file instead, all six test prompts got greedy replies word for word identical to `model.py`'s in FP32 PyTorch, with the same lengths. The Q4_K_M replies matched on the first prompt and drifted to different wording after a few words on the other five, as a 7% disagreement on the top token predicts.

## Step 4: the series in one table

### What each part built and measured

| Part | What it built | The number that mattered |
|---|---|---|
| [1: what a laptop can train](/blog/build-llm-from-scratch/) | The budget, from measured speed and 6ND | 88M parameters on 1.76B tokens, planned at about 17 hours |
| [2: tokenizer](/blog/train-bpe-tokenizer/) | Byte-level BPE, 32,768 tokens, trained in 42 seconds | 4.61 bytes per token on held-out text, against 4.06 for Llama 2's tokenizer |
| [3: data](/blog/pretraining-data-pipeline/) | Download, MinHash dedup, packing into shards | 42,867 near-duplicates removed; 2.47B training tokens |
| [4: model](/blog/write-gpt-model/) | `model.py`, 179 lines | 88,096,512 parameters, equal to the formula; Hugging Face's Llama within 0.0003 |
| [5: training loop](/blog/llm-training-loop/) | AdamW, warm-up, cosine, clipping, BF16, accumulation | A batch 8 times smaller ended 1.57 lower at equal tokens |
| [6: speed](/blog/fast-llm-training/) | BF16, `torch.compile`, a bigger micro-batch | 5,783 to about 30,000 tokens per second, 74% of the measured peak |
| [7: pretraining run](/blog/pretraining-run/) | 3,360 steps, unattended | 16 h 49 min, 0.83 kWh, validation loss 3.365 |
| [8: scaling laws](/blog/scaling-laws-experiment/) | Five more sizes, two fitted laws | Exponent 0.068 against Kaplan's 0.076; best ratio 22 tokens per parameter |
| [9: evaluation](/blog/evaluate-base-model/) | A 215-line harness matching lm-evaluation-harness | 1.045 bits per byte; 47.9% ARC-Easy, 28.0% HellaSwag |
| [10: SFT](/blog/sft-from-scratch/) | Chat template, loss masking, 26,000 conversations | Reply loss 2.82 to 1.96 in 15 minutes; the end token from 19.0 nats to 0.73 |
| [11: DPO](/blog/dpo-from-scratch/) | The DPO loss with cached reference log-probs | UltraFeedback pairs grew replies 45%; on-policy pairs made 97% end |
| [12: GRPO](/blog/grpo-from-scratch/) | GRPO with a Python checker | Addition from 17.8% to 77.4% in 20 minutes, and a leading-zero loophole |
| [13: distillation](/blog/distillation-synthetic-data/) | Logit and sequence distillation | 0.21 nats for 2.7 times the time; style transferred more than knowledge |
| [14: memory](/blog/training-memory/) | Memory accounting, checkpointing, 8-bit AdamW | Activations predicted within 0.6%; a 1B model fits in 8 GB at 2,347 tokens per second |
| 15: this part | `export.py`, GGUF files, llama.cpp | Logits within 2.3 × 10⁻⁵; F32 perplexity equal to PyTorch's to four decimals |

![A top-to-bottom flowchart of the series. Inside a box labeled Pretraining, Parts 1 to 7: the tokenizer with 32,768 BPE tokens leads to the data, 2.47 billion FineWeb-Edu tokens, which leads to the 88M model trained on 1.76 billion tokens in 16.8 hours. From the model, three branches: the scaling fit and evals of Parts 8 and 9, SFT with a chat template in Part 10, and distillation in Part 13. SFT leads to on-policy DPO in Part 11 and GRPO on sums in Part 12. The DPO model leads to a Hugging Face Llama checkpoint, then GGUF files in F16, Q8_0 and Q4_K_M, then llama.cpp on the laptop CPU.](/diagrams/t15-series-pipeline.svg "The series' pipeline, from tokenizer to a model running in llama.cpp.")

### What surprised us

- Nothing crashed when memory ran out. On this WSL2 laptop, a micro-batch too big for 8 GB spilled into system memory and trained 6 to 16 times slower with no error (Part 1).
- The biggest jumps in Part 7's training loss were hard batches, not instability: snapshots from before and after both scored them high.
- The preference data, not the DPO loss, set the direction: the same loss made replies 45% longer on UltraFeedback's pairs and shorter on the model's own (Part 11).
- Reinforcement learning found a hole in a five-line checker within 250 steps (Part 12), and Hinton's classic distillation temperature of 2 made the student worse than no teacher (Part 13).

### What an 88M model can and can't do

It writes fluent English about the topics FineWeb-Edu covers, answers "What is the capital of France?" correctly in eight tokens, and stops. It beats GPT-2 small on ARC-Easy, 47.9% to 43.8%, because its data was filtered for school material. It can't do arithmetic it wasn't trained on: asked 17 + 25, the chat model repeats "17 + 25 = 17" until the token limit. It invents facts, loses the thread after one turn, and scores 28% on HellaSwag, three points above guessing. Every one of those limits is the base model's: post-training changed the format of its answers and left its knowledge where it was.

### With more compute

Three next steps, sized with the series' own measurements:

- The same 88M model on 200 tokens per parameter, 17.6B tokens: 6.9 days on this laptop at Part 7's 29,400 tokens per second. SmolLM2-135M shows how far over-training goes on FineWeb-Edu-like data.
- A 1B model on 20B tokens, 1.2 × 10²⁰ FLOPs: 99 days on this laptop at Part 14's 2,347 tokens per second, or about 100 H100-hours at Llama 3's 342 trillion FLOPs per GPU-hour, roughly $200 at DeepSeek's assumed $2 per hour.
- Post-training in rounds: sample from the DPO model, rank its replies, train again, as Llama 3 did six times. Part 11's one on-policy round spent 14 minutes sampling 2,816 replies and 100 seconds ranking them before a 42-step DPO run.

The code for every part is in the repo's [`code/llm-from-scratch/`](https://github.com/vidyalai-in/vidyalai/tree/main/code/llm-from-scratch) folder, with each part's scripts and small result files under `experiments/`; `code/README.md` lists which file belongs to which part.

## In the real world

- OLMo 2's paper reports 131 MWh of GPU power for pretraining OLMo 2 7B and 1,022 MWh for Llama 3.1 8B. Part 7's run drew 0.834 kWh, 157,000 times less than OLMo 2 7B.
- DeepSeek-V3 spent "180K H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs" per trillion tokens. At Part 7's pace, this laptop would need about 400 days for one trillion tokens of our much smaller model.
- Failure is routine at scale: in 54 days of Llama 3 405B's pretraining there were 466 job interruptions, 419 unexpected, "approximately 78%" of them hardware issues. Part 7's run had none.
- MFU (model FLOPs utilization) depends on the yardstick. Llama 3 405B ran at 38 to 43% of the H100's spec-sheet peak; our 72% was against this laptop's measured peak, and about 51% against its spec sheet counting only 6N (Part 1).

## Common misconceptions

- **"A frontier model is our model with more layers."** The block is the same, but the recipe isn't: DeepSeek-V3 routes each token through 37B of its 671B parameters, trains in FP8, predicts an extra token, and extends its context in two stages. Llama 3 405B is the closest to a scaled-up copy of ours, and it still needed 16,384 GPUs in four kinds of parallelism.
- **"DeepSeek-V3 cost $5.6 million to build."** That's 2.788M GPU-hours at an assumed $2 rental price for the final run only; the report itself excludes "prior research and ablation experiments on architectures, algorithms, or data."
- **"Compute-optimal is what the labs train."** Of the dense runs above, only Llama 3 405B sits near its compute-optimal size. SmolLM2-135M trained on 14,800 tokens per parameter and OLMo 2 7B on 580, because a smaller model is cheaper to serve for its whole life.
- **"Quantizing to 4 bits makes a model 8 times smaller than FP32."** Our Q4_K_M file is 5.9 times smaller, at 5.30 bits per weight, because llama.cpp keeps the tied embedding, 29% of this model, in 6-bit Q6_K. In bigger models the embedding is a small share: Llama 3.1 8B's Q4_K_M file is 4.92 GB, about 4.9 bits per weight.
- **"If the converter runs, the model is right."** A wrong pre-tokenizer runs fine and tokenizes some text differently. What proved ours right was 310,414 identical token ids and a perplexity equal to PyTorch's to four decimals.

## Key takeaways

- Our 88M model used 9.3 × 10¹⁷ FLOPs. SmolLM2-135M used about 1,700 times that, OLMo 2 7B 190,000 times, DeepSeek-V3 3.5 million times and Llama 3 405B 41 million times, at about 22 times our per-GPU speed.
- The frontier block is still ours: RMSNorm, RoPE, grouped-query attention, SwiGLU. What grows is everything around it: data measured in trillions of tokens, mixtures of experts, FP8, long-context stages, thousands of GPUs and rounds of post-training.
- Exporting a homemade model is mostly bookkeeping: rename the weights, reorder the query and key rows for RoPE, set the norm epsilon and tied embeddings, and ship the tokenizer with its chat template. Then check it: logits within 2.3 × 10⁻⁵, the same 310,414 token ids, the same perplexity.
- In llama.cpp the chat model runs as a 59.6 MB file at Q4_K_M, generating about 600 tokens per second on the laptop CPU, with perplexity 0.9% higher than F32.
- Fourteen parts on one laptop produced a model that writes fluent English, answers simple questions and stops, but can't add, invents facts and loses the thread. Post-training changed how it answers; what it knows came from 1.76 billion tokens of pretraining.

## What's next

This is the last part of Build an LLM From Scratch; the [series page](/series/llm-from-scratch/) has the full reading order. To run models like this one faster and at scale, start with [How LLM Inference Works](/blog/how-llm-inference-works/), and for the kernels underneath, [From torch.matmul to Silicon](/blog/gpu-from-python-to-silicon/). Two series still to come will reuse these checkpoints: Inside the Model: Interpretability, which opens the 88M model's snapshots to watch features form, and Measuring LLMs, on how far to trust the benchmarks used here.

## Further reading

- [The Llama 3 Herd of Models (Llama Team, Meta, 2024)](https://arxiv.org/abs/2407.21783): the 405B recipe, its scaling-law extrapolation, 4D parallelism, failures in a 54-day window and six rounds of post-training.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437): MoE with MLA and multi-token prediction, FP8 training and the 2.788M GPU-hour cost table.
- [2 OLMo 2 Furious (OLMo team, 2024)](https://arxiv.org/abs/2501.00656): a fully open recipe, with data, mid-training, Tülu 3 post-training and energy use.
- [SmolLM2: When Smol Goes Big (Ben Allal et al., 2025)](https://arxiv.org/abs/2502.02737) and [SmolLM3: smol, multilingual, long-context reasoner (Hugging Face, 2025)](https://huggingface.co/blog/smollm3): the small-model recipes closest to ours.
- [Llama 3.1 model card (Meta)](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md): GPU-hours per model size.
- [llama.cpp (ggml-org)](https://github.com/ggml-org/llama.cpp): the engine used here, with `convert_hf_to_gguf.py`, `llama-quantize`, `llama-perplexity` and `llama-bench`.
- [GGUF specification (ggml)](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): the single-file format, its metadata keys and tensor types.
- [Chat templates (Hugging Face transformers docs)](https://huggingface.co/docs/transformers/main/en/chat_templating): `apply_chat_template`, generation prompts, and why templates should carry their own special tokens.
