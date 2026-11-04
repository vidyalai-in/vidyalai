---
title: "Audio Tokens and Voice Agents: Codecs, Turn-Taking and Latency"
description: "How neural codecs turn speech into tokens, how speech-to-speech models work, and where a voice agent's latency goes, measured stage by stage on a laptop."
slug: "voice-agents"
category: "multimodal"
tags: ["multimodal", "speech", "agents", "latency", "streaming"]
series: "multimodal-ai"
seriesOrder: 11
pubDate: 2026-10-24
updatedDate: 2026-11-04
---

## TL;DR

A voice agent must turn sound into tokens a model can read and write, and answer before the silence gets awkward. Mimi, the codec inside the Moshi voice model, keeps speech intelligible at 100 tokens per second (1.1 kbps), though its "semantic" first codebook alone turns "Three trucks arrived at dawn" into "Great car ride it on". A cascaded agent (Silero VAD, Whisper, Claude Haiku 5.5, Kokoro), timed stage by stage on a laptop, answers in about 2 seconds; after the LLM, the biggest cost is waiting to be sure the user has stopped, and streaming the first sentence into speech cuts the median from 2.41 s to 2.02 s. Speech-to-speech models like Moshi fold the chain into one network to approach the 200 ms gap of human conversation.

## Why it matters

People answer each other fast. Across 10 languages, [Stivers et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2705608/) measured a mean gap of 208 milliseconds between the end of a question and the start of its answer; the most common gap was zero.

On this laptop, a voice agent built from four open and hosted parts takes a median 2.41 seconds from the end of the user's speech to the first sound of its answer when each part waits for the one before it to finish. Streaming the first sentence of the reply into the speech synthesizer brings that down to 2.02 seconds. The language model takes the largest share. Next comes the 0.6 seconds the agent sits in silence to be sure the user has stopped, which involves no model at all.

The other half is the audio itself. To speak, a model must write sound as tokens, and a neural audio codec supplies them. [Part 1](/blog/multimodal-ai-overview/) counted Mimi's tokens; this part measures what they keep.

## The intuition

Think of two people on walkie-talkies. Only one can talk at a time, and since nobody says "over", the listener waits for a stretch of silence before pressing the button. Wait too briefly and you cut the other person off mid-thought; wait too long and the conversation drags. Then the listener has to think of an answer, and only then start speaking.

A cascaded voice agent is that walkie-talkie user. It hears through a speech recognizer, thinks with a text language model and speaks through a text-to-speech system, one after another, while a silence detector decides when its turn starts. It's an agent in the sense of [Agentic Part 1](/blog/what-is-an-ai-agent/), a model in a loop, with sound at both ends. A model like Moshi is closer to a phone call: both channels stay open, so it can listen while it talks.

The analogy breaks twice. People predict the end of a turn from its words and melody, so they start answering before the silence; a silence detector can't. And an agent can speak its first sentence while the rest is still being written.

## Step 1: speech as integers

A neural audio codec is an autoencoder for sound: an encoder turns the waveform into one vector per frame, a quantizer replaces each vector with integer codes, and a decoder turns codes back into a waveform, all trained together.

The quantizer is residual vector quantization (RVQ), introduced for neural codecs by SoundStream ([Zeghidour et al., 2021](https://arxiv.org/abs/2107.03312)). It "cascades Nq layers of VQ": the first codebook (a learned table of vectors) picks the entry nearest the frame vector, the second codes what the first missed, and so on. [Fundamentals Part 15](/blog/multimodal-llms/) works an example by hand. SoundStream also trained with "structured dropout applied to quantizer layers", a random number of codebooks per example, so "a single model can operate across variable bitrates from 3 kbps to 18 kbps". That quantizer dropout is why you can keep a codec's first few codebooks and drop the rest, as the experiment below does.

The size of the token stream is one multiplication:

```text
bitrate (bits/s) = frames per second × codebooks kept × bits per code
tokens per second = frames per second × codebooks kept
bits per code = log2(codebook size): 1,024 entries → 10 bits, 2,048 → 11 bits

Mimi:  12.5 frames/s × 8 codebooks × 11 bits = 1,100 bits/s = 1.1 kbps, 100 tokens/s
DAC:   75 frames/s × 32 codebooks × 10 bits  = 24,000 bits/s = 24 kbps, 2,400 tokens/s
```

| Codec | Audio in | Frames per second | Codebooks × bits | Bitrate | Tokens per second |
|---|---|---|---|---|---|
| Raw 16-bit audio | 24 kHz | 24,000 samples | 16 bits each | 384 kbps | 24,000 samples |
| SoundStream (Zeghidour et al., 2021) | 24 kHz | 75 | 8 × 10 at 6 kbps | 3-18 kbps | 300-1,800 |
| EnCodec (Défossez et al., 2022) | 24 kHz | 75 | up to 32 × 10 | 1.5-24 kbps | 150-2,400 |
| DAC, 44.1 kHz model (Kumar et al., 2023) | 44.1 kHz | 86 | 9 × 10 | 8 kbps | 775 |
| DAC, 24 kHz model (run here) | 24 kHz | 75 | 1-32 × 10 | 0.75-24 kbps | 75-2,400 |
| Mimi (Défossez et al., 2024; run here) | 24 kHz | 12.5 | 8 × 11 in Moshi; 32 in the checkpoint | 1.1 kbps (4.4 with 32) | 100 (400) |
| Text transcript | | | | | about 3-4 |

Frame rates and codebook sizes come from each paper; the Moshi paper, for example, uses "Q = 8 quantizers, each with a codebook size of NA = 2048. At 12.5Hz, this represents a bitrate of 1.1kbps," and notes that English speech needs "around 3 to 4 text tokens per second."

That last row is why the frame rate matters. A model that writes audio produces every token one step at a time. At 2,400 tokens per second a 10-second reply is 24,000 steps; at Mimi's 100 it's 1,000, still about 30 times more than the words need.

DAC ([Kumar et al., 2023](https://arxiv.org/abs/2306.06546)) attacked codebook collapse, "where a fraction of the codes are unused", by looking up codes "in a low-dimensional space (8d or 32d)" with L2-normalized vectors, and applied quantizer dropout only half the time ("p = 0.5").

Mimi's round trip in code, with the Hugging Face weights (the noise input is a placeholder for 24 kHz speech):

```python
import torch
from transformers import MimiModel

mimi = MimiModel.from_pretrained("kyutai/mimi").to("cuda").eval()
wav = 0.1 * torch.randn(1, 1, 10 * 24000, device="cuda")  # 10 s at 24 kHz
with torch.no_grad():
    codes = mimi.encode(wav, num_quantizers=8).audio_codes  # integers 0-2047
    out = mimi.decode(codes).audio_values
print(codes.shape, int(codes.max()), out.shape)
```

```text
torch.Size([1, 8, 125]) 2044 torch.Size([1, 1, 240000])
```

Ten seconds of audio, 240,000 samples, become an 8 × 125 grid of integers: 1,000 tokens.

## Step 2: semantic and acoustic tokens, measured

AudioLM ([Borsos et al., 2022](https://arxiv.org/abs/2209.03143)) used two token streams: semantic tokens, clustered features of the self-supervised speech model w2v-BERT at 25 per second, and acoustic tokens from SoundStream. "The semantic tokens enable long-term structural coherence, while modeling the acoustic tokens conditioned on the semantic tokens enables high-quality audio synthesis."

Mimi builds both into one codec. Training pushes its first codebook toward the features of WavLM, another self-supervised model (semantic distillation), and a "split RVQ" keeps that from hurting quality: "we distill semantic information into a plain VQ and apply an RVQ with 7 levels in parallel." In the paper's ablation, a phonetic discrimination error (ABX, lower is better) falls from 23.3% without distillation to 8.1%. So the first Mimi code should carry what was said, and the rest how it sounded. The experiment checks that.

### The codec experiment

The test audio is 32 sentences written for this article, each read by four Kokoro-82M stock voices (`af_heart`, `am_michael`, `bf_emma`, `bm_george`; [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0) at 24 kHz: 128 clips of 3.7 to 5.8 seconds, 9.7 minutes and 1,564 words in all. Each clip went through two codecs on the GPU in fp32, keeping 1, 2, 4, 8, 16 or 32 codebooks:

- Mimi, `kyutai/mimi` (CC-BY-4.0, 385 MB).
- DAC's 24 kHz model, `descript/dac_24khz` (299 MB; its repository says "Weights are released as part of this repo under MIT license").

EnCodec's Hugging Face card states no license for its weights, and its repository's MIT license covers "the code", so EnCodec appears here only through its paper.

Two scores say what survived. Word error rate (WER) is the share of words Whisper large-v3-turbo (`openai/whisper-large-v3-turbo`, MIT, fp16, greedy) gets wrong in the round-tripped audio: substitutions, deletions and insertions against the sentence as written, after Whisper's English normalizer. STOI (short-time objective intelligibility) compares short-time spectra of the original and the round trip and gives a score from 0 to 1, designed to predict how intelligible degraded speech is to listeners. Intervals are 95% bootstrap intervals over clips.

| Codec, codebooks kept | Bitrate | Tokens per second | WER (95% CI) | STOI (95% CI) | Encode 10 s | Decode 10 s |
|---|---|---|---|---|---|---|
| Original audio | 384 kbps | | 0.32% (0.06-0.64) | 1 | | |
| Mimi, 1 | 0.14 kbps | 12.5 | 39.1% (34.7-43.6) | 0.475 (0.470-0.480) | 27 ms | 26 ms |
| Mimi, 2 | 0.28 kbps | 25 | 2.81% (2.06-3.60) | 0.783 (0.780-0.786) | 28 ms | 26 ms |
| Mimi, 4 | 0.55 kbps | 50 | 1.09% (0.57-1.73) | 0.865 (0.861-0.868) | 29 ms | 28 ms |
| Mimi, 8 (Moshi's setting) | 1.1 kbps | 100 | 0.45% (0.13-0.82) | 0.919 (0.916-0.921) | 31 ms | 32 ms |
| Mimi, 32 | 4.4 kbps | 400 | 0.45% (0.13-0.82) | 0.972 (0.971-0.973) | 39 ms | 29 ms |
| DAC, 1 | 0.75 kbps | 75 | 6.71% (5.08-8.64) | 0.773 (0.769-0.778) | 119 ms | 212 ms |
| DAC, 2 | 1.5 kbps | 150 | 0.58% (0.20-1.02) | 0.858 (0.854-0.862) | 119 ms | 215 ms |
| DAC, 4 | 3 kbps | 300 | 0.45% (0.13-0.82) | 0.919 (0.917-0.922) | 121 ms | 215 ms |
| DAC, 8 | 6 kbps | 600 | 0.38% (0.13-0.71) | 0.960 (0.959-0.961) | 123 ms | 216 ms |
| DAC, 32 | 24 kbps | 2,400 | 0.32% (0.06-0.64) | 0.996 (0.996-0.996) | 136 ms | 224 ms |

Times are medians of 10 runs on one 10.0-second clip after 3 warm-up runs, with the GPU at 1,875 to 2,160 MHz and 41 to 49 W (nvidia-smi, sampled during the runs). The 16-codebook rows are in the chart.

![Two line charts sharing a log-scale bitrate axis from 0.1 to 30 kbps, one line for Mimi and one for DAC 24 kHz, each point labeled with the number of codebooks kept. Left, word error rate on a log scale: Mimi falls from 39% with 1 codebook to 2.8% with 2, 1.1% with 4 and 0.45% with 8, 16 and 32; DAC falls from 6.7% with 1 codebook to 0.58% with 2 and about 0.3 to 0.45% from 4 up, close to the original audio's 0.32%. Right, STOI: Mimi rises from 0.48 with 1 codebook through 0.78, 0.87, 0.92 and 0.95 to 0.97 with 32; DAC rises from 0.77 with 1 through 0.86, 0.92, 0.96 and 0.98 to 0.996 with 32. At any given bitrate, Mimi's line sits above DAC's in STOI and below it in error rate.](/diagrams/v11-codec-quality.svg "Word error rate and STOI after a codec round trip, 128 Kokoro clips. Bars: 95% bootstrap intervals over clips. Mimi reaches the same quality as DAC at about a third of the bits.")

Mimi gets more out of each bit. With 8 codebooks (1.1 kbps, 100 tokens per second) it matches DAC with 4 (3 kbps, 300 tokens per second) on both scores: 0.45% WER, STOI 0.919. Two Mimi codebooks (0.28 kbps) beat one DAC codebook, which spends nearly three times the bits (0.75 kbps).

WER stops improving at 8 Mimi codebooks while STOI climbs from 0.919 to 0.972 at 32: the last 24 codebooks carry how the words were said.

With only the semantic first codebook, Whisper's error rate was 39%: "Three trucks arrived at dawn with fresh bread for the market" came back as "Great car ride it on with fresh bread from our head." That code, one 11-bit number every 80 ms, was trained to make phonemes easy for a model to tell apart. Moshi predicts it first in each frame and fills in the 7 acoustic codes that make it sound like speech.

Kokoro's voices are clean, so the test was repeated on real recordings: 64 utterances of 3 to 12 seconds drawn at random (seed 0) from LibriSpeech test-clean ([openslr.org/12](https://www.openslr.org/12), CC-BY 4.0; measured only, never played here), 1,215 words. The ranking held: Mimi with 8 codebooks 3.87% WER (2.45-5.57), DAC with 4 3.95% (2.49-5.61), STOI 0.903 and 0.904, against 2.14% (1.02-3.53) for the original; Mimi's first codebook alone 40.3%. Real speech hit the lowest bitrates harder, so synthetic test speech flatters codecs: one DAC codebook went from 6.7% to 28.3%.

Both codecs are fast: Mimi encoded or decoded 10 seconds in about 30 ms, and DAC decoded it in about 215 ms. What a codec adds to a conversation is its frame: Mimi can't emit a code before it has heard 80 ms.

Here is one of the test sentences three times: Kokoro's original, then Mimi with 8 codebooks, then Mimi with only the first codebook.

<audio controls preload="none" src="/audio/v11-mimi-codebooks.mp3"></audio>

*Voice: Kokoro-82M stock voice `af_heart`, speed 1.0, text: "Three trucks arrived at dawn with fresh bread for the market." Played three times: the original, a Mimi round trip with 8 codebooks (1.1 kbps), and a Mimi round trip with 1 codebook (0.14 kbps), with half-second gaps. Mono MP3, 64 kbps.*

## Step 3: one model that listens and speaks

Moshi ([Défossez et al., 2024](https://arxiv.org/abs/2410.00037)) is the open reference design for speech-to-speech. At its core is Helium, "a 7B-parameter text LLM". An RQ-Transformer handles Mimi's grid: a large Temporal Transformer steps through time at 12.5 Hz and "a smaller Depth Transformer" predicts the 8 codebooks within each step, so the big model takes 12.5 steps per second instead of 100. Three choices make it conversational:

- Two streams. Moshi models "its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns." It hears the user while it speaks.
- Inner Monologue. It predicts "time-aligned text tokens as a prefix to audio tokens", writing its words as text just ahead of their audio.
- Acoustic delay. The acoustic codes lag the semantic one by a frame or two, because that "greatly improves the quality of the generation."

Mimi's 80 ms frame plus one frame of delay gives "a theoretical latency of 160ms, 200ms in practice". The paper compares this with "the 230 ms average in natural conversations" from Stivers et al.; that paper's own mean is 208 ms. Moshi's bf16 weights (`kyutai/moshiko-pytorch-bf16`) are about 15 GB, too large for this laptop's 8 GB GPU, so Moshi isn't measured here.

The closed systems document their interfaces and little else. OpenAI's GPT-4o system card says it "can respond to audio inputs in as little as 232 milliseconds, with an average of 320 milliseconds." OpenAI's Realtime API and Google's Gemini Live API stream raw 16-bit PCM audio, run voice activity detection on the server and let the user interrupt; neither documents its architecture or audio tokenizer.

| Aspect | Cascaded (recognizer, text LLM, synthesizer) | Speech-to-speech (Moshi, realtime APIs) |
|---|---|---|
| Turn-taking | A silence rule or a separate turn detector | Learned, full duplex in Moshi |
| Latency | Sum of stages; measured below | 200 ms for Moshi in practice; 320 ms average for GPT-4o (both reported, not reproduced) |
| What the LLM hears | Words only: "emotion and accent" and "non-speech audio" are lost (Moshi paper) | The audio itself |
| Control | Text in the middle: logs, tool calls, filters, any LLM | Harder to inspect or swap one part |
| Run here | Yes, every stage | No: Moshi needs about 15 GB in bf16 |

## Step 4: a cascaded voice agent, timed

![Flowchart of the voice agent loop. Microphone audio in 32 ms chunks goes to voice activity detection (Silero VAD). When silence lasts longer than the threshold, an endpoint fires: the user is done. Speech recognition (Whisper) transcribes the turn, a streamed LLM (Haiku 5.5) writes the reply, a sentence splitter passes the first complete sentence to text-to-speech (Kokoro), and the audio goes to the speaker. A second branch from the VAD: when speech starts while the agent talks, barge-in stops playback and cancels the reply.](/diagrams/v11-voice-loop.svg "The cascaded voice agent measured here. Every arrow on the left is on the critical path between the user's last word and the agent's first sound.")

### The setup

Every stage ran on this laptop: an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) and a 16-core CPU under WSL2, PyTorch 2.11, transformers 5.21. The user's side is 40 short requests written for this article ("Set a timer for twelve minutes, please."), spoken by the four Kokoro stock voices in turn, 2.6 to 4.7 seconds each, resampled to 16 kHz, padded with 0.5 s of silence before and 2 s after, over a faint noise floor (-60 dBFS). The end of speech is the last 10 ms window within 35 dB of the clip's loudest. The clock starts there and stops at the agent's first sound.

### Endpointing: has the user finished?

Voice activity detection (VAD) is a small classifier that gives, for each short chunk of audio, the probability that it contains speech. Silero VAD (silero-vad 6.2.3, MIT, "around two megabytes") takes 32 ms chunks at 16 kHz. Endpointing is the rule on top: the turn is over once the probability has stayed low for a set silence duration. Streaming one user turn:

```python
import soundfile as sf, torch
from silero_vad import load_silero_vad, VADIterator

model = load_silero_vad()
vad = VADIterator(model, threshold=0.5, sampling_rate=16000,
                  min_silence_duration_ms=500)
wav = torch.from_numpy(sf.read("turn.wav", dtype="float32")[0])  # 16 kHz mono
for i in range(0, len(wav) - 511, 512):              # 32 ms chunks, as from a mic
    event = vad(wav[i:i + 512], return_seconds=True)
    if event:
        print(f"at {(i + 512) / 16000:.2f} s: {event}")
```

```text
at 0.32 s: {'start': 0.3}
at 4.32 s: {'end': 3.8}
```

The detector places the end at 3.8 s but can only say so at 4.32 s. That turn was "I need a quick dinner idea. Something with rice and vegetables.", and the 500 ms rule waited through the pause between its sentences. Shorter rules don't:

| Silence rule | Endpoint delay, median (95% CI) | Endpoint delay, p90 | Turns cut off early (95% CI) |
|---|---|---|---|
| 100 ms | 221 ms (215-234) | 252 ms | 16 of 40, 40% (26-55) |
| 200 ms | 317 ms (311-330) | 348 ms | 6 of 40, 15% (7-29) |
| 300 ms | 413 ms (407-426) | 444 ms | 0 of 40 (0-9) |
| 500 ms | 605 ms (599-618) | 636 ms | 0 of 40 (0-9) |
| 800 ms | 893 ms (887-906) | 924 ms | 0 of 40 (0-9) |

Endpoint delay runs from the true end of speech to the moment the detector fires; p90 is the 90th percentile, the delay 1 turn in 10 exceeds. A turn is cut off early when the detector declares an end more than 50 ms before the real one (bootstrap intervals for delays, Wilson intervals for counts).

The delay is the silence rule plus about 95 to 120 ms, because the probability takes a few chunks to fall. A short rule cuts people off: at 100 ms, 16 turns ended at a pause between sentences or at a comma, and the agent would have answered half a question. Kokoro's pauses inside turns were short (24 over 100 ms, median 249 ms, longest 377 ms), so 300 ms sufficed here; people who stop to think pause longer.

The model is no bottleneck: 0.35 ms per chunk on one CPU thread (median of 39,165), within the README's "less than 1ms". Speech onsets were detected a median 55 ms after they began, so noticing a barge-in is fast; the harder parts are stopping playback, cancelling the LLM's stream, and telling an interruption from an "mm-hm".

### Speech recognition

After the endpoint, Whisper transcribes the turn. Each turn was timed 5 times after warm-up and the median kept; the table gives the median and p90 over the 40 turns.

| Whisper model | Time per turn, median (95% CI) | p90 | Word errors in 441 words | Peak VRAM | GPU clock, power |
|---|---|---|---|---|---|
| large-v3-turbo (809M, MIT) | 210 ms (206-214) | 223 ms | 0 | 1,598 MB | 2,025 MHz, 39 W |
| tiny (39M, MIT) | 83 ms (81-87) | 95 ms | 12, 2.7% (0.9-5.0) | 96 MB | 1,035 MHz, 10 W |

Whisper pads every clip to a 30-second window, so a short turn costs almost as much as a long one. Tiny's errors included "drains fast" heard as "drains first"; Part 9 measures Whisper's accuracy properly. A streaming recognizer, transcribing while the user talks, would take most of this stage off the critical path; it isn't built here.

### The LLM, streamed

Each turn's text went to `claude-haiku-5-5` (thinking disabled, effort at its default, `max_tokens` 200) with a 25-word system prompt asking for one to three short spoken sentences, through the Anthropic Python SDK's streaming interface. A sentence is complete when `.`, `!` or `?` is followed by a space. The 40 turns ran 3 times, interleaved, on November 4, 2026 between 22:41 and 22:56 IST over the public API from India; 118 of 120 calls finished before the run's time limit. Replies had a median of 80 output tokens and 234 characters, the first sentence 122. For comparison, Qwen2.5-1.5B-Instruct (Apache-2.0) ran locally in bf16, greedy, once per turn after warm-up, streamed with transformers' `TextIteratorStreamer`.

| LLM | First token, median (95% CI) | First token, p90 | First sentence, median | First sentence, p90 | Full reply, median | Full reply, p90 |
|---|---|---|---|---|---|---|
| Claude Haiku 5.5, API (118 calls) | 0.78 s (0.76-0.80) | 1.10 s | 1.03 s (1.00-1.07) | 1.35 s | 1.23 s (1.21-1.28) | 1.57 s |
| Qwen2.5-1.5B-Instruct, local (40 calls) | 0.053 s (0.051-0.071) | 0.076 s | 0.51 s (0.41-0.60) | 0.84 s | 0.82 s (0.69-1.21) | 1.81 s |

Claude's 0.78 s to the first token covers the network round trip, queueing and reading the prompt; then it writes fast, finishing the reply 0.45 s later. The local model starts in 53 ms and writes more slowly, and its answers were worse (asked how far the station is from the old town square, it offered "5-7 kilometers by car"); quality wasn't scored. The 118 Claude calls cost $0.0056.

### Text-to-speech

Speech synthesized for this article comes only from sentences written for it, so each reply was timed with a stand-in of the same length: for the first sentence, the written sentence nearest in character count; for the full reply, written sentences joined to within about 20 to 40 characters of it. Each stand-in ran 6 times on the GPU; the median of the last 5 was kept.

| Kokoro-82M on the GPU (118 Claude-length replies) | Median | p90 |
|---|---|---|
| Time to first audio, first sentence alone | 136 ms | 222 ms |
| Time to synthesize the whole reply | 335 ms | 460 ms |

A full reply was a median 17 seconds of speech, made in a third of a second (peak VRAM 813 MB, 2,055 MHz, 36 W). Part 10 measures Kokoro in detail.

### The whole budget

Adding the stages call by call gives the time from the user's last word to the agent's first sound. "Stream" sends the first sentence to Kokoro the moment it is complete; "wait" synthesizes the full reply in one call. Both use the 500 ms silence rule and Whisper large-v3-turbo. [Agentic Part 15](/blog/agent-cost-latency/) did the same accounting for text agents.

![Horizontal stacked bar chart, seconds from the end of the user's speech to the first sound of the reply, with a dashed line at the 0.21 s human turn gap. Each bar stacks the median endpoint wait, speech recognition, LLM and text-to-speech times. Claude Haiku 5.5 streaming the first sentence: 0.60 + 0.21 + 1.03 + 0.14, total median 2.02 s, 90th percentile 2.39 s. Claude waiting for the full reply: 0.60 + 0.21 + 1.23 + 0.33, median 2.41 s, p90 2.86 s. Qwen2.5-1.5B local streaming: 0.60 + 0.21 + 0.51 + 0.10, median 1.44 s, p90 1.83 s. Qwen waiting for the full reply: 0.60 + 0.21 + 0.82 + 0.26, median 1.89 s, p90 3.19 s.](/diagrams/v11-latency-budget.svg "The latency budget of a cascaded voice agent on the laptop. Segments are per-stage medians; the tick is the median of the per-call totals and the whisker reaches their 90th percentile.")

| Design | Median total (95% CI) | p90 total (95% CI) |
|---|---|---|
| Claude Haiku 5.5, stream the first sentence | 2.02 s (1.94-2.08) | 2.39 s (2.24-2.86) |
| Claude Haiku 5.5, wait for the full reply | 2.41 s (2.36-2.47) | 2.86 s (2.72-3.13) |
| Qwen2.5-1.5B local, stream the first sentence | 1.44 s (1.39-1.59) | 1.83 s (1.74-1.93) |
| Qwen2.5-1.5B local, wait for the full reply | 1.89 s (1.75-2.38) | 3.19 s (2.73-3.64) |

Streaming saved a median 0.38 s per call with Claude (95% CI 0.33-0.43) and 0.60 s with the slower-writing local model (0.37-0.75), and more in the tail, since waiting ties the first sound to the length of the whole answer.

With Claude, the first sentence is the largest piece, mostly the 0.78 s before the first token. Next comes the 0.60 s of silence the agent sits through before it believes the user has finished, more than recognition and synthesis together.

Even the fastest medians measured here, a 300 ms rule (0.41 s), Whisper tiny (0.08 s), Qwen's first sentence (0.51 s) and Kokoro (0.10 s), add up to about 1.1 seconds, five times the 208 ms human gap, before microphone and speaker buffers, echo cancellation and network hops for the audio. Speech-to-speech models were built to remove that chain.

## In the real world

- OpenAI's Realtime API defaults to the silence rule measured above. In `server_vad` mode, `silence_duration_ms` "Defaults to 500ms. With shorter values the model will respond more quickly, but may jump in on short pauses from the user," with a `threshold` that "defaults to 0.5". Here, the same 500 ms rule fired a median 605 ms after the last word.
- The same API offers `semantic_vad`, which "Chunks the audio when the model believes based on the words said by the user that they have completed their utterance." Its eagerness settings `low`, `medium` and `high` "have max timeouts of 8s, 4s, and 2s respectively." Barge-in is on by default: `interrupt_response` (default true) cancels "any ongoing response" when a VAD start event occurs. ([OpenAI docs](https://developers.openai.com/api/docs/guides/realtime-vad))
- Google's Gemini Live API takes 16-bit PCM audio, "natively 16kHz" in, and "Audio output always uses a sample rate of 24kHz." VAD is on by default, with start and end sensitivity, padding and silence-duration settings, and "When VAD detects an interruption, the ongoing generation is canceled and discarded." ([Google AI docs](https://aistudio.google.com/docs/live-api/capabilities))
- Moshi reports "200ms in practice" and GPT-4o "an average of 320 milliseconds" (not reproduced here). The cascade above took 1.4 to 2.4 s at the median: well under the "several seconds" the Moshi paper attributes to typical pipelines, but several times slower than either.
- SoundStream started the neural codec line with a listening test: "at 3 kbps outperforms Opus at 12 kbps and approaches EVS at 9.6 kbps," Opus and EVS being standard hand-designed codecs. Kokoro-82M, with "82 million parameters" and a StyleTTS 2 design per its [model card](https://huggingface.co/hexgrad/Kokoro-82M), peaked at 813 MB of GPU memory here, small enough to share an 8 GB laptop GPU with Whisper and a 1.5B-parameter LLM.

## Common misconceptions

- **"The language model is the only slow part of a voice agent."** It was the largest stage, but the 0.6 s spent making sure the user had stopped cost more than recognition and synthesis together.
- **"Silence means the user is done."** Turns have pauses between sentences and at commas; a 100 ms silence rule ended 16 of 40 turns early. That's why OpenAI's Realtime API also offers a semantic mode that judges from the words.
- **"Fewer bits means worse audio."** Mimi at 1.1 kbps matched DAC at 3 kbps on error rate and STOI.
- **"The semantic token is the speech."** Mimi's first codebook alone decoded to audio Whisper got 39% wrong. It helps a model predict what comes next; the acoustic codes turn it into a voice.
- **"Streaming only matters for long answers."** Replies here were two or three sentences, and streaming still saved 0.38 s with Claude and 0.60 s locally.

## Key takeaways

- A neural audio codec turns speech into a grid of integers. Mimi's 12.5 frames per second times 8 codebooks give 100 tokens per second (1.1 kbps), and Whisper reads the round trip nearly as well as the original.
- Codebooks are layered: the first carries what was said, the rest how it sounded, and without them the speech falls apart.
- A cascaded voice agent on a laptop answered in about 2 seconds: 0.6 s waiting for silence, 0.2 s of recognition, about 1 s for the LLM's first sentence, 0.14 s of synthesis.
- Stream everything. Speaking the first finished sentence saved 0.38 to 0.60 s per turn; a shorter silence rule saves more but interrupts people.
- Speech-to-speech models like Moshi report around 200 ms, near the 208 ms human gap. A cascade keeps text in the middle, easier to log, check and change.

## What's next

Part 12 turns to the hardest signal in the series: video. A few seconds of video is a stack of frames, and generating it means keeping a whole spacetime volume consistent, with spacetime patches, 3D autoencoders and diffusion transformers doing the work that codecs and turn-taking did here.

## Further reading

- [SoundStream: An End-to-End Neural Audio Codec (Zeghidour et al., 2021)](https://arxiv.org/abs/2107.03312): residual vector quantization and quantizer dropout for neural codecs.
- [High Fidelity Neural Audio Compression (Défossez et al., 2022)](https://arxiv.org/abs/2210.13438): EnCodec, 75 frames per second and 1.5 to 24 kbps from one model.
- [High-Fidelity Audio Compression with Improved RVQGAN (Kumar et al., 2023)](https://arxiv.org/abs/2306.06546): DAC, factorized and L2-normalized codes against codebook collapse.
- [AudioLM: a Language Modeling Approach to Audio Generation (Borsos et al., 2022)](https://arxiv.org/abs/2209.03143): the split into semantic and acoustic tokens.
- [Moshi: a speech-text foundation model for real-time dialogue (Défossez et al., 2024)](https://arxiv.org/abs/2410.00037): Mimi, split RVQ, two audio streams, Inner Monologue and the 160 ms latency budget.
- [Universals and cultural variation in turn-taking in conversation (Stivers et al., 2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2705608/): the 10-language measurement of gaps between turns.
- [GPT-4o System Card (OpenAI, 2024)](https://arxiv.org/abs/2410.21276): the 232 ms and 320 ms audio response figures.
- [Silero VAD (GitHub)](https://github.com/snakers4/silero-vad): the MIT-licensed voice activity detector used here.
- [Voice activity detection in the Realtime API (OpenAI docs)](https://developers.openai.com/api/docs/guides/realtime-vad): `server_vad`, `semantic_vad` and their settings.
- [Live API capabilities (Google AI docs)](https://aistudio.google.com/docs/live-api/capabilities): audio formats, automatic VAD and interruption in Gemini Live.
- [Mimi model card](https://huggingface.co/kyutai/mimi), [DAC 24 kHz model card](https://huggingface.co/descript/dac_24khz) and [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M): the weights run here.
- [Multimodal Models: How LLMs See and Hear (Fundamentals Part 15)](/blog/multimodal-llms/): the overview of audio encoders and codec tokens this part builds on.
- [Cost and Latency of Agents (Agentic Part 15)](/blog/agent-cost-latency/): latency measured on text agents with the same API.
