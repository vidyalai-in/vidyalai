---
title: "Text-to-Speech: How a Model Learns to Read Aloud"
description: "Text normalization, phonemes, duration, pitch and a waveform: how modern text-to-speech works, with Kokoro-82M timed and its intelligibility measured."
slug: "text-to-speech"
category: "multimodal"
tags: ["multimodal", "speech", "tts", "latency", "evals"]
series: "multimodal-ai"
seriesOrder: 10
pubDate: 2026-10-22
updatedDate: 2026-11-04
---

## TL;DR

A text-to-speech system turns text into audio in four stages: it rewrites numbers and abbreviations as words, converts the words into phonemes (the sounds of the language), predicts how long and at what pitch each sound is spoken, and renders a waveform. This article walks through each stage with Kokoro-82M, an open 82-million-parameter model with stock voices, on a laptop. It times the model on GPU and CPU, measures how much sooner the first audio arrives when you synthesize one sentence at a time, and checks intelligibility by having Whisper transcribe 2,060 synthesized clips. The weakest stage is the first: "Gen." became "Jen", and "St." in a street address became "Saint".

## Why it matters

On a laptop GPU, Kokoro-82M reads a 320-word passage, 102 seconds of speech, in 1.98 seconds: 51 times faster than real time. Split into sentences, the first audio of a five-sentence paragraph is ready after 72 milliseconds, 4.5 times sooner than in one piece. Whisper transcribes its speech with a word error rate of 0.55% on 200 public-domain sentences, against 2.6% for real people reading public-domain audiobooks.

And yet the same model reads "Gen. Ortiz met Lt. Brown" as "Jen Ortiz met L T Brown". The neural network that makes the sound is now good and cheap. Most of the remaining errors happen before it runs, when text is turned into something speakable.

[Part 1](/blog/multimodal-ai-overview/) showed how sound becomes numbers a model can read, [Fundamentals Part 15](/blog/multimodal-llms/) sketched speech in and out of language models, and Part 9 measured Whisper turning speech into text. This part runs the other way.

## The intuition

Think of an actor handed a script cold. First they decide how to say the awkward bits: is "St." Saint or Street, is "1998" a year or a quantity, is "read" past or present? Then they plan the delivery: which words to stretch, where to pause, where the voice rises. Only then do their vocal cords and mouth turn the plan into sound.

A text-to-speech (TTS) system has the same three jobs. A front end turns text into phonemes. An acoustic model plans the timing, pitch and loudness of each phoneme and describes the sound frame by frame. A vocoder, or a codec decoder in newer systems, turns that description into the waveform, the 24,000 numbers per second a speaker plays.

The analogy breaks in two places. An actor understands the script; the front end mostly looks words up and guesses the rest. And an actor can read a line a hundred ways, while most TTS models produce one reading, or a narrow spread, per voice.

## Step 1: from text to phonemes

![Flowchart, top to bottom. Text, Dr. Lee paid $45, goes to text normalization, which produces Doctor Lee paid forty-five dollars, then to grapheme-to-phoneme conversion, which produces a phoneme string. The phonemes go to phoneme encoders, PL-BERT and a CNN-LSTM. The encoders and a stock voice, a 256-number style vector, feed a prosody predictor for duration, pitch and energy. The prosody predictor, the encoders and the voice all feed the iSTFTNet decoder, which turns spectrum frames into a waveform: audio at 24 kHz.](/diagrams/v10-tts-pipeline.svg "Kokoro-82M's pipeline. The first two boxes run on the CPU in the misaki library; the rest is the neural network.")

The front end does two jobs. Text normalization rewrites everything that isn't a plain word: numbers, currency, dates, times, units and abbreviations. Grapheme-to-phoneme conversion, G2P for short, then maps the spelling (graphemes are letters) to phonemes, the distinct sounds a language uses. English needs this step more than most languages: "though", "through" and "tough" share the spelling "ough" and no vowel sound.

Kokoro hands both jobs to misaki, a small library by the same author. Its English G2P uses 49 phonemes in a modified IPA (International Phonetic Alphabet) where capital letters stand for diphthongs: `A` as in "day", `I` as in "high", `O` as in "go". Most words come from two dictionaries shipped with the library: 90,201 "gold" and 93,361 "silver" entries for American English. Numbers go through num2words, and words in neither dictionary fall back to espeak-ng, a rule-based speech synthesizer. You can run the front end without the neural network at all:

```python
from kokoro import KPipeline

g2p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)  # text to phonemes only
for text in ["Dr. Lee paid $45.", "Please record the record.", "Gen. Ortiz met Lt. Brown."]:
    for result in g2p(text):
        print(result.phonemes)
# dˈɑktəɹ lˈi pˈAd fˈɔɹTi fˈIv dˈɑləɹz.
# plˈiz ɹəkˈɔɹd ðə ɹˈɛkəɹd.
# ʤˈɛn ɔɹtˈiz mˈɛt ˌɛltˈi. bɹˈWn.
```

The marks `ˈ` and `ˌ` flag stressed syllables. "Dr." and "$45" came out right, and "record" came out two ways, as it should: the verb stresses the second syllable (`ɹəkˈɔɹd`), the noun the first (`ɹˈɛkəɹd`). Words spelled alike but pronounced differently are heteronyms, and the library resolves them with a part-of-speech tagger, the spaCy model `en_core_web_sm`, which labels each word noun, verb or adjective before the lookup; 790 gold entries carry pronunciations keyed by tag. "Gen." and "Lt." failed: the first became the name "Jen", the second the letters "L T".

### Sixty hard sentences

To see where the front end breaks, 60 sentences were written for this article, 10 in each of six categories, and misaki's phonemes (American English) were labeled by hand: does the phoneme string say what a careful human reader would?

| Category | Example | Read correctly | Acceptable | Wrong |
|---|---|---|---|---|
| Numbers | "Mix 3/4 of the paint with 1/2 a cup of water." | 9 | 0 | 1 |
| Currency | "He found ¥500 in the pocket of his coat." | 9 | 0 | 1 |
| Dates and times | "Your appointment is on Tue., Nov. 18 at 2:15 pm." | 5 | 2 | 3 |
| Abbreviations | "Dr. Lee lives at 42 Oak St. near the old mill." | 4 | 2 | 4 |
| Heteronyms | "The bass swam past the bass player on the dock." | 9 | 0 | 1 |
| Rare words and names | "The sommelier recommended a dry Gewürztraminer." | 8 | 1 | 1 |
| All | | 44 | 5 | 11 |

Each example is a failure. The fractions became "three four" and "one two", "¥500" became "yen five hundred", "Tue., Nov." stayed "tu., nov.", "Oak St." became "Oak Saint" (while "St. Paul's" was right), the second "bass" became the fish, and "Gewürztraminer" got a letter-by-letter guess. "Acceptable" means a person might say it that way, like "March three" or "prof". Eleven wrong out of 60 is 18%, with a 95% confidence interval (the range that would cover the true rate in 95% of repeated samples) of 11% to 30%. Sentences written to be hard aren't a sample of real text, so treat this as a list of failure modes, not a rate.

Heteronyms went well: 19 of 20 target words were right, from past and present "read" to the two "wind", "dove" and "desert". The one miss comes from how the lookup works: misaki's entry for "bass" switches to the music sense only when the tagger calls the word an adjective, and in "bass player" it's tagged as a noun. The tagger slipped once outside this set too: "read the first 15 lines" came out in the past tense.

## Step 2: from phonemes to a voice

An acoustic model faces a problem with no single right answer: one sentence can be spoken fast or slow, rising or falling, with a pause after any word. FastSpeech 2 ([Ren et al., 2020](https://arxiv.org/abs/2006.04558)) calls TTS "a typical one-to-many mapping problem": many readings fit one text, varying in pitch, duration, loudness and prosody, the rhythm, stress and melody of speech. A model trained to minimize average error learns the average of many readings, which sounds flat and muffled.

### A short history of the answers

The main designs of the last decade differ mostly in how they handle that spread:

| Model | Year | Acoustic model | Waveform from | How it handles one-to-many |
|---|---|---|---|---|
| Tacotron 2 | 2017 | Autoregressive: one mel frame at a time, with attention | WaveNet vocoder | Conditions each frame on the previous ones |
| FastSpeech 2 | 2020 | Parallel transformer | Separate vocoder | Predicts duration, pitch and energy explicitly |
| VITS | 2021 | Parallel, end to end | Its own GAN decoder | Samples durations from a learned distribution |
| StyleTTS 2 | 2023 | Parallel, style vector from diffusion | iSTFTNet or HiFi-GAN decoder | Samples a style vector per sentence |
| VALL-E | 2023 | Autoregressive language model over codec tokens | EnCodec decoder | Samples tokens like an LLM samples words |
| Voicebox, F5-TTS | 2023-2024 | Flow matching over mel frames | Vocoder | Starts from noise, like an image diffusion model |

Tacotron 2 ([Shen et al., 2017](https://arxiv.org/abs/1712.05884)) predicted a mel spectrogram, the picture of sound from Part 9 with energy in 80 frequency bands, one frame at a time with a 12.5 ms hop, and fed it to a WaveNet vocoder. It scored a mean opinion score of 4.53 against 4.58 for professional recordings. A mean opinion score (MOS) is the average of listeners' ratings from 1 (bad) to 5 (excellent), the field's headline number; it needs a panel of listeners for every comparison. Frame-by-frame generation was slow and could skip or repeat words when attention lost its place.

FastSpeech 2 generates all frames in parallel. It predicts each phoneme's duration, pitch and energy as explicit numbers, learned from values extracted from real recordings, so the frames no longer have to guess them. On a V100 its real-time factor was 0.0195. The real-time factor (RTF) is compute time divided by audio duration: 0.0195 means a second of speech takes 19.5 ms, 51 times faster than real time. VITS ([Kim et al., 2021](https://arxiv.org/abs/2106.06103)) merged acoustic model and vocoder into one network and sampled durations, so a sentence can come out with different rhythms; on LJ Speech, a single-speaker audiobook dataset, its MOS of 4.43 sat close to the recordings' 4.46.

StyleTTS 2 ([Li et al., 2023](https://arxiv.org/abs/2306.07691)) is the design Kokoro builds on. A diffusion model samples a style vector for each sentence, and the rest of the network is parallel and fast: the paper reports an RTF of 0.0185 on an RTX 2080 Ti with 5 diffusion steps, and says its speech "surpasses human recordings" on LJ Speech, with a comparative MOS (listeners compare two clips side by side) of +0.28. In the last rows, VALL-E ([Wang et al., 2023](https://arxiv.org/abs/2301.02111)) writes EnCodec tokens (the subject of Part 11) with transformers trained on 60,000 hours of English audiobooks, and the flow-matching models bring Part 7's technique to spectrograms.

### Inside Kokoro-82M

Kokoro-82M ([model card](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0) lists its architecture as StyleTTS 2 with an iSTFTNet decoder, and adds: "Decoder only: no diffusion, no encoder release". The style diffusion and the style encoder, which would build a voice from a recording, aren't published. A voice is a file: a table of 510 style vectors of 256 numbers, one per input length in phonemes. The card says its 54 voices in 8 languages came from "a few hundred hours" of permissively licensed and synthetic audio and about 1,000 A100 GPU hours of training.

Counting the parameters of the loaded model gives 81.8 million:

| Part | What it does | Parameters |
|---|---|---|
| PL-BERT (ALBERT, 12 layers sharing one set of weights) | Reads the phonemes in context for prosody | 6.3M |
| Prosody predictor (LSTMs and convolutions) | Duration of each phoneme, pitch and energy per frame | 16.2M |
| Text encoder (embedding, convolutions, LSTM) | Phoneme features for the decoder | 5.6M |
| Decoder (iSTFTNet) | Frames to waveform, driven by the pitch | 53.3M |
| Projection between BERT and predictor | Maps BERT's features to the predictor's width | 0.4M |

Two thirds of the model makes the waveform. PL-BERT ([Li et al., 2023](https://arxiv.org/abs/2301.08810)) is a BERT trained on phonemes; its ALBERT form reuses one layer's weights 12 times, hence so few parameters. The voice's 256 numbers split in two: 128 condition the decoder (what the voice sounds like), 128 the prosody predictor (how it speaks).

The network runs once per chunk of up to 510 phonemes, with no loop over time. The predictor gives every phoneme a duration in 25 ms frames, the phonemes are stretched to that length, and the predictor draws pitch and energy curves over the result. The decoder turns curves and phoneme features into a short-time spectrum, magnitude and phase, and an inverse short-time Fourier transform (iSTFT) turns that spectrum back into samples. The predicted pitch drives a sine-wave source inside the decoder, so the melody is fixed before any sound exists:

```python
from kokoro import KPipeline

pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
for r in pipe("Please record the record for the archive.", voice="af_heart"):
    print(len(r.phonemes), "phonemes ->", len(r.pred_dur), "durations, first six:", r.pred_dur[:6].tolist())
    print(len(r.audio), "samples =", len(r.audio) / 24000, "seconds")
# 40 phonemes -> 42 durations, first six: [14, 3, 2, 4, 5, 2]
# 69000 samples = 2.875 seconds
```

The two extra durations belong to start and end markers; the first, 14 frames, is 350 ms of lead-in before "p". Here is that sentence, and the plan the model made for it:

<audio controls preload="none" src="/audio/v10-record.mp3"></audio>

*Kokoro-82M stock voice `af_heart`, speed 1.0, American English: "Please record the record for the archive." 2.9 seconds, generated for this article.*

![Two aligned panels over 2.88 seconds. Top: the pitch contour Kokoro's prosody predictor chose, starting near 130 Hz, rising to about 290 Hz on "Please", dropping to between 140 and 220 Hz through the middle of the sentence, rising again to about 255 Hz at the start of "archive" and falling to 100 Hz at the end. Bottom: the log-mel spectrogram of the audio, with dashed lines at word boundaries taken from the predicted durations. Above it the words Please, record, the, record, for, the, archive; below it the phonemes misaki chose, with the two readings of record highlighted: rə-KORD for the verb and REK-ərd for the noun.](/diagrams/v10-record-spectrogram.svg "One Kokoro sentence as the model planned it: pitch from the prosody predictor, word spans from the predicted durations, and the spectrogram of the audio it rendered.")

The verb and the noun "record" got the same 350 ms with the stress in different places. "The" got 50 ms the first time and 100 ms (`ði`, before a vowel) the second, and the final "v" plus the period got 450 ms, mostly near silence.

## Step 3: measuring speed

Speed has two numbers: throughput, the real-time factor, and latency, the time to first audio, which is what a listener in a conversation notices. Kokoro-82M was timed in float32, as released, on the laptop's NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) and on its CPU (PyTorch 2.11, 8 threads), reading the first 5 to 320 words of the public-domain sentences used below in the `af_heart` voice. Each number is the median of 7 GPU or 5 CPU runs after 2 warm-ups, with `torch.cuda.synchronize()` around each, text-to-phoneme step included. `nvidia-smi` read a median SM clock of 2,040 MHz and 39 W board power (56 W peak).

![Line chart on log axes. Horizontal: length of the speech produced, 2 to 100 seconds. Vertical: time to synthesize it, 0.03 to 200 seconds. A dashed diagonal marks real time. The CPU line, 8 threads, runs from 0.55 seconds for 2.1 seconds of speech to 16.2 seconds for 51.8 seconds, three to four times faster than real time. The RTX 4060 Laptop GPU line runs from 79 milliseconds for 2.1 seconds of speech to 1.98 seconds for 101.9 seconds, about 51 times faster than real time.](/diagrams/v10-synthesis-time.svg "Kokoro-82M synthesis time against audio length. Points are medians of 7 (GPU) or 5 (CPU) runs after warm-up; bars, the min-max, are mostly smaller than the markers.")

From 20 words up, time grows in step with the audio: the GPU's real-time factor stays between 0.018 and 0.0196, about 19 ms per second of speech. Below that a fixed cost shows: 5 words (2.1 s of speech) took 79 ms, 10 words 84 ms. The CPU managed 0.23 to 0.31, three to four times faster than real time, enough to read aloud on a machine with no GPU. Text to phonemes took 1 to 24 ms, about 1% of the GPU time.

Memory is small: 312 MiB of weights, and peak allocated memory from 598 MiB for 5 words to 1,050 MiB for an 80-word chunk of 453 phonemes, near the 510 limit; longer passages are split into chunks. `nvidia-smi` showed at most 1,853 MiB in use, CUDA context included.

### Time to first audio

A voice assistant only needs the first sentence to start talking. The test paragraph has five sentences, 59 words and 281 phonemes, written for this article ("The kettle clicked off just as the rain began. ..."). Kokoro's pipeline can take it as one chunk, or split it at sentence boundaries and yield each sentence's audio when it's ready:

| Mode | Device | First audio after | All audio after | Audio length |
|---|---|---|---|---|
| Whole paragraph, one forward pass | GPU | 325 ms | 325 ms | 17.45 s |
| One sentence at a time | GPU | 72 ms | 403 ms | 18.00 s |
| Whole paragraph, one forward pass | CPU | 4.98 s | 4.98 s | 17.45 s |
| One sentence at a time | CPU | 0.59 s | 3.57 s | 18.00 s |

On the GPU, splitting cut the wait 4.5 times for 24% more total compute, since five small forward passes use the GPU less well than one large one. The audio got 0.55 s longer, because each sentence gets its own lead-in and closing pause. On the CPU, splitting helped twice: first audio after 0.59 s instead of 4.98 s, and the whole paragraph done sooner, because there the cost per second of audio grows with chunk length. (The CPU rows come from a rerun; the first attempt overlapped with other heavy jobs on the machine.) Part 11 builds a voice agent's latency budget around this number.

## Step 4: measuring intelligibility

MOS needs a panel of listeners. The round trip scales to thousands of clips: synthesize a sentence, transcribe it with a speech recognizer, and count the word error rate (WER), the substitutions, deletions and insertions that turn the transcript back into the original, divided by the number of original words. VALL-E and Voicebox report intelligibility this way.

The setup:

- Text: 200 random sentences of 8 to 30 words (seed 0), 100 each from two public-domain Project Gutenberg books, H. G. Wells's *The Time Machine* and L. Frank Baum's *The Wonderful Wizard of Oz*: 3,437 words. Plus the 60 hard sentences.
- Voices: six stock voices, four American and two British, spanning the grades Kokoro's voice list gives for "the quality and quantity" of each voice's training data, from A (`af_heart`) to F+ (`am_adam`).
- Recognizer: Whisper large-v3-turbo (`openai/whisper-large-v3-turbo`, 809M parameters, MIT license), fp16, greedy decoding, English. It transcribed all 2,060 clips, 3.3 hours of audio, in 323 seconds.
- Scoring: WER after Whisper's English text normalizer on both sides, so "$45" and "forty-five dollars" match; 95% confidence intervals from 2,000 bootstrap resamples of sentences.
- Reference: the same recognizer on 200 random utterances of LibriSpeech test-clean (40 real speakers reading public-domain audiobooks, CC BY 4.0), used for measurement only.

![Two dot charts with 95% confidence bars. Top, 200 public-domain sentences per voice: af_heart (grade A) 0.55%, af_bella (A-) 0.52%, bf_emma (B-) 0.61%, am_michael (C+) 0.66%, bm_george (C) 0.52%, am_adam (F+) 0.46%, all with intervals between about 0.2% and 1.0%; real speakers on LibriSpeech, for reference, 2.6% with an interval from 1.6% to 4.0%. Bottom, the 60 hard sentences with all six voices pooled: heteronyms 1.15%, numbers 1.46%, rare words and names 3.3%, currency 3.5%, abbreviations 7.1%, dates and times 17.0%.](/diagrams/v10-roundtrip-wer.svg "Round-trip word error rate: Kokoro speaks, Whisper large-v3-turbo transcribes. Bars are bootstrap 95% confidence intervals over sentences.")

On ordinary prose, Whisper misheard Kokoro less than it misheard people: 0.55% for the six voices pooled (115 errors in 20,802 words, 95% CI 0.43% to 0.68%) against 2.6% (1.6% to 4.0%) on real recordings. The synthetic clips are clean and evenly paced, without the hesitations, accents and room noise of real recordings, and the texts differ, so the gap says only that intelligibility is not Kokoro's weak point, not that it sounds more human. In all six voices, 167 of the 200 sentences came back perfect. Most remaining errors were spelling ("tender-hearted" as "tenderhearted"); a few were real mishearings, like "the Tin Woodman" as "the ten woodmen", twice in `am_michael`.

The voice grades made no measurable difference: all six intervals overlap, and the lowest-graded voice, `am_adam` at F+, had the lowest point estimate, 0.46%. The grades describe training data; listeners may hear the difference in naturalness, but a recognizer doesn't. The voices did differ in pace. The same 200 sentences took 1,104 seconds in `bf_emma` and 1,262 in `am_michael`, 187 and 163 words per minute, and for the median sentence the slowest voice took 17% longer than the fastest: each style vector picks its own durations for the same phonemes.

On the hard sentences the error rate rose to 17.0% for dates and times and 7.1% for abbreviations, and most of those errors trace back to the front end. "Oak St." came back as "Oaksent", "Tue." as "2.", "Gen. Ortiz met Lt. Brown" as "Jen Ortiz Med LT, Brown". Hear it for yourself:

<audio controls preload="none" src="/audio/v10-abbrev.mp3"></audio>

*Kokoro-82M stock voice `am_michael`, speed 1.0, American English: "Gen. Ortiz met Lt. Brown at 42 Oak St. on Tue., Nov. 18." 6.4 seconds, generated for this article. misaki's phonemes read it as "Jen Ortiz met L T Brown at forty-two Oak Saint on tu, nov eighteen".*

### What the round trip can't hear

Checked against the hand labels on the four American voices (240 clips), the test leaks. Of the 44 clips of sentences labeled wrong, the round trip flagged 34, 77% (95% CI 63% to 87%). It missed all four readings of "5 kg and 30 cm" as letters, because Whisper writes "kg" either way, and all four fish "bass", because a recognizer with a language model writes the word the sentence needs. It also flagged 15 of 176 clips labeled correct, 9% (5% to 14%), mostly over formatting, such as "9:30 am" coming back as "9, 30 a.m.". Use it to find gross failures at scale, then listen to what it flags and to a sample of what it passes.

### The speed setting

Kokoro's `speed` argument divides every predicted duration before rounding to whole 25 ms frames. On the first 50 of the 200 sentences, in two voices:

| Speed | Duration vs speed 1.0, `af_heart` | `am_michael` | Expected | Words per minute, `af_heart` | WER, `af_heart` | WER, `am_michael` |
|---|---|---|---|---|---|---|
| 0.5 | 2.01 | 2.01 | 2.00 | 87 | 1.75% (0.91-2.71) | 1.16% (0.37-2.02) |
| 0.75 | 1.37 | 1.36 | 1.33 | 128 | 1.40% (0.62-2.35) | 0.81% (0.23-1.50) |
| 1.0 | 1.00 | 1.00 | 1.00 | 175 | 1.16% (0.45-2.09) | 1.16% (0.45-2.09) |
| 1.25 | 0.85 | 0.84 | 0.80 | 206 | 1.05% (0.44-1.83) | 0.93% (0.24-1.78) |
| 1.5 | 0.65 | 0.65 | 0.67 | 268 | 5.01% (3.31-6.83) | 3.38% (2.09-4.80) |
| 2.0 | 0.55 | 0.53 | 0.50 | 320 | 14.1% (11.1-17.6) | 15.1% (12.6-17.8) |

Durations are medians over the 50 sentences; WER intervals are bootstrap 95% CIs over their 859 words. Intelligibility holds from half speed to 1.25, then breaks: at 1.5 the error rate roughly triples, and at 2.0 (320 words per minute) one word in seven is wrong. Durations don't scale exactly, because each is rounded to whole frames, at least one per phoneme. A sound predicted at 2.3 frames rounds to 2 at speed 1.0 and, at 1.84, still to 2 at speed 1.25, which is how 1.25 came out only 15% shorter instead of 20%.

## In the real world

- Kokoro's model card puts the market price of serving it over an API at "under $1 per million characters of text input", and says "1000 characters of input is about 1 minute of output". The six voices here spoke 853 to 975 characters per minute, close to that rule.
- F5-TTS ([Chen et al., 2024](https://arxiv.org/abs/2410.06885)), a flow-matching model trained on "a public 100K hours multilingual dataset", reports an RTF of 0.15 on an RTX 3090 with 16 flow steps, eight times Kokoro's figure here, in exchange for zero-shot voices.
- VALL-E's paper measured intelligibility by round trip with a HuBERT recognizer: 5.9% WER for VALL-E, 2.2% for the original recordings. Voicebox ([Le et al., 2023](https://arxiv.org/abs/2306.15687)), trained on 60,000 hours of English audiobooks, reported 1.9% against VALL-E's 5.9%, "while being up to 20 times faster".
- Cheap, natural speech makes voice cloning easier. VALL-E's paper warns that a model working from a 3-second recording "may carry potential risks in misuse of the model, such as spoofing voice identification or impersonating a specific speaker". Besides consent and policy, there are two defenses: not releasing the cloning parts, as Kokoro does, and watermarking. AudioSeal ([San Roman et al., 2024](https://arxiv.org/abs/2401.17264)) hides an inaudible signal in generated speech that its detector locates "up to the sample level", 1/16,000 of a second, so it can mark which part of an edited clip is synthetic. Nothing here clones anyone: every voice is a Kokoro stock voice.

## Common misconceptions

- **"The neural network decides how to pronounce words."** Kokoro's network never sees a word, only phonemes from a dictionary, a tagger and a fallback synthesizer. "Gen." becoming "Jen" is a front-end error no model quality can fix.
- **"A faster GPU always means faster replies."** Five words took 79 ms and ten took 84 ms: short inputs are dominated by fixed costs. Splitting the text into sentences cut the wait for the first sound from 325 ms to 72 ms on the same chip.
- **"If speech recognition transcribes it correctly, the TTS read it correctly."** The round trip flagged 34 of 44 clips of wrongly read sentences and missed every "kg" spelled out as letters and every wrong "bass": a recognizer writes the word the sentence needs.
- **"Same text, same voice, same audio."** Durations and pitch repeat exactly, but the decoder starts each harmonic at a random phase, so two runs differed by up to 0.10 in sample value. Compare durations or transcripts in tests, not audio bytes.
- **"Every open TTS model can clone voices."** Some can, from seconds of audio, as the VALL-E paper warns. Kokoro-82M ships without the encoder that would build a voice from a recording; its voices are fixed vectors in files.

## Key takeaways

- Text-to-speech is a pipeline: normalize the text, turn it into phonemes, predict duration and pitch, render a waveform. Kokoro does the first two on the CPU and the rest in one 82-million-parameter network.
- The front end is where the errors are. Eleven of 60 hard sentences were misread before any audio existed, mostly abbreviations, dates and times.
- Kokoro is fast on modest hardware: 51 times faster than real time on a laptop GPU, three to four times on its CPU, in about 1 GB of GPU memory.
- For a listener, latency beats throughput. Sentence-by-sentence synthesis brought the first audio from 325 ms to 72 ms.
- Intelligibility is not the weak point: Whisper misheard 0.55% of Kokoro's words, against 2.6% for real speakers. But the round trip missed a quarter of the misreadings a listener would hear.

## What's next

Kokoro makes its waveform with a decoder that never sees a discrete token. Part 11 goes the other way: neural audio codecs like EnCodec and Mimi that turn speech into a few hundred integers per second, the speech-to-speech models that read and write those integers directly, and a measured latency budget for a voice agent built from Whisper, Claude and Kokoro on this laptop.

## Further reading

- [Kokoro-82M model card (hexgrad, Hugging Face)](https://huggingface.co/hexgrad/Kokoro-82M) and its [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md): architecture, training data and cost, license, and the stock voices with their grades.
- [misaki (hexgrad, GitHub)](https://github.com/hexgrad/misaki): the G2P library behind Kokoro, with its phoneme set in EN_PHONES.md.
- [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models (Li et al., 2023)](https://arxiv.org/abs/2306.07691): the architecture Kokoro uses, with its RTF and listening tests.
- [iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform (Kaneko et al., 2022)](https://arxiv.org/abs/2203.02395): the decoder design that predicts a spectrum and inverts it.
- [Phoneme-Level BERT for Enhanced Prosody of Text-to-Speech with Grapheme Predictions (Li et al., 2023)](https://arxiv.org/abs/2301.08810): the phoneme BERT inside Kokoro.
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (Shen et al., 2017)](https://arxiv.org/abs/1712.05884): Tacotron 2 and its MOS of 4.53 against 4.58 for recordings.
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)](https://arxiv.org/abs/2006.04558): the one-to-many problem and explicit duration, pitch and energy.
- [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (Kim et al., 2021)](https://arxiv.org/abs/2106.06103): VITS and its stochastic duration predictor.
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (Wang et al., 2023)](https://arxiv.org/abs/2301.02111): VALL-E, TTS as next-token prediction over EnCodec codes, and its warning about misuse.
- [Voicebox (Le et al., 2023)](https://arxiv.org/abs/2306.15687) and [F5-TTS (Chen et al., 2024)](https://arxiv.org/abs/2410.06885): flow matching for speech.
- [Proactive Detection of Voice Cloning with Localized Watermarking (San Roman et al., 2024)](https://arxiv.org/abs/2401.17264): AudioSeal, sample-level watermark detection.
- [WaveNet: A Generative Model for Raw Audio (van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499) and [HiFi-GAN (Kong et al., 2020)](https://arxiv.org/abs/2010.05646): the autoregressive and adversarial vocoders that came before.
- [LibriSpeech ASR corpus (OpenSLR 12)](https://www.openslr.org/12): the real-speech reference used here for measurement only, CC BY 4.0.
- [Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022)](https://arxiv.org/abs/2212.04356) and the [whisper-large-v3-turbo model card](https://huggingface.co/openai/whisper-large-v3-turbo): the recognizer used for the round trips.
- [How AI Sees, Hears and Draws: Everything Becomes Tokens (Part 1)](/blog/multimodal-ai-overview/) and [Multimodal Models: How LLMs See and Hear (Fundamentals Part 15)](/blog/multimodal-llms/): how sound becomes vectors and tokens.
