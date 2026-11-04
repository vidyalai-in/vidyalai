---
title: "Speech Recognition with Whisper: From Waveform to Words"
description: "How Whisper turns sound into a log-mel spectrogram and then into text, and what word error rate, noise and silence do to it, measured on a laptop GPU."
slug: "speech-recognition-whisper"
category: "multimodal"
tags: ["multimodal", "speech", "transformers", "evals", "hallucination"]
series: "multimodal-ai"
seriesOrder: 9
pubDate: 2026-10-20
updatedDate: 2026-11-05
---

## TL;DR

Whisper turns 30 seconds of sound into a picture of 3,000 time steps by 80 frequency bands, reads it with a Transformer encoder, and writes the words with a decoder that works like a small language model. This article follows audio from samples to spectrogram to text, then measures four Whisper models on a laptop GPU: word error rate and how text cleanup changes it, speed, noise, and how often Whisper writes words when nobody is speaking.

## Why it matters

Play Whisper under three seconds of hiss too faint to hear and it may answer "you". On 150 synthetic clips with no voice in them, from near-silent hiss to beeps and hums, whisper-base wrote text for every single one. Its favorite word was "you"; 17 times it wrote "I'm going to go to the next", followed by "one", "room" or "station". Turbo, the largest model here, filled 88 of the clips, mostly with "Thank you." and "So".

On real speech the same models do well. On 500 read sentences from LibriSpeech's clean test set, large-v3-turbo got 2.2% of words wrong while transcribing 50 times faster than real time on a laptop GPU, and the 39-million-parameter tiny model got 7.9% wrong at 355 times real time. Yet score the same transcripts without cleaning up capitals and punctuation first and every model is wrong on about 99% of words.

Where do those errors come from, and which ones matter?

## The intuition

Think of a player-piano roll. Holes along the paper say which keys sound at each moment, bass at one edge and treble at the other, and a pianist who has read thousands of rolls can hum the tune from one without hearing a note. A log-mel spectrogram is that roll for a voice: time runs along it, frequency bands across it, and each cell holds the energy of one band in one short slice of time.

Whisper is the pianist. Its encoder reads the roll, and its decoder writes down the words, one token at a time, the way a language model writes the next word.

The analogy breaks in two places. A piano roll records notes a composer chose; a spectrogram records raw energy, with no gaps between words and no labels on the sounds. And the pianist never invents a tune for a blank roll, while Whisper's decoder is trained to keep writing: hand it silence and it sometimes writes a sentence anyway.

## Step 1: from air pressure to a log-mel spectrogram

### Sound as samples

A microphone turns changes in air pressure into a voltage, and a converter measures that voltage at fixed intervals. Each measurement is a sample, one number; the number of samples per second is the sample rate. Whisper resamples everything to 16,000 Hz, so one second of speech is 16,000 numbers and 30 seconds is 480,000.

The sample rate sets a ceiling on pitch. The Nyquist frequency, half the sample rate, is the highest frequency the samples can represent; above it, a fast wave and a slower one produce identical samples. At 16 kHz the ceiling is 8,000 Hz, which is why every Whisper feature stops there.

### The short-time Fourier transform

A waveform hides its pitches. A Fourier transform finds them: it measures how strongly each frequency is present in a stretch of signal. Speech changes every few tens of milliseconds, so the short-time Fourier transform (STFT) does this on short overlapping windows and lines the results up in time.

Whisper's [audio code](https://github.com/openai/whisper/blob/main/whisper/audio.py) fixes every number. The window is `N_FFT = 400` samples, 25 ms, tapered at both ends with a Hann window so the cut doesn't add false frequencies. The step is `HOP_LENGTH = 160` samples, 10 ms, so there are 100 frames per second. A 400-sample window gives 201 frequency bins from 0 to 8,000 Hz, 40 Hz apart.

### The mel scale and the log

Hearing isn't linear in frequency: listeners tell low pitches apart in fine steps and high ones only in coarse steps. The mel scale spaces bands to match; the version Whisper uses, [librosa's](https://librosa.org/doc/0.11.0/generated/librosa.filters.mel.html) default, is linear below 1,000 Hz and logarithmic above. Whisper multiplies each frame's 201 bins by 80 overlapping triangular filters. In that filter bank, 26 of the 80 band centers sit below 1,000 Hz; a band near 300 Hz spans about 75 Hz, one near 6,000 Hz about 470 Hz.

Loudness isn't linear either, so Whisper takes the base-10 log of each band's energy, clips everything more than 8 units (80 dB) below the loudest cell, and rescales with `(log_spec + 4) / 4`. The paper ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)) describes the result as "an 80-channel log-magnitude Mel spectrogram representation" computed on "25-millisecond windows with a stride of 10 milliseconds". Large-v3 and large-v3-turbo use 128 mel bins instead, as the [large-v3 model card](https://huggingface.co/openai/whisper-large-v3) states.

![Three stacked panels sharing a time axis from 0 to 4.6 seconds, for one synthetic sentence spoken by a Kokoro stock voice. Top: the waveform, 73,200 samples, with bursts of energy for each word and a thin box showing one 25-millisecond analysis window drawn to scale. Middle: the short-time Fourier transform, 457 frames by 201 frequency bins on a linear axis from 0 to 8,000 Hz; voiced sounds show as stacks of horizontal harmonic stripes in the lower part, and hissing sounds such as s as tall columns reaching 8,000 Hz. Bottom: Whisper's 80-bin log-mel features of the same sentence on a mel axis labeled 0, 500, 1,000, 2,000, 4,000 and 8,000 Hz; the low frequencies, where the harmonics are, now take up most of the height.](/diagrams/v09-waveform-to-logmel.svg "One sentence, three views. The text is \"Whisper listens to thirty seconds at a time and writes down what it heard.\", synthesized with Kokoro-82M's stock voice af_heart at speed 1.0 and resampled to 16 kHz.")

The mel axis gives the band below 1,000 Hz, where vowels put their harmonics, about a third of the rows; on the linear axis it gets an eighth.

The 30-second window is fixed. The feature extractor pads every clip with silence to 480,000 samples, 3,000 frames of 80 numbers. [Part 1](/blog/multimodal-ai-overview/) measured what that means downstream: 100 log-mel frames and 8,000 numbers per second, then 1,500 encoder vectors for a one-second clip, as many as for thirty seconds.

## Step 2: inside Whisper

### 680,000 hours of imperfect labels

Whisper's main novelty is its dataset: 680,000 hours of audio paired with transcripts found on the internet. That is weak supervision. The labels exist, but nobody checked them, so some are wrong, partial or misaligned. Of those hours, 117,000 cover 96 languages other than English and 125,000 are translation pairs. The team filtered out transcripts that looked machine-made ("an all-uppercase or all-lowercase transcript is very unlikely to be human generated"), cut everything into 30-second segments, and kept some segments with no speech so the model would learn to say so.

### An encoder-decoder with a small convolutional stem

The model is an encoder-decoder Transformer, the original 2017 design that [Fundamentals Part 9](/blog/encoders-decoders-gpt/) compares with decoder-only GPT. The encoder reads the whole spectrogram at once. The decoder writes text and, in every layer, uses cross-attention to look back at the encoder's output.

Before the encoder sits "a small stem consisting of two convolution layers with a filter width of 3 and the GELU activation function where the second convolution layer has a stride of two". It turns 3,000 spectrogram frames into 1,500 positions of 20 ms each. Sinusoidal position vectors are added, then the Transformer blocks run.

![Flowchart, top to bottom. Audio at 16 kHz padded to 30 seconds becomes a log-mel spectrogram of 3,000 frames by 80 bins. A convolutional stem of two Conv1D layers with GELU, the second with stride 2, feeds the encoder, a stack of Transformer blocks over 1,500 positions. The encoder's output reaches the decoder through cross-attention. Prompt tokens (start of transcript, English, transcribe, no timestamps) also feed the decoder, which outputs the next text token; that token is appended and the loop repeats until the end-of-text token.](/diagrams/v09-whisper-pipeline.svg "Whisper at inference. The encoder runs once per 30-second window; the decoder runs once per output token.")

The encoder runs once per window and the decoder once per token:

| Model | Encoder layers | Decoder layers | Width | Mel bins | Parameters |
|---|---|---|---|---|---|
| tiny | 4 | 4 | 384 | 80 | 39M |
| base | 6 | 6 | 512 | 80 | 74M |
| small | 12 | 12 | 768 | 80 | 244M |
| medium | 24 | 24 | 1,024 | 80 | 769M |
| large-v2 | 32 | 32 | 1,280 | 80 | 1,550M |
| large-v3-turbo | 32 | 4 | 1,280 | 128 | 809M |

Sizes from the paper's Table 1 (large-v2 shares its Large row) and the Hugging Face configs. The turbo row comes from its [model card](https://huggingface.co/openai/whisper-large-v3-turbo): "a finetuned version of a pruned" large-v3, "the exact same model, except that the number of decoding layers have reduced from 32 to 4." Large-v3 itself, its card says, was trained on 1 million hours of weakly labeled audio and 4 million hours labeled by large-v2.

### One decoder, many tasks, chosen by tokens

Whisper picks its task from the first tokens it writes: `<|startoftranscript|>`, a language token (one of 99), `<|transcribe|>` or `<|translate|>` (write English, whatever was spoken), then `<|notimestamps|>` or timestamp tokens, 1,501 of them from `<|0.00|>` to `<|30.00|>` in 20 ms steps. For a segment with no speech, the model is trained to predict a no-speech token instead of a language.

Software can force any of these. Every run below fixes English, transcription and no timestamps, so the model only chooses the words. The round trip fits in a few lines (Transformers 5.21, on the sentence in the figure):

```python
import jiwer, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

proc = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", dtype=torch.float16).to("cuda")

audio, sr = sf.read("speech.wav", dtype="float32")      # mono, 16 kHz
x = proc(audio, sampling_rate=sr, return_tensors="pt").input_features
print(x.shape)                                         # 80 mel bins x 3,000 frames
ids = model.generate(x.to("cuda", torch.float16), language="en", task="transcribe")
text = proc.batch_decode(ids, skip_special_tokens=True)[0]
print(text)

ref = "Whisper listens to thirty seconds at a time and writes down what it heard."
basic = BasicTextNormalizer()                          # lowercase, drop punctuation
print(jiwer.wer(basic(ref), basic(text)))
english = proc.tokenizer.normalize                     # Whisper's English normalizer
print(jiwer.wer(english(ref), english(text)))
```

```text
torch.Size([1, 80, 3000])
 Whisper listens to 30 seconds at a time and writes down what it heard.
0.07142857142857142
0.0
```

The transcript is right, yet the basic score is 1 error in 14 words, because Whisper wrote "30" where the reference says "thirty". Step 3 is about that difference.

### The other way: CTC and wav2vec 2.0

The main alternative has no decoder at all. wav2vec 2.0 ([Baevski et al., 2020](https://arxiv.org/abs/2006.11477)) first learns from unlabeled audio by hiding stretches of its own internal representation and picking the right quantized version of each hidden frame out of distractors. A linear layer on top is then fine-tuned to output one character per frame, about 49 per second, with a Connectionist Temporal Classification (CTC) loss, which lets the model emit blanks and repeats that are collapsed afterwards, so nobody has to align letters to frames by hand.

With all 960 labeled LibriSpeech hours, the paper reports 1.8% and 3.3% word error on the clean and other test sets. Since each output is tied to a frame, a CTC model can't run on and invent a paragraph, but it learns the style of its one labeled corpus. In the Whisper paper, a wav2vec 2.0 Large model and Whisper large-v2 both score 2.7% on LibriSpeech test-clean, yet across 13 other test sets they average 29.3% against 12.8%, "55.2% less errors on average" for Whisper (reported, not reproduced here).

### Smaller decoders: Distil-Whisper and turbo

Shrinking the decoder buys the most speed. Distil-Whisper ([Gandhi et al., 2023](https://arxiv.org/abs/2311.00430)) freezes large-v2's encoder, keeps two decoder layers (the teacher's first and last), and trains them on 21,170 hours of audio transcribed by large-v2, keeping only clips where those transcripts agree with the human ones. The paper reports it "5.8 times faster with 51% fewer parameters, while performing to within 1% WER on out-of-distribution test data" (reported, not reproduced here). Large-v3-turbo applies the idea to large-v3 with four decoder layers.

## Step 3: scoring a transcript with word error rate

### Counting edits

Word error rate (WER) lines the transcript up against a reference and counts the fewest word edits that turn one into the other: substitutions (S, a wrong word), deletions (D, a missing word) and insertions (I, an extra word), divided by the number of reference words N.

```text
reference:  switch the kitchen lights off at    ten
transcript: switch the kitchen light  off at a  ten
                               S            I
WER = (S + D + I) / N = (1 + 0 + 1) / 7 = 28.6%
```

Insertions have no upper bound, so WER can pass 100%. Over a test set, errors and reference words are summed over all utterances before dividing, so long utterances count for more.

### The normalizer decides the score

LibriSpeech references look like `MISTER SMITH PAID TWENTY DOLLARS`: capitals, no punctuation, numbers spelled out. Whisper writes `Mr. Smith paid $20.` Compared as raw strings, nearly every word differs. So every published WER first passes both texts through a normalizer. Whisper ships two. The basic one lowercases and strips punctuation and bracketed text. The English one goes further: it spells out titles ("mr" becomes "mister"), standardizes British spellings ("colour" to "color", "grey" to "gray"), expands contractions ("didn't" to "did not"), drops fillers such as "hmm", and turns spelled-out numbers into digits.

Here is what each step does to the same transcripts (greedy decoding, 500 test-clean utterances; 95% bootstrap intervals over utterances):

| Text cleaning | tiny | base | small | large-v3-turbo |
|---|---|---|---|---|
| None (exact strings) | 99.8% | 99.2% | 99.1% | 99.4% |
| Lowercase only | 19.5% | 18.0% | 16.8% | 11.1% |
| Basic normalizer | 8.6% (7.7-9.6) | 6.0% (5.3-6.7) | 4.2% (3.6-4.8) | 2.7% (2.2-3.2) |
| English normalizer | 7.9% (7.0-8.7) | 5.3% (4.6-6.0) | 3.5% (2.9-4.1) | 2.2% (1.7-2.7) |

Lowercasing alone leaves every word with punctuation attached wrong; the English normalizer then removes another 9% (tiny) to 19% (turbo) of the remaining errors. The normalizer decides what counts as a mistake, so a WER quoted without its normalizer can't be compared with another.

Normalizers have quirks of their own. Whisper's English one turns "ten thirty" into `1030` but "10:30" into `10 30`, so the same time spoken and written still disagrees.

### Four models on LibriSpeech

LibriSpeech ([Panayotov et al., 2015](https://www.openslr.org/12/)) is about 1,000 hours of read public-domain audiobooks under CC BY 4.0; its test-clean and test-other sets hold 5.4 and 5.3 hours, test-other being the harder one. Here they serve only to measure. From each, 500 utterances were drawn with a fixed seed, spread evenly over speakers (all 40 in test-clean, all 33 in test-other), leaving out the 16 utterances longer than 30 seconds so every clip fits one window: 10,184 and 9,322 reference words, 1.07 and 0.96 hours of audio.

Each model ran in fp16 on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) with PyTorch 2.11 and Transformers 5.21: greedy decoding, English forced, 16 clips per batch, clips sorted by length.

![Dot chart of word error rate for four Whisper models, each with a test-clean and a test-other dot and 95% bootstrap bars. tiny, 39M parameters: 7.9% clean, 16.6% other, 355 times real time. base, 74M: 5.3% and 11.8%, 281 times real time. small, 244M: 3.5% and 7.7%, 125 times real time. large-v3-turbo, 809M: 2.2% and 3.7%, 50 times real time. Hollow dots show the paper's greedy results on the full test sets for tiny, base and small, each within or near this run's interval.](/diagrams/v09-wer-by-model.svg "Word error rate after Whisper's English normalizer. Filled dots: this run, 500 utterances per test set. Hollow dots: the paper's Table 8 (greedy decoding, full test sets).")

Tiny, base and small land within 0.6 points of the paper's greedy numbers on the full test sets (7.6% and 16.9%, 5.0% and 12.4%, 3.4% and 7.6%), so 500 utterances per set reproduce the published picture. Large-v3-turbo, with 21 times tiny's parameters, makes 3.6 times fewer errors on test-clean and 4.5 times fewer on test-other: the harder the audio, the more size pays.

Speed depends mostly on the decoder:

| Model | Whole run, 16 per batch | One 10-second clip | Encoder, one window | Peak GPU memory, 16 per batch |
|---|---|---|---|---|
| tiny | 355× real time | 0.150 s | 4.3 ms | 309 MiB |
| base | 281× | 0.205 s | 8.3 ms | 517 MiB |
| small | 125× | 0.422 s (0.33-0.71) | 23.4 ms | 1,465 MiB |
| large-v3-turbo | 50× | 0.281 s | 126.7 ms | 2,281 MiB |

Speed is seconds of audio per second of compute (the inverse of the real-time factor Part 10 uses), feature extraction included, on the 500 test-clean clips. Single-clip times are medians of 5 runs after 2 warm-ups (range in brackets where wide), with `torch.cuda.synchronize()` around each; nvidia-smi read 1,620 to 2,175 MHz and 17 to 50 W.

Turbo's encoder takes 5.4 times as long as small's, yet turbo transcribes a single clip faster, because its decoder has 4 layers to small's 12 and runs once for every one of the clip's roughly 30 tokens. In batches the order flips: 16 clips share each decoder step, the encoder's cost dominates, and small pulls ahead, 125 to 50 times real time.

## Step 4: decoding, noise and silence

### Greedy or beam search

Greedy decoding takes the most probable token at every step; beam search keeps the five most probable partial transcripts, extends each, and returns the best. On whisper-small it cut test-clean WER from 3.5% to 3.1% (paired bootstrap difference −0.33 points, 95% interval −0.75 to −0.06) and test-other from 7.7% to 6.8% (−0.86, −1.29 to −0.47). It cost four times the compute: 125 down to 31 times real time, 1.60 to 6.14 seconds for a fixed batch of 16 clips (median of 5 runs, 1,740 to 2,175 MHz, 39 to 47 W), and 1,465 to 5,394 MiB of peak GPU memory for five decoder states per clip.

### Noise

The Whisper paper added white noise and pub noise at a fixed signal-to-noise ratio (SNR), speech power over noise power in decibels: 20 dB means noise at a hundredth of the speech power, 0 dB equal power. Here the same test used 200 test-clean utterances with white noise or babble (six LibriSpeech talkers from outside the test subsets, summed), the same seeded noise for both models.

![Line chart of word error rate on a log scale against signal-to-noise ratio: clean, 20, 10, 5 and 0 dB. tiny with white noise: 7.9, 9.2, 15.6, 31.9 and 73.7%. tiny with babble: 7.9, 9.1, 15.3, 39.0 and 157.5%. large-v3-turbo with white noise: 2.0, 2.1, 3.0, 4.6 and 11.6%. large-v3-turbo with babble: 2.0, 2.1, 2.8, 6.6 and 32.1%. Both models change little down to 10 dB, then rise steeply; babble at 0 dB hurts most.](/diagrams/v09-noise.svg "WER on 200 test-clean utterances with added noise. Dashed: white noise. Solid: six-talker babble.")

At 10 dB tiny's error has doubled, to about 15%, while turbo's has gone from 2.0% to under 3%. At 0 dB babble, tiny passes 100%: of its 6,829 errors there, 3,866 are insertions, words from the other voices or a phrase looped until it runs out of tokens. Turbo's error rate at 0 dB babble, 32.1% (23.4-47.2), is a fifth of tiny's, and at 0 dB white noise it stays at 11.6%.

### Speech that isn't there

Whisper's decoder was trained to write transcripts; the paper says segments without speech were kept "with sub-sampled probability". For anything that listens all the time, the test that matters is audio with nobody in it: 50 near-silent ones (white, pink or brown noise at −80 to −60 dB below full scale, too faint to hear on most speakers), 50 with audible noise (−35 to −15 dB), 50 of synthetic beeps, hums and chirps, each 2 to 30 seconds long, plus 10 seconds of digital zeros. Each model ran four ways:

- greedy decoding, as in the WER runs;
- the paper's fallback: if the average token log-probability is below −1 or the output compresses too well (a sign of repetition), decode again at temperature 0.2, 0.4 and so on up to 1.0 (Transformers measures compression on tokens, so its docs suggest a threshold of 1.35 instead of the paper's 2.4);
- fallback plus the no-speech check: skip the window if the no-speech token's probability is above 0.6 and the log-probability test also fails;
- a gate: [Silero VAD](https://github.com/snakers4/silero-vad) 6.2.3 (MIT), a 2.3 MB voice activity detector, decides on the CPU whether the clip contains speech, and Whisper runs only if it does.

| Model | Greedy | Fallback | Fallback and no-speech check | Silero VAD gate |
|---|---|---|---|---|
| tiny | 144 (91-98%) | 147 | 76 (42-58%) | 1 |
| base | 151 (98-100%) | 150 | 54 (29-44%) | 2 |
| small | 115 (69-82%) | 134 | 64 (35-50%) | 2 |
| large-v3-turbo | 89 (51-66%) | 104 | 104 (61-76%) | 0 |

Clips, out of 151, for which the model wrote at least one word; 95% Wilson intervals in brackets. Most outputs are one word, but tiny, base and turbo each looped at least once to more than 400 words, such as turbo's "oh oh oh" repeated until it hit the token limit.

Fallback alone makes it worse: when an output fails the log-probability test, sampling at a higher temperature finds a different sentence rather than none. Adding the no-speech check cuts the count by 48 to 64% from fallback alone for the original models but changed none of turbo's outputs in this Transformers version; the skip needs both conditions at once, and for turbo they never held together on these clips. The digital silence produced text from every model in every setting but the gate. The VAD flagged 2 of the 151 clips as speech, both tone sequences, and passed all 1,000 LibriSpeech clips, so the gate cost no real speech here; it ran at 75 times real time on four CPU threads.

## In the real world

Turbo is now Whisper's default. OpenAI's [release note](https://github.com/openai/whisper/discussions/2363) says the `whisper` command uses it unless told otherwise, that it was fine-tuned on large-v3's transcription data without the translation data, "on which we don't expect turbo to perform well", and that across languages it "performs similarly to large-v2", with larger drops on some, such as Thai and Cantonese. On this laptop it transcribes an hour of audiobook in about 72 seconds; small does it in 29 seconds, tiny in 10.

Longer recordings go through the window one piece at a time. The paper's long-form procedure transcribes 30 seconds, moves the window to the last predicted timestamp, and feeds the previous window's text back in as context. On seven long-form test sets, from TED talks to earnings calls, greedy decoding averaged 11.0% WER; beam search, temperature fallback, the no-speech check, previous-text conditioning and a rule that the first timestamp falls in the first second brought it to 10.0% (Table 7, reported, not reproduced here).

Hallucination is a documented limitation. The [large-v3 model card](https://huggingface.co/openai/whisper-large-v3) warns that predictions "may include texts that are not actually spoken in the audio input" because the models "combine trying to predict the next word in audio with trying to transcribe the audio itself." On long audio, the Distil-Whisper paper counts 23,792 repeated five-word sequences in large-v2's output and 7,971 in a wav2vec 2.0 CTC model's (reported, not reproduced here). Systems that listen all the time, like the voice agents in Part 11, put a voice activity detector in front of the recognizer.

Meaning is the other gap between WER and use. A Claude Haiku 5.5 judge (Batch API, thinking disabled, one-word answer) read all 2,017 flawed transcripts from the four models and said whether each error changes what a reader would understand. Against 60 hand labels made before it ran, 15 per model, it agreed on 50 (83%, 95% interval 72-91%; Cohen's kappa 0.65). Turbo's "the great hulk followed hurriedly to retrieve his prey", for "hawk", counts as changed; its "ironic air" for "ironical air" doesn't.

| Model | Test-clean sentences with any error | With an error that changes the meaning |
|---|---|---|
| tiny | 290 of 500 | 217 (43% of all 500; 39-48%) |
| base | 240 | 176 (35%; 31-39%) |
| small | 181 | 107 (21%; 18-25%) |
| large-v3-turbo | 119 | 52 (10%; 8-13%) |

A bigger model makes fewer errors, and more of the ones it still makes are harmless: 75% of tiny's flawed sentences changed meaning, 44% of turbo's. The judge cost $0.026; 1,510 replies hit the 5-token cap after the label word, and 2 were unreadable and count as neither.

## Common misconceptions

- **"A word error rate is a property of the model."** It's a property of the model, the test set and the text normalizer together. The same turbo transcripts score 99.4%, 11.1%, 2.7% or 2.2% on test-clean depending only on how the text is cleaned before comparing.
- **"A bigger Whisper is always slower."** Turbo has 3.3 times small's parameters but transcribed a single 10-second clip in 0.28 seconds to small's 0.42, because its decoder, which runs once per token, has 4 layers to small's 12.
- **"Silence gives an empty transcript."** Every model here wrote text for 10 seconds of digital zeros, and turbo for 88 of 150 clips of noise and tones. A voice activity detector in front of the recognizer is the fix that worked: it let 2 of the 151 clips through.
- **"Beam search is free accuracy."** On small it removed about one error in ten and cost four times the compute and 3.7 times the GPU memory at the same batch size.

## Key takeaways

- Whisper hears through a fixed 30-second window at 16 kHz, cut into 25-millisecond frames every 10 milliseconds, each squeezed into 80 or 128 mel bands on a log scale.
- An encoder reads that spectrogram once; a decoder writes the transcript token by token, and special tokens at the start pick the language and the task.
- Word error rate counts substitutions, deletions and insertions over reference words, and the text normalizer can move it from 99% to 2%. Always ask which normalizer a number used.
- On LibriSpeech, error fell from 7.9% for tiny to 2.2% for large-v3-turbo on clean speech, and from 16.6% to 3.7% on harder speech.
- With no speech to follow, Whisper's decoder writes "you" or "Thank you." A separate voice activity detector stopped nearly all of it.

## What's next

Part 10 runs the pipeline backwards: text to speech. It takes Kokoro, the voice in this article's figure, apart into text normalization, phonemes, an acoustic model and a vocoder, measures its speed on the same laptop, and uses Whisper from this part to check how intelligible each stock voice is.

## Further reading

- [Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022)](https://arxiv.org/abs/2212.04356): the Whisper paper; data, architecture, multitask tokens, long-form heuristics and the WER tables quoted here.
- [openai/whisper on GitHub](https://github.com/openai/whisper): the reference code; `audio.py` holds the spectrogram constants and `transcribe.py` the decoding thresholds.
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477): self-supervised pretraining plus CTC fine-tuning, the main alternative design.
- [Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling (Gandhi et al., 2023)](https://arxiv.org/abs/2311.00430): a two-layer decoder distilled from large-v2.
- [Whisper large-v3-turbo model card](https://huggingface.co/openai/whisper-large-v3-turbo): the pruned four-layer decoder.
- [Whisper large-v3 model card](https://huggingface.co/openai/whisper-large-v3): the 128-bin spectrogram, training data and known limitations.
- [LibriSpeech ASR corpus (Panayotov et al., 2015)](https://www.openslr.org/12/): the audiobook test sets used for measurement here.
- [librosa.filters.mel](https://librosa.org/doc/0.11.0/generated/librosa.filters.mel.html): the mel filter bank Whisper's features are built from.
- [Silero VAD](https://github.com/snakers4/silero-vad): the small voice activity detector used as a gate in the silence test.
- [Multimodal Models: How LLMs See and Hear (Fundamentals Part 15)](/blog/multimodal-llms/): where Whisper's encoder fits among multimodal designs.
- [How AI Sees, Hears and Draws (Part 1)](/blog/multimodal-ai-overview/): Whisper's input and encoder shapes measured, next to images and codec tokens.
