# Writing narration

## Voice and pacing

- Spoken, not written: short sentences, one idea each, active voice. Read every `say` aloud in
  your head; if you'd stumble, rewrite it.
- About 145 words per minute with Kokoro `af_heart` at speed 1.0. Budget the main video at
  1,300-1,700 words and each Short at 80-120 words.
- Open with a concrete hook (a number, a scenario), not "In this video". State the core idea in
  the first 45 seconds.
- Signpost: "First…", "Second…", "Now the online side." Viewers can't scroll back.
- Each step's narration should match what appears on screen during it. Introduce an item in the
  same step that reveals it.
- No filler ("let's dive in", "without further ado"), no hype ("revolutionary"), no claims beyond
  the article.

## Written form in, speech out

Write `say` text in the form you want in the captions. The build converts it for the voice:

| You write | Narrator says |
|---|---|
| `300-800 ms` | 300 to 800 milliseconds |
| `25B tokens`, `10M`, `100k` | 25 billion, 10 million, 100 thousand |
| `410 GB` | 410 gigabytes |
| `p95`, `BM25`, `HNSW`, `RAG` | P 95, B M 25, H N S W, rag (from `video/tools/lexicon.json`) |
| `~`, `≈`, `×`, `→` | about, about, times, to |

Avoid in `say`: `e.g.`/`i.e.` mid-sentence (the sentence splitter may cut there; the lexicon
expands them but prefer "for example"), symbols the table above doesn't cover (`/`, `@`, `#`),
and "$25k" (write "$25,000"). Write "recall at 50", not "recall@50".

When a term is misread, add it to the script's `lexicon:` (or to `video/tools/lexicon.json` if it
will recur across videos). Two forms:

- A respelling, e.g. `Faiss: face`, `SQL: sequel`. Quick, but approximate.
- Exact phonemes in Kokoro's markup, e.g. `vidyal.ai: "[Vidyalai](/vɪdjˈɑlˌA/)"`. Kokoro's US
  symbols are IPA plus `A` = eɪ, `I` = aɪ, `O` = oʊ, `W` = aʊ, `Y` = ɔɪ; `ˈ`/`ˌ` mark stress.

Matching a pronunciation the owner recorded (how `vidyal.ai` was set):
1. `ffmpeg -i Recording.mp3 -ac 1 -ar 16000 rec.wav`, split it at pauses if it has several takes
   (`-af silencedetect=noise=-35dB:d=0.25` finds them).
2. Decode it with `video/tools/phonemes.py` (usage in its docstring) to get IPA.
3. Synthesize 3-4 candidate markups *inside a sentence* (single words come out clipped) and decode
   those too. Keep the one whose phonemes match the recording. If the recognizer is unsure of a
   vowel, measure formants (F1 of the vowel's start: ~700+ Hz for aɪ, ~450 Hz for eɪ).
4. Prefer this over splicing the owner's own recording into the narration: a second voice in the
   same track sounds jarring.

## On-screen text

- Headings ≤ 6 words. Bullets ≤ 7 words, with an optional sub-line ≤ 12 words.
- Put the number on screen when the narration says it.
- Use `**highlight**` sparingly: one phrase per scene.

## Shorts

- First sentence is the hook; it must make sense with the sound off (the opening `statement`
  scene carries the same words).
- One idea, three or four beats, then the outro. No chapter titles.
- Title ends in `#shorts`; description links the article and (after upload) the main video.
