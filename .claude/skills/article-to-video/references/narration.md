# Writing narration

## Voice and pacing

- Spoken, not written: short sentences, one idea each, active voice. Read every `say` aloud in
  your head; if you'd stumble, rewrite it.
- About 145 words per minute with Kokoro `af_heart` at speed 1.0. Budget the main video at
  1,300-1,700 words and each Short at 80-120 words.
- Open with a concrete hook (a number, a scenario), not "In this video", and put it **before** the
  brand scene: the first narration should start within 1-2 seconds. State the core idea in the first
  45 seconds.
- Name the problem in the viewer's own words (the symptom, the question they'd type into a search
  box), not the article's terms for it. In the first 30 s, also say who it's for in a phrase ("if
  you've built a RAG demo and wondered why it falls apart at scale"), so the right viewers stay.
- Open a loop in the first minute: promise a specific payoff that comes later ("by the end, you'll
  know why a model 100 times smaller won"), and pay it off where the article does.
- Change the picture every 4-6 seconds. Keep steps to one or two sentences and give each step a visual
  change; the build warns when a step holds one picture for over 10 s, when three text scenes run in a
  row, or when 90 s pass without a visual scene (chart, tokens, quiz, diagram, stats).
- Give the viewer something to do once or twice: a `quiz` on a number they can reason about, asked
  in one step, then a 4 s countdown, then the reveal and why.
- A second voice (`speaker: skeptic`) asking the question a viewer would ask is a good way to bring
  in a misconception. Keep its lines to one short question; the narrator answers in the next step,
  from the article.
- Signpost: "First…", "Second…", "Now the online side." Viewers can't scroll back.
- Each step's narration should match what appears on screen during it. Introduce an item in the
  same step that reveals it.
- Land each reveal on its word. A step's visual change animates in as its narration starts, so open
  the step with the words that name what appears: "Recall at 50 drops by half", not "Now, the next
  thing worth looking at is recall at 50". If the name has to come late in the sentence, split the
  sentence and put the reveal on the second step, or keep the sentence and add `at: "<word>"`
  to the step (see `script-format.md`), which starts the visual on that word.
- Never point at the screen. "As you can see here", "this one", "look at that" mean nothing in the
  captions, the transcript, or to someone listening without looking. Say what is there: "The bar for
  Paris jumps to 85%." `video/tools/narration.mjs` flags the common cases with `!`.
- Don't read on-screen text word for word: a viewer can't read and listen to the same words at
  once. The bullet carries the label; the narration says it as a full sentence and adds the why.
- Leave room after a result. When a chart lands or a quiz reveals, give the viewer a moment to
  read it: end the step on a short sentence, or follow it with a silent step (`hold: 1.5`).
  Filling every second with talk is the most common pacing fault.
- For `code` scenes, say what the block does first, then show it and let it sit. Narration that
  competes with reading code loses both.
- No filler ("let's dive in", "without further ado"), no hype ("revolutionary"), no claims beyond
  the article.
- The `say` text is the final audio: there is no second take to fix a misread line. Read it as the
  voice will (after the conversions below) and watch for places the default reading will get wrong:
  a stress that changes the meaning, a pause the comma doesn't give, a term the lexicon doesn't know.

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
  scene carries the same words, and it is on screen from the first frame).
- One idea, three or four fast beats (2-4 s each). No chapter titles.
- End on a complete sentence that echoes the hook ("...It's why the same question can get a
  different answer." for a Short that opens "Same question, different answer."), so a replay reads
  naturally. Never end on a trailing half-sentence: it sounds cut off. The build holds the last
  frame an extra second. You can skip the outro scene; the brand mark in the corner carries the site.
- Title ends in `#shorts`; description links the article and (after upload) the main video.
