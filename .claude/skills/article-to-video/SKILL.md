---
name: article-to-video
description: Turn a VidyalAI article (src/content/blog/*.md) into a narrated YouTube video (landscape, ~8-12 min), several vertical Shorts (under 60 s), and a ready-to-paste upload kit (title, description with chapters and sources, tags, captions, thumbnail, checklist). Use when asked to make a video, YouTube video, Short, reel or "video version" of an article on this site.
---

# Article to video

Everything runs locally: Kokoro (open-weight TTS) for the voice, the `video/tools/stage/` HTML page
for visuals (screenshotted by headless Chromium), and ffmpeg for encoding. You write one file,
`video/<slug>/script.yaml`; the build does the rest.

Worked example to copy from: `video/rag-system-10m-documents/script.yaml` (main video + 4 Shorts).

## 1. Read before writing

1. Read the whole article. Every number, name and claim in the narration must come from it.
   If you want to say something the article doesn't, add it to the article first (with a source).
2. Read `references/script-format.md` (scene types and fields) and
   `references/narration.md` (writing for the ear, timing budgets, pronunciation).
3. Look at the article's diagrams in `public/diagrams/<name>.png` if you plan camera moves on them.

## 2. Plan the videos

**Main video** (landscape 1920×1080):
- Target 8-12 minutes. Narration runs at about 145 words/minute plus ~12% pauses, so budget
  **1,300-1,700 narrated words**. The RAG example is ~1,600 words → 11:02.
- 6-10 chapters, each 30 s or longer (YouTube needs ≥ 3 chapters, first at 0:00, each ≥ 10 s).
- Follow the article's arc: hook → the core idea → requirements → design (with diagrams and
  numbers) → measurement → tradeoffs → common mistakes → recap → outro pointing to the article →
  `endcard` (20 s, for YouTube's end-screen elements).
- Adapt, don't read: drop "Further reading" (it goes in the description), turn tables into
  `compare`/`budget`/`stats` scenes, and cut detail the visuals already carry.

**Shorts** (portrait 1080×1920, 3-4 per article):
- Each one stands alone, opens with a hook in the first sentence, and makes **one** point.
- 80-120 narrated words → 30-50 s. Hard limit: under 60 s (the build warns).
- Good sources: a striking number (sizing math), a counter-intuitive rule, a common bug, a
  latency or cost budget. End with an `outro` scene pointing to vidyal.ai.

## 3. Write `video/<slug>/script.yaml`

Use `slug` = the article's slug. Fill `youtube:` (title ≤ 100 chars, 2 alternatives, summary
paragraph, 10-15 tags, 3 hashtags, pinned comment with the article URL) and `thumbnail:` (3-5 word
title with one **highlight**, a short badge). Then `main.scenes` and `shorts`.

Rules that bite:
- **Quote any YAML value that contains a comma or colon** inside `{ ... }`. Unquoted, YAML splits
  it and the text is silently cut. The build refuses to run if it detects this.
- Write `say:` in written form ("25B tokens", "300-800 ms", "p95"): captions use it as-is, and
  the build converts it for speech (ranges, units, k/M/B, and `video/tools/lexicon.json`).
  Add per-video pronunciations under `lexicon:` when a term is misread.
- One sentence should never need a comma-separated list of more than ~4 items. Split it.

## 4. Build and review

```sh
node video/tools/preview.mjs video/<slug> main 5:2 6:1      # stills of scene:step, 1-based (fast)
node video/tools/build.mjs video/<slug> --fast               # all videos, quick encode
node video/tools/build.mjs video/<slug> --only short-x --fast
node video/tools/build.mjs video/<slug>                      # final quality
```

- The first run installs Kokoro and PyTorch through `uv` (several GB for the CUDA build). If the
  download keeps failing, or there's no NVIDIA GPU, add `--torch cpu` (about 4× real time instead
  of ~50×). TTS output is cached in `video/.tts-cache/`, so rebuilds only voice changed sentences.
- Change the voice with `voice:` in the script or `--voice` (`af_heart` default; others:
  `am_michael`, `bf_emma`, `bm_george`; see the Kokoro-82M model card for the full list).

Review every build, don't trust it blind:
1. Read the build output: durations, chapter count, and any WARNING lines.
2. Open `video/<slug>/out/<id>-contact.png` (16 frames per video) and check layout: nothing
   clipped, nothing overlapping, text readable, diagrams framed on what the narration is about.
3. Spot-check frames at specific times with
   `ffmpeg -ss <sec> -i out/<file>.mp4 -frames:v 1 frame.png` and look at them.
4. Check loudness (target about -14 to -15 LUFS, true peak under -1 dB):
   `ffmpeg -i out/<file>.mp4 -af ebur128=peak=true:framelog=quiet -f null - 2>&1 | grep -E ' I:| Peak:'`.
5. Check the pauses are silent: render a spectrogram of any 10 s
   (`ffmpeg -ss 140 -t 10 -i out/<file>.mp4 -vn -lavfi showspectrumpic=s=1600x400:scale=log spec.png`)
   and look at it. Gaps between sentences should be near-black (below about -90 dB). Bright noise in
   the gaps means something is amplifying the TTS noise floor; never put `loudnorm` or any other
   dynamic gain/compression on the narration (the build uses one static gain plus a limiter).
6. If the user reports a mispronounced word, follow "Matching a pronunciation" in
   `references/narration.md`, then re-check the rebuilt clips with `video/tools/phonemes.py`.
7. Re-read `youtube/main/description.txt` and the chapter titles.

## 5. Outputs and hand-off

| Path | Committed | What |
|---|---|---|
| `video/<slug>/script.yaml` | yes | The source of truth |
| `video/<slug>/youtube/` | yes | `UPLOAD.md` checklist; `main/` title, description, chapters, tags, captions, thumbnail, pinned comment; `shorts/<id>/` title, description, captions |
| `video/<slug>/out/*.mp4` | no (large) | The videos; rebuild any time from the script |

Tell the user where the MP4s are, the main video's length and chapters, each Short's length, and
point them to `youtube/UPLOAD.md`. Mention that the voice is AI-generated (the description credits
it) and that they should follow YouTube's current disclosure guidance.
