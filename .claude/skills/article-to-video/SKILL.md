---
name: article-to-video
description: Turn a VidyalAI article (src/content/blog/*.md) into a narrated YouTube video (landscape, ~8-12 min), 3 vertical Shorts (under 60 s), and a ready-to-paste upload kit (title, description with chapters and sources, tags, captions, thumbnail, checklist). Use when asked to make a video, YouTube video, Short, reel or "video version" of an article on this site.
---

# Article to video

Everything runs locally: Kokoro (open-weight TTS) for the voice, the `video/tools/stage/` HTML page
for visuals (screenshotted by headless Chromium), and ffmpeg for encoding. You write one file,
`video/<slug>/script.yaml`; the build does the rest.

Worked examples: `video/how-llms-work/script.yaml` uses every engagement feature (cold open,
charts, tokens scene, quizzes, quote card, series map, a second voice, sound effects, Shorts captions
and loops, three thumbnails); copy its structure for new videos.
`video/rag-system-10m-documents/script.yaml` is the older, plainer style.

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
- Open cold: the hook (a `tokens` demo, a striking number) comes **before** the brand scene, then an
  open loop (a payoff promised for later), then a `seriesmap` placing this part in its series.
- Follow the article's arc. System Design posts: hook → the core idea → requirements → design (with
  diagrams and numbers) → measurement → tradeoffs → common mistakes → recap. Explainer posts
  (Fundamentals, Inference): "Why it matters" → hook; "The intuition" → an analogy scene; the
  how-it-works sections → the body; "Common misconceptions" → a myth-vs-fact scene; "Key takeaways"
  → recap; "What's next" → a `seriesmap` lighting the next part, then the endcard teaser. Then an
  outro pointing to the article →
  `endcard` (for YouTube's end-screen elements: 20 s if the next video is already public, or a
  12 s `teaser` naming it if it isn't; see `references/script-format.md`).
- Adapt, don't read: drop "Further reading" (it goes in the description), turn tables into
  `chart`/`compare`/`budget`/`stats` scenes, and cut detail the visuals already carry.
- Engagement: turn on `fx` (drift, wordSync, sfx, shortsCaptions). Leave `music` off: a bed was
  tried on the Fundamentals Part 1 video and the owner found it added little under narration. Every 60-90 s, break the
  explanation with something different: a chart that animates, a `tokens` demo, a `quiz` (one or two
  per video), a `quote` card, a diagram `flow`. Put misconceptions in the mouth of a second voice.
  Offer 2-3 `thumbnail` designs for YouTube's Test & compare.
- Isometric `figure` scenes (the `anatomy` skill's style: a real machine whose working is the idea)
  cost about an hour each to build, so use them only where they carry the most: the one scene that
  walks through the whole mechanism, and at most one or two others where motion is the point (e.g.
  sampling and temperature). Everything else stays as charts, tokens and diagrams. See `figure` in
  `references/script-format.md`.

**Shorts** (portrait 1080×1920, 3 per article):
- Each one stands alone, opens with a hook in the first sentence, and makes **one** point.
- 60-120 narrated words → 25-50 s. Hard limit: under 60 s (the build warns).
- Pick the three strongest of these sources: a striking number (sizing math), a counter-intuitive rule, a common bug, a
  latency or cost budget. End on a complete sentence that echoes the hook (see narration.md), with
  or without an `outro` scene pointing to vidyal.ai. Never end on a trailing half-sentence.
- With `fx.shortsCaptions`, words are burned in above the Shorts UI; keep 2-4 s beats.

## 3. Write `video/<slug>/script.yaml`

Use `slug` = the article's slug. Match the article's series in the title suffix, summary and
kickers (see "Series" in `references/script-format.md`). Fill `youtube:` (title ≤ 100 chars, 2 alternatives, summary
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

Every frame is rendered (so drift, charts and captions move smoothly) by parallel headless browsers,
each piping to its own encoder: `--workers 6` by default, about 60 frames/s in total, so a 12-minute
video renders in about 5 minutes. Frames that would be identical reuse the previous screenshot.
PyTorch is pinned (`--torch-version`, default 2.14.1) so a new release doesn't force a multi-GB
download; upgrade on purpose.

- The first run installs Kokoro and PyTorch through `uv` (several GB for the CUDA build). If the
  download keeps failing, or there's no NVIDIA GPU, add `--torch cpu` (about 4× real time instead
  of ~50×). TTS output is cached in `video/.tts-cache/`, so rebuilds only voice changed sentences.
- Change the voice with `voice:` in the script or `--voice` (`af_heart` default; others:
  `am_michael`, `bf_emma`, `bm_george`; see the Kokoro-82M model card for the full list).

Review every build, don't trust it blind:
1. Read the build output: durations, chapter count, and any WARNING lines, including the pacing
   warnings (fix them in the script unless there's a reason).
2. Open `video/<slug>/out/<id>-contact.png` (16 frames per video) and check layout: nothing
   clipped, nothing overlapping, text readable, diagrams framed on what the narration is about.
3. Spot-check frames at specific times with
   `ffmpeg -ss <sec> -i out/<file>.mp4 -frames:v 1 frame.png` and look at them.
4. Check loudness (target about -14 to -15 LUFS, true peak under -1 dB):
   `ffmpeg -i out/<file>.mp4 -af ebur128=peak=true:framelog=quiet -f null - 2>&1 | grep -E ' I:| Peak:'`.
5. Check the pauses are silent in the narration stem, `out/<id>.narration.wav` (the final mix adds
   sound effects): render a spectrogram of any 10 s
   (`ffmpeg -ss 140 -t 10 -i out/main.narration.wav -lavfi showspectrumpic=s=1600x400:scale=log spec.png`)
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
| `video/<slug>/youtube/main/cards.txt` | yes | Times where the narration names another part, for YouTube cards (manual) |
| `video/<slug>/youtube/main/scenes.json` | yes | Scene start/end times, for `upload.mjs analytics` |
| `video/<slug>/youtube/main/thumbnail*.png` | yes | Thumbnail plus Test & compare variants |
| `video/<slug>/out/*.mp4` | no (large) | The videos; rebuild any time from the script |

Tell the user where the MP4s are, the main video's length and chapters, each Short's length, and
point them to `youtube/UPLOAD.md`. Mention that the voice is AI-generated (the description credits
it) and that they should follow YouTube's current disclosure guidance.

To upload, the user can follow `UPLOAD.md` by hand or use `video/tools/upload.mjs` (YouTube Data API):
`upload video/<slug>` puts every video up as private with all metadata, captions, thumbnail and playlist;
`publish video/<slug>` makes them public and posts the comment. It records video IDs in
`video/<slug>/youtube/uploaded.json` (committed). The end screen, each Short's Related video and pinning the
comment stay manual. OAuth credentials live in `~/.config/vidyalai/`, never in the repo; the user signs in
with `! node video/tools/upload.mjs auth`. Setup, release steps and quota limits: `video/UPLOADING.md`.

A few days after a video is public, run `node video/tools/upload.mjs analytics video/<slug>`: it
pulls the audience-retention curve, maps it onto the scenes, and writes `youtube/main/retention.md`
(the scenes losing viewers fastest). Use it when writing the next script. It needs the
`yt-analytics.readonly` scope: if the saved sign-in predates it, the user re-runs `upload.mjs auth`.

When the user shares the published video's link (or `publish` prints it), embed it in the article: add
`video: { youtube: "<ID>", duration: "<m:ss>" }` to the post's frontmatter and copy
`video/<slug>/youtube/main/thumbnail.png` to `src/assets/video/<slug>.png` (see `content-src/AUTHORING.md`).
