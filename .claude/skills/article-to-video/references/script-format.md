# `script.yaml` format

```yaml
slug: <article slug>                      # output file names use it
article: src/content/blog/NN-<slug>.md    # sources, URL and series come from here
voice: af_heart                           # Kokoro voice
speed: 1.0                                # 0.9-1.1 is natural
lexicon: { Kubernetes: "koober netties" } # optional per-video pronunciations (TTS only)

fx:                                       # engagement effects; all off when omitted (see below)
  drift: true                             # slow push-in on every scene, panning dot grid
  wordSync: true                          # **highlights** glow as the narrator says them
  sfx: true                               # quiet whoosh/pop/tick/ding cues
  # music: synth                          # optional bed; off by default and not recommended (see below)
  shortsCaptions: true                    # burned-in word-by-word captions on the Shorts

voices:                                   # second voices for `speaker:` steps (Kokoro voice names)
  skeptic: { voice: am_michael, label: "The skeptic" }

youtube:
  title: "..."                            # ≤ 100 chars
  altTitles: ["...", "..."]
  summary: |                              # first lines of the description
    ...
  tags: [...]                             # ≤ 500 chars total
  hashtags: ["#RAG", "#LLM", "#SystemDesign"]
  playlist: LLM System Design             # optional; defaults to the article's series name
  pinnedComment: |
    ...

thumbnail:
  title: "RAG for **10M** Docs"           # 3-5 words; **x** is highlighted
  badge: "410 GB → 6 GB"                  # optional
  diagram: q01-rag-pipeline               # optional; omit for wide diagrams (logo art is used)
# Or a list of up to 3 designs for YouTube's Test & compare: thumbnail.png, thumbnail-b.png, -c.png.

main:
  scenes: [ ... ]                         # landscape 1920×1080

shorts:
  - id: short-<name>                      # kebab-case, unique
    title: "... #shorts"
    description: "..."
    hashtags: [...]
    scenes: [ ... ]                       # portrait 1080×1920
```

### Series

The article's frontmatter `series` (default `llm-system-design`) decides the series; names, slugs and
playlists are in `src/data/series.mjs`. The build writes the series line of the description and the
default playlist from it. Write the rest to match that series by hand:
- the title suffix (`... | LLM Fundamentals`) and hashtags (`#SystemDesign` only for System Design),
- `summary`: "Part N of the <series name> series from VidyalAI", the earlier parts' video links, and the
  series playlist link once it exists,
- the title scene's `kicker` (`LLM Fundamentals · Part 1`) and the endcard's `kicker` and
  `path: /series/<series slug>`.

## Scenes and steps

Every scene has `type` and `steps`. A step is one beat of narration: its `say` text is voiced,
and the visual change it carries (a new bullet, a camera move) animates in as the narration starts.

Common scene fields:

| Field | Meaning |
|---|---|
| `chapter` | Starts a YouTube chapter at this scene (main video only). Also shown bottom-right. |
| `kicker` | Small uppercase line above the heading |
| `heading` | Scene heading; `**x**` highlights, `` `x` `` is code |

Common step fields:

| Field | Meaning |
|---|---|
| `say` | Narration, one or more sentences, written form (see narration.md) |
| `hold` | Seconds to show a step that has no `say` (default 2.5) |
| `animate: true` | Force an entrance animation on a step with no visual change |
| `minDuration` | Hold the step at least this many seconds (the end card uses 20) |
| `speaker` | Read this step in a second voice from `voices:`, with a speech bubble showing the line |

## Scene types

### `brand`
Full-frame banner (`src/assets/brand/banner1.png`, or the mobile crop in Shorts). Use once, first,
with `steps: [{ hold: 3 }]`.

### `title`
`kicker`, `heading` (large), `subtitle`, optional `subtitleStep` (step at which the subtitle appears).

### `statement`
One big line: `text`, optional `sub` and `subStep`. Good for the core idea, transitions and recaps.

### `bullets`
Each step with `add` reveals an item. `add` is a string or `{ text, sub }`.
Scene options: `numbered: true` (1, 2, 3 marks) or `style: warn` (amber "!" marks, for mistakes).
Keep to 3-5 items.

### `stats`
Each `add: { value, label }` reveals a big-number card; cards are joined by arrows and the newest is
highlighted. Values starting with a number count up (`"410 GB"`, `"$52k"`, `"6–13 GB"`); set
`count: false` to disable. 2-4 cards.

### `budget`
A waterfall of timing bars. Each `add: { label, sub?, ms }` where `ms` is a number or `[low, high]`.
Scene options: `total` (text shown on the step with `showTotal: true`, or the last step),
`max` (scale; default = sum of upper bounds), `waterfall: false` (bars all start at 0).

### `compare`
Option cards: `add: { name, pros, cons, when }` (`pros`/`cons` are a string or a list).
Scene option `cols` (default 3; use 2 for four options).

### `diagram`
`diagram: <name>` uses `public/diagrams/<name>-dark.svg` (render new ones with `npm run diagrams`).
Steps can set:
- `focus: [x, y, w, h]` — the camera frames that rectangle (fractions of the diagram, 0-1) and
  glides there from the previous focus. It carries over until changed. Default: whole diagram.
- `note` — caption shown under the diagram during that step only.

The diagram is inlined, so steps can also animate it (node ids are the Mermaid ids, e.g. `TK`):
- `highlight: [A, B]` — light those nodes and dim the rest (carries over until changed).
- `flow: [A, B, C]` — a glowing pulse travels A → B → C over `flowTime` seconds at an even speed
  (eased only at the start and end), passing behind each box and lighting it. Add `follow: 0.4` to have
  the camera glide along with it, a window 40% of the diagram wide; this is how to show a wide pipeline
  readably. Prefer one long flow to many short camera hops.
- For a walkthrough, give each block its own step with `flow: [previous, this]` and `flowTime: 1.2`:
  the pulse reaches the block just as that step's narration starts describing it, the block glows,
  and blocks already passed stay at full strength as a trail. End with a step that pulls back
  (`focus: [0, 0, 1, 1]`) and highlights every block.

Camera moves between `focus` rectangles take 1.4 s with a gentle start and stop, and zoom evenly. Avoid
big jumps between distant parts of a diagram in one step: pull back to the whole diagram instead.

To find focus rectangles, open `public/diagrams/<name>.png`, note node positions as fractions of
the width and height, and preview with `node video/tools/preview.mjs video/<slug> main <scene>:<step>`.
Wide diagrams (e.g. 7:1) are unreadable whole at 1080p; show the whole once as an establishing shot,
then focus on 2-4 nodes at a time.

### `figure`
An interactive-style isometric figure (a bench machine drawn with `video/tools/iso/`, the `anatomy`
skill's kit) whose moving parts are posed from the scene clock. Each figure is a build script next
to the video, `video/<slug>/figures/<name>.mjs`, which writes `<name>.json` (run it after editing;
`--page` also writes a looping `<name>.html` to check in a browser). The scene names it:

```yaml
- type: figure
  chapter: The whole machine
  figure: machine
  steps:
    - say: Here's the whole machine...
    - station: TK          # the figure's own step fields: what moves in this step
      focus: TK            # optional: a named target or [x, y, w, h] (fractions); defaults to station
      say: First, a tokenizer splits the text...
```

- The camera glides (1.4 s, sine-eased) to each step's `focus` and holds; the card's corners
  carry the figure number, title, the current part's name and a live readout.
- Step fields are defined by each figure (its header comment lists them). Examples in
  `video/how-llms-work/figures/`: `machine` (`station: T|TK|EM|ST|LG|SM|SP|RE|ALL`, or a list run
  in sequence) and `sampler` (`greedy`, `spin: <roll>|[rolls]`, `temp`, `light`, `focus`).
- Readability on video: put words on the upright front faces of tiles (`frontText`/`tile` in
  `video/tools/iso/video-parts.mjs`), not on top faces, in Inter at 6+ world units, and zoom the
  camera onto the part being narrated. Check a still at 1080p for every step.
- Every number a figure shows must come from the article, as with any scene.

### `code`
`code` (monospace block; `# comments` are dimmed) and optional `caption`.

### `quote`
A large quote card: `quote`, `source` (optional `sourceStep`), `kicker`. Quote exactly from the article
or its source. Good as a pattern interrupt in a run of explanation.

### `chart`
Data drawn from numbers in the article (never invented). Two kinds:

- **Bars** (default): `bars: [{ label, value, display?, sub? }]`, scene `unit`, `decimals`, `max`, `log`.
  Bars grow in on step 0. Steps can set `values` (bars morph to the new numbers), `max`/`unit`/`decimals`
  (e.g. logits morphing into percentages), `show: n` (reveal the first n bars), `tag` (a badge, e.g.
  `T = 2`), `highlight: <label>` (light one bar). Scene `note` adds a source line under the chart.
- **Line** (`chart: line`): `series: [{ name, points: [[x, y, "label"?], ...] }]`, `xLog`/`yLog`,
  `xMin`/`xMax`/`yMin`/`yMax`, `xTicks`/`yTicks` as `[[value, "label"], ...]`. A step with
  `draw: <name>` draws that series over `drawTime` seconds; points appear as the line reaches them.

### `tokens`
Next-token prediction on real tokens. Scene: `text`, `pieces` (token strings, leading spaces kept),
`ids`, `candidates: [{ t, p }]` (percent; the first is the winner), `typeTime`, `chips: false` (keep
the plain sentence and type the winner onto it, for a cold open), `from: <stage>` (earlier stages are
already shown). Each step's `show` moves to a stage: `text` (typed) → `pieces` (chips) → `ids` →
`candidates` (probability bars) → `append` (the winner joins the sequence).

### `quiz`
"Pause and guess": `question`, `options` (2-3 short answers), `answer` (0-based), `explain` (shown on
the reveal), `kicker` (default "Pause and guess"). Steps: one that asks, one with `countdown: 4` (no
`say`; a ring runs down with ticks), one with `reveal: true` (the right option lights up, with a bell).
Use one or two per main video, on a number the viewer can reason about.

### `seriesmap`
Every part of the article's series as tiles, with "You are here" on this one; parts come from
`src/content/blog` frontmatter. Steps can `highlight: [n, ...]` other parts ("Part 2 covers...").
Optional `labels: { 9: "Encoders and decoders" }` to shorten tile titles, `cols`.

### `endcard`
Last scene of the main video: a blank layout for YouTube end-screen elements, with a dashed 16:9
frame (left) for a video element and a dashed circle (right) for Subscribe. Fields: `kicker`,
`title` (default "Keep learning"), `url`, `path`, `videoLabel`, `subscribeLabel`, `teaser`,
`teaserNote` (default "Coming soon"). Give it one step with a short call to action and a
`minDuration` (YouTube end screens run 5-20 s). The brand bar and progress line are hidden, and
`UPLOAD.md` gets the exact end-card start time for the end-screen step. Not used in Shorts (they
don't support end screens).

- **Next video already public:** no `teaser`, `minDuration: 20`; the say line invites viewers to pick
  the next video. The uploader puts a Video element in the frame and Subscribe in the circle.
- **Next video not out yet:** set `teaser` to its title (it becomes the heading, under a "Next in the
  series · Coming soon" kicker; the frame stays empty, because YouTube shows end-screen elements
  see-through in the editor, so anything drawn in the frame shows behind them) and use `minDuration: 12`; the say line names the next topic and asks viewers to subscribe. YouTube
  requires at least one video or playlist element, so for now the uploader puts the series Playlist
  element in the frame and Subscribe in the circle. YouTube can't replace an uploaded video file,
  but end screens can be edited any time, so once the next video is public the uploader swaps the
  playlist for that video.

### `outro`
`title`, `url` (default vidyal.ai), `path`, `next` (shown from `nextStep`, a 0-based step index; default 1, the second step).

## Engagement effects (`fx`)

- `drift`: every scene slowly pushes in (2.8% over the scene) and the dot grid pans, so no frame is a
  still. Not applied to the end card (YouTube places its elements on fixed slots).
- `wordSync`: each `**highlighted**` phrase glows for ~0.9 s when the narration says its first word.
  Word timings come from Kokoro's duration model, mapped back onto the written words.
- `sfx`: a whoosh into each chapter, a soft pop on each reveal, ticks on a quiz countdown, a bell on
  the answer, key clicks while the `tokens` scene types. Synthesized; levels are fixed and quiet.
- `music` (off unless set; tried and dropped on the Fundamentals Part 1 video because it added little
  under narration): `music: synth` generates a calm ambient bed (chords, a soft pluck, reverb) per video, seeded by the
  slug, so it is ours: no licence and no Content ID claims. A file path loops that track instead; give
  `musicCredit`. Either way it gets one static gain, `musicGap` dB below the narration.
- `shortsCaptions`: big captions in the Shorts, 1-4 words at a time with the current word lit, above
  the Shorts UI; the scene area shrinks to make room. A Short's first frame shows its first scene fully
  (no fade-in), so the hook is visible before anyone hears it.

## Timing

The build measures each synthesized sentence and lays out: 0.15 s lead-in, 0.3 s between sentences,
0.45 s after each step, and 0.4 s extra at the end of each scene. Entrance animations take ~0.47 s.
Every step is at least ~0.7 s long.
