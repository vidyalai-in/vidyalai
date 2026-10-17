# `script.yaml` format

```yaml
slug: <article slug>                      # output file names use it
article: src/content/blog/NN-<slug>.md    # sources and URL come from here
voice: af_heart                           # Kokoro voice
speed: 1.0                                # 0.9-1.1 is natural
lexicon: { Kubernetes: "koober netties" } # optional per-video pronunciations (TTS only)

youtube:
  title: "..."                            # ≤ 100 chars
  altTitles: ["...", "..."]
  summary: |                              # first lines of the description
    ...
  tags: [...]                             # ≤ 500 chars total
  hashtags: ["#RAG", "#LLM", "#SystemDesign"]
  playlist: LLM System Design
  pinnedComment: |
    ...

thumbnail:
  title: "RAG for **10M** Docs"           # 3-5 words; **x** is highlighted
  badge: "410 GB → 6 GB"                  # optional
  diagram: q01-rag-pipeline               # optional; omit for wide diagrams (logo art is used)

main:
  scenes: [ ... ]                         # landscape 1920×1080

shorts:
  - id: short-<name>                      # kebab-case, unique
    title: "... #shorts"
    description: "..."
    hashtags: [...]
    scenes: [ ... ]                       # portrait 1080×1920
```

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

To find focus rectangles, open `public/diagrams/<name>.png`, note node positions as fractions of
the width and height, and preview with `node video/tools/preview.mjs video/<slug> main <scene>:<step>`.
Wide diagrams (e.g. 7:1) are unreadable whole at 1080p; show the whole once as an establishing shot,
then focus on 2-4 nodes at a time.

### `code`
`code` (monospace block; `# comments` are dimmed) and optional `caption`.

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
- **Next video not out yet:** set `teaser` to its title (the frame shows it with a "Coming soon" pill)
  and use `minDuration: 12`; the say line names the next topic and asks viewers to subscribe. Only
  Subscribe goes on now. YouTube can't replace an uploaded video file, but end screens can be edited
  any time, so once the next video is public the uploader adds a Video element over the teaser.

### `outro`
`title`, `url` (default vidyal.ai), `path`, `next` (shown from `nextStep`, a 0-based step index; default 1, the second step).

## Timing

The build measures each synthesized sentence and lays out: 0.15 s lead-in, 0.3 s between sentences,
0.45 s after each step, and 0.4 s extra at the end of each scene. Entrance animations take ~0.47 s.
Every step is at least ~0.7 s long.
