# Writing practice sets (`/practice/`)

The `/practice/` section is a separate, **unlisted** part of the site. It holds multiple-choice reasoning and mental-ability practice for Indian students in **Class 7 and 8** (age 12–13), the kind used in school foundation-course entrance tests, NTSE-style mental-ability papers and reasoning olympiads. Nothing on the main site links to it.

## Hard rules

- **Original questions only.** Do not copy or closely paraphrase questions from any book, website, coaching material or olympiad paper. Write your own numbers, names and situations.
- **Never name a coaching institute, test brand or olympiad** anywhere (content, ids, comments).
- **Exactly one correct option.** Every other option must be clearly wrong to a careful student. Check every answer by computing it, not by eye (see Verification).
- Language a 12-year-old reads easily. Short sentences. Indian English spelling (colour, metre).
- Indian context: names like Asha, Ravi, Meera, Arjun, Priya, Kabir, Farhan, Simran, Joseph, Anjali, Vikram, Neha; ₹ for money; km, kg, litres; Indian cities, festivals, trains, cricket. Keep it neutral and friendly. Avoid religion-, caste- or gender-stereotyped roles.

## Files and size

- One set = one file `src/content/practice/<slug>.md`, where the slug is `<family>-<part>`, for example `speed-time-work-2`.
- `family` must be one of the slugs in `src/data/practice.ts`.
- **Each set has 12 questions (11–13) and `timeMinutes: 12` (10–15).** Topics are split into parts. Each part gets its own subtopic and title, and the parts don't repeat each other.
- Difficulty within a set goes from easy to hard: about **4 easy (1), 5 medium (2), 3 hard (3)**. Order the questions roughly by difficulty.
  - Hard means olympiad level for Class 8: several steps, or a trap. It never means content beyond Class 8 maths.
- **Exactly 4 options per question**, and each letter is the answer 2–4 times per set. Put numeric options in increasing order where natural.
- Distractors should come from **real mistakes**: wrong operation, off-by-one, reversed ratio, the mirror instead of the water image, and so on.
- Question ids are kebab-case and **unique across the whole section**. Start each one with a short set prefix, for example `stw2-train-bridge`.

## Frontmatter

```yaml
---
title: "Speed, time and work 2: trains and boats"   # "<Topic> <part>: <subtopic>", ≤ 80 chars
description: "One sentence on what this part covers."  # ≤ 220 chars
slug: speed-time-work-2
family: speed-time-work
part: 2
timeMinutes: 12
stems:            # optional: shared material for several questions
  - id: shop
    title: "Sales at Meera's shop"
    text: "Short paragraph. **bold** and *italic* allowed."
    figure: { kind: bars, ... }      # optional
    lines: ["line 1", "line 2"]       # optional list
    # studySeconds: 40   <- ONLY for memory sets: shown for that long, then hidden
questions:
  - id: stw2-train-pole
    topic: Trains           # sub-skill label, shown in the score breakdown (2–5 distinct per set)
    difficulty: 1
    q: "A 150 m long train passes a pole in 10 seconds. What is its speed in km/h?"
    options: ["15 km/h", "36 km/h", "54 km/h", "60 km/h"]
    answer: C
    explanation: |
      Speed = 150 ÷ 10 = 15 m/s. To change m/s to km/h, multiply by 18/5: 15 × 18 ÷ 5 = **54 km/h**.
      15 km/h forgets to convert the units.
  - id: stw2-...
    stem: shop              # links to a stem; questions with the same stem appear under it
    ...
---
```

Text in `q`, `options`, `explanation`, stem `text` and `lines` can use `**bold**`, `*italic*`, `` `code` `` and line breaks, nothing else. **No HTML, no Markdown links, no LaTeX/$$.** Use the Unicode characters × ÷ − ² ³ √ π ₹ ° ½ ¼ ¾. Write other fractions as 3/5. Quote YAML strings that contain `:`, `#`, `*` at the start, or `[`.

**Explanations** are 1–4 short lines. Give the method or shortcut, **bold the final answer**, and where useful say why the most tempting wrong option is wrong.

## Primer (the Markdown body)

The body is 150–350 words. Open with one sentence on what the set tests, then use `##` sections:
- the method or rules;
- one short worked example;
- a tip or common trap.

Tables are fine. No HTML, no images, no links. Students read it before the questions, so it should teach what the questions need without giving away specific answers.

## Figures

All figures are data and get drawn at build time; see `src/lib/practice/figures.ts` and the schema in `src/content.config.ts`. Use them as `figure:` on a question or stem, or as `optionFigures:` (4 figures) instead of `options:`.

| kind | Use for | Example |
|---|---|---|
| `table` | data tables, timetables | `{ kind: table, caption: "Marks", header: [Name, Maths], rows: [[Asha, 78], [Ravi, 92]] }` |
| `grid` | magic squares, number/letter/symbol grids | `{ kind: grid, cells: [[8, 1, 6], [3, "?", 7]], shade: [[1, 1]] }` (null = no cell, "" = empty cell) |
| `bars` | bar charts (values printed on bars) | `{ kind: bars, title: "Books read", labels: [Jan, Feb], values: [12, 18], unit: "Books" }` |
| `pie` | pie charts, values are % adding to 100 | `{ kind: pie, title: "Spending", labels: [Food, Rent], values: [40, 60] }` |
| `venn` | circle diagrams, 200 × 140 box | `{ kind: venn, circles: [{ c: [70, 70], r: 45, label: Birds }, { c: [130, 70], r: 45 }], marks: [{ at: [100, 70], s: "5" }] }` |
| `shapes` | line drawings in a 100 × 100 box (y goes **down**) | `{ kind: shapes, frame: true, transform: flipH, items: [...] }` |
| `glyphs` | a word/number as a picture, mirrored or rotated | `{ kind: glyphs, text: "PUNE 47", transform: flipH }` |
| `dice`, `net`, `stack`, `cubegrid` | cubes (see `cubes-dice-1.md`) | |

For offset layouts such as number pyramids, don't use `grid` with null cells, because the gaps read as extra cells. Draw the bricks as `rect` and `text` items in a `shapes` figure instead (see `number-puzzles-3.md`).

Letters and numbers on `dice` faces are always drawn upright; they don't turn the way a real folded net would. In any question that compares a net with cubes, add "(The letters are only labels: ignore which way they face.)", or use symbols that look the same when turned (● ○ ■ + ×).

**`shapes` items** (all coordinates in the box; y down):
- `{ t: line, p: [x1, y1, x2, y2], dash: true }`
- `{ t: arrow, p: [x1, y1, x2, y2] }` (arrowhead at the second point)
- `{ t: poly, p: [[x, y], ...], fill: none|solid|grey|accent, open: true }`
- `{ t: circle, c: [x, y], r: 8, fill: solid }`
- `{ t: rect, p: [x, y, w, h], fill: grey }`
- `{ t: text, at: [x, y], s: "F", size: 22 }`

`transform` is one of `none`, `flipH` (mirror image, mirror on the right), `flipV` (water image), `rot90`, `rot180` or `rot270` (clockwise). It applies to the whole drawing, text included.

**Build mirror, rotation, folding and series questions by transforming one base drawing** (same `items`, different `transform`), or by computing the coordinates in a script. Never draw the answer by hand. Use shapes without symmetry, or the wrong options might equal the right one.

## Verification (required for every question)

1. **Compute every answer with a script.** Put it in your scratch directory and run it, not in the repo.
   - Arithmetic, dates (Python `datetime`), clock angles: compute directly.
   - Codes: implement the rule and apply it.
   - **Logic puzzles** (seating, ranking, blood relations, truth-tellers, scheduling, syllogisms): brute-force every arrangement with `itertools` and check that **exactly one** arrangement fits the clues, and that each question has one answer.
   - Figure transforms: compute where every point goes.
2. `npx astro sync`: validates the schema in about 2 seconds. Ignore errors in files that are not yours, since another author may be mid-edit.
3. `node scripts/practice-lint.mjs src/content/practice/<your files>`: size, difficulty mix, answer balance, banned text. Fix everything it reports.
4. **For figures, look at them.** With the dev server running on port 4330:
   `WIDTHS=900 THEMES=light node scripts/screenshots.mjs http://localhost:4330 <your scratch dir> /practice/<slug>/`
   Then view the PNG, cropping it with `sharp` if it is tall. Check that each figure shows what the question says.
5. Read each question once more as a student. Is there a second defensible answer? Is anything ambiguous: "left" (whose left?), "between", "after", "next to", or inclusive versus exclusive counting? Fix the wording.
