import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { CATEGORY_SLUGS } from './data/categories';
import { DEFAULT_SERIES, SERIES_SLUGS } from './data/series.mjs';
import { PRACTICE_FAMILY_SLUGS } from './data/practice';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
  schema: z.object({
    title: z.string().max(90),
    description: z.string().max(220),
    slug: z.string().regex(/^[a-z0-9]+(-[a-z0-9]+)*$/),
    category: z.enum(CATEGORY_SLUGS),
    tags: z.array(z.string().regex(/^[a-z0-9]+(-[a-z0-9]+)*$/)).min(1).max(8),
    /** Posts without one belong to LLM System Design. */
    series: z.enum(SERIES_SLUGS).default(DEFAULT_SERIES),
    seriesOrder: z.number().int().min(1).max(50),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    sourceDoc: z.string().optional(),
    /**
     * The article's video: YouTube ID, length (m:ss) and, optionally, the link to use for "Watch on
     * YouTube" (e.g. a share link). Thumbnail: src/assets/video/<slug>.png.
     */
    video: z
      .object({
        youtube: z.string().regex(/^[\w-]{11}$/),
        duration: z.string().regex(/^\d+:\d{2}$/),
        url: z.string().url().optional(),
      })
      .optional(),
  }),
});

/* ---------- Practice: unlisted MCQ sets under /practice/ (see src/data/practice.ts) ---------- */

const face = z.union([z.string(), z.number()]).nullable();
const xy = z.tuple([z.number(), z.number()]);
const fill = z.enum(['none', 'solid', 'grey', 'accent']).default('none');

/** Primitives for the `shapes` figure, in a box that is 100 × 100 unless w/h say otherwise. */
const shapeItem = z.discriminatedUnion('t', [
  z.object({ t: z.literal('line'), p: z.tuple([z.number(), z.number(), z.number(), z.number()]), dash: z.boolean().optional(), w: z.number().optional() }),
  z.object({ t: z.literal('arrow'), p: z.tuple([z.number(), z.number(), z.number(), z.number()]), w: z.number().optional() }),
  z.object({ t: z.literal('poly'), p: z.array(xy).min(2), fill, open: z.boolean().optional() }),
  z.object({ t: z.literal('circle'), c: xy, r: z.number(), fill }),
  z.object({ t: z.literal('rect'), p: z.tuple([z.number(), z.number(), z.number(), z.number()]), fill }),
  z.object({ t: z.literal('text'), at: xy, s: z.string(), size: z.number().optional() }),
]);

/** Figures are drawn from data at build time (src/lib/practice/figures.ts), so they follow the theme. */
export const figureSchema = z.discriminatedUnion('kind', [
  /** One or more isometric cubes; each shows its top, front and right faces. */
  z.object({
    kind: z.literal('dice'),
    pips: z.boolean().default(false),
    cubes: z.array(z.object({ top: face, front: face, right: face })).min(1).max(5),
    labels: z.array(z.string()).optional(),
  }),
  /** An unfolded cube: a grid of cells, null for empty. */
  z.object({ kind: z.literal('net'), pips: z.boolean().default(false), rows: z.array(z.array(face)).min(1).max(5) }),
  /** Unit cubes piled on a grid: heights[row][col], row 0 at the back. Hidden cubes under visible ones count. */
  z.object({ kind: z.literal('stack'), heights: z.array(z.array(z.number().int().min(0).max(6))).min(1).max(6) }),
  /** A big cube cut into n×n×n small cubes. */
  z.object({ kind: z.literal('cubegrid'), n: z.number().int().min(2).max(6) }),
  /** A hand-drawn SVG in src/assets/practice/<src>.svg. */
  z.object({ kind: z.literal('svg'), src: z.string(), alt: z.string().min(10) }),
  /** A data table. */
  z.object({ kind: z.literal('table'), caption: z.string().optional(), header: z.array(z.string()).optional(), rows: z.array(z.array(z.union([z.string(), z.number()]))).min(1) }),
  /** Square cells with numbers/letters/symbols (magic squares, number grids, letter grids). null = no cell. */
  z.object({
    kind: z.literal('grid'),
    cells: z.array(z.array(face)).min(1).max(12),
    shade: z.array(z.tuple([z.number().int(), z.number().int()])).optional(),
    size: z.number().optional(),
  }),
  /** Vertical bar chart; each bar is labelled with its value. */
  z.object({ kind: z.literal('bars'), title: z.string().optional(), labels: z.array(z.string()).min(2), values: z.array(z.number()).min(2), unit: z.string().optional() }),
  /** Pie chart; values are percentages adding up to 100. */
  z.object({ kind: z.literal('pie'), title: z.string().optional(), labels: z.array(z.string()).min(2), values: z.array(z.number()).min(2) }),
  /** Circles for Venn-diagram questions, in a box 200 × 140 unless w/h say otherwise. */
  z.object({
    kind: z.literal('venn'),
    w: z.number().optional(),
    h: z.number().optional(),
    circles: z.array(z.object({ c: xy, r: z.number(), label: z.string().optional() })).min(1).max(5),
    marks: z.array(z.object({ at: xy, s: z.string() })).optional(),
  }),
  /** Line drawing from primitives; `transform` mirrors or rotates the whole drawing (rotations are clockwise). */
  z.object({
    kind: z.literal('shapes'),
    w: z.number().optional(),
    h: z.number().optional(),
    frame: z.boolean().default(false),
    transform: z.enum(['none', 'flipH', 'flipV', 'rot90', 'rot180', 'rot270']).default('none'),
    items: z.array(shapeItem).min(1),
    alt: z.string().optional(),
  }),
  /** Text drawn as a picture, optionally mirrored (flipH = mirror image, flipV = water image) or rotated. */
  z.object({
    kind: z.literal('glyphs'),
    text: z.string(),
    transform: z.enum(['none', 'flipH', 'flipV', 'rot90', 'rot180', 'rot270']).default('none'),
    size: z.number().optional(),
  }),
]);

const LETTERS = ['A', 'B', 'C', 'D', 'E'] as const;

const question = z
  .object({
    id: z.string().regex(/^[a-z0-9-]+$/),
    /** Sub-skill, used for the score breakdown. */
    topic: z.string(),
    difficulty: z.union([z.literal(1), z.literal(2), z.literal(3)]),
    q: z.string().min(5),
    figure: figureSchema.optional(),
    options: z.array(z.string()).min(3).max(5).optional(),
    optionFigures: z.array(figureSchema).min(3).max(5).optional(),
    answer: z.enum(LETTERS),
    explanation: z.string().min(10),
    /** Links the question to a stem (shared passage, chart, puzzle or memorize block). */
    stem: z.string().optional(),
  })
  .superRefine((q, ctx) => {
    const opts = q.options ?? q.optionFigures;
    if (!opts || (q.options && q.optionFigures)) {
      ctx.addIssue({ code: 'custom', message: `${q.id}: give exactly one of options / optionFigures` });
      return;
    }
    if (LETTERS.indexOf(q.answer) >= opts.length) ctx.addIssue({ code: 'custom', message: `${q.id}: answer ${q.answer} has no option` });
    const keys = opts.map((o) => JSON.stringify(o).trim()); // exact: capitals and punctuation can be the point of a question
    if (new Set(keys).size !== keys.length) ctx.addIssue({ code: 'custom', message: `${q.id}: duplicate options` });
  });

const practice = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/practice' }),
  schema: z
    .object({
      title: z.string().max(80),
      description: z.string().max(220),
      slug: z.string().regex(/^[a-z0-9]+(-[a-z0-9]+)*$/),
      /** The topic this set is a part of (PRACTICE_FAMILIES in src/data/practice.ts). */
      family: z.enum(PRACTICE_FAMILY_SLUGS),
      part: z.number().int().min(1).max(9),
      /** Suggested time for a topic set; a hard limit (auto-submit) for a mock. */
      timeMinutes: z.number().int().min(5).max(60),
      /**
       * Shared material for several questions: a passage, chart, puzzle set-up or picture.
       * With studySeconds it is a memorize block: shown for that long, then hidden until Submit.
       */
      stems: z
        .array(
          z.object({
            id: z.string(),
            title: z.string(),
            text: z.string().optional(),
            figure: figureSchema.optional(),
            lines: z.array(z.string()).optional(),
            studySeconds: z.number().int().min(10).max(300).optional(),
          }),
        )
        .optional(),
      questions: z.array(question).min(5).max(30),
    })
    .superRefine((d, ctx) => {
      const ids = d.questions.map((q) => q.id);
      const dup = ids.find((id, i) => ids.indexOf(id) !== i);
      if (dup) ctx.addIssue({ code: 'custom', message: `duplicate question id ${dup}` });
      const stems = new Set((d.stems ?? []).map((m) => m.id));
      for (const q of d.questions) {
        if (q.stem && !stems.has(q.stem)) ctx.addIssue({ code: 'custom', message: `${q.id}: unknown stem ${q.stem}` });
      }
    }),
});

export const collections = { blog, practice };
