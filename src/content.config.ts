import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { CATEGORY_SLUGS } from './data/categories';
import { DEFAULT_SERIES, SERIES_SLUGS } from './data/series.mjs';

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

export const collections = { blog };
