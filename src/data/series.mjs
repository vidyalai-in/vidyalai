/**
 * Article series, in the order the site lists them. Plain JS so the video tools (video/tools/*.mjs)
 * can import it too.
 *
 * - `short`: label on post cards ("Fundamentals · Part 1").
 * - `total`: planned number of parts; series pages say "N of total published" until they're all out.
 * - `playlist`: the series' YouTube playlist, once it exists. The uploader finds it by `name`.
 * - `filePrefix`: post and diagram file prefix (`f01-how-llms-work.md`, `diagrams/f01-*.mmd`).
 */
export const SERIES = [
  {
    slug: 'llm-fundamentals',
    name: 'LLM Fundamentals',
    short: 'Fundamentals',
    blurb:
      'How large language models work, from tokens and embeddings to attention, the Transformer and training, built around “Attention Is All You Need”.',
    total: 10,
    filePrefix: 'f',
  },
  {
    slug: 'llm-inference',
    name: 'LLM Inference & Optimization',
    short: 'Inference',
    blurb:
      'What happens when a model generates text, and the tricks that make it fast: KV caching, batching, PagedAttention, quantization, speculative decoding and more.',
    total: 10,
    filePrefix: 'i',
  },
  {
    slug: 'llm-system-design',
    name: 'LLM System Design',
    short: 'System Design',
    blurb:
      'Twenty-five deep dives on building production LLM systems, from retrieval and evaluation to serving, security and operations.',
    total: 25,
    playlist: 'https://www.youtube.com/playlist?list=PLTy3a2upFmcM',
    filePrefix: 'q',
  },
];

/** Posts without a `series` field belong to this one. */
export const DEFAULT_SERIES = 'llm-system-design';

export const SERIES_SLUGS = /** @type {[string, ...string[]]} */ (SERIES.map((s) => s.slug));

export const CHANNEL = {
  url: 'https://www.youtube.com/@vidyalai',
  /** Opens YouTube's subscribe prompt for the channel. */
  subscribe: 'https://www.youtube.com/@vidyalai?sub_confirmation=1',
};

export function getSeries(slug = DEFAULT_SERIES) {
  const series = SERIES.find((s) => s.slug === slug);
  if (!series) throw new Error(`Unknown series: ${slug}`);
  return series;
}

export const seriesUrl = (slug) => `/series/${slug}/`;
