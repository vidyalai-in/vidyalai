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
    total: 16,
    filePrefix: 'f',
  },
  {
    slug: 'llm-math',
    name: 'The Math Behind LLMs',
    short: 'Math',
    blurb:
      'The math inside a language model, built up from high-school algebra: vectors, probability, gradients, backpropagation, attention, RoPE and the equations behind training and alignment.',
    total: 16,
    filePrefix: 'm',
  },
  {
    slug: 'llm-from-scratch',
    name: 'Build an LLM From Scratch',
    short: 'From Scratch',
    blurb:
      'Train a small GPT-style model on one laptop GPU, step by step: tokenizer, data pipeline, model, training loop, scaling laws, evaluation, then fine-tuning, DPO and RL.',
    total: 15,
    filePrefix: 't',
  },
  {
    slug: 'llm-inference',
    name: 'LLM Inference & Optimization',
    short: 'Inference',
    blurb:
      'What happens when a model generates text, and the tricks that make it fast: KV caching, batching, PagedAttention, quantization, speculative decoding and more.',
    total: 14,
    filePrefix: 'i',
  },
  {
    slug: 'llm-gpu',
    name: 'LLMs on the GPU: Kernels and Hardware',
    short: 'GPU',
    blurb:
      'How a GPU runs a language model, down to the kernels: CUDA, the memory hierarchy, tensor cores, Triton, attention and matmul kernels, and multi-GPU communication, measured on real hardware.',
    total: 15,
    filePrefix: 'g',
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
  {
    slug: 'agentic-ai',
    name: 'Agentic AI: How AI Agents Work',
    short: 'Agents',
    blurb:
      'How AI agents work, measured on the Claude API: the agent loop, tool calling, planning, context engineering, MCP, multi-agent systems, coding and computer-use agents, evals, reliability and cost.',
    total: 16,
    filePrefix: 'a',
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
