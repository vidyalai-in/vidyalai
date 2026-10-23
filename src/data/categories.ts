export const CATEGORIES = [
  {
    slug: 'llm-fundamentals',
    name: 'How LLMs Work',
    blurb: 'Tokens, embeddings, attention and the Transformer: what happens inside a language model.',
    icon: 'book',
  },
  {
    slug: 'llm-math',
    name: 'Math for LLMs',
    blurb: 'Vectors, probability, gradients and floating point: the math under every LLM, built up from high-school algebra.',
    icon: 'sigma',
  },
  {
    slug: 'inference-optimization',
    name: 'Inference & Optimization',
    blurb: 'Prefill and decode, KV caching, batching, quantization and the other tricks behind fast generation.',
    icon: 'bolt',
  },
  {
    slug: 'gpu-kernels',
    name: 'GPUs & Kernels',
    blurb: 'CUDA, Triton, memory hierarchies and tensor cores: how LLM math runs on the GPU, measured kernel by kernel.',
    icon: 'chip',
  },
  {
    slug: 'retrieval-rag',
    name: 'Retrieval & RAG',
    blurb: 'Chunking, hybrid search, reranking and debugging retrieval-augmented generation.',
    icon: 'search',
  },
  {
    slug: 'cost-latency-serving',
    name: 'Cost, Latency & Serving',
    blurb: 'Make LLM systems cheaper and faster: caching, routing, batching and GPU serving.',
    icon: 'gauge',
  },
  {
    slug: 'agents-tool-use',
    name: 'Agents & Tool Use',
    blurb: 'Reliable tool calling, agent loops, long-term memory and structured output.',
    icon: 'agent',
  },
  {
    slug: 'evaluation-reliability',
    name: 'Evaluation & Reliability',
    blurb: 'Eval suites, LLM-as-a-judge and keeping hallucinations in check.',
    icon: 'check',
  },
  {
    slug: 'security-privacy',
    name: 'Security & Privacy',
    blurb: 'Defend agents against prompt injection and keep personal data away from models.',
    icon: 'shield',
  },
  {
    slug: 'fine-tuning',
    name: 'Fine-tuning & Adaptation',
    blurb: 'When to fine-tune instead of prompting or RAG, and how to do it with LoRA.',
    icon: 'tune',
  },
  {
    slug: 'production-engineering',
    name: 'Production Engineering',
    blurb: 'Streaming, tracing, safe rollbacks and running chat for millions of users.',
    icon: 'server',
  },
] as const;

export type CategorySlug = (typeof CATEGORIES)[number]['slug'];
export const CATEGORY_SLUGS = CATEGORIES.map((c) => c.slug) as [CategorySlug, ...CategorySlug[]];

export function getCategory(slug: string) {
  const category = CATEGORIES.find((c) => c.slug === slug);
  if (!category) throw new Error(`Unknown category: ${slug}`);
  return category;
}
