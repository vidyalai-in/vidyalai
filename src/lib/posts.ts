import { getCollection, type CollectionEntry } from 'astro:content';
import { SERIES, getSeries } from '../data/series.mjs';

export type Post = CollectionEntry<'blog'>;

const seriesIndex = (p: Post) => SERIES.findIndex((s) => s.slug === p.data.series);

/** Series in site order, then part number. */
export const byReadingOrder = (a: Post, b: Post) => seriesIndex(a) - seriesIndex(b) || a.data.seriesOrder - b.data.seriesOrder;

/** All posts in reading order. */
export async function getPosts(): Promise<Post[]> {
  const posts = await getCollection('blog');
  return posts.sort(byReadingOrder);
}

/** One series' posts, in part order. */
export async function seriesPosts(slug: string): Promise<Post[]> {
  return (await getPosts()).filter((p) => p.data.series === slug);
}

export const seriesOf = (post: Post) => getSeries(post.data.series);

export const postUrl = (post: Post) => `/blog/${post.id}/`;
export const categoryUrl = (slug: string) => `/category/${slug}/`;
export const tagUrl = (tag: string) => `/tags/${tag}/`;

export function formatDate(date: Date) {
  return date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
}

export function tagLabel(tag: string) {
  const special: Record<string, string> = {
    rag: 'RAG', gpu: 'GPU', lora: 'LoRA', 'llm-as-judge': 'LLM-as-judge', evals: 'Evals',
  };
  return special[tag] ?? tag.replace(/-/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

/** Posts sharing the category first, then most shared tags. */
export function relatedPosts(post: Post, all: Post[], limit = 3) {
  return all
    .filter((p) => p.id !== post.id)
    .map((p) => ({
      p,
      score:
        (p.data.category === post.data.category ? 3 : 0) +
        p.data.tags.filter((t) => post.data.tags.includes(t)).length,
    }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || byReadingOrder(a.p, b.p))
    .slice(0, limit)
    .map((x) => x.p);
}
