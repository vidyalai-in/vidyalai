import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'blog'>;

/** All posts in series order. */
export async function getPosts(): Promise<Post[]> {
  const posts = await getCollection('blog');
  return posts.sort((a, b) => a.data.seriesOrder - b.data.seriesOrder);
}

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
    .sort((a, b) => b.score - a.score || a.p.data.seriesOrder - b.p.data.seriesOrder)
    .slice(0, limit)
    .map((x) => x.p);
}
