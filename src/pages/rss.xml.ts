import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { getPosts, postUrl } from '../lib/posts';
import { getCategory } from '../data/categories';

export async function GET(context: APIContext) {
  const posts = await getPosts();
  return rss({
    title: 'VidyalAI',
    description: 'Deep dives on designing production LLM systems.',
    site: context.site!,
    items: posts.map((p) => ({
      title: p.data.title,
      description: p.data.description,
      pubDate: p.data.pubDate,
      link: postUrl(p),
      categories: [getCategory(p.data.category).name, ...p.data.tags],
    })),
  });
}
