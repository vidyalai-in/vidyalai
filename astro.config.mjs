// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkReadingTime } from './src/plugins/remark-reading-time.mjs';
import { rehypeDiagrams } from './src/plugins/rehype-diagrams.mjs';
import { rehypeTables } from './src/plugins/rehype-tables.mjs';
import { rehypeHeadingAnchors } from './src/plugins/rehype-heading-anchors.mjs';
import { rehypeGlossary } from './src/plugins/rehype-glossary.mjs';

export default defineConfig({
  site: 'https://vidyal.ai',
  integrations: [sitemap()],
  markdown: {
    // Math is $$...$$ only (inline or on its own lines), so prices like "$2 to $10" stay plain text.
    remarkPlugins: [[remarkMath, { singleDollarTextMath: false }], remarkReadingTime],
    rehypePlugins: [
      [rehypeKatex, { output: 'htmlAndMathml' }],
      rehypeDiagrams,
      rehypeTables,
      rehypeHeadingAnchors,
      rehypeGlossary,
    ],
    shikiConfig: {
      themes: { light: 'github-light-high-contrast', dark: 'github-dark-dimmed' },
      defaultColor: false,
      wrap: false,
    },
  },
});
