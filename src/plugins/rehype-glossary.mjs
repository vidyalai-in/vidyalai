import { readFileSync, readdirSync } from 'node:fs';
import { GLOSSARY } from '../data/glossary.mjs';

/** Skip text inside these: code, links, headings, diagrams, math and anything already marked. */
const SKIP = new Set(['code', 'pre', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'figure', 'script', 'style', 'svg', 'button', 'kbd', 'sup', 'math']);
/** Diagrams and KaTeX output (`katex`, `katex-display`). */
const skipClass = (node) => [].concat(node.properties?.className ?? []).some((c) => c === 'diagram' || String(c).startsWith('katex'));
/** Sections whose text is mostly citations. */
const SKIP_SECTIONS = new Set(['further reading']);

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
// Word boundaries that also treat - and @ as part of a word, so "LLM-as-a-Judge" doesn't match "LLM".
const bounded = (source, flags) => new RegExp(`(?<![\\w@-])(?:${source})(?![\\w@-])`, flags);

const MATCHERS = GLOSSARY.map((e) => {
  if (!e.source?.url || !e.source?.title) throw new Error(`Glossary entry "${e.id}" needs a source`);
  const words = e.match ?? [];
  const sensitive = words.filter((w) => /[A-Z]/.test(w)).map(escape);
  const insensitive = words.filter((w) => !/[A-Z]/.test(w)).map(escape);
  const res = [];
  if (sensitive.length) res.push(bounded(sensitive.join('|'), 'g'));
  if (insensitive.length) res.push(bounded(insensitive.join('|'), 'gi'));
  if (e.pattern) res.push(bounded(e.pattern, /[A-Z]/.test(e.pattern.replace(/\\./g, '')) ? 'g' : 'gi'));
  return { entry: e, res };
});

/** slug -> title, read once from the articles' frontmatter, for the tooltip's "Read more" link. */
const TITLES = Object.fromEntries(
  readdirSync('src/content/blog')
    .filter((f) => f.endsWith('.md'))
    .map((f) => {
      const src = readFileSync(`src/content/blog/${f}`, 'utf8');
      const field = (k) => new RegExp(`^${k}:\\s*"?(.*?)"?\\s*$`, 'm').exec(src)?.[1];
      return [field('slug'), field('title')];
    }),
);

const textOf = (node) =>
  node.type === 'text' ? node.value : (node.children ?? []).map(textOf).join('');

/**
 * Earliest match among all terms; the longest wins a tie. A term already used in the section still
 * claims its text (the caller leaves it unmarked), so "canary" can't match inside a second "canary token".
 */
function firstMatch(text) {
  let best;
  for (const { entry, res } of MATCHERS) {
    for (const re of res) {
      re.lastIndex = 0;
      const m = re.exec(text);
      if (m && (!best || m.index < best.index || (m.index === best.index && m[0].length > best.text.length))) {
        best = { index: m.index, text: m[0], entry };
      }
    }
  }
  return best;
}

/**
 * Marks the first use of each glossary term in every H2 section with
 * `<a class="term" href="/glossary/#id" aria-describedby="term-id">`. The definitions go in one hidden
 * block at the end of the article (screen readers read them as the link's description;
 * GlossaryTips shows them as a tooltip). Without JS, the link opens the glossary page.
 */
export function rehypeGlossary() {
  return (tree, file) => {
    const self = file?.data?.astro?.frontmatter?.slug;
    const usedOnPage = new Map();
    let used = new Set();
    let skipSection = false;

    const walk = (node) => {
      if (!node.children) return;
      const out = [];
      for (const child of node.children) {
        if (child.type === 'element') {
          if (!SKIP.has(child.tagName) && !skipClass(child)) walk(child);
          out.push(child);
          continue;
        }
        if (child.type !== 'text') {
          out.push(child);
          continue;
        }
        let rest = child.value;
        for (let m = firstMatch(rest); m; m = firstMatch(rest)) {
          if (used.has(m.entry.id)) {
            out.push({ type: 'text', value: rest.slice(0, m.index + m.text.length) });
            rest = rest.slice(m.index + m.text.length);
            continue;
          }
          if (m.index) out.push({ type: 'text', value: rest.slice(0, m.index) });
          out.push({
            type: 'element',
            tagName: 'a',
            properties: { className: ['term'], href: `/glossary/#${m.entry.id}`, dataTerm: m.entry.id, ariaDescribedBy: `term-${m.entry.id}` },
            children: [{ type: 'text', value: m.text }],
          });
          used.add(m.entry.id);
          usedOnPage.set(m.entry.id, m.entry);
          rest = rest.slice(m.index + m.text.length);
        }
        if (rest) out.push({ type: 'text', value: rest });
      }
      node.children = out;
    };

    for (const node of tree.children) {
      if (node.type === 'element' && node.tagName === 'h2') {
        used = new Set();
        skipSection = SKIP_SECTIONS.has(textOf(node).trim().toLowerCase());
        continue;
      }
      if (!skipSection && node.type === 'element' && !SKIP.has(node.tagName)) walk({ children: [node] });
    }

    if (!usedOnPage.size) return;
    tree.children.push({
      type: 'element',
      tagName: 'div',
      properties: { className: ['term-defs'], hidden: true, dataPagefindIgnore: 'all' },
      children: [...usedOnPage.values()].map((e) => {
        const article = e.article && e.article !== self && TITLES[e.article];
        return {
          type: 'element',
          tagName: 'span',
          properties: {
            id: `term-${e.id}`,
            dataName: e.term,
            ...(e.expansion && { dataExpansion: e.expansion }),
            dataSource: e.source.url,
            dataSourceTitle: e.source.title,
            ...(article && { dataArticle: `/blog/${e.article}/`, dataArticleTitle: article }),
          },
          children: [{ type: 'text', value: `${e.expansion ? `${e.expansion}. ` : ''}${e.def}` }],
        };
      }),
    });
  };
}
