import { visit } from 'unist-util-visit';
import { toString } from 'hast-util-to-string';
import GithubSlugger from 'github-slugger';

/**
 * Gives every h2-h4 an id (github-slugger, same as Astro's own ids) and appends a "#" permalink to
 * h2/h3. Astro's heading collector keeps ids that already exist, so the TOC links still match.
 */
export function rehypeHeadingAnchors() {
  return (tree) => {
    const slugger = new GithubSlugger();
    visit(tree, 'element', (node) => {
      if (!/^h[2-4]$/.test(node.tagName)) return;
      const text = toString(node);
      node.properties ??= {};
      if (!node.properties.id) node.properties.id = slugger.slug(text);
      if (node.tagName === 'h4') return;
      node.children.push({
        type: 'element',
        tagName: 'a',
        properties: {
          className: ['heading-anchor'],
          href: `#${node.properties.id}`,
          ariaLabel: `Link to section: ${text}`,
        },
        // The "#" is drawn by CSS so it doesn't leak into heading text (TOC, search index).
        children: [],
      });
    });
  };
}
