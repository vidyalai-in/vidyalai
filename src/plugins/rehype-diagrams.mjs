import { readFileSync } from 'node:fs';
import { visit } from 'unist-util-visit';

const DIAGRAM = /^\/diagrams\/([\w-]+)\.svg$/;

/** Natural size from the rendered SVG's viewBox, so the page reserves space and phones can pan. */
function svgSize(name) {
  try {
    const svg = readFileSync(`public/diagrams/${name}.svg`, 'utf8');
    const [, , w, h] = /viewBox="([^"]+)"/.exec(svg)[1].trim().split(/[\s,]+/).map(Number);
    return { width: Math.round(w), height: Math.round(h) };
  } catch {
    return undefined;
  }
}

/**
 * Turns a paragraph that holds only `![alt](/diagrams/name.svg "caption")` into a figure with
 * light and dark variants. CSS shows the variant that matches the active theme; the hidden one is
 * `display: none`, so screen readers announce the alt text once.
 */
export function rehypeDiagrams() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'p' || !parent || index === undefined) return;
      const kids = node.children.filter((c) => !(c.type === 'text' && !c.value.trim()));
      if (kids.length !== 1 || kids[0].type !== 'element' || kids[0].tagName !== 'img') return;
      const img = kids[0];
      const match = DIAGRAM.exec(String(img.properties.src ?? ''));
      if (!match) return;

      const name = match[1];
      const alt = String(img.properties.alt ?? '');
      const caption = img.properties.title ? String(img.properties.title) : '';
      const size = svgSize(name);
      const variant = (cls, src) => ({
        type: 'element',
        tagName: 'img',
        properties: {
          className: [cls],
          src,
          alt,
          loading: 'lazy',
          decoding: 'async',
          ...(size && { width: size.width, height: size.height, style: `--dw:${size.width}` }),
        },
        children: [],
      });

      const figure = {
        type: 'element',
        tagName: 'figure',
        properties: { className: ['diagram'], ...(size && size.width > 560 && { 'data-wide': '' }) },
        children: [
          {
            type: 'element',
            tagName: 'a',
            properties: {
              className: ['diagram-link'],
              href: `/diagrams/${name}.svg`,
              title: 'Open the diagram full size',
              // DiagramViewer turns this into an in-page zoom/pan viewer; without JS it opens the SVG.
            },
            children: [
              variant('diagram-light', `/diagrams/${name}.svg`),
              variant('diagram-dark', `/diagrams/${name}-dark.svg`),
            ],
          },
        ],
      };
      if (size && size.width > 560) {
        figure.children.push({
          type: 'element',
          tagName: 'p',
          properties: { className: ['diagram-hint'] },
          children: [{ type: 'text', value: 'Swipe to pan the diagram, or tap it to zoom.' }],
        });
      }
      if (caption) {
        figure.children.push({
          type: 'element',
          tagName: 'figcaption',
          properties: {},
          children: [{ type: 'text', value: caption }],
        });
      }
      parent.children[index] = figure;
    });
  };
}
