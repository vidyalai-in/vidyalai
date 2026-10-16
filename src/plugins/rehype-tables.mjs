import { toString } from 'hast-util-to-string';

/**
 * Wraps each table in a focusable, labelled region so wide tables scroll horizontally on small
 * screens and keyboard users can scroll them. The label comes from the nearest preceding heading.
 */
export function rehypeTables() {
  return (tree) => {
    let heading = 'Table';
    const seen = new Map();
    tree.children = tree.children.map((node) => {
      if (node.type !== 'element') return node;
      if (/^h[2-4]$/.test(node.tagName)) heading = toString(node).trim();
      if (node.tagName !== 'table') return node;
      // Landmark labels must be unique: number repeat tables under the same heading.
      const n = (seen.get(heading) ?? 0) + 1;
      seen.set(heading, n);
      return {
        type: 'element',
        tagName: 'div',
        properties: {
          className: ['table-wrap'],
          role: 'region',
          tabIndex: 0,
          ariaLabel: n === 1 ? `${heading} (table)` : `${heading} (table ${n})`,
        },
        children: [node],
      };
    });
  };
}
