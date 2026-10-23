// Copies KaTeX's stylesheet and WOFF2 fonts into public/katex/, so math pages can load them
// without a CDN. Run after upgrading the katex package: node scripts/vendor-katex.mjs
import { copyFileSync, mkdirSync, readdirSync, rmSync } from 'node:fs';

const from = 'node_modules/katex/dist';
const to = 'public/katex';
rmSync(to, { recursive: true, force: true });
mkdirSync(`${to}/fonts`, { recursive: true });
copyFileSync(`${from}/katex.min.css`, `${to}/katex.min.css`);
// Every browser that runs the site supports WOFF2; the stylesheet lists it first.
const fonts = readdirSync(`${from}/fonts`).filter((f) => f.endsWith('.woff2'));
for (const f of fonts) copyFileSync(`${from}/fonts/${f}`, `${to}/fonts/${f}`);
console.log(`katex.min.css and ${fonts.length} fonts -> ${to}/`);
