// Build-time helpers shared by the video figures (video/<slug>/figures/*.mjs): type tiles with
// their words on the upright front face (readable on video, unlike text on a top face), camera
// focus rectangles, and the JSON bundle the stage loads.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as k from './iso-kit.mjs';

const KIT = join(dirname(fileURLToPath(import.meta.url)), 'iso-kit.mjs');

// Width of a word in world units at a font size (Inter semibold averages about 0.6 em per glyph).
export const textW = (text, size) => String(text).length * size * 0.6;

// Text standing on a face whose normal is +y (the front), baseline at height z.
export const frontText = (P, x, y, z, text, { size = 6, tone = 'hi', anchor = 'middle', weight = 600, cls = '' } = {}) =>
  `<text class="fig-text${cls ? ` ${cls}` : ''}" data-tone="${tone}" transform="${k.sideMatrix([x, y], z, P, 'left')}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${esc(text)}</text>`;

export const topText = (P, x, y, z, text, { size = 6, tone = 'hi', anchor = 'middle', weight = 600 } = {}) =>
  `<text class="fig-text" data-tone="${tone}" transform="${k.topMatrix([x, y], z, P, 'x')}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${esc(text)}</text>`;

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// A type tile: a block whose front face carries a word (and optionally a smaller second line).
export function tile(P, { x, y, z = 0, w, d = 14, h = 15, word, sub, lit = false, tone = 'hi', size = 6.2 }) {
  const plan = { x, y, w, d, r: 1.2 };
  const out = [k.solidSvg(k.slabOf(plan, z, h, P, 3, 0.5), { tone, crease: 'lo', lit })];
  out.push(k.lineSvg(k.planOutline(k.insetPlan(plan, 1.4), z + h, P, 3), { tone: 'faint' }));
  if (sub !== undefined) {
    out.push(frontText(P, x + w / 2, y + d, z + h * 0.56, word, { size }));
    out.push(frontText(P, x + w / 2, y + d, z + h * 0.18, sub, { size: size * 0.74, tone: 'mid', weight: 600 }));
  } else out.push(frontText(P, x + w / 2, y + d, z + h / 2 - size * 0.36, word, { size }));
  return out.join('');
}

// The screen rectangle of a world box, as fractions of the viewBox, padded by `pad` (fraction of
// its own size) for camera focus.
export function focusOf(P, width, height, [x0, y0, z0, x1, y1, z1], pad = 0.12) {
  const pts = [];
  for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [z0, z1]) pts.push(k.iso([x, y, z], P));
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  let [l, r, t, b] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const pw = (r - l) * pad;
  const ph = (b - t) * pad;
  [l, r, t, b] = [l - pw, r + pw, t - ph, b + ph];
  const f = (v) => Math.round(v * 1000) / 1000;
  return [f(l / width), f(t / height), f((r - l) / width), f((b - t) / height)];
}

// Extra styles every video figure uses.
export const FIGURE_CSS = `
.fig-text{font-family:var(--fig-font,Inter,system-ui,sans-serif);fill:var(--anatomy-hi);letter-spacing:.01em}
.fig-text[data-tone="mid"]{fill:var(--anatomy-mid)}
.fig-text[data-tone="lo"]{fill:var(--anatomy-lo)}
.fig-text[data-tone="lit"]{fill:var(--anatomy-lit)}
.fig-trace{fill:none;stroke:var(--anatomy-faint);stroke-width:1.1;stroke-linecap:round;stroke-linejoin:round}
.fig-trace[data-on]{stroke:var(--anatomy-lit);stroke-dasharray:3 5}
.fig-glass{fill:var(--anatomy-lit);fill-opacity:.05;stroke:var(--anatomy-mid);stroke-width:var(--anatomy-weight)}
.fig-sector{fill:var(--anatomy-lit);fill-opacity:.2;stroke:none}
[data-station]{transition:none}
[data-station][data-dim]{filter:brightness(.66) saturate(.8)}
`;

// Write <name>.json for the stage (and <name>.html, a looping preview page, with --page).
export function writeFigure(dir, name, { html, css = '', mount, constants, demo }) {
  const src = readFileSync(KIT, 'utf8');
  const kit = src.replace(/^export /gm, '');
  const kitNames = [...src.matchAll(/^export (?:const|function) (\w+)/gm)].map((m) => m[1]);
  const allCss = k.ISO_CSS + FIGURE_CSS + css;
  if (process.argv.includes('--page')) {
    const script = `${kit}
const C = ${JSON.stringify(constants)};
const api = (${mount.toString()})(document.querySelector('.iso-figure'), { ${kitNames.join(', ')} }, C);
const steps = ${JSON.stringify(demo.steps)};
const timing = { steps: steps.map((_, i) => ({ t0: i * ${demo.every ?? 5} })) };
const start = performance.now();
const tick = (now) => { const t = ((now - start) / 1000) % (steps.length * ${demo.every ?? 5}); let s = 0; timing.steps.forEach((st, i) => { if (st.t0 <= t) s = i; }); api.frame(t, s, timing, steps); requestAnimationFrame(tick); };
requestAnimationFrame(tick);`;
    const page = k.pageHtml({ title: name, theme: 'dark', body: html, script, width: 1100 }).replace('</style>', `${FIGURE_CSS}${css}</style>`);
    writeFileSync(join(dir, `${name}.html`), page);
    console.log(`wrote ${name}.html`);
  }
  writeFileSync(join(dir, `${name}.json`), JSON.stringify({ html, css: allCss, kit, kitNames, constants, mount: mount.toString() }));
  console.log(`wrote ${name}.json`);
}
