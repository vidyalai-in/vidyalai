/**
 * Builds the practice figures as inline SVG strings from their data (see figureSchema in
 * src/content.config.ts). Lines use currentColor and fills use CSS variables set by
 * src/components/practice/Quiz.astro, so every figure works in light, dark and print.
 */
import type { z } from 'astro/zod';
import type { figureSchema } from '../../content.config';

export type FigureSpec = z.infer<typeof figureSchema>;
type Face = string | number | null;
type Pt = { x: number; y: number };

const SVG_FILES = import.meta.glob('/src/assets/practice/**/*.svg', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const r = (n: number) => Math.round(n * 100) / 100;

const PIPS: Record<number, [number, number][]> = {
  1: [[0.5, 0.5]],
  2: [[0.27, 0.27], [0.73, 0.73]],
  3: [[0.27, 0.27], [0.5, 0.5], [0.73, 0.73]],
  4: [[0.27, 0.27], [0.73, 0.27], [0.27, 0.73], [0.73, 0.73]],
  5: [[0.27, 0.27], [0.73, 0.27], [0.5, 0.5], [0.27, 0.73], [0.73, 0.73]],
  6: [[0.27, 0.25], [0.27, 0.5], [0.27, 0.75], [0.73, 0.25], [0.73, 0.5], [0.73, 0.75]],
};

/** Content of one face, drawn in the unit square (0..1, y down). */
function faceContent(value: Face, pips: boolean): string {
  if (value === null || value === '') return '';
  if (pips && typeof value === 'number' && PIPS[value]) {
    return PIPS[value].map(([x, y]) => `<circle cx="${x}" cy="${y}" r="0.085" class="pf-pip"/>`).join('');
  }
  const t = String(value);
  const size = t.length > 2 ? 0.34 : t.length === 2 ? 0.42 : 0.52;
  return `<text x="0.5" y="0.52" font-size="${size}" text-anchor="middle" dominant-baseline="central" class="pf-text">${esc(t)}</text>`;
}

/** A unit square mapped onto the parallelogram O, O+u, O+v (affine), filled, outlined, with content. */
function face(o: Pt, u: Pt, v: Pt, fill: string, inner: string): string {
  const m = [u.x, u.y, v.x, v.y, o.x, o.y].map(r).join(' ');
  return `<g transform="matrix(${m})"><rect width="1" height="1" class="${fill}" vector-effect="non-scaling-stroke"/>${inner}</g>`;
}

/** Isometric cube whose top-front vertex is at (cx, cy), edge s. Returns its three faces. */
function cube(cx: number, cy: number, s: number, f: { top: string; front: string; right: string }) {
  const k = Math.cos(Math.PI / 6) * s;
  const L = { x: cx - k, y: cy - s / 2 };
  const B = { x: cx, y: cy - s };
  const F = { x: cx, y: cy };
  const top = face(L, { x: B.x - L.x, y: B.y - L.y }, { x: F.x - L.x, y: F.y - L.y }, 'pf-top', f.top);
  const front = face(L, { x: F.x - L.x, y: F.y - L.y }, { x: 0, y: s }, 'pf-front', f.front);
  const right = face(F, { x: k, y: -s / 2 }, { x: 0, y: s }, 'pf-right', f.right);
  return top + front + right;
}

function gridLines(n: number) {
  let out = '';
  for (let i = 1; i < n; i++) {
    const t = r(i / n);
    out += `<line x1="${t}" y1="0" x2="${t}" y2="1" class="pf-grid" vector-effect="non-scaling-stroke"/>`;
    out += `<line x1="0" y1="${t}" x2="1" y2="${t}" class="pf-grid" vector-effect="non-scaling-stroke"/>`;
  }
  return out;
}

const svg = (w: number, h: number, body: string, label: string, scale = 1, cls = '') =>
  `<svg viewBox="0 0 ${r(w)} ${r(h)}" width="${r(w * scale)}" height="${r(h * scale)}" role="img" aria-label="${esc(label)}" class="pf${cls}">${body}</svg>`;

const faceWord = (v: Face, pips: boolean) => (v === null || v === '' ? 'blank' : pips ? `${v} dot${v === 1 ? '' : 's'}` : String(v));

export function figureAlt(f: FigureSpec): string {
  switch (f.kind) {
    case 'dice':
      return f.cubes
        .map((c, i) => `${f.labels?.[i] ?? (f.cubes.length > 1 ? `Cube ${i + 1}` : 'A cube')}: top ${faceWord(c.top, f.pips)}, front ${faceWord(c.front, f.pips)}, right ${faceWord(c.right, f.pips)}`)
        .join('. ');
    case 'net':
      return `A cube net, row by row: ${f.rows.map((row) => row.map((c) => (c === null ? '·' : faceWord(c, f.pips))).join(' ')).join(' / ')}`;
    case 'stack':
      return `A pile of unit cubes; column heights from back row to front row: ${f.heights.map((row) => row.join(' ')).join(' / ')}`;
    case 'cubegrid':
      return `A large cube cut into ${f.n} × ${f.n} × ${f.n} small cubes`;
    case 'svg':
      return f.alt;
    case 'table':
      return f.caption ?? 'Table';
    case 'grid':
      return `Grid, row by row: ${f.cells.map((row) => row.map((c) => (c === null || c === '' ? '·' : String(c))).join(' ')).join(' / ')}`;
    case 'bars':
      return `${f.title ?? 'Bar chart'}: ${f.labels.map((l, i) => `${l} ${f.values[i]}${f.unit ? ' ' + f.unit : ''}`).join(', ')}`;
    case 'pie':
      return `${f.title ?? 'Pie chart'}: ${f.labels.map((l, i) => `${l} ${f.values[i]}%`).join(', ')}`;
    case 'venn':
      return `Venn diagram with ${f.circles.length} circles${f.circles.some((c) => c.label) ? ': ' + f.circles.map((c) => c.label ?? '').join(', ') : ''}`;
    case 'shapes':
      return f.alt ?? 'Figure';
    case 'glyphs':
      return 'Text figure';
  }
}

const PALETTE = ['#2f6fdc', '#e8772e', '#26a35a', '#d64545', '#8a4fd1', '#c9a227', '#1aa3b8', '#9a6b3f'];

/** SVG transform for mirroring/rotating a w × h box in place; returns [transform, newW, newH]. */
function boxTransform(t: string, w: number, h: number): [string, number, number] {
  switch (t) {
    case 'flipH':
      return [`matrix(-1 0 0 1 ${w} 0)`, w, h];
    case 'flipV':
      return [`matrix(1 0 0 -1 0 ${h})`, w, h];
    case 'rot90':
      return [`matrix(0 1 -1 0 ${h} 0)`, h, w];
    case 'rot180':
      return [`matrix(-1 0 0 -1 ${w} ${h})`, w, h];
    case 'rot270':
      return [`matrix(0 -1 1 0 0 ${w})`, h, w];
    default:
      return ['', w, h];
  }
}

const FILL: Record<string, string> = { none: 'none', solid: 'currentColor', grey: 'var(--pf-shade)', accent: 'var(--accent)' };

type ShapeItem = Extract<FigureSpec, { kind: 'shapes' }>['items'][number];
function shapeItem(it: ShapeItem): string {
  const sw = 'stroke="currentColor" stroke-linejoin="round" stroke-linecap="round"';
  switch (it.t) {
    case 'line':
      return `<line x1="${it.p[0]}" y1="${it.p[1]}" x2="${it.p[2]}" y2="${it.p[3]}" ${sw} stroke-width="${it.w ?? 2.5}"${it.dash ? ' stroke-dasharray="5 4"' : ''}/>`;
    case 'arrow': {
      const [x1, y1, x2, y2] = it.p;
      const a = Math.atan2(y2 - y1, x2 - x1);
      const L = 9;
      const h1 = [x2 - L * Math.cos(a - 0.45), y2 - L * Math.sin(a - 0.45)];
      const h2 = [x2 - L * Math.cos(a + 0.45), y2 - L * Math.sin(a + 0.45)];
      return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${sw} stroke-width="${it.w ?? 2.5}"/><polygon points="${x2},${y2} ${r(h1[0])},${r(h1[1])} ${r(h2[0])},${r(h2[1])}" fill="currentColor"/>`;
    }
    case 'poly': {
      const pts = it.p.map(([x, y]) => `${x},${y}`).join(' ');
      return it.open ? `<polyline points="${pts}" fill="none" ${sw} stroke-width="2.5"/>` : `<polygon points="${pts}" fill="${FILL[it.fill]}" ${sw} stroke-width="2.5"/>`;
    }
    case 'circle':
      return `<circle cx="${it.c[0]}" cy="${it.c[1]}" r="${it.r}" fill="${FILL[it.fill]}" ${sw} stroke-width="2.5"/>`;
    case 'rect':
      return `<rect x="${it.p[0]}" y="${it.p[1]}" width="${it.p[2]}" height="${it.p[3]}" fill="${FILL[it.fill]}" ${sw} stroke-width="2.5"/>`;
    case 'text':
      return `<text x="${it.at[0]}" y="${it.at[1]}" font-size="${it.size ?? 22}" text-anchor="middle" dominant-baseline="central" class="pf-text">${esc(it.s)}</text>`;
  }
}

/** Nice round step for a chart axis. */
function niceStep(max: number) {
  const raw = max / 5;
  const pow = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw)!;
}

export function renderFigure(f: FigureSpec): string {
  const alt = figureAlt(f);
  switch (f.kind) {
    case 'dice': {
      const s = 58;
      const k = Math.cos(Math.PI / 6) * s;
      const w = 2 * k;
      const gap = 34;
      const labelH = f.labels ? 26 : 0;
      const pad = 4;
      let body = '';
      f.cubes.forEach((c, i) => {
        const cx = pad + k + i * (w + gap);
        const cy = pad + s;
        body += cube(cx, cy, s, {
          top: faceContent(c.top, f.pips),
          front: faceContent(c.front, f.pips),
          right: faceContent(c.right, f.pips),
        });
        if (f.labels?.[i]) body += `<text x="${r(cx)}" y="${r(cy + s + 20)}" text-anchor="middle" class="pf-label">${esc(f.labels[i])}</text>`;
      });
      const width = pad * 2 + f.cubes.length * w + (f.cubes.length - 1) * gap;
      return svg(width, pad * 2 + 2 * s + labelH, body, alt);
    }
    case 'net': {
      const c = 46;
      const cols = Math.max(...f.rows.map((row) => row.length));
      let body = '';
      f.rows.forEach((row, y) =>
        row.forEach((v, x) => {
          if (v === null) return;
          body += face({ x: 2 + x * c, y: 2 + y * c }, { x: c, y: 0 }, { x: 0, y: c }, 'pf-front', faceContent(v, f.pips));
        }),
      );
      return svg(cols * c + 4, f.rows.length * c + 4, body, alt);
    }
    case 'stack': {
      // World: x to the right, y away from the viewer, z up; viewer looks from (1, -1, 1).
      const s = 30;
      const k = Math.cos(Math.PI / 6) * s;
      const rows = f.heights.length;
      const cubes: [number, number, number][] = [];
      f.heights.forEach((row, r) => row.forEach((h, x) => { for (let z = 0; z < h; z++) cubes.push([x, rows - 1 - r, z]); }));
      cubes.sort((a, b) => a[0] - a[1] + a[2] - (b[0] - b[1] + b[2]) || a[2] - b[2]);
      const P = (x: number, y: number, z: number) => ({ x: (x + y) * k, y: ((x - y) * s) / 2 - z * s });
      const pts = cubes.flatMap(([x, y, z]) => [P(x, y, z), P(x + 1, y + 1, z + 1), P(x, y + 1, z + 1), P(x + 1, y, z)]);
      const minX = Math.min(...pts.map((p) => p.x)), maxX = Math.max(...pts.map((p) => p.x));
      const minY = Math.min(...pts.map((p) => p.y)), maxY = Math.max(...pts.map((p) => p.y));
      const body = cubes
        .map(([x, y, z]) => {
          const F = P(x + 1, y, z + 1);
          return cube(F.x - minX + 4, F.y - minY + 4, s, { top: '', front: '', right: '' });
        })
        .join('');
      return svg(maxX - minX + 8, maxY - minY + 8, body, alt);
    }
    case 'cubegrid': {
      const s = 96;
      const k = Math.cos(Math.PI / 6) * s;
      const g = gridLines(f.n);
      return svg(2 * k + 8, 2 * s + 8, cube(4 + k, 4 + s, s, { top: g, front: g, right: g }), alt);
    }
    case 'svg': {
      const raw = SVG_FILES[`/src/assets/practice/${f.src}.svg`];
      if (!raw) throw new Error(`Practice figure not found: src/assets/practice/${f.src}.svg`);
      return raw
        .replace(/<\?xml[^>]*>\s*/, '')
        .replace(/<svg\b/, `<svg role="img" aria-label="${esc(alt)}" class="pf pf-file"`);
    }
    case 'table': {
      const head = f.header ? `<thead><tr>${f.header.map((h) => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead>` : '';
      const body = f.rows.map((row) => `<tr>${row.map((c, i) => (i === 0 && f.header ? `<th scope="row">${esc(String(c))}</th>` : `<td>${esc(String(c))}</td>`)).join('')}</tr>`).join('');
      return `<table class="pf-table">${f.caption ? `<caption>${esc(f.caption)}</caption>` : ''}${head}<tbody>${body}</tbody></table>`;
    }
    case 'grid': {
      const c = f.size ?? 42;
      const cols = Math.max(...f.cells.map((row) => row.length));
      const shade = new Set((f.shade ?? []).map(([a, b]) => `${a},${b}`));
      let body = '';
      f.cells.forEach((row, y) =>
        row.forEach((v, x) => {
          if (v === null) return;
          const X = 2 + x * c, Y = 2 + y * c;
          body += `<rect x="${X}" y="${Y}" width="${c}" height="${c}" class="${shade.has(`${y},${x}`) ? 'pf-right' : 'pf-front'} pf-cell"/>`;
          if (v !== '') {
            const t = String(v);
            body += `<text x="${X + c / 2}" y="${Y + c / 2 + 1}" font-size="${r(c * (t.length > 2 ? 0.36 : 0.48))}" text-anchor="middle" dominant-baseline="central" class="pf-text">${esc(t)}</text>`;
          }
        }),
      );
      return svg(cols * c + 4, f.cells.length * c + 4, body, alt);
    }
    case 'bars': {
      const n = f.labels.length;
      const bw = 34, gap = 22, left = 44, top = (f.title ? 34 : 14) + (f.unit ? 16 : 0), ph = 170;
      const max = Math.max(...f.values);
      const step = niceStep(max);
      const top_ = Math.ceil(max / step) * step;
      const W = left + n * (bw + gap) + 10, H = top + ph + 40;
      let body = f.title ? `<text x="${W / 2}" y="18" text-anchor="middle" class="pf-label">${esc(f.title)}</text>` : '';
      for (let v = 0; v <= top_ + 1e-9; v += step) {
        const y = top + ph - (v / top_) * ph;
        body += `<line x1="${left}" y1="${r(y)}" x2="${W - 6}" y2="${r(y)}" class="pf-gridline"/><text x="${left - 6}" y="${r(y)}" text-anchor="end" dominant-baseline="central" class="pf-small">${r(v)}</text>`;
      }
      f.values.forEach((v, i) => {
        const x = left + gap / 2 + i * (bw + gap);
        const h = (v / top_) * ph;
        body += `<rect x="${x}" y="${r(top + ph - h)}" width="${bw}" height="${r(h)}" fill="${PALETTE[0]}"/>`;
        body += `<text x="${x + bw / 2}" y="${r(top + ph - h - 7)}" text-anchor="middle" class="pf-small pf-strong">${v}</text>`;
        body += `<text x="${x + bw / 2}" y="${top + ph + 17}" text-anchor="middle" class="pf-small">${esc(f.labels[i])}</text>`;
      });
      body += `<line x1="${left}" y1="${top + ph}" x2="${W - 6}" y2="${top + ph}" stroke="currentColor" stroke-width="1.5"/>`;
      if (f.unit) body += `<text x="${left - 36}" y="${top - 14}" class="pf-small">${esc(f.unit)}</text>`;
      return svg(W, H, body, alt);
    }
    case 'pie': {
      const R = 82, cx = 92, cy = (f.title ? 30 : 8) + R;
      const legendX = cx + R + 22;
      const H = cy + R + 8, W = legendX + 150;
      let body = f.title ? `<text x="${W / 2}" y="18" text-anchor="middle" class="pf-label">${esc(f.title)}</text>` : '';
      let a0 = -Math.PI / 2;
      f.values.forEach((v, i) => {
        const a1 = a0 + (v / 100) * 2 * Math.PI;
        const large = a1 - a0 > Math.PI ? 1 : 0;
        const p0 = [cx + R * Math.cos(a0), cy + R * Math.sin(a0)];
        const p1 = [cx + R * Math.cos(a1), cy + R * Math.sin(a1)];
        body += `<path d="M${cx},${cy} L${r(p0[0])},${r(p0[1])} A${R},${R} 0 ${large} 1 ${r(p1[0])},${r(p1[1])} Z" fill="${PALETTE[i % PALETTE.length]}" stroke="var(--surface)" stroke-width="1.5"/>`;
        const ly = cy - R + 12 + i * 24;
        body += `<rect x="${legendX}" y="${ly - 8}" width="14" height="14" rx="2" fill="${PALETTE[i % PALETTE.length]}"/><text x="${legendX + 20}" y="${ly}" dominant-baseline="central" class="pf-small">${esc(f.labels[i])}: ${v}%</text>`;
        a0 = a1;
      });
      return svg(W, Math.max(H, cy - R + 12 + f.values.length * 24), body, alt);
    }
    case 'venn': {
      const W = f.w ?? 200, H = f.h ?? 140;
      const padTop = f.circles.some((c) => c.label) ? 18 : 2;
      let body = f.circles.map((c) => `<circle cx="${c.c[0]}" cy="${c.c[1]}" r="${c.r}" fill="var(--accent)" fill-opacity="0.07" stroke="currentColor" stroke-width="2"/>`).join('');
      body += f.circles.filter((c) => c.label).map((c) => `<text x="${c.c[0]}" y="${c.c[1] - c.r - 6}" text-anchor="middle" class="pf-small pf-strong">${esc(c.label!)}</text>`).join('');
      body += (f.marks ?? []).map((m) => `<text x="${m.at[0]}" y="${m.at[1]}" text-anchor="middle" dominant-baseline="central" class="pf-small pf-strong">${esc(m.s)}</text>`).join('');
      return svg(W, H + padTop, `<g transform="translate(0 ${padTop})">${body}</g>`, alt);
    }
    case 'shapes': {
      const w = f.w ?? 100, h = f.h ?? 100;
      const [tf, W, H] = boxTransform(f.transform, w, h);
      let inner = f.items.map(shapeItem).join('');
      if (f.frame) inner = `<rect x="1" y="1" width="${w - 2}" height="${h - 2}" fill="none" stroke="currentColor" stroke-width="1.5"/>` + inner;
      const pad = 4;
      const body = `<g transform="translate(${pad} ${pad})"><g${tf ? ` transform="${tf}"` : ''}>${inner}</g></g>`;
      return svg(W + pad * 2, H + pad * 2, body, alt, 1.3, ' pf-shapes');
    }
    case 'glyphs': {
      const size = f.size ?? 34;
      const w = Math.max(size, size * 0.62 * [...f.text].length) + 10, h = size * 1.3;
      const [tf, W, H] = boxTransform(f.transform, w, h);
      const text = `<text x="${w / 2}" y="${h / 2 + 1}" font-size="${size}" text-anchor="middle" dominant-baseline="central" class="pf-text pf-mono">${esc(f.text)}</text>`;
      return svg(W, H, `<g${tf ? ` transform="${tf}"` : ''}>${text}</g>`, alt);
    }
  }
}
