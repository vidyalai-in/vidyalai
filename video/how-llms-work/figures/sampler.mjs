// Fig: the token sampler. A weighted dial whose five sectors are the softmax probabilities of the
// five candidate tokens; a needle that either swings to the biggest sector (greedy) or is spun and
// lands wherever it lands (sampling); a temperature knob that reshapes the sectors; and a rack of
// the five candidate tiles in front, each showing its live probability.
//
//   node video/how-llms-work/figures/sampler.mjs [--page]
//
// Step fields: greedy: true | spin: <roll> or [<roll>, ...] | temp: <T> | light: <token>|knob
//              focus: DIAL|KNOB|RACK|ALL
// Numbers from the article: logits 6.0, 3.5, 3.0, 2.5, 1.5 for Paris, a, the, known, Lyon.

import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as k from '../../tools/iso/iso-kit.mjs';
import { frontText, topText, focusOf, writeFigure } from '../../tools/iso/video-parts.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const WIDTH = 640;
const HEIGHT = 330;

const TOKENS = ['Paris', 'a', 'the', 'known', 'Lyon'];
const LOGITS = [6.0, 3.5, 3.0, 2.5, 1.5];

const PLATE = { x: -158, y: -96, w: 316, d: 176, r: 10 };
const PLATE_H = 12;
const FOOT_H = 5;

const DIAL = [-34, -22];
const DIAL_R = 58;
const DIAL_Z = 4;
const DIAL_H = 10;
const DIAL_TOP = DIAL_Z + DIAL_H;
const HUB_R = 6;
const HUB_TOP = DIAL_TOP + 3;

const KNOB = [104, -42];
const KNOB_R = 11;
const KNOB_Z = 2.4;
const KNOB_TOP = KNOB_Z + 8;
const KNOB_DEG = 60; // angle = 60° × log2(T): T 0.5, 1, 2 at -60°, 0, +60°

const RACK = { x: -136, y: 46, w: 272, d: 26, r: 2 };
const RACK_H = 3;
const TILE = { w: 44, d: 16, h: 20, gap: 7 };
const TILE_X = (i) => -((TILE.w * 5 + TILE.gap * 4) / 2) + i * (TILE.w + TILE.gap);
const TILE_Y = RACK.y + 4;

const P = k.fitProjection([...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [DIAL[0], DIAL[1] - DIAL_R, HUB_TOP + 4]], WIDTH, HEIGHT, { pad: 16, azimuth: 60 });

const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);
const slab = (plan, z, h, steps = 3, bevel = 0.4) => k.slabOf(plan, z, h, P, steps, bevel);

function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  const feet = k.corners(PLATE, 26).sort((a, b) => k.depthOf([...a, 0], P) - k.depthOf([...b, 0], P));
  for (const [x, y] of feet) {
    out.push(S(k.cylinder(x, y, 8, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: 'lo' }));
    out.push(S(k.cylinder(x, y, 6.2, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: 'mid' }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: 'mid' }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: 'lo' }));
  const screws = k.corners(PLATE, 11);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.4, 0, P, 16)).join(''), { tone: 'lo' }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  const label = { x: 72, y: -84, w: 54, d: 18, r: 1.5 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 4), { tone: 'lo', crease: 'none' }));
  out.push(L(k.planOutline(k.insetPlan(label, 1.6), 0.8, P, 4), { tone: 'faint' }));
  [[34, 5.5], [26, 9.5], [30, 13.5]].forEach(([len, y]) => out.push(L(k.lineOnTop([label.x + 6, label.y + y], [label.x + 6 + len, label.y + y], 0.8, P), { tone: 'lo', free: true })));
  return out.join('');
}

function dial() {
  const [x, y] = DIAL;
  const out = [`<path class="iso-halo" d="${k.haloOf({ x: x - DIAL_R, y: y - DIAL_R, w: DIAL_R * 2, d: DIAL_R * 2, r: DIAL_R }, 0, 0.1, P)}" style="filter:blur(5px)"/>`];
  out.push(S(k.cylinder(x, y, DIAL_R - 7, 0, DIAL_Z, P, 72), { tone: 'mid' }));
  out.push(L(k.sideArc(x, y, DIAL_R - 7, 2, P, 40), { tone: 'faint' }));
  out.push(S(k.cylinder(x, y, DIAL_R, DIAL_Z, DIAL_H, P, 110, 0.8), { tone: 'hi', crease: 'lo' }));
  out.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 3.6, DIAL_TOP, 120, P, 0.3), { tone: 'lo' }));
  out.push(L(k.sideArc(x, y, DIAL_R, DIAL_TOP - 3.6, P, 60), { tone: 'faint' }));
  out.push(L(k.ring(x, y, DIAL_R - 2.2, DIAL_TOP, P, 110), { tone: 'lo' }));
  const t = k.radialTicks(x, y, DIAL_R - 3, 100, 10, DIAL_TOP, [1.5, 3.2], P);
  out.push(L(t.minor, { tone: 'lo' }), L(t.major, { tone: 'mid' }));
  out.push(L(k.ring(x, y, DIAL_R - 7.5, DIAL_TOP, P, 110), { tone: 'faint' }));
  out.push(L(k.ring(x, y, HUB_R + 3, DIAL_TOP, P, 32), { tone: 'faint' }));
  out.push('<g data-part="sectors"></g>');
  out.push(S(k.cylinder(x, y, HUB_R, DIAL_TOP, HUB_TOP - DIAL_TOP, P, 32), { tone: 'mid' }));
  out.push('<path data-part="needle" class="iso-line" data-tone="lit" style="stroke-width:1.6"/>');
  out.push(S(k.cylinder(x, y, 2.2, HUB_TOP, 1.2, P, 20), { tone: 'hi', lit: true }));
  return out.join('');
}

function knob() {
  const [x, y] = KNOB;
  const out = [S(k.cylinder(x, y, 36, 0, 0.8, P, 64), { tone: 'lo', crease: 'none' })];
  const span = (KNOB_DEG * 2 * Math.PI) / 180;
  const t = k.radialTicks(x, y, 24, 13, 6, 0.8, [2, 4], P, -span / 2 - span / 12, span + span / 6);
  out.push(L(t.minor, { tone: 'lo' }), L(t.major, { tone: 'mid' }));
  [[0.5, -1], [1, 0], [2, 1]].forEach(([T, s]) => {
    const a = (s * KNOB_DEG * Math.PI) / 180;
    out.push(topText(P, x + Math.sin(a) * 30, y - Math.cos(a) * 30 + 2, 0.8, String(T), { size: 6.4, tone: 'hi' }));
  });
  out.push(topText(P, x, y + 29, 0.8, 'TEMPERATURE', { size: 4, tone: 'mid', weight: 600 }));
  out.push(S(k.cylinder(x, y, KNOB_R + 2, 0.8, KNOB_Z - 0.8, P, 40), { tone: 'mid' }));
  out.push(`<g data-part="knob-body">${S(k.cylinder(x, y, KNOB_R, KNOB_Z, KNOB_TOP - KNOB_Z, P, 48, 0.6), { tone: 'mid' })}</g>`);
  out.push(L(k.knurl(x, y, KNOB_R, KNOB_Z + 0.8, KNOB_TOP - 0.8, 48, P), { tone: 'lo' }));
  out.push(L(k.ring(x, y, KNOB_R - 2, KNOB_TOP, P, 40), { tone: 'faint' }));
  out.push('<path data-part="knob-pointer" class="iso-line" data-tone="hi" style="stroke-width:1.4"/>');
  return out.join('');
}

function rack() {
  const out = [S(slab(RACK, 0, RACK_H), { tone: 'mid' })];
  out.push(S(slab({ x: RACK.x, y: RACK.y - 4, w: RACK.w, d: 4, r: 1 }, 0, 10), { tone: 'mid' }));
  out.push(D([RACK.x + 10, RACK.x + RACK.w / 2, RACK.x + RACK.w - 10].map((x) => at([x, RACK.y - 2, 10])), { size: 0.45 }));
  TOKENS.forEach((tok, i) => {
    const x = TILE_X(i);
    const plan = { x, y: TILE_Y, w: TILE.w, d: TILE.d, r: 1.4 };
    const face = TILE_Y + TILE.d;
    out.push(
      `<g data-tile="${i}">${S(slab(plan, RACK_H, TILE.h, 3, 0.5), { tone: 'hi', crease: 'lo' })}</g>` +
        L(k.planOutline(k.insetPlan(plan, 1.6), RACK_H + TILE.h, P, 3), { tone: 'faint' }) +
        frontText(P, x + TILE.w / 2, face, RACK_H + 11, tok, { size: 8 }) +
        frontText(P, x + TILE.w / 2, face, RACK_H + 3.4, '', { size: 6, tone: 'mid', weight: 600 }).replace('<text', `<text data-part="pct-${i}"`),
    );
  });
  return out.join('');
}

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    'A token sampler. A dial divided into five sectors, one per candidate token, sized by its probability; a needle that swings to the biggest sector or is spun to land at random; a temperature knob that reshapes the sectors; and a rack of the five candidate tiles, each showing its probability.',
  body: [plate(), knob(), dial(), rack()],
});

const html = k.plateHtml({ fig: 'Fig 5', title: 'Token sampler', hint: 'Pick one token', readout: '', body: `<div class="iso-stage">${svg}</div>` });

const F = (box, pad) => focusOf(P, WIDTH, HEIGHT, box, pad);
const focus = {
  DIAL: F([DIAL[0] - DIAL_R - 30, DIAL[1] - DIAL_R, 0, DIAL[0] + DIAL_R + 30, RACK.y + RACK.d, HUB_TOP], 0.04),
  KNOB: F([DIAL[0] - 10, -80, 0, KNOB[0] + 40, 0, 20], 0.06),
  RACK: F([RACK.x, -20, 0, RACK.x + RACK.w, RACK.y + RACK.d, 24], 0.04),
  ALL: [0, 0, 1, 1],
};

function mount(root, k, C) {
  const $ = (name) => root.querySelector(`[data-part="${name}"]`);
  const clamp = (v) => Math.min(1, Math.max(0, v));
  const ease = (u) => 0.5 - Math.cos(Math.PI * clamp(u)) / 2;
  const P = C.P;
  const sectors = $('sectors');
  const needle = $('needle');
  const pointer = $('knob-pointer');
  const knobBody = $('knob-body');
  const readout = root.querySelector('[data-readout]');
  const hint = root.querySelector('[data-corner="hint"]');
  const pcts = C.TOKENS.map((_, i) => $(`pct-${i}`));
  const tiles = [...root.querySelectorAll('[data-tile]')];
  const [cx, cy] = C.DIAL;
  const TAU = 2 * Math.PI;
  const onDial = (a, r) => [cx + Math.sin(a) * r, cy - Math.cos(a) * r];
  const softmax = (T) => {
    const e = C.LOGITS.map((l) => Math.exp((l - C.LOGITS[0]) / T));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  };
  const setLit = (g, on) => g.querySelectorAll(':scope > .iso-solid').forEach((s) => (on ? s.setAttribute('data-lit', '') : s.removeAttribute('data-lit')));
  // Text laid along the arc at radius r, reading clockwise, glyphs pointing away from the hub.
  // Where that would read right to left on screen, it runs the other way with glyphs toward the hub.
  const textMatrix = (a, r, z, size) => {
    let o = k.iso([...onDial(a, r), z], P);
    let u = k.iso([...onDial(a + 1 / r, r), z], P);
    let v = k.iso([...onDial(a, r - 1), z], P);
    if (u[0] < o[0]) {
      const r2 = r - size * 0.72;
      o = k.iso([...onDial(a, r2), z], P);
      u = k.iso([...onDial(a - 1 / r2, r2), z], P);
      v = k.iso([...onDial(a, r2 + 1), z], P);
    }
    const f = (n) => Math.round(n * 1000) / 1000;
    return `matrix(${f(u[0] - o[0])} ${f(u[1] - o[1])} ${f(v[0] - o[0])} ${f(v[1] - o[1])} ${f(o[0])} ${f(o[1])})`;
  };

  let plan = null;
  const schedule = (timing, steps) => {
    const temps = [];
    const moves = [];
    let T = 1;
    let angle = 0;
    steps.forEach((st, i) => {
      let t0 = timing.steps[i].t0 + 0.6;
      if (st.temp !== undefined) {
        temps.push({ t0, from: T, to: st.temp });
        T = st.temp;
      }
      if (st.greedy) {
        // Greedy: the needle glides to the middle of the biggest sector.
        const to = softmax(T)[0] * Math.PI;
        moves.push({ t0, dur: 1.6, from: angle, to, kind: 'greedy' });
        angle = to;
        t0 += 1.8;
      }
      for (const roll of st.spin === undefined ? [] : [].concat(st.spin)) {
        const to = angle + 3 * TAU + ((((roll * TAU - angle) % TAU) + TAU) % TAU);
        moves.push({ t0, dur: 2.6, from: angle, to, kind: 'spin', roll });
        angle = to;
        t0 += 3.4;
      }
    });
    return { temps, moves };
  };

  return {
    frame(t, step, timing, steps) {
      plan ??= schedule(timing, steps);
      let T = 1;
      let tempAt = -Infinity;
      for (const m of plan.temps) {
        if (t < m.t0) break;
        T = Math.exp(Math.log(m.from) + (Math.log(m.to) - Math.log(m.from)) * ease((t - m.t0) / 1.2));
        tempAt = m.t0;
      }
      const probs = softmax(T);

      // Needle: a spin decelerates under constant friction (quadratic ease-out).
      let a = 0;
      let move = null;
      let moving = false;
      for (const m of plan.moves) {
        if (t < m.t0) break;
        const u = clamp((t - m.t0) / m.dur);
        a = m.from + (m.to - m.from) * (m.kind === 'spin' ? 1 - (1 - u) * (1 - u) : ease(u));
        moving = u < 1;
        move = m;
      }
      const frac = (((a / TAU) % 1) + 1) % 1;
      let under = probs.length - 1;
      for (let i = 0, acc = 0; i < probs.length; i++) {
        if (frac < acc + probs[i]) { under = i; break; }
        acc += probs[i];
      }
      const landed = move && !moving;
      const glow = landed ? clamp((t - move.t0 - move.dur) / 0.35) : 0;
      // `light:` on the current step overrides what is lit.
      let light = null;
      for (let i = step; i >= 0; i--) if (steps[i].light !== undefined) { light = i === step ? steps[i].light : null; break; }
      const named = light && light !== 'knob';
      const litIndex = named ? C.TOKENS.indexOf(light) : landed && tempAt < move.t0 ? under : -1;
      const litGlow = named ? clamp((t - timing.steps[step].t0 - 0.6) / 0.4) : glow;

      let html = '';
      let start = 0;
      const R0 = C.HUB_R + 3;
      const R1 = C.DIAL_R - 7.5;
      probs.forEach((p, i) => {
        const a0 = start * TAU;
        const a1 = (start + p) * TAU;
        if (i === litIndex && litGlow > 0) {
          const n = Math.max(8, Math.ceil((a1 - a0) / 0.04));
          const outer = Array.from({ length: n + 1 }, (_, j) => onDial(a0 + ((a1 - a0) * j) / n, R1));
          const inner = Array.from({ length: n + 1 }, (_, j) => onDial(a1 - ((a1 - a0) * j) / n, R0));
          html += `<path class="fig-sector" style="opacity:${litGlow.toFixed(3)}" d="${k.onTop([...outer, ...inner], C.DIAL_TOP, P, true)}"/>`;
        }
        html += `<path class="iso-line" data-tone="mid" style="stroke-width:.9" d="${k.lineOnTop(onDial(a0, R0), onDial(a0, R1), C.DIAL_TOP, P)}"/>`;
        // A name along the arc, only where the sector is wide enough to hold it.
        const size = 7;
        const r = R1 - 9;
        const need = (C.TOKENS[i].length * size * 0.6 + 4) / r;
        if (a1 - a0 > need) {
          const mid = (a0 + a1) / 2;
          html += `<text class="fig-text" data-tone="${i === litIndex && litGlow > 0 ? 'lit' : 'hi'}" font-weight="700" font-size="${size}" text-anchor="middle" transform="${textMatrix(mid, r, C.DIAL_TOP, size)}">${C.TOKENS[i]}</text>`;
        }
        start += p;
      });
      sectors.innerHTML = html;
      needle.setAttribute('d', k.lineOnTop(onDial(a + Math.PI, 4.5), onDial(a, C.DIAL_R - 9), C.HUB_TOP, P));

      const ka = ((C.KNOB_DEG * Math.PI) / 180) * Math.log2(T);
      const [kx, ky] = C.KNOB;
      pointer.setAttribute('d', k.lineOnTop([kx - Math.sin(ka) * 2, ky + Math.cos(ka) * 2], [kx + Math.sin(ka) * (C.KNOB_R - 1.6), ky - Math.cos(ka) * (C.KNOB_R - 1.6)], C.KNOB_TOP, P));
      const knobLit = light === 'knob' || (tempAt > -Infinity && t - tempAt < 1.6);
      setLit(knobBody, knobLit);
      pointer.setAttribute('data-tone', knobLit ? 'lit' : 'hi');

      tiles.forEach((g, i) => setLit(g, i === litIndex && litGlow > 0.5));
      probs.forEach((p, i) => {
        const text = `${(p * 100).toFixed(1)}%`;
        if (pcts[i].textContent !== text) pcts[i].textContent = text;
      });

      const pct = (v) => `${(v * 100).toFixed(1)}%`;
      let r = `T = ${T.toFixed(2)}`;
      let h = 'Pick one token';
      if (move && move.t0 > tempAt) {
        if (move.kind === 'greedy') { h = 'Greedy'; r = moving ? 'take the top token…' : `always ${C.TOKENS[under]}`; }
        else { h = 'Sampling'; r = moving ? 'spinning…' : `roll ${move.roll.toFixed(2)} → ${C.TOKENS[under]} (${pct(probs[under])})`; }
      }
      if (tempAt > -Infinity && (!move || tempAt >= move.t0)) { h = 'Temperature'; r = `T = ${T.toFixed(2)} · Paris ${pct(probs[0])} · Lyon ${pct(probs[4])}`; }
      if (light === 'knob' && tempAt === -Infinity) { h = 'Temperature'; r = 'divide every logit by T'; }
      if (readout.textContent !== r) readout.textContent = r;
      if (hint.textContent !== h) hint.textContent = h;
      return [T.toFixed(4), a.toFixed(4), litIndex, litGlow.toFixed(3), knobLit, r].join(',');
    },
  };
}

const constants = { P, TOKENS, LOGITS, DIAL, DIAL_R, DIAL_TOP, HUB_R, HUB_TOP, KNOB, KNOB_R, KNOB_TOP, KNOB_DEG, focus };
writeFigure(HERE, 'sampler', {
  html,
  mount,
  constants,
  demo: { steps: [{ greedy: true }, { spin: [0.15, 0.88] }, { light: 'knob', focus: 'KNOB' }, { temp: 1, focus: 'RACK' }, { temp: 0.5 }, { temp: 2, light: 'Lyon', focus: 'ALL' }], every: 7 },
});
