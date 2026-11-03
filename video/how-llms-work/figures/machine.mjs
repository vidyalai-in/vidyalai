// Fig: the whole machine. One trip through an LLM as one bench machine, built around a loop:
//   T  the text so far, a type slug on a composing rail
//   TK a tokenizer press stamps it into five token tiles, each with its ID
//   EM an embedding cabinet (the lookup table): one drawer per ID gives each tile its vector (pins)
//   ST a tower of identical Transformer blocks; each block lights in turn and updates every vector
//   LG a bank of score bars, one per vocabulary token: the logits
//   SM five graduated tubes fill to the softmax probabilities
//   SP a weighted dial is spun and lands on Paris; the Paris tile comes up in the pocket
//   RE a jib crane carries Paris to the end of the sentence, and the loop runs again
//
//   node video/how-llms-work/figures/machine.mjs [--page]
//
// Scene steps use `station: T|TK|EM|ST|LG|SM|SP|RE|ALL`; the camera glides to that station, its
// part of the machine runs, later stations stay dimmed. Numbers from the article: IDs 791, 6864,
// 315, 9822, 374 (GPT-4's tokenizer); logits 6.0, 3.5, 3.0, 2.5, 1.5; probabilities 85.2, 7.0,
// 4.2, 2.6, 0.9%.

import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as k from '../../tools/iso/iso-kit.mjs';
import { tile, frontText, topText, textW, focusOf, writeFigure } from '../../tools/iso/video-parts.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const WIDTH = 640;
const HEIGHT = 330;

const SENTENCE = 'The capital of France is';
const WORDS = ['The', 'capital', 'of', 'France', 'is'];
const IDS = [791, 6864, 315, 9822, 374];
const TOKENS = ['Paris', 'a', 'the', 'known', 'Lyon'];
const LOGITS = [6.0, 3.5, 3.0, 2.5, 1.5];
const PROBS = [0.852, 0.07, 0.042, 0.026, 0.009];

// ---- world ----
const PLATE = { x: -256, y: -114, w: 506, d: 206, r: 10 };
const PLATE_H = 12;
const FOOT_H = 5;
const TILE_D = 14;
const TILE_H = 15;

const RAIL = { x: -244, y: 44, w: 158, d: 28, r: 2 };
const RAIL_H = 3;
const ROW_Y = 50; // tiles stand from y 50 to 64
const SLUG = { x: -240, w: textW(SENTENCE, 7) + 10 };
const PARIS_W = textW('Paris', 6.2) + 9;
const LAND = { x: SLUG.x + SLUG.w + 1.6, y: ROW_Y, z: RAIL_H };
const LX = LAND.x + PARIS_W / 2; // the crane's line

const PRESS = { x: -80, w: 30 };
const TRAY = { x: -84, y: 44, w: 194, d: 28, r: 2 };
const tileWs = WORDS.map((w, i) => Math.max(textW(w, 6.2), textW(IDS[i], 4.2)) + 9);
const tileXs = [];
{
  let x = -44;
  for (const w of tileWs) {
    tileXs.push(x);
    x += w + 3;
  }
}

const CAB = { x: 116, y: 32, w: 48, d: 42, r: 1.5 };
const CAB_Z = 3;
const CAB_H = 58;
const DRAWERS = 8;
const DRAWER_Z = (i) => CAB_Z + 3 + i * 6.8;
const DRAWER_OF = [2, 5, 0, 7, 1]; // row of each ID in the table (sorted by ID)

const TOWER = { x: 180, y: -42, w: 52, d: 60, r: 2 };
const BLOCKS = 8;
const BLOCK_Z = (j) => 10 + j * 10.5;
const BLOCK_H = 6.5;
const TOWER_TOP = 96;

const LOGB = { x: 18, y: -100, w: 96, d: 20, r: 1.5 };
const LOGB_H = 9;
// Only the article's five candidates: softmax over exactly these gives 85.2% for Paris, so the
// bank must not add other words (each would take a share and break the tubes' numbers).
const BARS = 5;
const NAMED = [0, 1, 2, 3, 4];
const BAR_X = (i) => LOGB.x + 10 + i * 17;
const barLogits = LOGITS;
const BAR_SCALE = 5;

const CYB = { x: -82, y: -102, w: 94, d: 24, r: 1.5 };
const CYB_H = 10;
const TUBE_X = (i) => -70 + i * 18;
const TUBE_Y = -92;
const TUBE_R = 6.5;
const TUBE_H = 42;

const DIAL = [-196, -62];
const DIAL_R = 30;
const DIAL_Z = 3;
const DIAL_H = 8;
const DIAL_TOP = DIAL_Z + DIAL_H;
const HUB_TOP = DIAL_TOP + 2.5;

const POCK = { x: LX - 18, y: -76, w: 36, d: 24, r: 1.5 };
const POCK_TILE_Y = -71;
const MAST = [LX, -100];
const ARM_Z = 46;

// Vectors: six pins per token; one set from the embedding table and one after each block.
const PINS = 6;
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const PIN_SETS = Array.from({ length: BLOCKS + 1 }, () => WORDS.map(() => Array.from({ length: PINS }, () => 1.5 + rand() * 8)));

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [TOWER.x, TOWER.y, TOWER_TOP + 4], [TOWER.x + TOWER.w, TOWER.y + TOWER.d, TOWER_TOP + 4], [MAST[0], -106, ARM_Z + 5]],
  WIDTH,
  HEIGHT,
  { pad: 14, azimuth: 60 },
);

const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);
const slab = (plan, z, h, steps = 3, bevel = 0.4) => k.slabOf(plan, z, h, P, steps, bevel);
const station = (id, body) => `<g data-station="${id}">${body}</g>`;
const trace = (id, pts) => `<path class="fig-trace" data-trace="${id}" d="${k.onTop(pts, 0.2, P)}"/>`;

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
  // Maker's plate, front right.
  const label = { x: 176, y: 56, w: 54, d: 20, r: 1.5 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 4), { tone: 'lo', crease: 'none' }));
  out.push(L(k.planOutline(k.insetPlan(label, 1.6), 0.8, P, 4), { tone: 'faint' }));
  [[34, 6], [26, 10.5], [30, 15]].forEach(([len, y]) => out.push(L(k.lineOnTop([label.x + 6, label.y + y], [label.x + 6 + len, label.y + y], 0.8, P), { tone: 'lo', free: true })));
  out.push(D([at([label.x + label.w - 4, label.y + 4, 0.8]), at([label.x + label.w - 4, label.y + label.d - 4, 0.8])], { size: 0.42 }));
  // Traces engraved in the plate: the path one trip takes.
  out.push(trace('ST', [[100, 58], [198, 58], [198, 20]]));
  out.push(trace('LG', [[198, -30], [198, -90], [118, -90]]));
  out.push(trace('SM', [[22, -90], [8, -90]]));
  out.push(trace('SP', [[-80, -94], [-180, -94], [-180, -82]]));
  out.push(trace('SP', [[-170, -62], [POCK.x + 4, -62]]));
  return out.join('');
}

function ruler(plan, z) {
  const strip = { x: plan.x + 2, y: plan.y + plan.d - 5, w: plan.w - 4, d: 4, r: 0.6 };
  const t = k.topTicks(strip.x + 2, strip.x + strip.w - 2, 4, 5, strip.y + 0.3, z + 0.4, [1.4, 2.8], P);
  return S(slab(strip, z, 0.4, 2, 0), { tone: 'lo', crease: 'none' }) + L(t.minor, { tone: 'lo' }) + L(t.major, { tone: 'mid' });
}

// ---- back row: crane mast, dial, pocket, tubes, score bars ----
function mast() {
  const [x, y] = MAST;
  return [
    S(k.cylinder(x, y, 7, 0, 2, P, 28), { tone: 'mid' }),
    D(k.circleRing(x, y, 5.4, 6).map(([px, py]) => at([px, py, 2])), { size: 0.4 }),
    S(k.cylinder(x, y, 3.4, 2, ARM_Z - 6, P, 24), { tone: 'mid', crease: 'none' }),
    L([16, 30].map((z) => k.sideArc(x, y, 3.4, z, P, 14)).join(''), { tone: 'faint' }),
    S(k.cylinder(x, y, 5, ARM_Z - 4, 4, P, 24), { tone: 'mid' }),
  ].join('');
}

function dial() {
  const [x, y] = DIAL;
  const out = [];
  out.push(S(k.cylinder(x, y, DIAL_R - 5, 0, DIAL_Z, P, 56), { tone: 'mid' }));
  out.push(S(k.cylinder(x, y, DIAL_R, DIAL_Z, DIAL_H, P, 80, 0.8), { tone: 'hi', crease: 'lo' }));
  out.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 3, DIAL_TOP, 80, P, 0.3), { tone: 'lo' }));
  out.push(L(k.ring(x, y, DIAL_R - 2, DIAL_TOP, P, 80), { tone: 'lo' }));
  const t = k.radialTicks(x, y, DIAL_R - 2.6, 100, 10, DIAL_TOP, [1.2, 2.6], P);
  out.push(L(t.minor, { tone: 'lo' }), L(t.major, { tone: 'mid' }));
  out.push('<g data-part="sector"></g>');
  // Sector edges at the cumulative probabilities.
  const onDial = (a, r) => [x + Math.sin(a) * r, y - Math.cos(a) * r];
  let acc = 0;
  for (const p of PROBS) {
    const a = acc * 2 * Math.PI;
    out.push(L(k.lineOnTop(onDial(a, 7), onDial(a, DIAL_R - 6), DIAL_TOP, P), { tone: 'mid' }));
    acc += p;
  }
  out.push(L(k.ring(x, y, DIAL_R - 6, DIAL_TOP, P, 80), { tone: 'faint' }));
  out.push(L(k.ring(x, y, 7, DIAL_TOP, P, 28), { tone: 'faint' }));
  out.push(topText(P, onDial(PROBS[0] * Math.PI, 15)[0], onDial(PROBS[0] * Math.PI, 15)[1] + 2, DIAL_TOP, 'Paris', { size: 6 }));
  out.push(S(k.cylinder(x, y, 4, DIAL_TOP, HUB_TOP - DIAL_TOP, P, 24), { tone: 'mid' }));
  out.push('<path data-part="needle" class="iso-line" data-tone="lit" style="stroke-width:1.4"/>');
  out.push(S(k.cylinder(x, y, 1.6, HUB_TOP, 1, P, 16), { tone: 'hi', lit: true }));
  return out.join('');
}

function pocket() {
  const out = [S(slab(POCK, 0, 3, 3, 0.5), { tone: 'mid' })];
  out.push(L(k.planOutline(k.insetPlan(POCK, 2.4), 3, P, 3), { tone: 'faint' }));
  out.push(D(k.corners(POCK, 2.2).map(([x, y]) => at([x, y, 3])), { size: 0.4 }));
  return out.join('');
}

function tubes() {
  const out = [S(slab(CYB, 0, CYB_H, 4, 0.6), { tone: 'mid' })];
  TOKENS.forEach((tok, i) => {
    const x = TUBE_X(i);
    out.push(L(k.ring(x, TUBE_Y, TUBE_R + 1.6, CYB_H, P, 28), { tone: 'faint' }));
    out.push(`<g data-part="liquid-${i}"></g>`);
    const glass = k.cylinder(x, TUBE_Y, TUBE_R, CYB_H, TUBE_H, P, 36);
    out.push(`<path class="fig-glass" d="${glass.outline}"/>`);
    out.push(L(k.ring(x, TUBE_Y, TUBE_R, CYB_H + TUBE_H, P, 36), { tone: 'mid' }));
    // Graduations at 25, 50 and 75%.
    out.push(L([0.25, 0.5, 0.75].map((f) => k.sideArc(x, TUBE_Y, TUBE_R, CYB_H + 0.5 + f * (TUBE_H - 1), P, 16)).join(''), { tone: 'lo' }));
    out.push(frontText(P, x, CYB.y + CYB.d, 5.4, tok, { size: 3.9 }));
    out.push(frontText(P, x, CYB.y + CYB.d, 1.4, `${(PROBS[i] * 100).toFixed(1)}%`, { size: 3.2, tone: 'mid', weight: 500 }));
  });
  return out.join('');
}

function scores() {
  const out = [S(slab(LOGB, 0, LOGB_H, 4, 0.6), { tone: 'mid' })];
  out.push(L(k.planOutline(k.insetPlan(LOGB, 2), LOGB_H, P, 3), { tone: 'faint' }));
  for (let i = 0; i < BARS; i++) out.push(`<g data-part="bar-${i}"></g>`);
  NAMED.forEach((b, i) => {
    const x = BAR_X(b);
    out.push(frontText(P, x, LOGB.y + LOGB.d, 4.6, TOKENS[i], { size: 3.8 }));
    out.push(frontText(P, x, LOGB.y + LOGB.d, 1.2, LOGITS[i].toFixed(1), { size: 3.1, tone: 'mid', weight: 500 }));
  });
  return out.join('');
}

// ---- the tower of Transformer blocks ----
function tower() {
  const out = [S(slab({ x: TOWER.x - 8, y: TOWER.y - 8, w: TOWER.w + 16, d: TOWER.d + 16, r: 2.5 }, 0, 4, 4, 0.6), { tone: 'mid' })];
  const posts = [[TOWER.x - 3, TOWER.y - 3], [TOWER.x + TOWER.w + 3, TOWER.y - 3], [TOWER.x - 3, TOWER.y + TOWER.d + 3], [TOWER.x + TOWER.w + 3, TOWER.y + TOWER.d + 3]];
  const post = ([x, y]) => S(k.cylinder(x, y, 2.6, 4, TOWER_TOP - 4, P, 18), { tone: 'mid', crease: 'none' });
  out.push(post(posts[0]));
  for (let j = 0; j < BLOCKS; j++) {
    const z = BLOCK_Z(j);
    const face = TOWER.y + TOWER.d;
    // Front face: a grid of dots (attention) and four vents (the MLP).
    const dots = [];
    for (let c = 0; c < 7; c++) for (let r = 0; r < 2; r++) dots.push(at([TOWER.x + 6 + c * 3.2, face, z + 2 + r * 2.6]));
    const vents = [0, 1, 2, 3].map((v) => {
      const x0 = TOWER.x + 32 + v * 4;
      return k.pathOf([at([x0, face, z + 1.4]), at([x0 + 1.4, face, z + 1.4]), at([x0 + 1.4, face, z + 5.2]), at([x0, face, z + 5.2])], true);
    });
    out.push(
      `<g data-block="${j}">${S(slab(TOWER, z, BLOCK_H, 4, 0.6), { tone: 'hi', crease: 'lo', inner: D(dots, { size: 0.42, tone: 'mid' }) + L(vents.join(''), { tone: 'lo' }) })}</g>`,
    );
    // Clips holding each block to the front posts.
    out.push(D([at([TOWER.x + 1, face, z + BLOCK_H / 2]), at([TOWER.x + TOWER.w - 1, face, z + BLOCK_H / 2])], { size: 0.5, tone: 'mid' }));
  }
  out.push(post(posts[1]), post(posts[2]), post(posts[3]));
  const cap = { x: TOWER.x - 6, y: TOWER.y - 6, w: TOWER.w + 12, d: TOWER.d + 12, r: 2.5 };
  out.push(S(slab(cap, TOWER_TOP, 4, 4, 0.6), { tone: 'mid' }));
  out.push(D(posts.map(([x, y]) => at([x, y, TOWER_TOP + 4])), { size: 0.5 }));
  out.push(L(k.planOutline(k.insetPlan(cap, 4), TOWER_TOP + 4, P, 3), { tone: 'faint' }));
  return out.join('');
}

// ---- front row ----
function railRow() {
  const out = [S(slab(RAIL, 0, RAIL_H), { tone: 'mid' }), ruler(RAIL, RAIL_H)];
  const fence = { x: RAIL.x, y: RAIL.y - 4, w: RAIL.w, d: 4, r: 1 };
  out.push(S(slab(fence, 0, 12), { tone: 'mid' }));
  out.push(D([RAIL.x + 20, RAIL.x + 80, RAIL.x + 140].map((x) => at([x, fence.y + 2, 12])), { size: 0.45 }));
  const head = { x: RAIL.x - 6, y: fence.y, w: 6, d: RAIL.d + 4, r: 1.2 };
  out.push(S(slab(head, 0, 18, 3, 0.5), { tone: 'mid' }));
  out.push(D([at([head.x + 3, head.y + 4, 18]), at([head.x + 3, head.y + head.d - 4, 18])], { size: 0.45 }));
  // The slug: the whole text so far, typed in letter by letter at step T.
  const plan = { x: SLUG.x, y: ROW_Y, w: SLUG.w, d: TILE_D, r: 1.2 };
  out.push(
    `<g data-litwhen="T">${S(slab(plan, RAIL_H, TILE_H, 3, 0.5), { tone: 'hi', crease: 'lo' })}</g>` +
      L(k.planOutline(k.insetPlan(plan, 1.4), RAIL_H + TILE_H, P, 3), { tone: 'faint' }) +
      frontText(P, SLUG.x + 5, ROW_Y + TILE_D, RAIL_H + TILE_H / 2 - 2.5, SENTENCE, { size: 7, anchor: 'start', cls: 'fig-typed' }).replace('<text', '<text data-part="slug-text"'),
  );
  return out.join('');
}

function pressBack() {
  const col = { x: PRESS.x, y: 30, w: PRESS.w, d: 10, r: 1.5 };
  return [
    S(slab({ x: col.x - 3, y: col.y - 3, w: col.w + 6, d: col.d + 4, r: 2 }, 0, 3, 3, 0.5), { tone: 'mid' }),
    S(slab(col, 3, 40, 3, 0.5), { tone: 'mid' }),
    D([at([col.x + 4, col.y + col.d, 34]), at([col.x + col.w - 4, col.y + col.d, 34])], { size: 0.5 }),
  ].join('');
}

function trayRow() {
  const out = [S(slab(TRAY, 0, RAIL_H), { tone: 'mid' }), ruler(TRAY, RAIL_H)];
  out.push(S(slab({ x: TRAY.x, y: TRAY.y - 4, w: TRAY.w, d: 4, r: 1 }, 0, 8), { tone: 'mid' }));
  WORDS.forEach((w, i) => {
    out.push(
      `<g data-part="tile-${i}" style="opacity:0"><g data-litwhen="TK">${tile(P, { x: 0, y: ROW_Y, z: RAIL_H, w: tileWs[i], d: TILE_D, h: TILE_H, word: w, sub: String(IDS[i]) })}</g><g data-part="pins-${i}"></g></g>`,
    );
  });
  return out.join('');
}

function pressFront() {
  const arm = { x: PRESS.x - 2, y: 30, w: PRESS.w + 4, d: 40, r: 2 };
  return [
    S(slab(arm, 43, 6, 4, 0.6), { tone: 'mid' }),
    D([at([arm.x + 4, arm.y + arm.d - 4, 49]), at([arm.x + arm.w - 4, arm.y + arm.d - 4, 49])], { size: 0.5 }),
    '<g data-part="ram"></g>',
    `<g data-part="head">${S(slab({ x: PRESS.x, y: 49, w: PRESS.w, d: 18, r: 1.5 }, 0, 10, 3, 0.5), { tone: 'mid' })}${L(k.planOutline({ x: PRESS.x + 3, y: 52, w: PRESS.w - 6, d: 12, r: 1 }, 10, P, 3), { tone: 'faint' })}</g>`,
  ].join('');
}

function cabinet() {
  const out = [S(slab({ x: CAB.x - 2, y: CAB.y - 2, w: CAB.w + 4, d: CAB.d + 4, r: 2 }, 0, CAB_Z, 3, 0.5), { tone: 'mid' })];
  out.push(S(slab(CAB, CAB_Z, CAB_H, 3, 0.6), { tone: 'mid' }));
  out.push(L(k.planOutline(k.insetPlan(CAB, 3), CAB_Z + CAB_H, P, 3), { tone: 'faint' }));
  for (let i = 0; i < DRAWERS; i++) out.push(`<g data-part="drawer-${i}"></g>`);
  return out.join('');
}

// ---- the crane's moving parts and the picked tile (drawn last) ----
function crane() {
  const arm = { x: LX - 3, y: -106, w: 6, d: 174, r: 1.5 };
  return [
    S(slab(arm, ARM_Z, 5, 3, 0.5), { tone: 'mid' }),
    D([-96, -40, 20, 62].map((y) => at([LX, y, ARM_Z + 5])), { size: 0.42 }),
    S(slab({ x: LX - 4, y: arm.y + arm.d - 2, w: 8, d: 3, r: 1 }, ARM_Z - 3, 9, 2), { tone: 'mid' }),
    '<g data-part="trolley"></g>',
  ].join('');
}

const pickedTile = `<g data-part="picked" style="opacity:0"><g data-litwhen="SP RE">${tile(P, { x: 0, y: 0, z: 0, w: PARIS_W, d: TILE_D, h: TILE_H, word: 'Paris' })}</g></g>`;

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    'One trip through a language model as a bench machine. A type slug holds the text so far; a press stamps it into five token tiles with their ID numbers; drawers of an embedding cabinet give each tile a vector of pins; a tower of identical Transformer blocks updates the vectors; a bank of bars shows one score per vocabulary token; five glass tubes fill to the softmax probabilities; a weighted dial is spun and lands on Paris; and a jib crane sets the Paris tile at the end of the sentence.',
  body: [
    plate(),
    station('RE', mast()),
    station('LG', scores()),
    station('SM', tubes()),
    station('SP', dial() + pocket()),
    station('ST', tower()),
    station('TK', pressBack()),
    station('T', railRow()),
    station('TK', trayRow() + pressFront()),
    station('EM', cabinet()),
    station('SP', pickedTile),
    station('RE', crane()),
  ],
});

const html = k.plateHtml({ fig: 'Fig 1', title: 'One trip, one token', hint: 'The whole machine', readout: '', body: `<div class="iso-stage">${svg}</div>` });

// Camera targets, as fractions of the drawing.
const F = (box, pad) => focusOf(P, WIDTH, HEIGHT, box, pad);
const focus = {
  T: F([-250, 30, 0, -80, 72, 20], 0.1),
  TK: F([-86, 28, 0, 110, 72, 50], 0.08),
  EM: F([-24, 24, 0, 172, 84, 64], 0.06),
  ST: F([30, -50, 0, 242, 72, 100], 0.04),
  LG: F([10, -104, 0, 132, -78, 36], 0.1),
  SM: F([-90, -106, 0, 40, -76, 52], 0.12),
  SP: F([-230, -110, 0, -90, -30, 20], 0.12),
  RE: F([-250, -110, 0, -60, 72, 72], 0.04),
  ALL: [0, 0, 1, 1],
};

// ---- runtime: everything from the scene clock ----
function mount(root, k, C) {
  const $ = (name) => root.querySelector(`[data-part="${name}"]`);
  const clamp = (v) => Math.min(1, Math.max(0, v));
  const ease = (u) => 0.5 - Math.cos(Math.PI * clamp(u)) / 2;
  const lerp = (a, b, u) => a + (b - a) * u;
  const P = C.P;
  const ORDER = ['T', 'TK', 'EM', 'ST', 'LG', 'SM', 'SP', 'RE'];
  const DELAY = 0.7; // let the camera arrive before the part moves
  const hint = root.querySelector('[data-corner="hint"]');
  const readout = root.querySelector('[data-readout]');
  const slugText = $('slug-text');
  const tiles = C.WORDS.map((_, i) => $(`tile-${i}`));
  const pins = C.WORDS.map((_, i) => $(`pins-${i}`));
  const bars = Array.from({ length: C.BARS }, (_, i) => $(`bar-${i}`));
  const liquids = C.TOKENS.map((_, i) => $(`liquid-${i}`));
  const drawers = Array.from({ length: C.DRAWERS }, (_, i) => $(`drawer-${i}`));
  const blocks = [...root.querySelectorAll('[data-block]')];
  const groups = [...root.querySelectorAll('[data-station]')];
  const litGroups = [...root.querySelectorAll('[data-litwhen]')];
  const traces = [...root.querySelectorAll('[data-trace]')];
  const slab = (plan, z, h, steps = 3, bevel = 0.4) => k.slabOf(plan, z, h, P, steps, bevel);
  const solid = (paths, style) => k.solidSvg(paths, style);
  const setLit = (g, on) => g.querySelectorAll(':scope > .iso-solid').forEach((s) => (on ? s.setAttribute('data-lit', '') : s.removeAttribute('data-lit')));

  let plan = null;
  const schedule = (timing, steps) => {
    const start = {};
    // A step may run several stations one after another: station: [SP, RE].
    const DUR = { T: 1.6, TK: 2.8, EM: 2.8, ST: 5.2, LG: 1.8, SM: 1.8, SP: 3.4, RE: 3.8 };
    steps.forEach((st, i) => {
      let t0 = timing.steps[i].t0 + DELAY;
      for (const id of [].concat(st.station ?? [])) {
        if (id === 'ALL') continue;
        if (start[id] === undefined) start[id] = t0;
        t0 += DUR[id];
      }
    });
    return start;
  };
  // Seconds since a station's part started moving; Infinity when the scene never visits it.
  const since = (id, t) => (plan[id] === undefined ? Infinity : t - plan[id]);

  const cache = new Map();
  const put = (el, key, html) => {
    if (cache.get(el) !== key) {
      cache.set(el, key);
      el.innerHTML = html;
    }
  };

  return {
    frame(t, step, timing, steps) {
      plan ??= schedule(timing, steps);
      let current = null;
      for (let i = step; i >= 0; i--) {
        if (!steps[i].station) continue;
        const ids = [].concat(steps[i].station);
        current = ids.filter((id) => id === 'ALL' || !(plan[id] > t)).at(-1) ?? ids[0];
        break;
      }
      const ci = ORDER.indexOf(current);
      const keys = [current];

      // Dim the stations still to come; light the moving parts of the current one.
      for (const g of groups) {
        const gi = ORDER.indexOf(g.dataset.station);
        const dim = current && current !== 'ALL' && gi > ci;
        dim ? g.setAttribute('data-dim', '') : g.removeAttribute('data-dim');
      }
      for (const g of litGroups) setLit(g, current !== null && current !== 'ALL' && g.dataset.litwhen.split(' ').includes(current));

      // T: the text is typed in.
      const tT = since('T', t);
      const chars = Math.round(clamp(tT / 1.4) * C.SENTENCE.length);
      const text = C.SENTENCE.slice(0, chars);
      if (slugText.textContent !== text) slugText.textContent = text;

      // TK: the press head comes down, then the five tiles slide out, the last word first.
      const tK = since('TK', t);
      const stroke = tK < 0 ? 0 : tK < 0.35 ? ease(tK / 0.35) : tK < 0.7 ? 1 - ease((tK - 0.35) / 0.35) : 0;
      const headZ = 22 - 3.5 * stroke;
      $('head').setAttribute('transform', k.translateAlong(P, [0, 0, headZ]));
      put($('ram'), headZ.toFixed(2), solid(k.cylinder(C.PRESS.x + C.PRESS.w / 2, 58, 3.2, headZ + 10, 43 - headZ - 10, P, 18), { tone: 'mid', crease: 'none' }));
      let flying = 0;
      C.WORDS.forEach((_, i) => {
        const local = tK - 0.7 - (C.WORDS.length - 1 - i) * 0.3;
        const u = clamp(local / 1.0);
        const x = lerp(C.PRESS.x + 2, C.tileXs[i], ease(u));
        tiles[i].setAttribute('transform', k.translateAlong(P, [x, 0, 0]));
        tiles[i].style.opacity = local <= 0 ? '0' : String(clamp(u * 3));
        if (u > 0 && u < 1) flying++;
      });

      // EM: one drawer per token slides out and the token takes its vector.
      const tE = since('EM', t);
      const out = new Array(C.DRAWERS).fill(0);
      const grown = C.WORDS.map((_, i) => {
        const local = tE - i * 0.45;
        const row = C.DRAWER_OF[i];
        out[row] = Math.max(out[row], local < 0 ? 0 : local < 0.3 ? ease(local / 0.3) : local < 0.5 ? 1 : 1 - ease((local - 0.5) / 0.3));
        return clamp((local - 0.15) / 0.45);
      });
      drawers.forEach((el, i) => {
        const s = out[i] * 11;
        const z = C.DRAWER_Z(i);
        put(el, s.toFixed(2), solid(slab({ x: C.CAB.x + 3, y: C.CAB.y + C.CAB.d, w: C.CAB.w - 6, d: 1.5 + s, r: 0.8 }, z, 5.8, 2, 0.3), { tone: out[i] > 0 ? 'hi' : 'mid', lit: out[i] > 0 }) +
          k.dotsSvg([k.iso([C.CAB.x + C.CAB.w / 2, C.CAB.y + C.CAB.d + 1.5 + s, z + 2.9], P)], { size: 0.55, tone: 'mid' }));
      });

      // ST: the blocks light from the bottom up; each one updates every vector.
      const tS = since('ST', t);
      const per = 0.6;
      const active = tS < 0.3 ? -1 : Math.floor((tS - 0.3) / per);
      blocks.forEach((b, j) => setLit(b, current === 'ST' && j === active));
      const stage = clamp((tS - 0.3) / per / C.BLOCKS) * C.BLOCKS; // 0..8, fractional
      const s0 = Math.min(C.BLOCKS, Math.floor(stage));
      const sf = stage - s0;
      const pinsLit = current === 'EM' || current === 'ST';
      C.WORDS.forEach((_, i) => {
        if (grown[i] <= 0) return put(pins[i], 'none', '');
        const a = C.PIN_SETS[s0][i];
        const b = C.PIN_SETS[Math.min(C.BLOCKS, s0 + 1)][i];
        const hs = a.map((v, p) => lerp(v, b[p], ease(sf)) * ease(grown[i]));
        const key = hs.map((h) => h.toFixed(2)).join() + pinsLit;
        const w = C.tileWs[i];
        const gap = (w - 4) / C.PINS;
        const z = C.RAIL_H + C.TILE_H;
        put(pins[i], key, hs.map((h, p) => solid(slab({ x: 2 + p * gap + (gap - 1.4) / 2, y: C.ROW_Y + 5.5, w: 1.4, d: 3, r: 0.4 }, z, Math.max(0.2, h), 2, 0), { tone: 'hi', crease: 'none', lit: pinsLit })).join(''));
      });

      // LG: one score bar per vocabulary token grows to its logit.
      const tL = since('LG', t);
      bars.forEach((el, i) => {
        const u = ease((tL - i * 0.12) / 1.2);
        const h = Math.max(0, C.barLogits[i] * C.BAR_SCALE * u);
        const named = C.NAMED.includes(i);
        put(el, h.toFixed(2) + (current === 'LG'), h <= 0.05 ? '' : solid(slab({ x: C.BAR_X(i) - 2.5, y: C.LOGB.y + 5, w: 5, d: 6, r: 0.6 }, C.LOGB_H, h, 2, 0), { tone: named ? 'hi' : 'mid', crease: 'none', lit: named && i === C.NAMED[0] && current === 'LG' }));
      });

      // SM: the tubes fill to the probabilities.
      const tM = since('SM', t);
      liquids.forEach((el, i) => {
        const u = ease((tM - i * 0.12) / 1.3);
        const h = (C.TUBE_H - 1) * C.PROBS[i] * u;
        put(el, h.toFixed(2) + (current === 'SM'), h <= 0.05 ? '' : solid(k.cylinder(C.TUBE_X(i), C.TUBE_Y, C.TUBE_R - 0.9, C.CYB_H + 0.4, h, P, 32), { tone: 'hi', crease: 'none', lit: i === 0 }));
      });

      // SP: the needle is spun (constant friction: quadratic ease-out) and lands on Paris.
      const tP = since('SP', t);
      const SPIN = 2.8;
      const target = (C.ROLL + 3) * 2 * Math.PI;
      const a = tP <= 0 ? 0 : target * (1 - Math.pow(1 - clamp(tP / SPIN), 2));
      const [dx, dy] = C.DIAL;
      const onDial = (ang, r) => [dx + Math.sin(ang) * r, dy - Math.cos(ang) * r];
      $('needle').setAttribute('d', k.lineOnTop(onDial(a + Math.PI, 3), onDial(a, C.DIAL_R - 7), C.HUB_TOP, P));
      const glow = clamp((tP - SPIN) / 0.35);
      if (glow > 0) {
        const n = 60;
        const a1 = C.PROBS[0] * 2 * Math.PI;
        const outer = Array.from({ length: n + 1 }, (_, j) => onDial((a1 * j) / n, C.DIAL_R - 6));
        const inner = Array.from({ length: n + 1 }, (_, j) => onDial(a1 - (a1 * j) / n, 7));
        put($('sector'), glow.toFixed(2), `<path class="fig-sector" style="opacity:${glow.toFixed(3)}" d="${k.onTop([...outer, ...inner], C.DIAL_TOP, P, true)}"/>`);
      } else put($('sector'), '0', '');

      // The Paris tile comes up in the pocket, then RE: the crane sets it at the end of the line.
      const rise = clamp((tP - SPIN - 0.2) / 0.5);
      const tR = since('RE', t);
      const LIFT = 40;
      const REST = 21; // gripper height with the tile on a floor at z 3
      let ty = C.POCK_Y;
      let gz = LIFT;
      let tileZ = 3 - 6 * (1 - ease(rise));
      let tileY = C.POCK_Y - 7;
      let carried = false;
      const u = tR / 3.6;
      if (tR > 0) {
        const seg = (a0, a1) => ease((u - a0) / (a1 - a0));
        if (u < 0.14) gz = lerp(LIFT, REST, seg(0, 0.14));
        else if (u < 0.2) gz = REST;
        else if (u < 0.34) { gz = lerp(REST, LIFT, seg(0.2, 0.34)); carried = true; }
        else if (u < 0.7) { gz = LIFT; ty = lerp(C.POCK_Y, C.RAIL_Y, seg(0.34, 0.7)); carried = true; }
        else if (u < 0.84) { gz = lerp(LIFT, REST, seg(0.7, 0.84)); ty = C.RAIL_Y; carried = true; }
        else { gz = lerp(REST, LIFT, seg(0.84, 1)); ty = C.RAIL_Y; }
        if (carried) { tileZ = gz - 3 - C.TILE_H; tileY = ty - 7; }
        else if (u >= 0.84) { tileZ = 3; tileY = C.RAIL_Y - 7; }
      }
      const picked = $('picked');
      picked.style.opacity = tP > SPIN ? String(ease(rise)) : '0';
      picked.setAttribute('transform', k.translateAlong(P, [C.LAND_X, tileY, tileZ]));
      const LX = C.LX;
      const trolley = [
        solid(slab({ x: LX - 5, y: ty - 6, w: 10, d: 12, r: 1 }, C.ARM_Z - 4, 4, 2, 0.3), { tone: 'mid' }),
        k.lineSvg(k.segment([LX, ty, C.ARM_Z - 4], [LX, ty, gz], P), { tone: 'mid' }),
        solid(slab({ x: LX - 15.4, y: ty - 4, w: 30.8, d: 8, r: 1 }, gz - 3, 3, 2, 0.3), { tone: 'mid', lit: current === 'RE' }),
        solid(slab({ x: LX - 15.4, y: ty - 3, w: 1.4, d: 6, r: 0.4 }, gz - 9, 6, 2), { tone: 'mid' }),
        solid(slab({ x: LX + 14, y: ty - 3, w: 1.4, d: 6, r: 0.4 }, gz - 9, 6, 2), { tone: 'mid' }),
      ].join('');
      put($('trolley'), `${ty.toFixed(2)},${gz.toFixed(2)},${current === 'RE'}`, trolley);

      // The engraved trace into the current station flows while its part runs.
      const phase = (-t * 14).toFixed(1);
      for (const tr of traces) {
        const on = tr.dataset.trace === current;
        on ? tr.setAttribute('data-on', '') : tr.removeAttribute('data-on');
        tr.style.strokeDashoffset = on ? phase : '';
      }

      // Corner labels.
      const NAMES = { T: '1 · The text so far', TK: '2 · Tokenizer', EM: '3 · Embedding', ST: '4 · Transformer blocks', LG: '5 · Logits', SM: '6 · Softmax', SP: '7 · Sample', RE: '8 · Append, repeat', ALL: 'The whole machine' };
      const name = NAMES[current] ?? 'The whole machine';
      if (hint.textContent !== name) hint.textContent = name;
      const pct = (v) => `${(v * 100).toFixed(1)}%`;
      const R = {
        T: chars ? `"${text}"` : 'typing…',
        TK: flying || tK < 0.7 ? 'splitting into tokens…' : '5 tokens · France = 9822',
        EM: 'each ID → a vector of numbers',
        ST: active >= 0 && active < C.BLOCKS ? `block ${active + 1} · attention + MLP` : 'every vector updated',
        LG: 'top 5 scores shown (made-up logits)',
        SM: `Paris ${pct(C.PROBS[0])} · a ${pct(C.PROBS[1])} · the ${pct(C.PROBS[2])}`,
        SP: tP < SPIN ? 'spinning…' : 'picked: Paris',
        RE: tR < 3.6 ? 'appending Paris…' : '"…France is Paris" · run again',
        ALL: 'one trip = one token',
      };
      const r = R[current] ?? '';
      if (readout.textContent !== r) readout.textContent = r;

      keys.push(chars, stroke.toFixed(3), tK.toFixed(2), tE.toFixed(2), stage.toFixed(3), active, tL.toFixed(2), tM.toFixed(2), a.toFixed(3), glow.toFixed(2), rise.toFixed(2), u.toFixed(3), current === null ? '' : phase);
      return keys.map((v) => (v === Infinity || v === -Infinity ? 'x' : v)).join(',');
    },
  };
}

const constants = {
  P, SENTENCE, WORDS, TOKENS, PROBS, PINS, PIN_SETS, BLOCKS, BARS, NAMED, barLogits, BAR_SCALE, DRAWERS, DRAWER_OF,
  tileXs, tileWs, RAIL_H, TILE_H, ROW_Y, CAB, LOGB, LOGB_H, CYB_H, TUBE_H, TUBE_R, TUBE_Y, DIAL, DIAL_R, DIAL_TOP, HUB_TOP,
  PRESS, ARM_Z, LX, LAND_X: LAND.x, POCK_Y: POCK_TILE_Y + 7, RAIL_Y: ROW_Y + 7, ROLL: 0.3, focus,
};
// Functions the runtime needs, passed as data.
constants.DRAWER_Z = undefined;
constants.BAR_X = undefined;
constants.TUBE_X = undefined;

const mountSrc = mount
  .toString()
  .replace('const P = C.P;', `const P = C.P;\n  C.DRAWER_Z = ${DRAWER_Z.toString()};\n  C.BAR_X = ${BAR_X.toString().replace('LOGB.x', String(LOGB.x))};\n  C.TUBE_X = ${TUBE_X.toString()};\n  C.CAB_Z = ${CAB_Z};`)
  .replace('CAB_Z + 3', `${CAB_Z} + 3`);

const STEPS = ['T', 'TK', 'EM', 'ST', 'LG', 'SM', 'SP', 'RE', 'ALL'].map((station) => ({ station }));
writeFigure(HERE, 'machine', {
  html,
  mount: { toString: () => mountSrc },
  constants,
  demo: { steps: [{}, ...STEPS], every: 5 },
});
