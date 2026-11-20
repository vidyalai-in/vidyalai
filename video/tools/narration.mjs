#!/usr/bin/env node
// Print a video's narration on its own, for the listen-only pass (see SKILL.md, step 3).
//
//   node video/tools/narration.mjs video/<slug>              # main video and every Short
//   node video/tools/narration.mjs video/<slug> short-name   # one video
//
// Each line is "scene:step [type] (speaker) text", 1-based like preview.mjs. Lines that only make
// sense with the picture ("as you can see", "this one here") are flagged with !. Reveals whose item
// is named 2 s or more into the step's narration are flagged with ~ (give the step `at:`, see
// script-format.md). Each video ends with its word count and a rough runtime at 145 words/minute
// plus ~12% pauses.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import YAML from 'yaml';

const [dir, only] = process.argv.slice(2);
if (!dir) {
  console.error('usage: node video/tools/narration.mjs video/<slug> [main|<short id>]');
  process.exit(1);
}
const script = YAML.parse(readFileSync(join(dir, 'script.yaml'), 'utf8'));

// Phrases that point at the screen instead of saying what is on it.
const DEICTIC = [
  /\bas you can see\b/i,
  /\byou can see (here|this|that)\b/i,
  /\blook (at )?(here|this)\b/i,
  /\blook at that\b/i,
  /\bclick (on )?(here|this|that)\b/i,
  /\b(this|that) (one|thing|bit|part) (here|there)\b/i,
  /\bover here\b/i,
  /\bright here\b/i,
  /\bthis (is|was) (it|that)\b/i,
  /\blike (this|so)\b[.!]?$/i,
];

// The item a step reveals, and the first word of it worth listening for.
const STOP = new Set('the a an of to and in on is are for with by at from it its this that be as or how why what your you'.split(' '));
const toks = (x) => String(x).toLowerCase().replace(/\*\*|`/g, '').replace(/[^a-z0-9%.\s-]/g, ' ').split(/\s+/).filter(Boolean);
const WPS = 145 / 60; // words per second
function lateReveal(step, ti) {
  if (!step.add || step.at !== undefined) return null;
  const a = step.add;
  const label = typeof a === 'string' ? a : (a.text ?? a.label ?? a.name ?? '');
  // The item counts as named when any of its words is said, so take the earliest one.
  const keys = toks(label).filter((w) => !STOP.has(w) && w.length > 2);
  const said = toks(step.say);
  const hits = keys.map((k) => said.findIndex((w) => w.startsWith(k.slice(0, 5)))).filter((i) => i >= 0);
  if (!hits.length) return null;
  const i = Math.min(...hits);
  if (i / WPS < 2) return null;
  const fix = ti === 0 ? 'split: lead-in on its own first step, item on the next' : `try at: "${toks(step.say)[i]}"`;
  return `"${label}" is named ~${(i / WPS).toFixed(1)} s in (${fix})`;
}

const videos = [
  { id: 'main', scenes: script.main?.scenes ?? [] },
  ...(script.shorts ?? []).map((s) => ({ id: s.id, scenes: s.scenes ?? [] })),
].filter((v) => !only || v.id === only);
if (!videos.length) {
  console.error(`no video "${only}" in ${dir}/script.yaml`);
  process.exit(1);
}

let flagged = 0;
let late = 0;
for (const v of videos) {
  let words = 0;
  console.log(`\n=== ${v.id}`);
  v.scenes.forEach((scene, si) => {
    if (scene.chapter) console.log(`\n-- ${scene.chapter}`);
    (scene.steps ?? []).forEach((step, ti) => {
      if (!step.say) return;
      const text = String(step.say).replace(/\s+/g, ' ').trim();
      words += text.split(' ').length;
      const hit = DEICTIC.some((re) => re.test(text));
      if (hit) flagged++;
      const who = step.speaker ? ` (${step.speaker})` : '';
      console.log(`${hit ? '!' : ' '} ${si + 1}:${ti + 1} [${scene.type}]${who} ${text}`);
      const lr = lateReveal(step, ti);
      if (lr) {
        late++;
        console.log(`~    ${lr}`);
      }
    });
  });
  const min = (words / 145) * 1.12;
  const m = Math.floor(min);
  const s = String(Math.round((min - m) * 60)).padStart(2, '0');
  console.log(`\n   ${words} words, about ${m}:${s}`);
}
if (late) console.log(`\n${late} reveal(s) marked ~ appear well before the narration names them.`);
if (flagged) console.log(`\n${flagged} line(s) marked ! point at the screen: say what is there instead.`);
