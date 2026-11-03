#!/usr/bin/env node
// Renders stills of chosen scene steps (no TTS, no encoding) for fast layout checks.
//
//   node video/tools/preview.mjs video/<slug> [main|<short-id>] <scene>:<step> [<scene>:<step> ...]
//
// Scene and step numbers are 1-based, as in build warnings. Output: video/<slug>/out/preview/*.png

import { createRequire } from 'node:module';
import { existsSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { prepareScene } from './assets.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(join(ROOT, 'package.json'));
const puppeteer = require('puppeteer');
const YAML = require('yaml');

const [dirArg, ...rest] = process.argv.slice(2);
const videoDir = resolve(dirArg ?? '');
if (!existsSync(join(videoDir, 'script.yaml')) || !rest.length) {
  console.error('Usage: node video/tools/preview.mjs video/<slug> [main|<short-id>] <scene>:<step> ...');
  process.exit(1);
}
const script = YAML.parse(readFileSync(join(videoDir, 'script.yaml'), 'utf8'));
const front = YAML.parse(readFileSync(join(ROOT, script.article), 'utf8').split(/^---$/m)[1]);
const fx = script.fx ?? {};
const which = /^\d+:\d+$/.test(rest[0]) ? 'main' : rest.shift();
const video = which === 'main' ? { format: 'landscape', ...script.main } : { format: 'portrait', ...script.shorts.find((s) => s.id === which) };
if (!video.scenes) throw new Error(`No video "${which}" in script.yaml`);

const fileUrl = (p) => pathToFileURL(join(ROOT, p)).href;
const sizes = { landscape: [1920, 1080], portrait: [1080, 1920] };
const pw = join(homedir(), '.cache/ms-playwright');
const executablePath = existsSync(pw)
  ? readdirSync(pw).filter((d) => d.startsWith('chromium-')).map((d) => join(pw, d, 'chrome-linux64/chrome')).find(existsSync)
  : undefined;
const browser = await puppeteer.launch({ executablePath, args: ['--no-sandbox'] });
const page = await browser.newPage();
const [w, h] = sizes[video.format];
await page.setViewport({ width: w, height: h });
await page.goto(fileUrl('video/tools/stage/stage.html'), { waitUntil: 'networkidle0' });
const outDir = join(videoDir, 'out', 'preview');
mkdirSync(outDir, { recursive: true });

for (const pick of rest) {
  const [sn, tn] = pick.split(':').map(Number);
  const scene = video.scenes[sn - 1];
  if (!scene) throw new Error(`No scene ${sn}`);
  scene.steps = scene.steps?.length ? scene.steps : [{}];
  const assets = {
    logo: fileUrl('src/assets/brand/logo-mark.png'),
    banner: fileUrl(video.format === 'portrait' ? 'src/assets/brand/banner-mobile.png' : 'src/assets/brand/banner1.png'),
  };
  Object.assign(assets, prepareScene(ROOT, scene, front));
  // Stills show the end of the step (no TTS timing: each step is 4 s), without drift.
  const stageFx = { wordSync: false, drift: false, captions: video.format === 'portrait' && fx.shortsCaptions };
  await page.evaluate((spec) => window.stage.load(spec), { format: video.format, scene, assets, fx: stageFx, chapter: scene.chapter ?? 'Preview', progress: video.format === 'landscape' && scene.type !== 'endcard', chrome: !['brand', 'endcard'].includes(scene.type) });
  await page.evaluate((a) => window.stage.set(a), tn - 1);
  const file = join(outDir, `${which}-${sn}-${tn}.png`);
  await page.screenshot({ path: file });
  console.log(relative(ROOT, file));
}
await browser.close();
