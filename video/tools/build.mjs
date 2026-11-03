#!/usr/bin/env node
// Article-to-video build. Turns video/<slug>/script.yaml into narrated MP4s plus a YouTube upload kit.
//
//   node video/tools/build.mjs video/<slug> [--only main,short-id] [--skip-tts] [--torch cuda|cpu]
//                                           [--voice af_heart] [--no-video] [--fast]
//
// Pipeline: script.yaml -> TTS per sentence (tts.py, Kokoro, with word timings) -> timeline from real
// audio durations -> every frame rendered by video/tools/stage/stage.html in headless Chromium, split
// across parallel browsers that each pipe into their own x264 encoder (identical frames reuse the
// previous screenshot) -> narration + optional music bed and sound effects -> captions, chapters,
// description, thumbnails, YouTube card times, checklist.
//
//   [--workers 6]   parallel browsers/encoders

import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { createRequire } from 'node:module';
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync, copyFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import YAML from 'yaml';
import { getSeries } from '../../src/data/series.mjs';
import { prepareScene, seriesParts } from './assets.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(join(ROOT, 'package.json'));
const puppeteer = require('puppeteer');

const FPS = 30;
const TRANSITION_FRAMES = 14; // ~0.47 s entrance animation per step
const SENTENCE_GAP = 0.3; // silence between sentences within a step (s)
const STEP_TAIL = 0.45; // silence after a step's narration (s)
const SCENE_TAIL = 0.4; // extra silence at the end of each scene (s)
const SHORT_END = 1.0; // a Short holds its last frame this much longer, so it doesn't end mid-breath (s)
const SIZES = { landscape: [1920, 1080], portrait: [1080, 1920], thumb: [1280, 720] };
const SITE = 'https://vidyal.ai';

// ---------- CLI ----------
const argv = process.argv.slice(2);
const flag = (name) => argv.includes(`--${name}`);
const opt = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : fallback;
};
const videoDir = resolve(argv[0] ?? '');
if (!existsSync(join(videoDir, 'script.yaml'))) {
  console.error('Usage: node video/tools/build.mjs video/<slug> [--only main,short-id] [--skip-tts] [--torch cuda|cpu]');
  process.exit(1);
}
const only = opt('only', '')?.split(',').filter(Boolean);
const fast = flag('fast'); // smaller x264 preset for quick previews

const script = YAML.parse(readFileSync(join(videoDir, 'script.yaml'), 'utf8'));
const slug = script.slug;
// Engagement effects, all off unless the script turns them on (see references/script-format.md).
const fx = { drift: false, wordSync: false, sfx: false, music: null, musicGap: 18, shortsCaptions: false, ...(script.fx ?? {}) };
// Second voices: `voices: { skeptic: { voice: am_michael, label: Skeptic } }`, used by `speaker:` on a step.
const voices = script.voices ?? {};
const WORKERS = Math.max(1, Number(opt('workers', 6)));
const outDir = join(videoDir, 'out');
const kitDir = join(videoDir, 'youtube');
mkdirSync(outDir, { recursive: true });
mkdirSync(kitDir, { recursive: true });
// Voice: a profile in video/voices/<name>/profile.json (e.g. a cloned voice), or a Kokoro voice name.
const voiceName = opt('voice', script.voice ?? 'af_heart');
const profilePath = join(ROOT, 'video/voices', voiceName, 'profile.json');
const profile = existsSync(profilePath)
  ? JSON.parse(readFileSync(profilePath, 'utf8'))
  : { engine: 'kokoro', voice: voiceName, speed: Number(script.speed ?? 1.0), lang: script.lang ?? 'a' };
const engineName = profile.engine;
if (profile.reference) {
  profile.reference = join(dirname(profilePath), profile.reference);
  if (!existsSync(profile.reference)) throw new Error(`Voice "${voiceName}" needs its reference recording at ${relative(ROOT, profile.reference)} (not in git)`);
}
const tag = opt('tag', ''); // e.g. --tag voice-sample: writes <name>--<tag>.mp4 and skips the upload kit

// ---------- Article: URL and sources ----------
const articlePath = join(ROOT, script.article);
const articleText = readFileSync(articlePath, 'utf8');
const front = YAML.parse(articleText.split(/^---$/m)[1]);
const articleUrl = `${SITE}/blog/${front.slug}/`;
const series = getSeries(front.series);
const sources = [];
{
  const section = articleText.split(/^## Further reading\s*$/m)[1]?.split(/^## /m)[0] ?? '';
  for (const m of section.matchAll(/^- \[([^\]]+)\]\((https?:[^)]+)\)/gm)) sources.push({ title: m[1], url: m[2] });
}

// ---------- Text helpers ----------
const lexicon = { ...JSON.parse(readFileSync(join(ROOT, 'video/tools/lexicon.json'), 'utf8')), ...(script.lexicon ?? {}) };
delete lexicon._comment;
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
// A lexicon value is a string (all engines) or { kokoro?, chatterbox?, default?, ipa? } per engine.
// `ipa` makes cloned-voice takes verify that pronunciation with a phoneme recognizer.
const forEngine = (v) => (typeof v === 'string' ? v : (v[engineName] ?? v.default));
const pronunciationChecks = Object.values(lexicon)
  .filter((v) => typeof v === 'object' && v.ipa && engineName !== 'kokoro')
  .map((v) => ({ say: forEngine(v), ipa: v.ipa }))
  .filter((c, i, all) => all.findIndex((o) => o.say === c.say) === i); // one check per spelling
const lexRules = Object.entries(lexicon)
  .filter(([, v]) => forEngine(v) !== undefined)
  .sort((a, b) => b[0].length - a[0].length)
  .map(([k, v]) => [new RegExp(`(?<![\\w.])${escapeRe(k)}(?![\\w])`, 'g'), forEngine(v)]);

const plain = (s) => s.replace(/\*\*(.+?)\*\*/g, '$1').replace(/`(.+?)`/g, '$1');
function toSpeech(s) {
  let t = plain(s);
  t = t.replace(/(\d)\s*[-–]\s*(\d)/g, '$1 to $2'); // ranges: 6-13 -> 6 to 13
  t = t.replace(/~\s*/g, 'about ').replace(/≈/g, 'about ').replace(/×/g, ' times ').replace(/→/g, ' to ');
  t = t.replace(/(\d)\s*M\b/g, '$1 million').replace(/(\d)\s*B\b/g, '$1 billion').replace(/(\d)\s*k\b/g, '$1 thousand');
  t = t.replace(/(\d)\s*GB\b/g, '$1 gigabytes').replace(/(\d)\s*TB\b/g, '$1 terabytes').replace(/(\d)\s*ms\b/g, '$1 milliseconds');
  t = t.replace(/(\d)\s*s\b/g, '$1 seconds').replace(/&/g, ' and ');
  for (const [re, v] of lexRules) t = t.replace(re, v);
  return t.replace(/\s+/g, ' ').trim();
}
// One sentence per TTS item; split after . ! ? followed by a capital, digit or quote.
const sentences = (s) =>
  String(s ?? '')
    .replace(/\s+/g, ' ')
    .trim()
    .split(/(?<=[.!?])\s+(?=[A-Z0-9"“'(])/)
    .filter(Boolean);
const fmtTime = (sec, srt = false) => {
  const ms = Math.round(sec * 1000);
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  if (srt) return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms % 1000).padStart(3, '0')}`;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
};

// ---------- Validate ----------
// An unquoted comma inside YAML flow syntax `{ sub: a, b }` silently truncates text and leaves a
// stray null key. Catch that (and empty narration) before spending time on TTS and rendering.
{
  const problems = [];
  for (const v of [script.main, ...(script.shorts ?? [])])
    for (const [si, sc] of (v?.scenes ?? []).entries())
      for (const [ti, st] of (sc.steps ?? []).entries()) {
        if (st.add && typeof st.add === 'object')
          for (const [k, val] of Object.entries(st.add))
            if (val === null) problems.push(`${v.id ?? 'main'} scene ${si + 1} step ${ti + 1}: stray key "${k}" (quote values that contain commas)`);
        if (st.say !== undefined && !String(st.say).trim()) problems.push(`${v.id ?? 'main'} scene ${si + 1} step ${ti + 1}: empty say`);
        if (st.speaker && !voices[st.speaker]) problems.push(`${v.id ?? 'main'} scene ${si + 1} step ${ti + 1}: speaker "${st.speaker}" is not in voices:`);
        if (st.values && sc.bars && st.values.length !== sc.bars.length) problems.push(`${v.id ?? 'main'} scene ${si + 1} step ${ti + 1}: values has ${st.values.length} numbers for ${sc.bars.length} bars`);
      }
  if (problems.length) {
    console.error(problems.join('\n'));
    process.exit(1);
  }
}

// ---------- Videos to build ----------
const videos = [
  { id: 'main', format: 'landscape', ...script.main },
  ...(script.shorts ?? []).map((s) => ({ format: 'portrait', ...s })),
].filter((v) => !only?.length || only.includes(v.id));
// --only thumbnail: re-render just the thumbnails (needs one earlier full build for the timeline).
const thumbsOnly = only?.length === 1 && only[0] === 'thumbnail';

// ---------- TTS ----------
const manifestPath = join(outDir, 'tts-manifest.json');
const items = [];
for (const v of videos)
  v.scenes.forEach((scene, si) =>
    (scene.steps ?? []).forEach((st, ti) =>
      sentences(st.say).forEach((text, k) =>
        items.push({ id: `${v.id}/${si}/${ti}/${k}`, text, tts: toSpeech(text), ...(st.speaker && engineName === 'kokoro' && { voice: voices[st.speaker].voice }) }),
      ),
    ),
  );
writeFileSync(
  manifestPath,
  JSON.stringify(
    {
      engine: { name: engineName, ...Object.fromEntries(Object.entries(profile).filter(([k]) => !['engine', 'postfx', 'label', 'credit', '_tuning'].includes(k))), ...(pronunciationChecks.length && { checks: pronunciationChecks }) },
      cacheDir: join(ROOT, 'video/.tts-cache'),
      items,
    },
    null,
    1,
  ),
);

// PyTorch is pinned so a new release doesn't trigger a multi-GB download on the next build.
// Override with --torch-version or VIDEO_TORCH_VERSION when upgrading on purpose.
const torchVersion = opt('torch-version', process.env.VIDEO_TORCH_VERSION ?? '2.14.1');
function uv(args) {
  const torch = opt('torch', process.env.VIDEO_TORCH ?? 'cuda');
  const index = torch === 'cpu' ? ['--index', 'https://download.pytorch.org/whl/cpu', '--index-strategy', 'unsafe-best-match'] : [];
  // Chatterbox needs Python 3.14 to accept a current PyTorch (it pins torch 2.6 on older Pythons);
  // setuptools<81 still ships pkg_resources, which its watermarker imports.
  const deps =
    engineName === 'chatterbox' && args[0] === 'synth'
      ? ['--python', '3.14', '--with', 'chatterbox-tts', '--with', 'soundfile', '--with', 'num2words', '--with', 'setuptools<81']
      : ['--python', '3.12', '--with', 'kokoro>=0.9.4', '--with', `torch==${torchVersion}`, '--with', 'soundfile', '--with', 'pip', ...index];
  const cmd = ['run', ...deps, 'python', join(ROOT, 'video/tools/tts.py'), ...args];
  const r = spawnSync('uv', cmd, { stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf8', env: { ...process.env, PYTHONWARNINGS: 'ignore' } });
  const lines = `${r.stdout}\n${r.stderr}`.split('\n').filter((l) => /^tts:|Error|error:|Traceback|^\s+File /.test(l));
  if (lines.length) console.log(lines.join('\n'));
  if (r.status !== 0) {
    console.error(r.stderr.split('\n').slice(-25).join('\n'));
    throw new Error('TTS failed');
  }
}

if (!flag('skip-tts') || !JSON.parse(readFileSync(manifestPath, 'utf8')).durations) uv(['synth', manifestPath]);
const durations = JSON.parse(readFileSync(manifestPath, 'utf8')).durations;
if (!durations) throw new Error('No TTS durations; run without --skip-tts');

// ---------- Word timings ----------
// Kokoro times its own tokens (the speech form: "25 billion"); captions and highlights need the
// written words ("25B"). Map one onto the other by position in the letters each side spells.
const letters = (s) => (String(s).toLowerCase().match(/[a-z0-9]/g) ?? []).length;
function wordTimes(text, d, start) {
  const words = plain(text).split(/\s+/).filter(Boolean);
  let acc = 0;
  const spans = words.map((w) => {
    const n = Math.max(1, letters(toSpeech(w)));
    return [acc, (acc += n)];
  });
  const toks = (d.words ?? []).filter(([w]) => letters(w) > 0);
  if (!toks.length) return words.map((w, i) => ({ w, start: start + (d.seconds * spans[i][0]) / acc, end: start + (d.seconds * spans[i][1]) / acc }));
  let tacc = 0;
  const tpos = toks.map(([w, s0, s1]) => [tacc, (tacc += letters(w)), s0, s1]);
  return words.map((w, i) => {
    const a = (spans[i][0] / acc) * tacc;
    const b = (spans[i][1] / acc) * tacc;
    const hit = tpos.filter(([x, y]) => y > a + 1e-6 && x < b - 1e-6);
    const first = hit[0] ?? tpos.find(([x]) => x >= a) ?? tpos.at(-1);
    const last = hit.at(-1) ?? first;
    return { w, start: start + first[2], end: start + Math.max(first[2] + 0.05, last[3]) };
  });
}

// ---------- Pacing checks ----------
// Viewers leave when the picture stops changing or every scene looks the same. These are warnings:
// fix them in the script (split the step, add a visual scene) unless there's a reason not to.
const TEXT_SCENES = new Set(['title', 'statement', 'bullets', 'code', 'compare', 'quote']);
const VISUAL_SCENES = new Set(['diagram', 'figure', 'chart', 'tokens', 'quiz', 'stats', 'budget', 'seriesmap']);
const CHANGES = ['add', 'show', 'values', 'highlight', 'flow', 'draw', 'reveal', 'focus', 'note', 'countdown', 'speaker', 'tag'];
function lintPacing(v) {
  const warn = (msg) => console.warn(`  WARNING (pacing): ${v.id} ${msg}`);
  const firstSaid = v.scenes.flatMap((sc) => sc.steps).find((st) => st.sentences?.length);
  if (firstSaid && firstSaid.start > 6) warn(`first narration starts at ${firstSaid.start.toFixed(1)} s; open with the hook, then the brand`);
  let run = 0;
  let lastVisual = 0;
  v.scenes.forEach((sc, si) => {
    sc.steps.forEach((st, ti) => {
      const changes = ti === 0 || CHANGES.some((k) => st[k] !== undefined);
      if (st.duration > 10 && !changes && TEXT_SCENES.has(sc.type)) warn(`scene ${si + 1} step ${ti + 1} holds one picture for ${st.duration.toFixed(0)} s; split it or reveal something`);
    });
    run = TEXT_SCENES.has(sc.type) ? run + 1 : 0;
    if (run === 3) warn(`scenes ${si - 1}-${si + 1} are three text scenes in a row; break them up with a visual`);
    if (VISUAL_SCENES.has(sc.type) || ['brand', 'outro', 'endcard'].includes(sc.type)) {
      if (sc.start - lastVisual > 90) warn(`${(sc.start - lastVisual).toFixed(0)} s without a visual scene before scene ${si + 1}`);
      lastVisual = sc.start + sc.dur;
    }
  });
}

// ---------- Sound cues ----------
// Quiet interface sounds on the picture's changes: a whoosh into each chapter, a pop per reveal,
// ticks for a countdown, a bell for an answer, key clicks while text types.
function soundCues(v) {
  const cues = [];
  v.scenes.forEach((sc, si) => {
    if (si > 0 && (sc.chapter || sc.type === 'brand')) cues.push({ name: 'whoosh', t: Math.max(0, sc.start - 0.3) });
    sc.steps.forEach((st, ti) => {
      if (ti > 0 && CHANGES.some((k) => st[k] !== undefined) && !st.countdown && !st.reveal && !st.speaker) cues.push({ name: 'pop', t: st.start + 0.03 });
      if (st.countdown) for (let k = 0; k < st.countdown; k++) cues.push({ name: 'tick', t: st.start + k, db: k === st.countdown - 1 ? 2 : 0 });
      if (st.reveal) cues.push({ name: 'ding', t: st.start + 0.05 });
      if (sc.type === 'tokens' && st.show === 'text') {
        const text = sc.text ?? sc.pieces.join('');
        const typeT = sc.typeTime ?? 1.4;
        for (let k = 0; k < text.length; k++) if (text[k] !== ' ') cues.push({ name: 'type', t: st.start + ((k + 0.5) * typeT) / text.length });
      }
      if (sc.type === 'tokens' && (st.show === 'pieces' || st.show === 'ids')) sc.pieces.forEach((_, k) => cues.push({ name: 'type', t: st.start + k * 0.14, db: 4 }));
    });
  });
  return cues;
}

// ---------- Timeline ----------
const frameRound = (s) => Math.ceil(s * FPS - 1e-6) / FPS;
const minStep = (TRANSITION_FRAMES + 6) / FPS;
for (const v of videos) {
  let t = 0;
  v.chapters = [];
  v.clips = [];
  v.captions = [];
  v.scenes.forEach((scene, si) => {
    scene.steps = scene.steps?.length ? scene.steps : [{}];
    if (scene.chapter) v.chapters.push({ title: scene.chapter, start: t });
    scene.chapterTitle = v.chapters.at(-1)?.title ?? '';
    scene.start = t;
    scene.steps.forEach((st, ti) => {
      st.start = t;
      let offset = 0.15;
      const said = sentences(st.say);
      st.sentences = [];
      said.forEach((text, k) => {
        const d = durations[`${v.id}/${si}/${ti}/${k}`];
        v.clips.push({ wav: d.wav, start: t + offset });
        v.captions.push({ text: plain(text), start: t + offset, end: t + offset + d.seconds });
        st.sentences.push({ start: t + offset, end: t + offset + d.seconds, words: wordTimes(text, d, t + offset) });
        offset += d.seconds + (k < said.length - 1 ? SENTENCE_GAP : 0);
      });
      let dur = said.length ? offset + STEP_TAIL : (st.hold ?? st.countdown ?? 2.5);
      if (st.countdown) dur = Math.max(dur, st.countdown + 0.4); // the ring runs out, then a beat
      if (ti === scene.steps.length - 1) dur += SCENE_TAIL;
      if (v.format === 'portrait' && si === v.scenes.length - 1 && ti === scene.steps.length - 1) dur += SHORT_END;
      if (st.minDuration) dur = Math.max(dur, st.minDuration); // e.g. a 20 s end card
      st.duration = frameRound(Math.max(dur, minStep));
      t += st.duration;
    });
    scene.dur = t - scene.start;
  });
  v.total = t;
  lintPacing(v);
  v.sfx = fx.sfx ? soundCues(v) : null;
  if (v.chapters.length && v.chapters[0].start > 0) v.chapters[0].start = 0;
  // YouTube only shows chapters when there are 3+, the first is 0:00, and each lasts 10 s or more.
  if (v.id === 'main') {
    if (v.chapters.length < 3) console.warn('  WARNING: fewer than 3 chapters; YouTube will not show chapters');
    v.chapters.forEach((c, i) => {
      const end = v.chapters[i + 1]?.start ?? v.total;
      if (end - c.start < 10) console.warn(`  WARNING: chapter "${c.title}" is under 10 s`);
    });
  }
  console.log(`${v.id}: ${fmtTime(v.total)} (${v.total.toFixed(1)} s), ${v.clips.length} sentences, ${v.chapters.length} chapters`);
  if (v.format === 'portrait' && v.total > 60) console.warn(`  WARNING: ${v.id} is over 60 s`);
}

// ---------- Assets ----------
const fileUrl = (p) => pathToFileURL(join(ROOT, p)).href;
const baseAssets = (format) => ({
  logo: fileUrl('src/assets/brand/logo-mark.png'),
  banner: fileUrl(format === 'portrait' ? 'src/assets/brand/banner-mobile.png' : 'src/assets/brand/banner1.png'),
});

// ---------- Render ----------
const pwDir = join(homedir(), '.cache/ms-playwright');
const executablePath = existsSync(pwDir)
  ? readdirSync(pwDir).filter((d) => d.startsWith('chromium-')).map((d) => join(pwDir, d, 'chrome-linux64/chrome')).find(existsSync)
  : undefined;
const stageUrl = pathToFileURL(join(ROOT, 'video/tools/stage/stage.html')).href;
// One browser per worker: pages that share a browser are background tabs and stop painting.
const launch = () => puppeteer.launch({ executablePath, args: ['--no-sandbox', '--font-render-hinting=none'], protocolTimeout: 600000 });

async function openStage(browser, format) {
  const page = await browser.newPage();
  const [w, h] = SIZES[format];
  await page.setViewport({ width: w, height: h, deviceScaleFactor: 1 });
  await page.goto(stageUrl, { waitUntil: 'networkidle0' });
  return page;
}

// What the stage needs to know about a scene's timing, relative to the scene's start.
function sceneSpec(v, scene, si) {
  const rel = (x) => +(x - scene.start).toFixed(4);
  const timing = {
    dur: scene.dur,
    steps: scene.steps.map((st) => ({
      t0: rel(st.start),
      dur: st.duration,
      say: plain(String(st.say ?? '')),
      speaker: st.speaker,
      speakerLabel: st.speaker ? (voices[st.speaker].label ?? st.speaker) : undefined,
      sentences: (st.sentences ?? []).map((x) => ({ start: rel(x.start), end: rel(x.end), words: x.words.map((w) => ({ w: w.w, start: rel(w.start), end: rel(w.end) })) })),
    })),
  };
  const assets = { ...baseAssets(v.format), ...prepareScene(ROOT, scene, front) };
  return {
    format: v.format,
    scene,
    assets,
    timing,
    chapter: scene.chapterTitle,
    progress: v.format === 'landscape' && scene.type !== 'endcard',
    chrome: !['brand', 'endcard'].includes(scene.type),
    instant: v.format === 'portrait' && si === 0, // a Short's first frame already shows its hook
    fx: { drift: fx.drift, wordSync: fx.wordSync, captions: v.format === 'portrait' && fx.shortsCaptions },
  };
}

// Render a contiguous run of scenes into one video-only chunk. Every frame is computed from time;
// when the stage reports the same key as the previous frame, the previous screenshot is reused.
async function renderChunk(v, group, gi, report) {
  const browser = await launch();
  const page = await openStage(browser, v.format);
  const file = join(outDir, `${v.id}.chunk${gi}.mp4`);
  const ff = spawn('ffmpeg', [
    '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'png', '-i', '-',
    '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', fast ? 'veryfast' : 'slow', '-crf', '18',
    '-g', String(FPS * 2), '-threads', '4', '-r', String(FPS), file,
  ], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('close', (c) => (c === 0 ? res() : rej(new Error(`ffmpeg chunk ${gi} failed`)))));
  for (const { si, f0, n } of group) {
    const scene = v.scenes[si];
    await page.evaluate((spec) => window.stage.load(spec), sceneSpec(v, scene, si));
    let prevKey = null;
    let shot = null;
    for (let i = 0; i < n; i++) {
      const gT = (f0 + i) / FPS;
      const key = await page.evaluate((a, b, c) => window.stage.frame(a, b, c), gT - scene.start, gT, gT / v.total);
      if (key !== prevKey) {
        shot = await page.screenshot({ type: 'png', optimizeForSpeed: true });
        prevKey = key;
        report.shots++;
      }
      if (!ff.stdin.write(shot)) await once(ff.stdin, 'drain');
      report.frames++;
    }
  }
  ff.stdin.end();
  await done;
  await browser.close();
  return file;
}

for (const v of videos) {
  if (flag('no-video')) break;
  const started = Date.now();
  const totalFrames = Math.round(v.total * FPS);
  const spans = v.scenes.map((scene, si) => {
    const f0 = Math.round(scene.start * FPS);
    const f1 = si + 1 < v.scenes.length ? Math.round(v.scenes[si + 1].start * FPS) : totalFrames;
    return { si, f0, n: f1 - f0 };
  });
  // Contiguous groups of about equal length, one per worker.
  const groups = [];
  let cur = [];
  let acc = 0;
  for (const sp of spans) {
    cur.push(sp);
    acc += sp.n;
    if (acc >= totalFrames / WORKERS && groups.length < WORKERS - 1) {
      groups.push(cur);
      cur = [];
      acc = 0;
    }
  }
  if (cur.length) groups.push(cur);
  const report = { frames: 0, shots: 0 };
  const ticker = setInterval(() => process.stdout.write(`\r  ${v.id}: ${report.frames}/${totalFrames} frames (${report.shots} rendered) on ${groups.length} workers`), 1000);
  let chunks;
  try {
    chunks = await Promise.all(groups.map((g, gi) => renderChunk(v, g, gi, report)));
  } finally {
    clearInterval(ticker);
  }
  console.log(`\r  ${v.id}: ${report.frames} frames (${report.shots} rendered) in ${((Date.now() - started) / 1000).toFixed(0)} s on ${groups.length} workers`);
  const list = join(outDir, `${v.id}.chunks.txt`);
  writeFileSync(list, chunks.map((c) => `file '${c}'`).join('\n') + '\n');

  // Audio: narration, plus the music bed and sound effects when the script asks for them.
  const timelinePath = join(outDir, `${v.id}.timeline.json`);
  const wav = join(outDir, `${v.id}.narration.wav`);
  writeFileSync(timelinePath, JSON.stringify({ out: wav, totalSeconds: v.total, clips: v.clips }));
  uv(['mix', timelinePath]);
  const musicFile = fx.music === 'synth' ? join(outDir, `${v.id}.music.wav`) : fx.music ? resolve(videoDir, fx.music) : null;
  const sfxFile = v.sfx ? join(outDir, `${v.id}.sfx.wav`) : null;
  if (fx.music === 'synth' || v.sfx) {
    const bedPath = join(outDir, `${v.id}.bed.json`);
    const seed = [...`${slug}/${v.id}`].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
    writeFileSync(bedPath, JSON.stringify({ seconds: v.total, seed, music: fx.music === 'synth' ? 'synth' : null, musicOut: musicFile, sfx: v.sfx, sfxOut: sfxFile }));
    uv(['bed', bedPath]);
  }
  const mp4 = join(outDir, `${v.id === 'main' ? slug : `${slug}--${v.id}`}${tag ? `--${tag}` : ''}.mp4`);
  const postfx = profile.postfx ? `${profile.postfx},` : ''; // static voice polish (EQ, de-esser) from the profile
  // Loudness: one static gain to -14 LUFS on the narration plus a peak limiter at -2.5 dBFS, which
  // leaves room for AAC overshoot (target: true peak under -1 dB). (ffmpeg's loudnorm falls back to
  // dynamic gain when it can't stay linear, and that pumps the TTS noise floor up ~45 dB in every
  // pause, which is audible as hiss and grinding. A fixed gain can't do that.) The music bed also
  // gets one static gain, `musicGap` dB under the narration; the effects are pre-levelled.
  const lufs = (file, filter = '') => {
    const probe = spawnSync('ffmpeg', ['-hide_banner', '-nostats', '-i', file, '-af', `${filter}loudnorm=print_format=json`, '-f', 'null', '-'], { encoding: 'utf8' });
    return Number(JSON.parse(probe.stderr.slice(probe.stderr.lastIndexOf('{'))).input_i);
  };
  const gain = -14 - lufs(wav, postfx);
  const limiter = `alimiter=limit=${Math.pow(10, -2.5 / 20).toFixed(4)}:attack=5:release=50:level=false`;
  const inputs = ['-i', wav];
  const st = 'aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo';
  const graph = [`[1:a]${postfx}volume=${gain.toFixed(2)}dB,${st}[n]`];
  const mixIn = ['[n]'];
  if (musicFile) {
    const musicGain = -14 - fx.musicGap - lufs(musicFile);
    inputs.push(...(fx.music === 'synth' ? [] : ['-stream_loop', '-1']), '-i', musicFile);
    const fade = fx.music === 'synth' ? '' : `,afade=t=in:d=2,afade=t=out:st=${Math.max(0, v.total - 4).toFixed(2)}:d=4`;
    graph.push(`[${inputs.filter((x) => x === '-i').length}:a]${st},atrim=0:${v.total.toFixed(3)}${fade},volume=${musicGain.toFixed(2)}dB[m]`);
    mixIn.push('[m]');
  }
  if (sfxFile) {
    inputs.push('-i', sfxFile);
    graph.push(`[${inputs.filter((x) => x === '-i').length}:a]${st}[s]`);
    mixIn.push('[s]');
  }
  graph.push(`${mixIn.join('')}${mixIn.length > 1 ? `amix=inputs=${mixIn.length}:normalize=0:duration=first,` : ''}${limiter}[a]`);
  const ff = spawnSync(
    'ffmpeg',
    [
      '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, ...inputs,
      '-filter_complex', graph.join(';'), '-map', '0:v', '-map', '[a]', '-c:v', 'copy',
      '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-t', v.total.toFixed(3), '-movflags', '+faststart', mp4,
    ],
    { stdio: 'inherit' },
  );
  if (ff.status !== 0) throw new Error(`ffmpeg failed for ${v.id}`);
  for (const c of chunks) rmSync(c, { force: true });
  v.mp4 = mp4;
  // Contact sheet for a quick visual check of the whole video.
  spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', mp4, '-vf', `fps=${(16 / v.total).toFixed(5)},scale=${v.format === 'portrait' ? 270 : 480}:-1,tile=4x4`, '-frames:v', '1', join(outDir, `${v.id}-contact.png`)]);
  console.log(`  -> ${relative(ROOT, mp4)}`);
}

// ---------- Thumbnails ----------
// `thumbnail` is one design or a list of up to 3 for YouTube's "Test & compare": thumbnail.png,
// thumbnail-b.png, thumbnail-c.png.
const thumbs = [].concat(script.thumbnail ?? []);
if (thumbs.length && !flag('no-video') && !tag && (thumbsOnly || videos.some((v) => v.id === 'main'))) {
  const browser = await launch();
  const page = await openStage(browser, 'thumb');
  mkdirSync(join(kitDir, 'main'), { recursive: true });
  for (const [i, th] of thumbs.slice(0, 3).entries()) {
    const assets = { ...baseAssets('thumb'), ...(th.diagram ? { diagram: fileUrl(`public/diagrams/${th.diagram}-dark.svg`) } : {}) };
    await page.evaluate((spec) => window.stage.load(spec), { format: 'thumb', scene: { type: 'thumb', steps: [{}], ...th }, assets, progress: false });
    await page.evaluate(() => window.stage.set(0));
    await page.screenshot({ path: join(kitDir, 'main', `thumbnail${i ? `-${'abc'[i]}` : ''}.png`) });
  }
  await browser.close();
}

if (thumbsOnly) {
  console.log(`Thumbnails: ${relative(ROOT, join(kitDir, 'main'))}/thumbnail*.png`);
  process.exit(0);
}

// ---------- Captions (SRT) ----------
function srt(captions) {
  const cues = [];
  for (const c of captions) {
    // Split long sentences into cues of <= 2 lines x 42 chars, timed by character share.
    const words = c.text.split(' ');
    const chunks = [];
    let cur = '';
    for (const w of words) {
      if ((cur + ' ' + w).trim().length > 84 && cur) {
        chunks.push(cur);
        cur = w;
      } else cur = (cur + ' ' + w).trim();
    }
    if (cur) chunks.push(cur);
    const total = chunks.reduce((s, x) => s + x.length, 0);
    let t = c.start;
    for (const ch of chunks) {
      const d = ((c.end - c.start) * ch.length) / total;
      let line = ch;
      if (ch.length > 42) {
        const mid = ch.lastIndexOf(' ', Math.ceil(ch.length / 2) + 6);
        line = `${ch.slice(0, mid)}\n${ch.slice(mid + 1)}`;
      }
      cues.push({ start: t, end: t + d, text: line });
      t += d;
    }
  }
  return cues.map((q, i) => `${i + 1}\n${fmtTime(q.start, true)} --> ${fmtTime(q.end, true)}\n${q.text}\n`).join('\n');
}

// ---------- YouTube cards ----------
// Wherever the narration names another part of this series ("Part 2"), a card can point to that
// video at that moment. The API can't add cards, so they go in the checklist with their times.
function youtubeCards(v) {
  const parts = seriesParts(ROOT, front.series);
  const uploadedId = (partSlug) => {
    for (const d of readdirSync(join(ROOT, 'video'))) {
      const sp = join(ROOT, 'video', d, 'script.yaml');
      const up = join(ROOT, 'video', d, 'youtube', 'uploaded.json');
      if (!existsSync(sp) || !existsSync(up)) continue;
      if (YAML.parse(readFileSync(sp, 'utf8')).slug !== partSlug) continue;
      const id = JSON.parse(readFileSync(up, 'utf8')).videos?.main?.id;
      if (id) return id;
    }
    return null;
  };
  const cards = [];
  const seen = new Set();
  for (const c of v.captions)
    for (const m of c.text.matchAll(/\bParts? (\d+)/g)) {
      const n = Number(m[1]);
      const part = parts.find((p) => p.n === n);
      if (!part || n === front.seriesOrder || seen.has(n) || cards.length >= 5) continue;
      seen.add(n);
      const id = uploadedId(part.slug);
      cards.push({ at: c.start, label: `Part ${n}: ${part.title}`, target: id ? `https://youtu.be/${id}` : `not on YouTube yet (use the ${series.name} playlist, or skip)` });
    }
  return cards;
}

// ---------- Upload kit ----------
if (tag) {
  console.log('Tagged build: upload kit not updated.');
  process.exit(0);
}
const yt = script.youtube ?? {};
const musicCredit = fx.music === 'synth' ? '\nMusic: generated by VidyalAI.' : fx.musicCredit ? `\nMusic: ${fx.musicCredit}` : '';
const credit = `${profile.credit ?? 'Narration: AI-generated voice (Kokoro open-weight text-to-speech). Script, diagrams and sources by VidyalAI.'}${musicCredit}\n© ${new Date().getFullYear()} VidyalAI. All rights reserved.`;
for (const v of videos) {
  const dir = join(kitDir, v.id === 'main' ? 'main' : join('shorts', v.id));
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'captions.en.srt'), srt(v.captions));
  if (v.id === 'main') {
    const chapters = v.chapters.map((c) => `${fmtTime(c.start)} ${c.title}`).join('\n');
    writeFileSync(join(dir, 'chapters.txt'), chapters + '\n');
    writeFileSync(join(dir, 'title.txt'), [yt.title, '', 'Alternatives:', ...(yt.altTitles ?? []).map((t) => `- ${t}`)].join('\n') + '\n');
    const description = [
      yt.summary?.trim(),
      '',
      `📖 Read the full article (with every number and source): ${articleUrl}`,
      `📚 The whole ${series.name} series: ${SITE}/series/${series.slug}/`,
      '',
      'Chapters',
      chapters,
      '',
      'Sources and further reading',
      ...sources.map((s) => `• ${s.title}: ${s.url}`),
      '',
      credit,
      '',
      (yt.hashtags ?? []).join(' '),
    ].join('\n');
    writeFileSync(join(dir, 'description.txt'), description + '\n');
    writeFileSync(join(dir, 'tags.txt'), (yt.tags ?? []).join(', ') + '\n');
    if (yt.pinnedComment) writeFileSync(join(dir, 'pinned-comment.txt'), yt.pinnedComment.trim() + '\n');
    // Scene index for `upload.mjs analytics`, which maps the audience-retention curve onto scenes.
    const sceneIndex = v.scenes.map((sc, i) => ({
      scene: i + 1,
      type: sc.type,
      chapter: sc.chapterTitle || undefined,
      label: plain(String(sc.heading ?? sc.text ?? sc.question ?? sc.quote ?? sc.title ?? sc.type)).slice(0, 80),
      start: +sc.start.toFixed(2),
      end: +(sc.start + sc.dur).toFixed(2),
    }));
    writeFileSync(join(dir, 'scenes.json'), JSON.stringify({ total: +v.total.toFixed(2), scenes: sceneIndex }, null, 1) + '\n');
    v.cards = youtubeCards(v);
    writeFileSync(join(dir, 'cards.txt'), v.cards.length ? v.cards.map((c) => `${fmtTime(c.at)}  ${c.label}  ->  ${c.target}`).join('\n') + '\n' : '');
    if (description.length > 5000) console.warn(`WARNING: description is ${description.length} chars (YouTube max 5,000)`);
    if ((yt.tags ?? []).join(',').length > 500) console.warn('WARNING: tags exceed 500 characters');
    if ((yt.title ?? '').length > 100) console.warn('WARNING: title exceeds 100 characters');
    v.kit = { chapters };
  } else {
    writeFileSync(join(dir, 'title.txt'), `${v.title}\n`);
    writeFileSync(
      join(dir, 'description.txt'),
      [v.description?.trim(), '', 'Watch the full breakdown: tap the related video link on this Short.', `Read the full article: ${articleUrl.replace(/^https:\/\//, '')}`, '', credit, '', (v.hashtags ?? yt.hashtags ?? []).join(' ')].join('\n') + '\n',
    );
  }
}

// Checklist covering every video in the script (not only the ones built in this run).
const allIds = ['main', ...(script.shorts ?? []).map((s) => s.id)];
// Where the end card sits, for the end screen step (from this run's timeline, or the last known one).
const mainVideo = videos.find((v) => v.id === 'main');
const endcard = mainVideo?.scenes.find((sc) => sc.type === 'endcard');
// YouTube Studio's end screen editor shows times as m:ss:ff (ff = frame number, 30 fps).
const studioTime = (sec) => {
  const f = Math.round(sec * FPS);
  return `${Math.floor(f / FPS / 60)}:${String(Math.floor(f / FPS) % 60).padStart(2, '0')}:${String(f % FPS).padStart(2, '0')}`;
};
const endcardHelp = endcard?.teaser
  ? `the last ${(mainVideo.total - endcard.start).toFixed(0)} s (from ${studioTime(endcard.start)} in Studio's m:ss:frames format) is the end card. Choose
     **Add element → Subscribe**, drag it into the dashed circle on the right, resize it to fill the circle, and set it
     to run for the whole end card (drag the element's bar in the timeline if the typed time turns red). YouTube requires at least one video or playlist
     element, so also add **Playlist** → the series playlist and drag it into the dashed frame on the left, resized to
     fill it. **When the next video is public**, come back to this video's end screen and replace the playlist with
     **Video → Choose specific video** (same position and size). End screens can be edited any time; no re-upload needed.`
  : endcard
  ? `the last ${(mainVideo.total - endcard.start).toFixed(0)} s (from ${studioTime(endcard.start)} in Studio's m:ss:frames format) is a blank end card. Choose
     **Add element → Video** (Best for viewer, or the next video in the series) and **Subscribe**. Drag the video element
     into the dashed 16:9 frame on the left and the Subscribe element into the dashed circle on the right, resize them to
     fill the frames, and set both to run for the whole end card.`
  : 'Subscribe + "Best for viewer" over the last 15-20 s; check that the elements don\'t cover on-screen text.';
// For upload.mjs, which prints the end-card start next to the end-screen step.
if (endcard) writeFileSync(join(kitDir, 'main', 'endcard.txt'), `${studioTime(endcard.start)}\n`);
const mp4Name = (id) => `${id === 'main' ? slug : `${slug}--${id}`}.mp4`;
const checklist = `# YouTube upload checklist: ${yt.title ?? slug}

Videos are in \`video/${slug}/out/\` (not committed; rebuild with \`node video/tools/build.mjs video/${slug}\`).
Everything to paste is in this folder. Upload the main video first: the Shorts link to it.

## Automated alternative

\`video/tools/upload.mjs\` does every step below that the YouTube Data API supports: upload, title,
description, tags, category, language, audience, AI disclosure, thumbnail, captions and playlist, all as
private, then publishing and posting the comment. Progress is saved in \`uploaded.json\`, so re-runs resume.

\`\`\`sh
node video/tools/upload.mjs upload  video/${slug} --dry-run   # check what will be sent
node video/tools/upload.mjs upload  video/${slug}             # everything up as private
node video/tools/upload.mjs status  video/${slug}             # processing state
node video/tools/upload.mjs publish video/${slug}             # public (main first) + post the comment
\`\`\`

Still by hand (the API can't): the main video's end screen, each Short's Related video, and pinning the
comment. The script prints Studio links for each. One-time setup, the release steps and daily limits are
in \`video/UPLOADING.md\`.

## Once per channel

- **Upload defaults** (Studio → Settings → Upload defaults → Advanced settings), so API uploads match the
  manual steps: comments **Hold potentially inappropriate comments → Strict**, **Automatic chapters**,
  **Featured places** and **Automatic concepts** unchecked, **Caption certification** "This content has never
  aired on television in the U.S.", **Category** Education, **Language** English.

- **Verify the channel by phone** at https://www.youtube.com/verify. Custom thumbnails need it (Studio → Settings →
  Channel → Feature eligibility shows what is unlocked). Chapters need the channel to have no active strikes; if they
  don't appear, apply for Advanced features on the same page.
- Create the playlist **${yt.playlist ?? series.name}** (it can also be created from the upload form).

## Main video: \`out/${mp4Name('main')}\`

1. YouTube Studio → **Create → Upload videos** → select the MP4.
2. **Details**
   - **Title**: first line of \`main/title.txt\`.
   - **Description**: paste \`main/description.txt\`. Its timestamps (starting at 00:00) become the chapters.
   - **Thumbnail**: upload \`main/thumbnail.png\` (1280×720 PNG).${thumbs.length > 1 ? `
     **Test & compare** (Studio → the video's Details → Thumbnail → Test & compare): upload
     ${thumbs.slice(0, 3).map((_, i) => `\`main/thumbnail${i ? `-${'abc'[i]}` : ''}.png\``).join(', ')}. YouTube shows each to a share of viewers
     and keeps the one with the most watch time. It needs Advanced features on the channel and is set from the web Studio
     only; the API can't start a test.` : ''}
   - **Playlists**: ${yt.playlist ?? series.name}.
   - **Audience**: "No, it's not made for kids".
   - **Attributes → AI use**: **Yes**. The narration is an AI-generated voice. YouTube's policy only requires
     disclosure for realistic depictions of real people or events, but it labels AI-generated non-music audio in the
     expanded description, which matches the credit line already in the description.
   - **Show more**: **Tags**: paste \`main/tags.txt\`. **Language**: English. **Caption certification**: "This content
     has never aired on television in the U.S." **Category**: Education.
3. **Video elements**
   - **Add subtitles → Upload file → With timing** → \`main/captions.en.srt\`.
   - **Add end screen**: ${endcardHelp}
   - **Add cards**${mainVideo?.cards?.length ? `: one per line of \`main/cards.txt\` (time, the part it names, and the video to link):
${mainVideo.cards.map((c) => `     - ${fmtTime(c.at)} ${c.label} → ${c.target}`).join('\n')}
     Studio → **Cards → Video**, then drag each card to its time. Up to 5 per video.` : ': none (the narration names no other part of the series).'}
4. **Checks**: wait for the copyright check to finish.
5. **Visibility**: Public, or schedule. After publishing, pin the comment from \`main/pinned-comment.txt\`.

## Shorts

Vertical videos up to 3 minutes are published as Shorts automatically. Links in Shorts descriptions and comments are
not clickable, so each Short links to the main video with **Related video** instead.

${allIds
  .filter((id) => id !== 'main')
  .map(
    (id) => `### \`out/${mp4Name(id)}\`

1. **Create → Upload videos** → select the MP4.
2. **Title**: \`shorts/${id}/title.txt\`. **Description**: \`shorts/${id}/description.txt\`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: ${yt.playlist ?? series.name}.
3. **Video elements → Add subtitles → Upload file → With timing** → \`shorts/${id}/captions.en.srt\`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.`,
  )
  .join('\n\n')}
`;
writeFileSync(join(kitDir, 'UPLOAD.md'), checklist);
console.log(`Upload kit: ${relative(ROOT, kitDir)}/`);
