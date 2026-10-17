#!/usr/bin/env node
// Article-to-video build. Turns video/<slug>/script.yaml into narrated MP4s plus a YouTube upload kit.
//
//   node video/tools/build.mjs video/<slug> [--only main,short-id] [--skip-tts] [--torch cuda|cpu]
//                                           [--voice af_heart] [--no-video] [--fast]
//
// Pipeline: script.yaml -> TTS per sentence (tts.py, Kokoro) -> timeline from real audio durations
// -> frames rendered by video/tools/stage/stage.html in headless Chromium -> ffmpeg (concat
// demuxer, so held frames cost nothing) -> captions, chapters, description, thumbnail, checklist.

import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync, copyFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import YAML from 'yaml';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(join(ROOT, 'package.json'));
const puppeteer = require('puppeteer');

const FPS = 30;
const TRANSITION_FRAMES = 14; // ~0.47 s entrance animation per step
const SENTENCE_GAP = 0.3; // silence between sentences within a step (s)
const STEP_TAIL = 0.45; // silence after a step's narration (s)
const SCENE_TAIL = 0.4; // extra silence at the end of each scene (s)
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

// ---------- TTS ----------
const manifestPath = join(outDir, 'tts-manifest.json');
const items = [];
for (const v of videos)
  v.scenes.forEach((scene, si) =>
    (scene.steps ?? []).forEach((st, ti) =>
      sentences(st.say).forEach((text, k) => items.push({ id: `${v.id}/${si}/${ti}/${k}`, text, tts: toSpeech(text) })),
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

function uv(args) {
  const torch = opt('torch', process.env.VIDEO_TORCH ?? 'cuda');
  const index = torch === 'cpu' ? ['--index', 'https://download.pytorch.org/whl/cpu', '--index-strategy', 'unsafe-best-match'] : [];
  // Chatterbox needs Python 3.14 to accept a current PyTorch (it pins torch 2.6 on older Pythons);
  // setuptools<81 still ships pkg_resources, which its watermarker imports.
  const deps =
    engineName === 'chatterbox' && args[0] === 'synth'
      ? ['--python', '3.14', '--with', 'chatterbox-tts', '--with', 'soundfile', '--with', 'num2words', '--with', 'setuptools<81']
      : ['--python', '3.12', '--with', 'kokoro>=0.9.4', '--with', 'soundfile', '--with', 'pip', ...index];
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
    scene.start = t;
    scene.steps.forEach((st, ti) => {
      st.start = t;
      let offset = 0.15;
      const said = sentences(st.say);
      said.forEach((text, k) => {
        const d = durations[`${v.id}/${si}/${ti}/${k}`];
        v.clips.push({ wav: d.wav, start: t + offset });
        v.captions.push({ text: plain(text), start: t + offset, end: t + offset + d.seconds });
        offset += d.seconds + (k < said.length - 1 ? SENTENCE_GAP : 0);
      });
      let dur = said.length ? offset + STEP_TAIL : (st.hold ?? 2.5);
      if (ti === scene.steps.length - 1) dur += SCENE_TAIL;
      if (st.minDuration) dur = Math.max(dur, st.minDuration); // e.g. a 20 s end card
      st.duration = frameRound(Math.max(dur, minStep));
      t += st.duration;
    });
  });
  v.total = t;
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
function diagramAsset(name) {
  const p = `public/diagrams/${name}-dark.svg`;
  const svg = readFileSync(join(ROOT, p), 'utf8');
  const [, , w, h] = /viewBox="([^"]+)"/.exec(svg)[1].trim().split(/[\s,]+/).map(Number);
  return { diagram: fileUrl(p), diagramSize: { w, h } };
}
const baseAssets = (format) => ({
  logo: fileUrl('src/assets/brand/logo-mark.png'),
  banner: fileUrl(format === 'portrait' ? 'src/assets/brand/banner-mobile.png' : 'src/assets/brand/banner1.png'),
});

// ---------- Render ----------
const pwDir = join(homedir(), '.cache/ms-playwright');
const executablePath = existsSync(pwDir)
  ? readdirSync(pwDir).filter((d) => d.startsWith('chromium-')).map((d) => join(pwDir, d, 'chrome-linux64/chrome')).find(existsSync)
  : undefined;
const browser = await puppeteer.launch({ executablePath, args: ['--no-sandbox', '--font-render-hinting=none'] });
const stageUrl = pathToFileURL(join(ROOT, 'video/tools/stage/stage.html')).href;

async function openStage(format) {
  const page = await browser.newPage();
  const [w, h] = SIZES[format];
  await page.setViewport({ width: w, height: h, deviceScaleFactor: 1 });
  await page.goto(stageUrl, { waitUntil: 'networkidle0' });
  return page;
}

const animates = (scene, st, ti) => ti === 0 || st.add || st.focus || st.note || st.animate;

for (const v of videos) {
  if (flag('no-video')) break;
  const framesDir = join(outDir, 'frames', v.id);
  rmSync(framesDir, { recursive: true, force: true });
  mkdirSync(framesDir, { recursive: true });
  const page = await openStage(v.format);
  const concat = ['ffconcat version 1.0'];
  let n = 0;
  let chapterTitle = '';
  const started = Date.now();
  for (const [si, scene] of v.scenes.entries()) {
    if (scene.chapter) chapterTitle = scene.chapter;
    const assets = { ...baseAssets(v.format), ...(scene.diagram ? diagramAsset(scene.diagram) : {}) };
    await page.evaluate(
      (spec) => window.stage.load(spec),
      { format: v.format, scene, assets, chapter: chapterTitle, progress: v.format === 'landscape' && scene.type !== 'endcard', chrome: !['brand', 'endcard'].includes(scene.type) },
    );
    for (const [ti, st] of scene.steps.entries()) {
      const frames = animates(scene, st, ti) ? TRANSITION_FRAMES : 1;
      for (let f = 1; f <= frames; f++) {
        const time = st.start + (f - 1) / FPS;
        await page.evaluate((a, b, c) => window.stage.set(a, b, c), ti, f / frames, time / v.total);
        const file = join(framesDir, `${String(n++).padStart(5, '0')}.png`);
        await page.screenshot({ path: file, optimizeForSpeed: true });
        const hold = f < frames ? 1 / FPS : st.duration - (frames - 1) / FPS;
        concat.push(`file '${file}'`, `duration ${hold.toFixed(6)}`);
      }
    }
    process.stdout.write(`\r  ${v.id}: rendered scene ${si + 1}/${v.scenes.length} (${n} frames)`);
  }
  concat.push(concat[concat.length - 2]); // concat demuxer needs the last file repeated
  writeFileSync(join(outDir, `${v.id}.ffconcat`), concat.join('\n'));
  await page.close();
  console.log(` in ${((Date.now() - started) / 1000).toFixed(0)} s`);

  // Narration track, then mux and normalize loudness to YouTube's -14 LUFS.
  const timelinePath = join(outDir, `${v.id}.timeline.json`);
  const wav = join(outDir, `${v.id}.narration.wav`);
  writeFileSync(timelinePath, JSON.stringify({ out: wav, totalSeconds: v.total, clips: v.clips }));
  uv(['mix', timelinePath]);
  const mp4 = join(outDir, `${v.id === 'main' ? slug : `${slug}--${v.id}`}${tag ? `--${tag}` : ''}.mp4`);
  const postfx = profile.postfx ? `${profile.postfx},` : ''; // static voice polish (EQ, de-esser) from the profile
  // Loudness: one static gain to -14 LUFS plus a peak limiter at -2.5 dBFS, which leaves room for AAC overshoot (target: true peak under -1 dB). (ffmpeg's loudnorm falls back to
  // dynamic gain when it can't stay linear, and that pumps the TTS noise floor up ~45 dB in every
  // pause, which is audible as hiss and grinding. A fixed gain can't do that.)
  const probe = spawnSync('ffmpeg', ['-hide_banner', '-nostats', '-i', wav, '-af', `${postfx}loudnorm=print_format=json`, '-f', 'null', '-'], { encoding: 'utf8' });
  const measured = JSON.parse(probe.stderr.slice(probe.stderr.lastIndexOf('{')));
  const gain = -14 - Number(measured.input_i);
  const loudnorm = `${postfx}volume=${gain.toFixed(2)}dB,alimiter=limit=${Math.pow(10, -2.5 / 20).toFixed(4)}:attack=5:release=50:level=false`;
  const ff = spawnSync(
    'ffmpeg',
    [
      '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', join(outDir, `${v.id}.ffconcat`), '-i', wav,
      '-vf', `fps=${FPS},format=yuv420p`, '-c:v', 'libx264', '-preset', fast ? 'veryfast' : 'slow', '-crf', '18',
      '-g', String(FPS * 2), '-af', loudnorm, '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
      '-t', v.total.toFixed(3), '-movflags', '+faststart', mp4,
    ],
    { stdio: 'inherit' },
  );
  if (ff.status !== 0) throw new Error(`ffmpeg failed for ${v.id}`);
  v.mp4 = mp4;
  // Contact sheet for a quick visual check of the whole video.
  spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', mp4, '-vf', `fps=${(16 / v.total).toFixed(5)},scale=${v.format === 'portrait' ? 270 : 480}:-1,tile=4x4`, '-frames:v', '1', join(outDir, `${v.id}-contact.png`)]);
  console.log(`  -> ${relative(ROOT, mp4)}`);
}

// ---------- Thumbnail ----------
if (script.thumbnail && !flag('no-video') && !tag && videos.some((v) => v.id === 'main')) {
  const page = await openStage('thumb');
  const assets = { ...baseAssets('thumb'), ...(script.thumbnail.diagram ? diagramAsset(script.thumbnail.diagram) : {}) };
  await page.evaluate((spec) => window.stage.load(spec), { format: 'thumb', scene: { type: 'thumb', steps: [{}], ...script.thumbnail }, assets, progress: false });
  await page.evaluate(() => window.stage.set(0, 1, 0));
  mkdirSync(join(kitDir, 'main'), { recursive: true });
  await page.screenshot({ path: join(kitDir, 'main', 'thumbnail.png') });
  await page.close();
}
await browser.close();

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

// ---------- Upload kit ----------
if (tag) {
  console.log('Tagged build: upload kit not updated.');
  process.exit(0);
}
const yt = script.youtube ?? {};
const credit = `${profile.credit ?? 'Narration: AI-generated voice (Kokoro open-weight text-to-speech). Script, diagrams and sources by VidyalAI.'}\n© ${new Date().getFullYear()} VidyalAI. All rights reserved.`;
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
      `📚 The whole LLM System Design series: ${SITE}/series/llm-system-design/`,
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
     to run for the whole end card (drag the element's bar in the timeline if the typed time turns red). The left panel shows the next episode as "coming soon", so leave it uncovered for
     now. **When that video is public**, come back to this video's end screen, add **Video → Choose specific video**,
     and place it over the panel (same position and size). End screens can be edited any time; no re-upload needed.`
  : endcard
  ? `the last ${(mainVideo.total - endcard.start).toFixed(0)} s (from ${studioTime(endcard.start)} in Studio's m:ss:frames format) is a blank end card. Choose
     **Add element → Video** (Best for viewer, or the next video in the series) and **Subscribe**. Drag the video element
     into the dashed 16:9 frame on the left and the Subscribe element into the dashed circle on the right, resize them to
     fill the frames, and set both to run for the whole end card.`
  : 'Subscribe + "Best for viewer" over the last 15-20 s; check that the elements don\'t cover on-screen text.';
const mp4Name = (id) => `${id === 'main' ? slug : `${slug}--${id}`}.mp4`;
const checklist = `# YouTube upload checklist: ${yt.title ?? slug}

Videos are in \`video/${slug}/out/\` (not committed; rebuild with \`node video/tools/build.mjs video/${slug}\`).
Everything to paste is in this folder. Upload the main video first: the Shorts link to it.

## Once per channel

- **Verify the channel by phone** at https://www.youtube.com/verify. Custom thumbnails need it (Studio → Settings →
  Channel → Feature eligibility shows what is unlocked). Chapters need the channel to have no active strikes; if they
  don't appear, apply for Advanced features on the same page.
- Create the playlist **${yt.playlist ?? 'LLM System Design'}** (it can also be created from the upload form).

## Main video: \`out/${mp4Name('main')}\`

1. YouTube Studio → **Create → Upload videos** → select the MP4.
2. **Details**
   - **Title**: first line of \`main/title.txt\`.
   - **Description**: paste \`main/description.txt\`. Its timestamps (starting at 00:00) become the chapters.
   - **Thumbnail**: upload \`main/thumbnail.png\` (1280×720 PNG).
   - **Playlists**: ${yt.playlist ?? 'LLM System Design'}.
   - **Audience**: "No, it's not made for kids".
   - **Attributes → AI use**: **Yes**. The narration is an AI-generated voice. YouTube's policy only requires
     disclosure for realistic depictions of real people or events, but it labels AI-generated non-music audio in the
     expanded description, which matches the credit line already in the description.
   - **Show more**: **Tags**: paste \`main/tags.txt\`. **Language**: English. **Caption certification**: "This content
     has never aired on television in the U.S." **Category**: Education.
3. **Video elements**
   - **Add subtitles → Upload file → With timing** → \`main/captions.en.srt\`.
   - **Add end screen**: ${endcardHelp}
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
   kids. **Attributes → AI use**: Yes. **Playlists**: ${yt.playlist ?? 'LLM System Design'}.
3. **Video elements → Add subtitles → Upload file → With timing** → \`shorts/${id}/captions.en.srt\`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.`,
  )
  .join('\n\n')}
`;
writeFileSync(join(kitDir, 'UPLOAD.md'), checklist);
console.log(`Upload kit: ${relative(ROOT, kitDir)}/`);
