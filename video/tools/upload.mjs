#!/usr/bin/env node
// YouTube uploader for the kit that build.mjs writes. Does every step the YouTube Data API v3 supports;
// prints the few that stay manual (end screen, a Short's Related video, pinning the comment).
//
//   node video/tools/upload.mjs auth                          # one-time OAuth sign-in
//   node video/tools/upload.mjs upload  video/<slug> [--dry-run] [--only main,short-id]
//   node video/tools/upload.mjs status  video/<slug>
//   node video/tools/upload.mjs publish video/<slug> [--only main,short-id]
//
// upload: every video goes up private, with title, description, tags, category, language, audience,
// AI disclosure, captions, thumbnail (main) and playlist. publish: makes them public (main first) and
// posts the comment from main/pinned-comment.txt. Progress is saved in youtube/uploaded.json after each
// step, so a re-run resumes where it stopped and never uploads a video twice.
//
// Credentials live outside the repo, in ~/.config/vidyalai/: youtube-client.json (a Desktop-app OAuth
// client from Google Cloud Console) and youtube-token.json (written by `auth`).

import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, openSync, readFileSync, readSync, closeSync, statSync, writeFileSync, renameSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { google } from 'googleapis';
import YAML from 'yaml';
import { getSeries } from '../../src/data/series.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const CONFIG_DIR = join(homedir(), '.config', 'vidyalai');
const CLIENT_FILE = join(CONFIG_DIR, 'youtube-client.json');
const TOKEN_FILE = join(CONFIG_DIR, 'youtube-token.json');
const SCOPES = ['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube.force-ssl'];
const CATEGORY_EDUCATION = '27';
const CHUNK = 32 * 1024 * 1024; // resumable upload chunk; must be a multiple of 256 KiB

const [cmd, target] = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const flag = (name) => process.argv.includes(`--${name}`);
const opt = (name) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : undefined;
};
const dryRun = flag('dry-run');

// Full status object. videos.update replaces the whole `status` part, so every write sends all of it.
const status = (privacyStatus) => ({
  privacyStatus,
  selfDeclaredMadeForKids: false,
  containsSyntheticMedia: true, // Studio's "AI use" answer: the narration is an AI-generated voice
  embeddable: true,
  license: 'youtube',
  publicStatsViewable: true,
});

// ---------- Auth ----------
function loadClient() {
  if (!existsSync(CLIENT_FILE)) {
    fail(`No OAuth client at ${CLIENT_FILE}.
  Google Cloud Console → APIs & Services: enable "YouTube Data API v3", set up the OAuth consent screen
  (External, add yourself as a test user), then Credentials → Create OAuth client ID → Desktop app →
  Download JSON, and save it as that file.`);
  }
  const json = JSON.parse(readFileSync(CLIENT_FILE, 'utf8'));
  const { client_id, client_secret } = json.installed ?? json.web ?? {};
  if (!client_id) fail(`${CLIENT_FILE} has no "installed" client. Create the OAuth client as a Desktop app.`);
  return { client_id, client_secret };
}

function saveToken(token) {
  mkdirSync(CONFIG_DIR, { recursive: true, mode: 0o700 });
  writeFileSync(TOKEN_FILE, JSON.stringify(token, null, 2), { mode: 0o600 });
}

async function auth() {
  const { client_id, client_secret } = loadClient();
  const server = createServer();
  await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
  const redirect = `http://localhost:${server.address().port}`;
  const client = new google.auth.OAuth2(client_id, client_secret, redirect);
  const url = client.generateAuthUrl({ access_type: 'offline', prompt: 'consent', scope: SCOPES });
  console.log(`Open this link, sign in with the channel's Google account, and allow access:\n\n${url}\n`);
  const code = await new Promise((ok, no) => {
    server.on('request', (req, res) => {
      const params = new URL(req.url, redirect).searchParams;
      res.end(params.get('code') ? 'Signed in. You can close this tab.' : `Sign-in failed: ${params.get('error')}`);
      params.get('code') ? ok(params.get('code')) : no(new Error(params.get('error') ?? 'no code'));
    });
  });
  server.close();
  const { tokens } = await client.getToken(code);
  saveToken(tokens);
  console.log(`Saved ${TOKEN_FILE}`);
}

function authorized() {
  const { client_id, client_secret } = loadClient();
  if (!existsSync(TOKEN_FILE)) fail('Not signed in. Run: node video/tools/upload.mjs auth');
  const client = new google.auth.OAuth2(client_id, client_secret);
  const saved = JSON.parse(readFileSync(TOKEN_FILE, 'utf8'));
  client.setCredentials(saved);
  client.on('tokens', (t) => saveToken({ ...saved, ...t }));
  return client;
}

// ---------- Kit ----------
function loadKit(dir) {
  const videoDir = resolve(ROOT, dir);
  const script = YAML.parse(readFileSync(join(videoDir, 'script.yaml'), 'utf8'));
  const slug = script.slug;
  const kitDir = join(videoDir, 'youtube');
  const read = (p) => readFileSync(join(kitDir, p), 'utf8').trim();
  const only = opt('only')?.split(',');
  const entries = [
    {
      id: 'main',
      file: join(videoDir, 'out', `${slug}.mp4`),
      title: read('main/title.txt').split('\n')[0].trim(),
      description: read('main/description.txt'),
      tags: read('main/tags.txt').split(',').map((t) => t.trim()).filter(Boolean),
      captions: join(kitDir, 'main/captions.en.srt'),
      thumbnail: join(kitDir, 'main/thumbnail.png'),
      comment: existsSync(join(kitDir, 'main/pinned-comment.txt')) ? read('main/pinned-comment.txt') : null,
    },
    ...(script.shorts ?? []).map((s) => ({
      id: s.id,
      file: join(videoDir, 'out', `${slug}--${s.id}.mp4`),
      title: read(`shorts/${s.id}/title.txt`).split('\n')[0].trim(),
      description: read(`shorts/${s.id}/description.txt`),
      tags: [],
      captions: join(kitDir, `shorts/${s.id}/captions.en.srt`),
    })),
  ].filter((e) => !only || only.includes(e.id));
  const statePath = join(kitDir, 'uploaded.json');
  const state = existsSync(statePath) ? JSON.parse(readFileSync(statePath, 'utf8')) : { videos: {} };
  const save = () => {
    if (dryRun) return;
    writeFileSync(`${statePath}.tmp`, JSON.stringify(state, null, 2) + '\n');
    renameSync(`${statePath}.tmp`, statePath);
  };
  const endcard = existsSync(join(kitDir, 'main/endcard.txt'))
    ? read('main/endcard.txt')
    : readFileSync(join(kitDir, 'UPLOAD.md'), 'utf8').match(/from (\d+:\d\d:\d\d) in Studio/)?.[1];
  return { slug, script, kitDir, entries, state, save, endcard, playlist: script.youtube?.playlist ?? articleSeries(script).name };
}

/** The series of the article a video is made from (frontmatter `series`, default LLM System Design). */
function articleSeries(script) {
  const front = YAML.parse(readFileSync(join(ROOT, script.article), 'utf8').split(/^---$/m)[1]);
  return getSeries(front.series);
}

// ---------- Resumable upload (resumes across runs via the saved session URI) ----------
async function uploadVideo(client, entry, rec, save) {
  const size = statSync(entry.file).size;
  const token = async () => (await client.getAccessToken()).token;
  if (!rec.session) {
    const res = await fetch('https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${await token()}`,
        'Content-Type': 'application/json; charset=UTF-8',
        'X-Upload-Content-Length': String(size),
        'X-Upload-Content-Type': 'video/mp4',
      },
      body: JSON.stringify(metadata(entry)),
    });
    if (!res.ok) throw new Error(`upload session for ${entry.id}: ${res.status} ${await res.text()}`);
    rec.session = res.headers.get('location');
    save();
  }
  // Ask the server how much it already has (0 for a fresh session).
  let offset = await committed(rec.session, size, await token());
  if (typeof offset === 'object') return offset; // already complete: the video resource
  const fd = openSync(entry.file, 'r');
  try {
    for (;;) {
      const len = Math.min(CHUNK, size - offset);
      const buf = Buffer.alloc(len);
      readSync(fd, buf, 0, len, offset);
      let res;
      try {
        res = await fetch(rec.session, {
          method: 'PUT',
          headers: { Authorization: `Bearer ${await token()}`, 'Content-Length': String(len), 'Content-Range': `bytes ${offset}-${offset + len - 1}/${size}` },
          body: buf,
        });
      } catch (e) {
        console.log(`  ${entry.id}: network error (${e.message}), resuming`);
        await sleep(5000);
        const now = await committed(rec.session, size, await token());
        if (typeof now === 'object') return now;
        offset = now;
        continue;
      }
      if (res.status === 200 || res.status === 201) return res.json();
      if (res.status === 308) {
        offset = rangeEnd(res);
        process.stdout.write(`\r  ${entry.id}: ${((offset / size) * 100).toFixed(0)}% of ${(size / 1e6).toFixed(0)} MB`);
        continue;
      }
      if (res.status >= 500) {
        await sleep(5000);
        offset = await committed(rec.session, size, await token());
        continue;
      }
      if (res.status === 404 || res.status === 410) {
        delete rec.session; // session expired: start a new one on the next run
        save();
      }
      throw new Error(`upload ${entry.id}: ${res.status} ${await res.text()}`);
    }
  } finally {
    closeSync(fd);
    process.stdout.write('\n');
  }
}

async function committed(session, size, token) {
  const res = await fetch(session, { method: 'PUT', headers: { Authorization: `Bearer ${token}`, 'Content-Range': `bytes */${size}`, 'Content-Length': '0' } });
  if (res.status === 200 || res.status === 201) return res.json();
  if (res.status === 308) return rangeEnd(res);
  throw new Error(`upload status: ${res.status} ${await res.text()}`);
}
const rangeEnd = (res) => {
  const r = res.headers.get('range');
  return r ? Number(r.split('-')[1]) + 1 : 0;
};

const metadata = (entry) => ({
  snippet: {
    title: entry.title,
    description: entry.description,
    ...(entry.tags.length ? { tags: entry.tags } : {}),
    categoryId: CATEGORY_EDUCATION,
    defaultLanguage: 'en',
    defaultAudioLanguage: 'en',
  },
  status: status('private'),
});

// ---------- Commands ----------
async function upload(dir) {
  const kit = loadKit(dir);
  const { entries, state, save } = kit;
  for (const e of entries) if (!existsSync(e.file)) fail(`Missing ${relative(ROOT, e.file)}. Build it first.`);
  if (dryRun) {
    for (const e of entries) {
      console.log(`\n${e.id}: ${relative(ROOT, e.file)} (${(statSync(e.file).size / 1e6).toFixed(1)} MB)`);
      const m = metadata(e);
      m.snippet.description = `${m.snippet.description.slice(0, 80).replace(/\n/g, ' ')}… (${m.snippet.description.length} chars)`;
      console.log(JSON.stringify(m, null, 2));
      console.log(`  + captions ${relative(ROOT, e.captions)}${e.thumbnail ? `, thumbnail ${relative(ROOT, e.thumbnail)}` : ''}, playlist "${kit.playlist}"`);
    }
    return;
  }
  const client = authorized();
  const yt = google.youtube({ version: 'v3', auth: client });
  state.playlistId ??= await findPlaylist(yt, kit.playlist);
  save();

  for (const e of entries) {
    const rec = (state.videos[e.id] ??= { steps: {} });
    if (!rec.id) {
      console.log(`${e.id}: uploading ${relative(ROOT, e.file)}`);
      const video = await uploadVideo(client, e, rec, save);
      rec.id = video.id;
      delete rec.session;
      save();
      console.log(`  ${e.id}: https://youtu.be/${rec.id} (private)`);
    }
    if (e.thumbnail && !rec.steps.thumbnail) {
      await step(e, 'thumbnail', () => yt.thumbnails.set({ videoId: rec.id, media: { mimeType: 'image/png', body: createReadStream(e.thumbnail) } }), rec, save);
    }
    if (!rec.steps.captions) {
      await step(
        e,
        'captions',
        () =>
          yt.captions.insert({
            part: ['snippet'],
            requestBody: { snippet: { videoId: rec.id, language: 'en', name: 'English', isDraft: false } },
            media: { mimeType: 'application/octet-stream', body: createReadStream(e.captions) },
          }),
        rec,
        save,
      );
    }
    if (!rec.steps.playlist) {
      await step(
        e,
        'playlist',
        () => yt.playlistItems.insert({ part: ['snippet'], requestBody: { snippet: { playlistId: state.playlistId, resourceId: { kind: 'youtube#video', videoId: rec.id } } } }),
        rec,
        save,
      );
    }
  }
  manualChecklist(kit);
}

async function step(entry, name, call, rec, save) {
  for (let attempt = 1; ; attempt++) {
    try {
      await call();
      rec.steps[name] = true;
      save();
      console.log(`  ${entry.id}: ${name} ✓`);
      return;
    } catch (err) {
      const reason = apiError(err);
      // Fresh uploads can briefly be "not found" by the other endpoints; retry those a few times.
      if (attempt < 4 && /videoNotFound|notFound|backendError|internalError/i.test(reason)) {
        await sleep(15000 * attempt);
        continue;
      }
      console.log(`  ${entry.id}: ${name} FAILED: ${reason}`);
      if (name === 'thumbnail' && /forbidden|permission/i.test(reason)) {
        console.log('    Custom thumbnails need a phone-verified channel: https://www.youtube.com/verify');
      }
      return; // left unmarked, so the next run retries it
    }
  }
}

async function findPlaylist(yt, title) {
  let pageToken;
  do {
    const { data } = await yt.playlists.list({ part: ['snippet'], mine: true, maxResults: 50, pageToken });
    const hit = data.items.find((p) => p.snippet.title === title);
    if (hit) return hit.id;
    pageToken = data.nextPageToken;
  } while (pageToken);
  fail(`No playlist titled "${title}" on this channel. Create it in YouTube Studio first.`);
}

function manualChecklist(kit) {
  const { state, endcard } = kit;
  const main = state.videos.main?.id;
  const studio = (id) => `https://studio.youtube.com/video/${id}/edit`;
  const shorts = Object.entries(state.videos).filter(([id, v]) => id !== 'main' && v.id);
  const failed = Object.entries(state.videos).flatMap(([id, v]) => ['thumbnail', 'captions', 'playlist'].filter((s) => v.id && v.steps[s] === undefined && (s !== 'thumbnail' || id === 'main')).map((s) => `${id} ${s}`));
  console.log(`
Uploaded as private. Left to do by hand in YouTube Studio:
${failed.length ? `\n  ! These steps failed; re-run upload to retry: ${failed.join(', ')}\n` : ''}${
    main
      ? `
1. End screen on the main video: ${studio(main)} → Video elements → End screen.
   ${endcard ? `The end card starts at ${endcard} (m:ss:frames). ` : ''}Subscribe in the circle, and the playlist (or the next
   video, once it's public) in the frame. See UPLOAD.md for sizing.`
      : ''
  }${
    shorts.length
      ? `
2. Related video on each Short → the main video${main ? ` (https://youtu.be/${main})` : ''}:
${shorts.map(([id, v]) => `   - ${id}: ${studio(v.id)}`).join('\n')}`
      : ''
  }
3. Check each video's details (Checks tab: copyright) and that the Upload defaults applied: comment
   moderation, automatic chapters/places/concepts off, caption certification.
4. Then: node video/tools/upload.mjs publish ${relative(ROOT, resolve(ROOT, target))}
   (it makes everything public, main first, and posts the comment; pin it from the main video's comments)
`);
}

async function showStatus(dir) {
  const { state } = loadKit(dir);
  const ids = Object.entries(state.videos).filter(([, v]) => v.id);
  if (!ids.length) return console.log('Nothing uploaded yet.');
  const yt = google.youtube({ version: 'v3', auth: authorized() });
  const { data } = await yt.videos.list({ part: ['status', 'processingDetails', 'snippet'], id: ids.map(([, v]) => v.id) });
  const byId = new Map(data.items.map((v) => [v.id, v]));
  for (const [name, v] of ids) {
    const r = byId.get(v.id);
    if (!r) {
      console.log(`${name.padEnd(24)} ${v.id}  NOT FOUND (deleted?)`);
      continue;
    }
    const s = r.status;
    console.log(`${name.padEnd(24)} https://youtu.be/${v.id}  ${s.privacyStatus.padEnd(8)} upload=${s.uploadStatus} processing=${r.processingDetails?.processingStatus ?? '?'}${s.rejectionReason ? ` rejected=${s.rejectionReason}` : ''}`);
  }
}

async function publish(dir) {
  const kit = loadKit(dir);
  const { state, save, entries } = kit;
  const order = entries.filter((e) => state.videos[e.id]?.id); // main first, then Shorts
  if (!order.length) fail('Nothing uploaded yet. Run upload first.');
  const yt = google.youtube({ version: 'v3', auth: authorized() });
  const { data } = await yt.videos.list({ part: ['status'], id: order.map((e) => state.videos[e.id].id) });
  const byId = new Map(data.items.map((v) => [v.id, v.status]));
  const notReady = order.filter((e) => byId.get(state.videos[e.id].id)?.uploadStatus !== 'processed');
  if (notReady.length) fail(`Still processing (or failed): ${notReady.map((e) => e.id).join(', ')}. Check with: status ${dir}`);

  for (const e of order) {
    const rec = state.videos[e.id];
    if (byId.get(rec.id).privacyStatus !== 'public') {
      await yt.videos.update({ part: ['status'], requestBody: { id: rec.id, status: status('public') } });
      console.log(`${e.id}: public  https://youtu.be/${rec.id}`);
    }
    rec.published = true;
    save();
    if (e.comment && !rec.steps.comment) {
      await step(
        e,
        'comment',
        () => yt.commentThreads.insert({ part: ['snippet'], requestBody: { snippet: { videoId: rec.id, topLevelComment: { snippet: { textOriginal: e.comment } } } } }),
        rec,
        save,
      );
    }
  }
  const main = state.videos.main?.id;
  if (main) {
    console.log(`
Published. Pin the comment: https://studio.youtube.com/video/${main}/comments
Main video: https://youtu.be/${main}`);
  }
}

// ---------- Helpers ----------
function apiError(err) {
  const d = err?.response?.data?.error;
  if (d?.errors?.[0]) return `${d.errors[0].reason}: ${d.message}`;
  if (err?.response?.data?.error === 'invalid_grant') return 'invalid_grant: sign-in expired. Run: node video/tools/upload.mjs auth';
  return err?.message ?? String(err);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function fail(msg) {
  console.error(msg);
  process.exit(1);
}

const commands = { auth, upload, status: showStatus, publish };
if (!commands[cmd] || (cmd !== 'auth' && !target)) {
  fail('Usage: node video/tools/upload.mjs auth | upload <video/slug> [--dry-run] [--only ids] | status <video/slug> | publish <video/slug> [--only ids]');
}
try {
  await commands[cmd](target);
} catch (err) {
  fail(`Error: ${apiError(err)}`);
}
