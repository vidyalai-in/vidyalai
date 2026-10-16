// Checks every external link in src/content/blog/*.md. Prints failures; exits 1 if any.
// Usage: node scripts/check-links.mjs
import { readdirSync, readFileSync } from 'node:fs';

const dir = 'src/content/blog';
const urls = new Map(); // url -> [files]
for (const f of readdirSync(dir).filter((f) => f.endsWith('.md'))) {
  const text = readFileSync(`${dir}/${f}`, 'utf8');
  for (const m of text.matchAll(/\]\((https?:\/\/[^)\s]+)\)/g)) {
    const u = m[1];
    urls.set(u, [...(urls.get(u) ?? []), f.slice(0, 2)]);
  }
}

const UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36';
async function check(url) {
  for (const method of ['HEAD', 'GET']) {
    try {
      const res = await fetch(url, { method, redirect: 'follow', headers: { 'user-agent': UA }, signal: AbortSignal.timeout(20000) });
      if (res.ok) return { ok: true, status: res.status, final: res.url };
      if (method === 'GET') return { ok: false, status: res.status, final: res.url };
    } catch (e) {
      if (method === 'GET') return { ok: false, status: e.name };
    }
  }
}

const list = [...urls.keys()];
const results = [];
for (let i = 0; i < list.length; i += 12) {
  results.push(...(await Promise.all(list.slice(i, i + 12).map(async (u) => ({ u, ...(await check(u)) })))));
}
const bad = results.filter((r) => !r.ok);
const moved = results.filter((r) => r.ok && r.final && r.final.replace(/\/$/, '') !== r.u.replace(/\/$/, '') && new URL(r.final).host !== new URL(r.u).host);
console.log(`${list.length} unique links, ${bad.length} failing, ${moved.length} cross-host redirects`);
for (const r of bad) console.log(`FAIL ${r.status} ${r.u}  [${urls.get(r.u).join(',')}]`);
for (const r of moved) console.log(`MOVED ${r.u} -> ${r.final}  [${urls.get(r.u).join(',')}]`);
process.exit(bad.length ? 1 : 0);
