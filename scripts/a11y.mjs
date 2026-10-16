// Runs axe-core (WCAG 2.2 A/AA) against every built page in light and dark themes.
// Usage: node scripts/a11y.mjs <baseUrl>   (serve dist/ first, e.g. `npx astro preview`)
import puppeteer from 'puppeteer';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { createRequire } from 'node:module';

const base = process.argv[2] || 'http://localhost:4321';
const axeSource = readFileSync(createRequire(import.meta.url).resolve('axe-core/axe.min.js'), 'utf8');
const pages = [];
(function walk(dir, prefix) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.isDirectory() && !['_astro', 'pagefind', 'diagrams'].includes(e.name)) walk(join(dir, e.name), `${prefix}${e.name}/`);
    else if (e.name === 'index.html') pages.push(`/${prefix}`);
  }
})('dist', '');

const pw = join(homedir(), '.cache/ms-playwright');
const executablePath = readdirSync(pw).filter((d) => d.startsWith('chromium-')).map((d) => join(pw, d, 'chrome-linux64/chrome')).find(existsSync);
const browser = await puppeteer.launch({ executablePath, args: ['--no-sandbox'] });
const summary = new Map();
for (const theme of ['light', 'dark']) {
  for (const path of pages) {
    const page = await browser.newPage();
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: theme }]);
    await page.goto(base + path, { waitUntil: 'networkidle0' });
    await page.evaluate(axeSource);
    const res = await page.evaluate(() => axe.run(document, { runOnly: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice'] }));
    for (const v of res.violations) {
      const key = `${v.impact} ${v.id}`;
      const s = summary.get(key) ?? { help: v.help, pages: new Set(), sample: v.nodes[0]?.target.join(' ') + ' :: ' + (v.nodes[0]?.failureSummary ?? '').split('\n')[1] };
      s.pages.add(`${theme}:${path}`);
      summary.set(key, s);
    }
    await page.close();
  }
}
await browser.close();
console.log(`${pages.length} pages x 2 themes checked`);
for (const [k, s] of summary) console.log(`${k} (${s.pages.size} page-themes): ${s.help}\n   e.g. ${[...s.pages][0]} ${s.sample}`);
process.exit([...summary.keys()].some((k) => /^(serious|critical)/.test(k)) ? 1 : 0);
