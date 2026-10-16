// Screenshots pages at several widths and both themes, and reports horizontal overflow.
// Usage: node scripts/screenshots.mjs <baseUrl> <outDir> <path> [path ...]
import puppeteer from 'puppeteer';
import { existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const [base, outDir, ...paths] = process.argv.slice(2);
const widths = (process.env.WIDTHS || '375,768,1280,1600').split(',').map(Number);
const themes = (process.env.THEMES || 'light,dark').split(',');
const fullPage = process.env.FULL !== '0';
mkdirSync(outDir, { recursive: true });

const pw = join(homedir(), '.cache/ms-playwright');
const executablePath = existsSync(pw)
  ? readdirSync(pw).filter((d) => d.startsWith('chromium-')).map((d) => join(pw, d, 'chrome-linux64/chrome')).find(existsSync)
  : undefined;
const browser = await puppeteer.launch({ executablePath, args: ['--no-sandbox'] });
for (const path of paths) {
  for (const theme of themes) {
    for (const width of widths) {
      const page = await browser.newPage();
      await page.setViewport({ width, height: 900 });
      await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: theme }]);
      await page.goto(base + path, { waitUntil: 'networkidle0' });
      await page.evaluate(() => document.fonts.ready);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      const file = join(outDir, `${path.replace(/\W+/g, '_') || 'home'}-${theme}-${width}.png`);
      await page.screenshot({ path: file, fullPage });
      console.log(`${file}${overflow > 0 ? `  OVERFLOW ${overflow}px` : ''}`);
      await page.close();
    }
  }
}
await browser.close();
