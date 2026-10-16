// Renders diagrams/*.mmd into public/diagrams/: <name>.svg (light), <name>-dark.svg, <name>.png (light, 2x, for Google Docs).
// Usage: node scripts/render-diagrams.mjs [--force] [name ...]
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, basename } from 'node:path';
import { homedir, tmpdir } from 'node:os';

const SRC = 'diagrams';
const OUT = 'public/diagrams';
const args = process.argv.slice(2);
const force = args.includes('--force');
const only = args.filter((a) => !a.startsWith('--'));
mkdirSync(OUT, { recursive: true });

// Reuse Playwright's Chromium when puppeteer's own download was skipped.
const puppeteerConfig = join(tmpdir(), 'vidyalai-puppeteer.json');
const pwDir = join(homedir(), '.cache/ms-playwright');
const chromium = existsSync(pwDir)
  ? readdirSync(pwDir)
      .filter((d) => d.startsWith('chromium-'))
      .map((d) => join(pwDir, d, 'chrome-linux64/chrome'))
      .find((p) => existsSync(p))
  : undefined;
writeFileSync(puppeteerConfig, JSON.stringify({ ...(chromium && { executablePath: chromium }), args: ['--no-sandbox'] }));

const mmdc = join('node_modules', '.bin', 'mmdc');
const files = readdirSync(SRC).filter((f) => f.endsWith('.mmd')).filter((f) => !only.length || only.includes(basename(f, '.mmd')));
let failed = 0;

for (const file of files) {
  const name = basename(file, '.mmd');
  const input = join(SRC, file);
  const outputs = [
    { out: join(OUT, `${name}.svg`), config: 'scripts/mermaid/light.json', extra: ['-b', 'transparent'] },
    { out: join(OUT, `${name}-dark.svg`), config: 'scripts/mermaid/dark.json', extra: ['-b', 'transparent'] },
    { out: join(OUT, `${name}.png`), config: 'scripts/mermaid/light.json', extra: ['-b', 'white', '-s', '2'] },
  ];
  const srcTime = statSync(input).mtimeMs;
  for (const { out, config, extra } of outputs) {
    if (!force && existsSync(out) && statSync(out).mtimeMs > srcTime) continue;
    try {
      execFileSync(mmdc, ['-i', input, '-o', out, '-c', config, '-p', puppeteerConfig, '-q', ...extra], { stdio: 'pipe' });
      console.log(`rendered ${out}`);
    } catch (err) {
      failed++;
      console.error(`FAILED ${input} -> ${out}\n${err.stderr?.toString() || err.message}`);
      break;
    }
  }
}
if (failed) {
  console.error(`${failed} diagram(s) failed`);
  process.exit(1);
}
