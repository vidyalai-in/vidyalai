// Browser test for the unlisted /practice/ section: answers every set, grades, retries, checks the
// memorize flow and the print layout. Usage: node scripts/practice-test.mjs http://localhost:4321 [outDir]
import puppeteer from 'puppeteer';
import { existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const base = process.argv[2] ?? 'http://localhost:4321';
const outDir = process.argv[3];
if (outDir) mkdirSync(outDir, { recursive: true });
const pw = join(homedir(), '.cache/ms-playwright');
const executablePath = existsSync(pw)
  ? readdirSync(pw).filter((d) => d.startsWith('chromium-')).map((d) => join(pw, d, 'chrome-linux64/chrome')).find(existsSync)
  : undefined;
const browser = await puppeteer.launch({ executablePath, args: ['--no-sandbox'] });
let failures = 0;
const check = (ok, msg) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failures += 1;
};

const page = await browser.newPage();
page.on('pageerror', (e) => check(false, `page error: ${e.message}`));
page.on('dialog', (d) => d.accept());
await page.setViewport({ width: 1100, height: 900 });

// Hub lists every set.
await page.goto(`${base}/practice/`, { waitUntil: 'networkidle0' });
let slugs = await page.$$eval('[data-set]', (els) => els.map((e) => e.dataset.set));
if (process.env.ONLY) slugs = slugs.filter((s) => s.includes(process.env.ONLY));
check(slugs.length > 0, `hub lists ${slugs.length} sets`);
check((await page.$('meta[name="robots"][content="noindex"]')) !== null, 'hub is noindex');

for (const slug of slugs) {
  await page.goto(`${base}/practice/${slug}/`, { waitUntil: 'networkidle0' });
  await page.evaluate(() => localStorage.clear());
  await page.reload({ waitUntil: 'networkidle0' });
  const key = await page.$eval('[data-key]', (s) => JSON.parse(s.textContent));
  check((await page.$('meta[name="robots"][content="noindex"]')) !== null, `${slug}: noindex`);
  check((await page.$('[data-pagefind-body]')) === null, `${slug}: not in search index`);
  check((await page.$$eval('.q-feedback:not([hidden])', (els) => els.length)) === 0, `${slug}: no explanations before submit`);

  if (key.kind === 'mock') await page.click('[data-start]');

  // Memorize blocks: questions hidden until studied.
  const mems = await page.$$('[data-mem]');
  for (const mem of mems) {
    const hiddenBefore = await mem.$eval('[data-mem-qs]', (el) => getComputedStyle(el).display === 'none');
    check(hiddenBefore, `${slug}: memorize questions hidden before study`);
    await mem.$eval('[data-mem-start]', (b) => b.click());
    const showing = await mem.$eval('[data-mem-body]', (el) => getComputedStyle(el).display !== 'none');
    check(showing, `${slug}: picture shown while studying`);
    await mem.$eval('[data-mem-done]', (b) => b.click());
    const after = await mem.evaluate((el) => ({
      body: getComputedStyle(el.querySelector('[data-mem-body]')).display,
      qs: getComputedStyle(el.querySelector('[data-mem-qs]')).display,
    }));
    check(after.body === 'none' && after.qs !== 'none', `${slug}: picture hidden, questions shown after study`);
  }

  // Answer all correctly except the first two (one wrong, one blank).
  const wrongOf = (a) => (a === 'A' ? 'B' : 'A');
  for (const [i, q] of key.questions.entries()) {
    if (i === 1) continue;
    const v = i === 0 ? wrongOf(q.answer) : q.answer;
    await page.$eval(`input[name="q-${q.id}"][value="${v}"]`, (el) => el.click());
  }
  await page.reload({ waitUntil: 'networkidle0' });
  const restored = await page.$$eval('input:checked', (els) => els.length);
  check(restored === key.questions.length - 1, `${slug}: ${restored} answers restored after reload`);

  await page.click('[data-submit]');
  const score = await page.$eval('[data-score]', (el) => el.textContent);
  const expected = `${key.questions.length - 2} / ${key.questions.length}`;
  check(score === expected, `${slug}: score ${score} (expected ${expected})`);
  const shown = await page.$$eval('.q-feedback:not([hidden])', (els) => els.length);
  check(shown === key.questions.length, `${slug}: ${shown} explanations shown after submit`);
  const marks = await page.$$eval('.q', (els) => ({
    right: els.filter((e) => e.classList.contains('is-right')).length,
    wrong: els.filter((e) => e.classList.contains('is-wrong')).length,
    blank: els.filter((e) => e.classList.contains('is-blank')).length,
  }));
  check(marks.wrong === 1 && marks.blank === 1, `${slug}: marks ${JSON.stringify(marks)}`);
  if (outDir) await page.screenshot({ path: join(outDir, `${slug}-results.png`) });

  // Retry wrong ones: fix them, submit again.
  await page.click('[data-retry]');
  for (const q of key.questions.slice(0, 2)) await page.$eval(`input[name="q-${q.id}"][value="${q.answer}"]`, (el) => el.click());
  await page.click('[data-submit]');
  const score2 = await page.$eval('[data-score]', (el) => el.textContent);
  check(score2 === `${key.questions.length} / ${key.questions.length}`, `${slug}: retry score ${score2}`);

  // Start over: options are shuffled (where allowed), grading and feedback letters still line up.
  await Promise.all([page.waitForNavigation({ waitUntil: 'networkidle0' }), page.click('[data-reset]')]);
  const moved = await page.$$eval('fieldset.q', (fss) => fss.filter((fs) => [...fs.querySelectorAll('.opt')].map((l) => l.dataset.letter).join('') !== 'ABCD').length);
  const shufflable = key.questions.filter((q) => q.shuffle).length;
  check(shufflable === 0 || moved > 0, `${slug}: ${moved} of ${shufflable} shuffleable questions reordered after Start over`);
  if (key.kind === 'mock') await page.click('[data-start]');
  for (const mem of await page.$$('[data-mem]')) {
    const st = await mem.evaluate((m) => `${m.dataset.state} hidden=${m.querySelector('[data-mem-start]').hidden}`);
    if (!st.startsWith('ready')) console.log('  mem state after Start over:', st);
    await mem.$eval('[data-mem-start]', (b) => b.click());
    await mem.$eval('[data-mem-done]', (b) => b.click());
  }
  for (const [i, q] of key.questions.entries()) {
    const v = i === 0 ? wrongOf(q.answer) : q.answer;
    await page.$eval(`input[name="q-${q.id}"][value="${v}"]`, (el) => el.click());
  }
  await page.click('[data-submit]');
  const score3 = await page.$eval('[data-score]', (el) => el.textContent);
  check(score3 === `${key.questions.length - 1} / ${key.questions.length}`, `${slug}: score after shuffle ${score3}`);
  const fb = await page.$eval(`[data-q="${key.questions[0].id}"]`, (fs) => ({
    head: fs.querySelector('.fb-head').textContent,
    correct: fs.querySelector('.opt-correct .opt-letter').textContent,
    picked: fs.querySelector('.opt-wrong .opt-letter').textContent,
  }));
  check(fb.head.includes(`You chose ${fb.picked}.`) && fb.head.includes(`answer is ${fb.correct}.`), `${slug}: feedback letters match shown letters (${fb.head})`);

  // Print layout: key appended, controls hidden.
  await page.emulateMediaType('print');
  await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
  const print = await page.evaluate(() => ({
    keyItems: document.querySelectorAll('.print-key li').length,
    bar: getComputedStyle(document.querySelector('[data-bar]')).display,
    results: getComputedStyle(document.querySelector('[data-results]')).display,
  }));
  check(print.keyItems === key.questions.length && print.bar === 'none' && print.results === 'none', `${slug}: print ${JSON.stringify(print)}`);
  if (outDir) {
    const pdf = join(outDir, `${slug}.pdf`);
    await page.pdf({ path: pdf, format: 'A4', margin: { top: '14mm', bottom: '14mm', left: '12mm', right: '12mm' } });
  }
  await page.emulateMediaType('screen');
  await page.evaluate(() => localStorage.clear());
}

await browser.close();
console.log(failures ? `${failures} failure(s)` : 'all passed');
process.exit(failures ? 1 : 0);
