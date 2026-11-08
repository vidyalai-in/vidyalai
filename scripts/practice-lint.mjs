// Style checks for practice sets (the schema in src/content.config.ts checks structure).
// Usage: node scripts/practice-lint.mjs [files...]   (default: all of src/content/practice)
import { readdirSync, readFileSync } from 'node:fs';
import YAML from 'yaml';

const dir = 'src/content/practice';
const files = process.argv.slice(2).length ? process.argv.slice(2) : readdirSync(dir).filter((f) => f.endsWith('.md')).map((f) => `${dir}/${f}`);
// Names that must never appear in practice content (kept encoded so they don't appear in the repo either).
const NAMES = Buffer.from('YmFrbGl3YWx8bG9naXFpZHM=', 'base64').toString();
const BANNED = new RegExp(`${NAMES}|\\$\\$|<[a-z]+[\\s>/]`, 'i');
let problems = 0;
const allIds = new Map();

for (const file of files) {
  const text = readFileSync(file, 'utf8');
  const [, fmText, body] = text.split(/^---$/m);
  const fm = YAML.parse(fmText);
  const issues = [];
  const qs = fm.questions ?? [];
  const mock = fm.family === 'mock-tests';

  if (!mock && (qs.length < 10 || qs.length > 14)) issues.push(`${qs.length} questions (want 10–14)`);
  if (!mock && (fm.timeMinutes < 10 || fm.timeMinutes > 15)) issues.push(`timeMinutes ${fm.timeMinutes} (want 10–15)`);
  if (fm.slug !== file.split('/').pop().replace(/\.md$/, '')) issues.push(`slug ${fm.slug} does not match file name`);

  const letters = { A: 0, B: 0, C: 0, D: 0 };
  for (const q of qs) letters[q.answer] = (letters[q.answer] ?? 0) + 1;
  const most = Math.max(...Object.values(letters)), least = Math.min(...Object.values(letters));
  if (qs.length >= 8 && (most > Math.ceil(qs.length / 4) + 2 || least === 0)) issues.push(`answer letters unbalanced ${JSON.stringify(letters)}`);

  const diff = [1, 2, 3].map((d) => qs.filter((q) => q.difficulty === d).length);
  if (!mock && (diff[0] < 2 || diff[1] < 3 || diff[2] < 2)) issues.push(`difficulty mix ${diff.join('/')} (want at least 2 easy, 3 medium, 2 hard)`);

  for (const q of qs) {
    const n = (q.options ?? q.optionFigures ?? []).length;
    if (n !== 4) issues.push(`${q.id}: ${n} options (want 4)`);
    if (allIds.has(q.id)) issues.push(`${q.id}: id also used in ${allIds.get(q.id)}`);
    allIds.set(q.id, file);
    for (const field of ['q', 'explanation']) if (BANNED.test(q[field] ?? '')) issues.push(`${q.id}: banned text/HTML/LaTeX in ${field}`);
    for (const o of q.options ?? []) if (BANNED.test(o)) issues.push(`${q.id}: banned text in option`);
    if (q.explanation && !/\*\*[^*]+\*\*/.test(q.explanation)) issues.push(`${q.id}: explanation should bold the answer`);
  }
  if (BANNED.test(body.replace(/<!--[\s\S]*?-->/g, ''))) issues.push('banned text/HTML/LaTeX in primer');
  const words = body.split(/\s+/).filter(Boolean).length;
  if (words < 100 || words > 500) issues.push(`primer is ${words} words (want 100–500)`);

  console.log(`${issues.length ? 'FIX ' : 'ok  '} ${file}  ${qs.length} Qs  ${diff.join('/')}  ${Object.entries(letters).map(([k, v]) => k + v).join(' ')}`);
  for (const i of issues) console.log(`       - ${i}`);
  problems += issues.length;
}
process.exit(problems ? 1 : 0);
