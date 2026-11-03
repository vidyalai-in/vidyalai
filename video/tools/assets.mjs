// Scene inputs shared by build.mjs and preview.mjs: diagrams (inlined so steps can light up nodes)
// and the list of parts for `seriesmap` scenes.

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import YAML from 'yaml';

export function diagramAsset(root, name) {
  const svg = readFileSync(join(root, `public/diagrams/${name}-dark.svg`), 'utf8');
  const [, , w, h] = /viewBox="([^"]+)"/.exec(svg)[1].trim().split(/[\s,]+/).map(Number);
  return { diagramSvg: svg, diagramSize: { w, h } };
}

// "Reading "Attention Is All You Need", the 2017 Transformer Paper" -> "Reading "Attention Is All You Need""
const shortTitle = (title) => {
  let t = title.split(':')[0].trim();
  if (t.length > 40) {
    const cut = t.lastIndexOf(', ', 40);
    if (cut > 12) t = t.slice(0, cut);
  }
  return t;
};

// Every published part of a series, in order: [{ n, title, slug }].
export function seriesParts(root, series) {
  const dir = join(root, 'src/content/blog');
  const parts = [];
  for (const f of readdirSync(dir).filter((x) => x.endsWith('.md'))) {
    const front = YAML.parse(readFileSync(join(dir, f), 'utf8').split(/^---$/m)[1]);
    if (front.series === series && front.seriesOrder) parts.push({ n: front.seriesOrder, title: shortTitle(front.title), slug: front.slug });
  }
  return parts.sort((a, b) => a.n - b.n);
}

// Fill in what a scene needs from the repo before it goes to the stage.
export function prepareScene(root, scene, front) {
  const assets = {};
  if (scene.diagram) Object.assign(assets, diagramAsset(root, scene.diagram));
  // figure: an isometric figure built by video/<slug>/figures/<name>.mjs (see tools/iso/README.md).
  if (scene.figure) {
    const file = join(root, 'video', front.slug, 'figures', `${scene.figure}.json`);
    assets.figure = JSON.parse(readFileSync(file, 'utf8'));
  }
  if (scene.type === 'seriesmap') {
    const labels = scene.labels ?? {};
    scene.parts ??= seriesParts(root, scene.series ?? front.series).map((p) => ({ ...p, title: labels[p.n] ?? p.title }));
    scene.current ??= front.seriesOrder;
  }
  return assets;
}
