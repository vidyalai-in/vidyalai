// VidyalAI video stage. build.mjs calls stage.load(spec) once per scene, then stage.frame(t, ...) for
// every frame it captures, where t is the time in seconds since the scene started. Everything on
// screen is a function of t: which narration step is showing, how far its entrance animation has
// run, the slow camera drift, words pulsing as they are spoken, charts morphing, burned-in captions.
// Elements carry data-step="n" (visible from step n on) or data-only="n" (visible only during step n).
// frame() returns a key that changes whenever the picture does, so the build can reuse the previous
// screenshot for frames that would be identical.

(() => {
  const TRANSITION = 14 / 30; // entrance animation length (s)
  const MORPH = 0.9; // chart bars and lines easing between values (s)
  const PULSE = 0.9; // a highlighted phrase glows for this long after it is spoken (s)
  const CAMERA = 1.4; // camera moves between diagram focus rectangles take this long (s)

  const $ = (tag, cls, html) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (html !== undefined) el.innerHTML = html;
    return el;
  };
  const svgEl = (tag, attrs = {}) => {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  };
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  // **highlight** and `code` inline markup.
  const md = (s) =>
    esc(s ?? '')
      .replace(/\*\*(.+?)\*\*/g, '<em class="hl">$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>');
  const clamp = (t) => Math.min(1, Math.max(0, t));
  const ease = (t) => 1 - Math.pow(1 - clamp(t), 3);
  const easeInOut = (t) => {
    t = clamp(t);
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  };
  const lerp = (a, b, t) => a + (b - a) * t;
  // Gentle at both ends, no sudden start: for camera moves and the flow pulse.
  const easeSine = (t) => 0.5 - Math.cos(Math.PI * clamp(t)) / 2;
  // Between two focus rectangles [x, y, w, h]: the centre moves in a line and the size changes
  // geometrically, so a zoom feels even instead of rushing at one end.
  const camLerp = (a, b, k) => {
    const cx = lerp(a[0] + a[2] / 2, b[0] + b[2] / 2, k);
    const cy = lerp(a[1] + a[3] / 2, b[1] + b[3] / 2, k);
    const w = a[2] * Math.pow(b[2] / a[2], k);
    const h = a[3] * Math.pow(b[3] / a[3], k);
    return [cx - w / 2, cy - h / 2, w, h];
  };
  const norm = (w) => String(w).toLowerCase().replace(/[^a-z0-9.%$]+/g, '').replace(/\.$/, '');
  const fmtNum = (v, decimals) => v.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });

  let state;
  let spec;

  function stepEl(el, step) {
    el.dataset.step = String(step);
    return el;
  }

  function header(scene, parent) {
    if (scene.kicker) parent.append(stepEl($('p', 'kicker', md(scene.kicker)), 0));
    if (scene.heading) parent.append(stepEl($('h1', 'heading', md(scene.heading)), 0));
  }

  // Index of the last step at or before `step` that sets `field`, or -1.
  const lastWith = (steps, step, field) => {
    for (let i = step; i >= 0; i--) if (steps[i][field] !== undefined) return i;
    return -1;
  };

  // Time-driven effects registered by builders: each gets (t, step, p) and returns a key fragment.
  const effect = (fn) => state.effects.push(fn);

  const builders = {
    title(scene, root) {
      header(scene, root);
      if (scene.subtitle) root.append(stepEl($('p', 'subtitle', md(scene.subtitle)), scene.subtitleStep ?? 0));
    },

    statement(scene, root) {
      if (scene.kicker) root.append(stepEl($('p', 'kicker', md(scene.kicker)), 0));
      root.append(stepEl($('div', 'statement', md(scene.text)), 0));
      if (scene.sub) root.append(stepEl($('div', 'statement-sub', md(scene.sub)), scene.subStep ?? 0));
    },

    quote(scene, root) {
      if (scene.kicker) root.append(stepEl($('p', 'kicker', md(scene.kicker)), 0));
      const box = stepEl($('figure', 'quote'), 0);
      box.append($('div', 'quote-mark', '“'), $('blockquote', 'quote-text', md(scene.quote)));
      if (scene.source) box.append(stepEl($('figcaption', 'quote-source', md(scene.source)), scene.sourceStep ?? 0));
      root.append(box);
    },

    bullets(scene, root) {
      header(scene, root);
      const list = $('ul', 'bullets');
      let n = 0;
      scene.steps.forEach((st, i) => {
        if (!st.add) return;
        n++;
        const item = typeof st.add === 'string' ? { text: st.add } : st.add;
        const li = stepEl($('li', 'bullet'), i);
        const mark = scene.numbered ? String(n) : scene.style === 'warn' ? '!' : '✓';
        li.append($('span', `mark${scene.style === 'warn' ? ' warn' : ''}`, mark));
        li.append($('span', 'txt', md(item.text) + (item.sub ? `<span class="sub">${md(item.sub)}</span>` : '')));
        list.append(li);
      });
      root.append(list);
    },

    // An isometric figure (tools/iso): its build script supplies the markup, styles and a mount()
    // whose frame(t, step, timing, steps) poses every moving part from the scene clock.
    figure(scene, root, assets) {
      header(scene, root);
      const F = assets.figure;
      if (!document.getElementById('iso-css')) {
        const style = document.createElement('style');
        style.id = 'iso-css';
        style.textContent = F.css;
        document.head.append(style);
      }
      const wrap = stepEl($('div', 'figure-wrap iso', F.html), 0);
      root.append(wrap);
      const kit = new Function(`${F.kit}\nreturn { ${F.kitNames.join(', ')} };`)();
      const api = new Function(`return (${F.mount});`)()(wrap.firstElementChild, kit, F.constants);
      // Camera: each step may name a focus (a rectangle [x, y, w, h] as fractions of the drawing, or
      // a named target the figure defines, defaulting to the step's `station`). It carries over until
      // changed, and moves are the same 1.4 s sine-eased glides as on diagrams.
      const named = F.constants.focus ?? {};
      let last = [0, 0, 1, 1];
      const rects = scene.steps.map((st) => {
        const f = st.focus ?? (Array.isArray(st.station) ? st.station[0] : st.station);
        last = Array.isArray(f) ? f : named[f] ?? last;
        return last;
      });
      const view = wrap.querySelector('.iso-stage');
      const svg = view.querySelector('svg');
      const place = ([x, y, w, h]) => {
        const s = Math.min(1 / w, 1 / h);
        const tx = 0.5 - (x + w / 2) * s;
        const ty = 0.5 - (y + h / 2) * s;
        svg.style.transform = `translate(${(tx * 100).toFixed(3)}%, ${(ty * 100).toFixed(3)}%) scale(${s.toFixed(4)})`;
      };
      effect((t, step) => {
        const from = step > 0 ? rects[step - 1] : rects[0];
        const u = step === 0 ? 1 : easeSine((t - state.timing.steps[step].t0) / CAMERA);
        place(camLerp(from, rects[step], u));
        return `fig${u.toFixed(4)},${api.frame(t, step, state.timing, scene.steps)}`;
      });
    },

    diagram(scene, root, assets) {
      header(scene, root);
      const viewport = $('div', 'viewport');
      const camera = $('div', 'camera');
      // Inline the SVG (not an <img>) so steps can light up nodes and send a pulse along the edges.
      camera.innerHTML = assets.diagramSvg;
      const svg = camera.querySelector('svg');
      svg.removeAttribute('width');
      svg.style.maxWidth = 'none';
      svg.style.display = 'block';
      viewport.append(camera);
      const notes = $('div', 'notes');
      scene.steps.forEach((st, i) => {
        if (st.note) {
          const note = $('div', 'note', md(st.note));
          note.dataset.only = String(i);
          notes.append(note);
        }
      });
      root.append(viewport);
      if (notes.children.length) root.append(notes);
      // Camera focus per step: [x, y, w, h] as fractions of the diagram; carries over until changed.
      let focus = [0, 0, 1, 1];
      state.cameras = scene.steps.map((st) => (focus = st.focus ?? focus));
      Object.assign(state, { viewport, svg, camera });

      // Mermaid ids look like "my-svg-flowchart-TK-1" (nodes) and "my-svg-L_TK_EM_0" (edges).
      const nodes = {};
      for (const g of svg.querySelectorAll('g.node')) {
        const m = /flowchart-(.+)-\d+$/.exec(g.id);
        if (m) nodes[m[1]] = g;
      }
      const edge = (a, b) => svg.querySelector(`path[id$="L_${a}_${b}_0"]`);
      const vb = svg.viewBox.baseVal;
      const center = (id) => {
        const m = /translate\(([-\d.]+)[ ,]+([-\d.]+)/.exec(nodes[id]?.getAttribute('transform') ?? '');
        return m ? { x: Number(m[1]), y: Number(m[2]) } : null;
      };
      // A flow's route: node centre -> along the edge -> next node centre, as points with running
      // arc length. The pulse moves along it at an even speed (eased only at the very start and end)
      // and passes behind the boxes, so it never jumps from one edge to the next.
      const route = (ids) => {
        const pts = [];
        const marks = []; // arc length at which the pulse reaches each node's centre
        const push = (p) => {
          const last = pts.at(-1);
          const d = last ? Math.hypot(p.x - last.x, p.y - last.y) : 0;
          pts.push({ x: p.x, y: p.y, s: (last?.s ?? 0) + d });
        };
        ids.forEach((id, k) => {
          const c = center(id);
          if (c) push(c);
          marks.push(pts.at(-1)?.s ?? 0);
          const e = k + 1 < ids.length ? edge(id, ids[k + 1]) : null;
          if (e) {
            const len = e.getTotalLength();
            for (let j = 0; j <= 24; j++) push(e.getPointAtLength((j / 24) * len));
          }
        });
        return { pts, marks, length: pts.at(-1)?.s ?? 0 };
      };
      const pointAt = ({ pts }, s) => {
        const i = Math.max(1, pts.findIndex((p) => p.s >= s));
        const a = pts[i - 1];
        const b = pts[i] ?? a;
        const k = b.s > a.s ? (s - a.s) / (b.s - a.s) : 0;
        return { x: lerp(a.x, b.x, k), y: lerp(a.y, b.y, k) };
      };
      const routes = scene.steps.map((st) => (st.flow ? route([].concat(st.flow)) : null));
      // follow: the camera window (a fraction of the diagram's width) glides from the first node to
      // the last in step with the pulse; later steps start from where it ended.
      const followX = (st, r, x) => {
        const xs = [r.pts[0].x, r.pts.at(-1).x].map((v) => Math.min(1 - st.follow, Math.max(0, (v - vb.x) / vb.width - st.follow / 2)));
        const span = r.pts.at(-1).x - r.pts[0].x || 1;
        return lerp(xs[0], xs[1], clamp((x - r.pts[0].x) / span));
      };
      scene.steps.forEach((st, i) => {
        if (!st.follow || !routes[i]?.pts.length) return;
        const x = followX(st, routes[i], routes[i].pts.at(-1).x);
        for (let j = i; j < scene.steps.length && (j === i || !scene.steps[j].focus); j++) state.cameras[j] = [x, 0, st.follow, 1];
      });
      const dot = svgEl('circle', { r: Math.max(6, vb.height / 22), class: 'flow-dot' });
      // Under the nodes, so the pulse goes into a box and comes out the other side.
      const nodeLayer = svg.querySelector('g.nodes');
      nodeLayer ? nodeLayer.before(dot) : svg.append(dot);
      effect((t, step) => {
        const hi = lastWith(scene.steps, step, 'highlight');
        const fl = lastWith(scene.steps, step, 'flow');
        // A highlight carries over until a later flow step takes its place.
        const lit = new Set(hi >= 0 && hi >= (fl === step ? step : -1) ? [].concat(scene.steps[hi].highlight) : []);
        // Blocks an earlier flow step already passed through stay at full strength, without the glow,
        // so a block-by-block walkthrough keeps its trail while lighting only the current block.
        const done = new Set(scene.steps.slice(0, step).flatMap((st) => (st.flow ? [].concat(st.flow) : [])));
        let key = '';
        // flow: [A, B, C, ...] sends a glowing pulse through those nodes over `flowTime` seconds.
        if (fl >= 0 && fl === step && routes[fl]?.length) {
          const st = scene.steps[fl];
          const r = routes[fl];
          const ids = [].concat(st.flow);
          const t0 = state.timing.steps[fl].t0;
          const total = st.flowTime ?? Math.min(state.timing.steps[fl].dur, 1.1 * (ids.length - 1));
          const u = clamp((t - t0) / total);
          const s = easeSine(u) * r.length;
          const pt = pointAt(r, s);
          dot.setAttribute('cx', pt.x.toFixed(2));
          dot.setAttribute('cy', pt.y.toFixed(2));
          dot.style.opacity = String(clamp(Math.min(u * 12, (1 - u) * 12)));
          if (st.follow) {
            const into = fl === 0 ? 1 : easeSine((t - t0) / CAMERA);
            const from = state.cameras[Math.max(0, fl - 1)];
            placeCamera(camLerp(from, [followX(st, r, pt.x), 0, st.follow, 1], into));
          }
          // Light the block the pulse has reached; the one it left becomes part of the trail.
          let reached = 0;
          ids.forEach((id, k) => {
            if (s >= r.marks[k] - 1) reached = k;
          });
          ids.forEach((id, k) => {
            if (k < reached) done.add(id);
          });
          lit.add(ids[reached]);
          key += `f${s.toFixed(1)}`;
        } else dot.style.opacity = '0';
        for (const [id, g] of Object.entries(nodes)) {
          g.classList.toggle('lit', lit.has(id));
          g.classList.toggle('dim', lit.size > 0 && !lit.has(id) && !done.has(id));
        }
        return key + [...lit].join(',') + '|' + [...done].join(',');
      });
    },

    stats(scene, root) {
      header(scene, root);
      const row = $('div', 'stats');
      let first = true;
      scene.steps.forEach((st, i) => {
        if (!st.add) return;
        if (!first) row.append(stepEl($('div', 'arrow', '→'), i));
        first = false;
        const card = stepEl($('div', 'stat'), i);
        const value = $('div', `value${String(st.add.value).length > 6 ? ' long' : ''}`, esc(st.add.value));
        const m = /^([^0-9]*)([0-9][0-9,]*\.?[0-9]*)(.*)$/.exec(st.add.value);
        if (m && st.add.count !== false) {
          const target = parseFloat(m[2].replace(/,/g, ''));
          const decimals = (m[2].split('.')[1] || '').length;
          state.counters.push({ el: value, step: i, pre: m[1], post: m[3], target, decimals, comma: m[2].includes(',') });
        }
        card.append(value, $('div', 'label', md(st.add.label)));
        row.append(card);
      });
      root.append(row);
      state.statCards = [...row.querySelectorAll('.stat')];
    },

    budget(scene, root) {
      header(scene, root);
      const box = $('div', 'budget');
      const rows = scene.steps.map((st, i) => (st.add ? { ...st.add, step: i } : null)).filter(Boolean);
      const hi = (r) => (Array.isArray(r.ms) ? r.ms[1] : r.ms);
      const lo = (r) => (Array.isArray(r.ms) ? r.ms[0] : r.ms);
      const max = scene.max ?? rows.reduce((s, r) => s + hi(r), 0);
      let offset = 0;
      for (const r of rows) {
        const row = stepEl($('div', 'budget-row'), r.step);
        row.append($('div', 'budget-label', md(r.label) + (r.sub ? `<small>${md(r.sub)}</small>` : '')));
        const track = $('div', 'budget-track');
        const bar = $('div', `budget-bar${Array.isArray(r.ms) ? ' range' : ''}`);
        const text = Array.isArray(r.ms) ? `${r.ms[0]}–${r.ms[1]} ms` : `${r.ms} ms`;
        const start = scene.waterfall === false ? 0 : offset;
        bar.style.left = `${(start / max) * 100}%`;
        state.bars.push({ el: bar, step: r.step, width: (hi(r) / max) * 100 });
        track.append(bar);
        row.append(track, $('div', 'budget-value', esc(r.text ?? text)));
        box.append(row);
        offset += scene.waterfall === false ? 0 : lo(r) + (hi(r) - lo(r)) / 2;
      }
      root.append(box);
      if (scene.total) {
        const totalStep = scene.steps.findIndex((s) => s.showTotal);
        root.append(stepEl($('div', 'budget-total', md(scene.total)), totalStep >= 0 ? totalStep : scene.steps.length - 1));
      }
    },

    compare(scene, root) {
      header(scene, root);
      const grid = $('div', 'compare');
      grid.style.setProperty('--cols', scene.cols ?? 3);
      scene.steps.forEach((st, i) => {
        if (!st.add) return;
        const c = stepEl($('div', 'card'), i);
        c.append($('h3', '', md(st.add.name)));
        for (const p of [].concat(st.add.pros ?? [])) c.append($('div', 'row pro', `<i>+</i><span>${md(p)}</span>`));
        for (const p of [].concat(st.add.cons ?? [])) c.append($('div', 'row con', `<i>−</i><span>${md(p)}</span>`));
        if (st.add.when) c.append($('div', 'row when', `<i>→</i><span>${md(st.add.when)}</span>`));
        grid.append(c);
      });
      root.append(grid);
    },

    // Horizontal bars that grow in, and morph when a step sets new `values`.
    //   bars: [{ label, value, display?, sub? }], unit, decimals, max, log
    //   step: show (bars visible), values (new numbers), tag (badge, e.g. "T = 2"), highlight (label)
    chart(scene, root) {
      if (scene.chart === 'line') return lineChart(scene, root);
      header(scene, root);
      const box = $('div', 'barchart');
      const tag = $('div', 'chart-tag');
      const rows = scene.bars.map((b) => {
        const row = $('div', 'bc-row');
        const label = $('div', 'bc-label', md(b.label) + (b.sub ? `<small>${md(b.sub)}</small>` : ''));
        const track = $('div', 'bc-track');
        const bar = $('div', 'bc-bar');
        const value = $('div', 'bc-value');
        track.append(bar);
        row.append(label, track, value);
        box.append(row);
        return { row, bar, value, label: b.label };
      });
      root.append(box);
      if (scene.steps.some((s) => s.tag)) root.append(tag);
      if (scene.note) root.append(stepEl($('p', 'chart-note', md(scene.note)), scene.noteStep ?? 0));
      const n = scene.bars.length;
      // The data in force at each step, and when it was set.
      // A step can also change the scale and unit (logits -> percentages): `max`, `unit`, `decimals`.
      const at = (step) => {
        const i = lastWith(scene.steps, step, 'values');
        const st = i >= 0 ? scene.steps[i] : scene;
        const values = i >= 0 ? st.values : scene.bars.map((b) => b.value);
        return { values, from: i, max: st.max ?? scene.max ?? Math.max(...values), unit: st.unit ?? scene.unit ?? '', dec: st.decimals ?? scene.decimals ?? 1 };
      };
      const min = scene.log ? Math.min(...scene.bars.map((b) => b.value)) / 3 : 0;
      const frac = (v, max) => (scene.log ? Math.log(Math.max(v, min) / min) / Math.log(max / min) : v / max);
      const anyShow = scene.steps.some((s) => s.show !== undefined);
      const shownAt = (step) => {
        if (step < 0) return 0;
        const i = lastWith(scene.steps, step, 'show');
        return i >= 0 ? scene.steps[i].show : anyShow ? 0 : n;
      };
      // When bar i first appears: the first step that shows it, staggered within that step.
      const showT0 = (i, step) => {
        for (let s = 0; s <= step; s++)
          if (shownAt(s) > i) return state.timing.steps[s].t0 + Math.max(0, i - shownAt(s - 1)) * 0.12;
        return Infinity;
      };
      effect((t, step) => {
        const shown = shownAt(step);
        const cur = at(step);
        const prev = cur.from > 0 ? at(cur.from - 1) : null;
        const k = cur.from >= 0 && prev ? easeInOut((t - state.timing.steps[cur.from].t0) / MORPH) : 1;
        const hiStep = lastWith(scene.steps, step, 'highlight');
        const hiLabel = hiStep >= 0 ? scene.steps[hiStep].highlight : null;
        let key = '';
        rows.forEach((r, i) => {
          const g = ease((t - showT0(i, step)) / MORPH);
          const visible = i < shown;
          const v = prev ? lerp(prev.values[i], cur.values[i], k) : cur.values[i];
          const max = prev ? lerp(prev.max, cur.max, k) : cur.max;
          r.row.style.opacity = visible ? String(clamp(g * 1.6)) : '0';
          r.bar.style.width = `${(visible ? Math.min(1, frac(v, max)) * g : 0) * 100}%`;
          const disp = scene.bars[i].display;
          const fmt = prev && k < 0.5 ? prev : cur;
          r.value.textContent = disp && !prev && cur.from < 0 ? disp : `${fmtNum(v * (disp ? 1 : g), fmt.dec)}${fmt.unit}`;
          r.row.classList.toggle('lead', hiLabel === r.label);
          key += `${r.bar.style.width}|`;
        });
        const tagStep = lastWith(scene.steps, step, 'tag');
        tag.innerHTML = tagStep >= 0 ? md(scene.steps[tagStep].tag) : '';
        tag.style.opacity = tagStep >= 0 ? '1' : '0';
        return key + tag.textContent + hiLabel;
      });
    },

    // Next-token prediction, shown on real tokens: text -> token pieces -> IDs -> candidate
    // probabilities -> the winner appended. Each step's `show` moves to one of those stages.
    tokens(scene, root) {
      header(scene, root);
      const LEVELS = ['text', 'pieces', 'ids', 'candidates', 'append'];
      const box = $('div', 'tokens');
      const typed = $('div', 'tok-text');
      const chips = $('div', 'tok-chips');
      const pieces = scene.pieces.map((p, i) => {
        const chip = $('div', `tok-chip c${i % 4}`);
        const lead = p.startsWith(' ');
        chip.append($('span', 'tok-piece', (lead ? '<i>␣</i>' : '') + esc(p.trimStart())), $('span', 'tok-id', esc(scene.ids?.[i] ?? '')));
        chips.append(chip);
        return chip;
      });
      const winner = scene.candidates?.[0];
      const add = $('div', 'tok-chip appended');
      if (winner) {
        add.append($('span', 'tok-piece', `<i>␣</i>${esc(winner.t)}`), $('span', 'tok-id', esc(winner.id ?? '')));
        chips.append(add);
      }
      const cands = $('div', 'tok-cands');
      const candRows = (scene.candidates ?? []).map((c) => {
        const row = $('div', 'tc-row');
        const bar = $('div', 'tc-bar');
        const val = $('div', 'tc-val');
        const track = $('div', 'tc-track');
        track.append(bar);
        row.append($('div', 'tc-label', esc(c.t)), track, val);
        cands.append(row);
        return { row, bar, val, p: c.p };
      });
      // chips: false keeps the plain sentence (a cold open): the winner is typed onto its end instead.
      if (scene.chips === false) chips.style.display = 'none';
      box.append(typed, chips, cands);
      root.append(box);
      const level = (step) => {
        const i = lastWith(scene.steps, step, 'show');
        return i >= 0 ? { lv: LEVELS.indexOf(scene.steps[i].show), t0: state.timing.steps[i].t0 } : { lv: 0, t0: 0 };
      };
      const fromLv = LEVELS.indexOf(scene.from ?? 'text'); // stages before `from` are already done
      const levelStart = (lv, step) => {
        if (lv < fromLv) return -100;
        for (let s = 0; s <= step; s++) if (LEVELS.indexOf(scene.steps[s].show) >= lv) return state.timing.steps[s].t0;
        return 0;
      };
      effect((t, step) => {
        const { lv } = level(step);
        const typeT = scene.typeTime ?? 1.4;
        const text = scene.text ?? scene.pieces.join('');
        const nChars = Math.round(clamp((t - levelStart(0, step)) / typeT) * text.length);
        const won = scene.chips === false && lv >= 4 && winner ? ease((t - levelStart(4, step)) / TRANSITION) : 0;
        typed.innerHTML =
          esc(text.slice(0, nChars)) +
          (won ? ` <em class="hl tok-won" style="opacity:${won.toFixed(2)}">${esc(winner.t)}</em>` : '') +
          (nChars < text.length || lv === 0 ? '<span class="caret"></span>' : '');
        typed.style.opacity = lv >= 1 && scene.chips !== false ? '0.35' : '1';
        const tp = levelStart(1, step);
        let key = `${nChars}|${lv}`;
        pieces.forEach((chip, i) => {
          const g = lv >= 1 ? ease((t - tp - i * 0.14) / TRANSITION) : 0;
          chip.style.opacity = String(g);
          chip.style.transform = `translateY(${(1 - g) * 30}px)`;
          chip.classList.toggle('show-id', lv >= 2);
          const gi = lv >= 2 ? ease((t - levelStart(2, step) - i * 0.14) / TRANSITION) : 0;
          chip.style.setProperty('--id', String(gi));
          key += `${g.toFixed(2)}${gi.toFixed(2)}`;
        });
        const tc = levelStart(3, step);
        cands.style.opacity = lv >= 3 ? '1' : '0';
        candRows.forEach((r, i) => {
          const g = lv >= 3 ? ease((t - tc - i * 0.12) / MORPH) : 0;
          r.bar.style.width = `${r.p * g}%`;
          r.val.textContent = `${fmtNum(r.p * g, 1)}%`;
          r.row.classList.toggle('lead', lv >= 4 && i === 0);
          key += g.toFixed(3);
        });
        const ga = lv >= 4 ? ease((t - levelStart(4, step)) / (TRANSITION * 1.5)) : 0;
        add.style.opacity = String(ga);
        add.style.transform = `translate(${(1 - ga) * 120}px, ${(1 - ga) * 260}px) scale(${1 + 0.25 * (1 - ga)})`;
        return key + ga.toFixed(3) + won.toFixed(2);
      });
    },

    // "Pause and guess": a question, options, a countdown ring, then the answer.
    //   question, options [..], answer (0-based); steps: countdown: N, reveal: true
    quiz(scene, root) {
      root.append(stepEl($('p', 'kicker', md(scene.kicker ?? 'Pause and guess')), 0));
      root.append(stepEl($('h1', 'heading quiz-q', md(scene.question)), 0));
      const opts = $('div', 'quiz-opts');
      const optStep = scene.optionsStep ?? 0;
      const cards = scene.options.map((o, i) => {
        const c = stepEl($('div', 'quiz-opt'), optStep);
        c.append($('span', 'quiz-letter', 'ABCD'[i]), $('span', 'quiz-text', md(o)));
        opts.append(c);
        return c;
      });
      root.append(opts);
      const ring = $('div', 'quiz-ring');
      ring.innerHTML = '<svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="52"/><circle class="fill" cx="60" cy="60" r="52"/></svg><span></span>';
      root.append(ring);
      const fill = ring.querySelector('.fill');
      const num = ring.querySelector('span');
      const C = 2 * Math.PI * 52;
      fill.style.strokeDasharray = String(C);
      if (scene.explain) root.append(stepEl($('p', 'quiz-explain', md(scene.explain)), scene.steps.findIndex((s) => s.reveal)));
      effect((t, step) => {
        const st = scene.steps[step];
        const revealed = lastWith(scene.steps, step, 'reveal') >= 0;
        cards.forEach((c, i) => {
          c.classList.toggle('right', revealed && i === scene.answer);
          c.classList.toggle('wrong', revealed && i !== scene.answer);
        });
        if (st.countdown) {
          const u = clamp((t - state.timing.steps[step].t0) / st.countdown);
          ring.style.opacity = String(clamp((t - state.timing.steps[step].t0) / 0.3) * (u < 1 ? 1 : 0.4));
          fill.style.strokeDashoffset = String(C * u);
          num.textContent = String(Math.max(1, Math.ceil(st.countdown * (1 - u))));
          return `c${u.toFixed(3)}${revealed}`;
        }
        ring.style.opacity = '0';
        return `r${revealed}`;
      });
    },

    // Where this part sits in its series: a grid of every part, "you are here" on the current one.
    seriesmap(scene, root) {
      header(scene, root);
      const grid = $('div', 'smap');
      grid.style.setProperty('--cols', scene.cols ?? (spec.format === 'portrait' ? 2 : 4));
      const tiles = scene.parts.map((p) => {
        const tile = $('div', `smap-tile${p.n === scene.current ? ' here' : ''}`);
        tile.append($('span', 'smap-n', String(p.n)), $('span', 'smap-title', md(p.title)));
        if (p.n === scene.current) tile.append($('span', 'smap-here', 'You are here'));
        grid.append(tile);
        return { tile, n: p.n };
      });
      root.append(grid);
      effect((t, step) => {
        const hi = lastWith(scene.steps, step, 'highlight');
        const lit = new Set(hi >= 0 ? [].concat(scene.steps[hi].highlight) : []);
        let key = '';
        tiles.forEach(({ tile, n }, i) => {
          const g = ease((t - i * 0.06) / TRANSITION);
          tile.style.opacity = String(lit.size && !lit.has(n) && n !== scene.current ? 0.35 * g : g);
          tile.style.transform = `scale(${0.9 + 0.1 * g})`;
          tile.classList.toggle('lit', lit.has(n));
          key += g.toFixed(2);
        });
        return key + [...lit].join(',');
      });
    },

    code(scene, root) {
      header(scene, root);
      root.append(stepEl($('div', 'codeblock', md(scene.code).replace(/(#[^\n]*)/g, '<span class="c">$1</span>')), 0));
      if (scene.caption) root.append(stepEl($('p', 'code-caption', md(scene.caption)), scene.captionStep ?? 0));
    },

    brand(scene, root, assets) {
      const img = $('img');
      img.src = assets.banner;
      img.alt = '';
      root.append(stepEl(img, 0));
    },

    outro(scene, root, assets) {
      const logo = $('img', 'outro-logo');
      logo.src = assets.logo;
      logo.alt = '';
      root.append(stepEl(logo, 0));
      root.append(stepEl($('div', 'outro-title', md(scene.title ?? 'Read the full article')), 0));
      root.append(stepEl($('div', 'outro-url', esc(scene.url ?? 'vidyal.ai')), 0));
      if (scene.path) root.append(stepEl($('div', 'outro-path', esc(scene.path)), 0));
      if (scene.next) root.append(stepEl($('div', 'outro-next', md(scene.next)), scene.nextStep ?? 1));
    },

    // End card for YouTube end screens: a 16:9 frame for the video element and a circle for the
    // Subscribe element, both empty so the elements YouTube overlays sit inside them. With `teaser`,
    // the heading names the next episode as "coming soon".
    endcard(scene, root) {
      // With a teaser (next episode not out yet), its title is the heading and the video frame stays
      // empty: YouTube draws end-screen elements see-through in the editor, so anything inside the
      // frame would show behind the Playlist/Video element placed there.
      const top = $('div', 'endcard-top');
      const kicker = scene.teaser ? `Next in the series · ${scene.teaserNote ?? 'Coming soon'}` : scene.kicker;
      if (kicker) top.append($('p', 'kicker', md(kicker)));
      top.append($('h1', scene.teaser ? 'heading heading-teaser' : 'heading', md(scene.teaser ?? scene.title ?? 'Keep learning')));
      root.append(stepEl(top, 0));
      const slots = $('div', 'endcard-slots');
      const video = $('div', 'slot-wrap');
      video.append($('div', 'slot-label', md(scene.videoLabel ?? 'Watch next')), $('div', 'slot slot-video'));
      const sub = $('div', 'slot-wrap slot-wrap-sub');
      sub.append($('div', 'slot slot-subscribe'), $('div', 'slot-label', md(scene.subscribeLabel ?? 'Subscribe')));
      slots.append(video, sub);
      root.append(stepEl(slots, 0));
      root.append(stepEl($('div', 'endcard-url', `<b>${esc(scene.url ?? 'vidyal.ai')}</b>${scene.path ? `<span>${esc(scene.path)}</span>` : ''}`), 0));
    },

    thumb(scene, root, assets) {
      const top = $('div', 'thumb-top', `<img src="${assets.logo}" alt=""><span>Vidyal<b>AI</b></span>`);
      root.append(top);
      const art = $('img', assets.diagram ? 'thumb-art' : 'thumb-logo');
      art.src = assets.diagram ?? assets.logo;
      art.alt = '';
      root.append(art);
      root.append($('div', 'thumb-title', md(scene.title)));
      if (scene.badge) root.append($('div', 'thumb-badge', md(scene.badge)));
    },
  };

  // Line chart on linear or log axes. series: [{ name, points: [[x, y, label?], ...] }];
  // xTicks/yTicks: [[value, label], ...]; step `draw: <name>` draws that series over `drawTime`.
  function lineChart(scene, root) {
    header(scene, root);
    const W = spec.format === 'portrait' ? 900 : 1640;
    const H = spec.format === 'portrait' ? 900 : 640;
    const pad = { l: 150, r: 60, t: 30, b: 80 };
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'linechart', width: W, height: H });
    const all = scene.series.flatMap((s) => s.points);
    const xs = all.map((p) => p[0]);
    const ys = all.map((p) => p[1]);
    const ext = (v, log, lo, hi) => (log ? [Math.log10(lo ?? Math.min(...v)), Math.log10(hi ?? Math.max(...v))] : [lo ?? Math.min(...v), hi ?? Math.max(...v)]);
    const [x0, x1] = ext(xs, scene.xLog, scene.xMin, scene.xMax);
    const [y0, y1] = ext(ys, scene.yLog, scene.yMin, scene.yMax);
    const X = (x) => pad.l + (((scene.xLog ? Math.log10(x) : x) - x0) / (x1 - x0)) * (W - pad.l - pad.r);
    const Y = (y) => H - pad.b - (((scene.yLog ? Math.log10(y) : y) - y0) / (y1 - y0)) * (H - pad.t - pad.b);
    for (const [v, label] of scene.yTicks ?? []) {
      svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: Y(v), y2: Y(v), class: 'grid' }));
      const tx = svgEl('text', { x: pad.l - 18, y: Y(v) + 10, class: 'tick', 'text-anchor': 'end' });
      tx.textContent = label;
      svg.append(tx);
    }
    for (const [v, label] of scene.xTicks ?? []) {
      const tx = svgEl('text', { x: X(v), y: H - pad.b + 50, class: 'tick', 'text-anchor': 'middle' });
      tx.textContent = label;
      svg.append(tx);
    }
    svg.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b, class: 'axis' }));
    const lines = scene.series.map((s, si) => {
      const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
      const path = svgEl('path', { d, class: `series s${si}` });
      svg.append(path);
      const marks = s.points.map((p) => {
        const g = svgEl('g', { class: `pt s${si}` });
        g.append(svgEl('circle', { cx: X(p[0]), cy: Y(p[1]), r: 9 }));
        if (p[2]) {
          const right = X(p[0]) > W * 0.72;
          const tx = svgEl('text', { x: X(p[0]) + (right ? -18 : 18), y: Y(p[1]) - 18, 'text-anchor': right ? 'end' : 'start', class: 'pt-label' });
          tx.textContent = p[2];
          g.append(tx);
        }
        svg.append(g);
        return { g, x: X(p[0]) };
      });
      return { path, marks, name: s.name, x0: X(s.points[0][0]), x1: X(s.points.at(-1)[0]) };
    });
    const wrap = $('div', 'linechart-wrap');
    wrap.append(svg);
    root.append(wrap);
    if (scene.note) root.append(stepEl($('p', 'chart-note', md(scene.note)), scene.noteStep ?? 0));
    effect((t, step) => {
      let key = '';
      for (const l of lines) {
        // Drawn from the first step that names it (or step 0 when no step does).
        let s = scene.steps.findIndex((st) => st.draw === l.name);
        if (s < 0 && !scene.steps.some((st) => st.draw)) s = 0;
        const t0 = s >= 0 ? state.timing.steps[s].t0 : Infinity;
        const dt = scene.steps[s]?.drawTime ?? Math.min(6, state.timing.steps[Math.max(0, s)]?.dur ?? 3);
        const u = s >= 0 && s <= step ? clamp((t - t0) / dt) : 0;
        const len = l.path.getTotalLength();
        l.path.style.strokeDasharray = `${len}`;
        l.path.style.strokeDashoffset = `${len * (1 - u)}`;
        const head = lerp(l.x0, l.x1, u);
        for (const m of l.marks) m.g.style.opacity = u > 0 && m.x <= head + 1 ? '1' : '0';
        key += u.toFixed(4);
      }
      return key;
    });
  }

  function placeCamera(focus) {
    const { viewport, svg, camera } = state;
    const vw = viewport.clientWidth;
    const vh = viewport.clientHeight;
    const [x, y, w, h] = focus;
    const { w: iw, h: ih } = state.imgSize;
    const s = Math.min(vw / (w * iw), vh / (h * ih)) * 0.94;
    // Lay the SVG out at the zoomed size (not a CSS scale) so Chrome rasterizes it crisply.
    svg.style.width = `${iw * s}px`;
    svg.style.height = `${ih * s}px`;
    camera.style.left = `${vw / 2 - (x + w / 2) * iw * s}px`;
    camera.style.top = `${vh / 2 - (y + h / 2) * ih * s}px`;
  }

  // ---------- Overlays: second-voice bubble, burned-in word captions ----------
  function speakerBubble(t, step) {
    const bubble = document.getElementById('speaker');
    const st = state.timing.steps[step];
    if (!st?.speaker) {
      bubble.style.opacity = '0';
      return '';
    }
    const g = ease((t - st.t0) / TRANSITION);
    bubble.querySelector('.who').textContent = st.speakerLabel ?? st.speaker;
    bubble.querySelector('.what').textContent = st.say;
    bubble.style.opacity = String(g);
    bubble.style.transform = `translateY(${(1 - g) * 24}px)`;
    return `s${step}${g.toFixed(2)}`;
  }

  // Shorts: the phrase being spoken, 1-4 words at a time, with the current word lit.
  function captions(t) {
    const box = document.getElementById('captions');
    if (!spec.fx?.captions) return '';
    for (const st of state.timing.steps)
      for (const s of st.sentences ?? []) {
        if (t < s.start || t > s.end + 0.25 || !s.words?.length) continue;
        const groups = [];
        let cur = [];
        for (const w of s.words) {
          if (cur.length && (cur.length >= 4 || cur.map((x) => x.w).join(' ').length + w.w.length > 18)) {
            groups.push(cur);
            cur = [];
          }
          cur.push(w);
          if (/[,.;:!?]$/.test(w.w)) {
            groups.push(cur);
            cur = [];
          }
        }
        if (cur.length) groups.push(cur);
        const g = groups.find((gr) => t < gr.at(-1).end + 0.05) ?? groups.at(-1);
        const active = g.findLastIndex((w) => t >= w.start);
        box.innerHTML = g.map((w, i) => `<span class="${i === active ? 'on' : i < active ? 'said' : ''}">${esc(w.w)}</span>`).join(' ');
        box.style.opacity = '1';
        return `c${groups.indexOf(g)}.${active}${s.start}`;
      }
    box.style.opacity = '0';
    return '';
  }

  // **Highlighted** phrases glow briefly when the narration says them.
  function wordPulse(t) {
    if (!spec.fx?.wordSync) return '';
    let key = '';
    for (const h of state.hls) {
      let v = 0;
      for (const e of h.events) {
        const k = (t - e) / PULSE;
        if (k >= 0 && k < 1) v = Math.max(v, Math.sin(Math.PI * Math.sqrt(k)));
      }
      v = Math.round(v * 20) / 20;
      h.el.style.textShadow = v ? `0 0 ${Math.round(28 * v)}px rgb(34 211 238 / ${(0.85 * v).toFixed(2)})` : '';
      h.el.style.color = v ? `color-mix(in srgb, var(--cyan) ${Math.round(100 - 60 * v)}%, white)` : '';
      key += v;
    }
    return key;
  }

  window.stage = {
    async load(s) {
      spec = s;
      const frame = document.getElementById('frame');
      frame.className = `frame ${spec.format}${spec.chrome === false ? ' no-chrome' : ''}${spec.progress === false ? ' no-progress' : ''}${spec.fx?.captions ? ' burn' : ''}`;
      document.getElementById('brand-mark').src = spec.assets.logo;
      document.getElementById('chapter').textContent = spec.chapter ?? '';
      const root = document.getElementById('scene');
      root.className = `scene t-${spec.scene.type}`;
      root.style.transform = '';
      root.innerHTML = '';
      // Without timing (thumbnails, previews), each step lasts 4 s.
      const timing = spec.timing ?? {
        dur: 4 * spec.scene.steps.length,
        steps: spec.scene.steps.map((st, i) => ({ t0: 4 * i, dur: 4, speaker: st.speaker, speakerLabel: st.speaker, say: String(st.say ?? '').replace(/\*\*/g, '') })),
      };
      state = { steps: spec.scene.steps.length, timing, cameras: [], counters: [], bars: [], effects: [], hls: [], viewport: null, svg: null, imgSize: null };
      const build = builders[spec.scene.type];
      if (!build) throw new Error(`Unknown scene type: ${spec.scene.type}`);
      build(spec.scene, root, spec.assets);
      await document.fonts.ready;
      await Promise.all([...document.images].map((im) => (im.complete ? im.decode().catch(() => {}) : im.decode())));
      if (state.svg) state.imgSize = spec.assets.diagramSize;
      // Word-sync: when is each highlighted phrase's first word spoken in this scene?
      const spoken = timing.steps.flatMap((st) => (st.sentences ?? []).flatMap((s) => s.words ?? []));
      for (const el of root.querySelectorAll('.hl')) {
        const first = norm(el.textContent.split(/\s+/)[0]);
        state.hls.push({ el, events: spoken.filter((w) => norm(w.w) === first).map((w) => w.start) });
      }
      return state.steps;
    },

    // t: seconds since the scene started; globalT: since the video started; progress: 0..1.
    frame(t, globalT = 0, progress = 0) {
      const steps = state.timing.steps;
      let step = 0;
      for (let i = 0; i < steps.length; i++) if (steps[i].t0 <= t + 1e-6) step = i;
      const p = spec.instant && step === 0 ? 1 : clamp((t - steps[step].t0) / TRANSITION);
      const e = ease(p);
      const keys = [step, p.toFixed(3)];

      const bar = Math.round((progress ?? 0) * 1920);
      document.getElementById('progress-bar').style.width = `${bar}px`;
      keys.push(bar);
      for (const el of document.querySelectorAll('#scene [data-step]')) {
        const s = Number(el.dataset.step);
        const shown = s < step ? 1 : s === step ? e : 0;
        el.style.opacity = String(shown);
        el.style.transform = shown < 1 ? `translateY(${(1 - shown) * 28}px)` : '';
      }
      for (const el of document.querySelectorAll('#scene [data-only]')) {
        const s = Number(el.dataset.only);
        el.style.opacity = s === step ? String(e) : '0';
        el.style.transform = s === step && e < 1 ? `translateY(${(1 - e) * 20}px)` : '';
      }
      for (const c of state.counters) {
        const v = c.step < step ? c.target : c.step === step ? c.target * e : 0;
        let num = v.toFixed(c.decimals);
        if (c.comma) num = Number(num).toLocaleString('en-US', { minimumFractionDigits: c.decimals, maximumFractionDigits: c.decimals });
        c.el.textContent = `${c.pre}${num}${c.post}`;
      }
      for (const b of state.bars) {
        const k = b.step < step ? 1 : b.step === step ? e : 0;
        b.el.style.width = `${b.width * k}%`;
      }
      if (state.statCards) {
        const latest = state.statCards.filter((c) => Number(c.dataset.step) <= step).pop();
        state.statCards.forEach((c) => c.classList.toggle('focus', c === latest));
      }
      if (state.viewport && state.imgSize) {
        const from = state.cameras[Math.max(0, step - 1)];
        const to = state.cameras[step];
        const k = step === 0 ? 1 : easeSine((t - steps[step].t0) / CAMERA);
        placeCamera(camLerp(from, to, k));
        keys.push(`cam${k.toFixed(4)}`);
      }
      for (const fn of state.effects) keys.push(fn(t, step, p));

      // Drift: the scene pushes in slowly and the dot grid pans, so no frame is ever a still.
      const drift = spec.fx?.drift && !['endcard', 'thumb'].includes(spec.scene.type);
      if (drift) {
        const u = clamp(t / Math.max(1, state.timing.dur));
        const root = document.getElementById('scene');
        root.style.transformOrigin = '50% 45%';
        root.style.transform = `scale(${(1 + 0.028 * u).toFixed(5)})`;
        document.querySelector('.bg-grid').style.backgroundPosition = `${(globalT * 7).toFixed(2)}px ${(globalT * 3.5).toFixed(2)}px`;
        keys.push(t.toFixed(3));
      }
      keys.push(speakerBubble(t, step), captions(t), wordPulse(t));
      return keys.join('/');
    },

    // Stills (thumbnails, previews): the end of step i, when its animations have finished.
    set(step) {
      const st = state.timing.steps[step];
      return this.frame(st.t0 + st.dur - 0.01, st.t0 + st.dur, 0.5);
    },
  };
})();
