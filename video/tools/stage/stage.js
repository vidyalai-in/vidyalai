// VidyalAI video stage. build.mjs calls stage.load(spec) once per scene, then stage.set(step, p)
// for every frame it captures. `step` is the narration step being shown and `p` (0..1) is how far
// its entrance animation has progressed. Elements carry data-step="n" (visible from step n on) or
// data-only="n" (visible only during step n).

(() => {
  const $ = (tag, cls, html) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (html !== undefined) el.innerHTML = html;
    return el;
  };
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  // **highlight** and `code` inline markup.
  const md = (s) =>
    esc(s ?? '')
      .replace(/\*\*(.+?)\*\*/g, '<em class="hl">$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>');
  const ease = (t) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
  const lerp = (a, b, t) => a + (b - a) * t;

  let state = { steps: 0, cameras: [], counters: [], bars: [], viewport: null, img: null, imgSize: null };

  function stepEl(el, step) {
    el.dataset.step = String(step);
    return el;
  }

  function header(scene, parent) {
    if (scene.kicker) parent.append(stepEl($('p', 'kicker', md(scene.kicker)), 0));
    if (scene.heading) parent.append(stepEl($('h1', 'heading', md(scene.heading)), 0));
  }

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

    diagram(scene, root, assets) {
      header(scene, root);
      const viewport = $('div', 'viewport');
      const camera = $('div', 'camera');
      const img = $('img');
      img.src = assets.diagram;
      img.alt = '';
      camera.append(img);
      viewport.append(camera);
      // Notes sit in a caption strip under the diagram so they never cover the boxes being discussed.
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
      state.viewport = viewport;
      state.img = img;
      state.camera = camera;
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
        const value = $('div', 'value', esc(st.add.value));
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
    // the frame instead shows the next episode as "coming soon"; a video element can cover it later.
    endcard(scene, root, assets) {
      const top = $('div', 'endcard-top');
      if (scene.kicker) top.append($('p', 'kicker', md(scene.kicker)));
      top.append($('h1', 'heading', md(scene.title ?? 'Keep learning')));
      root.append(stepEl(top, 0));
      const slots = $('div', 'endcard-slots');
      const video = $('div', 'slot-wrap');
      const frame = $('div', 'slot slot-video');
      if (scene.teaser) {
        frame.classList.add('slot-teaser');
        frame.append($('p', 'teaser-title', md(scene.teaser)), $('p', 'teaser-soon', md(scene.teaserNote ?? 'Coming soon')));
      }
      video.append($('div', 'slot-label', md(scene.videoLabel ?? (scene.teaser ? 'Next in the series' : 'Watch next'))), frame);
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

  function placeCamera(focus) {
    const { viewport, img, imgSize, camera } = state;
    const vw = viewport.clientWidth;
    const vh = viewport.clientHeight;
    const [x, y, w, h] = focus;
    const s = Math.min(vw / (w * imgSize.w), vh / (h * imgSize.h)) * 0.94;
    // Lay the SVG out at the zoomed size (not a CSS scale) so Chrome rasterizes it crisply.
    img.style.width = `${imgSize.w * s}px`;
    img.style.height = `${imgSize.h * s}px`;
    camera.style.left = `${vw / 2 - (x + w / 2) * imgSize.w * s}px`;
    camera.style.top = `${vh / 2 - (y + h / 2) * imgSize.h * s}px`;
  }

  window.stage = {
    async load(spec) {
      const frame = document.getElementById('frame');
      frame.className = `frame ${spec.format}${spec.chrome === false ? ' no-chrome' : ''}${spec.progress === false ? ' no-progress' : ''}`;
      document.getElementById('brand-mark').src = spec.assets.logo;
      document.getElementById('chapter').textContent = spec.chapter ?? '';
      const root = document.getElementById('scene');
      root.className = `scene t-${spec.scene.type}`;
      root.innerHTML = '';
      state = { steps: spec.scene.steps.length, cameras: [], counters: [], bars: [], viewport: null, img: null, imgSize: null };
      const build = builders[spec.scene.type];
      if (!build) throw new Error(`Unknown scene type: ${spec.scene.type}`);
      build(spec.scene, root, spec.assets);
      await document.fonts.ready;
      await Promise.all([...document.images].map((im) => (im.complete ? im.decode().catch(() => {}) : im.decode())));
      // Mermaid SVGs say width="100%", so take their real size from the viewBox (build.mjs reads it).
      if (state.img) state.imgSize = spec.assets.diagramSize ?? { w: state.img.naturalWidth, h: state.img.naturalHeight };
      return state.steps;
    },

    set(step, p, progress) {
      const t = ease(p);
      document.getElementById('progress-bar').style.width = `${(progress ?? 0) * 100}%`;
      for (const el of document.querySelectorAll('[data-step]')) {
        const s = Number(el.dataset.step);
        const shown = s < step ? 1 : s === step ? t : 0;
        el.style.opacity = String(shown);
        el.style.transform = shown < 1 ? `translateY(${(1 - shown) * 28}px)` : '';
      }
      for (const el of document.querySelectorAll('[data-only]')) {
        const s = Number(el.dataset.only);
        el.style.opacity = s === step ? String(t) : '0';
        el.style.transform = s === step && t < 1 ? `translateY(${(1 - t) * 20}px)` : '';
      }
      for (const c of state.counters) {
        const v = c.step < step ? c.target : c.step === step ? c.target * t : 0;
        let num = v.toFixed(c.decimals);
        if (c.comma) num = Number(num).toLocaleString('en-US', { minimumFractionDigits: c.decimals, maximumFractionDigits: c.decimals });
        c.el.textContent = `${c.pre}${num}${c.post}`;
      }
      for (const b of state.bars) {
        const k = b.step < step ? 1 : b.step === step ? t : 0;
        b.el.style.width = `${b.width * k}%`;
      }
      if (state.statCards) {
        const latest = state.statCards.filter((c) => Number(c.dataset.step) <= step).pop();
        state.statCards.forEach((c) => c.classList.toggle('focus', c === latest));
      }
      if (state.viewport && state.imgSize) {
        const from = state.cameras[Math.max(0, step - 1)];
        const to = state.cameras[step];
        const k = step === 0 ? 1 : t;
        placeCamera(from.map((v, i) => lerp(v, to[i], k)));
      }
    },
  };
})();
