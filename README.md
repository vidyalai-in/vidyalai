# VidyalAI

Source for [vidyal.ai](https://vidyal.ai): in-depth, sourced articles on designing production LLM systems.
Built with [Astro](https://astro.build), deployed to GitHub Pages.

## Develop

```sh
npm install
npm run dev        # http://localhost:4321
npm run build      # static site in dist/, plus the Pagefind search index
```

## Content

| What | Where |
|---|---|
| Articles (Markdown, source of truth) | `src/content/blog/NN-slug.md` |
| Authoring rules: structure, tone, frontmatter, diagrams | `content-src/AUTHORING.md` |
| Review of each original answer, and the roll-up | `content-src/reviews/`, `content-src/REVIEW.md` |
| Snapshots of the original Google Docs | `content-src/original/` |
| Categories and series metadata | `src/data/categories.ts` |
| Diagram sources (Mermaid) | `diagrams/*.mmd` |

Diagrams are pre-rendered and committed, so CI doesn't need a browser. After editing a `.mmd` file:

```sh
npm run diagrams   # writes public/diagrams/<name>.svg, <name>-dark.svg and <name>.png
```

Reference a diagram from an article with
`![Full alt text](/diagrams/<name>.svg "Caption")` on its own line. The build turns it into a figure
that follows the light/dark theme.

## Videos

Each article can become a narrated YouTube video, Shorts and an upload kit. The Claude Code skill
`.claude/skills/article-to-video/` describes the whole process; in short:

```sh
node video/tools/preview.mjs video/<slug> main 3:1   # still of scene 3, step 1
node video/tools/build.mjs video/<slug>              # TTS (Kokoro, local), render, encode, upload kit
```

The source is `video/<slug>/script.yaml`; the upload kit (titles, descriptions, chapters, captions,
thumbnail, `UPLOAD.md`) is written to `video/<slug>/youtube/`. Rendered MP4s go to
`video/<slug>/out/` and are not committed. Needs `ffmpeg`, `uv` and (optionally) an NVIDIA GPU.

## Checks

```sh
npm run check:links    # every external link resolves
npm run check:arxiv    # every arXiv ID matches the cited paper title
npx astro preview &    # then:
npm run check:a11y     # axe-core WCAG 2.2 AA over every page, light and dark
npm run screenshots    # screenshots at several widths into .screenshots/
```

## Deploy

Pushing to `main` runs `.github/workflows/deploy.yml`, which builds the site and publishes it to
GitHub Pages. In the repo settings, Pages → Source must be **GitHub Actions**.

### Custom domains

GitHub Pages serves one custom domain per site. `vidyal.ai` is primary (`public/CNAME`), and
`vidyalai.in` redirects to it.

**vidyal.ai** (at its DNS provider):

| Type | Name | Value |
|---|---|---|
| A | `@` | `185.199.108.153` |
| A | `@` | `185.199.109.153` |
| A | `@` | `185.199.110.153` |
| A | `@` | `185.199.111.153` |
| AAAA | `@` | `2606:50c0:8000::153` |
| AAAA | `@` | `2606:50c0:8001::153` |
| AAAA | `@` | `2606:50c0:8002::153` |
| AAAA | `@` | `2606:50c0:8003::153` |
| CNAME | `www` | `vidyalai-in.github.io` |

Then, in GitHub: verify `vidyal.ai` under the **vidyalai-in organization settings → Pages → Add a domain** (adds a TXT
record; protects against domain takeover), set the custom domain in the repo's Pages settings, and
tick **Enforce HTTPS** once the certificate is issued.

**vidyalai.in** → 301 to `https://vidyal.ai`. Either use the registrar's URL forwarding (choose a
permanent redirect that preserves the path, with HTTPS support), or move its DNS to Cloudflare
(free plan), add proxied placeholder records (`A @ 192.0.2.1`, `CNAME www vidyalai.in`), and add a
Redirect Rule: *all incoming requests* → dynamic `concat("https://vidyal.ai", http.request.uri.path)`,
status 301, preserve query string.
