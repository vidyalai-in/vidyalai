# Authoring guide: VidyalAI blog posts

Every post belongs to one series (`src/data/series.mjs`):

| Series | `series:` | File prefix | Parts | Template |
|---|---|---|---|---|
| LLM Fundamentals | `llm-fundamentals` | `f` | 16 | [Explainer](#explainer-template-fundamentals-and-inference) |
| LLM Inference & Optimization | `llm-inference` | `i` | 14 | [Explainer](#explainer-template-fundamentals-and-inference) |
| LLM System Design | `llm-system-design` (the default; existing posts omit it) | `q` | 25 | [Post structure](#post-structure) |

LLM System Design posts are standalone engineering articles, each built from one answer in the
"LLM System Design Answers" Google Docs. Repo Markdown is the source of truth: the website builds
from it, and the Google Docs are later rewritten from it. The Fundamentals and Inference series are
written directly in the repo; they have no source doc.

To add a series, add it to `src/data/series.mjs` (and, usually, a matching category below).

## Files

| What | Where |
|---|---|
| Original doc snapshot (verbatim `read_file_content` output) | `content-src/original/qNN-qMM.md` |
| Post | System Design: `src/content/blog/NN-<slug>.md` (NN = zero-padded question number, e.g. `01-rag-system-10m-documents.md`). Other series: `<prefix>NN-<slug>.md` (e.g. `f01-how-llms-work.md`) |
| Review notes for one post | `content-src/reviews/qNN.md` |
| Diagram source | `diagrams/<prefix>NN-<name>.mmd` (Mermaid), e.g. `q01-rag-pipeline.mmd`, `i01-prefill-decode.mmd` |

Posts are plain Markdown (`.md`), not MDX. No imports, no JSX, no raw HTML.

## Frontmatter

```yaml
---
title: "Designing a RAG System for 10 Million Documents"   # topic-style, not a question, ≤ 70 chars
description: "One or two sentences (≤ 160 chars) that say what the reader will learn."
slug: "rag-system-10m-documents"        # kebab-case, ≤ 6 words, unique
category: "retrieval-rag"               # one of the category slugs below
tags: ["rag", "vector-search", "evals"] # 3-6 lowercase kebab-case tags; reuse the shared tag list when possible
series: "llm-fundamentals"               # omit for LLM System Design posts
seriesOrder: 1                          # part number within the series
pubDate: 2026-10-16
sourceDoc: "1h3m8wtZykYFchMjt8t57Aqya0AhrdYy0tlem7E4EI3A"  # System Design only: Drive file ID of the original doc
---
```

Once the article's video is on YouTube, add `video: { youtube: "<11-char ID>", duration: "11:06", url: "<share link, optional>" }` and copy
its thumbnail to `src/assets/video/<slug>.png`. The post then shows a click-to-play poster above the
contents; nothing loads from YouTube until the reader presses play.

Category slugs:

| Slug | Name | Posts |
|---|---|---|
| `llm-fundamentals` | How LLMs Work | LLM Fundamentals series |
| `inference-optimization` | Inference & Optimization | LLM Inference & Optimization series |
| `retrieval-rag` | Retrieval & RAG | 1-5 |
| `cost-latency-serving` | Cost, Latency & Serving | 6, 7, 8, 9, 21 |
| `agents-tool-use` | Agents & Tool Use | 10, 11, 12, 13 |
| `evaluation-reliability` | Evaluation & Reliability | 14, 15, 16 |
| `security-privacy` | Security & Privacy | 17, 18 |
| `fine-tuning` | Fine-tuning & Adaptation | 19, 20 |
| `production-engineering` | Production Engineering | 22, 23, 24, 25 |

Shared tags (prefer these; add a new one only when nothing fits): `rag`, `retrieval`,
`vector-search`, `reranking`, `chunking`, `evals`, `llm-as-judge`, `cost`, `latency`, `caching`,
`routing`, `agents`, `tool-use`, `memory`, `structured-output`, `hallucination`, `security`,
`prompt-injection`, `privacy`, `fine-tuning`, `lora`, `serving`, `gpu`, `streaming`,
`observability`, `deployment`, `architecture`.

## Post structure

This section is for LLM System Design posts. The other series use the
[explainer template](#explainer-template-fundamentals-and-inference).

Use exactly these H2 headings, in this order. Omit a section only if the original truly has nothing
for it (and say why in the review file). H3 headings are free-form inside sections. Do not add an
H1: the title comes from frontmatter.

1. `## TL;DR`: from the original "Short answer". One tight paragraph of 3-6 sentences, prose (no bullets).
2. `## Requirements to pin down first`: from "Clarifying questions" / "Clarify first". Phrase as
   questions the team building the system should answer, with why each one changes the design.
3. One or more design sections with descriptive H2 titles (e.g. `## The ingestion pipeline`,
   `## The query path`). This is the bulk of the original "Detailed answer".
4. `## How to measure it`: metrics, targets, eval sets, online monitoring. Write one if the original
   lacked it (Q8, Q15, Q24 in particular).
5. `## Tradeoffs`: a GFM table with a real header row (`| Option | Pros | Cons | When to choose |`),
   followed by the original's 2-3 summary bullets.
6. `## Common mistakes`: from "Pitfalls interviewers probe". Bullets; each states the mistake and its consequence.
7. `## Thinking at scale`: from "What a principal would add" (or the equivalent inline remarks). Org,
   ownership, migration and long-term concerns. Omit if there is nothing.
8. `## Further reading`: from "Sources". Bulleted `[Title (Authors, Year)](url): one-line why it matters`.

## Explainer template (Fundamentals and Inference)

Readers are curious people who want to understand LLMs properly, not only engineers building
products. Every article will later become a YouTube video (see the `article-to-video` skill), so
write it to be read and to be narrated.

Use these H2 headings, in this order:

1. `## TL;DR`: 3-5 plain sentences that state what the reader will understand by the end.
2. `## Why it matters`: open with a question, a surprising number or a puzzle the article answers.
   It becomes the video's cold open, so the first two sentences must hook.
3. `## The intuition`: one everyday analogy, before any math. Say where the analogy breaks.
4. Two to four descriptive H2s that explain how it works, step by step (e.g. `## Step 1: turning
   text into tokens`). Each gets a worked example with real numbers: a small table, a short code
   snippet, or a formula in code font explained in words right after it. At least one diagram.
5. `## In the real world`: where this shows up in real models and systems, with their actual
   numbers (model sizes, speeds, memory), each with a source.
6. `## Common misconceptions`: bullets, each "**The belief.** What's actually true, and why."
7. `## Key takeaways`: 4-6 bullets. They become the video's recap scene.
8. `## What's next`: two or three sentences that tease the next part and link it once it exists.
   It becomes the end-card teaser.
9. `## Further reading`: as below. Primary sources first (papers, official docs), then the best
   explainers (blog posts, talks, videos).

Writing rules for these series, on top of the voice rules below:
- Define every term at first use, in plain words, even ones that feel obvious (token, parameter,
  GPU memory). Expand every acronym.
- Numbers before symbols: work an example with small concrete numbers, then show the general
  formula. Never leave an equation unexplained.
- Short paragraphs (2-4 sentences). Vary the rhythm.
- No system-design framing ("requirements to pin down", "tradeoffs table") unless it helps.
- Link earlier parts and related System Design posts with `/blog/<slug>/` links.
- Length: 2,000-3,500 words. A main video uses 1,300-1,700 narrated words, so the article can
  carry more detail than the video.

Video-readiness checklist (check before publishing):
- A hook in the first two sentences of `## Why it matters`.
- At least three striking numbers that would work as on-screen stat cards.
- At least one diagram or table that can be shown full screen.
- Three self-contained ideas that could each be a 30-50 second Short.
- `## Key takeaways` and `## What's next` written as if spoken.

When the article is done, run the `humanizer` skill on it in file mode (prose only: code,
numbers, links and frontmatter stay as they are), then re-check every number against its source.

## Voice and tone

- Engineering deep-dive for working engineers. Second person ("you") or neutral voice.
- **Never**: "I'd", "I would", "my default", "in my experience", "interviewer", "candidate",
  "principal would add", "say this explicitly", invented first-hand anecdotes ("most loops I've debugged").
- Each post must stand alone. Define jargon on first use (e.g. "TTFT (time to first token)").
  Expand acronyms once. Link related posts with relative links `/blog/<slug>/` instead of "as above".
- Short sentences. One idea per paragraph. Prefer concrete numbers with units over adjectives.
- Keep all the substance of the original: numbers, tables, worked examples, sources. Restructure and
  re-voice; do not shorten into a summary. Improve where the rubric finds gaps.
- Use code formatting for code, protocol frames, config keys, formulas that are code-like
  (e.g. `score(d) = Σ 1/(k + rank_i(d))`). Use fenced code blocks with a language tag for multi-line code.
- No stray escapes: write `~`, `+`, `>`, `&`, `<` as plain characters (inside tables too).
- Plain ASCII-friendly punctuation is fine; avoid em-dash chains.

## Diagrams

Add one diagram per post where structure, flow or sequence matters (target: most posts; skip only
when a picture adds nothing). In the post, reference it as an image whose path ends in `.svg`:

```md
![Alt text that fully describes what the diagram shows, for screen readers.](/diagrams/q01-rag-pipeline.svg "Caption shown under the figure.")
```

Then immediately follow it with one plain-text sentence that summarizes the diagram's key point.

The build renders `diagrams/q01-rag-pipeline.mmd` into light and dark SVGs plus a PNG. Mermaid rules:

- Use `flowchart LR|TD`, `sequenceDiagram`, `stateDiagram-v2`, or `graph` only.
- **No styling**: no `classDef`, `style`, `linkStyle`, `%%{init}%%` or colors. The theme is applied at render time.
- Quote every label that contains spaces or punctuation: `A["Hybrid retrieval (top 200)"]`.
  Avoid `"` inside labels; use `#quot;` if unavoidable. Use `<br/>` for line breaks inside labels.
- Keep it legible: ≤ ~14 nodes, labels ≤ ~5 words per line, subgraphs to group stages.
- Must be accurate to the text; do not add components the post doesn't discuss.

## Glossary terms

Specialised terms get a hover/tap tooltip automatically: the build marks the first use of each
term in every H2 section (not in headings, links, code, diagrams or "Further reading"). Terms
live in `src/data/glossary.mjs`, which also feeds the `/glossary/` page. When a post introduces a
term a reader might not know, add an entry:

- `match`: every form used in posts ("reranker", "rerankers", "reranking"). Phrases with a capital
  letter match case-sensitively; all-lowercase ones match any case.
- `def`: one or two plain sentences that make sense without the article.
- `source`: the paper, standard or official docs that define the term. Check that the URL opens and,
  for arXiv, that the ID's title matches.
- `article`: the post that explains it best, if any.

Still expand an acronym in the text on first use; the tooltip is a convenience, not a substitute.

## Review file (`content-src/reviews/qNN.md`)

```md
# QNN: <post title>

| Dimension | Before (1-5) | After (1-5) | Notes |
|---|---|---|---|
| Correctness | | | |
| Detail | | | |
| Sources | | | |
| Readability | | | |

## Issues found
- ...

## Changes made
- ...

## Sources checked
- url: OK / replaced with ... / added
```

Rubric:
- **Correctness**: recompute every arithmetic claim; check numbers, product names, prices and
  capabilities against primary sources; update anything stale as of October 2026. If a number has
  no source and can't be verified, either cite one or present it as illustrative ("for example").
- **Detail**: fill missing measurement sections; add a worked example where one would clarify.
- **Sources**: WebFetch every link you add or change, and any link that looks suspicious; confirm it
  resolves and supports the claim. Prefer primary sources (papers, official docs). Normalize
  Anthropic links to the current canonical URLs (WebFetch to confirm where they redirect).
- **Readability**: tone rules above; scannable structure.
