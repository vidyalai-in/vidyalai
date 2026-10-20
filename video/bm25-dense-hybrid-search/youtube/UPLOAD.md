# YouTube upload checklist: BM25 vs Dense vs Hybrid Search: When to Use Each | LLM System Design

Videos are in `video/bm25-dense-hybrid-search/out/` (not committed; rebuild with `node video/tools/build.mjs video/bm25-dense-hybrid-search`).
Everything to paste is in this folder. Upload the main video first: the Shorts link to it.

## Once per channel

- **Verify the channel by phone** at https://www.youtube.com/verify. Custom thumbnails need it (Studio → Settings →
  Channel → Feature eligibility shows what is unlocked). Chapters need the channel to have no active strikes; if they
  don't appear, apply for Advanced features on the same page.
- Create the playlist **LLM System Design** (it can also be created from the upload form).

## Main video: `out/bm25-dense-hybrid-search.mp4`

1. YouTube Studio → **Create → Upload videos** → select the MP4.
2. **Details**
   - **Title**: first line of `main/title.txt`.
   - **Description**: paste `main/description.txt`. Its timestamps (starting at 00:00) become the chapters.
   - **Thumbnail**: upload `main/thumbnail.png` (1280×720 PNG).
   - **Playlists**: LLM System Design.
   - **Audience**: "No, it's not made for kids".
   - **Attributes → AI use**: **Yes**. The narration is an AI-generated voice. YouTube's policy only requires
     disclosure for realistic depictions of real people or events, but it labels AI-generated non-music audio in the
     expanded description, which matches the credit line already in the description.
   - **Show more**: **Tags**: paste `main/tags.txt`. **Language**: English. **Caption certification**: "This content
     has never aired on television in the U.S." **Category**: Education.
3. **Video elements**
   - **Add subtitles → Upload file → With timing** → `main/captions.en.srt`.
   - **Add end screen**: Subscribe + "Best for viewer" over the last 15-20 s; check that the elements don't cover on-screen text.
4. **Checks**: wait for the copyright check to finish.
5. **Visibility**: Public, or schedule. After publishing, pin the comment from `main/pinned-comment.txt`.

## Shorts

Vertical videos up to 3 minutes are published as Shorts automatically. Links in Shorts descriptions and comments are
not clickable, so each Short links to the main video with **Related video** instead.

### `out/bm25-dense-hybrid-search--short-rrf-math.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-rrf-math/title.txt`. **Description**: `shorts/short-rrf-math/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-rrf-math/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/bm25-dense-hybrid-search--short-dont-add-scores.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-dont-add-scores/title.txt`. **Description**: `shorts/short-dont-add-scores/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-dont-add-scores/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/bm25-dense-hybrid-search--short-gains-stack.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-gains-stack/title.txt`. **Description**: `shorts/short-gains-stack/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-gains-stack/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/bm25-dense-hybrid-search--short-pooling-bias.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-pooling-bias/title.txt`. **Description**: `shorts/short-pooling-bias/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-pooling-bias/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.
