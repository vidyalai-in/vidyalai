# YouTube upload checklist: Designing a RAG System for 10 Million Documents | LLM System Design

Videos are in `video/rag-system-10m-documents/out/` (not committed; rebuild with `node video/tools/build.mjs video/rag-system-10m-documents`).
Everything to paste is in this folder. Upload the main video first: the Shorts link to it.

## Once per channel

- **Verify the channel by phone** at https://www.youtube.com/verify. Custom thumbnails need it (Studio → Settings →
  Channel → Feature eligibility shows what is unlocked). Chapters need the channel to have no active strikes; if they
  don't appear, apply for Advanced features on the same page.
- Create the playlist **LLM System Design** (it can also be created from the upload form).

## Main video: `out/rag-system-10m-documents.mp4`

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
   - **Add end screen**: the last 20 s (from 10:59) is a blank end card. Choose
     **Add element → Video** (Best for viewer, or the next video in the series) and **Subscribe**. Drag the video element
     into the dashed 16:9 frame on the left and the Subscribe element into the dashed circle on the right, resize them to
     fill the frames, and set both to run for the whole end card.
4. **Checks**: wait for the copyright check to finish.
5. **Visibility**: Public, or schedule. After publishing, pin the comment from `main/pinned-comment.txt`.

## Shorts

Vertical videos up to 3 minutes are published as Shorts automatically. Links in Shorts descriptions and comments are
not clickable, so each Short links to the main video with **Related video** instead.

### `out/rag-system-10m-documents--short-index-sizing.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-index-sizing/title.txt`. **Description**: `shorts/short-index-sizing/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-index-sizing/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/rag-system-10m-documents--short-retrieval-first.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-retrieval-first/title.txt`. **Description**: `shorts/short-retrieval-first/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-retrieval-first/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/rag-system-10m-documents--short-acl-filter.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-acl-filter/title.txt`. **Description**: `shorts/short-acl-filter/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-acl-filter/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/rag-system-10m-documents--short-latency-budget.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-latency-budget/title.txt`. **Description**: `shorts/short-latency-budget/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-latency-budget/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.
