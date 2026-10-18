# YouTube upload checklist: Debugging a RAG System That Gives Wrong Answers | LLM System Design

Videos are in `video/debugging-rag-wrong-answers/out/` (not committed; rebuild with `node video/tools/build.mjs video/debugging-rag-wrong-answers`).
Everything to paste is in this folder. Upload the main video first: the Shorts link to it.

## Once per channel

- **Verify the channel by phone** at https://www.youtube.com/verify. Custom thumbnails need it (Studio → Settings →
  Channel → Feature eligibility shows what is unlocked). Chapters need the channel to have no active strikes; if they
  don't appear, apply for Advanced features on the same page.
- Create the playlist **LLM System Design** (it can also be created from the upload form).

## Main video: `out/debugging-rag-wrong-answers.mp4`

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
   - **Add end screen**: the last 12 s (from 10:30:02 in Studio's m:ss:frames format) is the end card. Choose
     **Add element → Subscribe**, drag it into the dashed circle on the right, resize it to fill the circle, and set it
     to run for the whole end card (drag the element's bar in the timeline if the typed time turns red). YouTube requires at least one video or playlist
     element, so also add **Playlist** → the series playlist and drag it over the left "coming soon" panel, resized to
     fill it. **When the next video is public**, come back to this video's end screen and replace the playlist with
     **Video → Choose specific video** (same position and size). End screens can be edited any time; no re-upload needed.
4. **Checks**: wait for the copyright check to finish.
5. **Visibility**: Public, or schedule. After publishing, pin the comment from `main/pinned-comment.txt`.

## Shorts

Vertical videos up to 3 minutes are published as Shorts automatically. Links in Shorts descriptions and comments are
not clickable, so each Short links to the main video with **Related video** instead.

### `out/debugging-rag-wrong-answers--short-gold-only.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-gold-only/title.txt`. **Description**: `shorts/short-gold-only/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-gold-only/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/debugging-rag-wrong-answers--short-eval-noise.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-eval-noise/title.txt`. **Description**: `shorts/short-eval-noise/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-eval-noise/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/debugging-rag-wrong-answers--short-exact-vs-ann.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-exact-vs-ann/title.txt`. **Description**: `shorts/short-exact-vs-ann/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-exact-vs-ann/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/debugging-rag-wrong-answers--short-bad-label.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-bad-label/title.txt`. **Description**: `shorts/short-bad-label/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-bad-label/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.
