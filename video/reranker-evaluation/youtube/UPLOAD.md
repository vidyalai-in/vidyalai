# YouTube upload checklist: Adding a Reranker and Proving It Helped | LLM System Design

Videos are in `video/reranker-evaluation/out/` (not committed; rebuild with `node video/tools/build.mjs video/reranker-evaluation`).
Everything to paste is in this folder. Upload the main video first: the Shorts link to it.

## Automated alternative

`video/tools/upload.mjs` does every step below that the YouTube Data API supports: upload, title,
description, tags, category, language, audience, AI disclosure, thumbnail, captions and playlist, all as
private, then publishing and posting the comment. Progress is saved in `uploaded.json`, so re-runs resume.

```sh
node video/tools/upload.mjs upload  video/reranker-evaluation --dry-run   # check what will be sent
node video/tools/upload.mjs upload  video/reranker-evaluation             # everything up as private
node video/tools/upload.mjs status  video/reranker-evaluation             # processing state
node video/tools/upload.mjs publish video/reranker-evaluation             # public (main first) + post the comment
```

Still by hand (the API can't): the main video's end screen, each Short's Related video, and pinning the
comment. The script prints Studio links for each. One-time setup, the release steps and daily limits are
in `video/UPLOADING.md`.

## Once per channel

- **Upload defaults** (Studio → Settings → Upload defaults → Advanced settings), so API uploads match the
  manual steps: comments **Hold potentially inappropriate comments → Strict**, **Automatic chapters**,
  **Featured places** and **Automatic concepts** unchecked, **Caption certification** "This content has never
  aired on television in the U.S.", **Category** Education, **Language** English.

- **Verify the channel by phone** at https://www.youtube.com/verify. Custom thumbnails need it (Studio → Settings →
  Channel → Feature eligibility shows what is unlocked). Chapters need the channel to have no active strikes; if they
  don't appear, apply for Advanced features on the same page.
- Create the playlist **LLM System Design** (it can also be created from the upload form).

## Main video: `out/reranker-evaluation.mp4`

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
   - **Add end screen**: the last 12 s (from 11:45:02 in Studio's m:ss:frames format) is the end card. Choose
     **Add element → Subscribe**, drag it into the dashed circle on the right, resize it to fill the circle, and set it
     to run for the whole end card (drag the element's bar in the timeline if the typed time turns red). YouTube requires at least one video or playlist
     element, so also add **Playlist** → the series playlist and drag it into the dashed frame on the left, resized to
     fill it. **When the next video is public**, come back to this video's end screen and replace the playlist with
     **Video → Choose specific video** (same position and size). End screens can be edited any time; no re-upload needed.
4. **Checks**: wait for the copyright check to finish.
5. **Visibility**: Public, or schedule. After publishing, pin the comment from `main/pinned-comment.txt`.

## Shorts

Vertical videos up to 3 minutes are published as Shorts automatically. Links in Shorts descriptions and comments are
not clickable, so each Short links to the main video with **Related video** instead.

### `out/reranker-evaluation--short-reranker-ceiling.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-reranker-ceiling/title.txt`. **Description**: `shorts/short-reranker-ceiling/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-reranker-ceiling/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/reranker-evaluation--short-net-cost.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-net-cost/title.txt`. **Description**: `shorts/short-net-cost/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-net-cost/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/reranker-evaluation--short-detectable-gain.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-detectable-gain/title.txt`. **Description**: `shorts/short-detectable-gain/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-detectable-gain/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/reranker-evaluation--short-ship-rule.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-ship-rule/title.txt`. **Description**: `shorts/short-ship-rule/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: LLM System Design.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-ship-rule/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.
