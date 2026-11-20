# YouTube upload checklist: The Math You Need for LLMs: Vectors, Matrices and Shapes | The Math Behind LLMs

Videos are in `video/math-for-llms/out/` (not committed; rebuild with `node video/tools/build.mjs video/math-for-llms`).
Everything to paste is in this folder. Upload the main video first: the Shorts link to it.

## Automated alternative

`video/tools/upload.mjs` does every step below that the YouTube Data API supports: upload, title,
description, tags, category, language, audience, AI disclosure, thumbnail, captions and playlist, all as
private, then publishing and posting the comment. Progress is saved in `uploaded.json`, so re-runs resume.

```sh
node video/tools/upload.mjs upload  video/math-for-llms --dry-run   # check what will be sent
node video/tools/upload.mjs upload  video/math-for-llms             # everything up as private
node video/tools/upload.mjs status  video/math-for-llms             # processing state
node video/tools/upload.mjs publish video/math-for-llms             # public (main first) + post the comment
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
- Create the playlist **The Math Behind LLMs** if the channel doesn't have it yet (`upload.mjs upload` creates it
  for you; by hand, it can also be created from the upload form).

## Main video: `out/math-for-llms.mp4`

1. YouTube Studio → **Create → Upload videos** → select the MP4.
2. **Details**
   - **Title**: first line of `main/title.txt`.
   - **Description**: paste `main/description.txt`. Its timestamps (starting at 00:00) become the chapters.
   - **Thumbnail**: upload `main/thumbnail.png` (1280×720 PNG).
     **Test & compare** (Studio → the video's Details → Thumbnail → Test & compare): upload
     `main/thumbnail.png`, `main/thumbnail-b.png`, `main/thumbnail-c.png`. YouTube shows each to a share of viewers
     and keeps the one with the most watch time. It needs Advanced features on the channel and is set from the web Studio
     only; the API can't start a test.
   - **Playlists**: The Math Behind LLMs.
   - **Audience**: "No, it's not made for kids".
   - **Attributes → AI use**: **Yes**. The narration is an AI-generated voice. YouTube's policy only requires
     disclosure for realistic depictions of real people or events, but it labels AI-generated non-music audio in the
     expanded description, which matches the credit line already in the description.
   - **Show more**: **Tags**: paste `main/tags.txt`. **Language**: English. **Caption certification**: "This content
     has never aired on television in the U.S." **Category**: Education.
3. **Video elements**
   - **Add subtitles → Upload file → With timing** → `main/captions.en.srt`.
   - **Add end screen**: the last 12 s (from 11:40:09 in Studio's m:ss:frames format) is the end card. Choose
     **Add element → Subscribe**, drag it into the dashed circle on the right, resize it to fill the circle, and set it
     to run for the whole end card (drag the element's bar in the timeline if the typed time turns red). YouTube requires at least one video or playlist
     element, so also add **Playlist** → the series playlist and drag it into the dashed frame on the left, resized to
     fill it. **When the next video is public**, come back to this video's end screen and replace the playlist with
     **Video → Choose specific video** (same position and size). End screens can be edited any time; no re-upload needed.
   - **Add cards**: one per line of `main/cards.txt` (time, the part it names, and the video to link):
     - 09:55 Part 7: Derivatives, Gradients and the Chain Rule → not on YouTube yet (use the The Math Behind LLMs playlist, or skip)
     - 10:22 Part 3: Matrices as Transformations → not on YouTube yet (use the The Math Behind LLMs playlist, or skip)
     - 11:20 Part 2: Vectors and Dot Products → not on YouTube yet (use the The Math Behind LLMs playlist, or skip)
     Studio → **Cards → Video**, then drag each card to its time. Up to 5 per video.
4. **Checks**: wait for the copyright check to finish.
5. **Visibility**: Public, or schedule. After publishing, pin the comment from `main/pinned-comment.txt`.

## Shorts

Vertical videos up to 3 minutes are published as Shorts automatically. Links in Shorts descriptions and comments are
not clickable, so each Short links to the main video with **Related video** instead.

### `out/math-for-llms--short-dot-products.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-dot-products/title.txt`. **Description**: `shorts/short-dot-products/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: The Math Behind LLMs.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-dot-products/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/math-for-llms--short-shape-rule.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-shape-rule/title.txt`. **Description**: `shorts/short-shape-rule/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: The Math Behind LLMs.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-shape-rule/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.

### `out/math-for-llms--short-long-context.mp4`

1. **Create → Upload videos** → select the MP4.
2. **Title**: `shorts/short-long-context/title.txt`. **Description**: `shorts/short-long-context/description.txt`. **Audience**: not made for
   kids. **Attributes → AI use**: Yes. **Playlists**: The Math Behind LLMs.
3. **Video elements → Add subtitles → Upload file → With timing** → `shorts/short-long-context/captions.en.srt`.
4. Publish, then **Studio → Content → Shorts → ✏️ (Details) → Related video** → choose the main video → Save.
