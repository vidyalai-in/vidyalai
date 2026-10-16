# YouTube upload checklist: Designing a RAG System for 10 Million Documents | LLM System Design

Videos are in `video/rag-system-10m-documents/out/` (not committed; rebuild with `node video/tools/build.mjs video/rag-system-10m-documents`).
Everything to paste is in this folder.

## Main video: `out/rag-system-10m-documents.mp4`

1. YouTube Studio → **Create → Upload videos** → select the MP4.
2. **Title**: first line of `main/title.txt`.
3. **Description**: paste `main/description.txt`. The timestamps in it create the chapters automatically.
4. **Thumbnail**: upload `main/thumbnail.png`.
5. **Playlist**: LLM System Design (create it on first upload).
6. **Audience**: "No, it's not made for kids".
7. **Show more**:
   - **Altered or synthetic content**: the narration is an AI-generated voice; no real person, place or event is depicted. Check YouTube's current disclosure policy (Help: "Disclosing use of altered or synthetic content") and answer accordingly. The description already credits the AI voice either way.
   - **Tags**: paste `main/tags.txt`.
   - **Language**: English. **Caption certification**: "This content has never aired on television in the U.S."
   - **Category**: Education.
8. **Video elements**: add **Subtitles → Upload file → With timing** → `main/captions.en.srt`. Add an **end screen** (last 15 s: subscribe + "best for viewer" video).
9. **Visibility**: Public (or schedule). After publishing, pin the comment from `main/pinned-comment.txt`.

## Shorts

### `out/rag-system-10m-documents--short-index-sizing.mp4`

- Title: `shorts/short-index-sizing/title.txt`. Description: `shorts/short-index-sizing/description.txt` (replace the "Full video" placeholder with the main video's link).
- Upload captions from `shorts/short-index-sizing/captions.en.srt`.
- In the YouTube app or Studio, set **Related video** to the main video so viewers can jump to it.

### `out/rag-system-10m-documents--short-retrieval-first.mp4`

- Title: `shorts/short-retrieval-first/title.txt`. Description: `shorts/short-retrieval-first/description.txt` (replace the "Full video" placeholder with the main video's link).
- Upload captions from `shorts/short-retrieval-first/captions.en.srt`.
- In the YouTube app or Studio, set **Related video** to the main video so viewers can jump to it.

### `out/rag-system-10m-documents--short-acl-filter.mp4`

- Title: `shorts/short-acl-filter/title.txt`. Description: `shorts/short-acl-filter/description.txt` (replace the "Full video" placeholder with the main video's link).
- Upload captions from `shorts/short-acl-filter/captions.en.srt`.
- In the YouTube app or Studio, set **Related video** to the main video so viewers can jump to it.

### `out/rag-system-10m-documents--short-latency-budget.mp4`

- Title: `shorts/short-latency-budget/title.txt`. Description: `shorts/short-latency-budget/description.txt` (replace the "Full video" placeholder with the main video's link).
- Upload captions from `shorts/short-latency-budget/captions.en.srt`.
- In the YouTube app or Studio, set **Related video** to the main video so viewers can jump to it.
